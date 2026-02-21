"""Stage 3 – student finetuning on prompt/response traces with response-only loss."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from config import FinetuneConfig, ModelSpec
from models.loader import load_causal_lm
from models.prompts import OASST1_SYSTEM_PROMPT, PromptBuilder
from utils.env import set_global_seed
from utils.io import read_jsonl_rows
from utils.tokenization import load_tokenizer


def _mask_prompt(tokenizer, prompt: str, response: str, max_length: int) -> Dict[str, List[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + response_ids)[:max_length]
    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return {"input_ids": input_ids, "labels": labels}


def _pad_batch(features: List[Dict], tokenizer, max_length: int) -> Dict:
    input_batch = tokenizer.pad(
        {"input_ids": [f["input_ids"] for f in features]},
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = tokenizer.pad(
        {"input_ids": [f["labels"] for f in features]},
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )["input_ids"]
    if tokenizer.pad_token_id is not None:
        labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)
    input_batch["labels"] = labels
    return input_batch


def _prompt_from_row(builder: PromptBuilder, tokenizer, row: Dict, *, add_system: bool) -> str:
    if "messages" in row:
        messages = row.get("messages") or []
        return builder.build_from_messages(tokenizer, messages, add_system=add_system)
    prompt = row.get("prompt")
    if not prompt:
        raise ValueError("Trace row missing prompt/messages")
    return prompt


def run_stage3(cfg: FinetuneConfig) -> Path:
    set_global_seed(cfg.seed)
    rows = read_jsonl_rows(cfg.traces_jsonl)
    if not rows:
        raise RuntimeError("Trace file is empty")

    tokenizer = load_tokenizer(cfg.student, padding_side="left")
    model = load_causal_lm(cfg.student)
    if hasattr(model, "config"):
        model.config.use_cache = False

    add_system_for_messages = cfg.dataset == "oasst1"
    builder = PromptBuilder(system_prompt=OASST1_SYSTEM_PROMPT if add_system_for_messages else None)
    wrapped_rows = []
    for row in rows:
        response = row.get("response")
        if not response:
            continue
        prompt_text = _prompt_from_row(builder, tokenizer, row, add_system=add_system_for_messages)
        wrapped_rows.append(_mask_prompt(tokenizer, prompt_text, response, cfg.max_seq_length))
    train_dataset = Dataset.from_list(wrapped_rows)

    lora = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        max_steps=-1,
        warmup_ratio=0.0,
        logging_steps=10,
        save_strategy="no",
        dataset_text_field=None,
        report_to=[],
        remove_unused_columns=False,
        packing=False,
        gradient_checkpointing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora,
        data_collator=lambda batch: _pad_batch(batch, tokenizer, cfg.max_seq_length),
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(cfg.output_dir)
    return cfg.output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3 – student finetune")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--student-model", type=str, required=True)
    parser.add_argument("--student-dtype", type=str, default="bfloat16")
    parser.add_argument("--student-pad-token", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = FinetuneConfig(
        dataset=args.dataset,
        student=ModelSpec(name=args.student_model, dtype=args.student_dtype, pad_token=args.student_pad_token),
        traces_jsonl=args.traces,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
    )
    run_stage3(cfg)


if __name__ == "__main__":
    main()
