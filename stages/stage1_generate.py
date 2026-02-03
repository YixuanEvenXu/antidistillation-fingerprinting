"""Stage 1 – teacher trace generation (radioactive or ADS)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import torch
import re
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import LogitsProcessorList

from config import GenerationConfig, ModelSpec
from data.gsm8k import GSM8KProvider
from data.oasst1 import OASST1Provider
from hashing import BigramHash, HashConfig, load_hash_config
from models.loader import load_causal_lm
from models.logits import ADSLogitsProcessor, RadioactiveLogitsProcessor
from models.prompts import OASST1_SYSTEM_PROMPT, PromptBuilder
from utils.env import set_global_seed
from utils.io import write_json
from utils.tokenization import load_tokenizer


DATASET_PROVIDERS = {
    "gsm8k": GSM8KProvider(),
    "oasst1": OASST1Provider(),
}


def _get_provider(name: str):
    """Return a dataset provider by name.

    Args:
        name: Dataset name (e.g., "gsm8k", "oasst1").

    Returns:
        Dataset provider instance.

    Raises:
        ValueError: If the dataset name is unsupported.
    """
    if name not in DATASET_PROVIDERS:
        raise ValueError(f"Unsupported dataset: {name}")
    return DATASET_PROVIDERS[name]


def _batched(seq: List, batch_size: int) -> Iterable[List]:
    """Yield list slices of length batch_size.

    Args:
        seq: List to batch.
        batch_size: Size of each batch.

    Yields:
        Lists containing up to batch_size items.
    """
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size]


def _prepare_prompt(
    builder: PromptBuilder,
    tokenizer,
    example,
    *,
    add_system: bool,
) -> tuple[str, dict]:
    """Render a prompt and extract trace metadata from a dataset example.

    Args:
        builder: PromptBuilder for chat formatting.
        tokenizer: Tokenizer used to render prompts.
        example: Dataset example with prompt or messages.
        add_system: Whether to include a system prompt when using messages.

    Returns:
        Tuple of (rendered prompt text, trace payload dict).
    """
    if getattr(example, "messages", None):
        rendered = builder.build_from_messages(tokenizer, example.messages, add_system=add_system)
        return rendered, {"messages": example.messages}
    if example.prompt is None:
        raise ValueError("Dataset example missing prompt/messages")
    rendered = builder.build(tokenizer, example.prompt)
    return rendered, {"prompt": rendered}


def _needs_think_prefix(model_name: str) -> bool:
    """Return True if the model expects a <think> prefix.

    Args:
        model_name: Full model name string.

    Returns:
        Boolean indicating whether to prepend "<think>".
    """
    lowered = model_name.lower()
    return "r1" in lowered and "qwen" in lowered


def _extract_gsm8k_solution(solution_text: str) -> str:
    """Extract the numeric answer from a GSM8K solution string.

    Args:
        solution_text: Raw GSM8K solution containing a '####' delimiter.

    Returns:
        Cleaned numeric answer string.

    Raises:
        ValueError: If the expected delimiter or numeric answer is missing.
    """
    if "####" not in solution_text:
        raise ValueError("GSM8K solution missing '####' delimiter")
    tail = solution_text.split("####")[-1]
    pattern = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")
    match = pattern.search(tail)
    if not match:
        raise ValueError("Unable to extract numeric solution from GSM8K answer")
    numeric = match.group(0).replace(",", "").strip()
    if not numeric:
        raise ValueError("Extracted GSM8K solution is empty after cleaning")
    return numeric


def run_stage1(cfg: GenerationConfig, hash_cfg: HashConfig) -> Path:
    """Run Stage 1 to generate teacher traces with optional watermarking.

    Args:
        cfg: GenerationConfig for dataset, models, and generation settings.
        hash_cfg: HashConfig with seed and gamma.

    Returns:
        Path to the merged traces JSONL file.
    """
    accelerator = Accelerator()
    set_global_seed(cfg.seed)

    provider = _get_provider(cfg.dataset)
    dataset_rows = list(provider.load(cfg.split, cfg.max_examples))
    indexed_examples = list(enumerate(dataset_rows))
    total_examples = len(indexed_examples)
    if total_examples == 0:
        raise RuntimeError("No dataset examples loaded")

    local_examples = indexed_examples[accelerator.process_index :: accelerator.num_processes]
    teacher_tokenizer = load_tokenizer(cfg.teacher, padding_side="left")
    proxy_tokenizer = load_tokenizer(cfg.proxy, padding_side="left")

    teacher_model = load_causal_lm(cfg.teacher)
    teacher_model.resize_token_embeddings(len(teacher_tokenizer))
    if hasattr(teacher_model, "generation_config"):
        teacher_model.generation_config.pad_token_id = teacher_tokenizer.pad_token_id
        teacher_model.generation_config.eos_token_id = teacher_tokenizer.eos_token_id
    teacher_model.to(accelerator.device)

    proxy_model = None
    if cfg.method == "ads":
        proxy_model = load_causal_lm(cfg.proxy)
        proxy_model.resize_token_embeddings(len(proxy_tokenizer))
        proxy_model.to(accelerator.device)
    hash_fn = BigramHash(
        hash_cfg,
        vocab_size=len(teacher_tokenizer),
        excluded_token_ids=getattr(teacher_tokenizer, "all_special_ids", None),
    )

    builder = PromptBuilder()
    add_system_for_messages = False
    if cfg.dataset == "oasst1":
        builder.system_prompt = OASST1_SYSTEM_PROMPT
        add_system_for_messages = True
    add_think_prefix = _needs_think_prefix(cfg.teacher.name)
    rows: List[dict] = []

    total_batches = (len(local_examples) + cfg.batch_size - 1) // cfg.batch_size
    iterator = _batched(local_examples, cfg.batch_size)
    if accelerator.is_local_main_process:
        iterator = tqdm(
            iterator,
            total=total_batches,
            desc="Stage 1: teacher generation",
        )

    for batch in iterator:
        gen_prompts: List[str] = []
        trace_payloads: List[dict] = []
        for _, example in batch:
            rendered, payload = _prepare_prompt(
                builder,
                teacher_tokenizer,
                example,
                add_system=add_system_for_messages,
            )
            gen_prompts.append(rendered + ("<think>" if add_think_prefix else ""))
            trace_payloads.append(payload)
        enc = teacher_tokenizer(
            gen_prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(accelerator.device)
        attention_mask = enc["attention_mask"].to(accelerator.device)
        logits_processor = None
        if cfg.method == "radioactive":
            if cfg.delta is None:
                raise ValueError("delta must be specified for radioactive watermarking")
            logits_processor = RadioactiveLogitsProcessor(hash_fn, cfg.delta)
        elif cfg.method == "ads":
            if cfg.lam is None:
                raise ValueError("lambda must be specified for ADS watermarking")
            if proxy_model is None:
                raise RuntimeError("ADS method requires a proxy model")
            logits_processor = ADSLogitsProcessor(
                hash_fn,
                cfg.lam,
                proxy_model=proxy_model,
                pad_token_id=teacher_tokenizer.pad_token_id,
                eos_token_id=teacher_tokenizer.eos_token_id,
            )
        elif cfg.method == "control":
            pass
        else:
            raise ValueError(f"Unsupported method: {cfg.method}")

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            eos_token_id=teacher_tokenizer.eos_token_id,
            pad_token_id=teacher_tokenizer.pad_token_id,
            repetition_penalty=cfg.repetition_penalty,
            do_sample=cfg.temperature > 0,
        )
        if logits_processor is not None:
            gen_kwargs.update(dict(logits_processor=LogitsProcessorList([logits_processor])))
        if cfg.temperature > 0:
            gen_kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p)
        if hasattr(logits_processor, "reset"):
            logits_processor.reset()
        with torch.no_grad():
            outputs = teacher_model.generate(**gen_kwargs)

        prompt_window = input_ids.shape[-1]
        responses = outputs[:, prompt_window:]
        decoded_responses = teacher_tokenizer.batch_decode(
            responses,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        for (global_idx, example), response_text, trace_payload in zip(
            batch, decoded_responses, trace_payloads
        ):
            solution_val = None
            if cfg.dataset == "gsm8k":
                if example.solution is None:
                    raise ValueError("GSM8K example missing solution text")
                solution_val = _extract_gsm8k_solution(example.solution)
            elif example.solution is not None:
                solution_val = example.solution

            response_value = response_text.strip()
            if add_think_prefix:
                response_value = "<think>" + response_value

            row = {
                "index": int(global_idx),
                "response": response_value,
                "solution": solution_val,
            }
            row.update(trace_payload)
            rows.append(row)

    tmp_dir = cfg.output_jsonl.parent / "_tmp_stage1"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rank_path = tmp_dir / f"rank_{accelerator.process_index:03d}.json"
    with rank_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        merged: List[dict] = []
        for idx in range(accelerator.num_processes):
            shard = tmp_dir / f"rank_{idx:03d}.json"
            if not shard.exists():
                continue
            with shard.open("r", encoding="utf-8") as handle:
                merged.extend(json.load(handle))
        merged = merged[: total_examples]
        merged.sort(key=lambda row: row["index"])

        cfg.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with cfg.output_jsonl.open("w", encoding="utf-8") as handle:
            for row in merged:
                payload = {
                    "response": row["response"],
                    "solution": row["solution"],
                }
                if "prompt" in row:
                    payload["prompt"] = row["prompt"]
                else:
                    payload["messages"] = row["messages"]
                json.dump(payload, handle, ensure_ascii=False)
                handle.write("\n")

        meta_payload = {
            "dataset": cfg.dataset,
            "split": cfg.split,
            "method": cfg.method,
            "delta": cfg.delta,
            "lambda": cfg.lam,
            "num_examples": len(merged),
        }
        write_json(cfg.metadata_path, meta_payload)

        for shard in tmp_dir.glob("rank_*.json"):
            shard.unlink()
        tmp_dir.rmdir()
    accelerator.wait_for_everyone()
    return cfg.output_jsonl


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for Stage 1.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Stage 1 – teacher generation")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--teacher-model", type=str, required=True)
    parser.add_argument("--teacher-dtype", type=str, default="bfloat16")
    parser.add_argument("--teacher-pad-token", type=str, default=None)
    parser.add_argument("--proxy-model", type=str, required=True)
    parser.add_argument("--proxy-dtype", type=str, default="bfloat16")
    parser.add_argument("--proxy-pad-token", type=str, default=None)
    parser.add_argument("--method", type=str, choices=["radioactive", "ads", "control"], required=True)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--hash-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for Stage 1.

    Args:
        argv: Optional list of CLI arguments (defaults to sys.argv).

    Returns:
        None.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = GenerationConfig(
        dataset=args.dataset,
        split=args.split,
        max_examples=args.max_examples,
        teacher=ModelSpec(name=args.teacher_model, dtype=args.teacher_dtype, pad_token=args.teacher_pad_token),
        proxy=ModelSpec(name=args.proxy_model, dtype=args.proxy_dtype, pad_token=args.proxy_pad_token),
        method=args.method,  # type: ignore[arg-type]
        delta=args.delta,
        lam=args.lam,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        batch_size=args.batch_size,
        seed=args.seed,
        output_jsonl=args.output,
        metadata_path=args.metadata,
    )
    hash_cfg = load_hash_config(args.hash_config)
    run_stage1(cfg, hash_cfg)


if __name__ == "__main__":
    main()
