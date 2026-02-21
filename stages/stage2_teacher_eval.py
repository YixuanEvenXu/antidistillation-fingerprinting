"""Stage 2 – teacher evaluation via answer forcing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from math_verify import parse, verify
from tqdm.auto import tqdm

from config import ModelSpec, TeacherEvalConfig
from models.loader import load_causal_lm
from models.prompts import OASST1_SYSTEM_PROMPT, PromptBuilder
from utils.env import set_global_seed
from utils.io import read_jsonl_rows, write_json
from utils.tokenization import load_tokenizer

ANSWER_FORCE_STRING = "\n\n**Final Answer**\n\\[\\boxed{"


def _is_correct(candidate: str, solution: str) -> bool:
    try:
        sol = parse(solution)
        cand = parse(candidate)
        return bool(verify(sol, cand))
    except Exception:
        return False


def _batch(seq: Sequence, size: int) -> List[Sequence]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _pad_batch(features: List[Dict], tokenizer) -> Dict:
    input_batch = tokenizer.pad(
        {"input_ids": [f["input_ids"] for f in features]},
        padding=True,
        return_tensors="pt",
    )
    labels = tokenizer.pad(
        {"input_ids": [f["labels"] for f in features]},
        padding=True,
        return_tensors="pt",
    )["input_ids"]
    if tokenizer.pad_token_id is not None:
        labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)
    input_batch["labels"] = labels
    return input_batch


def _prepare_nll_example(tokenizer, prompt: str, response: str, max_length: int | None) -> Dict[str, List[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    if max_length is not None and max_length > 0:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {"input_ids": input_ids, "labels": labels}


def _split_think_prefix(response: str) -> tuple[str, str]:
    if response.startswith("<think>"):
        return "<think>", response[len("<think>"):]
    return "", response


def _run_oasst1_nll(cfg: TeacherEvalConfig, traces: List[Dict]) -> Path:
    accelerator = Accelerator()
    set_global_seed(cfg.seed)

    tokenizer = load_tokenizer(cfg.teacher, padding_side="left")
    teacher = load_causal_lm(cfg.teacher)
    teacher.to(accelerator.device)
    teacher.eval()

    builder = PromptBuilder(system_prompt=OASST1_SYSTEM_PROMPT)

    local_traces = traces[accelerator.process_index :: accelerator.num_processes]
    batch_size = max(1, cfg.batch_size)
    iterator = _batch(local_traces, batch_size)
    if accelerator.is_local_main_process:
        iterator = tqdm(iterator, total=(len(local_traces) + batch_size - 1) // batch_size, desc="Stage 2: OASST1 NLL")

    max_length = tokenizer.model_max_length
    if not isinstance(max_length, int) or max_length > 100000:
        max_length = None

    total_nll = 0.0
    total_tokens = 0
    total_examples = 0

    for chunk in iterator:
        features: List[Dict] = []
        for row in chunk:
            response = row.get("response") or ""
            if not response:
                continue
            think_prefix, response = _split_think_prefix(response)
            if not response:
                continue
            if "messages" in row:
                messages = row.get("messages") or []
                prompt_text = builder.build_from_messages(tokenizer, messages, add_system=True)
            else:
                prompt_text = row.get("prompt")
                if not prompt_text:
                    continue
            if think_prefix:
                prompt_text = prompt_text + think_prefix
            features.append(_prepare_nll_example(tokenizer, prompt_text, response, max_length))
        if not features:
            continue
        batch_inputs = _pad_batch(features, tokenizer)
        input_ids = batch_inputs["input_ids"].to(accelerator.device)
        attention_mask = batch_inputs["attention_mask"].to(accelerator.device)
        labels = batch_inputs["labels"].to(accelerator.device)

        with torch.no_grad():
            logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        mask = shift_labels != -100
        if mask.sum() == 0:
            continue
        vocab = shift_logits.shape[-1]
        losses = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            reduction="none",
        ).view_as(shift_labels)
        total_nll += float(losses[mask].sum().item())
        total_tokens += int(mask.sum().item())
        total_examples += len(features)

    counts = torch.tensor(
        [total_nll, float(total_tokens), float(total_examples)],
        dtype=torch.float32,
        device=accelerator.device,
    )
    reduced = accelerator.reduce(counts, reduction="sum")
    if accelerator.is_main_process:
        total_nll_val, token_count, example_count = reduced.tolist()
        mean_nll = float(total_nll_val / token_count) if token_count > 0 else None
        payload = {
            "num_examples": int(example_count),
            "num_tokens": int(token_count),
            "total_nll": float(total_nll_val),
            "mean_nll": mean_nll,
        }
        write_json(cfg.output_path, payload)
    accelerator.wait_for_everyone()
    return cfg.output_path


def run_stage2(cfg: TeacherEvalConfig) -> Path:
    accelerator = Accelerator()
    set_global_seed(cfg.seed)

    traces = read_jsonl_rows(cfg.traces_jsonl)
    if not traces:
        raise RuntimeError("Trace file is empty")
    if cfg.dataset == "oasst1":
        return _run_oasst1_nll(cfg, traces)
    missing = sum(1 for row in traces if not row.get("solution"))
    if missing > 0:
        raise RuntimeError(f"{missing} trace rows are missing solutions; regenerate Stage 1 traces.")

    tokenizer = load_tokenizer(cfg.teacher, padding_side="left")
    teacher = load_causal_lm(cfg.teacher)
    teacher.to(accelerator.device)

    local_traces = traces[accelerator.process_index :: accelerator.num_processes]

    raw_correct = 0
    forced_correct = 0
    total = 0

    iterator = _batch(local_traces, max(1, cfg.batch_size))
    if accelerator.is_local_main_process:
        iterator = tqdm(iterator, total=(len(local_traces) + cfg.batch_size - 1) // cfg.batch_size, desc="Stage 2: teacher eval")

    for chunk in iterator:
        prompts: List[str] = []
        solutions: List[str] = []
        raw_predictions: List[str] = []
        for trace_row in chunk:
            prompt = trace_row["prompt"]
            response = trace_row["response"]
            full_text = prompt + response
            prompts.append(full_text + ANSWER_FORCE_STRING)
            solutions.append(trace_row.get("solution"))
            raw_predictions.append(response)
        inputs = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = teacher.generate(
                **inputs,
                max_new_tokens=cfg.max_answer_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        for raw_pred, forced_text, solution in zip(raw_predictions, decoded, solutions):
            if not solution:
                continue
            total += 1
            if _is_correct(raw_pred, solution):
                raw_correct += 1
            if _is_correct(forced_text, solution):
                forced_correct += 1

    counts = torch.tensor(
        [raw_correct, forced_correct, total],
        dtype=torch.float32,
        device=accelerator.device,
    )
    reduced = accelerator.reduce(counts, reduction="sum")
    if accelerator.is_main_process:
        raw, forced, denom = reduced.tolist()
        if denom == 0:
            raise RuntimeError("No evaluable references found")
        payload = {
            "num_examples": int(denom),
            "raw_correct": int(raw),
            "answer_forced_correct": int(forced),
            "raw_accuracy": float(raw / denom),
            "answer_forced_accuracy": float(forced / denom),
        }
        write_json(cfg.output_path, payload)
    # accelerator.wait_for_everyone()
    # If the barrier is buggy, lets wait instead.
    wait_seconds = 60*5
    accelerator.print(f"Waiting {wait_seconds} seconds for other processes to finish...")
    import time
    time.sleep(wait_seconds)
    return cfg.output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2 – teacher evaluation")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--teacher-model", type=str, required=True)
    parser.add_argument("--teacher-dtype", type=str, default="bfloat16")
    parser.add_argument("--teacher-pad-token", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-answer-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = TeacherEvalConfig(
        dataset=args.dataset,
        teacher=ModelSpec(name=args.teacher_model, dtype=args.teacher_dtype, pad_token=args.teacher_pad_token),
        traces_jsonl=args.traces,
        output_path=args.output,
        batch_size=args.batch_size,
        max_answer_tokens=args.max_answer_tokens,
        seed=args.seed,
    )
    run_stage2(cfg)


if __name__ == "__main__":
    main()
