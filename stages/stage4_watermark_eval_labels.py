"""Stage 4 sanity check – evaluate watermark using teacher labels instead of logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from accelerate import Accelerator
from tqdm.auto import tqdm

from config import ModelSpec, WatermarkEvalConfig
from hashing import BigramHash, load_hash_config
from utils.env import set_global_seed
from utils.io import read_jsonl_rows, write_json
from utils.tokenization import compute_overlap, load_tokenizer

EvalEntry = Tuple[Tuple[int, int], float]


def _aligned_offsets(offsets: Sequence[Tuple[int, int]], limit: int) -> Dict[int, int]:
    """Build a lookup from end-offset -> student token index for aligned positions."""
    result: Dict[int, int] = {}
    for idx, (_, end) in enumerate(offsets[:limit]):
        result[int(end)] = idx
    return result


def run_stage4_labels(cfg: WatermarkEvalConfig) -> Path:
    """
    Sanity check evaluation: instead of using student logits, compute the
    empirical frequency that the *actual* next token in the trace lies in the
    green list. Keeps the same bigram selection logic as stage4_watermark_eval.
    """
    accelerator = Accelerator()
    set_global_seed(cfg.seed)

    traces = read_jsonl_rows(cfg.traces_jsonl)
    if not traces:
        raise RuntimeError("Trace file is empty")

    hash_cfg = load_hash_config(cfg.hash_config)
    teacher_tokenizer = load_tokenizer(cfg.teacher, padding_side="left")
    student_tokenizer = load_tokenizer(cfg.student, padding_side="left")

    hash_fn = BigramHash(
        hash_cfg,
        vocab_size=len(teacher_tokenizer),
        excluded_token_ids=getattr(teacher_tokenizer, "all_special_ids", None),
    )

    overlap = compute_overlap(teacher_tokenizer, student_tokenizer)
    map_tensor = overlap.source_to_target
    student_vocab = len(student_tokenizer)
    teacher_vocab = len(teacher_tokenizer)
    map_dev = map_tensor
    if map_dev.shape[0] < teacher_vocab:
        padded = torch.full((teacher_vocab,), -1, dtype=torch.long)
        padded[: map_dev.shape[0]] = map_dev
        map_dev = padded
    else:
        map_dev = map_dev[:teacher_vocab]

    local_rows = traces[accelerator.process_index :: accelerator.num_processes]
    tmp_dir = cfg.output_path.parent / "_tmp_stage4_labels"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rank_path = tmp_dir / f"rank_{accelerator.process_index:03d}.json"

    local_values: List[EvalEntry] = []

    batch_size = max(1, cfg.batch_size)
    iterator = range(0, len(local_rows), batch_size)
    if accelerator.is_local_main_process:
        iterator = tqdm(iterator, total=(len(local_rows) + batch_size - 1) // batch_size, desc="Stage 4 labels")

    for start in iterator:
        batch = local_rows[start : start + batch_size]
        texts = [row["prompt"] + row["response"] for row in batch]
        student_inputs = student_tokenizer(
            texts,
            padding=True,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets_batch = student_inputs["offset_mapping"]
        attention_mask = torch.tensor(student_inputs["attention_mask"])
        seq_lengths = attention_mask.sum(dim=1).tolist()

        batch_bigrams: List[Tuple[int, int]] = []
        batch_next_tokens: List[int] = []
        for idx, row in enumerate(batch):
            text = texts[idx]
            teacher_encoding = teacher_tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            teacher_ids = torch.tensor(teacher_encoding["input_ids"], dtype=torch.long)
            teacher_offsets = [
                (int(start), int(end))
                for start, end in teacher_encoding["offset_mapping"]
            ]
            offsets_list = [
                (int(start), int(end))
                for start, end in offsets_batch[idx][: seq_lengths[idx]]
            ]
            alignment = _aligned_offsets(offsets_list, len(offsets_list))
            response_start = len(row["prompt"])
            for t_idx, (_, end) in enumerate(teacher_offsets):
                if end <= response_start:
                    continue
                if end not in alignment:
                    continue
                if t_idx < 1:
                    continue
                # token t ends at `end`, next token is t_idx+1
                if t_idx + 1 >= len(teacher_ids):
                    continue
                bigram = (int(teacher_ids[t_idx - 1]), int(teacher_ids[t_idx]))
                next_token = int(teacher_ids[t_idx + 1])
                batch_bigrams.append(bigram)
                batch_next_tokens.append(next_token)

        if not batch_bigrams:
            continue

        bigrams_tensor = torch.tensor(batch_bigrams, dtype=torch.long)
        next_tokens_tensor = torch.tensor(batch_next_tokens, dtype=torch.long)
        valid_mask = (
            (next_tokens_tensor >= 0)
            & (next_tokens_tensor < map_dev.shape[0])
        )
        # require the next token to map to student vocab
        mapped = map_dev[next_tokens_tensor.clamp(min=0, max=map_dev.shape[0]-1)]
        valid_mask = valid_mask & (mapped >= 0) & (mapped < student_vocab)

        if not valid_mask.any():
            continue

        bigrams_tensor = bigrams_tensor[valid_mask]
        next_tokens_tensor = next_tokens_tensor[valid_mask]
        # Chunk to avoid large GPU allocations
        chunk_size = max(1, cfg.mask_chunk if hasattr(cfg, "mask_chunk") else 64)
        for offset in range(0, bigrams_tensor.shape[0], chunk_size):
            bg_chunk = bigrams_tensor[offset : offset + chunk_size].to(accelerator.device)
            next_chunk = next_tokens_tensor[offset : offset + chunk_size].to(accelerator.device)
            teacher_masks = hash_fn.mask_batch(
                bg_chunk.tolist(), device=accelerator.device, dtype=torch.bool
            )
            row_idx = torch.arange(next_chunk.shape[0], device=accelerator.device, dtype=torch.long)
            green_hits = teacher_masks[row_idx, next_chunk]
            for bg, hit in zip(bg_chunk.tolist(), green_hits.tolist()):
                local_values.append((tuple(bg), 1.0 if hit else 0.0))

    with rank_path.open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {"bigram": list(bigram), "value": value}
                for bigram, value in local_values
            ],
            handle,
        )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        all_entries: List[Tuple[Tuple[int, int], float]] = []
        for idx in range(accelerator.num_processes):
            shard = tmp_dir / f"rank_{idx:03d}.json"
            if not shard.exists():
                continue
            with shard.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                for entry in data:
                    all_entries.append((tuple(entry["bigram"]), float(entry["value"])))
        if not all_entries:
            raise RuntimeError("No watermark statistics collected")
        all_entries.sort(key=lambda x: x[0])
        values: List[float] = []
        prev_bigram: Tuple[int, int] | None = None
        for bigram, value in all_entries:
            if bigram == prev_bigram:
                continue
            values.append(value)
            prev_bigram = bigram
        payload = {
            "num_measurements": len(values),
            "mean": float(sum(values) / len(values)),
            "mode": cfg.mode,
            "supervision": cfg.supervision,
        }
        write_json(cfg.output_path, payload)
        for shard in tmp_dir.glob("rank_*.json"):
            shard.unlink()
        tmp_dir.rmdir()
    accelerator.wait_for_everyone()
    return cfg.output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 4 labels sanity check")
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--hash-config", type=Path, required=True)
    parser.add_argument("--teacher-model", type=str, required=True)
    parser.add_argument("--teacher-dtype", type=str, default="bfloat16")
    parser.add_argument("--teacher-pad-token", type=str, default=None)
    parser.add_argument("--student-model", type=str, required=True)
    parser.add_argument("--student-dtype", type=str, default="bfloat16")
    parser.add_argument("--student-pad-token", type=str, default=None)
    parser.add_argument("--lora-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["open", "closed"], default="open")
    parser.add_argument("--supervision", choices=["supervised", "unsupervised"], default="supervised")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = WatermarkEvalConfig(
        dataset=args.dataset,
        teacher=ModelSpec(name=args.teacher_model, dtype=args.teacher_dtype, pad_token=args.teacher_pad_token),
        student=ModelSpec(name=args.student_model, dtype=args.student_dtype, pad_token=args.student_pad_token),
        hash_config=args.hash_config,
        traces_jsonl=args.traces,
        lora_dir=args.lora_dir,
        mode=args.mode,  # type: ignore[arg-type]
        supervision=args.supervision,  # type: ignore[arg-type]
        output_path=args.output,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    run_stage4_labels(cfg)


if __name__ == "__main__":
    main()
