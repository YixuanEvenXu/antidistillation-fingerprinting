"""Stage 4 – student watermark evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from peft import PeftModel
from tqdm.auto import tqdm

from config import ModelSpec, WatermarkEvalConfig
from hashing import BigramHash, HashConfig, load_hash_config
from models.loader import load_causal_lm
from utils.env import set_global_seed
from utils.io import read_jsonl_rows, write_json
from models.prompts import OASST1_SYSTEM_PROMPT, PromptBuilder
from utils.tokenization import compute_overlap, load_tokenizer

EvalEntry = Tuple[Tuple[int, int], float]


def _build_student_shared_mask(student_tokenizer, shared_tokens: set[str]) -> torch.BoolTensor:
    """Build a mask over the student vocab for shared token strings.

    Args:
        student_tokenizer: Student tokenizer instance.
        shared_tokens: Set of token strings shared with the teacher.

    Returns:
        Boolean tensor of shape [student_vocab] marking shared tokens.
    """
    mask = torch.zeros(len(student_tokenizer), dtype=torch.bool)
    vocab = student_tokenizer.get_vocab()
    for token, idx in vocab.items():
        if token in shared_tokens:
            mask[idx] = True
    return mask


def _aligned_offsets(offsets: Sequence[Tuple[int, int]], limit: int) -> Dict[int, int]:
    """Build a lookup from end-offset -> student token index for aligned positions.

    Args:
        offsets: Sequence of (start, end) offsets from tokenizer output.
        limit: Maximum number of offsets to consider.

    Returns:
        Mapping from end-offset integer to token index.
    """
    result: Dict[int, int] = {}
    for idx, (_, end) in enumerate(offsets[:limit]):
        result[int(end)] = idx
    return result


def _prompt_from_row(builder: PromptBuilder, tokenizer, row: Dict, *, add_system: bool) -> str:
    """Extract or build a prompt from a trace row.

    Args:
        builder: PromptBuilder used for message-based prompts.
        tokenizer: Tokenizer used in the prompt builder.
        row: Trace row dict with "prompt" or "messages".
        add_system: Whether to insert a system prompt for messages.

    Returns:
        Prompt text ready for concatenation with the response.
    """
    if "messages" in row:
        messages = row.get("messages") or []
        return builder.build_from_messages(tokenizer, messages, add_system=add_system)
    prompt = row.get("prompt")
    if not prompt:
        raise ValueError("Trace row missing prompt/messages")
    return prompt


def run_stage4(cfg: WatermarkEvalConfig) -> Path:
    """Run Stage 4 to evaluate watermark statistics for a student model.

    Args:
        cfg: WatermarkEvalConfig with dataset, models, and evaluation settings.

    Returns:
        Path to the written watermark JSON file.
    """
    accelerator = Accelerator()
    set_global_seed(cfg.seed)

    traces = read_jsonl_rows(cfg.traces_jsonl)
    if not traces:
        raise RuntimeError("Trace file is empty")

    hash_cfg = load_hash_config(cfg.hash_config)
    teacher_tokenizer = load_tokenizer(cfg.teacher, padding_side="left")
    student_tokenizer = load_tokenizer(cfg.student, padding_side="left")
    add_system_for_messages = cfg.dataset == "oasst1"
    builder = PromptBuilder(system_prompt=OASST1_SYSTEM_PROMPT if add_system_for_messages else None)

    hash_fn = BigramHash(
        hash_cfg,
        vocab_size=len(teacher_tokenizer),
        excluded_token_ids=getattr(teacher_tokenizer, "all_special_ids", None),
    )

    overlap = compute_overlap(teacher_tokenizer, student_tokenizer)
    shared_mask = _build_student_shared_mask(student_tokenizer, overlap.shared_token_strings)
    student_vocab = len(student_tokenizer)
    map_tensor = overlap.source_to_target
    map_dev = map_tensor.to(accelerator.device)
    teacher_vocab = len(teacher_tokenizer)
    if map_dev.shape[0] < teacher_vocab:
        padded = torch.full((teacher_vocab,), -1, device=accelerator.device, dtype=torch.long)
        padded[: map_dev.shape[0]] = map_dev
        map_dev = padded
    else:
        map_dev = map_dev[:teacher_vocab]
    sentinel = student_vocab
    map_indices = map_dev.clone()
    map_indices[map_indices < 0] = sentinel

    base_model = load_causal_lm(cfg.student)
    base_model.resize_token_embeddings(len(student_tokenizer))
    student_model = PeftModel.from_pretrained(base_model, cfg.lora_dir)
    if hasattr(student_model, "config"):
        student_model.config.use_cache = False
    student_model.to(accelerator.device)
    student_model.eval()

    local_rows = traces[accelerator.process_index :: accelerator.num_processes]
    tmp_dir = cfg.output_path.parent / "_tmp_stage4"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rank_path = tmp_dir / f"rank_{accelerator.process_index:03d}.json"

    student_shared_mask = shared_mask
    local_values: List[EvalEntry] = []

    batch_size = max(1, cfg.batch_size)
    iterator = range(0, len(local_rows), batch_size)
    if accelerator.is_local_main_process:
        iterator = tqdm(iterator, total=(len(local_rows) + batch_size - 1) // batch_size, desc="Stage 4: watermark eval")

    for start in iterator:
        batch = local_rows[start : start + batch_size]
        prompts: List[str] = []
        texts: List[str] = []
        prompt_lengths: List[int] = []
        for row in batch:
            prompt_text = _prompt_from_row(builder, student_tokenizer, row, add_system=add_system_for_messages)
            response_text = row.get("response") or ""
            prompts.append(prompt_text)
            texts.append(prompt_text + response_text)
            prompt_lengths.append(len(prompt_text))
        # Tokenize with student tokenizer to get offsets/token ids.
        student_inputs = student_tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        attention_mask = student_inputs["attention_mask"].to(accelerator.device)
        input_ids = student_inputs["input_ids"].to(accelerator.device)
        offsets_batch = student_inputs["offset_mapping"]
        with torch.no_grad():
            logits = student_model(input_ids=input_ids, attention_mask=attention_mask).logits
        seq_lengths = attention_mask.sum(dim=1).to(torch.long)

        batch_bigrams: List[Tuple[int, int]] = []
        batch_student_positions: List[int] = []
        batch_sample_indices: List[int] = []

        for idx, row in enumerate(batch):
            text = texts[idx]
            # Teacher tokenization to find bigrams + offsets on teacher vocab.
            teacher_encoding = teacher_tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            teacher_ids = torch.tensor(teacher_encoding["input_ids"], dtype=torch.long)
            teacher_offsets = [
                (int(start), int(end)) for start, end in teacher_encoding["offset_mapping"]
            ]
            offsets_list = [
                (int(start), int(end))
                for start, end in offsets_batch[idx][: seq_lengths[idx]]
            ]
            alignment = _aligned_offsets(offsets_list, len(offsets_list))
            response_start = prompt_lengths[idx]
            for t_idx, (_, end) in enumerate(teacher_offsets):
                if end <= response_start:
                    continue
                if end not in alignment:
                    continue
                if t_idx < 1:
                    continue
                s_idx = alignment[end]
                if s_idx < 0 or s_idx >= seq_lengths[idx]:
                    continue
                bigram = (int(teacher_ids[t_idx - 1]), int(teacher_ids[t_idx]))
                batch_bigrams.append(bigram)
                batch_student_positions.append(int(s_idx))
                batch_sample_indices.append(int(idx))

        if not batch_bigrams:
            continue

        mask_chunk = max(1, cfg.mask_chunk)
        shared = student_shared_mask.to(accelerator.device)
        for offset in range(0, len(batch_bigrams), mask_chunk):
            bg_chunk = batch_bigrams[offset : offset + mask_chunk]
            sample_chunk = batch_sample_indices[offset : offset + mask_chunk]
            pos_chunk = batch_student_positions[offset : offset + mask_chunk]

            teacher_masks = hash_fn.mask_batch(bg_chunk, device=accelerator.device, dtype=torch.bool)
            teacher_vocab = teacher_masks.shape[1]
            map_slice = map_indices[:teacher_vocab]

            student_masks = torch.zeros(
                (teacher_masks.shape[0], student_vocab + 1),
                device=accelerator.device,
                dtype=torch.bool,
            )
            student_masks.scatter_(
                1, map_slice.unsqueeze(0).expand_as(teacher_masks), teacher_masks
            )
            student_masks = student_masks[:, :student_vocab]

            sample_idx_tensor = torch.tensor(sample_chunk, device=accelerator.device, dtype=torch.long)
            token_idx_tensor = torch.tensor(pos_chunk, device=accelerator.device, dtype=torch.long)
            selected_logits = logits[sample_idx_tensor, token_idx_tensor]

            probs = F.softmax(selected_logits, dim=-1)

            if cfg.mode == "open":
                shared_mass = (probs * shared).sum(dim=-1)
                green_mass = (probs * student_masks).sum(dim=-1)
                values = (green_mass / shared_mass).tolist()
                for bigram, value in zip(bg_chunk, values):
                    local_values.append((bigram, float(value)))
            else:
                samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
                shared_flags = shared[samples]
                hits = student_masks.gather(1, samples.unsqueeze(-1)).squeeze(-1).to(torch.float32)
                for bigram, is_shared, hit in zip(bg_chunk, shared_flags.tolist(), hits.tolist()):
                    if not is_shared:
                        continue  # discard samples outside the shared vocabulary
                    local_values.append((bigram, float(hit)))

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
            payload = {
                "num_measurements": 0,
                "mean": 0.0,
                "mode": cfg.mode,
                "supervision": cfg.supervision,
                "note": "no_measurements_collected",
            }
        else:
            # Deduplicate by bigram after sorting by bigram only.
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
                "mean": float(sum(values) / len(values)) if values else 0.0,
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
    """Build the CLI argument parser for Stage 4.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Stage 4 – watermark evaluation")
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
    parser.add_argument("--mask-chunk", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for Stage 4.

    Args:
        argv: Optional list of CLI arguments (defaults to sys.argv).

    Returns:
        None.
    """
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
        mask_chunk=args.mask_chunk,
    )
    run_stage4(cfg)


if __name__ == "__main__":
    main()
