"""Full pipeline orchestration for watermark experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    ExperimentLayout,
    FinetuneConfig,
    GenerationConfig,
    HashStageConfig,
    ModelSpec,
    TeacherEvalConfig,
    WatermarkEvalConfig,
)
from hashing import load_hash_config
from stages.stage0_hash import run_stage0
from stages.stage1_generate import run_stage1
from stages.stage2_teacher_eval import run_stage2
from stages.stage3_finetune import run_stage3
from stages.stage4_watermark_eval import run_stage4


def _scale_label(method: str, delta: float | None, lam: float | None) -> str:
    if method == "radioactive":
        if delta is None:
            raise ValueError("delta must be provided for radioactive runs")
        return f"radioactive-delta{delta:g}".replace(".", "_")
    if method == "ads":
        if lam is None:
            raise ValueError("lambda must be provided for ADS runs")
        return f"ads-lambda{lam:g}".replace(".", "_")
    raise ValueError(f"Unknown method: {method}")


def orchestrate(args: argparse.Namespace) -> None:
    method_label = _scale_label(args.method, args.delta, args.lam)
    layout = ExperimentLayout(
        root=args.exp_root,
        dataset=args.dataset,
        num_examples=args.num_examples,
        teacher_model=args.teacher_model,
        proxy_model=args.proxy_model,
    )

    hash_path = layout.hash_path()
    train_traces_dir = layout.trace_dir("training", method_label)
    alt_traces_dir = layout.trace_dir("alternative", method_label)
    train_traces_jsonl = train_traces_dir / "traces.jsonl"
    alt_traces_jsonl = alt_traces_dir / "traces.jsonl"
    train_meta = train_traces_dir / "metadata.json"
    alt_meta = alt_traces_dir / "metadata.json"
    train_teacher_eval = train_traces_dir / "teacher_eval.json"
    alt_teacher_eval = alt_traces_dir / "teacher_eval.json"

    model_dir = layout.model_dir(args.student_model, method_label, args.learning_rate, args.epochs)
    lora_dir = model_dir / "student_lora"
    metrics_dir = layout.metrics_dir(args.student_model, method_label, args.learning_rate, args.epochs)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for folder in [layout.hash_dir(), train_traces_dir, alt_traces_dir, model_dir, metrics_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    teacher_spec = ModelSpec(name=args.teacher_model, dtype=args.teacher_dtype, pad_token=args.teacher_pad_token)
    proxy_spec = ModelSpec(name=args.proxy_model, dtype=args.proxy_dtype, pad_token=args.proxy_pad_token)
    student_spec = ModelSpec(name=args.student_model, dtype=args.student_dtype, pad_token=args.student_pad_token)

    # Stage 0 (shared hash)
    run_stage0(
        HashStageConfig(
            teacher=teacher_spec,
            exp_dir=layout.hash_dir(),
            seed=args.hash_seed,
            gamma=args.gamma,
            output_file=hash_path,
        )
    )
    hash_cfg = load_hash_config(hash_path)

    # Stage 1 (training traces)
    train_gen_cfg = GenerationConfig(
        dataset=args.dataset,
        split=args.split,
        max_examples=args.num_examples,
        teacher=teacher_spec,
        proxy=proxy_spec,
        method=args.method,  # type: ignore[arg-type]
        delta=args.delta,
        lam=args.lam,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        batch_size=args.batch_size,
        seed=args.train_seed,
        output_jsonl=train_traces_jsonl,
        metadata_path=train_meta,
    )
    run_stage1(train_gen_cfg, hash_cfg)

    # Stage 1 (alternative traces for unsupervised eval)
    alt_gen_cfg = GenerationConfig(
        dataset=args.dataset,
        split=args.split,
        max_examples=args.num_examples,
        teacher=teacher_spec,
        proxy=proxy_spec,
        method=args.method,  # type: ignore[arg-type]
        delta=args.delta,
        lam=args.lam,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        batch_size=args.batch_size,
        seed=args.alt_seed,
        output_jsonl=alt_traces_jsonl,
        metadata_path=alt_meta,
    )
    run_stage1(alt_gen_cfg, hash_cfg)

    # Stage 2 (teacher eval on training traces)
    if args.dataset in {"gsm8k", "oasst1"}:
        train_eval_cfg = TeacherEvalConfig(
            dataset=args.dataset,
            teacher=teacher_spec,
            traces_jsonl=train_traces_jsonl,
            output_path=train_teacher_eval,
            batch_size=args.eval_batch_size,
            max_answer_tokens=args.max_answer_tokens,
            seed=args.train_seed,
        )
        run_stage2(train_eval_cfg)

    # Stage 2 (teacher eval on alternative traces)
    if args.dataset in {"gsm8k", "oasst1"}:
        alt_eval_cfg = TeacherEvalConfig(
            dataset=args.dataset,
            teacher=teacher_spec,
            traces_jsonl=alt_traces_jsonl,
            output_path=alt_teacher_eval,
            batch_size=args.eval_batch_size,
            max_answer_tokens=args.max_answer_tokens,
            seed=args.alt_seed,
        )
        run_stage2(alt_eval_cfg)

    # Stage 3 (finetune student on training traces)
    finetune_cfg = FinetuneConfig(
        dataset=args.dataset,
        student=student_spec,
        traces_jsonl=train_traces_jsonl,
        output_dir=lora_dir,
        epochs=args.epochs,
        batch_size=args.ft_batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        seed=args.train_seed,
        max_seq_length=args.max_seq_length,
    )
    run_stage3(finetune_cfg)

    # Stage 4 – four evaluations
    eval_jobs = [
        ("open", "supervised", train_traces_jsonl, metrics_dir / "watermark_open_supervised.json"),
        ("closed", "supervised", train_traces_jsonl, metrics_dir / "watermark_closed_supervised.json"),
        ("open", "unsupervised", alt_traces_jsonl, metrics_dir / "watermark_open_unsupervised.json"),
        ("closed", "unsupervised", alt_traces_jsonl, metrics_dir / "watermark_closed_unsupervised.json"),
    ]
    for mode, supervision, traces_path, output_path in eval_jobs:
        watermark_cfg = WatermarkEvalConfig(
            dataset=args.dataset,
            teacher=teacher_spec,
            student=student_spec,
            hash_config=hash_path,
            traces_jsonl=traces_path,
            lora_dir=lora_dir,
            mode=mode,  # type: ignore[arg-type]
            supervision=supervision,  # type: ignore[arg-type]
            output_path=output_path,
            batch_size=args.eval_batch_size,
            seed=args.train_seed if supervision == "supervised" else args.alt_seed,
        )
        run_stage4(watermark_cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end watermark pipeline")
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--method", choices=["radioactive", "ads"], required=True)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--num-examples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--hash-seed", type=int, default=None, help="Seed for hash generation (shared across traces)")
    parser.add_argument("--train-seed", type=int, default=42, help="Stage 1 seed for training traces")
    parser.add_argument("--alt-seed", type=int, default=43, help="Stage 1 seed for alternative traces")

    parser.add_argument("--teacher-model", type=str, required=True)
    parser.add_argument("--teacher-dtype", type=str, default="bfloat16")
    parser.add_argument("--teacher-pad-token", type=str, default=None)

    parser.add_argument("--proxy-model", type=str, required=True)
    parser.add_argument("--proxy-dtype", type=str, default="bfloat16")
    parser.add_argument("--proxy-pad-token", type=str, default=None)

    parser.add_argument("--student-model", type=str, required=True)
    parser.add_argument("--student-dtype", type=str, default="bfloat16")
    parser.add_argument("--student-pad-token", type=str, default=None)

    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8, help="Stage 1 batch size")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-answer-tokens", type=int, default=32)

    parser.add_argument("--ft-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int, default=4096)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    orchestrate(args)


if __name__ == "__main__":
    main()
