# ⚠️ Old version of code for student accuracy and E[FPR] experiments ⚠️ 

This branch is based on an older development version of the codebase configured for a special cluster (frontier) and includes an additional pipeline stage for evaluating the student checkpoints on the GSM8K test set using the lm-evaluation-harness (stage 6 in `pipeline_frontier.sh`). 

It also includes special logic for running and plotting 100 trials of the same two watermarking settings and control (no-watermark) configuration to simulate detection attempts and develop ROC plots and compute E[FPR] (`launch_exps_frontier.py` and `pipeline_efpr_frontier.sh`). 

This code is included in the public repo on this branch mostly for transparency and therefore is separated from the main code which has been simplfied and cleaned for ease of use and understanding.

# Antidistillation Fingerprinting Supplementary

This repository contains the code for the paper **Antidistillation Fingerprinting**. It provides a standalone, stage-based pipeline that compares two fingerprinting techniques for language models: Antidistillation Fingerprinting (ads) and the fingerprinting scheme induced by Red-and-Green-List Watermarking (radioactive). The pipeline generates teacher traces, fine-tunes a student via LoRA, evaluates fingerprint statistics under multiple settings, and produces plots that relate p-values to teacher quality metrics.

## Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Quick Start](#quick-start)
4. [Pipeline Stages](#pipeline-stages)
5. [Configuration](#configuration)
6. [Outputs and Layout](#outputs-and-layout)
7. [Dataset Notes](#dataset-notes)
8. [Predefined Sweep Scripts](#predefined-sweep-scripts)

## Overview
- Orchestration entrypoint: `pipeline.sh`
- Stage scripts: `stages/stage0_hash.py` through `stages/stage5_plotting.py`
- Dataset providers: `data/gsm8k.py`, `data/oasst1.py`
- Hashing and watermark logic: `hashing.py`, `models/logits.py`, `stages/stage4_watermark_eval.py`

## Requirements
- Python 3.12 (see `pyproject.toml`)
- GPU environment with 8x NVIDIA H100s for the default pipeline
- Dependencies installed via `uv`:

```bash
uv sync
```

Notes:
- `pipeline.sh` invokes `uv run python` and `uv run accelerate`. If you do not use `uv`, replace `PY_CMD` and `ACC_CMD` in `pipeline.sh` accordingly.
- Accelerate config lives in `accelerate_config.yaml` (multi-GPU by default). Adjust `ACC_NUM_PROCS` in `pipeline.sh` if needed.

## Quick Start
```bash
./pipeline.sh
```

By default, this runs the full 8x H100 pipeline on GSM8K using:
- `METHOD=radioactive` with `DELTA=2` and `GAMMA=0.5`
- Teacher: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- Proxy: `Qwen/Qwen2.5-3B`
- Student: `meta-llama/Llama-3.2-3B`
- `NUM_EXAMPLES=1024`, `EPOCHS=3`, `STAGE1_BATCH=64`, `EVAL_BATCH=32`

## Pipeline Stages
1. Stage 0 (hash): sample or load a hash seed and gamma, write `hash_config.json`.
2. Stage 1 (teacher generation): generate traces with radioactive or ADS perturbations.
3. Stage 2 (teacher eval): compute teacher accuracy or NLL on traces.
4. Stage 3 (student finetune): LoRA SFT on traces with response-only loss.
5. Stage 4 (watermark eval): compute watermark statistics in open/closed and supervised/unsupervised modes.
6. Stage 5 (plotting): plot p-values against teacher metrics across variants.

## Configuration
`pipeline.sh` is driven by environment variables. Common ones include:
- `DATASET`, `SPLIT`, `NUM_EXAMPLES`
- `METHOD` (radioactive, ads, control), `DELTA`, `LAMBDA`, `GAMMA`
- `TEACHER_MODEL`, `PROXY_MODEL`, `STUDENT_MODEL` and their `*_DTYPE`, `*_PAD` overrides
- `STAGE1_BATCH`, `EVAL_BATCH`, `EFFECTIVE_FT_BATCH`, `GRAD_ACCUM`, `LEARNING_RATE`
- `EPOCHS`, `MAX_NEW_TOKENS`, `MAX_SEQ_LEN`
- `TRAIN_SEED`, `ALT_SEED`, `HASH_SEED`
`ACC_NUM_PROCS` in `pipeline.sh` controls the process count used by accelerate.

Sentinel files are written under `.sentinels` in each experiment directory. To rerun a stage, set the corresponding `FORCE_STAGE*` variable to `1` (for example, `FORCE_STAGE3=1`).

## Outputs and Layout
Each experiment is stored under:
```
experiments/{teacher_abbrev}_{proxy_abbrev}_{dataset}_n{N}/
```
Key subdirectories:
- `hash_seed/hash_config.json`
- `training_traces/{method_label}/traces.jsonl` and `metadata.json`
- `alternative_traces/{method_label}/traces.jsonl` and `metadata.json`
- `models/{student_tag}_{method_label}_lr{lr}_e{epochs}/student_lora/`
- `metrics/{student_tag}_{method_label}_lr{lr}_e{epochs}/watermark_*.json`
- `figures/` (Stage 5 plots)

## Dataset Notes
- GSM8K is loaded directly from Hugging Face via `datasets`.
- OASST1 requires a prebuilt JSONL of ChatML contexts. Generate it with:

```bash
uv run python data/oasst1.py --split train --output data/oasst1_chatml_messages.jsonl
```

Set `OASST1_PATH` to point to the JSONL if you store it elsewhere.

## Predefined Sweep Scripts
The following scripts run batches of pipeline sweeps for different deltas/lambdas:
- `run-llama-gsm8k.sh`
- `run-llama-oasst1.sh`
- `run-qwen-gsm8k.sh`
- `run-qwen-oasst1.sh`
