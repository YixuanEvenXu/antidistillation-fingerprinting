#!/bin/bash
set -euo pipefail

# ----------------------------------------------------------------------------
# End-to-end watermark pipeline orchestrator (radioactive vs ADS)
# Runs all four evaluations: (open/closed) x (supervised/unsupervised)
# using shared hashing + dual trace sets. Assumes 8xH100 GPUs by default.
# ----------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${ROOT_DIR}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
ACC_CONFIG_MULTI="${ROOT_DIR}/acc_config_gpu_multi.yaml"
ACC_CONFIG_SINGLE="${ROOT_DIR}/acc_config_gpu.yaml"
ACC_CONFIG="${ACC_CONFIG_MULTI}"
ACC_NUM_PROCS=8
if [[ "${ACCELERATE_SINGLE_GPU:-0}" == "1" ]]; then
    ACC_CONFIG="${ACC_CONFIG_SINGLE}"
    ACC_NUM_PROCS=1
fi

YELLOW='\033[0;33m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RESET='\033[0m'

READABLE_TIME() {
    local secs=$1
    printf "%02dh:%02dm:%02ds" $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
}

abbrev_model() {
    local name="$1"
    name="${name##*/}"
    local lowered="$(echo "$name" | tr '[:upper:]' '[:lower:]')"
    if [[ "$lowered" == *"deepseek-r1-distill"* ]]; then
        name="r1-distill-7b"
    fi
    name="$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:].-')"
    [[ -z "$name" ]] && name="model"
    echo "$name"
}

run_stage() {
    local label="$1"; shift
    local sentinel="$1"; shift
    local force="$1"; shift
    local -a cmd=("$@")
    if [[ -f "$sentinel" && "$force" != "1" ]]; then
        echo -e "${YELLOW}⏭️  Skipping ${label}: sentinel present.${RESET}"
        return 0
    fi
    echo -e "${CYAN}▶ ${label}${RESET}"
    echo "Command to be run: ${cmd[@]}"
    "${cmd[@]}"
    mkdir -p "$(dirname "$sentinel")"
    touch "$sentinel"
    echo -e "${GREEN}✅ ${label} completed.${RESET}"
}

# ----------------------------------------------------------------------------
# Configuration (environment overrides available)
# ----------------------------------------------------------------------------
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/experiments}"
DATASET="${DATASET:-gsm8k}"
SPLIT="${SPLIT:-train}"
METHOD="${METHOD:-radioactive}" # radioactive | ads
DELTA="${DELTA:-2}"              # required for radioactive
LAMBDA="${LAMBDA:-16}"             # required for ads
GAMMA="${GAMMA:-0.5}"
HASH_SEED="${HASH_SEED:-}"
TRAIN_SEED="${TRAIN_SEED:-42}"
ALT_SEED="${ALT_SEED:-43}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1024}"
EPOCHS="${EPOCHS:-3}"
TRIAL="${TRIAL:-0}"
NUM_TRIALS="${NUM_TRIALS:-1}"

USE_LOCAL_MODELS="${USE_LOCAL_MODELS:-0}"

TEACHER_MODEL="${TEACHER_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
TEACHER_DTYPE="${TEACHER_DTYPE:-bfloat16}"
TEACHER_PAD="${TEACHER_PAD:-}"
PROXY_MODEL="${PROXY_MODEL:-Qwen/Qwen2.5-3B}"
PROXY_DTYPE="${PROXY_DTYPE:-bfloat16}"
PROXY_PAD="${PROXY_PAD:-}"
STUDENT_MODEL="${STUDENT_MODEL:-meta-llama/Llama-3.2-3B}"
STUDENT_DTYPE="${STUDENT_DTYPE:-bfloat16}"
STUDENT_PAD="${STUDENT_PAD:-}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
STAGE1_BATCH="${STAGE1_BATCH:-64}"
EVAL_BATCH="${EVAL_BATCH:-32}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-32}"
EFFECTIVE_FT_BATCH="${EFFECTIVE_FT_BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_RANK="${LORA_RANK:-128}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
PLOT_LABELS="${PLOT_LABELS:-0}"

LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-8}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-gsm8k_cot}"
LM_EVAL_PROJECT="${LM_EVAL_PROJECT:-adsfp-evals}"
LM_EVAL_EXTRA_ARGS="${LM_EVAL_EXTRA_ARGS:-}"
if [[ "${STUDENT_MODEL,,}" == *"qwen"* ]]; then
    LM_EVAL_EXTRA_ARGS='--apply_chat_template --fewshot_as_multiturn'
fi

FT_BATCH=$(( EFFECTIVE_FT_BATCH / ACC_NUM_PROCS ))
if [[ $FT_BATCH -lt 1 ]]; then
    FT_BATCH=1
fi

# Derived experiment + directories
teacher_tag=$(abbrev_model "$TEACHER_MODEL")
proxy_tag=$(abbrev_model "$PROXY_MODEL")
student_tag=$(abbrev_model "$STUDENT_MODEL")
method_label="${METHOD}"
if [[ "$METHOD" == "radioactive" ]]; then
    method_label="${method_label}-delta${DELTA//./_}"
elif [[ "$METHOD" == "ads" ]]; then
    method_label="${method_label}-lambda${LAMBDA//./_}"
fi
# add the trial to differentiate the samples from different trials
# justifiy the trial name so that it sorts properly
# method_label="${method_label}-trial$(printf "%04d" "${TRIAL}")of$(printf "%04d" "${NUM_TRIALS}")"
method_label="${method_label}-trial$(printf "%04d" "${TRIAL}")" # might allow adding more trials later

# Now that tags are safely extracted, bc internet access on compute nodes is flaky ...
# if using local models convert to hardcoded paths.
if [[ "$USE_LOCAL_MODELS" == "1" ]]; then
    if [[ "$TEACHER_MODEL" == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" ]]; then
        TEACHER_MODEL="/lustre/orion/lrn089/scratch/jkirchen/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
    fi
    if [[ "$PROXY_MODEL" == "Qwen/Qwen2.5-3B" ]]; then
        PROXY_MODEL="/lustre/orion/lrn089/scratch/jkirchen/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
    fi
    if [[ "$STUDENT_MODEL" == "meta-llama/Llama-3.2-3B" ]]; then
        STUDENT_MODEL="/lustre/orion/lrn089/scratch/jkirchen/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062"
    elif [[ "$STUDENT_MODEL" == "Qwen/Qwen2.5-3B" ]]; then
        STUDENT_MODEL="/lustre/orion/lrn089/scratch/jkirchen/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
    fi
fi

EXP_DIR="${EXP_ROOT%/}/efpr_${teacher_tag}_${proxy_tag}_${DATASET}_n${NUM_EXAMPLES}"
HASH_DIR="${EXP_DIR}/hash_seed"
HASH_CFG="${HASH_DIR}/hash_config.json"

LR_TAG=$(LEARNING_RATE="$LEARNING_RATE" python3 - <<'PY'
import os
rate = os.environ.get("LEARNING_RATE", "")
try:
    val = float(rate)
    print(f"{val:g}")
except Exception:
    print(rate)
PY
)

TRAIN_TRACES_DIR="${EXP_DIR}/training_traces/${method_label}"
ALT_TRACES_DIR="${EXP_DIR}/alternative_traces/${method_label}"
TRAIN_TRACES_JSONL="${TRAIN_TRACES_DIR}/traces.jsonl"
ALT_TRACES_JSONL="${ALT_TRACES_DIR}/traces.jsonl"
TRAIN_TRACES_META="${TRAIN_TRACES_DIR}/metadata.json"
ALT_TRACES_META="${ALT_TRACES_DIR}/metadata.json"
TRAIN_TEACHER_EVAL="${TRAIN_TRACES_DIR}/teacher_eval.json"
ALT_TEACHER_EVAL="${ALT_TRACES_DIR}/teacher_eval.json"

MODEL_IDENTIFIER="${student_tag}_${method_label}_lr${LR_TAG}_e${EPOCHS}"
MODEL_DIR="${EXP_DIR}/models/${MODEL_IDENTIFIER}"
LORA_DIR="${MODEL_DIR}/student_lora"

METRICS_DIR="${EXP_DIR}/metrics/${MODEL_IDENTIFIER}"
METRIC_OPEN_SUP="${METRICS_DIR}/watermark_open_supervised.json"
METRIC_CLOSED_SUP="${METRICS_DIR}/watermark_closed_supervised.json"
METRIC_OPEN_UNSUP="${METRICS_DIR}/watermark_open_unsupervised.json"
METRIC_CLOSED_UNSUP="${METRICS_DIR}/watermark_closed_unsupervised.json"

mkdir -p "$TRAIN_TRACES_DIR" "$ALT_TRACES_DIR" "$MODEL_DIR" "$METRICS_DIR"
FIG_DIR="${EXP_DIR}/figures"
mkdir -p "$FIG_DIR"
SENTINELS_DIR="${EXP_DIR}/.sentinels"
mkdir -p "$SENTINELS_DIR"
LM_EVAL_DIR="${EXP_DIR}/lm_eval"
mkdir -p "$LM_EVAL_DIR"

PY_CMD=(python -u)
ACC_CMD=(accelerate launch --config_file "$ACC_CONFIG")
plot_label_flag=()
if [[ "${PLOT_LABELS}" == "1" ]]; then
    plot_label_flag=(--show-labels)
fi

# ----------------------------------------------------------------------------
# Stage 0 – shared hash
# ----------------------------------------------------------------------------
run_stage "Stage 0 – Hash" "$SENTINELS_DIR/stage0_hash.done" "${FORCE_STAGE0:-0}" \
    "${PY_CMD[@]}" stages/stage0_hash.py \
    --exp-dir "$HASH_DIR" \
    --gamma "$GAMMA" \
    ${HASH_SEED:+--seed "$HASH_SEED"} \
    --output "$HASH_CFG"

# ----------------------------------------------------------------------------
# Stage 1 – teacher generation (training traces)
# ----------------------------------------------------------------------------
stage1_train_args=(
    --dataset "$DATASET"
    --split "$SPLIT"
    --max-examples "$NUM_EXAMPLES"
    --teacher-model "$TEACHER_MODEL"
    --teacher-dtype "$TEACHER_DTYPE"
    --teacher-pad-token "$TEACHER_PAD"
    --proxy-model "$PROXY_MODEL"
    --proxy-dtype "$PROXY_DTYPE"
    --proxy-pad-token "$PROXY_PAD"
    --method "$METHOD"
    --hash-config "$HASH_CFG"
    --output "$TRAIN_TRACES_JSONL"
    --metadata "$TRAIN_TRACES_META"
    --batch-size "$STAGE1_BATCH"
    --seed "$TRAIN_SEED"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --repetition-penalty "$REPETITION_PENALTY"
)
if [[ "$METHOD" == "radioactive" ]]; then
    stage1_train_args+=(--delta "$DELTA")
else
    stage1_train_args+=(--lam "$LAMBDA")
fi
run_stage "Stage 1 – Teacher Generation (train traces)" "$SENTINELS_DIR/stage1_train_${method_label}_seed${TRAIN_SEED}.done" "${FORCE_STAGE1_TRAIN:-0}" \
    "${ACC_CMD[@]}" --num_processes "${ACC_NUM_PROCS}" stages/stage1_generate.py "${stage1_train_args[@]}"

# ----------------------------------------------------------------------------
# Stage 1 – alternative traces for unsupervised eval
# ----------------------------------------------------------------------------
stage1_alt_args=(
    --dataset "$DATASET"
    --split "$SPLIT"
    --max-examples "$NUM_EXAMPLES"
    --teacher-model "$TEACHER_MODEL"
    --teacher-dtype "$TEACHER_DTYPE"
    --teacher-pad-token "$TEACHER_PAD"
    --proxy-model "$PROXY_MODEL"
    --proxy-dtype "$PROXY_DTYPE"
    --proxy-pad-token "$PROXY_PAD"
    --method "$METHOD"
    --hash-config "$HASH_CFG"
    --output "$ALT_TRACES_JSONL"
    --metadata "$ALT_TRACES_META"
    --batch-size "$STAGE1_BATCH"
    --seed "$ALT_SEED"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --repetition-penalty "$REPETITION_PENALTY"
)
if [[ "$METHOD" == "radioactive" ]]; then
    stage1_alt_args+=(--delta "$DELTA")
else
    stage1_alt_args+=(--lam "$LAMBDA")
fi
run_stage "Stage 1 – Teacher Generation (alt traces)" "$SENTINELS_DIR/stage1_alt_${method_label}_seed${ALT_SEED}.done" "${FORCE_STAGE1_ALT:-0}" \
    "${ACC_CMD[@]}" --num_processes "${ACC_NUM_PROCS}" stages/stage1_generate.py "${stage1_alt_args[@]}"

# ----------------------------------------------------------------------------
# Stage 2 – teacher eval on training traces
# ----------------------------------------------------------------------------
if [[ "$DATASET" == "gsm8k" ]]; then
    run_stage "Stage 2 – Teacher Eval (train traces)" "$SENTINELS_DIR/stage2_train_${method_label}_seed${TRAIN_SEED}.done" "${FORCE_STAGE2_TRAIN:-0}" \
        "${ACC_CMD[@]}" --num_processes "${ACC_NUM_PROCS}" stages/stage2_teacher_eval.py \
        --traces "$TRAIN_TRACES_JSONL" \
        --teacher-model "$TEACHER_MODEL" \
        --teacher-dtype "$TEACHER_DTYPE" \
        --teacher-pad-token "$TEACHER_PAD" \
        --batch-size "$EVAL_BATCH" \
        --max-answer-tokens "$MAX_ANSWER_TOKENS" \
        --seed "$TRAIN_SEED" \
        --output "$TRAIN_TEACHER_EVAL" \
        --dataset "$DATASET"
else
    echo -e "${YELLOW}⏭️  Skipping Stage 2 (train): dataset ${DATASET} not supported for answer-forced eval.${RESET}"
fi

# ----------------------------------------------------------------------------
# Stage 2 – teacher eval on alternative traces
# ----------------------------------------------------------------------------
if [[ "$DATASET" == "gsm8k" ]]; then
    run_stage "Stage 2 – Teacher Eval (alt traces)" "$SENTINELS_DIR/stage2_alt_${method_label}_seed${ALT_SEED}.done" "${FORCE_STAGE2_ALT:-0}" \
        "${ACC_CMD[@]}" --num_processes "${ACC_NUM_PROCS}" stages/stage2_teacher_eval.py \
        --traces "$ALT_TRACES_JSONL" \
        --teacher-model "$TEACHER_MODEL" \
        --teacher-dtype "$TEACHER_DTYPE" \
        --teacher-pad-token "$TEACHER_PAD" \
        --batch-size "$EVAL_BATCH" \
        --max-answer-tokens "$MAX_ANSWER_TOKENS" \
        --seed "$ALT_SEED" \
        --output "$ALT_TEACHER_EVAL" \
        --dataset "$DATASET"
else
    echo -e "${YELLOW}⏭️  Skipping Stage 2 (alt): dataset ${DATASET} not supported for answer-forced eval.${RESET}"
fi

# ----------------------------------------------------------------------------
# Stage 3 – Student finetune (training traces)
# ----------------------------------------------------------------------------
run_stage "Stage 3 – Student Finetune" "$SENTINELS_DIR/stage3_${student_tag}_${method_label}_lr${LR_TAG}_e${EPOCHS}.done" "${FORCE_STAGE3:-0}" \
    "${ACC_CMD[@]}" --num_processes "${ACC_NUM_PROCS}" stages/stage3_finetune.py \
    --traces "$TRAIN_TRACES_JSONL" \
    --student-model "$STUDENT_MODEL" \
    --student-dtype "$STUDENT_DTYPE" \
    --student-pad-token "$STUDENT_PAD" \
    --output-dir "$LORA_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$FT_BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    --learning-rate "$LEARNING_RATE" \
    --rank "$LORA_RANK" \
    --alpha "$LORA_ALPHA" \
    --dropout "$LORA_DROPOUT" \
    --seed "$TRAIN_SEED" \
    --max-seq-length "$MAX_SEQ_LEN" \
    --dataset "$DATASET"

# ----------------------------------------------------------------------------
# Stage 4 – Watermark evals
# ----------------------------------------------------------------------------
run_stage "Stage 4 – Watermark Eval (open, supervised)" "$SENTINELS_DIR/stage4_open_sup_${student_tag}_${method_label}_lr${LR_TAG}_e${EPOCHS}.done" "${FORCE_STAGE4_OPEN_SUP:-0}" \
    "${ACC_CMD[@]}" --num_processes "${ACC_NUM_PROCS}" stages/stage4_watermark_eval.py \
    --traces "$TRAIN_TRACES_JSONL" \
    --hash-config "$HASH_CFG" \
    --teacher-model "$TEACHER_MODEL" \
    --teacher-dtype "$TEACHER_DTYPE" \
    --teacher-pad-token "$TEACHER_PAD" \
    --student-model "$STUDENT_MODEL" \
    --student-dtype "$STUDENT_DTYPE" \
    --student-pad-token "$STUDENT_PAD" \
    --lora-dir "$LORA_DIR" \
    --mode "open" \
    --supervision "supervised" \
    --output "$METRIC_OPEN_SUP" \
    --batch-size "$EVAL_BATCH" \
    --seed "$TRAIN_SEED" \
    --dataset "$DATASET"

run_stage "Stage 4 – Watermark Eval (closed, supervised)" "$SENTINELS_DIR/stage4_closed_sup_${student_tag}_${method_label}_lr${LR_TAG}_e${EPOCHS}.done" "${FORCE_STAGE4_CLOSED_SUP:-0}" \
    "${ACC_CMD[@]}" --num_processes "${ACC_NUM_PROCS}" stages/stage4_watermark_eval.py \
    --traces "$TRAIN_TRACES_JSONL" \
    --hash-config "$HASH_CFG" \
    --teacher-model "$TEACHER_MODEL" \
    --teacher-dtype "$TEACHER_DTYPE" \
    --teacher-pad-token "$TEACHER_PAD" \
    --student-model "$STUDENT_MODEL" \
    --student-dtype "$STUDENT_DTYPE" \
    --student-pad-token "$STUDENT_PAD" \
    --lora-dir "$LORA_DIR" \
    --mode "closed" \
    --supervision "supervised" \
    --output "$METRIC_CLOSED_SUP" \
    --batch-size "$EVAL_BATCH" \
    --seed "$TRAIN_SEED" \
    --dataset "$DATASET"

run_stage "Stage 4 – Watermark Eval (open, unsupervised)" "$SENTINELS_DIR/stage4_open_unsup_${student_tag}_${method_label}_lr${LR_TAG}_e${EPOCHS}.done" "${FORCE_STAGE4_OPEN_UNSUP:-0}" \
    "${ACC_CMD[@]}" --num_processes "${ACC_NUM_PROCS}" stages/stage4_watermark_eval.py \
    --traces "$ALT_TRACES_JSONL" \
    --hash-config "$HASH_CFG" \
    --teacher-model "$TEACHER_MODEL" \
    --teacher-dtype "$TEACHER_DTYPE" \
    --teacher-pad-token "$TEACHER_PAD" \
    --student-model "$STUDENT_MODEL" \
    --student-dtype "$STUDENT_DTYPE" \
    --student-pad-token "$STUDENT_PAD" \
    --lora-dir "$LORA_DIR" \
    --mode "open" \
    --supervision "unsupervised" \
    --output "$METRIC_OPEN_UNSUP" \
    --batch-size "$EVAL_BATCH" \
    --seed "$ALT_SEED" \
    --dataset "$DATASET"

run_stage "Stage 4 – Watermark Eval (closed, unsupervised)" "$SENTINELS_DIR/stage4_closed_unsup_${student_tag}_${method_label}_lr${LR_TAG}_e${EPOCHS}.done" "${FORCE_STAGE4_CLOSED_UNSUP:-0}" \
    "${ACC_CMD[@]}" --num_processes "${ACC_NUM_PROCS}" stages/stage4_watermark_eval.py \
    --traces "$ALT_TRACES_JSONL" \
    --hash-config "$HASH_CFG" \
    --teacher-model "$TEACHER_MODEL" \
    --teacher-dtype "$TEACHER_DTYPE" \
    --teacher-pad-token "$TEACHER_PAD" \
    --student-model "$STUDENT_MODEL" \
    --student-dtype "$STUDENT_DTYPE" \
    --student-pad-token "$STUDENT_PAD" \
    --lora-dir "$LORA_DIR" \
    --mode "closed" \
    --supervision "unsupervised" \
    --output "$METRIC_CLOSED_UNSUP" \
    --batch-size "$EVAL_BATCH" \
    --seed "$ALT_SEED" \
    --dataset "$DATASET"

# ----------------------------------------------------------------------------
# Stage 5 – plotting (all variants)
# ----------------------------------------------------------------------------
run_stage "Stage 5 – Plotting" "$SENTINELS_DIR/stage5_${student_tag}_${method_label}_lr${LR_TAG}_e${EPOCHS}.done" "${FORCE_STAGE5:-0}" \
    "${PY_CMD[@]}" stages/stage5_plotting.py \
    --exp-dir "$EXP_DIR" \
    --fig-dir "$FIG_DIR" \
    --student-tag "$student_tag" \
    --lr "$LR_TAG" \
    --epochs "$EPOCHS" \
    "${plot_label_flag[@]}" \
    --variants open_supervised open_unsupervised closed_supervised closed_unsupervised


echo -e "${GREEN}Pipeline complete. Results under ${EXP_DIR}.${RESET}"