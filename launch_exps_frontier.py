# fmt: off
import os
from itertools import product, chain
import hashlib

LIST_CFGS = True
# LIST_CFGS = False

# WRITE_ONLY = True
WRITE_ONLY = False

RELAUNCH_ONLY = True
# RELAUNCH_ONLY = False

RELAUNCH_STATE_CHECK = "running"
# RELAUNCH_STATE_CHECK = "pending"

# TARGET_JOB_CT = 1
TARGET_JOB_CT = 1
# TARGET_JOB_CT = 10

LAUNCHER_FILEPATH = "$WRKSPC/llnl-tools/launch_frontier.py"

RCCL_INSTALL_DIR = "$WRKSPC/aws-ofi-rccl_frontier_uv_adsfp/lib"

# env
ENVIRONMENT = "$WRKSPC/frontier_uv_adsfp"
VENV = ".venv"
ENV_ACT_STYLE = "conda_activate_source_venv"
MODULES = "PrgEnv-gnu+craype-accel-amd-gfx90a+rocm/6.4.2"

# job qos
# QOS = "debug"
QOS = "normal"
TIME_LIMIT = 119  # in minutes
# REPETITIONS = 1
REPETITIONS = 1
# REPETITIONS = 30
DEPENDENCY = "singleton"


BASE_OUT_DIR = f"$WRKSPC/adsfp-root/antidistillation-fingerprinting/outputs"

# BASE_RUN_NAME = f"debug_interactive"

# BASE_RUN_NAME = f"repro_sweep_try0"
# BASE_RUN_NAME = f"repro_sweep_try1"
# BASE_RUN_NAME = f"repro_sweep_add_ctrl_eval"
# BASE_RUN_NAME = f"repro_sweep_qwen_fixes"
# BASE_RUN_NAME = f"debug_efpr_runs"
BASE_RUN_NAME = f"sweep_efpr_runs"

NODES = 1
GPUS_PER_NODE = 8
TASKS_PER_NODE = 1
CPUS_PER_TASK = 56

# HF_OFFLINE_ONLY = False
HF_OFFLINE_ONLY = True

# USE_LOCAL_MODELS = False
USE_LOCAL_MODELS = True

HF_OFFLINE_FLAGS = f"""\
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export USE_LOCAL_MODELS={1 if USE_LOCAL_MODELS else 0}
"""

STATIC_VARS = f"""\
export HF_HUB_CACHE=$WRKSPC/.cache/huggingface/hub
export HF_DATASETS_CACHE=$WRKSPC/.cache/huggingface/datasets
export HF_XET_CACHE=$WRKSPC/.cache/huggingface/xet
export TRITON_HOME=$WRKSPC/.triton
export TRITON_CACHE_DIR=$WRKSPC/.triton
export VLLM_CACHE_ROOT=$WRKSPC/.cache/vllm
export VLLM_CONFIG_ROOT=$WRKSPC/.config/vllm

{HF_OFFLINE_FLAGS if HF_OFFLINE_ONLY else ''}

export NUM_EXAMPLES=7473 # or 8192
export EPOCHS=1
export DATASET=gsm8k # or oasst1

# oasst1 specific potentially
# export STAGE1_BATCH=32
# export EVAL_BATCH=16
"""

# Cfgs
exp_list = [
    # ["pipeline_frontier.sh"],
    ["pipeline_efpr_frontier.sh"],
]

# NUM_TRIALS = None
# NUM_TRIALS = 10
# NUM_TRIALS = 20
# NUM_TRIALS = 30
# NUM_TRIALS = 35
# NUM_TRIALS = 50
# NUM_TRIALS = 60
# NUM_TRIALS = 75
NUM_TRIALS = 100

MIN_TRIAL_IDX = None
# MIN_TRIAL_IDX = 10
# MIN_TRIAL_IDX = 20
# MIN_TRIAL_IDX = 30
# MIN_TRIAL_IDX = 35
# MIN_TRIAL_IDX = 50
# MIN_TRIAL_IDX = 60
# MIN_TRIAL_IDX = 75
MIN_TRIAL_IDX = 85
# bc of determinism in seeding below, should be able to add more trials later if needed

# student model
sweep_hparam = [
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-3B",
]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))

# method and hparams
sweep_hparam = [
    # ["control",None,[None]],
    # ["ads","LAMBDA",[4,8,16,32,48,64,80,96,112,128,140,172,196,224,256,320,384,448,512]],
    # ["radioactive","DELTA",[1,2,3,4,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,11,12,13,14,15,16]],
    ["control",None,[None]],
    ["ads","LAMBDA",[140,]],
    ["radioactive","DELTA",[6,]],
]
exp_list = list(chain(*[[exp + hp_grp[:2]+[hp_val] for hp_grp in sweep_hparam for hp_val in hp_grp[2]] for exp in exp_list]))

# trials and seeds
if NUM_TRIALS is not None:
    # for each trial, we need to set NUM_TRIALS and TRIAL, and then randomly generate a TRAIN_SEED and ALT_SEED
    LAUNCHER_SEED_BASE = 123456789
    all_train_seeds = set()
    all_alt_seeds = set()
    final_exp_list = []
    for exp in exp_list:
        for trial in range(NUM_TRIALS):
            # generate train and alt seeds
            base_str = "_".join(map(str, exp + [trial, LAUNCHER_SEED_BASE]))
            hash_obj = hashlib.md5(base_str.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            train_seed = hash_int % 100000
            alt_seed = (hash_int // 100000) % 100000
            if train_seed in all_train_seeds:
                train_seed = (train_seed + trial) % 100000
            if alt_seed in all_alt_seeds:
                alt_seed = (alt_seed + trial) % 100000
            assert train_seed not in all_train_seeds, "Train seed collision despite resampling"
            assert alt_seed not in all_alt_seeds, "Alt seed collision despite resampling"
            all_train_seeds.add(train_seed)
            all_alt_seeds.add(alt_seed)
            if (MIN_TRIAL_IDX is not None) and (trial < MIN_TRIAL_IDX):
                continue
            final_exp_list.append(exp + ["NUM_TRIALS", NUM_TRIALS, "TRIAL", trial, "TRAIN_SEED", train_seed, "ALT_SEED", alt_seed])
else:
    final_exp_list = []
    # append dummy hparams to match unpacking
    for exp in exp_list:
        final_exp_list.append(exp + [None, None, None, None, None, None, None, None])

for exp in final_exp_list:
    print(exp)

total_launches = 0
total_skips = 0
total_relaunches = 0
tot_incl_repetitions = 0

unique_run_names = set()

# queue all jobs
for exp in final_exp_list:

    (
        driver_script,
        stud_model,
        method,
        hp0_name,
        hp0_value,
        hp1_name,
        hp1_value,
        hp2_name,
        hp2_value,
        hp3_name,
        hp3_value,
        hp4_name,
        hp4_value,
    ) = exp

    if driver_script == "pipeline_efpr_frontier.sh":
        assert hp1_name == "NUM_TRIALS", "E[FPR] runs require NUM_TRIALS to be set"
    if hp1_name == "NUM_TRIALS":
        assert driver_script == "pipeline_efpr_frontier.sh", "Only E[FPR] runs should set NUM_TRIALS"
    stud_model_str = stud_model.split("/")[-1].replace(".","-")

    run_name = driver_script.replace(".sh","").replace("_frontier","") 
    run_name += f"_{stud_model_str}_{method}"
    if hp0_name is not None:
        run_name += f"_{hp0_name}{hp0_value}"
    if hp1_name is not None:
        run_name += f"_{hp1_name}{hp1_value}"
    if hp2_name is not None:
        run_name += f"_{hp2_name}{hp2_value}"
    if hp3_name is not None:
        run_name += f"_{hp3_name}{hp3_value}"
    if hp4_name is not None:
        run_name += f"_{hp4_name}{hp4_value}"

    unique_run_names.add(run_name)

    hparam_vars = f"""\
export STUDENT_MODEL={stud_model}
export METHOD={method}
{f'export {hp0_name}={hp0_value}' if hp0_name is not None else ''}
{f'export {hp1_name}={hp1_value}' if hp1_name is not None else ''}
{f'export {hp2_name}={hp2_value}' if hp2_name is not None else ''}
{f'export {hp3_name}={hp3_value}' if hp3_name is not None else ''}
{f'export {hp4_name}={hp4_value}' if hp4_name is not None else ''}
"""

    # put together the actual "train.py" command
    launch_out_dir = f"{BASE_OUT_DIR}/{BASE_RUN_NAME}"

    custom_invocation = f"""\
{STATIC_VARS}
{hparam_vars}
./{driver_script}\
"""

    REMAINING_REPS = REPETITIONS
    if RELAUNCH_ONLY:
        # check squeue for a run with the same name
        sq_out = os.popen(f"squeue -u $USER -t {RELAUNCH_STATE_CHECK} -n {run_name}").read()
        # count the lines
        res = sq_out.strip("\n").split("\n")
        nlines = len(res)
        njobs = nlines - 1  # subtract header
        # assert nlines in [1,2], "Only cases I expect"
        if njobs >= TARGET_JOB_CT:
            jobhit = res[1]
            # print(f"Skipping {run_name}, {njobs} instances already in queue: \n{'\n'.join(res)}")
            print(f"Skipping {run_name}, {njobs} instances already in queue.")
            total_skips += 1
            continue
        else:
            # set repetitions to remaining needed
            REMAINING_REPS = TARGET_JOB_CT - njobs
            REMAINING_REPS = min(REMAINING_REPS, REPETITIONS)
            print(f"Relaunching {run_name} w/ {REMAINING_REPS} repetitions, as {'no' if njobs==0 else f'only {njobs}'} runs found in queue with same name and state={RELAUNCH_STATE_CHECK}.")
            total_relaunches += 1

    # make the complete launcher command
    command = f"""\
    python {LAUNCHER_FILEPATH} \
        --output_dir={launch_out_dir} \
        --rccl_installdir={RCCL_INSTALL_DIR} \
        --environment={ENVIRONMENT} \
        --venv={VENV} \
        --env_act_style={ENV_ACT_STYLE} \
        --modules="{MODULES}" \
        --qos={QOS} \
        --minutes={TIME_LIMIT} \
        --repetitions={REMAINING_REPS}{f' --dependency={DEPENDENCY}' if DEPENDENCY is not None else ''} \
        --nodes={NODES} \
        --gpus_per_node={GPUS_PER_NODE} \
        --ntasks_per_node={TASKS_PER_NODE} \
        --cpus_per_task={CPUS_PER_TASK} \
        --run_name={run_name} \
        --custom_invocation='{custom_invocation}' \
        --pass_run_name=False \
        {'--dryrun' if WRITE_ONLY else ''}
    """

    total_launches += 1
    tot_incl_repetitions += REMAINING_REPS    
    if not LIST_CFGS:
        os.system(command)
    else:
        print(run_name)
        # print(command)
if RELAUNCH_ONLY:
    print(f"Total skips/relaunches (incl repetitions): {total_skips}/{total_relaunches} ({tot_incl_repetitions})")
    assert (total_skips+total_relaunches) == len(unique_run_names), f"{total_launches} != {len(unique_run_names)} Jobs might be overwriting eachother, plz check that all hparams factor into naming."
else:
    print(f"Total launches (incl repetitions): {total_launches} ({tot_incl_repetitions})")
    assert total_launches == len(unique_run_names), f"{total_launches} != {len(unique_run_names)} Jobs might be overwriting eachother, plz check that all hparams factor into naming."
