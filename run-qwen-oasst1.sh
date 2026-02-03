# Sweep pipeline runs for a Qwen student on OASST1.
# Inputs: environment variables exported below.
# Output: executes pipeline.sh across delta/lambda settings.
export STUDENT_MODEL=Qwen/Qwen2.5-3B
export NUM_EXAMPLES=8192
export EPOCHS=1
export DATASET=oasst1
export STAGE1_BATCH=32
export EVAL_BATCH=16

export METHOD=radioactive
export DELTA=1
./pipeline.sh
export DELTA=2
./pipeline.sh
export DELTA=3
./pipeline.sh
export DELTA=4
./pipeline.sh
export DELTA=5
./pipeline.sh
export DELTA=6
./pipeline.sh
export DELTA=7
./pipeline.sh
export DELTA=8
./pipeline.sh
export DELTA=9
./pipeline.sh
export DELTA=10
./pipeline.sh
export DELTA=11
./pipeline.sh
export DELTA=12
./pipeline.sh
export DELTA=13
./pipeline.sh
export DELTA=14
./pipeline.sh
export DELTA=15
./pipeline.sh
export DELTA=16
./pipeline.sh

export METHOD=control
./pipeline.sh

export STAGE1_BATCH=16
export EVAL_BATCH=8
export METHOD=ads
export LAMBDA=4
./pipeline.sh
export LAMBDA=8
./pipeline.sh
export LAMBDA=16
./pipeline.sh
export LAMBDA=24
./pipeline.sh
export LAMBDA=32
./pipeline.sh
export LAMBDA=48
./pipeline.sh
export LAMBDA=64
./pipeline.sh
export LAMBDA=96
./pipeline.sh
export LAMBDA=128
./pipeline.sh
export LAMBDA=192
./pipeline.sh
export LAMBDA=256
./pipeline.sh
export LAMBDA=320
./pipeline.sh
export LAMBDA=384
./pipeline.sh
export LAMBDA=448
./pipeline.sh
export LAMBDA=512
./pipeline.sh
export LAMBDA=640
./pipeline.sh
export LAMBDA=768
./pipeline.sh
export LAMBDA=896
./pipeline.sh
export LAMBDA=1024
./pipeline.sh
