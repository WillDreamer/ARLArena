#!/bin/bash
set -x
ulimit -n 131072
export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL

# ======================== GPU auto selection ========================
GPU_LIST=(4 5 6 7)  # <<<------  which GPUs to use, directly fill here
# Automatically concatenate CUDA_VISIBLE_DEVICES according to GPU_LIST
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
# Automatically detect the number of n_gpus_per_node
NUM_GPUS=${#GPU_LIST[@]}
echo "Detected ${NUM_GPUS} GPUs for this run"

MODEL=Qwen/Qwen3-0.6B

PORT=$(( ( RANDOM % 10000 +1000) ))
ray status >/dev/null 2>&1 || ray start --head --port $PORT --dashboard-host=0.0.0.0 --dashboard-port=7777 --include-dashboard=true

export RAY_TMPDIR="/data2/whx/ray_out"
rm -rf "$RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR"

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size 256 \
    --val_data_size 256 # evaluate 2 × val_data_size tasks during each iteration

python3 -m recipe.shop_agent.eval_shop_agent \
    data.train_files=$HOME/data/text/train.parquet \
    data.val_files=$HOME/data/text/test.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8\
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.99 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=1
