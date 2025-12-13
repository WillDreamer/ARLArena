set -x
# ======================== GPU auto selection ========================
GPU_LIST=(6 7)  # <<<------  which GPUs to use, directly fill here
# Automatically concatenate CUDA_VISIBLE_DEVICES according to GPU_LIST
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
# Automatically detect the number of n_gpus_per_node
NUM_GPUS=${#GPU_LIST[@]}
echo "NUM_GPUS=${NUM_GPUS}"
echo "Detected ${NUM_GPUS} GPUs for this run"

source /data1/xw27/miniconda3/etc/profile.d/conda.sh
cd /data1/xw27/agent/ARLArena
conda activate agentrl_science_async
# ======================== Hyper-parameters ========================
# dataset_name=deepmath_torl # or math_torl_offical to use torl training dat
train_data=[/data1/xw27/agent/ARLArena/datasets/simplelr_math_35/train.parquet,/data1/xw27/agent/ARLArena/datasets/deepscaler/train.parquet]
val_data=[/data1/xw27/agent/ARLArena/datasets/simplelr_math_35/test.parquet,/data1/xw27/agent/ARLArena/datasets/deepscaler/aime.parquet,/data1/xw27/agent/ARLArena/datasets/deepscaler/aime25.parquet,/data1/xw27/agent/ARLArena/datasets/deepscaler/olympiad.parquet,/data1/xw27/agent/ARLArena/datasets/deepscaler/math.parquet]

# train_data=$(pwd)/data/${dataset_name}/train.parquet
# val_data=[$(pwd)/data/${dataset_name}/test.parquet,\
# $(pwd)/data/${dataset_name}/math500_test.parquet,\
# $(pwd)/data/${dataset_name}/aime24_test.parquet,\
# $(pwd)/data/${dataset_name}/aime25_test.parquet]
model_name=Qwen/Qwen3-1.7B-Base
rl_alg=grpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective
n_gpus_per_node=$NUM_GPUS
n_nodes=1
n=8
batch_size=512
ppo_mini_batch_size=128
max_prompt_length=1024
max_response_length=4096
max_num_seqs=8
max_obs_length=256
temperature=1.0
top_p=1.0
enable_agent=True # enable agent for tool use
use_kl_loss=False
strategy="fsdp"
action_stop_tokens='```output'
max_turns=5
kl_loss_coef=0.0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=1e-6
reward_manager=torl
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
tensor_model_parallel_size=1
gpu_memory_utilization=0.4 # higher gpu_memory_utilization will likely cause the vllm to OOM and get stuck, so set it to a lower value like 0.4 or 0.5
do_offload=True # control actor's fsdp.[param|optimizer]_offload and actor_rollout_ref.rollout.fsdp.[param|optimizer]_offload; if gpu_memory_utilization is set to > 0.6, then do_offload should be set to True otherwise it will cause OOM
use_dynamic_bsz=True # faster
ulysses_sequence_parallel_size=1 # set to 1 for normal verl behavior, otherwise it will cause OOM
fsdp_size=-1
additional_eos_token_ids=[151645] # <|im_end|> token id
mask_observations=True # mask observations for kl loss and gradient descent
enable_mtrl=False # enable multi-turn training
max_action_length=2048
val_before_train=False
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name_postfix="acc-only"
if [ "$enable_agent" = "True" ]; then
    run_name="${reward_manager}-${strategy}-agent-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
else
    run_name="${reward_manager}-${strategy}-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
fi
checkpoint_path=/local/xw27/ARLArena/outputs_$run_name
rollout_data_dir=/local/xw27/ARLArena/rollout_data_$run_name
mkdir -p $checkpoint_path
mkdir -p $rollout_data_dir
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=INFO
export VLLM_USE_V1=1
rollout_mode='async'

RAY_TMP=/data1/xw27/ARLArena/outputs
mkdir -p $RAY_TMP
export RAY_TMPDIR="$RAY_TMP"
export TMPDIR="$RAY_TMP"

# Start a new Ray head node on a separate port to avoid conflicts with existing Ray instances
RAY_HEAD_PORT=2039  # Default Ray head port, change if needed
RAY_HEAD_DASHBOARD_PORT=2040
# RAY_OBJECT_STORE_PORT=10002  # Default object store port
# RAY_NODE_MANAGER_PORT=10003  # Default node manager port
# RAY_GCS_SERVER_PORT=10004  # Default GCS server port

# Kill any existing Ray processes on these ports (optional, for safety)
# ray stop --force 2>/dev/null || true

# Start a new Ray head node
echo "Starting new Ray head node on port $RAY_HEAD_PORT..."
ray start --head --port=$RAY_HEAD_PORT --dashboard-port=$RAY_HEAD_DASHBOARD_PORT 


# Set Ray address to connect to the new head node
export RAY_ADDRESS="127.0.0.1:$RAY_HEAD_PORT"
echo "Ray head node started with PID $ray_head_pid on $RAY_ADDRESS"

export WANDB_PROMPT_VERSION="math_agent_async"
export WANDB_PROJECT="${WANDB_PROMPT_VERSION}"
WANDB_API_KEY="09286f9b4dcf8784b832ad623eb07a6d5541f59a" # Modify your wandb key


# temp file for action tokens as verl cannot pass special strs as params
action_stop_tokens_file="$(pwd)$(mktemp)"
mkdir -p $(dirname $action_stop_tokens_file)
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

host=$(hostname -i | awk '{print $1}')
# port=$(shuf -i 30000-31000 -n 1)
port=2041
tool_server_url=http://$host:$port/get_observation
python -m recipe.math_agent_async.servers.serve --host $host --port $port --tool_type "ipython_code" --workers_per_tool 4 > logs/tool_server.log &
server_pid=$!


export RAY_OVERRIDE_RESOURCES='{"num_cpus":64,"num_gpus":'${NUM_GPUS}'}'
PYTHONUNBUFFERED=1 python3 -m recipe.math_agent_async.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=1024 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    +actor_rollout_ref.agent.retokenization=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=$max_num_seqs \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$reward_manager \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=$val_before_train \
    trainer.default_hdfs_dir=null \
    trainer.rollout_data_dir=$rollout_data_dir \
    trainer.validation_data_dir=$val_data_dir \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=10


pkill -P -9 $server_pid
kill -9 $kill $server_pid