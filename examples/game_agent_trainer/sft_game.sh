#!/bin/bash
set -x

[ -d "$HOME/data" ] || mkdir -p "$HOME/data"
SAVE_PATH=../../checkpoints/
[ -d "$SAVE_PATH" ] || mkdir -p "$SAVE_PATH"

# wget -O $HOME/data/train.parquet https://huggingface.co/datasets/willamazon1/webshop_sft/resolve/main/train.parquet


TRAIN_DATA=$HOME/data/sft/train.parquet
EVAL_DATA=$HOME/data/sft/train.parquet
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
MODEL_SHORT="${MODEL_PATH##*/}"

WANDB_API_KEY="a7be45528eb0e10c37315748df65f21e5c09d71c" # Modify your wandb key
# ============================ Preparation ============================
# Login to WandB (if API key is provided)
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m recipe.game_agent.fsdp_sft_mm_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=16384 \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=$MODEL_PATH \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=sft-qwen-2.5-$MODEL_SHORT \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=2 \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size=8 \
    use_remove_padding=true
