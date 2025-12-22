#!/bin/bash
set -x

[ -d "$HOME/data" ] || mkdir -p "$HOME/data"
SAVE_PATH=../../checkpoints/
[ -d "$SAVE_PATH" ] || mkdir -p "$SAVE_PATH"



TRAIN_DATA=$HOME/ARLArena/datasets/train.parquet
EVAL_DATA=$HOME/ARLArena/datasets/train.parquet
wget -O $HOME/data/train.parquet https://huggingface.co/datasets/willamazon1/alfworld_sft/resolve/main/train.parquet


TRAIN_DATA=$HOME/data/train.parquet
EVAL_DATA=$HOME/data/train.parquet
TASK=Alfworld
MODEL_PATH=Qwen/Qwen3-4B
MODEL_SHORT="${MODEL_PATH##*/}"

WANDB_API_KEY="ba70fcbc92808cc7a1750dd80ac3908295e6854f" # Modify your wandb key
# ============================ Preparation ============================
# Login to WandB (if API key is provided)
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=163840 \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=$MODEL_PATH \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=sft-qwen-3-$MODEL_SHORT\
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    ulysses_sequence_parallel_size=8 \
    use_remove_padding=true
