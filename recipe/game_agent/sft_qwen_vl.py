import os
import sys
import importlib

# IMPORTANT: Add project root to Python path to use local verl instead of installed package
project_root = "/home/ubuntu/Yidan/ARLArena"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

# Change to the project root directory for Hydra config resolution
original_cwd = os.getcwd()
if os.getcwd() != project_root:
    os.chdir(project_root)
    print(f"Changed working directory from {original_cwd} to: {os.getcwd()}")

import pandas as pd
import torch
from huggingface_hub import notebook_login
from hydra import initialize, compose
from omegaconf import OmegaConf
from peft import LoraConfig
from torch.utils.data import Dataset
from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig

from recipe.game_agent.multiturn_sft_mm_dataset import MultiTurnSFTDataset
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer, hf_processor

# 设置可见 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 读取并查看 SFT parquet 数据
df = pd.read_parquet("/home/ubuntu/Yidan/ARLArena/checkpoints/sft/parquet/train.parquet")

print(df.head())       # 查看前几行
print(df.columns)      # 查看列名
print(df.info())       # 查看结构和类型

print("Up sft parquet--------------------------------Down visual parquet")

df_2 = pd.read_parquet("/home/ubuntu/data/visual/train.parquet")

print(df["turns"][0][0]["inputs"][0].keys())       # 查看前几行
print(df["turns"][0][0])
# print(df["turns"][0][0]["inputs"][0]["role"])
# print(df.columns)      # 查看列名
# print(df.info())       # 查看结构和类型


# Verify we're using the local verl
try:
    import verl
    print(f"Using verl from: {verl.__file__}")
except Exception as e:
    print(f"Warning: Could not import verl: {e}")

with initialize(config_path="config", version_base=None):
    cfg = compose(config_name="sft_trainer.yaml")

cfg.model.partial_pretrain = "Qwen/Qwen3-VL-4B-Instruct"
cfg.data.max_length = 10000

data_paths = ["/home/ubuntu/Yidan/ARLArena/checkpoints/sft/parquet/train.parquet"]
data_config = cfg.data

df_sample = pd.read_parquet(data_paths[0])
print(f"Data columns: {df_sample.columns.tolist()}")

trust_remote_code = cfg.data.get("trust_remote_code", False)
local_path = copy_to_local(
    cfg.model.partial_pretrain,
    use_shm=cfg.model.get("use_shm", False),
)

tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

train_dataset = MultiTurnSFTDataset(
    parquet_files=data_paths,
    tokenizer=tokenizer,
    processor=processor,
    config=data_config,
)

print(train_dataset.column_names)

model_name = "Qwen/Qwen3-VL-4B-Instruct"  # "Qwen/Qwen3-VL-8B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,                        # Load the model in 4-bit precision to save memory
        bnb_4bit_compute_dtype=torch.float16,     # Data type used for internal computations in quantization
        bnb_4bit_use_double_quant=True,           # Use double quantization to improve accuracy
        bnb_4bit_quant_type="nf4",                # Type of quantization. "nf4" is recommended for recent LLMs
    ),
)

# You may need to update `target_modules` depending on the architecture of your chosen model.
# For example, different VLMs might have different attention/projection layer names.
peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["down_proj", "o_proj", "k_proj", "q_proj", "gate_proj", "up_proj", "v_proj"],
)

output_dir = "Qwen3-VL-4B-Instruct-trl-sft"

# Configure training arguments using SFTConfig
training_args = SFTConfig(
    # Training schedule / optimization
    # num_train_epochs=1,
    max_steps=15,                       # Number of dataset passes. For full trainings, use `num_train_epochs` instead
    per_device_train_batch_size=2,      # Batch size per GPU/CPU
    gradient_accumulation_steps=8,      # Gradients are accumulated over multiple steps → effective batch size = 4 * 8 = 32
    warmup_steps=5,                     # Gradually increase LR during first N steps
    learning_rate=2e-4,                 # Learning rate for the optimizer
    optim="adamw_8bit",                 # Optimizer
    max_length=None,                    # For VLMs, truncating may remove image tokens, leading to errors during training. max_length=None avoids it

    # Logging / reporting
    output_dir=output_dir,              # Where to save model checkpoints and logs
    logging_steps=1,                    # Log training metrics every N steps
    report_to="trackio",                # Experiment tracking tool

    # Hub integration
    push_to_hub=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
)

dataset_sample = next(iter(train_dataset))
print(dataset_sample.keys())