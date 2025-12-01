# LoRA/QLoRA Adapter 合并指南

## 合并原理

### 1. LoRA/QLoRA 的工作原理

当你使用 QLoRA 进行微调时：
- **Base Model**: `Qwen/Qwen3-VL-4B-Instruct` - 原始模型权重（冻结，不更新）
- **LoRA Adapter**: 只训练少量参数（通常是原模型的 0.1-1%）
  - LoRA 通过低秩分解：`W_new = W_base + ΔW = W_base + BA`
  - 其中 B 和 A 是低秩矩阵（rank=r，如 r=32）
  - 训练时只更新 B 和 A，不更新 W_base

### 2. 保存的内容

使用 TRL 的 SFTTrainer 保存后，checkpoint 包含：
- **LoRA adapter 权重**（adapter_model.safetensors）
- **Adapter 配置**（adapter_config.json）
- **训练状态**（optimizer, scheduler 等，可选）

**不包含**完整的 base model 权重（因为 base model 没有被修改）

### 3. 合并的目的

合并 adapter 到 base model 可以：
- 创建一个**独立的完整模型**，不需要分别加载 base model 和 adapter
- 提高推理速度（不需要额外的 adapter 计算）
- 方便部署和分享

## 合并方法

### 方法1: 使用 PEFT 库（推荐，适用于 TRL 训练的模型）

```python
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

# 1. 加载 base model（不使用量化，因为要合并）
base_model_name = "Qwen/Qwen3-VL-4B-Instruct"
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 2. 加载 LoRA adapter
adapter_path = "DannieSYD/Qwen3-VL-4B-Instruct-trl-sft"  # 你的 checkpoint 路径
model = PeftModel.from_pretrained(base_model, adapter_path)

# 3. 合并 adapter 到 base model
merged_model = model.merge_and_unload()

# 4. 保存合并后的完整模型
output_dir = "./merged_qwen3_vl_4b_instruct"
merged_model.save_pretrained(output_dir)

# 5. 保存 processor 和 tokenizer
processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
processor.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)
```

### 方法2: 使用 verl.model_merger（适用于 verl FSDP 训练的模型）

**注意**: `verl.model_merger` 主要用于合并 **verl FSDP/Megatron 分布式训练的 checkpoint**，不是专门用于合并 PEFT adapter。

如果你使用的是 TRL 训练的模型，应该使用方法1。

如果你确实需要使用 verl.model_merger（例如你的 checkpoint 是 verl FSDP 格式），命令如下：

```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir DannieSYD/Qwen3-VL-4B-Instruct-trl-sft \
    --target_dir ./merged_model
```

## 重要说明

### 对于你的情况（TRL + QLoRA）

1. **Base Model**: `Qwen/Qwen3-VL-4B-Instruct`
2. **Adapter**: `DannieSYD/Qwen3-VL-4B-Instruct-trl-sft`（你保存的 checkpoint）

**应该使用方法1（PEFT merge_and_unload）**，因为：
- TRL 使用 PEFT 库管理 LoRA adapter
- 保存的格式是标准的 PEFT adapter 格式
- `verl.model_merger` 主要用于 verl 自己的训练框架

### 合并后的模型

合并后你会得到一个完整的模型，可以像普通模型一样使用：

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# 直接加载合并后的模型（不需要 adapter）
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "./merged_qwen3_vl_4b_instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    "./merged_qwen3_vl_4b_instruct",
    trust_remote_code=True
)
```

## 数学原理

LoRA 的合并公式：
```
W_merged = W_base + (B × A) × (alpha / r)
```

其中：
- `W_base`: 原始权重矩阵
- `B`: LoRA 的 B 矩阵（shape: [out_features, r]）
- `A`: LoRA 的 A 矩阵（shape: [r, in_features]）
- `alpha`: LoRA alpha 参数（你的配置中是 32）
- `r`: LoRA rank（你的配置中是 32）

合并时，`merge_and_unload()` 会：
1. 计算 `ΔW = B × A × (alpha / r)`
2. 将 `ΔW` 加到 `W_base` 上：`W_merged = W_base + ΔW`
3. 移除 adapter 层，得到标准的模型结构

