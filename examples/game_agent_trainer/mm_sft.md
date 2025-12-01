# Multi-Modal Supervised Fine-Tuning (SFT) for Qwen3-VL

This guide documents the workflow for supervised fine-tuning of Qwen3-VL models in a multi-turn setting, with multimodal inputs (image + text) for game agent training.

## Overview

The SFT pipeline consists of the following steps:
1. **Trajectory Generation**: Sample multiple trajectories using multi-turn VLM inference
2. **Data Filtering**: Filter and save high-quality trajectories to JSON format
3. **Data Conversion**: Convert JSON trajectories to Parquet format for training
4. **Model Training**: Fine-tune the model using LoRA/QLoRA with Hugging Face TRL
5. **Model Upload**: Upload the fine-tuned model to Hugging Face Hub
6. **Checkpoint Merging**: Merge LoRA adapter with base model
7. **RL Training**: Use the merged checkpoint for reinforcement learning

---

## Prerequisites

- Qwen3-VL model (e.g., `Qwen/Qwen3-VL-4B-Instruct`)
- Required Python packages: `transformers`, `trl`, `peft`, `bitsandbytes`
- Access to GPU(s) for training
- Hugging Face account and token (for model upload)

---

## Step-by-Step Workflow

### Step 1: Multi-Turn Trajectory Generation

Generate multiple trajectories using multi-turn inference with the vision-language model.

**Scripts Modified:**
- `eval_game_agent.sh` - Main evaluation script
- `eval_game_agent.py` - Evaluation logic
- `agent_proxy.py` - Agent proxy with multi-modality support

**Command:**
```bash
bash examples/game_agent_trainer/eval_game_agent.sh
```

This script performs multi-turn inference with the VLM to sample multiple trajectories, which are used as training data for SFT.

---

### Step 2: Filter and Save High-Quality Trajectories

The generated trajectories are filtered to keep only high-quality samples, which are then saved to JSON files.

**Output Location:**
- Trajectories are saved to: `outputs/sft_trajectories_json/`

---

### Step 3: Convert JSON to Parquet Format

Convert the trajectory JSON files to Parquet format, following the multi-turn chat template format required for training.

**Script:** `recipe/game_agent/json_to_parquet.py`

**Usage:**
```bash
python recipe/game_agent/json_to_parquet.py \
    /data1/dannie/projects/ARLArena/outputs/sft_trajectories_json \
    /home/dannie/data/sft
```

**Arguments:**
- `input_dir`: Directory containing JSON trajectory files
- `output_dir`: Directory to save Parquet files (e.g., `/home/dannie/data/sft`)

The script processes the JSON files and converts them to Parquet format with proper multi-turn chat template formatting.

---

### Step 4: Train LoRA SFT

Fine-tune the model using LoRA (Low-Rank Adaptation) or QLoRA for parameter-efficient training.

**Training Framework:**
- ❌ **verl**: Does not support multi-modality SFT (attempted fixes failed due to model not supporting global [CLS] token)
- ✅ **Hugging Face TRL**: Successfully used for multi-modal SFT

**Notebook:** `recipe/game_agent/sft_qwen_vl.ipynb`

Follow the notebook to:
- Load the Parquet dataset
- Configure QLoRA parameters
- Set up the SFT trainer
- Train the LoRA adapter

**Key Configuration:**
- Model: `Qwen/Qwen3-VL-4B-Instruct` (or other Qwen3-VL variants)
- Quantization: 4-bit with BitsAndBytesConfig
- LoRA target modules: Attention and projection layers

---

### Step 5: Upload SFT Model to Hugging Face

After training, upload the fine-tuned LoRA adapter to Hugging Face Hub for sharing and version control.

**Note:** Ensure you're logged in to Hugging Face:
```python
from huggingface_hub import notebook_login
notebook_login()
```

---

### Step 6: Merge Checkpoint

Merge the LoRA adapter with the base model to create a standalone checkpoint.

**Script:** `recipe/game_agent/merge_lora_adapter.py`

**Notebook:** `recipe/game_agent/sft_qwen_vl.ipynb`

Follow the merging section in the notebook to combine the LoRA weights with the base model.

**Documentation:** See `recipe/game_agent/merge_lora_adapter.md` for detailed instructions.

---

### Step 7: Use Merged Checkpoint for RL Training

Use the merged checkpoint as the base model for reinforcement learning training.

**Script:** `examples/game_agent_trainer/train_gigpo_ablation.sh`

**Required Change:**
Update the `MERGED_MODEL` variable in `train_gigpo_ablation.sh` to point to your merged checkpoint path.

```bash
# Example:
MERGED_MODEL="/path/to/merged/checkpoint"
```

---

## File Structure

```
examples/game_agent_trainer/
├── mm_sft.md                    # This file
├── eval_game_agent.sh           # Multi-turn trajectory generation
├── train_gigpo_ablation.sh      # RL training script
└── ...

recipe/game_agent/
├── sft_qwen_vl.ipynb            # Main SFT training notebook
├── json_to_parquet.py           # JSON to Parquet converter
├── merge_lora_adapter.py        # LoRA merging script
└── merge_lora_adapter.md        # Merging documentation

outputs/
└── sft_trajectories_json/       # Generated trajectory JSON files
```

---

## Troubleshooting

### Issue: verl doesn't support multi-modality SFT
**Solution:** Use Hugging Face TRL instead, as documented in Step 4.

### Issue: Model doesn't support global [CLS] token
**Solution:** This was encountered when attempting to fix verl. Use Hugging Face TRL which properly handles Qwen3-VL architecture.

---

## Additional Resources

- [TRL Documentation](https://huggingface.co/docs/trl)
- [Qwen3-VL Fine-tuning Examples](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune/)
- [PEFT Documentation](https://huggingface.co/docs/peft)