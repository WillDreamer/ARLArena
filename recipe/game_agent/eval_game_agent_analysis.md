# Code Analysis: eval_game_agent.py

## Code Overview

This script evaluates a game agent by:
1. Loading validation dataset
2. Processing batches through a multi-turn rollout loop
3. Collecting trajectories and success metrics
4. Saving high-scoring multi-turn dialogues to JSON

## Key Components

### Main Flow
1. **Setup** (lines 25-38): Initialize tokenizer, actor model, environments, and trajectory collector
2. **Data Loading** (lines 40-61): Create train/val datasets and dataloaders
3. **Evaluation Loop** (lines 63-143): 
   - Process each batch from validation dataloader
   - Repeat batch for multiple rollouts
   - Run multi-turn evaluation
   - Extract and save high-scoring trajectories

### Critical Operations
- **Line 67**: `test_batch.repeat()` - Repeats each sample `n` times (interleaved), increasing batch size from B to B×n
- **Line 99-103**: `multi_turn_loop_for_eval()` - Runs the multi-turn agent-environment interaction
- **Line 107-112**: Extracts success scores from result dictionary
- **Line 120-136**: Filters and saves high-scoring trajectories (score > 0.9)

## Logical Errors Found

### 🔴 **CRITICAL ERROR #1: KeyError on 'raw_prompt' (Line 360 in rollout_loop_eval.py)**

**Location**: The error occurs in `rollout_loop_eval.py` but affects this script's functionality.

**Problem**: 
- In `rollout_loop_eval.py` line 338-339, `raw_prompt` is added to keys to be popped
- Line 342 pops `raw_prompt` from `batch_input`
- Line 360 tries to access `batch_input.non_tensor_batch['raw_prompt']` which no longer exists

**Impact**: This will cause a `KeyError` when trying to save step I/O history, preventing the script from working.

**Fix**: Save `raw_prompt` from `batch.non_tensor_batch` BEFORE popping, or don't pop it if it's needed later.

```python
# Current (BROKEN):
if "raw_prompt" in batch.non_tensor_batch:
    non_tensor_batch_keys_to_pop.append("raw_prompt")
batch_input = batch.pop(...)
# Later tries to access batch_input.non_tensor_batch['raw_prompt'] - KEYERROR!

# Should be:
raw_prompt_backup = batch.non_tensor_batch.get('raw_prompt', None)
if "raw_prompt" in batch.non_tensor_batch:
    non_tensor_batch_keys_to_pop.append("raw_prompt")
batch_input = batch.pop(...)
# Then use raw_prompt_backup instead
```

### 🟡 **POTENTIAL ERROR #2: Batch Size Mismatch in Indexing (Lines 120-129)**

**Problem**: 
- Line 67 repeats the batch, changing size from B to B×n
- `task_scores` array has size B×n (one score per repeated sample)
- `sample_idx` ranges from 0 to B×n-1
- However, `turn['inputs']` and `turn['outputs']` come from the repeated batch, so they should also have size B×n
- **BUT**: If `raw_prompt` was popped before being saved (Error #1), this section won't even execute

**Analysis**: If Error #1 is fixed, this should work correctly because:
- `step_io_history` contains data from the repeated batch
- `sample_idx` correctly indexes into the repeated batch size
- The indexing `[sample_idx]` should work if arrays have matching sizes

**Potential Issue**: Need to verify that `turn['inputs']` and `turn['outputs']` are indeed lists/arrays of length B×n, not B.

### 🟡 **POTENTIAL ERROR #3: Unused Variable (Line 72)**

**Problem**: `input_texts` is created but never used.

**Impact**: Minor - just dead code, but suggests incomplete implementation or leftover debugging code.

### 🟡 **POTENTIAL ERROR #4: Missing Error Handling (Lines 107-112)**

**Problem**: 
- Line 107 assumes `result['success']` exists and is a dict
- If `multi_turn_loop_for_eval` returns a different structure or fails, this will raise KeyError
- No try-except around the result extraction

**Impact**: Script will crash if result structure is unexpected.

### 🟡 **POTENTIAL ERROR #5: File Overwriting (Line 117)**

**Problem**: 
- Output file path only includes seed: `f'high_score_multiturn_texts_seed{config.env.seed}.json'`
- If multiple batches are processed, each will overwrite the previous file
- Only the last batch's results will be saved

**Impact**: Data loss - earlier batches' high-scoring trajectories will be lost.

**Fix**: Include batch index or timestamp in filename, or append to file instead of overwriting.

### 🟡 **POTENTIAL ERROR #6: Unused train_dataset and train_sampler (Lines 40, 43)**

**Problem**: 
- `train_dataset` and `train_sampler` are created but never used
- There's commented-out code (lines 46-52) that would use `train_dataset` with `train_sampler`

**Impact**: Minor - just unused variables, but suggests incomplete refactoring.

## Recommendations

1. **Fix Critical Error #1** - This will prevent the script from running
2. **Fix Error #5** - Prevent data loss from file overwriting
3. **Add error handling** around result extraction
4. **Remove unused code** (input_texts, train_dataset, train_sampler)
5. **Add validation** to ensure batch sizes match after repeat operation
6. **Add logging** to track which batches are processed and saved

## Code Flow Diagram

```
main()
├── Setup (tokenizer, model, envs, collector)
├── Create datasets (train, val)
├── Create dataloader (val_dataset)
└── For each batch in val_dataloader:
    ├── Convert to DataProto
    ├── Repeat batch (B → B×n)
    ├── Pop unnecessary keys
    ├── Run multi_turn_loop_for_eval()
    │   └── [CRITICAL ERROR: tries to access popped 'raw_prompt']
    ├── Extract task_scores
    ├── Filter high scores (>0.9)
    └── Save to JSON (overwrites previous file)
```


