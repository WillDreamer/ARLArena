# Game Agent Evaluation with Multimodal Support

## 测试结果

✅ 所有代码检查通过：
- ✅ 语法检查通过
- ✅ `multi_modal_data` 正确保留（不被弹出）
- ✅ 使用 `success_rate` 作为游戏代理的评估指标
- ✅ `generate_sequences` 正确处理多模态数据
- ✅ vLLM 多模态输入格式正确
- ✅ 文本模式回退逻辑存在

## 运行评估脚本

### 基本运行

```bash
cd /data1/dannie/projects/ARLArena
python recipe/game_agent/eval_game_agent.py
```

### 使用自定义配置

```bash
# 使用不同的配置文件
python recipe/game_agent/eval_game_agent.py --config-name=your_config

# 覆盖配置参数
python recipe/game_agent/eval_game_agent.py \
    actor_rollout_ref.model.path=/path/to/your/model \
    data.val_files=/path/to/val/data.parquet \
    env.seed=42
```

### 多模态设置

确保配置文件中包含：
- `data.image_key: images` (如果使用图像)
- `data.video_key: videos` (如果使用视频)
- `processor` 正确初始化（会自动从模型路径加载）

## 主要修改说明

### 1. `eval_game_agent.py`
- **保留 `multi_modal_data`**: 不再从批次中弹出，以便在 rollout 循环中使用
- **更新任务分数提取**: 使用 `success_rate` 替代 `webshop_task_score`
- **添加回退逻辑**: 如果 `success_rate` 不存在，尝试其他可用的成功指标

### 2. `llm_agent/agent_proxy.py`
- **多模态支持**: 检测 `multi_modal_data` 和 `raw_prompt_ids`，使用正确的 vLLM 格式
- **文本模式回退**: 如果没有多模态数据，使用标准的文本生成
- **灵活的数据处理**: 处理 `input_ids` 被弹出或不存在的情况

## 输出文件

评估完成后，会生成：
- `high_score_multiturn_texts_seed{seed}.json`: 包含高分（>0.9）任务的多轮对话记录

## 故障排除

### 如果遇到多模态相关错误：

1. **检查模型是否支持多模态**:
   ```python
   from transformers import AutoProcessor
   processor = AutoProcessor.from_pretrained(model_path)
   # 应该包含 image_processor
   ```

2. **检查数据格式**:
   - 确保数据文件包含 `images` 或 `videos` 字段
   - 检查 `data.image_key` 和 `data.video_key` 配置

3. **检查 vLLM 版本**:
   - 确保 vLLM 版本支持多模态（通常需要 >= 0.3.0）

### 如果遇到配置错误：

```bash
# 打印完整配置
python recipe/game_agent/eval_game_agent.py --cfg job
```

## 测试脚本

运行测试脚本验证代码：

```bash
python recipe/game_agent/test_multimodal_support.py
```

## 注意事项

1. **内存使用**: 多模态模型通常需要更多 GPU 内存
2. **批处理大小**: 如果遇到 OOM，减小 `batch_size` 或 `tensor_model_parallel_size`
3. **数据路径**: 确保 `train_files` 和 `val_files` 路径正确

