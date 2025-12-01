#!/bin/bash
# 上传合并后的模型到 HuggingFace Hub 的示例脚本

# ========== 配置参数 ==========
MODEL_PATH="./checkpoints/merged_qwen3_vl_4b_instruct"  # 你的合并后模型路径
REPO_ID="DannieSYD/Qwen3-VL-4B-Instruct-merged"  # 你的 HuggingFace Hub repo ID (格式: username/model-name)
PRIVATE=false  # 是否设为私有仓库 (true/false)
COMMIT_MESSAGE="Upload merged Qwen3-VL-4B-Instruct model after SFT training"

# ========== 检查登录 ==========
echo "检查 HuggingFace 登录状态..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "未登录 HuggingFace Hub，请先登录："
    echo "  huggingface-cli login"
    echo "或者运行："
    echo "  python3 -c 'from huggingface_hub import login; login()'"
    exit 1
fi

echo "✅ 已登录 HuggingFace Hub"

# ========== 上传模型 ==========
echo ""
echo "开始上传模型..."
python3 -m recipe.game_agent.upload_model_to_hub \
    --model_path "$MODEL_PATH" \
    --repo_id "$REPO_ID" \
    --commit_message "$COMMIT_MESSAGE" \
    $([ "$PRIVATE" = "true" ] && echo "--private")

echo ""
echo "✅ 上传完成！"
echo "模型地址: https://huggingface.co/$REPO_ID"

