#!/bin/bash

set -e  # 遇到错误立即退出

echo "================================================"
echo "从 agentrl_game 迁移到 agentrl_qwen3"
echo "================================================"

# 检查源环境是否存在
if ! conda env list | grep -q "agentrl_game"; then
    echo "错误: agentrl_game 环境不存在！"
    exit 1
fi

# 1. 克隆环境
echo -e "\n[1/4] 克隆 agentrl_game 环境..."
conda create --name agentrl_qwen3 --clone agentrl_game -y
echo "✓ 环境克隆完成"

# 2. 激活新环境
echo -e "\n[2/4] 激活 agentrl_qwen3 环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate agentrl_qwen3

# 3. 升级关键包
echo -e "\n[3/4] 升级关键包..."

# 升级 PyTorch (从 2.6.0+cu124 到 2.8.0)
echo "  - 升级 PyTorch 到 2.8.0..."
pip install --upgrade torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu124

# 升级 vLLM (关键升级: 0.8.5 -> 0.11.0)
echo "  - 升级 vLLM 到 0.11.0..."
pip install --upgrade vllm==0.11.0

# 升级 flash-attention (2.7.4 -> 2.8.3)
echo "  - 升级 flash-attention..."
pip install --upgrade flash-attn==2.8.3 --no-build-isolation

# 升级 transformers (4.51.1 -> 4.57.1)
echo "  - 升级 transformers..."
pip install --upgrade transformers==4.57.1

# 升级 outlines (0.1.11 -> 1.2.8, 重大版本升级)
echo "  - 升级 outlines..."
pip install --upgrade outlines==1.2.8

# 升级其他依赖
echo "  - 升级其他依赖包..."
pip install --upgrade \
    xformers==0.0.32.post1 \
    triton==3.4.0 \
    ray==2.51.1 \
    tokenizers==0.22.1 \
    compressed-tensors==0.11.0 \
    lm-format-enforcer==0.11.3

echo "✓ 包升级完成"

# 4. 验证安装
echo -e "\n[4/4] 验证环境..."
python -c "
import torch
import vllm
import transformers
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ vLLM: {vllm.__version__}')
print(f'✓ Transformers: {transformers.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
"

echo -e "\n================================================"
echo "迁移完成！新环境: agentrl_qwen3"
echo "================================================"
echo -e "\n使用方法:"
echo "  conda activate agentrl_qwen3"
echo -e "\n运行测试:"
echo "  python test_qwen3.py"