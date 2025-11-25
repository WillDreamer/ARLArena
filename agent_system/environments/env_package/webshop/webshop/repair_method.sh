# === 彻底修复所有问题 ===

conda activate verl_agent
cd ~/ARLArena

echo "🔧 开始彻底修复..."

# 1. 重新安装 faiss (带 MKL)
echo "1️⃣ 重新安装 faiss-cpu (完整版)..."
conda remove faiss-cpu -y
conda install -c conda-forge faiss-cpu mkl mkl-service -y

# 2. 验证 faiss
echo "验证 faiss:"
python -c "import faiss; print('✅ faiss 正常')" || echo "❌ faiss 仍有问题"

# 3. 先安装 torch,再装其他包
echo "2️⃣ 安装 torch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. 跳过 flash-attn,安装其他依赖
echo "3️⃣ 安装其他依赖(跳过 flash-attn)..."
cd ~/ARLArena
sed -i 's/^flash-attn/# flash-attn/' requirements.txt
pip install -r requirements.txt

# 5. 重新运行数据转换
echo "4️⃣ 重新转换产品数据..."
cd agent_system/environments/env_package/webshop/webshop/search_engine
python convert_product_file_format.py

# 6. 检查转换结果
echo "检查转换结果:"
ls -lh resources/
ls -lh resources_1k/

# 7. 重新构建索引
echo "5️⃣ 重新构建搜索引擎索引..."
bash run_indexing.sh

echo ""
echo "✅ 修复完成!"