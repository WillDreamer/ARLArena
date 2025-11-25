cd ~/ARLArena/agent_system/environments/env_package/webshop/webshop

# 注释掉 sudo 行
sed -i 's/^sudo apt/# sudo apt/' setup.sh

# 验证修改
grep sudo setup.sh

# 应该看到:
# # sudo apt update
# # sudo apt install -y default-jdk

# 然后运行
conda activate verl_agent
source setup.sh