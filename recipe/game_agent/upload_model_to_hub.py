"""
上传合并后的模型到 HuggingFace Hub
"""
import os
from huggingface_hub import HfApi, login, create_repo
from huggingface_hub.utils import HfHubHTTPError
import argparse


def upload_model_to_hub(
    model_path: str,
    repo_id: str,
    private: bool = False,
    trust_remote_code: bool = True,
    commit_message: str = "Upload merged model",
):
    """
    上传模型到 HuggingFace Hub
    
    Args:
        model_path: 本地模型路径
        repo_id: HuggingFace Hub 的 repo ID (格式: username/model-name)
        private: 是否设为私有仓库
        trust_remote_code: 是否信任远程代码
        commit_message: 提交信息
    """
    print("=" * 80)
    print("上传模型到 HuggingFace Hub")
    print("=" * 80)
    print(f"模型路径: {model_path}")
    print(f"Repo ID: {repo_id}")
    print(f"私有仓库: {private}")
    print("=" * 80)
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    # 检查必要的文件
    required_files = ["config.json"]
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"警告: 缺少以下文件: {missing_files}")
        print("但会继续尝试上传...")
    
    # 登录 HuggingFace（如果需要）
    try:
        api = HfApi()
        # 尝试获取当前用户信息，如果失败则提示登录
        try:
            user = api.whoami()
            print(f"✅ 已登录 HuggingFace Hub")
            print(f"   用户名: {user.get('name', 'Unknown')}")
        except Exception:
            print("❌ 未登录 HuggingFace Hub")
            print("   请先运行: huggingface-cli login")
            print("   或者使用: from huggingface_hub import login; login()")
            response = input("是否现在登录? (y/n): ")
            if response.lower() == 'y':
                login()
            else:
                raise Exception("需要先登录 HuggingFace Hub")
    except Exception as e:
        print(f"❌ 登录失败: {e}")
        raise
    
    # 创建仓库（如果不存在）
    print(f"\n[1/3] 创建/检查仓库: {repo_id}...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        print(f"✅ 仓库已准备就绪: https://huggingface.co/{repo_id}")
    except HfHubHTTPError as e:
        if "already exists" in str(e).lower():
            print(f"✅ 仓库已存在: https://huggingface.co/{repo_id}")
        else:
            print(f"❌ 创建仓库失败: {e}")
            raise
    
    # 上传模型文件
    print(f"\n[2/3] 上传模型文件...")
    print("   这可能需要几分钟时间，取决于模型大小...")
    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            ignore_patterns=[".git", "__pycache__", "*.pyc", ".DS_Store"],
        )
        print("✅ 模型上传完成")
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        raise
    
    # 验证上传
    print(f"\n[3/3] 验证上传...")
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"✅ 验证成功")
        print(f"   仓库大小: {repo_info.siblings} 个文件")
    except Exception as e:
        print(f"⚠️  验证时出现警告: {e}")
    
    print("\n" + "=" * 80)
    print("✅ 上传完成！")
    print("=" * 80)
    print(f"\n模型已上传到: https://huggingface.co/{repo_id}")
    print("\n使用方法:")
    print(f"  from transformers import Qwen3VLForConditionalGeneration, AutoProcessor")
    print(f"  ")
    print(f"  model = Qwen3VLForConditionalGeneration.from_pretrained(")
    print(f"      '{repo_id}',")
    print(f"      torch_dtype=torch.bfloat16,")
    print(f"      device_map='auto',")
    print(f"      trust_remote_code=True")
    print(f"  )")
    print(f"  ")
    print(f"  processor = AutoProcessor.from_pretrained('{repo_id}', trust_remote_code=True)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="上传模型到 HuggingFace Hub")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/merged_qwen3_vl_4b_instruct",
        help="本地模型路径",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace Hub 的 repo ID (格式: username/model-name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="是否设为私有仓库",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload merged Qwen3-VL model",
        help="提交信息",
    )
    
    args = parser.parse_args()
    
    upload_model_to_hub(
        model_path=args.model_path,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
    )

