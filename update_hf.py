# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /workspace/AgentRL/sft_ckpt/global_step_56 \
#     --target_dir /workspace/Qwen-4B-webshop


from huggingface_hub import create_repo, upload_folder

repo_name = "willamazon1/Qwen3-4B-rft-webshop"
repo_id = repo_name  
create_repo(repo_id, exist_ok=True)

upload_folder(
    repo_id=repo_id,
    folder_path="/workspace/Qwen3-4B-rft-webshop",  
    path_in_repo=".",                
)
