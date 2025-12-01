"""
合并 LoRA Adapter 到 Base Model
适用于 TRL + QLoRA 训练的模型
"""
import torch
from peft import PeftModel
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer
)
import os
from huggingface_hub import HfApi


def merge_lora_adapter(
    base_model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
    adapter_path: str = "DannieSYD/Qwen3-VL-4B-Instruct-trl-sft",
    output_dir: str = "./merged_qwen3_vl_4b_instruct",
    push_to_hub: bool = False,
    hub_model_id: str = None,
    trust_remote_code: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """
    合并 LoRA adapter 到 base model
    
    Args:
        base_model_name: Base model 的路径或 HuggingFace ID
        adapter_path: LoRA adapter 的路径或 HuggingFace ID
        output_dir: 合并后模型的保存路径
        push_to_hub: 是否推送到 HuggingFace Hub
        hub_model_id: HuggingFace Hub 的模型 ID（如果 push_to_hub=True）
        trust_remote_code: 是否信任远程代码
        torch_dtype: 模型的数据类型
    """
    print("=" * 80)
    print("LoRA Adapter 合并工具")
    print("=" * 80)
    print(f"Base Model: {base_model_name}")
    print(f"Adapter Path: {adapter_path}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)
    
    # 1. 加载 base model（不使用量化，因为要合并）
    print("\n[1/5] 加载 base model...")
    print("注意: 合并时需要加载完整的模型权重，不使用量化")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    print(f"✅ Base model 加载完成")
    print(f"   模型参数量: {sum(p.numel() for p in base_model.parameters()):,}")
    
    # 2. 加载 LoRA adapter
    print(f"\n[2/5] 加载 LoRA adapter from {adapter_path}...")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            trust_remote_code=trust_remote_code,
        )
        print("✅ Adapter 加载完成")
        
        # 显示 adapter 信息
        if hasattr(model, 'peft_config'):
            adapter_config = model.peft_config
            print(f"   Adapter 配置:")
            for key, config in adapter_config.items():
                if hasattr(config, 'r'):
                    print(f"     - Rank (r): {config.r}")
                if hasattr(config, 'lora_alpha'):
                    print(f"     - Alpha: {config.lora_alpha}")
                if hasattr(config, 'target_modules'):
                    print(f"     - Target Modules: {config.target_modules}")
    except Exception as e:
        print(f"❌ 加载 adapter 失败: {e}")
        print("   请检查 adapter_path 是否正确")
        raise
    
    # 3. 合并 adapter 到 base model
    print(f"\n[3/5] 合并 adapter 到 base model...")
    print("   这可能需要几分钟时间...")
    try:
        merged_model = model.merge_and_unload()
        print("✅ 合并完成")
        print(f"   合并后的模型参数量: {sum(p.numel() for p in merged_model.parameters()):,}")
    except Exception as e:
        print(f"❌ 合并失败: {e}")
        raise
    
    # 4. 保存合并后的完整模型
    print(f"\n[4/5] 保存合并后的模型到 {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    try:
        merged_model.save_pretrained(
            output_dir,
            safe_serialization=True,  # 使用 safetensors 格式
        )
        print("✅ 模型保存完成")
    except Exception as e:
        print(f"❌ 保存模型失败: {e}")
        raise
    
    # 5. 保存 processor 和 tokenizer
    print(f"\n[5/5] 保存 processor 和 tokenizer...")
    try:
        processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=trust_remote_code,
        )
        processor.save_pretrained(output_dir)
        print("✅ Processor 保存完成")
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=trust_remote_code,
        )
        tokenizer.save_pretrained(output_dir)
        print("✅ Tokenizer 保存完成")
    except Exception as e:
        print(f"❌ 保存 processor/tokenizer 失败: {e}")
        raise
    
    # 6. 可选：推送到 HuggingFace Hub
    if push_to_hub:
        if hub_model_id is None:
            hub_model_id = output_dir.split("/")[-1]
        
        print(f"\n[6/6] 推送模型到 HuggingFace Hub: {hub_model_id}...")
        try:
            merged_model.push_to_hub(
                hub_model_id,
                private=False,
                trust_remote_code=trust_remote_code,
            )
            processor.push_to_hub(hub_model_id)
            tokenizer.push_to_hub(hub_model_id)
            print(f"✅ 模型已推送到: https://huggingface.co/{hub_model_id}")
        except Exception as e:
            print(f"❌ 推送失败: {e}")
            print("   请确保已登录 HuggingFace: huggingface-cli login")
    
    print("\n" + "=" * 80)
    print("✅ 合并完成！")
    print("=" * 80)
    print(f"\n合并后的模型保存在: {output_dir}")
    print("\n使用方法:")
    print(f"  from transformers import Qwen3VLForConditionalGeneration, AutoProcessor")
    print(f"  ")
    print(f"  model = Qwen3VLForConditionalGeneration.from_pretrained(")
    print(f"      '{output_dir}',")
    print(f"      torch_dtype=torch.bfloat16,")
    print(f"      device_map='auto',")
    print(f"      trust_remote_code=True")
    print(f"  )")
    print(f"  ")
    print(f"  processor = AutoProcessor.from_pretrained('{output_dir}', trust_remote_code=True)")
    print("=" * 80)
    
    return merged_model, processor, tokenizer


if __name__ == "__main__":
    # 配置参数
    BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
    ADAPTER_PATH = "DannieSYD/Qwen3-VL-4B-Instruct-trl-sft"  # 你的 checkpoint 路径
    OUTPUT_DIR = "./merged_qwen3_vl_4b_instruct"
    
    # 可选：推送到 HuggingFace Hub
    PUSH_TO_HUB = False
    HUB_MODEL_ID = None  # 例如: "DannieSYD/Qwen3-VL-4B-Instruct-merged"
    
    # 执行合并
    merge_lora_adapter(
        base_model_name=BASE_MODEL,
        adapter_path=ADAPTER_PATH,
        output_dir=OUTPUT_DIR,
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=HUB_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

