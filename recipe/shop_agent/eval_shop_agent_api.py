# eval_shop_agent_api.py
# Version for API-based evaluation (DeepSeek/OpenAI)
# Replaces VllmWrapperWg with ApiCallingWrapperWg

#from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
import time
import pickle
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from torchdata.stateful_dataloader import StatefulDataLoader
#from recipe.shop_agent.llm_agent.agent_proxy import VllmWrapperWg
from recipe.shop_agent.llm_agent.agent_proxy import ApiCallingWrapperWg, ApiCallingWrapperWg_MAS, ApiCallingWrapperWg_Mixture, ApiCallingWrapperWg_Sequential, ApiCallingWrapperWg_Memory
from agent_system.multi_turn_rollout.rollout_loop_eval import TrajectoryCollector
from agent_system.environments import make_envs
from verl.utils import hf_processor
from recipe.shop_agent.main_shop_agent import create_rl_dataset, create_rl_sampler

from pathlib import Path

def safe_load_tokenizer(model_path: str):
    """安全加载 tokenizer，处理路径问题"""
    # 1. 如果是 HF ID 格式，直接用
    if '/' in model_path and len(model_path.split('/')) == 2:
        if not any(x in model_path for x in ['~', '\\', '..', ':']):
            try:
                return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
            except:
                pass
    
    # 2. 尝试展开路径
    try:
        expanded = os.path.expanduser(model_path)
        expanded = os.path.expandvars(expanded)
        path_obj = Path(expanded)
        
        if path_obj.exists():
            return AutoTokenizer.from_pretrained(str(path_obj), trust_remote_code=True, use_fast=True)
    except:
        pass
    
    # 3. 最后的 fallback
    print(f"⚠️  Loading fallback tokenizer: Qwen/Qwen2.5-7B-Instruct")
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True, use_fast=True)




@hydra.main(version_base=None, config_path="config", config_name="base_eval") # base_eval
def main(config):
    # detect config name from python -m webshop.llm_agent.agent_proxy --config_name frozen_lake
    # os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # Fix tokenizer parallelism warning in multiprocessing environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print(config.data)
    
    
    tokenizer = safe_load_tokenizer(config.actor_rollout_ref.model.path)
    
    # actor_wg = ApiCallingWrapperWg(config, tokenizer)
    actor_wg = ApiCallingWrapperWg_Memory(config, tokenizer)
    # actor_wg = ApiCallingWrapperWg_MAS(config, tokenizer)
    # actor_wg = ApiCallingWrapperWg_Mixture(config, tokenizer)
    # actor_wg = ApiCallingWrapperWg_Sequential(config, tokenizer)
    
    
    
    envs, val_envs = make_envs(config)
    processor = hf_processor(config.actor_rollout_ref.model.path, trust_remote_code=True, use_fast=True)

    traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    print("111")
    # train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
    print("222")
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
    print("333")
    
    # train_sampler = create_rl_sampler(config.data, train_dataset)
    from verl.utils.dataset.rl_dataset import collate_fn

    # val_dataloader = StatefulDataLoader(
    #         dataset=train_dataset,
    #         batch_size=256,
    #         num_workers=8,
    #         drop_last=True,
    #         collate_fn=collate_fn,
    #         sampler=train_sampler)
    print("444")


    
    val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=64,
            num_workers=0,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    
    #测试数据的load
    print(f"val_dataloader: {len(val_dataloader)}")

    # if True:
    #     data_iter = iter(val_dataloader)

    #     # Skip the first batch
    #     _ = next(data_iter) 

    #     # Skip the second batch
    #     _ = next(data_iter)

    #     # Skip the third batch
    #     _ = next(data_iter)

    #     # Capture the third batch
    #     test_data = next(data_iter)

    for test_data in val_dataloader:
        test_batch = DataProto.from_single_dict(test_data)

        # repeat test batch
        test_batch = test_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

        # Store original inputs
        input_ids = test_batch.batch["input_ids"]
        # TODO: Can we keep special tokens except for padding tokens?
        input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
        if "multi_modal_data" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "env_kwargs" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("env_kwargs")
        test_gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        test_gen_batch.meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
        }
        print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

    
        start_time = time.time()
        #result: important 存下来
        result = traj_collector.multi_turn_loop_for_eval(
                        gen_batch=test_gen_batch,
                        actor_rollout_wg=actor_wg,
                        envs=envs
                        )

        # Persist every evaluation result for later inspection
        snapshot_dir = Path("outputs_webshop") / "eval_results"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        snapshot_path = snapshot_dir / f"result_seed{config.env.seed}_{timestamp}.pkl"
        try:
            with snapshot_path.open("wb") as f:
                pickle.dump(result, f)
            print(f"Saved evaluation result to {snapshot_path}")
        except Exception as exc:
            print(f"Warning: failed to save evaluation result to {snapshot_path}: {exc}")

        task_scores = result['success'].get('webshop_task_score (not success_rate)', None)
        response_texts = result['step_io_history']

        if task_scores is not None and response_texts is not None:
            import json
            
            # Create a specific directory to hold these separate files
            traj_save_dir = Path(f"outputs_webshop/trajectories_seed{config.env.seed}_{timestamp}")
            traj_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving individual trajectories to: {traj_save_dir}")

            for sample_idx, score in enumerate(task_scores):
                
                # CORRECTION: 
                # response_texts[sample_idx] is already the clean, truncated list of dicts 
                # for this specific sample (from the post-processing in the previous function).
                # We can use it directly.
                sample_turns = response_texts[sample_idx]

                # Construct the data object for this single file
                single_traj_data = {
                    "sample_idx": sample_idx,
                    "task_score": float(score) if score is not None else 0.0,
                    "turns": sample_turns
                }

                # Generate a unique filename for this sample
                file_name = f"traj_sample_{sample_idx}.json"
                file_path = traj_save_dir / file_name
                
                # Write individual file
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(single_traj_data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Error saving sample {sample_idx}: {e}")

            print(f"Finished saving {len(task_scores)} separate trajectory files.")



        end_time = time.time()
        print(f'rollout time: {end_time - start_time} seconds')


        break

if __name__ == "__main__":
    main()
