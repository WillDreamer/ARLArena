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
from recipe.shop_agent.llm_agent.agent_proxy import ApiCallingWrapperWg
from agent_system.multi_turn_rollout.rollout_loop_eval import TrajectoryCollector
from agent_system.environments import make_envs
from verl.utils import hf_processor
from recipe.shop_agent.main_shop_agent import create_rl_dataset, create_rl_sampler

from pathlib import Path
from numbers import Number


def to_serializable(obj):
    """Convert common numpy/torch scalars and arrays to JSON friendly types."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    # Handle numpy/torch scalar with .item()
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    # Handle numpy arrays
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, (str, Number)) or obj is None:
        return obj
    return str(obj)


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




@hydra.main(version_base=None, config_path="config", config_name="base_eval")
def main(config):
    # detect config name from python -m webshop.llm_agent.agent_proxy --config_name frozen_lake
    # os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # Fix tokenizer parallelism warning in multiprocessing environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print(config.data)
    
    
    tokenizer = safe_load_tokenizer(config.actor_rollout_ref.model.path)
    
    actor_wg = ApiCallingWrapperWg(config, tokenizer)
    
    
    envs, val_envs = make_envs(config)
    processor = hf_processor(config.actor_rollout_ref.model.path, trust_remote_code=True, use_fast=True)

    traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    
    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
    
    train_sampler = create_rl_sampler(config.data, train_dataset)
    from verl.utils.dataset.rl_dataset import collate_fn

    # val_dataloader = StatefulDataLoader(
    #         dataset=train_dataset,
    #         batch_size=256,
    #         num_workers=8,
    #         drop_last=True,
    #         collate_fn=collate_fn,
    #         sampler=train_sampler)

    # Use batch_size from config, fallback to 64 if not specified
    val_bs = config.data.val_batch_size if config.data.val_batch_size is not None else 64
    val_workers = min(8, os.cpu_count() or 2)
    print(f"[DEBUG] val_batch_size from config: {config.data.val_batch_size}, using: {val_bs}")

    val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_bs,
            num_workers=val_workers,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    
    #测试数据的load
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

        # Extract metrics from result
        success_rate = result.get("success_rate", {})
        success = result.get("success", {})
        print("\nSuccess rate:")
        for k, v in success_rate.items():
            print(f"  {k}: {v}")

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

        # Save human-readable metrics to JSON alongside the pickle
        metrics_json = snapshot_dir / f"result_seed{config.env.seed}_{timestamp}.json"
        try:
            import json
            metrics_payload = {
                "seed": config.env.seed,
                "timestamp": timestamp,
                "model": config.actor_rollout_ref.model.path,
                "data": {
                    "train": config.data.train_files,
                    "val": config.data.val_files,
                },
                "success_rate": to_serializable(success_rate),
                "success": to_serializable(success),
                "step_io_history_len": len(result.get("step_io_history", [])),
                "pkl_path": str(snapshot_path),
            }
            with metrics_json.open("w", encoding="utf-8") as jf:
                json.dump(metrics_payload, jf, ensure_ascii=False, indent=2)
            print(f"Saved metrics summary to {metrics_json}")
        except Exception as exc:
            print(f"Warning: failed to save metrics JSON to {metrics_json}: {exc}")

        # 提取success['webshop_task_score']大于0.8对应的response_texts，并保存
        task_scores = result['success'].get('webshop_task_score (not success_rate)', None)
        response_texts = result['step_io_history']

        if task_scores is not None and response_texts is not None:
            import json
            output_path = f'high_score_multiturn_texts_seed{config.env.seed}.json'
            # response_texts是长度为15的列表，每个元素包含dict_keys(['step', 'inputs', 'outputs'])
            all_turns = []
            for sample_idx, score in enumerate(task_scores):
                if score is not None and score >= 0.0: #原本是0.9
                    turns = []
                    # response_texts为每轮list，每轮包含该batch（n）的信息
                    for turn in response_texts:
                        # turn['inputs']和turn['outputs']都是n个sample
                        turn_result = {
                            "step": turn.get('step', None),
                            "inputs": list(turn.get('inputs', []))[sample_idx] if turn.get('inputs', None) is not None else None,
                            "outputs": list(turn.get('outputs', []))[sample_idx] if turn.get('outputs', None) is not None else None
                        }

                        # 添加 env_feedback（如果存在的话）
                        if 'env_feedback' in turn:
                            env_feedback = turn['env_feedback']
                            try:
                                # next_obs is a dict with keys ['text', 'image', 'anchor']
                                # Each value is a list of observations for all samples
                                next_obs_dict = env_feedback.get('next_obs', {})
                                next_obs_data = None
                                if next_obs_dict:
                                    next_obs_data = {
                                        'text': next_obs_dict.get('text', [None]*64)[sample_idx] if next_obs_dict.get('text') else None,
                                        'image': next_obs_dict.get('image'),  # Usually None
                                        'anchor': next_obs_dict.get('anchor', [None]*64)[sample_idx] if next_obs_dict.get('anchor') else None,
                                    }

                                turn_result['env_feedback'] = {
                                    "next_obs": next_obs_data,
                                    "reward": list(env_feedback.get('rewards', []))[sample_idx] if env_feedback.get('rewards') is not None else None,
                                    "done": list(env_feedback.get('dones', []))[sample_idx] if env_feedback.get('dones') is not None else None,
                                    "info": list(env_feedback.get('infos', []))[sample_idx] if env_feedback.get('infos') is not None else None
                                }
                            except (IndexError, KeyError, TypeError) as e:
                                # If extraction fails, skip env_feedback for this turn
                                print(f"Warning: Failed to extract env_feedback for sample_idx={sample_idx}: {e}")
                                pass

                        # Check if environment is done based on CURRENT turn's input
                        # The ONLY reliable indicator: admissible actions is empty
                        is_done = False

                        # Check if current turn has empty admissible actions (env is done)
                        inp = turn_result.get('inputs', [])
                        if isinstance(inp, list) and len(inp) > 0 and isinstance(inp[0], dict):
                            inp_text = inp[0].get('content', '')
                        else:
                            inp_text = str(inp) if inp else ''

                        if inp_text and 'admissible actions' in inp_text.lower():
                            # Extract the actions section
                            start = inp_text.find('[', inp_text.lower().find('admissible actions'))
                            if start != -1:
                                end = inp_text.find(']', start) + 1
                                actions_part = inp_text[start:end].strip()
                                # Check if it's essentially empty (only whitespace/newlines between brackets)
                                content = actions_part[1:-1].strip()  # Remove [ and ]
                                if not content or content.replace('\n', '').replace(' ', '') == '':
                                    # Environment is done, don't include this turn
                                    is_done = True

                        # If done, stop WITHOUT including this turn (it's noise after task completion)
                        turns.append(turn_result)
                        
                        if is_done:
                            break

                        # Otherwise, include this turn
                        # turns.append(turn_result)

                    all_turns.append({
                        "sample_idx": sample_idx,
                        "task_score": score,
                        "turns": turns,
                        "actual_turns": len(turns)  # Record actual number of turns before done
                    })

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(to_serializable(all_turns), f, ensure_ascii=False, indent=2)
            print(f"Saved all {len(all_turns)} turns of the multi-turn dialogue to {output_path}")
        
        end_time = time.time()
        print(f'rollout time: {end_time - start_time} seconds')

if __name__ == "__main__":
    main()
