from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
from typing import List, Dict, Tuple, Union
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from recipe.shop_agent.llm_agent.base_llm import ConcurrentLLM
import numpy as np
import re
from collections import Counter


class VllmWrapperWg: # Thi is a developing class for eval and test
	def __init__(self, config, tokenizer):
		self.config = config
		self.tokenizer = tokenizer
		model_name = config.actor_rollout_ref.model.path
		ro_config = config.actor_rollout_ref.rollout
		self.llm = LLM(
			model_name,
            enable_sleep_mode=True,
            tensor_parallel_size=ro_config.tensor_model_parallel_size,
            dtype=ro_config.dtype,
            enforce_eager=ro_config.enforce_eager,
            gpu_memory_utilization=ro_config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            # disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=ro_config.max_model_len,
            disable_log_stats=ro_config.disable_log_stats,
            max_num_batched_tokens=ro_config.max_num_batched_tokens,
            enable_chunked_prefill=ro_config.enable_chunked_prefill,
            enable_prefix_caching=True,
			trust_remote_code=True,
		)
		print("LLM initialized")
		self.sampling_params = SamplingParams(
			max_tokens=ro_config.response_length,
			temperature=ro_config.val_kwargs.temperature,
			top_p=ro_config.val_kwargs.top_p,
			top_k=ro_config.val_kwargs.top_k,
			# min_p=0.1,
		)

	def generate_sequences(self, lm_inputs: DataProto):
		"""
		Convert the input ids to text, and then generate the sequences. Finally create a dataproto. 
		This aligns with the verl Worker Group interface.
		"""
		# NOTE: free_cache_engine is not used in the vllm wrapper. Only used in the verl vllm.
		# cache_action = lm_inputs.meta_info.get('cache_action', None)

		input_ids = lm_inputs.batch['input_ids']
		input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
		input_texts = [i.replace("<|endoftext|>", "") for i in input_texts]

		outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
		texts = [output.outputs[0].text for output in outputs] 
		lm_outputs = DataProto()
		lm_outputs.non_tensor_batch = {
			'response_texts': texts,
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
		lm_outputs.meta_info = lm_inputs.meta_info

		return lm_outputs

	
class ApiCallingWrapperWg:
    """Wrapper class for API-based LLM calls that fits into the VERL framework"""
    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        # model_info adjust
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs

        # concurrent LLM需要改
        self.llm = ConcurrentLLM(
			provider="gemini", # model_info.provider_name,
            model_name="gemini-2.5-pro", # model_info.model_name,
            max_concurrency=config.model_config.max_concurrency
        )
        
        print(f'API-based LLM ({model_info.provider_name} - {model_info.model_name}) initialized')

        import time
        time.sleep(10)
        # exit()


    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        """
        Convert the input ids to text, make API calls to generate responses, 
        and create a DataProto with the results.
        """

        messages_list = lm_inputs.non_tensor_batch.get('messages_list', None)
        
        if messages_list is not None:
            messages_list = messages_list.tolist()
        else:
            # 回退：直接用 input_ids 构建单轮聊天消息
            input_ids = lm_inputs.batch.get('input_ids', None)
            if input_ids is None:
                raise KeyError("messages_list missing and input_ids unavailable to build fallback prompts")
            if hasattr(input_ids, "detach"):
                input_ids = input_ids.detach().cpu()
            fallback_messages = []
            for ids in input_ids:
                text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                fallback_messages.append([{"role": "user", "content": text}])
            messages_list = fallback_messages
            print("[WARN] messages_list not provided; constructed fallback prompts from input_ids")

        results, failed_messages = self.llm.run_batch(
            messages_list=messages_list,
            **self.llm_kwargs
        )
        assert not failed_messages, f"Failed to generate responses for the following messages: {failed_messages}"

        texts = [result["response"] for result in results]
        env_ids = lm_inputs.non_tensor_batch.get('env_ids')
        if env_ids is None:
            env_ids = np.arange(len(texts), dtype=object)
        group_ids = lm_inputs.non_tensor_batch.get('group_ids')
        if group_ids is None:
            group_ids = np.zeros(len(texts), dtype=object)
        # print(f'[DEBUG] texts: {texts}')
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': env_ids,
			'group_ids': group_ids
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
        lm_outputs.meta_info = lm_inputs.meta_info
        
        return lm_outputs
    

class ApiCallingWrapperWg_Memory:
    """Wrapper class for API-based LLM calls that fits into the VERL framework"""
    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        # model_info adjust
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs

        # concurrent LLM需要改
        self.llm = ConcurrentLLM(
			provider="gemini", # model_info.provider_name,
            model_name="gemini-2.5-pro", # model_info.model_name,
            max_concurrency=config.model_config.max_concurrency
        )
        
        print(f'API-based LLM ({model_info.provider_name} - {model_info.model_name}) initialized')

        import time
        time.sleep(10)
        # exit()

    def _get_messages_list(self, lm_inputs):
        messages_list = lm_inputs.non_tensor_batch.get('messages_list', None)
        
        if messages_list is not None:
            messages_list = messages_list.tolist()
        else:
            input_ids = lm_inputs.batch.get('input_ids', None)
            if input_ids is None:
                raise KeyError("messages_list missing and input_ids unavailable")
            if hasattr(input_ids, "detach"):
                input_ids = input_ids.detach().cpu()
            
            fallback_messages = []
            for ids in input_ids:
                text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                fallback_messages.append([{"role": "user", "content": text}])
            messages_list = fallback_messages
            print("[WARN] Constructed fallback prompts from input_ids")

        return messages_list

    def _pack_output(self, inputs, final_texts):
        lm_outputs = DataProto()
        batch_len = len(final_texts)
        env_ids = inputs.non_tensor_batch.get('env_ids', np.arange(batch_len, dtype=object))
        group_ids = inputs.non_tensor_batch.get('group_ids', np.zeros(batch_len, dtype=object))
        
        lm_outputs.non_tensor_batch = {
            'response_texts': final_texts,
            'env_ids': env_ids,
            'group_ids': group_ids
        }
        lm_outputs.meta_info = inputs.meta_info
        return lm_outputs
    
    
    def _check_has_history(self, messages_list):
        has_history_mask = []
        history_marker = "Below are the most recent"

        for history in messages_list:
            last_message_content = history[-1]['content']
            
            if history_marker in last_message_content:
                has_history_mask.append(True)
            else:
                has_history_mask.append(False)
                
        return has_history_mask

    def _extract_history_blocks(self, messages_list, has_history_mask):
        history_blocks = []
        pattern = r"(Below are the most recent .*? observations and the corresponding actions you took: .*?)(?=\nYou are now at step)"

        for i, history in enumerate(messages_list):
            if has_history_mask[i]:
                last_msg_content = history[-1]['content']
                
                match = re.search(pattern, last_msg_content, re.DOTALL)
                
                if match:
                    history_blocks.append(match.group(1))
                else:
                    print(f"[WARN] History marker found but extraction failed for item {i}")
                    history_blocks.append(None)
            else:
                history_blocks.append(None)
                
        return history_blocks
    
    def _extract_background_info(self, messages_list):
        background_infos = []
    
        pattern = (
            r"(You are an expert autonomous agent operating in the WebShop e‑commerce environment\.\n"
            r"Your task is to: .*?)"
            r"(?=\nPrior to this step|\nYour current observation is)"
        )

        for history in messages_list:
            last_msg_content = history[-1]['content']
            
            match = re.search(pattern, last_msg_content, re.DOTALL)
            
            if match:
                background_infos.append(match.group(1))
            else:
                print("[WARN] Background info not found.")
                background_infos.append(None)
                
        return background_infos
    
    def _prepare_summary_prompts(self, background_infos, history_blocks, has_history_mask):
        summary_messages_list = []
        active_indices = []

        for i, has_history in enumerate(has_history_mask):
            if has_history:
                bg_info = background_infos[i]
                hist_block = history_blocks[i]
                
                prompt_text = (
                    f"=== TASK CONTEXT (READ ONLY) ===\n"
                    f"{bg_info}\n\n"
                    f"The following is the raw log of your recent interactions:\n"
                    f"---------------------\n"
                    f"{hist_block}\n"
                    f"---------------------\n"
                    f"Please summarize the past interaction history above concisely using words, "
                    f"highlighting the key information that will be helpful for the next step decision making."
                )
                
                summary_messages_list.append([{"role": "user", "content": prompt_text}])
                active_indices.append(i)

        return summary_messages_list, active_indices
    
    def _run_summarization(self, summary_messages_list):
        if not summary_messages_list:
            return []
        
        results, failed_messages = self.llm.run_batch(
            messages_list=summary_messages_list, 
            **self.llm_kwargs
        )
    
        assert not failed_messages, "Summarization agent failed on some messages"

        summaries = [res["response"] for res in results]
        
        return summaries
    
    def _inject_summaries(self, messages_list, history_blocks, summaries, active_indices):
        final_message_lists = []
        summary_cursor = 0

        summary_label = "Summarization of Previous Observation and Past Actions: "

        for i, history in enumerate(messages_list):
            new_history = [dict(msg) for msg in history]
            
            last_message = new_history[-1]
            original_content = last_message['content']
            
            if i in active_indices:
                target_text = history_blocks[i]

                generated_summary = summaries[summary_cursor]
                formatted_summary = f"{summary_label}{generated_summary}"

                new_content = original_content.replace(target_text, formatted_summary)
                
                last_message['content'] = new_content
                summary_cursor += 1
            
            final_message_lists.append(new_history)
                
        return final_message_lists
    

    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        messages_list = self._get_messages_list(lm_inputs)
        batch_size = len(messages_list)

        target_phrase = "You are Qwen, created by Alibaba Cloud."
        for history in messages_list:
            for msg in history:
                if target_phrase in msg.get("content", ""):
                    msg["content"] = msg["content"].replace(target_phrase, "")

        has_history_mask = self._check_has_history(messages_list)
        # print(f"has_history_mask: {has_history_mask}")

        if all(has_history_mask):
            history_blocks = self._extract_history_blocks(messages_list, has_history_mask)
            # print(f"history_blocks: {history_blocks[0]}")
            background_infos = self._extract_background_info(messages_list)
            # print(f"background_infos: {background_infos[0]}")
            summary_messages_list, active_indices = self._prepare_summary_prompts(background_infos, history_blocks, has_history_mask)
            # print(f"summary_messages_list: {summary_messages_list[0]}")
            # print(f"active_indices: {active_indices}")
            summaries = self._run_summarization(summary_messages_list)
            # print(f"summaries: {summaries[0]}")
            # _ = input("Next: ")

            final_prompts = self._inject_summaries(messages_list, history_blocks, summaries, active_indices)
            # print(f"final_prompts: {final_prompts[0]}")
            # _ = input("Next: ")

        else:
            final_prompts = messages_list
        
        results, failed_messages = self.llm.run_batch(
            messages_list=final_prompts,
            **self.llm_kwargs
        )
        assert not failed_messages, f"Failed to generate responses for the following messages: {failed_messages}"

        final_texts = [res["response"] for res in results]
        # print(f"final_texts: {final_texts[0]}")
        # _ = input("Next: ")
        
        return self._pack_output(lm_inputs, final_texts)
    


class ApiCallingWrapperWg_MAS:
    """Multi-Agent Wrapper."""
    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs

        self.max_iterations = 5
        self.num_agents = 3
        self.agents = []

        agent_configs = [
            {"provider": "gemini", "model": "gemini-2.5-pro"},
            {"provider": "openai", "model": "o3"},
            {"provider": "openai", "model": "gpt-5.2"},
        ]

        for i, config_item in enumerate(agent_configs):
            agent = ConcurrentLLM(
                provider=config_item["provider"],
                model_name=config_item["model"],
                max_concurrency=config.model_config.max_concurrency
            )
            self.agents.append(agent)
            print(f'Multi-Agent Setup: Agent {i} ({config_item["provider"]}/{config_item["model"]}) initialized')

        import time
        time.sleep(10)


    def _get_messages_list(self, lm_inputs):
        messages_list = lm_inputs.non_tensor_batch.get('messages_list', None)
        
        if messages_list is not None:
            messages_list = messages_list.tolist()
        else:
            input_ids = lm_inputs.batch.get('input_ids', None)
            if input_ids is None:
                raise KeyError("messages_list missing and input_ids unavailable")
            if hasattr(input_ids, "detach"):
                input_ids = input_ids.detach().cpu()
            
            fallback_messages = []
            for ids in input_ids:
                text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                fallback_messages.append([{"role": "user", "content": text}])
            messages_list = fallback_messages
            print("[WARN] Constructed fallback prompts from input_ids")

        return messages_list
    
    def _run_all_agents(self, inputs_map):
        agent_outputs = {}
        
        for i, agent in enumerate(self.agents):
            results, failed_messages = agent.run_batch(messages_list=inputs_map[i], **self.llm_kwargs)
            
            assert not failed_messages, f"Agent {i} failed on some messages"

            agent_outputs[i] = [result["response"] for result in results]
        
        return agent_outputs
    
    def _prepare_debate_prompts(self, base_messages, current_candidates, previous_outputs=None):
        inputs_per_agent = {i: [] for i in range(self.num_agents)}
        batch_size = len(base_messages)

        for b in range(batch_size):
            original_hist = base_messages[b]
            candidates = current_candidates[b]
            
            candidates_str = ""
            for idx, text in enumerate(candidates):
                snippet = text.replace("<think>", "").replace("</think>", "").strip()
                candidates_str += f"=== CANDIDATE SOLUTION {idx} ===\n{snippet}\n\n"

            
            transcript_str = ""
            if previous_outputs:
                transcript_str = ""
                for i in range(self.num_agents):
                    prev_text = previous_outputs[i][b].strip()
                    transcript_str += f"=== PREVIOUS ROUND OUTPUTS Agent {i} ===\n{prev_text}\n\n"

            for i in range(self.num_agents):
                instruction = (
                    # "===============\n"
                    f"Review the candidate solutions above for the user's request.\n"
                    f"You are Agent {i}. Your previous answer is represented by CANDIDATE SOLUTION {i}.\n\n"
                    "Your Goal: Reach consensus on the BEST correct action.\n"
                    "1. If you believe another candidate is BETTER:\n"
                    "   Output <vote>INDEX</vote> (e.g., <vote>1</vote>).\n"
                    "2. If you believe your solution is BETTER:\n"
                    "   Justify why in <think> tags, then vote your solution in <vote> tags.\n"
                    "3. If you can combine ideas to make a better solution:\n"
                    "   Output the new solution in <action> tags.\n\n"
                    "FORMAT:\n"
                    "<think> reasoning... </think>\n"
                    "<vote>ID</vote>  OR  <action>...full solution...</action>"
                )
                
                new_hist = [dict(msg) for msg in original_hist]
                
                last_msg = new_hist[-1]
                base_content = last_msg['content']

                if "assistant" in base_content[-20:]:
                    base_content = base_content.rsplit("assistant", 1)[0].strip()
                combined_content = f"=== TASK CONTEXT (READ ONLY) ===\n{base_content}\n\n{candidates_str}{transcript_str}{instruction}\n\nassistant"


                last_msg['content'] = combined_content

                # if b == 0:
                #     print(f"new_hist: {new_hist}")
                #     _ = input("Next: ")

                
                inputs_per_agent[i].append(new_hist)

        return inputs_per_agent
    
    def _parse_decision(self, response: str) -> Tuple[str, Union[int, str]]:
        """
        Parses the LLM output.
        Priority: 
        1. <vote>ID</vote>
        2. <refine>Text</refine>
        3. Fallback: Treat entire text as refinement (if it contains <action>)
        """
        # 1. Check Vote
        vote_match = re.search(r"<vote>\s*(\d+)\s*</vote>", response, re.IGNORECASE)
        if vote_match:
            try:
                return ("vote", int(vote_match.group(1)))
            except: pass
            
        return ("refine", response.strip())
    
    def _pack_output(self, inputs, final_texts):
        lm_outputs = DataProto()
        batch_len = len(final_texts)
        env_ids = inputs.non_tensor_batch.get('env_ids', np.arange(batch_len, dtype=object))
        group_ids = inputs.non_tensor_batch.get('group_ids', np.zeros(batch_len, dtype=object))
        
        lm_outputs.non_tensor_batch = {
            'response_texts': final_texts,
            'env_ids': env_ids,
            'group_ids': group_ids
        }
        lm_outputs.meta_info = inputs.meta_info
        return lm_outputs

    def _print_debug(self, iteration, is_solved, raw_response):
        print("-" * 50)
        print(f"DEBUG: End of Round {iteration} (Item 0)")
        print(f"Solved: {is_solved}")
        print(f"Agent 0 Output:\n{raw_response[0]}")
        print(f"Agent 1 Output:\n{raw_response[1]}")
        print(f"Agent 2 Output:\n{raw_response[2]}")
        print("-" * 50)

    def _check_textual_consensus(self, candidates):
        """Checks if any two candidates have identical normalized actions."""
        normalized = []
        
        for text in candidates:
            try:
                match = re.search(r"<action>(.*?)</action>", text, re.DOTALL | re.IGNORECASE)
            except:
                print(candidates)
            if match:
                clean = re.sub(r'[\s\W_]+', '', match.group(1).lower())
                normalized.append(clean)
            else:
                normalized.append(None)


        if normalized[0] and normalized[0] == normalized[1]:
            return True, candidates[0]
        if normalized[0] and normalized[0] == normalized[2]:
            return True, candidates[0]
        if normalized[1] and normalized[1] == normalized[2]:
            return True, candidates[1]
            
        return False, None

    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        messages_list = self._get_messages_list(lm_inputs)
        batch_size = len(messages_list)


        
        # current_candidates[batch_idx] = [text_agent_0, text_agent_1, text_agent_2]
        current_candidates = [["" for _ in range(self.num_agents)] for _ in range(batch_size)]
        final_solutions = [None] * batch_size
        is_solved = [False] * batch_size

        # print(f"messages_list: {messages_list}")
        target_phrase = "You are Qwen, created by Alibaba Cloud."
        for history in messages_list:
            for msg in history:
                if target_phrase in msg.get("content", ""):
                    # Remove the phrase regardless of the message role
                    msg["content"] = msg["content"].replace(target_phrase, "")

        # print(f"messages_list: {messages_list}")
        # exit()


        interaction_history = [
            {
                "batch_index": b,
                "round_0_initial": {},
                "debate_rounds": []
            }
            for b in range(batch_size)
        ]

        # print("\n=== Round 0: Initial Generation ===")
        init_prompts = {i: messages_list for i in range(self.num_agents)}
        
        results = self._run_all_agents(init_prompts)

        for b in range(batch_size):
            interaction_history[b]["round_0_initial"] = {
                "input": messages_list[b],
                "outputs": {
                    f"agent_{i}": results[i][b] for i in range(self.num_agents)
                }
            }

            current_candidates[b] = [results[i][b] for i in range(self.num_agents)]

            has_consensus, consensus_text = self._check_textual_consensus(current_candidates[b])
            if has_consensus:
                is_solved[b] = True
                final_solutions[b] = consensus_text
                # Sync candidates so future loops (if entered) see the consensus
                current_candidates[b] = [consensus_text] * self.num_agents

        # self._print_debug("First", is_solved[0], [results[i][0] for i in range(self.num_agents)])

        # print(f"current_candidates[0]: {current_candidates[0]}")
        # _ = input("Next: ")

        last_round_responses = results

        for iteration in range(1, self.max_iterations + 1):
            # print(f"\n=== Round {iteration}: Vote or Refine ===")

            prompts_map = self._prepare_debate_prompts(messages_list, current_candidates, last_round_responses)

            raw_responses = self._run_all_agents(prompts_map)

            last_round_responses = raw_responses

            next_candidates = []

            # print(f"{iteration}")
            # print(f"final_solutions: {final_solutions}")
            # print(f"is_solved: {is_solved}")
            
            for b in range(batch_size):
                if is_solved[b]:
                    next_candidates.append([final_solutions[b]] * self.num_agents) # Just carry forward the winner to keep lists aligned (though irrelevant)
                    continue

                candidates_b = current_candidates[b]
                responses_b = [raw_responses[i][b] for i in range(self.num_agents)]

                round_history = {
                    "round": iteration,
                    "inputs_per_agent": {
                        f"agent_{i}": prompts_map[i][b] for i in range(self.num_agents)
                    },
                    "outputs_per_agent": {
                         f"agent_{i}": responses_b[i] for i in range(self.num_agents)
                    }
                }
                interaction_history[b]["debate_rounds"].append(round_history)
                
                decisions = [self._parse_decision(r) for r in responses_b]
                
                # if b == 0:
                #     self._print_debug(iteration, is_solved[0], [raw_responses[i][0] for i in range(self.num_agents)])
                #     _ = input("Next: ")
                
                # --- Consensus Check ---
                votes_for_index = Counter()
                for d_type, d_val in decisions:
                    if d_type == "vote":
                        votes_for_index[d_val] += 1

                
                # print(f"votes_for_index: {votes_for_index}")
                
                # Determine if we have a winner
                winner_idx = None
                if votes_for_index:
                    best_idx, count = votes_for_index.most_common(1)[0]
                    if count >= 2:
                        winner_idx = best_idx
                        # print(f"winner_idx: {winner_idx}")

                if winner_idx is not None:
                    # CONVERGENCE REACHED
                    is_solved[b] = True
                    final_solutions[b] = candidates_b[winner_idx]
                    next_candidates.append([final_solutions[b]] * self.num_agents)
                else:
                    # NO CONSENSUS -> Update candidates for next round
                    new_row = []
                    for i in range(self.num_agents):
                        d_type, d_val = decisions[i]
                        
                        if d_type == "vote":
                            # Agent adopted another solution
                            # target_idx = d_val if 0 <= d_val < self.num_agents else i
                            # new_row.append(candidates_b[target_idx])

                            new_row.append(candidates_b[i])
                        else:
                            # Agent refined or stood ground
                            new_row.append(d_val)
                    next_candidates.append(new_row)

            current_candidates = next_candidates

            for b in range(batch_size):
                if not is_solved[b]:
                    has_consensus, consensus_text = self._check_textual_consensus(current_candidates[b])
                    if has_consensus:
                        is_solved[b] = True
                        final_solutions[b] = consensus_text
                        current_candidates[b] = [consensus_text] * self.num_agents
            
            # Check for global convergence
            if all(is_solved):
                print(">>> Full batch converged.")
                break

        
        final_texts = []
        for b in range(batch_size):
            if final_solutions[b]:
                # Case A: Consensus was reached during the loop
                final_texts.append(final_solutions[b])
            else:
                # Case B: No Consensus -> Strictly Fallback to Agent 0
                # print(f"[WARN] Item {b} not solved. Falling back to Agent 0's final candidate.")
                final_texts.append(current_candidates[b][0])

        return self._pack_output(lm_inputs, final_texts), interaction_history
    

class ApiCallingWrapperWg_Mixture:    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs

        self.num_agents = 3
        self.agents = []
        
        # agent_configs = [
        #     {"provider": "openai",   "model": "gpt-4o"},
        #     {"provider": "deepseek", "model": "deepseek-chat"},
        #     {"provider": "kimi",     "model": "kimi-latest"}
        # ]

        # for i, cfg in enumerate(agent_configs):
        #     agent = ConcurrentLLM(
        #         provider=cfg["provider"],
        #         model_name=cfg["model"],
        #         max_concurrency=config.model_config.max_concurrency
        #     )
            
        #     self.agents.append(agent)
        #     print(f'Multi-Agent Setup: Agent {i} {cfg["model"]} initialized')

        for i in range(self.num_agents):
            agent = ConcurrentLLM(
                provider="openai", # model_info.provider_name,
                model_name="gpt-4o", # model_info.model_name,
                max_concurrency=config.model_config.max_concurrency
            )
            self.agents.append(agent)
            print(f'Multi-Agent Setup: Agent {i} ({model_info.model_name}) initialized')

        import time
        time.sleep(10)

    def _get_messages_list(self, lm_inputs):
        messages_list = lm_inputs.non_tensor_batch.get('messages_list', None)
        
        if messages_list is not None:
            messages_list = messages_list.tolist()
        else:
            input_ids = lm_inputs.batch.get('input_ids', None)
            if input_ids is None:
                raise KeyError("messages_list missing and input_ids unavailable")
            if hasattr(input_ids, "detach"):
                input_ids = input_ids.detach().cpu()
            
            fallback_messages = []
            for ids in input_ids:
                text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                fallback_messages.append([{"role": "user", "content": text}])
            messages_list = fallback_messages
            print("[WARN] Constructed fallback prompts from input_ids")

        return messages_list
    
    def _run_all_agents(self, inputs_map):
        agent_outputs = {}
        
        for i, agent in enumerate(self.agents):
            results, failed_messages = agent.run_batch(messages_list=inputs_map[i], **self.llm_kwargs)
            
            assert not failed_messages, f"Agent {i} failed on some messages"

            agent_outputs[i] = [result["response"] for result in results]
        
        return agent_outputs
    
    def _prepare_orchestrator_prompts(self, base_messages, current_candidates):
        orchestrator_inputs = []
        batch_size = len(base_messages)

        for b in range(batch_size):
            original_hist = base_messages[b]
            candidates = current_candidates[b] 
            
            candidates_str = "=== CANDIDATE SOLUTIONS FROM AGENTS ===\n"
            for idx, text in enumerate(candidates):
                snippet = text.replace("<think>", "").replace("</think>", "").strip()
                candidates_str += f"--- Agent {idx} Proposed: ---\n{snippet}\n\n"

            last_msg_dict = original_hist[-1]
            base_content = last_msg_dict['content']

            if "assistant" in base_content[-20:]:
                base_content = base_content.rsplit("assistant", 1)[0].strip()

            if base_content.startswith("system") and "user\n" in base_content:
                base_content = base_content.split("user\n", 1)[1].strip()

            
            orchestrator_instruction = (
                f"{candidates_str}"
                "=== ORCHESTRATOR INSTRUCTIONS ===\n"
                "You are the Lead Orchestrator. You are NOT the agent acting in the environment.\n"
                "Your goal is to SELECT the single best solution from the candidates provided above.\n\n"
                "1. Read the 'Task Context' to understand the goal.\n"
                "2. Critically evaluate EACH agent's proposal. explicitly mentioning 'Agent 0', 'Agent 1', etc.\n"
                "3. Select the ONE agent that offers the optimal solution.\n"
                "4. You are STRICTLY FORBIDDEN from creating a new solution, merging proposals, or modifying the chosen agent's output. You must pick exactly one existing candidate.\n\n"
                "REQUIRED OUTPUT FORMAT:\n"
                "<think>\n"
                "Critique of Agent 0: ...\n"
                "Critique of Agent 1: ...\n"
                "Critique of Agent 2: ...\n"
                "Conclusion: The best path is ...\n"
                "</think>\n"
                "<action>...final action...</action>"
            )
            
            combined_content = (
                f"=== TASK CONTEXT (READ ONLY) ===\n"
                f"{base_content}\n"
                f"================================\n\n"
                f"{orchestrator_instruction}\n\n"
                f"assistant"
            )

            new_hist = [dict(msg) for msg in original_hist]
            new_hist[-1]['content'] = combined_content
            
            orchestrator_inputs.append(new_hist)

        return orchestrator_inputs
    
    def _pack_output(self, inputs, final_texts):
        lm_outputs = DataProto()
        batch_len = len(final_texts)
        env_ids = inputs.non_tensor_batch.get('env_ids', np.arange(batch_len, dtype=object))
        group_ids = inputs.non_tensor_batch.get('group_ids', np.zeros(batch_len, dtype=object))
        
        lm_outputs.non_tensor_batch = {
            'response_texts': final_texts,
            'env_ids': env_ids,
            'group_ids': group_ids
        }
        lm_outputs.meta_info = inputs.meta_info
        return lm_outputs
    
    def _print_debug(self, iteration, raw_response):
        print("-" * 50)
        print(f"DEBUG: End of Round {iteration} (Item 0)")
        print(f"Agent 0 Output:\n{raw_response[0]}")
        print(f"Agent 1 Output:\n{raw_response[1]}")
        print(f"Agent 2 Output:\n{raw_response[2]}")
        print("-" * 50)
    
    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        messages_list = self._get_messages_list(lm_inputs)
        
        # print(f"messages_list: {messages_list}")
        target_phrase = "You are Qwen, created by Alibaba Cloud."
        for history in messages_list:
            for msg in history:
                if target_phrase in msg.get("content", ""):
                    # Remove the phrase regardless of the message role
                    msg["content"] = msg["content"].replace(target_phrase, "")

        # print(f"messages_list: {messages_list}")
        # exit()

        batch_size = len(messages_list)
        
        # --- Step 1: Initial Independent Generation ---
        init_prompts = {i: messages_list for i in range(self.num_agents)}
        results = self._run_all_agents(init_prompts)
        
        candidates_by_batch = [] # candidates[batch_idx] = [response_0, response_1, response_2]
        for b in range(batch_size):
            row = [results[i][b] for i in range(self.num_agents)]
            candidates_by_batch.append(row)

        # self._print_debug("First", [results[i][0] for i in range(self.num_agents)])

        # --- Step 2: Orchestrator Aggregation ---
        orchestrator_prompts = self._prepare_orchestrator_prompts(messages_list, candidates_by_batch)

        # print(f"orchestrator_prompts[0]: {orchestrator_prompts[0]}")
        # _ = input("Next")
        
        orchestrator_results, failed = self.agents[0].run_batch(
            messages_list=orchestrator_prompts, 
            **self.llm_kwargs
        )
        assert not failed, "Orchestrator failed on some messages"

        # print(f"orchestrator_results[0]: {orchestrator_results[0]}")
        # _ = input("Next")
        
        final_texts = [res["response"] for res in orchestrator_results]

        # print(f"final_texts[0]: {final_texts[0]}")
        # _ = input("Next")

        interaction_history = []
        
        for b in range(batch_size):
            # Gather Phase 1 Outputs (Independent Agents)
            phase1_outputs = {
                f"agent_{i}": candidates_by_batch[b][i] 
                for i in range(self.num_agents)
            }
            
            # Construct the history object for this specific batch item
            batch_item_history = {
                "batch_index": b,
                "phase_1_independent": {
                    "input": messages_list[b],
                    "outputs": phase1_outputs
                },
                "phase_2_orchestration": {
                    "input": orchestrator_prompts[b],
                    "output": final_texts[b]
                }
            }
            interaction_history.append(batch_item_history)
        
        return self._pack_output(lm_inputs, final_texts), interaction_history
    

class ApiCallingWrapperWg_Sequential_v1:    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs

        self.num_agents = 3
        self.agents = []
        
        # agent_configs = [
        #     {"provider": "deepseek", "model": "deepseek-chat"},
        #     {"provider": "kimi",     "model": "kimi-latest"},
        #     {"provider": "openai",   "model": "gpt-4o"}
        # ]

        # for i, cfg in enumerate(agent_configs):
        #     agent = ConcurrentLLM(
        #         provider=cfg["provider"],
        #         model_name=cfg["model"],
        #         max_concurrency=config.model_config.max_concurrency
        #     )
            
        #     self.agents.append(agent)
        #     print(f'Multi-Agent Setup: Agent {i} {cfg["model"]} initialized')

        for i in range(self.num_agents):
            agent = ConcurrentLLM(
                provider=model_info.provider_name,
                model_name=model_info.model_name,
                max_concurrency=config.model_config.max_concurrency
            )
            self.agents.append(agent)
            print(f'Multi-Agent Setup: Agent {i} ({model_info.model_name}) initialized')

        import time
        time.sleep(10)

    def _get_messages_list(self, lm_inputs):
        messages_list = lm_inputs.non_tensor_batch.get('messages_list', None)
        
        if messages_list is not None:
            messages_list = messages_list.tolist()
        else:
            input_ids = lm_inputs.batch.get('input_ids', None)
            if input_ids is None:
                raise KeyError("messages_list missing and input_ids unavailable")
            if hasattr(input_ids, "detach"):
                input_ids = input_ids.detach().cpu()
            
            fallback_messages = []
            for ids in input_ids:
                text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                fallback_messages.append([{"role": "user", "content": text}])
            messages_list = fallback_messages
            print("[WARN] Constructed fallback prompts from input_ids")

        return messages_list
    
    def _prepare_refinement_prompts(self, base_messages, previous_responses):
        """
        Constructs the prompt for Agent [i].
        Input: Original Task + Output from Agent [i-1].
        """
        refinement_inputs = []
        batch_size = len(base_messages)

        for b in range(batch_size):
            original_hist = base_messages[b]
            prev_response = previous_responses[b]
            
            snippet = prev_response.replace("<think>", "").replace("</think>", "").strip()

            last_msg_dict = original_hist[-1]
            base_content = last_msg_dict['content']

            if "assistant" in base_content[-20:]:
                base_content = base_content.rsplit("assistant", 1)[0].strip()

            if base_content.startswith("system") and "user\n" in base_content:
                base_content = base_content.split("user\n", 1)[1].strip()


            refinement_instruction = (
                f"You are a Verifier and Refiner Agent. You are NOT the agent acting in the environment.\n"
                f"Your goal is to verify the Previous Proposal below. Correct logical errors and refine the reasoning.\n\n"
                "=== PREVIOUS PROPOSAL ===\n"
                f"{snippet}\n"
                "=========================\n\n"
                "=== INSTRUCTIONS ===\n"
                "1. Read the 'Task Context' to understand the user's original goal.\n"
                "2. Analyze the 'Previous Proposal' for correctness and completeness.\n"
                "3. If the proposal is correct, output it. If it is flawed, generate a BETTER version.\n"
                "4. Output the solution wrapped in <action> tags.\n\n"
                "FORMAT:\n"
                "<think> Critique of previous proposal... </think>\n"
                "<action>...solution...</action>"
            )
            
            combined_content = (
                f"=== TASK CONTEXT (READ ONLY) ===\n"
                f"{base_content}\n"
                f"================================\n\n"
                f"{refinement_instruction}\n\n"
                f"assistant"
            )

            new_hist = [dict(msg) for msg in original_hist]
            new_hist[-1]['content'] = combined_content
            
            refinement_inputs.append(new_hist)

        return refinement_inputs
    
    def _pack_output(self, inputs, final_texts):
        lm_outputs = DataProto()
        batch_len = len(final_texts)
        env_ids = inputs.non_tensor_batch.get('env_ids', np.arange(batch_len, dtype=object))
        group_ids = inputs.non_tensor_batch.get('group_ids', np.zeros(batch_len, dtype=object))
        
        lm_outputs.non_tensor_batch = {
            'response_texts': final_texts,
            'env_ids': env_ids,
            'group_ids': group_ids
        }
        lm_outputs.meta_info = inputs.meta_info
        return lm_outputs

    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        messages_list = self._get_messages_list(lm_inputs)

        target_phrase = "You are Qwen, created by Alibaba Cloud."
        for history in messages_list:
            for msg in history:
                if target_phrase in msg.get("content", ""):
                    # Remove the phrase regardless of the message role
                    msg["content"] = msg["content"].replace(target_phrase, "")

        batch_size = len(messages_list)
        interaction_history = [
            {"batch_index": b, "sequence_steps": []} 
            for b in range(batch_size)
        ]
        
        current_responses = []
        for i in range(self.num_agents):
            # print(f"\n--- Running Agent {i} ---")
            
            if i == 0:
                current_prompts = messages_list
                # print("Task: Initial Generation")
            else:
                current_prompts = self._prepare_refinement_prompts(messages_list, current_responses)
                # print(f"Task: Verifying Agent {i-1}")

            step_description = f"agent_{i}"

            results, failed = self.agents[i].run_batch(
                messages_list=current_prompts, 
                **self.llm_kwargs
            )
            assert not failed, f"Agent {i} failed on some messages"
            
            current_responses = [res["response"] for res in results]

            debug_snippet = current_responses[0].replace("\n", " ")
            # print(f" > Agent {i} Result (Item 0): {results[0]}...")
            # _ = input("Next")

            # print(f" > Agent {i} Result (Item 0): {debug_snippet}...")
            # _ = input("Next")

            for b in range(batch_size):
                step_record = {
                    "agent_index": i,
                    "step_description": step_description,
                    "input": current_prompts[b],
                    "output": current_responses[b]
                }
                interaction_history[b]["sequence_steps"].append(step_record)
        
        return self._pack_output(lm_inputs, current_responses), interaction_history




class ApiCallingWrapperWg_Sequential_v2:    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs

        self.num_agents = 3
        self.agents = []
        

        for i in range(self.num_agents):
            agent = ConcurrentLLM(
                provider="openai",
                model_name="gpt-4o",
                max_concurrency=config.model_config.max_concurrency
            )
            self.agents.append(agent)
            print(f'Multi-Agent Setup: Agent {i} ({model_info.model_name}) initialized')

        import time
        time.sleep(10)

    def _get_messages_list(self, lm_inputs):
        messages_list = lm_inputs.non_tensor_batch.get('messages_list', None)
        
        if messages_list is not None:
            messages_list = messages_list.tolist()
        else:
            input_ids = lm_inputs.batch.get('input_ids', None)
            if input_ids is None:
                raise KeyError("messages_list missing and input_ids unavailable")
            if hasattr(input_ids, "detach"):
                input_ids = input_ids.detach().cpu()
            
            fallback_messages = []
            for ids in input_ids:
                text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                fallback_messages.append([{"role": "user", "content": text}])
            messages_list = fallback_messages
            print("[WARN] Constructed fallback prompts from input_ids")

        return messages_list
    
    def _prepare_verification_prompts(self, base_messages, agent0_responses):
        prompts = []
        batch_size = len(base_messages)

        for b in range(batch_size):
            original_hist = base_messages[b]
            proposal = agent0_responses[b]
            
            last_msg_dict = original_hist[-1]
            base_content = last_msg_dict['content']

            if "assistant" in base_content[-20:]:
                base_content = base_content.rsplit("assistant", 1)[0].strip()
            if base_content.startswith("system") and "user\n" in base_content:
                base_content = base_content.split("user\n", 1)[1].strip()

            verification_instruction = (
                "=== INITIAL PROPOSAL (Agent 0) ===\n"
                f"{proposal}\n"
                "==================================\n\n"
                f"You are a Verifier Agent. You are NOT acting in the environment.\n"
                f"Your goal is to EVALUATE the Initial Proposal below. Identify logical errors, missing steps, or potential risks.\n\n"
                "=== INSTRUCTIONS ===\n"
                "1. Read the 'Task Context' to understand the goal.\n"
                "2. Critique the 'Initial Proposal' for correctness.\n"
                "3. Output a purely evaluative message detailing pros and cons.\n"
                "4. DO NOT generate the final action yet, just the evaluation.\n"
            )
            
            combined_content = (
                f"=== TASK CONTEXT (READ ONLY) ===\n"
                f"{base_content}\n"
                f"================================\n\n"
                f"{verification_instruction}\n\n"
                f"assistant"
            )

            new_hist = [dict(msg) for msg in original_hist]
            new_hist[-1]['content'] = combined_content
            prompts.append(new_hist)

        return prompts

    def _prepare_final_refinement_prompts(self, base_messages, agent0_responses, agent1_responses):
        prompts = []
        batch_size = len(base_messages)

        for b in range(batch_size):
            original_hist = base_messages[b]
            proposal = agent0_responses[b]
            evaluation = agent1_responses[b]
            
            # Extract clean task description
            last_msg_dict = original_hist[-1]
            base_content = last_msg_dict['content']

            if "assistant" in base_content[-20:]:
                base_content = base_content.rsplit("assistant", 1)[0].strip()
            if base_content.startswith("system") and "user\n" in base_content:
                base_content = base_content.split("user\n", 1)[1].strip()

            refinement_instruction = (
                "=== INITIAL PROPOSAL (Agent 0) ===\n"
                f"{proposal}\n"
                "==================================\n\n"
                "=== EVALUATION (Agent 1) ===\n"
                f"{evaluation}\n"
                "============================\n\n"
                f"You are the Final Refiner Agent.\n"
                f"Your goal is to produce the final correct action based on the Proposal and the Evaluation provided below.\n\n"
                "=== INSTRUCTIONS ===\n"
                "1. Synthesize the original goal, the proposal, and the evaluation.\n"
                "2. Explain your reasoning for the final fix in <think> tags.\n"
                "3. Output the final solution wrapped in <action> tags.\n"
            )
            
            combined_content = (
                f"=== TASK CONTEXT (READ ONLY) ===\n"
                f"{base_content}\n"
                f"================================\n\n"
                f"{refinement_instruction}\n\n"
                f"assistant"
            )

            new_hist = [dict(msg) for msg in original_hist]
            new_hist[-1]['content'] = combined_content
            prompts.append(new_hist)

        return prompts
    
    def _pack_output(self, inputs, final_texts):
        lm_outputs = DataProto()
        batch_len = len(final_texts)
        env_ids = inputs.non_tensor_batch.get('env_ids', np.arange(batch_len, dtype=object))
        group_ids = inputs.non_tensor_batch.get('group_ids', np.zeros(batch_len, dtype=object))
        
        lm_outputs.non_tensor_batch = {
            'response_texts': final_texts,
            'env_ids': env_ids,
            'group_ids': group_ids
        }
        lm_outputs.meta_info = inputs.meta_info
        return lm_outputs

    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        messages_list = self._get_messages_list(lm_inputs)

        target_phrase = "You are Qwen, created by Alibaba Cloud."
        for history in messages_list:
            for msg in history:
                if target_phrase in msg.get("content", ""):
                    msg["content"] = msg["content"].replace(target_phrase, "")

        batch_size = len(messages_list)
        interaction_history = [
            {"batch_index": b, "sequence_steps": []} 
            for b in range(batch_size)
        ]
        

        agent0_output = [] 
        agent1_output = [] 
        current_responses = []

        for i in range(self.num_agents):
            if i == 0:
                current_prompts = messages_list
            elif i == 1:
                agent0_output = current_responses
                current_prompts = self._prepare_verification_prompts(messages_list, agent0_output)
            elif i == 2:
                agent1_output = current_responses
                current_prompts = self._prepare_final_refinement_prompts(messages_list, agent0_output, agent1_output)

            step_description = f"agent_{i}"

            results, failed = self.agents[i].run_batch(
                messages_list=current_prompts, 
                **self.llm_kwargs
            )
            assert not failed, f"Agent {i} failed on some messages"
            
            current_responses = [res["response"] for res in results]

            for b in range(batch_size):
                step_record = {
                    "agent_index": i,
                    "step_description": step_description,
                    "input": current_prompts[b],
                    "output": current_responses[b]
                }
                interaction_history[b]["sequence_steps"].append(step_record)
        
        return self._pack_output(lm_inputs, current_responses), interaction_history
    


class ApiCallingWrapperWg_Sequential:    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs

        self.num_agents = 3
        self.agents = []
        

        for i in range(self.num_agents):
            agent = ConcurrentLLM(
                provider="openai",
                model_name="gpt-4o",
                max_concurrency=config.model_config.max_concurrency
            )
            self.agents.append(agent)
            print(f'Multi-Agent Setup: Agent {i} ({model_info.model_name}) initialized')


        self.previous_step_thoughts = None

        import time
        time.sleep(10)

    def _get_messages_list(self, lm_inputs):
        messages_list = lm_inputs.non_tensor_batch.get('messages_list', None)
        
        if messages_list is not None:
            messages_list = messages_list.tolist()
        else:
            input_ids = lm_inputs.batch.get('input_ids', None)
            if input_ids is None:
                raise KeyError("messages_list missing and input_ids unavailable")
            if hasattr(input_ids, "detach"):
                input_ids = input_ids.detach().cpu()
            
            fallback_messages = []
            for ids in input_ids:
                text = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                fallback_messages.append([{"role": "user", "content": text}])
            messages_list = fallback_messages
            print("[WARN] Constructed fallback prompts from input_ids")

        return messages_list
    
    def _prepare_verification_prompts(self, base_messages, agent0_responses):
        prompts = []
        batch_size = len(base_messages)

        for b in range(batch_size):
            original_hist = base_messages[b]
            proposal = agent0_responses[b]
            
            last_msg_dict = original_hist[-1]
            base_content = last_msg_dict['content']

            if "assistant" in base_content[-20:]:
                base_content = base_content.rsplit("assistant", 1)[0].strip()
            if base_content.startswith("system") and "user\n" in base_content:
                base_content = base_content.split("user\n", 1)[1].strip()

            verification_instruction = (
                "=== INITIAL PROPOSAL (Agent 0) ===\n"
                f"{proposal}\n"
                "==================================\n\n"
                f"You are a Verifier Agent. You are NOT acting in the environment.\n"
                f"Your goal is to EVALUATE the Initial Proposal below. Identify logical errors, missing steps, or potential risks.\n\n"
                "=== INSTRUCTIONS ===\n"
                "1. Read the 'Task Context' to understand the goal.\n"
                "2. Critique the 'Initial Proposal' for correctness.\n"
                "3. Output a purely evaluative message detailing pros and cons.\n"
                "4. DO NOT generate the final action yet, just the evaluation.\n"
            )
            
            combined_content = (
                f"=== TASK CONTEXT (READ ONLY) ===\n"
                f"{base_content}\n"
                f"================================\n\n"
                f"{verification_instruction}\n\n"
                f"assistant"
            )

            new_hist = [dict(msg) for msg in original_hist]
            new_hist[-1]['content'] = combined_content
            prompts.append(new_hist)

        return prompts

    def _prepare_final_refinement_prompts(self, base_messages, agent0_responses, agent1_responses):
        prompts = []
        batch_size = len(base_messages)

        for b in range(batch_size):
            original_hist = base_messages[b]
            proposal = agent0_responses[b]
            evaluation = agent1_responses[b]
            
            # Extract clean task description
            last_msg_dict = original_hist[-1]
            base_content = last_msg_dict['content']

            if "assistant" in base_content[-20:]:
                base_content = base_content.rsplit("assistant", 1)[0].strip()
            if base_content.startswith("system") and "user\n" in base_content:
                base_content = base_content.split("user\n", 1)[1].strip()

            refinement_instruction = (
                "=== INITIAL PROPOSAL (Agent 0) ===\n"
                f"{proposal}\n"
                "==================================\n\n"
                "=== EVALUATION (Agent 1) ===\n"
                f"{evaluation}\n"
                "============================\n\n"
                f"You are the Final Refiner Agent.\n"
                f"Your goal is to produce the final correct action based on the Proposal and the Evaluation provided below.\n\n"
                "=== INSTRUCTIONS ===\n"
                "1. Synthesize the original goal, the proposal, and the evaluation.\n"
                "2. Explain your reasoning for the final fix in <think> tags.\n"
                "3. Output the final solution wrapped in <action> tags.\n"
            )
            
            combined_content = (
                f"=== TASK CONTEXT (READ ONLY) ===\n"
                f"{base_content}\n"
                f"================================\n\n"
                f"{refinement_instruction}\n\n"
                f"assistant"
            )

            new_hist = [dict(msg) for msg in original_hist]
            new_hist[-1]['content'] = combined_content
            prompts.append(new_hist)

        return prompts
    
    def _pack_output(self, inputs, final_texts):
        lm_outputs = DataProto()
        batch_len = len(final_texts)
        env_ids = inputs.non_tensor_batch.get('env_ids', np.arange(batch_len, dtype=object))
        group_ids = inputs.non_tensor_batch.get('group_ids', np.zeros(batch_len, dtype=object))
        
        lm_outputs.non_tensor_batch = {
            'response_texts': final_texts,
            'env_ids': env_ids,
            'group_ids': group_ids
        }
        lm_outputs.meta_info = inputs.meta_info
        return lm_outputs

    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        messages_list = self._get_messages_list(lm_inputs)

        target_phrase = "You are Qwen, created by Alibaba Cloud."
        for history in messages_list:
            for msg in history:
                if target_phrase in msg.get("content", ""):
                    msg["content"] = msg["content"].replace(target_phrase, "")
        
        
        if self.previous_step_thoughts is not None:
            for b_idx, history in enumerate(messages_list):
                prev_thought = self.previous_step_thoughts[b_idx]
                
                if prev_thought and len(history) > 0:
                    last_msg = history[-1]
                    content = last_msg.get("content", "")

                    pattern = r"Below are the most recent (\d+) observations and the corresponding actions you took:"
                    
                    if re.search(pattern, content):
                        def replace_callback(match):
                            num_obs = match.group(1)
                            new_sentence = f"Below are the previous thought process, the most recent {num_obs} observations and the corresponding actions you took:"
                            return f"{new_sentence}\n[{prev_thought}]"

                        last_msg["content"] = re.sub(pattern, replace_callback, content, count=1)
                    else:
                        print(f"[WARN] Target phrase not found in batch {b_idx}, appending thought to end.")
                        last_msg["content"] += f"\n\n[{prev_thought}]"
        

        batch_size = len(messages_list)
        interaction_history = [
            {"batch_index": b, "sequence_steps": []} 
            for b in range(batch_size)
        ]
        

        agent0_output = [] 
        agent1_output = [] 
        current_responses = []

        for i in range(self.num_agents):
            if i == 0:
                current_prompts = messages_list
            elif i == 1:
                agent0_output = current_responses
                current_prompts = self._prepare_verification_prompts(messages_list, agent0_output)
            elif i == 2:
                agent1_output = current_responses
                current_prompts = self._prepare_final_refinement_prompts(messages_list, agent0_output, agent1_output)

            step_description = f"agent_{i}"

            results, failed = self.agents[i].run_batch(
                messages_list=current_prompts, 
                **self.llm_kwargs
            )
            assert not failed, f"Agent {i} failed on some messages"
            
            current_responses = [res["response"] for res in results]

            for b in range(batch_size):
                step_record = {
                    "agent_index": i,
                    "step_description": step_description,
                    "input": current_prompts[b],
                    "output": current_responses[b]
                }
                interaction_history[b]["sequence_steps"].append(step_record)

        
        self.previous_step_thoughts = []
        for response in current_responses:
            match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            if match:
                self.previous_step_thoughts.append(match.group(1).strip())
            else:
                self.previous_step_thoughts.append(None)
        
        return self._pack_output(lm_inputs, current_responses), interaction_history
