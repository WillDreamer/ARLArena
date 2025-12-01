from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
# from .base_llm import ConcurrentLLM
# import time

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
		Supports multimodal inputs (images/videos) via multi_modal_data in non_tensor_batch.
		
		Note: In the rollout loop, raw_prompt_ids and multi_modal_data are popped from the batch
		and passed separately. This method handles both cases:
		1. If multi_modal_data is in non_tensor_batch, use it with raw_prompt_ids for multimodal generation
		2. Otherwise, decode input_ids to text for text-only generation
		"""
		# NOTE: free_cache_engine is not used in the vllm wrapper. Only used in the verl vllm.
		# cache_action = lm_inputs.meta_info.get('cache_action', None)

		import numpy as np
		non_tensor_batch = lm_inputs.non_tensor_batch
		
		# Handle multimodal data if present
		if "multi_modal_data" in non_tensor_batch and "raw_prompt_ids" in non_tensor_batch:
			# For multimodal, we need to use raw_prompt_ids (token IDs) instead of decoded text
			# to preserve the multimodal token structure
			raw_prompt_ids = non_tensor_batch["raw_prompt_ids"]
			multi_modal_data = non_tensor_batch["multi_modal_data"]
			
			# Prepare vLLM inputs with multimodal data using token IDs
			vllm_inputs = []
			for prompt_token_ids, mm_data in zip(raw_prompt_ids, multi_modal_data):
				# Ensure token IDs are in the correct format (list of ints)
				if isinstance(prompt_token_ids, np.ndarray):
					prompt_token_ids = prompt_token_ids.tolist()
				elif not isinstance(prompt_token_ids, list):
					# Handle other array-like types
					prompt_token_ids = list(prompt_token_ids) if hasattr(prompt_token_ids, '__iter__') else [prompt_token_ids]
				vllm_input = {"prompt_token_ids": prompt_token_ids, "multi_modal_data": mm_data}
				vllm_inputs.append(vllm_input)
			outputs = self.llm.generate(vllm_inputs, sampling_params=self.sampling_params)
		else:
			# Text-only generation: decode input_ids to text
			# Note: input_ids might not be in batch if they were popped, so we need to handle that
			if "input_ids" in lm_inputs.batch:
				input_ids = lm_inputs.batch['input_ids']
				input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
				input_texts = [i.replace("<|endoftext|>", "") for i in input_texts]
				outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
			else:
				# Fallback: if input_ids were popped, try to reconstruct from raw_prompt_ids if available
				if "raw_prompt_ids" in non_tensor_batch:
					# Decode raw_prompt_ids to text
					raw_prompt_ids = non_tensor_batch["raw_prompt_ids"]
					input_texts = []
					for prompt_ids in raw_prompt_ids:
						if isinstance(prompt_ids, np.ndarray):
							prompt_ids = prompt_ids.tolist()
						text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
						input_texts.append(text.replace("<|endoftext|>", ""))
					outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
				else:
					raise ValueError("Cannot generate sequences: neither input_ids nor raw_prompt_ids available in batch")
		
		texts = [output.outputs[0].text for output in outputs] 
		lm_outputs = DataProto()
		lm_outputs.non_tensor_batch = {
			'response_texts': texts,
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
		lm_outputs.meta_info = lm_inputs.meta_info

		return lm_outputs
	
# class ApiCallingWrapperWg:
#     """Wrapper class for API-based LLM calls that fits into the VERL framework"""
    
#     def __init__(self, config, tokenizer):
#         self.config = config
#         self.tokenizer = tokenizer
#         model_info = config.model_info[config.model_config.model_name]
#         self.llm_kwargs = model_info.generation_kwargs
        
        
#         self.llm = ConcurrentLLM(
# 			provider=model_info.provider_name,
#             model_name=model_info.model_name,
#             max_concurrency=config.model_config.max_concurrency
#         )
        
#         print(f'API-based LLM ({model_info.provider_name} - {model_info.model_name}) initialized')


#     def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
#         """
#         Convert the input ids to text, make API calls to generate responses, 
#         and create a DataProto with the results.
#         """

#         messages_list = lm_inputs.non_tensor_batch['messages_list'].tolist()
#         results, failed_messages = self.llm.run_batch(
#             messages_list=messages_list,
#             **self.llm_kwargs
#         )
#         assert not failed_messages, f"Failed to generate responses for the following messages: {failed_messages}"

#         texts = [result["response"] for result in results]
#         print(f'[DEBUG] texts: {texts}')
#         lm_outputs = DataProto()
#         lm_outputs.non_tensor_batch = {
# 			'response_texts': texts,
# 			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
# 			'group_ids': lm_inputs.non_tensor_batch['group_ids']
# 		} # this is a bit hard-coded to bypass the __init__ check in DataProto
#         lm_outputs.meta_info = lm_inputs.meta_info
        
#         return lm_outputs

