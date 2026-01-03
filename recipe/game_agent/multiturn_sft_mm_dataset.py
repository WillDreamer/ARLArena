# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

import copy
import logging
import re
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs


def convert_nested_value_to_list_recursive(data_item):
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        # Convert to list, then recursively process the elements of the new list
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        # Base case: item is already a primitive type (int, str, float, bool, etc.)
        return data_item


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self, parquet_files: str | list[str], tokenizer, config=None, processor=None):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.processor: Optional[ProcessorMixin] = processor
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.tools_key = multiturn_config.get("tools_key", "tools")
        self.enable_thinking_key = multiturn_config.get("enable_thinking_key", "enable_thinking")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        # Multimodal keys
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "video")
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        
        self.column_names = ["input_ids", "attention_mask", "position_ids", "loss_mask"]

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        # Check if messages_key exists in dataframe columns
        if self.messages_key not in self.dataframe.columns:
            # If messages_key is "messages" but not found, try to convert from "turns"
            if self.messages_key == "messages" and "turns" in self.dataframe.columns:
                # Convert turns to messages format
                self.messages = self.dataframe["turns"].apply(
                    lambda turns: self._convert_turns_to_messages(series_to_item(turns))
                ).tolist()
            else:
                available_columns = list(self.dataframe.columns)
                raise KeyError(
                    f"Column '{self.messages_key}' not found in dataframe. "
                    f"Available columns: {available_columns}. "
                    f"Please configure 'multiturn.messages_key' to match one of the available columns, "
                    f"or ensure your dataframe contains a '{self.messages_key}' column. "
                    f"If your data is in 'turns' format, the code will automatically convert it to 'messages' format."
                )
        else:
            # Extract messages list from dataframe
            self.messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()

        # Extract tools list from dataframe
        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = None
        # Extract enable_thinking list from dataframe
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None


    def __len__(self):
        return len(self.messages)

    def _convert_turns_to_messages(self, turns):
        """
        Convert turns format to messages format.
        Turns format: [{"step": int, "inputs"/"messages": [{"content": str, "role": str}], "outputs": str}, ...]
        Messages format: [{"role": str, "content": str}, ...]
        Supports both "inputs" and "messages" keys for backward compatibility.
        Also handles cases where turns["messages"] is already a flat list of messages.
        """
        messages = []
        for turn in turns:
            # Check if turn["messages"] is already a flat list of messages (already converted)
            if "messages" in turn and turn["messages"]:
                # Check if it's already in the correct format (list of dicts with "role" and "content")
                if isinstance(turn["messages"], list) and len(turn["messages"]) > 0:
                    first_msg = turn["messages"][0]
                    if isinstance(first_msg, dict) and "role" in first_msg and "content" in first_msg:
                        # Already in correct format, use directly
                        messages.extend(turn["messages"])
                    else:
                        # Not in correct format, convert it
                        for input_msg in turn["messages"]:
                            messages.append({
                                "role": input_msg.get("role", "user"),
                                "content": input_msg.get("content", "")
                            })
                else:
                    # Empty or invalid, skip
                    pass
            elif "inputs" in turn and turn["inputs"]:
                # Legacy format with "inputs" key
                for input_msg in turn["inputs"]:
                    messages.append({
                        "role": input_msg.get("role", "user"),
                        "content": input_msg.get("content", "")
                    })
            # Add assistant output as a message
            if "outputs" in turn and turn["outputs"]:
                messages.append({
                    "role": "assistant",
                    "content": turn["outputs"]
                })
        return messages

    def _build_messages(self, example: dict):
        """
        Build messages with multimodal content support.
        Similar to RLHFDataset._build_messages, this method:
        1. Extracts messages from example using messages_key (or converts from turns if needed)
        2. If image_key or video_key exists in example, parses <image> and <video> tags in message content
        
        Args:
            example: Dictionary containing the row data from dataframe
            
        Returns:
            List of message dictionaries with multimodal content format
        """
        # Get messages from example (don't pop, as we may need the original structure)
        messages = example.get(self.messages_key, [])
        
        # If messages_key is not found but turns exists, convert from turns
        if not messages and "turns" in example:
            turns = example.get("turns", [])
            # Handle both list and numpy array cases
            if isinstance(turns, list) and len(turns) > 0:
                messages = self._convert_turns_to_messages(turns)
            elif hasattr(turns, 'any') and turns.any():
                messages = self._convert_turns_to_messages(turns)
        
        # Handle different message formats (list, single dict, or nested structure)
        if not isinstance(messages, list):
            messages = [messages]
        
        # Create a copy to avoid modifying the original
        messages = copy.deepcopy(messages)

        # Check if example has image_key or video_key (similar to rl_dataset)
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str):
                    content_list = []
                    segments = re.split("(<image>|<video>)", content)
                    segments = [item for item in segments if item != ""]
                    for segment in segments:
                        if segment == "<image>":
                            content_list.append({"type": "image"})
                        elif segment == "<video>":
                            content_list.append({"type": "video"})
                        else:
                            content_list.append({"type": "text", "text": segment})

                    message["content"] = content_list

        return messages

    def _process_message_tokens(
        self,
        messages: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        is_assistant: bool = False,
        enable_thinking: Optional[bool] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        images=None,
        videos=None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Process tokens for a single message or a group of messages.

        Args:
            messages: List of message dictionaries
            start_idx: Start index in messages list
            end_idx: End index in messages list
            is_assistant: Whether this is an assistant message
            enable_thinking: Whether to enable thinking mode
            tools: Optional tools for the conversation
            images: Optional list of images for multimodal processing
            videos: Optional list of videos for multimodal processing

        Returns:
            Tuple of (tokens, loss_mask, attention_mask)
        """
        # Use processor if available, otherwise use tokenizer
        tokenizer_or_processor = self.processor if self.processor is not None else self.tokenizer

        if start_idx > 0:
            prev_applied_text = tokenizer_or_processor.apply_chat_template(
                messages[:start_idx],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                tools=tools,
                **self.apply_chat_template_kwargs,
            )
            if is_assistant:
                prev_applied_text_w_generation_prompt = tokenizer_or_processor.apply_chat_template(
                    messages[:start_idx],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                    tools=tools,
                    **self.apply_chat_template_kwargs,
                )
        else:
            prev_applied_text = ""
            if is_assistant:
                # When start_idx=0, we need to handle empty messages list
                # For empty list, we can't apply chat template, so use empty string
                # The generation prompt will be added by the collator
                prev_applied_text_w_generation_prompt = ""

        cur_applied_text = tokenizer_or_processor.apply_chat_template(
            messages[:end_idx],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            tools=tools,
            **self.apply_chat_template_kwargs,
        )
        # Get tokens for the current message only
        if is_assistant:
            generation_prompt_text = prev_applied_text_w_generation_prompt[len(prev_applied_text) :]
            if self.processor is not None:
                # For processor, we need to handle multimodal inputs differently
                # This is a simplified version - full multimodal processing happens in __getitem__
                generation_prompt_tokens = self.tokenizer.encode(
                    generation_prompt_text,
                    add_special_tokens=False,
                )
                _message_text = cur_applied_text[len(prev_applied_text_w_generation_prompt) :]
                _message_tokens = self.tokenizer.encode(
                    _message_text,
                    add_special_tokens=False,
                )
            else:
                generation_prompt_tokens = self.tokenizer.encode(
                    generation_prompt_text,
                    add_special_tokens=False,
                )
                _message_tokens = self.tokenizer.encode(
                    cur_applied_text[len(prev_applied_text_w_generation_prompt) :],
                    add_special_tokens=False,
                )
            message_tokens = generation_prompt_tokens + _message_tokens
            loss_mask = [0] * (len(generation_prompt_tokens)) + [1] * (
                len(message_tokens) - len(generation_prompt_tokens)
            )
        else:
            if self.processor is not None:
                # For processor, use tokenizer for text-only encoding in this context
                message_tokens = self.tokenizer.encode(
                    cur_applied_text[len(prev_applied_text) :],
                    add_special_tokens=False,
                )
            else:
                message_tokens = self.tokenizer.encode(
                    cur_applied_text[len(prev_applied_text) :],
                    add_special_tokens=False,
                )
            loss_mask = [0] * len(message_tokens)

        attention_mask = [1] * len(message_tokens)

        return message_tokens, loss_mask, attention_mask

    def _validate_and_convert_tokens(
        self,
        full_tokens: torch.Tensor,
        concat_tokens: list[int],
        concat_loss_mask: list[int],
        concat_attention_mask: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Validate tokenization and convert to tensors.

        Args:
            full_tokens: Full conversation tokens
            concat_tokens: Concatenated tokens
            concat_loss_mask: Concatenated loss mask
            concat_attention_mask: Concatenated attention mask

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask) as tensors
        """
        full_tokens_list = full_tokens.tolist()

        if len(concat_tokens) != len(full_tokens_list) or not all(
            a == b for a, b in zip(concat_tokens, full_tokens_list, strict=True)
        ):
            logging.warning(
                f"Token mismatch detected! Full tokenization length: {len(full_tokens_list)}, Concatenated tokens "
                f"length: {len(concat_tokens)}. Using concatenated version."
                # f"full tokens text: {self.tokenizer.decode(full_tokens_list)}"
                # f"concat tokens text: {self.tokenizer.decode(concat_tokens)}"
            )
            return (
                torch.tensor(concat_tokens, dtype=torch.long),
                torch.tensor(concat_loss_mask, dtype=torch.long),
                torch.tensor(concat_attention_mask, dtype=torch.long),
            )

        return (
            full_tokens,
            torch.tensor(concat_loss_mask, dtype=torch.long),
            torch.tensor(concat_attention_mask, dtype=torch.long),
        )

    def __getitem__(self, item):
        from verl.utils.dataset.vision_utils import process_image, process_video

        # Get the row as a dictionary (similar to rl_dataset)
        row_dict = self.dataframe.iloc[item].to_dict()
        
        # Build messages with multimodal content support (similar to rl_dataset)
        messages = self._build_messages(row_dict)
        
        # Validate messages
        if not messages or len(messages) == 0:
            raise ValueError(
                f"Empty messages for item {item}. "
                f"Row dict keys: {list(row_dict.keys())}. "
                f"messages_key: {self.messages_key}. "
                f"turns in row_dict: {'turns' in row_dict}. "
                f"Please check the data format."
            )
        
        # Get tools and enable_thinking from pre-extracted lists or row_dict
        tools = self.tools[item] if self.tools is not None else row_dict.get(self.tools_key, None)
        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else row_dict.get(self.enable_thinking_key, None)

        # Get images and videos from row_dict (similar to rl_dataset)
        images = None
        if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
            image_data = row_dict[self.image_key]
            # Handle both list and single image cases
            if not isinstance(image_data, list):
                image_data = [image_data]
            
            # Convert base64 format to process_image-compatible format
            processed_images = []
            for img in image_data:
                if isinstance(img, dict):
                    # Handle base64 format from JSON: {"format": "png_base64", "data"/"images": "base64_string"}
                    # Support both "data" and "images" keys for backward compatibility
                    data_key = None
                    if img.get("format") == "png_base64":
                        if "images" in img:
                            data_key = "images"
                        elif "data" in img:
                            data_key = "data"
                    
                    if data_key:
                        import base64
                        from io import BytesIO
                        # Decode base64 to bytes
                        image_bytes = base64.b64decode(img[data_key])
                        # Convert to process_image-compatible format: {"bytes": bytes}
                        img = {"bytes": image_bytes}
                    # Also handle simple dict format: {"images": "base64_string"} or {"data": "base64_string"}
                    elif "images" in img and isinstance(img["images"], str):
                        import base64
                        image_bytes = base64.b64decode(img["images"])
                        img = {"bytes": image_bytes}
                    elif "data" in img and isinstance(img["data"], str):
                        import base64
                        image_bytes = base64.b64decode(img["data"])
                        img = {"bytes": image_bytes}
                processed_images.append(process_image(img))
            images = processed_images
        
        # 在 __getitem__ 里，构造完 messages 和 images 之后，加一段：
        num_image_tokens = 0
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, str):
                num_image_tokens += c.count("<image>")
            elif isinstance(c, list):
                # 如果你前面把 string 转成了 [{"type": "image"}, {"type": "text", ...}] 之类，这里也可以数一下
                num_image_tokens += sum(1 for x in c if isinstance(x, dict) and x.get("type") == "image")

        if images is not None and num_image_tokens > 0:
            if len(images) == 1 and num_image_tokens > 1:
                # 把同一张图复制 num_image_tokens 份
                images = images * num_image_tokens
            elif len(images) != num_image_tokens:
                raise ValueError(f"images number ({len(images)}) is not equal to <image> number ({num_image_tokens})")
        
        videos = None
        if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
            video_data = row_dict[self.video_key]
            # Handle both list and single video cases
            if not isinstance(video_data, list):
                video_data = [video_data]
            videos = [process_video(video) for video in video_data]

        # TRL's DataCollatorForVisionLanguageModeling expects "messages" and optionally "images"
        # Return the format that TRL's collator expects - no tokenization needed here
        result = {
            "messages": messages,  # Original messages format for TRL
        }
        
        # Add images if available (TRL expects "images" key, not "image")
        # Note: For TRL, we should return raw image data, not processed images
        # But let's keep processed images for now and see if TRL can handle them
        if images is not None:
            result["images"] = images
        
        # Add tools and enable_thinking if available (for TRL's collator)
        if tools is not None:
            result["tools"] = tools
        if enable_thinking is not None:
            result["enable_thinking"] = enable_thinking

        # Final validation: ensure result is not empty
        if not result or len(result) == 0:
            raise ValueError(f"Empty result dictionary for item {item}. This should not happen.")
        
        # Ensure required keys are present for TRL
        if "messages" not in result:
            raise ValueError(f"Missing required key 'messages' in result for item {item}")

        return result
