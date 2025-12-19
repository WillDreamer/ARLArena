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
        self.video_key = config.get("video_key", "videos")
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
        Turns format: [{"step": int, "inputs": [{"content": str, "role": str}], "outputs": str}, ...]
        Messages format: [{"role": str, "content": str}, ...]
        """
        messages = []
        for turn in turns:
            # Add all input messages
            if "inputs" in turn and turn["inputs"]:
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
            if turns.any():
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
                    # Handle base64 format from JSON: {"format": "png_base64", "data": "base64_string"}
                    if img.get("format") == "png_base64" and "data" in img:
                        import base64
                        from io import BytesIO
                        # Decode base64 to bytes
                        image_bytes = base64.b64decode(img["data"])
                        # Convert to process_image-compatible format: {"bytes": bytes}
                        img = {"bytes": image_bytes}
                processed_images.append(process_image(img))
            images = processed_images

        videos = None
        if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
            video_data = row_dict[self.video_key]
            # Handle both list and single video cases
            if not isinstance(video_data, list):
                video_data = [video_data]
            videos = [process_video(video) for video in video_data]

        # Process with processor if available, otherwise use tokenizer
        if self.processor is not None:
            # Use processor for multimodal processing
            raw_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                tools=tools,
                **self.apply_chat_template_kwargs,
            )

            # Process multimodal inputs
            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
            full_tokens = model_inputs["input_ids"]
            full_attention_mask = model_inputs["attention_mask"]
            
            # Store multimodal data
            multi_modal_data = {}
            if images is not None:
                multi_modal_data["image"] = images
            if videos is not None:
                multi_modal_data["video"] = [video.numpy() if hasattr(video, 'numpy') else video for video in videos]

            # Extract other multimodal inputs if needed
            multi_modal_inputs = {}
            if self.return_multi_modal_inputs:
                for key, value in model_inputs.items():
                    if key not in ["input_ids", "attention_mask"]:
                        multi_modal_inputs[key] = value
                # Remove second_per_grid_ts if present (not used for training)
                multi_modal_inputs.pop("second_per_grid_ts", None)

        else:
            # Use tokenizer for text-only processing
            tokenizer = self.tokenizer
            try:
                full_tokens = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=False,
                    enable_thinking=enable_thinking,
                    **self.apply_chat_template_kwargs,
                )
                full_attention_mask = torch.ones_like(full_tokens[0])
            except Exception as e:
                logging.error(
                    f"Error applying chat template: {e}\nMessages: {messages}\nTools: {tools}\nEnable thinking: "
                    f"{enable_thinking}"
                )
                raise
            multi_modal_data = {}
            multi_modal_inputs = {}

        # Track concatenated tokens for validation
        concat_tokens = []
        concat_loss_mask = []
        concat_attention_mask = []

        i = 0
        while i < len(messages):
            cur_messages = messages[i]
            if cur_messages["role"] == "assistant":
                # Process assistant message
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, is_assistant=True, enable_thinking=enable_thinking, 
                    tools=tools, images=images, videos=videos
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i += 1
            elif cur_messages["role"] == "tool":
                # Process consecutive tool messages
                st = i
                ed = i + 1
                while ed < len(messages) and messages[ed]["role"] == "tool":
                    ed += 1
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, st, ed, enable_thinking=enable_thinking, tools=tools,
                    images=images, videos=videos
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i = ed
            elif cur_messages["role"] in ["user", "system"]:
                # Process user or system message
                if cur_messages["role"] == "system" and i != 0:
                    raise ValueError("System message should be the first message")
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, enable_thinking=enable_thinking, tools=tools,
                    images=images, videos=videos
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i += 1
            else:
                raise ValueError(f"Unknown role: {cur_messages['role']}")

        # For processor, we need to use the full tokens from processor
        # For tokenizer, we validate against concatenated tokens
        if self.processor is not None:
            # With processor, use the full tokens directly
            input_ids = full_tokens[0]
            attention_mask = full_attention_mask[0]
            # For loss_mask, we need to align with the processor's tokenization
            # This is a simplified approach - in practice, you may need more sophisticated alignment
            # We'll use the concatenated loss_mask as a reference, but pad/truncate to match processor output
            if len(concat_loss_mask) > 0:
                # Try to use concatenated loss_mask, but adjust length to match processor output
                if len(concat_loss_mask) <= len(input_ids):
                    loss_mask = torch.tensor(concat_loss_mask, dtype=torch.long)
                    # Pad if needed
                    if len(loss_mask) < len(input_ids):
                        padding = torch.zeros(len(input_ids) - len(loss_mask), dtype=loss_mask.dtype)
                        loss_mask = torch.cat([loss_mask, padding])
                else:
                    # Truncate if needed
                    loss_mask = torch.tensor(concat_loss_mask[:len(input_ids)], dtype=torch.long)
            else:
                # Fallback: create zero loss_mask
                loss_mask = torch.zeros_like(input_ids)
        else:
            # Validate and convert tokens for tokenizer-only case
            input_ids, loss_mask, attention_mask = self._validate_and_convert_tokens(
                full_tokens[0], concat_tokens, concat_loss_mask, concat_attention_mask
            )

        # Handle sequence length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
            padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        # Create position IDs
        # For Qwen2VL and Qwen2.5-VL, we may need special position ID handling
        # Both Qwen2-VL and Qwen2.5-VL use Qwen2VLImageProcessor
        if self.processor is not None and "Qwen2VLImageProcessor" in str(type(self.processor.image_processor)):
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=multi_modal_inputs.get("image_grid_thw") if multi_modal_inputs else None,
                    video_grid_thw=multi_modal_inputs.get("video_grid_thw") if multi_modal_inputs else None,
                    second_per_grid_ts=multi_modal_inputs.get("second_per_grid_ts") if multi_modal_inputs else None,
                    attention_mask=attention_mask,
                )
            ]
            position_ids = position_ids[0]  # Extract from list
        else:
            position_ids = torch.arange(len(input_ids), dtype=torch.long)
            # Zero out position IDs for padding
            position_ids = position_ids * attention_mask

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

        # Add multimodal data if available
        if multi_modal_data:
            result["multi_modal_data"] = multi_modal_data

        if multi_modal_inputs and self.return_multi_modal_inputs:
            result["multi_modal_inputs"] = multi_modal_inputs

        return result
