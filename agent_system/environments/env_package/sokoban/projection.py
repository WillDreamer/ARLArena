# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import random
from typing import List
import re


def sokoban_projection(actions: List[str]):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        <think>some reasoning...</think><action>up/down/left/right/still</action>
    Sokoban action mappings:
    - 0: Still (Invalid Action)
    - 1: Up
    - 2: Down
    - 3: Left
    - 4: Right
    """

    action_pools = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "still": 0,
    }

    format_valids = [0] * len(actions)
    valids = [0] * len(actions)

    strict_pattern = re.compile(r'^\s*<think>(.*?)</think>\s*<action>(.*?)</action>\s*$', flags=re.IGNORECASE | re.DOTALL)
    action_first_re = re.compile(r'<\s*action\s*>(.*?)</\s*action\s*>', flags=re.IGNORECASE | re.DOTALL)    

    def count_tag(s: str, tag: str):
        return len(re.findall(fr'<\s*{re.escape(tag)}\s*>', s, flags=re.IGNORECASE))

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()
        # stay still if invalid
        default_invalid_store = 0

        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', original_str))

        think_open_count = count_tag(original_str, 'think')
        action_open_count = count_tag(original_str, 'action')
        think_close_count = len(re.findall(r'</\s*think\s*>', original_str, flags=re.IGNORECASE))
        action_close_count = len(re.findall(r'</\s*action\s*>', original_str, flags=re.IGNORECASE))
        m = strict_pattern.match(original_str)

        strict_ok = bool(m) and not has_chinese and (
            think_open_count == think_close_count == 1 and action_open_count == action_close_count == 1
        )

        if strict_ok:
            # perfectly formatted
            extracted_action = m.group(2).strip().lower()
            format_valids[i] = 1
            for act in action_pools.keys():
                if act in extracted_action:
                    actions[i] = action_pools[act]
                    valids[i] = 1
                else:
                    actions[i] = default_invalid_store
                    valids[i] = 0
            continue

        m2 = action_first_re.search(original_str)
        if m2:
            extracted_action = m2.group(1).strip().lower()
            format_valids[i] = 0  # penalty: format not strictly valid
            valids[i] = 0        # penalty: considered invalid even if extracted
            for act in action_pools.keys():
                if act in extracted_action:
                    actions[i] = action_pools[act]
                else:
                    actions[i] = default_invalid_store
            continue

        actions[i] = default_invalid_store
        format_valids[i] = 0
        valids[i] = 0

    return actions, valids, format_valids