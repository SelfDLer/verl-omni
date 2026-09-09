# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Lazy vLLM model entry point retaining the upstream model/processor registry.

Use the original class so weight loading, multimodal registration and model
capability checks remain unchanged. Only prompt position calculation differs.
Remove this compatibility entry when the pinned vLLM-Omni implements HF parity.
"""

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

from verl_omni.pipelines.qwen3_omni.rope import get_mrope_input_positions

Qwen3OmniMoeThinkerForConditionalGeneration.get_mrope_input_positions = get_mrope_input_positions

__all__ = ["Qwen3OmniMoeThinkerForConditionalGeneration"]
