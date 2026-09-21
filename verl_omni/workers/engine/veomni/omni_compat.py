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

"""Compatibility for pinned VeOmni Qwen3-Omni on the repository's Transformers."""

import inspect
import sys
from types import MethodType

import torch


def _restore_verl_position_layout(module, args, kwargs):
    positions = kwargs.get("position_ids")
    # verl supplies (4, batch, tokens). VeOmni 0.1.11 mistakes batch=3 for
    # its native (batch, 3, tokens) layout and transposes it in Thinker.forward.
    if positions is not None and positions.ndim == 3 and positions.shape[:2] == (3, 4):
        kwargs["position_ids"] = positions.transpose(0, 1).contiguous()
    return args, kwargs


def _audio_attention_forward(module, hidden_states, cu_seqlens, attention_mask=None, **kwargs):
    # SDPA/eager ignore FA's cu_seqlens. Run each audio window independently
    # so adjacent clips (and windows within a clip) cannot attend to each other.
    boundaries = cu_seqlens.tolist()
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = None if attention_mask is None else attention_mask[..., start:end, start:end]
        outputs.append(
            module._veomni_window_forward(
                hidden_states[start:end],
                cu_seqlens.new_tensor([0, end - start]),
                attention_mask=mask,
                **kwargs,
            )
        )
    return torch.cat(outputs, dim=0)


def configure_omni_compat(module):
    """Bridge the pinned generated model's mask API and verl's four-axis RoPE."""
    if getattr(module, "_omni_compat_configured", False):
        return module
    modeling = sys.modules[type(module.thinker).__module__]
    create_mask = modeling.create_causal_mask
    if "cache_position" not in inspect.signature(create_mask).parameters:
        # Transformers 5.13+ derives query positions from the cache/input shape.
        # Patch only VeOmni's generated module, not Transformers' global API.
        def create_causal_mask(*args, cache_position=None, **kwargs):
            return create_mask(*args, **kwargs)

        modeling.create_causal_mask = create_causal_mask
    module.thinker.model.register_forward_pre_hook(_restore_verl_position_layout, with_kwargs=True)
    for layer in module.thinker.audio_tower.layers:
        attention = layer.self_attn
        if attention.config._attn_implementation in ("sdpa", "eager"):
            attention._veomni_window_forward = attention.forward
            attention.forward = MethodType(_audio_attention_forward, attention)
    module._omni_compat_configured = True
    return module
