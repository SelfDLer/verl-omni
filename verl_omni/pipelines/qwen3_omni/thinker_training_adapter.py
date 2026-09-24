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
"""Qwen3-Omni Thinker training adapter.

Implements ``OmniModelBase`` for thinker-stage training of
Qwen3-Omni: sub-module stripping, forward redirection,
processor/tokenizer configuration, and native VeOmni model/input adaptation.
"""

import inspect
import json
import logging
import os
import sys
from types import MethodType
from typing import Any

import numpy as np
import torch

from verl_omni.pipelines.model_base import OmniModelBase

logger = logging.getLogger(__name__)


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


@OmniModelBase.register("Qwen3OmniMoeForConditionalGeneration", stage="thinker")
class Qwen3OmniThinkerAdapter(OmniModelBase):
    """Thinker-stage training adapter for Qwen3-Omni.

    Provides separate setup hooks for HF/FSDP and native VeOmni models,
    alongside shared processor/tokenizer configuration. Native VeOmni hooks
    preserve the backend's forward wrapper and sharding hints.
    """

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        return ["talker", "code2wav", "code_predictor"]

    @classmethod
    def configure_model(cls, module, model_config):
        """Strip non-training stages and redirect forward to thinker.

        Args:
            module: The loaded Qwen3-Omni model before FSDP wrapping.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured module with talker/codec stripped and
            forward/embedding accessors redirected to thinker.
        """
        module = super().configure_model(module, model_config)
        module.forward = module.thinker.forward
        module.get_input_embeddings = module.thinker.get_input_embeddings
        module.set_input_embeddings = module.thinker.set_input_embeddings
        module._no_split_modules = ["Qwen3OmniMoeThinkerTextDecoderLayer"]
        return module

    @classmethod
    def configure_veomni_model(cls, module, model_config):
        """Bridge pinned VeOmni APIs without replacing native forward or split hints."""
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

    @classmethod
    def prepare_veomni_inputs(cls, model_inputs: dict, full_input_ids: torch.Tensor, hf_config) -> dict:
        """Add global modality masks and flatten the HF processor's valid audio frames."""
        thinker_config = hf_config.thinker_config
        for modality in ("image", "video", "audio"):
            model_inputs[f"{modality}_mask"] = full_input_ids == getattr(thinker_config, f"{modality}_token_id")

        features = model_inputs.get("input_features")
        feature_mask = model_inputs.pop("feature_attention_mask", None)
        if features is not None:
            if features.ndim != 3 or feature_mask is None:
                raise ValueError("VeOmni expects HF input_features (clips, mel, frames) and feature_attention_mask.")
            if feature_mask.shape != (features.shape[0], features.shape[2]):
                raise ValueError("feature_attention_mask must match the clip and frame dimensions of input_features.")
            feature_mask = feature_mask.to(device=features.device, dtype=torch.bool)
            model_inputs["audio_feature_lengths"] = feature_mask.sum(-1).to(torch.long)
            model_inputs["input_features"] = features.transpose(1, 2)[feature_mask].contiguous()
        elif feature_mask is not None:
            raise ValueError("feature_attention_mask was provided without input_features.")
        return model_inputs

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """Load the Qwen3-Omni multimodal processor with RoPE + dedup helpers.

        Swaps ``processor.config`` to ``thinker_config`` (Qwen3-Omni nests
        multimodal settings under sub-configs). Binds ``get_rope_index`` and
        ``get_llm_pos_ids_for_vision`` (model methods the omni agent loop
        calls on the processor), and ``dedup_pad_tokens`` (collapses
        consecutive multimodal pad tokens before vLLM-Omni re-expands them).

        Args:
            model_path: Local path to the model checkpoint.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured processor with RoPE and dedup helpers bound.
        """
        import types

        from transformers import AutoConfig
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

        from verl_omni.pipelines.qwen3_omni.video_processor import Qwen3OmniVideoProcessor

        processor = Qwen3OmniVideoProcessor.from_pretrained(
            model_path, trust_remote_code=model_config.trust_remote_code
        )
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)

        processor.config = config.thinker_config
        processor.spatial_merge_size = config.thinker_config.vision_config.spatial_merge_size
        processor.config.vision_start_token_id = config.talker_config.vision_start_token_id

        model_cls = Qwen3OmniMoeThinkerForConditionalGeneration

        # Cast to int64: HF returns float32, FSDP would otherwise bf16-round positions.
        def _get_rope_index_long(self, *args, **kwargs):
            vision_position_ids, deltas = model_cls.get_rope_index(self, *args, **kwargs)
            return vision_position_ids.long(), deltas

        processor.get_rope_index = types.MethodType(_get_rope_index_long, processor)
        processor.get_llm_pos_ids_for_vision = types.MethodType(model_cls.get_llm_pos_ids_for_vision, processor)

        # Provide audio lengths to verl's generic V1 agent loop via get_rope_index_kwargs.
        def _get_rope_index_kwargs(multi_modal_inputs: dict) -> dict:
            result = {}
            seconds = multi_modal_inputs.get("video_second_per_grid")
            if seconds is not None:
                result["second_per_grids"] = seconds
            feature_attention_mask = multi_modal_inputs.get("feature_attention_mask")
            if feature_attention_mask is not None:
                result["audio_seqlens"] = feature_attention_mask.sum(-1)
            return result

        processor.get_rope_index_kwargs = _get_rope_index_kwargs

        # Collapse consecutive multimodal pad tokens before vLLM-Omni re-expands
        # them (token-IDs path still unfixed: https://github.com/vllm-project/vllm/issues/33672);
        # mirrors verl's qwen2_5_vl_dedup_image_tokens.
        def _dedup_pad_tokens(self, prompt_ids: list[int]) -> list[int]:
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None:
                return prompt_ids
            pad_ids: set[int] = set()
            for tok_attr in ("image_token", "video_token", "audio_token"):
                tok = getattr(self, tok_attr, None)
                if tok is None:
                    continue
                try:
                    tid = tokenizer.convert_tokens_to_ids(tok)
                except Exception:
                    continue
                if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
                    continue
                pad_ids.add(int(tid))
            if not pad_ids:
                return prompt_ids
            arr = np.asarray(prompt_ids, dtype=np.int64)
            if arr.size == 0:
                return prompt_ids
            is_pad = np.isin(arr, list(pad_ids))
            keep = np.ones(arr.size, dtype=bool)
            same_as_prev = is_pad[1:] & is_pad[:-1] & (arr[1:] == arr[:-1])
            keep[1:] &= ~same_as_prev
            return arr[keep].tolist()

        processor.dedup_pad_tokens = types.MethodType(_dedup_pad_tokens, processor)
        return processor

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        """Load the tokenizer with chat template from ``chat_template.json``.

        Args:
            model_path: Local path to the model checkpoint.
            model_config: The ``OmniModelConfig``.

        Returns:
            The configured tokenizer with ``chat_template`` loaded from
            ``chat_template.json``.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
        chat_template_path = os.path.join(model_path, "chat_template.json")
        if not os.path.isfile(chat_template_path):
            raise FileNotFoundError(
                f"Qwen3-Omni chat template not found at {chat_template_path}. "
                f"Ensure the model checkpoint includes chat_template.json."
            )
        with open(chat_template_path) as f:
            tokenizer.chat_template = json.load(f)["chat_template"]
        return tokenizer
