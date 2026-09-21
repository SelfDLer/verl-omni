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

"""Translate HF processor outputs to VeOmni's Qwen3-Omni training inputs."""

import torch


def prepare_omni_inputs(model_inputs: dict, full_input_ids: torch.Tensor, thinker_config) -> dict:
    """Add full-sequence modality masks and flatten valid HF audio frames.

    HF audio has shape (clips, mel, padded_frames); VeOmni consumes
    (valid_frames, mel), with one audio_feature_lengths entry per clip.
    Masks remain global when sequence parallelism shards the embeddings.
    """
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


def packed_attention_kwargs(offsets: torch.Tensor, pad_size: int) -> dict:
    """Derive sample boundaries from token offsets, never from multimodal RoPE."""
    offsets = offsets.to(dtype=torch.int32)
    if pad_size:
        offsets = torch.cat((offsets, offsets[-1:] + pad_size))
    max_length = int(offsets.diff().max().item())
    return {
        "cu_seq_lens_q": offsets,
        "cu_seq_lens_k": offsets,
        "max_length_q": max_length,
        "max_length_k": max_length,
    }
