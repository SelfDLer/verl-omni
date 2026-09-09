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
"""HF-compatible Qwen3-Omni positions for vLLM's multimodal feature interface.

The pinned vLLM-Omni implementation rounds timestamps before AV interleaving
and advances from the whole block's maximum. HF orders floating timestamps
and advances from its final appended segment. Keep both conventions identical
to the actor; cast only the completed positions to int64, as the actor does.
"""

import numpy as np
import torch


def _vision_positions(start, data, position_id_per_seconds):
    t, h, w = data["grid_t"], data["grid_h"], data["grid_w"]
    positions = np.indices((t, h, w), dtype=np.float32)
    # Match HF's float32 multiplication order, rather than rounding a product
    # computed in float64. This matters at timestamp ties with audio tokens.
    seconds = np.float32(data["t_factor"] / position_id_per_seconds)
    times = np.arange(t, dtype=np.float32) * seconds * np.float32(position_id_per_seconds)
    positions[0] = times[:, None, None]
    return positions.reshape(3, -1) + start


def _interleave(video, audio):
    pieces = []
    vi = ai = 0
    while vi < video.shape[1] and ai < audio.shape[1]:
        if video[0, vi] <= audio[0, ai]:
            pieces.append(video[:, vi : vi + 1])
            vi += 1
        else:
            pieces.append(audio[:, ai : ai + 1])
            ai += 1
    if vi < video.shape[1]:
        pieces.append(video[:, vi:])
    if ai < audio.shape[1]:
        pieces.append(audio[:, ai:])
    return np.concatenate(pieces, axis=1), pieces[-1].max() + np.float32(1)


def get_mrope_input_positions(self, input_tokens, mm_features, **kwargs):
    """Compute positions from vLLM feature offsets using the actor's semantics."""
    seq_len = len(input_tokens)
    parts = []
    cursor = 0
    next_position = np.float32(0)

    def text_positions(count, start):
        return np.broadcast_to(np.arange(count, dtype=np.float32), (3, count)) + start

    for offset, modality, data in self.iter_mm_features(mm_features):
        if offset < cursor:
            raise ValueError("Overlapping Qwen3-Omni multimodal placeholders.")
        text_len = offset - cursor
        if text_len:
            parts.append(text_positions(text_len, next_position))
            next_position += np.float32(text_len)

        if modality == "audio":
            count = self._compute_audio_token_count(data["audio_feature_length"])
            positions = text_positions(count, next_position)
            parts.append(positions)
            next_position = positions.max() + np.float32(1)
            cursor = offset + int(data.get("placeholder_len", count))
        elif modality in ("image", "video"):
            paired_audio = modality == "video" and data["use_audio_in_video"]
            if paired_audio:
                parts.append(text_positions(1, next_position))  # audio BOS
                next_position += np.float32(1)
            video = _vision_positions(next_position, data, self.config.position_id_per_seconds)
            count = video.shape[1]
            if paired_audio:
                placeholder_len = data.get("placeholder_len")
                audio_count = (
                    self._compute_audio_token_count(data["audio_feature_length"])
                    if placeholder_len is None
                    else int(placeholder_len) - count - 2
                )
                if audio_count <= 0:
                    raise ValueError("Qwen3-Omni audio/video placeholder must contain audio tokens.")
                audio = text_positions(audio_count, next_position)
                positions, next_position = _interleave(video, audio)
                parts.append(positions)
                parts.append(text_positions(1, next_position))  # audio EOS
                next_position += np.float32(1)
                count += audio_count + 2
            else:
                parts.append(video)
                next_position = video.max() + np.float32(1)
            cursor = offset + int(data.get("placeholder_len", count))
        else:
            raise ValueError(f"Unsupported Qwen3-Omni modality: {modality}")

    if cursor < seq_len:
        parts.append(text_positions(seq_len - cursor, next_position))
    if not parts:
        return torch.empty((3, 0), dtype=torch.int64), 0
    positions = np.concatenate(parts, axis=1).astype(np.int64)
    if positions.shape != (3, seq_len):
        raise ValueError("Qwen3-Omni position IDs must align with every prompt token.")
    # Decode must continue the final segment just as actor teacher forcing
    # does. For very short AV clips an earlier spatial coordinate can exceed
    # the final text position; using the global maximum would shift responses.
    return torch.from_numpy(positions), int(positions[:, -1].max()) + 1 - seq_len


def install_vllm_rope_fix():
    """Keep vLLM's lazy model loading in every process, including workers."""
    from vllm_omni.model_executor.models.registry import OmniModelRegistry

    OmniModelRegistry.register_model(
        "Qwen3OmniMoeThinkerForConditionalGeneration",
        "verl_omni.pipelines.qwen3_omni.vllm_thinker:Qwen3OmniMoeThinkerForConditionalGeneration",
    )
