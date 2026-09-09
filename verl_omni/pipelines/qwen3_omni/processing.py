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
"""Preserve Qwen3-Omni media boundaries and sampled video time across backends."""

import copy
import functools
import math

import numpy as np


def collapse_multimodal_tokens(processor, prompt_ids: list[int]) -> list[int]:
    """Restore one placeholder per media item before vLLM expands the prompt.

    Audio embedded in video is one joint item. Collapsing its individual pad
    runs leaves audio boundaries and extra pads behind after vLLM replacement.
    """
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return prompt_ids

    def token_id(attr):
        token = getattr(processor, attr, None)
        if token is None:
            return None
        value = tokenizer.convert_tokens_to_ids(token)
        return None if value is None or value == getattr(tokenizer, "unk_token_id", None) else int(value)

    image, video, audio = (token_id(attr) for attr in ("image_token", "video_token", "audio_token"))
    vision_start, vision_end = (token_id(attr) for attr in ("vision_bos_token", "vision_eos_token"))
    audio_start, audio_end = (token_id(attr) for attr in ("audio_bos_token", "audio_eos_token"))
    pads = {value for value in (image, video, audio) if value is not None}
    paired_tokens_known = all(
        value is not None for value in (video, audio, vision_start, vision_end, audio_start, audio_end)
    )
    result = []
    index = 0
    while index < len(prompt_ids):
        if paired_tokens_known and prompt_ids[index : index + 2] == [vision_start, audio_start]:
            end = index + 2
            while end < len(prompt_ids) and prompt_ids[end] in (video, audio):
                end += 1
            if video not in prompt_ids[index + 2 : end] or prompt_ids[end : end + 2] != [audio_end, vision_end]:
                raise ValueError("Malformed Qwen3-Omni audio/video segment; cannot restore a video placeholder.")
            result.extend([vision_start, video, vision_end])
            index = end + 2
        else:
            token = prompt_ids[index]
            if token not in pads or not result or result[-1] != token:
                result.append(token)
            index += 1
    return result


def video_seconds_per_grid(metadata, temporal_patch_size):
    """Use the sampled frame count and source duration, including frame caps.

    Accept both qwen-omni-utils dictionaries and HF VideoMetadata objects.
    The reader's total_num_frames describes the selected clip, not necessarily
    the entire source file. frames_indices may be a tensor, so avoid truth tests.
    """
    result = []
    for item in metadata:
        get = item.get if isinstance(item, dict) else lambda key, default=None, item=item: getattr(item, key, default)
        indices = get("frames_indices")
        duration = get("duration")
        if duration is None:
            total_frames, fps = get("total_num_frames"), get("fps")
            if total_frames is not None and fps is not None and fps > 0:
                duration = float(total_frames) / float(fps)
        if indices is None or len(indices) == 0 or duration is None or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Presampled Qwen3-Omni video requires frame indices and a positive source duration.")
        result.append(float(temporal_patch_size) * float(duration) / len(indices))
    return result


def install_video_timing_fix(processor_cls):
    """Make the HF processor use per-video timing for presampled tensors.

    HF 5.10--5.14 computes all video times from one scalar FPS. Keep its media
    preprocessing/tokenization, but supply actual times to prompt expansion
    and to the returned RoPE metadata. A shallow per-call copy keeps concurrent
    requests independent, including requests with different FPS per video.
    """
    original_call = processor_cls.__call__
    if getattr(original_call, "_verl_omni_video_timing", False):
        return

    @functools.wraps(original_call)
    def call(self, text=None, images=None, videos=None, audio=None, **kwargs):
        video_kwargs = dict(kwargs.get("videos_kwargs") or {})
        metadata = video_kwargs.get("video_metadata", kwargs.get("video_metadata"))
        if videos is None or metadata is None:
            return original_call(self, text=text, images=images, videos=videos, audio=audio, **kwargs)
        # Both backends return float32 processor tensors. Use those same values
        # during prompt expansion, before either backend materializes tensors.
        seconds = [
            float(np.float32(value))
            for value in video_seconds_per_grid(metadata, self.video_processor.temporal_patch_size)
        ]
        # Already sampled by qwen-omni-utils: FPS is used only for temporal
        # metadata. Avoid conflicting top-level and nested processor kwargs.
        kwargs.pop("do_sample_frames", None)
        kwargs.pop("fps", None)
        video_kwargs["do_sample_frames"] = False
        video_kwargs["fps"] = 1.0
        kwargs["videos_kwargs"] = video_kwargs

        current = copy.copy(self)
        replace_tokens = self.replace_multimodal_special_tokens

        @functools.wraps(replace_tokens)
        def replace(
            text, audio_lengths, image_grid_thw, video_grid_thw, video_second_per_grid, *args, **replace_kwargs
        ):
            return replace_tokens(
                text, audio_lengths, image_grid_thw, video_grid_thw, iter(seconds), *args, **replace_kwargs
            )

        current.replace_multimodal_special_tokens = replace
        result = original_call(current, text=text, images=images, videos=videos, audio=audio, **kwargs)
        if len(result["video_grid_thw"]) != len(seconds):
            raise ValueError("Video metadata must contain one entry per processed video.")
        # Preserve the processor's requested tensor representation.
        previous = result["video_second_per_grid"]
        if hasattr(previous, "new_tensor"):
            result["video_second_per_grid"] = previous.new_tensor(seconds)
        elif isinstance(previous, np.ndarray):
            result["video_second_per_grid"] = np.asarray(seconds, dtype=previous.dtype)
        else:
            result["video_second_per_grid"] = seconds
        return result

    call._verl_omni_video_timing = True
    processor_cls.__call__ = call
