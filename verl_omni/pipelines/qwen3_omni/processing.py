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
"""Preserve Qwen3-Omni media boundaries when rebuilding rollout prompts."""


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
