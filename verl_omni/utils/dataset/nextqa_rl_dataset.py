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

"""NExT-QA video and soundtrack inputs for the synchronous omni trainer."""

from copy import deepcopy

from verl_omni.utils.dataset.omni_rl_datasets import QwenOmniRLHFDataset, pad_audio_to_hop_multiple


def with_video_soundtracks(messages):
    """Insert a separate audio item after each video without changing parquet.

    Explicit video + audio items use the model's ordinary media placeholders.
    They deliberately do not request joint audio/video token interleaving.
    """
    messages = deepcopy(messages)
    for message in messages:
        if not isinstance(message["content"], list):
            continue
        content = []
        for item in message["content"]:
            content.append(item)
            if item.get("type") == "video":
                audio = {"type": "audio", "audio": item["video"]}
                for source, target in (("video_start", "audio_start"), ("video_end", "audio_end")):
                    if source in item:
                        audio[target] = item[source]
                content.append(audio)
        message["content"] = content
    return messages


class NextQARLHFDataset(QwenOmniRLHFDataset):
    """Keep the existing converter's schema; expand its soundtrack at load time."""

    def _build_messages(self, example, key):
        return with_video_soundtracks(super()._build_messages(deepcopy(example), key))

    @classmethod
    def _process_multi_modal_info(cls, messages, image_patch_size, config):
        from qwen_omni_utils import process_mm_info

        options = (config or {}).get("mm_processor_kwargs", {})
        if options.get("use_audio_in_video", False) is not False:
            raise ValueError("NextQARLHFDataset uses explicit audio items; set use_audio_in_video=false.")
        if options.get("sampling_rate", 16000) != 16000:
            raise ValueError("qwen-omni-utils decodes soundtracks at 16000 Hz; set sampling_rate=16000.")
        audios, images, videos = process_mm_info(
            messages,
            use_audio_in_video=False,
            image_patch_size=image_patch_size,
            return_video_metadata=True,
        )
        if videos is not None:
            # vLLM's VideoMetadata uses Python sequences in truth tests.
            normalized = []
            for frames, meta in videos:
                meta = {**meta, "frames_indices": list(map(int, meta["frames_indices"]))}
                # qwen-omni-utils omits duration, whereas the pinned vLLM-Omni
                # uses duration (not source FPS) to recover the sampled clock.
                if meta.get("duration") is None:
                    meta["duration"] = meta["total_num_frames"] / meta["fps"]
                normalized.append((frames, meta))
            videos = normalized
        if audios is not None:
            audios = [pad_audio_to_hop_multiple(audio) for audio in audios]
        return images, videos, audios
