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

"""Exercise media boundaries with mocked decoders, without a rollout engine."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dataset_module(monkeypatch):
    # Only the external base class is isolated. Load the real shared padding
    # implementation and the complete new dataset module from disk.
    base = ModuleType("verl.utils.dataset.rl_dataset")
    base.RLHFDataset = type("RLHFDataset", (), {})
    monkeypatch.setitem(sys.modules, "verl.utils.dataset.rl_dataset", base)
    shared = load_file("omni_dataset_test", "verl_omni/utils/dataset/omni_rl_datasets.py")
    monkeypatch.setitem(sys.modules, "verl_omni.utils.dataset.omni_rl_datasets", shared)
    return load_file("nextqa_dataset_test", "verl_omni/utils/dataset/nextqa_rl_dataset.py")


def test_soundtrack_is_explicit_and_input_is_not_modified(dataset_module):
    video = {"type": "video", "video": "/clip.mp4", "video_start": 2, "video_end": 5, "fps": 1}
    messages = [{"role": "user", "content": [video, {"type": "text", "text": "Why?"}]}]
    result = dataset_module.with_video_soundtracks(messages)
    assert result[0]["content"][1] == {"type": "audio", "audio": "/clip.mp4", "audio_start": 2, "audio_end": 5}
    assert [item["type"] for item in result[0]["content"]] == ["video", "audio", "text"]
    assert len(messages[0]["content"]) == 2
    assert result[0]["content"][0] is not video


def test_decode_preserves_metadata_and_pads_audio(dataset_module, monkeypatch):
    frames = object()
    audio = np.ones(801, dtype=np.float32)
    metadata = {"fps": 30, "total_num_frames": 1800, "frames_indices": np.arange(32)}
    calls = []

    def decode(messages, **kwargs):
        calls.append(kwargs)
        return [audio], None, [(frames, metadata)]

    monkeypatch.setitem(sys.modules, "qwen_omni_utils", SimpleNamespace(process_mm_info=decode))
    images, videos, audios = dataset_module.NextQARLHFDataset._process_multi_modal_info([], 16, {})
    assert images is None
    assert videos[0][0] is frames
    assert videos[0][1]["frames_indices"] == list(range(32))
    assert videos[0][1]["total_num_frames"] == 1800
    assert videos[0][1]["duration"] == 60
    assert audios[0].shape == (960,)
    np.testing.assert_array_equal(audios[0][:801], audio)
    np.testing.assert_array_equal(audios[0][801:], 0)
    assert calls == [{"use_audio_in_video": False, "image_patch_size": 16, "return_video_metadata": True}]


@pytest.mark.parametrize("options", [{"use_audio_in_video": True}, {"sampling_rate": 48000}])
def test_incompatible_audio_configuration_fails_before_decode(dataset_module, monkeypatch, options):
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", SimpleNamespace(process_mm_info=None))
    with pytest.raises(ValueError):
        dataset_module.NextQARLHFDataset._process_multi_modal_info([], 14, {"mm_processor_kwargs": options})


@pytest.fixture
def processor_module():
    return load_file("nextqa_processor_test", "verl_omni/pipelines/qwen3_omni/video_processor.py")


def test_processor_passes_actual_sampled_fps_without_resampling(processor_module, monkeypatch):
    calls = []
    monkeypatch.setattr(processor_module.Qwen3OmniMoeProcessor, "__call__", lambda self, **kw: calls.append(kw))
    processor = object.__new__(processor_module.Qwen3OmniVideoProcessor)
    meta = [{"fps": 30, "total_num_frames": 1800, "frames_indices": list(range(32))}]
    original = {"fps": 1, "do_sample_frames": True}
    processor(videos=[object()], video_metadata=meta, videos_kwargs=original, fps=99)
    assert calls[0]["videos_kwargs"] == {"fps": 32 / 60, "do_sample_frames": False}
    assert "fps" not in calls[0]
    assert original == {"fps": 1, "do_sample_frames": True}


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_processor_rejects_invalid_video_clock(processor_module, duration):
    processor = object.__new__(processor_module.Qwen3OmniVideoProcessor)
    with pytest.raises(ValueError, match="duration"):
        processor(videos=[object()], video_metadata=[{"duration": duration, "frames_indices": [0, 1]}])


def test_processor_rejects_mixed_clocks(processor_module):
    processor = object.__new__(processor_module.Qwen3OmniVideoProcessor)
    with pytest.raises(ValueError, match="one sampled FPS"):
        processor(
            videos=[object(), object()],
            video_metadata=[{"duration": 2, "frames_indices": [0, 1]}, {"duration": 4, "frames_indices": [0, 1]}],
        )


def test_real_hf_processor_returns_sampled_video_time(processor_module):
    import torch

    class Tokenizer:
        init_kwargs = {}

        def __call__(self, text, **kwargs):
            return {"input_ids": [[1, 2]]}

    class VideoProcessor:
        temporal_patch_size = 2

        def __call__(self, videos, **kwargs):
            assert kwargs["do_sample_frames"] is False
            return {"video_grid_thw": [[16, 2, 2]]}

    processor = object.__new__(processor_module.Qwen3OmniVideoProcessor)
    processor.tokenizer = Tokenizer()
    processor.video_processor = VideoProcessor()
    processor.replace_multimodal_special_tokens = lambda text, *args, **kwargs: text
    result = processor(
        text="video",
        videos=[torch.zeros(32, 3, 28, 28)],
        video_metadata=[{"fps": 30, "total_num_frames": 1800, "frames_indices": list(range(32))}],
        return_tensors="pt",
    )
    torch.testing.assert_close(result["video_second_per_grid"], torch.tensor([3.75]))
