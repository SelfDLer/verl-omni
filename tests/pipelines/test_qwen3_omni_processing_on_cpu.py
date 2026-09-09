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
"""CPU regressions for complete media spans and per-video sampling time.

Load these pure helpers directly so testing them does not require vLLM/GPU
initialization through the package's registration imports.
"""

import importlib.util
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _load(name):
    path = Path(__file__).resolve().parents[2] / "verl_omni" / "pipelines" / "qwen3_omni" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"qwen3_omni_{name}_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


processing = _load("processing")
rope = _load("rope")


@pytest.fixture
def processor():
    attrs = {
        "image_token": "I",
        "video_token": "V",
        "audio_token": "A",
        "vision_bos_token": "VS",
        "vision_eos_token": "VE",
        "audio_bos_token": "AS",
        "audio_eos_token": "AE",
    }
    ids = dict(zip(attrs.values(), (101, 102, 103, 104, 107, 105, 106), strict=True))
    return SimpleNamespace(**attrs, tokenizer=SimpleNamespace(unk_token_id=-1, convert_tokens_to_ids=ids.get))


@pytest.mark.parametrize("video_runs", [1, 2, 16])
def test_audio_video_round_trip_through_one_video_replacement(processor, video_runs):
    # vLLM replaces one video pad per item with this *entire* joint segment.
    expanded = [105] + ([102] * 32 + [103] * 26) * video_runs + [106]
    original = [7, 104, *expanded, 107, 8]
    collapsed = processing.collapse_multimodal_tokens(processor, original)
    assert collapsed == [7, 104, 102, 107, 8]
    index = collapsed.index(102)
    assert collapsed[:index] + expanded + collapsed[index + 1 :] == original
    assert processing.collapse_multimodal_tokens(processor, collapsed) == collapsed


def test_multiple_media_spans_preserve_text_and_separate_audio(processor):
    original = [
        7,
        7,
        104,
        101,
        101,
        107,
        105,
        103,
        103,
        106,
        104,
        105,
        102,
        102,
        103,
        102,
        103,
        106,
        107,
        8,
        104,
        102,
        102,
        107,
        9,
        104,
        105,
        102,
        103,
        106,
        107,
    ]
    assert processing.collapse_multimodal_tokens(processor, original) == [
        7,
        7,
        104,
        101,
        107,
        105,
        103,
        106,
        104,
        102,
        107,
        8,
        104,
        102,
        107,
        9,
        104,
        102,
        107,
    ]


@pytest.mark.parametrize("tokens", [[104, 105, 102], [104, 105, 103, 106, 107], [104, 105, 102, 8, 106, 107]])
def test_incomplete_joint_segment_is_rejected(processor, tokens):
    with pytest.raises(ValueError, match="Malformed"):
        processing.collapse_multimodal_tokens(processor, tokens)


@pytest.mark.parametrize("indices", [list(range(32)), np.arange(32), torch.arange(32)])
def test_capped_video_uses_source_duration(indices):
    metadata = {"fps": 30.0, "total_num_frames": 1800, "frames_indices": indices}
    assert processing.video_seconds_per_grid([metadata], 2) == pytest.approx([3.75])
    assert processing.video_seconds_per_grid(
        [SimpleNamespace(duration=60, frames_indices=indices)], 2
    ) == pytest.approx([3.75])


@pytest.mark.parametrize(
    "metadata", [{}, {"duration": 0, "frames_indices": [0]}, {"duration": float("nan"), "frames_indices": [0]}]
)
def test_missing_sampling_information_fails_explicitly(metadata):
    with pytest.raises(ValueError, match="Presampled"):
        processing.video_seconds_per_grid([metadata], 2)


class _Processor:
    """Minimal HF processor contract; real media/model execution is unnecessary."""

    video_processor = SimpleNamespace(temporal_patch_size=2)

    def replace_multimodal_special_tokens(
        self, text, audio_lengths, image_grid_thw, video_grid_thw, video_second_per_grid, **kwargs
    ):
        return list(video_second_per_grid)

    def __call__(self, text=None, images=None, videos=None, audio=None, **kwargs):
        videos_kwargs = kwargs.get("videos_kwargs", {})
        seconds = [2.0 / videos_kwargs.get("fps", 1.0)] * len(videos)
        expanded_times = self.replace_multimodal_special_tokens(text, iter([]), iter([]), iter([]), iter(seconds))
        return {
            "video_grid_thw": torch.ones((len(videos), 3)),
            "video_second_per_grid": torch.tensor(seconds),
            "expanded_times": expanded_times,
            "do_sample_frames": videos_kwargs.get("do_sample_frames", True),
        }


def test_per_video_timing_is_used_during_expansion_and_in_returned_tensors():
    class Processor(_Processor):
        pass

    processing.install_video_timing_fix(Processor)
    patched = Processor.__call__
    processing.install_video_timing_fix(Processor)
    assert Processor.__call__ is patched
    instance = Processor()
    original_replace = instance.replace_multimodal_special_tokens
    metadata = [
        {"duration": 60, "frames_indices": list(range(32))},
        {"duration": 10, "frames_indices": list(range(10))},
    ]
    for nested in (False, True):
        kwargs = {"video_metadata": metadata}
        if nested:
            kwargs = {"videos_kwargs": {**kwargs, "fps": 7, "do_sample_frames": True}}
        result = instance(text="video video", videos=[object(), object()], **kwargs)
        assert result["expanded_times"] == pytest.approx([3.75, 2.0])
        torch.testing.assert_close(result["video_second_per_grid"], torch.tensor([3.75, 2.0]))
        assert result["do_sample_frames"] is False
    assert instance.replace_multimodal_special_tokens == original_replace
    raw = instance(text="video", videos=[object()], videos_kwargs={"fps": 2})
    assert raw["expanded_times"] == [1.0]


def test_concurrent_processor_calls_do_not_share_video_times():
    class Processor(_Processor):
        pass

    processing.install_video_timing_fix(Processor)
    processor = Processor()

    def run(duration):
        return processor(
            text="video", videos=[object()], video_metadata=[{"duration": duration, "frames_indices": [0, 1]}]
        )["expanded_times"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(run, range(1, 25)))
    assert result == [[float(value)] for value in range(1, 25)]


@pytest.mark.parametrize("use_audio", [False, True])
@pytest.mark.parametrize("metadata_form", ["dict", "hf"])
def test_real_hf_processor_expands_each_video_with_its_own_time(processor, use_audio, metadata_form):
    pytest.importorskip("transformers")
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor
    from transformers.video_utils import VideoMetadata

    class Processor(Qwen3OmniMoeProcessor):
        pass

    class Tokenizer:
        init_kwargs = {}

        def __call__(self, texts, **kwargs):
            return {"input_ids": [[int(value) for value in re.findall(r"<(\d+)>", text)] for text in texts]}

    class VideoProcessor:
        merge_size = temporal_patch_size = 2

        def __call__(self, videos, **kwargs):
            assert kwargs["do_sample_frames"] is False
            return {"video_grid_thw": np.array([[2, 4, 8]] * len(videos))}

    class AudioProcessor:
        def __call__(self, audio, **kwargs):
            return {
                "attention_mask": np.ones((len(audio), 400), dtype=np.int64),
                "input_features": np.zeros((len(audio), 128, 400), dtype=np.float32),
            }

    # Skip checkpoint/tokenizer construction; execute the actual HF __call__,
    # kwargs merging, token expansion and BatchFeature tensor conversion.
    instance = object.__new__(Processor)
    for name, value in vars(processor).items():
        if name != "tokenizer":
            setattr(instance, name, f"<{processor.tokenizer.convert_tokens_to_ids(value)}>")
    instance.tokenizer = Tokenizer()
    instance.image_processor = SimpleNamespace(merge_size=2)
    instance.video_processor = VideoProcessor()
    instance.feature_extractor = AudioProcessor()
    processing.install_video_timing_fix(Processor)
    text = "<104><102><107><104><102><107>"
    metadata = [
        {"fps": 30.0, "total_num_frames": 1800, "frames_indices": list(range(32))},
        {"fps": 30.0, "total_num_frames": 960, "frames_indices": list(range(32))},
    ]
    if metadata_form == "hf":
        metadata = [VideoMetadata(**item) for item in metadata]
    result = instance(
        text=[text],
        videos=[np.zeros((4, 3, 32, 32))] * 2,
        audio=[np.zeros(64000)] * 2 if use_audio else None,
        video_metadata=metadata,
        use_audio_in_video=use_audio,
        return_tensors="pt",
    )
    expected_text = instance.replace_multimodal_special_tokens(
        [text], iter([52, 52]), iter([]), iter(np.array([[2, 4, 8], [2, 4, 8]])), iter([3.75, 2.0]), use_audio, 13, 2.0
    )
    expected_ids = instance.tokenizer(expected_text)["input_ids"]
    torch.testing.assert_close(result["input_ids"], torch.tensor(expected_ids))
    torch.testing.assert_close(result["video_second_per_grid"], torch.tensor([3.75, 2.0]))


def test_vllm_registration_keeps_model_loading_lazy(monkeypatch):
    calls = []
    registry = SimpleNamespace(register_model=lambda *args: calls.append(args))
    monkeypatch.setitem(
        sys.modules, "vllm_omni.model_executor.models.registry", SimpleNamespace(OmniModelRegistry=registry)
    )
    rope.install_vllm_rope_fix()
    assert calls == [
        (
            "Qwen3OmniMoeThinkerForConditionalGeneration",
            "verl_omni.pipelines.qwen3_omni.vllm_thinker:Qwen3OmniMoeThinkerForConditionalGeneration",
        )
    ]


def test_lazy_entry_applies_rope_to_the_registered_upstream_class(monkeypatch):
    upstream_cls = type("Qwen3OmniMoeThinkerForConditionalGeneration", (), {})
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker",
        SimpleNamespace(Qwen3OmniMoeThinkerForConditionalGeneration=upstream_cls),
    )
    monkeypatch.setitem(sys.modules, "verl_omni.pipelines.qwen3_omni.rope", rope)
    entry = _load("vllm_thinker")
    assert entry.Qwen3OmniMoeThinkerForConditionalGeneration is upstream_cls
    assert upstream_cls.get_mrope_input_positions is rope.get_mrope_input_positions


@pytest.mark.parametrize("seconds", [1.5, 2.0, 2.3, 3.75])
@pytest.mark.parametrize("temporal_grid", [1, 2, 8])
@pytest.mark.parametrize("audio_frames", [7, 100, 1000])
def test_rollout_rope_matches_hf_for_fractional_time_and_short_audio(seconds, temporal_grid, audio_frames):
    pytest.importorskip("transformers")
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoePreTrainedModelForConditionalGeneration as HFModel,
    )
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        _get_feat_extract_output_lengths,
    )

    config = SimpleNamespace(
        image_token_id=101,
        video_token_id=102,
        audio_token_id=103,
        vision_start_token_id=104,
        audio_start_token_id=105,
        audio_end_token_id=106,
        position_id_per_seconds=13,
        audio_config=SimpleNamespace(n_window=50),
    )
    hf = SimpleNamespace(config=config, spatial_merge_size=2)
    hf.get_llm_pos_ids_for_vision = HFModel.get_llm_pos_ids_for_vision.__get__(hf)
    audio_count = int(_get_feat_extract_output_lengths(torch.tensor(audio_frames)))
    # Token types are read by HF to locate the single combined media segment.
    tokens = [7, 104, 105] + [102] * (temporal_grid * 32) + [103] * audio_count + [106, 107, 8, 9]
    grid = torch.tensor([[temporal_grid, 8, 16]])
    ids = torch.tensor([tokens])
    expected, _ = HFModel.get_rope_index(
        hf,
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        video_grid_thw=grid,
        use_audio_in_video=True,
        audio_seqlens=torch.tensor([audio_frames]),
        second_per_grids=torch.tensor([seconds]),
    )
    data = dict(
        grid_t=temporal_grid,
        grid_h=4,
        grid_w=8,
        t_factor=seconds * 13,
        use_audio_in_video=True,
        audio_feature_length=audio_frames,
        placeholder_len=temporal_grid * 32 + audio_count + 2,
    )
    model = SimpleNamespace(config=config, iter_mm_features=lambda _: iter([(2, "video", data)]))
    actual, delta = rope.get_mrope_input_positions(model, tokens, [])
    torch.testing.assert_close(actual, expected[:, 0].long())
    full_ids = torch.tensor([tokens + [10, 11, 12]])
    full_positions, _ = HFModel.get_rope_index(
        hf,
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids),
        video_grid_thw=grid,
        use_audio_in_video=True,
        audio_seqlens=torch.tensor([audio_frames]),
        second_per_grids=torch.tensor([seconds]),
    )
    decode_positions = torch.arange(len(tokens), len(tokens) + 3) + delta
    torch.testing.assert_close(decode_positions.expand(3, -1), full_positions[:, 0, -3:].long())
