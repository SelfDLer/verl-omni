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
"""CPU regressions for complete media spans.

Load these pure helpers directly so testing them does not require vLLM/GPU
initialization through the package's registration imports.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load(name):
    path = Path(__file__).resolve().parents[2] / "verl_omni" / "pipelines" / "qwen3_omni" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"qwen3_omni_{name}_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


processing = _load("processing")


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
