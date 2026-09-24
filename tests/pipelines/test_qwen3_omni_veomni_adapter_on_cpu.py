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

"""VeOmni-specific adapter hooks must preserve the native and HF model contracts."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.model_base import OmniModelBase
from verl_omni.pipelines.qwen3_omni.thinker_training_adapter import Qwen3OmniThinkerAdapter


@pytest.fixture
def hf_config():
    return SimpleNamespace(thinker_config=SimpleNamespace(image_token_id=10, video_token_id=11, audio_token_id=12))


def test_audio_frames_and_global_masks(hf_config):
    features = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    inputs = {"input_features": features, "feature_attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])}
    full_ids = torch.tensor([[10, 11, 12, 12, -1]])
    result = Qwen3OmniThinkerAdapter.prepare_veomni_inputs(inputs, full_ids, hf_config)
    expected = torch.cat((features[0, :, :2].T, features[1, :, :3].T))
    torch.testing.assert_close(result["input_features"], expected)
    assert result["audio_feature_lengths"].tolist() == [2, 3]
    assert result["audio_mask"].tolist() == [[False, False, True, True, False]]
    assert result["image_mask"].sum() == 1
    assert result["video_mask"].sum() == 1
    assert "feature_attention_mask" not in result


@pytest.mark.parametrize(
    "payload",
    [
        {"input_features": torch.zeros(2, 3, 4)},
        {"input_features": torch.zeros(2, 3, 4), "feature_attention_mask": torch.ones(2, 5)},
        {"feature_attention_mask": torch.ones(2, 4)},
    ],
)
def test_invalid_audio_rejected(hf_config, payload):
    with pytest.raises(ValueError):
        Qwen3OmniThinkerAdapter.prepare_veomni_inputs(payload, torch.tensor([[1]]), hf_config)


def test_text_only_and_empty_audio(hf_config):
    result = Qwen3OmniThinkerAdapter.prepare_veomni_inputs({}, torch.tensor([[1, 2]]), hf_config)
    assert not any(result[key].any() for key in ("audio_mask", "image_mask", "video_mask"))
    inputs = {"input_features": torch.zeros(0, 3, 4), "feature_attention_mask": torch.zeros(0, 4)}
    result = Qwen3OmniThinkerAdapter.prepare_veomni_inputs(inputs, torch.tensor([[1]]), hf_config)
    assert result["input_features"].shape == (0, 3)


def test_unsupported_adapter_fails_closed():
    # Existing adapters need not implement the new backend hooks to remain
    # usable with FSDP, but must not silently accept native VeOmni models.
    with pytest.raises(NotImplementedError, match="native VeOmni model"):
        OmniModelBase.configure_veomni_model(None, None)
    with pytest.raises(NotImplementedError, match="native VeOmni inputs"):
        OmniModelBase.prepare_veomni_inputs({}, torch.tensor([[1]]), None)


@pytest.fixture
def model_and_modeling(monkeypatch):
    modeling = ModuleType("veomni_adapter_fixture")
    modeling.create_causal_mask = lambda value: value
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)
    thinker_class = type("Thinker", (torch.nn.Module,), {"__module__": modeling.__name__})
    model = torch.nn.Module()
    model.thinker = thinker_class()
    model.thinker.model = torch.nn.Linear(2, 2)
    model.thinker.get_input_embeddings = lambda: model.thinker.model
    model.thinker.set_input_embeddings = lambda value: None
    model.thinker.audio_tower = torch.nn.Module()
    attention = torch.nn.Linear(2, 2)
    attention.config = SimpleNamespace(_attn_implementation="sdpa")
    layer = torch.nn.Module()
    layer.self_attn = attention
    model.thinker.audio_tower.layers = torch.nn.ModuleList([layer])
    model.talker = torch.nn.Linear(2, 2)
    model._no_split_modules = ["TextDecoder", "AudioEncoder", "VisionEncoder"]
    return model, modeling


def test_native_configure_is_idempotent_and_preserves_forward_and_split_hints(model_and_modeling):
    model, modeling = model_and_modeling
    forward, split_modules = model.forward, model._no_split_modules
    attention = model.thinker.audio_tower.layers[0].self_attn
    audio_forward = attention.forward
    Qwen3OmniThinkerAdapter.configure_veomni_model(model, None)
    patched_mask, patched_audio = modeling.create_causal_mask, attention.forward
    assert Qwen3OmniThinkerAdapter.configure_veomni_model(model, None) is model
    assert model.forward == forward
    assert model._no_split_modules is split_modules
    assert len(model.thinker.model._forward_pre_hooks) == 1
    assert attention._veomni_window_forward == audio_forward
    assert attention.forward == patched_audio
    assert modeling.create_causal_mask is patched_mask
    assert modeling.create_causal_mask(42, cache_position=None) == 42
    assert hasattr(model, "talker")  # Native stage selection belongs to VeOmni's builder.


def test_hf_configuration_does_not_apply_native_compatibility(model_and_modeling):
    model, modeling = model_and_modeling
    attention = model.thinker.audio_tower.layers[0].self_attn
    mask, audio_forward = modeling.create_causal_mask, attention.forward
    assert Qwen3OmniThinkerAdapter.configure_model(model, None) is model
    assert model.forward == model.thinker.forward
    assert model.get_input_embeddings() is model.thinker.model
    assert model._no_split_modules == ["Qwen3OmniMoeThinkerTextDecoderLayer"]
    assert not hasattr(model, "talker")
    assert not model.thinker.model._forward_pre_hooks
    assert attention.forward == audio_forward
    assert modeling.create_causal_mask is mask
    assert not hasattr(model, "_omni_compat_configured")
