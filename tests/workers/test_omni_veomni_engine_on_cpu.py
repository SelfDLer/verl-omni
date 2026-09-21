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

"""Exercise real tensor conversion with distributed/model-loading boundaries mocked."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = ROOT / "verl_omni/workers/engine/veomni"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs_module():
    return load_module("omni_inputs_test", ENGINE_DIR / "omni_inputs.py")


@pytest.fixture
def thinker_config():
    return SimpleNamespace(
        image_token_id=10, video_token_id=11, audio_token_id=12, vision_config=SimpleNamespace(spatial_merge_size=2)
    )


def test_audio_frames_and_global_masks(inputs_module, thinker_config):
    features = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    inputs = {"input_features": features, "feature_attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])}
    full_ids = torch.tensor([[10, 11, 12, 12, -1]])
    result = inputs_module.prepare_omni_inputs(inputs, full_ids, thinker_config)
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
def test_invalid_audio_rejected(inputs_module, thinker_config, payload):
    with pytest.raises(ValueError):
        inputs_module.prepare_omni_inputs(payload, torch.tensor([[1]]), thinker_config)


def test_text_only_and_empty_audio(inputs_module, thinker_config):
    result = inputs_module.prepare_omni_inputs({}, torch.tensor([[1, 2]]), thinker_config)
    assert not any(result[key].any() for key in ("audio_mask", "image_mask", "video_mask"))
    inputs = {"input_features": torch.zeros(0, 3, 4), "feature_attention_mask": torch.zeros(0, 4)}
    result = inputs_module.prepare_omni_inputs(inputs, torch.tensor([[1]]), thinker_config)
    assert result["input_features"].shape == (0, 3)


def test_attention_boundaries_preserve_samples_and_padding(inputs_module):
    result = inputs_module.packed_attention_kwargs(torch.tensor([0, 5, 12]), 4)
    assert result["cu_seq_lens_q"].tolist() == [0, 5, 12, 16]
    assert result["cu_seq_lens_q"].dtype == torch.int32
    assert result["max_length_q"] == 7
    assert inputs_module.packed_attention_kwargs(torch.tensor([0, 5, 12]), 0)["cu_seq_lens_k"].tolist() == [0, 5, 12]


@pytest.fixture
def engine_module(inputs_module, monkeypatch):
    names = [
        "veomni",
        "veomni.arguments",
        "veomni.distributed",
        "veomni.distributed.offloading",
        "veomni.distributed.torch_parallelize",
        "veomni.models",
        "veomni.models.auto",
        "verl",
        "verl.utils",
        "verl.workers",
        "verl.workers.engine",
        "verl.workers.engine.base",
        "verl.workers.engine.veomni",
        "verl.workers.engine.veomni.transformer_impl",
        "verl.workers.engine.veomni.utils",
        "omni_test_engine",
    ]
    mocks = {name: ModuleType(name) for name in names}
    for module in mocks.values():
        module.__path__ = []
    mocks["omni_test_engine.omni_inputs"] = inputs_module
    mocks["omni_test_engine.omni_compat"] = SimpleNamespace(configure_omni_compat=lambda module: module)
    mocks["veomni.arguments"].MixedPrecisionConfig = lambda enable: SimpleNamespace(enable=enable)
    mocks["veomni.distributed.offloading"].build_activation_offloading_context = MagicMock(return_value=(None, None))
    mocks["veomni.distributed.torch_parallelize"].build_parallelize_model = MagicMock(side_effect=lambda m, **kw: m)
    mocks["veomni.models.auto"].build_foundation_model = MagicMock()
    mocks["verl.utils"].tensordict_utils = SimpleNamespace(
        get_non_tensor_data=lambda data, key, default: data.get(key, default)
    )
    mocks["verl.workers.engine.base"].EngineRegistry = SimpleNamespace(register=lambda **kwargs: lambda cls: cls)

    class Parent:
        def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config, **kwargs):
            self.model_config, self.engine_config = model_config, engine_config

        def _get_model_config_path(self):
            return self.model_config.local_path

        def _build_optimizer(self, module):
            return torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=0.1)

        def _build_lr_scheduler(self, optimizer):
            return None

        def get_per_tensor_param(self, **kwargs):
            return iter(self.module.state_dict().items()), None

    upstream = mocks["verl.workers.engine.veomni.transformer_impl"]
    upstream.VeOmniEngineWithLMHead = Parent
    upstream.OmniSequenceShardCollator = MagicMock()
    upstream._build_ops_implementation_config = lambda config: config
    mocks["verl.workers.engine.veomni.utils"].load_safetensors_index = lambda path: {}
    for name, module in mocks.items():
        monkeypatch.setitem(sys.modules, name, module)
    yield load_module("omni_test_engine.omni_impl", ENGINE_DIR / "omni_impl.py")


@pytest.fixture
def configs():
    model = SimpleNamespace(
        architecture="Qwen3OmniMoeForConditionalGeneration",
        model_stage="thinker",
        lora_rank=0,
        lora={},
        lora_adapter_path=None,
        use_fused_kernels=False,
        use_liger=False,
        use_remove_padding=True,
        local_path="fixture",
        enable_gradient_checkpointing=True,
        enable_activation_offload=False,
    )
    engine = SimpleNamespace(
        force_use_huggingface=False,
        expert_parallel_size=1,
        moe_implementation="eager",
        router_replay=SimpleNamespace(mode="disabled"),
        ulysses_parallel_size=1,
        attn_implementation="flash_attention_2",
        mixed_precision=True,
        init_device="meta",
        freeze_vision_tower=True,
        freeze_audio_tower=True,
        enable_full_shard=True,
        enable_fsdp_offload=False,
        basic_modules=[],
        enable_reentrant=False,
        forward_prefetch=False,
        forward_only=False,
        activation_gpu_limit=0.0,
    )
    return model, engine


@pytest.mark.parametrize("field,value", [("model_stage", "talker"), ("lora_rank", 8), ("use_fused_kernels", True)])
def test_unsupported_modes_fail_before_distributed_init(engine_module, configs, field, value):
    model, engine = configs
    setattr(model, field, value)
    with pytest.raises(ValueError):
        engine_module.OmniVeOmniEngine(model, engine, None, None)


@pytest.mark.parametrize("forward_only", [False, True])
def test_build_preserves_native_forward_and_freezes_before_sharding(engine_module, configs, forward_only):
    model_config, engine_config = configs
    engine_config.forward_only = forward_only
    model = torch.nn.Module()
    model.thinker = torch.nn.Module()
    model.thinker.visual = torch.nn.Linear(2, 2)
    model.thinker.audio_tower = torch.nn.Linear(2, 2)
    model.thinker.model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(thinker_config=SimpleNamespace())
    model._no_split_modules = ["Decoder", "VisionBlock", "AudioBlock"]
    native_forward = model.forward
    engine_module.build_foundation_model.return_value = model

    def parallelize(module, **kwargs):
        assert not module.thinker.visual.weight.requires_grad
        assert not module.thinker.audio_tower.weight.requires_grad
        assert module.thinker.model.weight.requires_grad
        assert kwargs["basic_modules"] == ["AudioBlock", "Decoder", "VisionBlock"]
        assert module.forward == native_forward
        return module

    engine_module.build_parallelize_model.side_effect = parallelize
    engine = engine_module.OmniVeOmniEngine(model_config, engine_config, None, None)
    engine._build_model_optimizer()
    assert (engine.optimizer is None) == forward_only
    params, metadata = engine.get_per_tensor_param()
    assert all(t.dtype == torch.bfloat16 for _, t in params)
    assert metadata is None


def test_packed_transform_does_not_use_mrope_resets(engine_module, configs, thinker_config):
    engine = engine_module.OmniVeOmniEngine(*configs, None, None)
    engine.module = SimpleNamespace(config=SimpleNamespace(thinker_config=thinker_config))
    engine.ulysses_sequence_parallel_size = 1
    engine.use_ulysses_sp = False
    tokens = torch.nested.nested_tensor([torch.tensor([1, 10, 10]), torch.tensor([2, 12])], layout=torch.jagged)
    model_inputs = {
        "input_ids": torch.tensor([[1, 10, 10, 2, 12, 0]]),
        "position_ids": torch.tensor([[[0, 0, 0, 0, 1, 0]]]).expand(4, -1, -1),
    }
    engine._apply_veomni_input_transforms(model_inputs, {"input_ids": tokens, "use_remove_padding": True})
    assert model_inputs["cu_seq_lens_q"].tolist() == [0, 3, 5, 6]
    assert model_inputs["image_mask"].tolist() == [[False, True, True, False, False, False]]
    assert model_inputs["use_cache"] is False
