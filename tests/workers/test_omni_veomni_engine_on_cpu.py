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

from verl_omni.pipelines.model_base import OmniModelBase
from verl_omni.pipelines.qwen3_omni.thinker_training_adapter import Qwen3OmniThinkerAdapter

ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = ROOT / "verl_omni/workers/engine/veomni"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def thinker_config():
    return SimpleNamespace(
        image_token_id=10, video_token_id=11, audio_token_id=12, vision_config=SimpleNamespace(spatial_merge_size=2)
    )


def test_attention_boundaries_preserve_samples_and_padding(engine_module):
    result = engine_module._packed_attention_kwargs(torch.tensor([0, 5, 12]), 4)
    assert result["cu_seq_lens_q"].tolist() == [0, 5, 12, 16]
    assert result["cu_seq_lens_q"].dtype == torch.int32
    assert result["max_length_q"] == 7
    assert engine_module._packed_attention_kwargs(torch.tensor([0, 5, 12]), 0)["cu_seq_lens_k"].tolist() == [0, 5, 12]


@pytest.fixture
def engine_module(monkeypatch):
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
        "verl.utils.device",
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
    mocks["veomni.arguments"].MixedPrecisionConfig = lambda enable: SimpleNamespace(enable=enable)
    mocks["veomni.distributed.offloading"].build_activation_offloading_context = MagicMock(return_value=(None, None))
    mocks["veomni.distributed.torch_parallelize"].build_parallelize_model = MagicMock(side_effect=lambda m, **kw: m)
    mocks["veomni.models.auto"].build_foundation_model = MagicMock()
    mocks["verl.utils"].tensordict_utils = SimpleNamespace(
        get_non_tensor_data=lambda data, key, default: data.get(key, default)
    )
    mocks["verl.utils.device"].get_device_id = MagicMock(return_value="meta")
    mocks["verl.workers.engine.base"].EngineRegistry = SimpleNamespace(register=lambda **kwargs: lambda cls: cls)

    class Parent:
        def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config, **kwargs):
            self.model_config, self.engine_config = model_config, engine_config
            self._is_offload_param = engine_config.param_offload
            self._is_offload_optimizer = engine_config.optimizer_offload
            self._uses_fsdp2_cpu_offload_policy = engine_config.enable_fsdp_offload

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
        param_offload=True,
        optimizer_offload=True,
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
@pytest.mark.parametrize("native_offload", [False, True])
def test_build_preserves_native_forward_and_freezes_before_sharding(
    engine_module, configs, forward_only, native_offload, monkeypatch
):
    model_config, engine_config = configs
    engine_config.forward_only = forward_only
    engine_config.enable_fsdp_offload = native_offload
    model = torch.nn.Module()
    model.thinker = torch.nn.Module()
    model.thinker.visual = torch.nn.Linear(2, 2)
    model.thinker.audio_tower = torch.nn.Linear(2, 2)
    model.thinker.model = torch.nn.Linear(2, 2)
    model.register_buffer("root_buffer", torch.ones(2))
    model.thinker.audio_tower.register_buffer("position_buffer", torch.ones(2), persistent=False)
    model.thinker.audio_tower.register_buffer("unused_buffer", None)
    original_parameters = tuple(model.parameters())
    model.config = SimpleNamespace(thinker_config=SimpleNamespace())
    model._no_split_modules = ["Decoder", "VisionBlock", "AudioBlock"]
    native_forward = model.forward
    engine_module.build_foundation_model.return_value = model
    configure = MagicMock(side_effect=lambda module, config: module)
    monkeypatch.setattr(Qwen3OmniThinkerAdapter, "configure_veomni_model", configure)

    def parallelize(module, **kwargs):
        configure.assert_called_once_with(module, model_config)
        assert not module.thinker.visual.weight.requires_grad
        assert not module.thinker.audio_tower.weight.requires_grad
        assert module.thinker.model.weight.requires_grad
        assert kwargs["basic_modules"] == ["AudioBlock", "Decoder", "VisionBlock"]
        assert module.forward == native_forward
        return module

    engine_module.build_parallelize_model.side_effect = parallelize
    engine = engine_module.OmniVeOmniEngine(model_config, engine_config, None, None)
    engine._build_model_optimizer()
    assert engine.model_adapter_cls is Qwen3OmniThinkerAdapter
    assert (engine.optimizer is None) == forward_only
    assert engine._is_offload_param is not native_offload
    assert engine._is_offload_optimizer is not native_offload
    assert engine._uses_fsdp2_cpu_offload_policy is native_offload
    assert engine_config.param_offload and engine_config.optimizer_offload
    # Meta stands in for the accelerator: buffers move, but optimizer parameters must not.
    buffer_device = "meta" if native_offload else "cpu"
    assert model.root_buffer.device.type == buffer_device
    assert model.thinker.audio_tower.position_buffer.device.type == buffer_device
    assert model.thinker.audio_tower.unused_buffer is None
    for original, parameter in zip(original_parameters, model.parameters(), strict=True):
        assert parameter is original
        assert parameter.device.type == "cpu"
    if engine.optimizer is not None:
        assert all(p.device.type == "cpu" for group in engine.optimizer.param_groups for p in group["params"])
    if native_offload:
        engine_module.get_device_id.assert_called_once_with()
    else:
        engine_module.get_device_id.assert_not_called()
    params, metadata = engine.get_per_tensor_param()
    assert all(t.dtype == torch.bfloat16 for _, t in params)
    assert metadata is None


def test_packed_transform_does_not_use_mrope_resets(engine_module, configs, thinker_config):
    engine = engine_module.OmniVeOmniEngine(*configs, None, None)
    engine.model_adapter_cls = Qwen3OmniThinkerAdapter
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
    assert "use_cache" not in model_inputs


def test_engine_resolves_registered_adapter(engine_module, configs, monkeypatch):
    class CustomAdapter(Qwen3OmniThinkerAdapter):
        configure_veomni_model = MagicMock(side_effect=RuntimeError("adapter reached"))

    monkeypatch.setattr(
        OmniModelBase, "_registry", {("Qwen3OmniMoeForConditionalGeneration", "thinker"): CustomAdapter}
    )
    engine = engine_module.OmniVeOmniEngine(*configs, None, None)
    with pytest.raises(RuntimeError, match="adapter reached"):
        engine._build_model_optimizer()
    assert engine.model_adapter_cls is CustomAdapter


@pytest.mark.parametrize("new_dict", [False, True])
def test_engine_preserves_adapter_input_return_contract(engine_module, configs, new_dict):
    def prepare(inputs, full_ids, hf_config):
        result = {"input_ids": full_ids, "adapter_field": 42}
        if new_dict:
            return result
        inputs.clear()
        inputs.update(result)
        return inputs

    engine = engine_module.OmniVeOmniEngine(*configs, None, None)
    engine.model_adapter_cls = SimpleNamespace(prepare_veomni_inputs=prepare)
    engine.module = SimpleNamespace(config=object())
    engine.use_ulysses_sp = False
    inputs = {"input_ids": torch.tensor([[1, 2]]), "removed_field": True}
    engine._apply_veomni_input_transforms(inputs, {"use_remove_padding": False})
    assert inputs["adapter_field"] == 42
    assert "removed_field" not in inputs
    assert "use_cache" not in inputs


def test_engine_rejects_invalid_adapter_input_return(engine_module, configs):
    engine = engine_module.OmniVeOmniEngine(*configs, None, None)
    engine.model_adapter_cls = SimpleNamespace(prepare_veomni_inputs=lambda *args: None)
    engine.module = SimpleNamespace(config=object())
    with pytest.raises(TypeError, match="must return a dict"):
        engine._apply_veomni_input_transforms({"input_ids": torch.tensor([[1]])}, {"use_remove_padding": False})
