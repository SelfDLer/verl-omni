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
"""Check the 0.28 backport's guard placement without loading inference engines."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def omni_impl(monkeypatch):
    modules = {
        "transformers": {"AutoModelForMultimodalLM": MagicMock()},
        "verl.utils.activation_offload": {"enable_activation_offloading": MagicMock()},
        "verl.utils.debug": {"log_gpu_memory_usage": MagicMock()},
        "verl.utils.device": {"get_device_id": lambda: 0},
        "verl.utils.fsdp_utils": {
            name: MagicMock()
            for name in (
                "CPUOffloadPolicy",
                "MixedPrecisionPolicy",
                "apply_fsdp2",
                "fsdp2_load_full_state_dict",
                "fsdp_version",
                "get_fsdp_wrap_policy",
                "get_init_weight_context_manager",
                "init_fn",
                "load_fsdp_model_to_gpu",
                "merged_lora_context",
                "normalize_peft_param_name",
                "offload_fsdp_model_to_cpu",
                "replace_lora_wrapper",
            )
        },
        "verl.utils.model": {"convert_weight_keys": MagicMock()},
        "verl.utils.torch_dtypes": {"PrecisionType": SimpleNamespace(to_dtype=lambda value: value)},
        "verl.workers.engine.base": {
            "EngineRegistry": SimpleNamespace(register=lambda **kwargs: lambda cls: cls),
        },
        "verl.workers.engine.fsdp.transformer_impl": {"FSDPEngineWithLMHead": type("BaseEngine", (), {})},
        "verl.workers.engine.fsdp.utils": {"get_sharding_strategy": MagicMock()},
        "verl_omni.utils.fsdp_utils": {"collect_lora_params": MagicMock()},
        "verl_omni.utils.fsdp_offload": {"install_fsdp2_cpu_transfer_guard": MagicMock(return_value=True)},
        "verl_omni.workers.config": {"OmniModelConfig": type("OmniModelConfig", (), {})},
    }
    for name, attributes in modules.items():
        parts = name.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                package = ModuleType(parent)
                package.__path__ = []
                monkeypatch.setitem(sys.modules, parent, package)
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).resolve().parents[2] / "verl_omni/workers/engine/fsdp/omni_impl.py"
    spec = importlib.util.spec_from_file_location("_omni_fsdp_offload_backport_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    module.fsdp_version.return_value = 2
    return module


def _engine(omni_impl, device_type, offload_policy, forward_only):
    engine = object.__new__(omni_impl.OmniFSDPEngine)
    engine.engine_config = SimpleNamespace(
        strategy="fsdp2",
        mixed_precision=None,
        wrap_policy={},
        reshard_after_forward=True,
        offload_policy=offload_policy,
        forward_only=forward_only,
    )
    engine.model_config = SimpleNamespace(
        lora_rank=0,
        enable_activation_offload=False,
        enable_gradient_checkpointing=False,
    )
    engine.device_mesh = SimpleNamespace(device_type=device_type)
    engine._is_offload_param = True
    engine._is_offload_optimizer = True
    engine._uses_fsdp2_cpu_offload_policy = False
    return engine


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
@pytest.mark.parametrize("offload_policy,forward_only", [(False, False), (True, False), (False, True), (True, True)])
def test_guard_is_installed_before_initial_state_loading(omni_impl, device_type, offload_policy, forward_only):
    engine = _engine(omni_impl, device_type, offload_policy, forward_only)
    model = torch.nn.Linear(2, 2)
    calls = []
    pinned = offload_policy or forward_only
    policy = SimpleNamespace(pin_memory=True)
    omni_impl.CPUOffloadPolicy.return_value = policy

    def wrap(target, kwargs, config):
        assert target is model and config is engine.engine_config
        assert kwargs["mesh"] is engine.device_mesh
        assert kwargs["offload_policy"] is (policy if pinned else None)
        calls.append("wrap")

    def guard(target, backend, *, pin_memory):
        assert target is model and backend == device_type and pin_memory == pinned
        calls.append("guard")
        return True

    def load(target, state, mesh, offload):
        assert target is model and mesh is engine.device_mesh
        assert offload is (policy if pinned else None)
        torch.testing.assert_close(state["weight"], model.weight)
        # The loader may call model.to("cpu", non_blocking=True) itself.
        assert calls == ["wrap", "guard"]
        calls.append("load")

    omni_impl.apply_fsdp2.side_effect = wrap
    omni_impl.install_fsdp2_cpu_transfer_guard.side_effect = guard
    omni_impl.fsdp2_load_full_state_dict.side_effect = load
    assert engine._build_fsdp_module(model) is model
    assert calls == ["wrap", "guard", "load"]
    assert engine._uses_fsdp2_cpu_offload_policy == pinned
    assert engine._is_offload_param == (not pinned)
    assert engine._is_offload_optimizer == (not pinned)
    assert engine._autocast_dtype == torch.bfloat16
    assert engine.scaler is None


def test_fsdp1_does_not_install_fsdp2_guard(omni_impl, monkeypatch):
    engine = _engine(omni_impl, "cuda", False, False)
    engine.engine_config.strategy = "fsdp"
    engine.engine_config.forward_prefetch = False
    engine.engine_config.use_orig_params = True
    model = torch.nn.Linear(2, 2)
    wrapped = object()
    fsdp = MagicMock(return_value=wrapped)
    monkeypatch.setattr(omni_impl, "FSDP", fsdp)
    omni_impl.fsdp_version.return_value = 1

    assert engine._build_fsdp_module(model) is wrapped
    assert fsdp.call_args.args == (model,)
    fsdp.set_state_dict_type.assert_called_once()
    omni_impl.install_fsdp2_cpu_transfer_guard.assert_not_called()
    omni_impl.fsdp2_load_full_state_dict.assert_not_called()
