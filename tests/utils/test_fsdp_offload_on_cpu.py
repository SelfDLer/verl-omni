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
"""Exercise model-scoped transfer ordering with deterministic deferred host copies."""

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

GUARD = runpy.run_path(Path(__file__).resolve().parents[2] / "verl_omni/utils/fsdp_offload.py")
install_guard = GUARD["install_fsdp2_cpu_transfer_guard"]


class Tensor:
    def __init__(self, values, device="npu", index=15):
        self.values = values
        self.device = SimpleNamespace(type=device, index=index)


class Device:
    def __init__(self):
        self.pending = []
        self.waited_devices = []

    def synchronize(self, device=None):
        self.waited_devices.append(device)
        for copy in self.pending:
            copy()
        self.pending.clear()

    def to_cpu(self, tensor):
        result = Tensor([-999.0] * len(tensor.values), "cpu")

        def copy():
            result.values[:] = tensor.values

        self.pending.append(copy)
        return result


class FSDPModule:
    """Model the recursive _apply -> per-child CPU repadding order."""

    def __init__(self, children=(), parameter=None, gradient=None, buffer=None):
        self.children = children
        self.parameter = parameter
        self.gradient = gradient
        self.buffer = buffer

    def _apply(self, fn, recurse=True):
        if recurse:
            for child in self.children:
                child._apply(fn)
        for name in ("parameter", "gradient", "buffer"):
            if (tensor := getattr(self, name)) is not None:
                setattr(self, name, fn(tensor))
        if self.parameter is not None and self.parameter.device.type == "cpu":
            # Like an uneven FSDP shard, this allocation is detached from the
            # DMA destination; a later device wait cannot repair its contents.
            self.parameter = Tensor(self.parameter.values.copy(), "cpu")
        return self


@pytest.fixture
def device(monkeypatch):
    accelerator = Device()
    fsdp = ModuleType("torch.distributed.fsdp")
    fsdp.FSDPModule = FSDPModule
    utilities = ModuleType("verl.utils.device")
    utilities.get_torch_device = lambda: accelerator
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", fsdp)
    monkeypatch.setitem(sys.modules, "verl.utils.device", utilities)
    return accelerator


def test_guard_waits_before_nested_child_repads_not_after_root_returns(device):
    expected = [float(i) for i in range(119)]
    bad_leaf = FSDPModule(parameter=Tensor(expected.copy()))
    bad_root = FSDPModule(children=[FSDPModule(children=[bad_leaf])])
    bad_root._apply(device.to_cpu)
    device.synchronize()
    assert bad_leaf.parameter.values == [-999.0] * 119

    good_leaf = FSDPModule(parameter=Tensor(expected.copy()))
    good_root = FSDPModule(children=[FSDPModule(children=[good_leaf])])
    assert install_guard(good_root, "npu")
    assert good_root._apply(device.to_cpu) is good_root
    assert good_leaf.parameter.values == expected
    assert device.pending == []
    assert device.waited_devices[-1].index == 15


def test_guard_preserves_gradients_buffers_and_multiple_source_devices(device):
    root = FSDPModule(parameter=Tensor([1.0], index=2), gradient=Tensor([2.0], index=2), buffer=Tensor([3.0], index=5))
    install_guard(root, "npu")
    root._apply(device.to_cpu)
    assert root.parameter.values == [1.0]
    assert root.gradient.values == [2.0]
    assert root.buffer.values == [3.0]
    assert [value.index for value in device.waited_devices] == [2, 2, 5]


def test_guard_is_idempotent_and_does_not_patch_other_models_or_the_class(device):
    guarded, other = FSDPModule(parameter=Tensor([7.0])), FSDPModule(parameter=Tensor([7.0]))
    original = FSDPModule._apply
    assert install_guard(guarded, "npu")
    bound = guarded._apply
    assert not install_guard(guarded, "npu")
    assert guarded._apply is bound
    guarded._apply(device.to_cpu)
    other._apply(device.to_cpu)
    device.synchronize()
    assert guarded.parameter.values == [7.0]
    assert other.parameter.values == [-999.0]
    assert FSDPModule._apply is original


@pytest.mark.parametrize("source,target", [("cpu", "cpu"), ("cpu", "npu"), ("meta", "npu"), ("npu", "npu")])
def test_guard_does_not_synchronize_other_conversion_directions(device, source, target):
    root = FSDPModule(parameter=Tensor([7.0], source))
    install_guard(root, "npu")
    root._apply(lambda tensor: Tensor(tensor.values.copy(), target))
    assert root.parameter.values == [7.0]
    assert device.waited_devices == []


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_non_npu_install_is_a_noop_without_importing_accelerator_dependencies(backend):
    assert not install_guard(object(), backend)


def test_recurse_false_is_preserved(device):
    child = FSDPModule(parameter=Tensor([9.0]))
    root = FSDPModule(children=[child], parameter=Tensor([7.0]))
    install_guard(root, "npu")
    root._apply(fn=device.to_cpu, recurse=False)
    assert root.parameter.values == [7.0]
    assert child.parameter.device.type == "npu"
    assert len(device.waited_devices) == 1


def test_guard_propagates_conversion_and_device_errors(device):
    root = FSDPModule(parameter=Tensor([7.0]))
    install_guard(root, "npu")

    def fail(*args):
        raise RuntimeError("transfer failed")

    with pytest.raises(RuntimeError, match="transfer failed"):
        root._apply(fail)
    assert device.waited_devices == []
    device.synchronize = fail
    with pytest.raises(RuntimeError, match="transfer failed"):
        root._apply(device.to_cpu)


def test_guard_rejects_non_fsdp_modules(device):
    with pytest.raises(TypeError, match="FSDP2 root"):
        install_guard(SimpleNamespace(), "npu")
