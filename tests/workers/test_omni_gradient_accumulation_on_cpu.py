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
"""Test the real sync context with an isolated distributed-engine boundary."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def make_engine(device, strategy):
    """Compile the production method without importing Ray or accelerator engines."""
    source = Path(__file__).resolve().parents[2] / "verl_omni/workers/engine/fsdp/omni_impl.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OmniFSDPEngine")
    cls.body = [
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_gradient_sync_context"
    ]
    cls.decorator_list = []

    class Base:
        @contextmanager
        def _gradient_sync_context(self, *, is_last_micro_batch):
            self.calls.append(is_last_micro_batch)
            self.sync_enabled = is_last_micro_batch
            try:
                yield
            finally:
                self.sync_enabled = True

    namespace = {
        "contextmanager": contextmanager,
        "get_device_name": lambda: device,
        "FSDPEngineWithLMHead": Base,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    engine = namespace["OmniFSDPEngine"]()
    engine.engine_config = SimpleNamespace(strategy=strategy)
    engine.calls = []
    engine.sync_enabled = True
    return engine


@pytest.mark.parametrize("last", [False, True])
def test_npu_fsdp2_synchronizes_every_backward(last):
    engine = make_engine("npu", "fsdp2")
    with engine._gradient_sync_context(is_last_micro_batch=last):
        assert engine.sync_enabled
    assert engine.calls == []


@pytest.mark.parametrize("device,strategy", [("cuda", "fsdp2"), ("npu", "fsdp"), ("cpu", "fsdp2")])
def test_other_backends_delegate_and_restore_on_exception(device, strategy):
    engine = make_engine(device, strategy)
    with pytest.raises(RuntimeError, match="backward failed"):
        with engine._gradient_sync_context(is_last_micro_batch=False):
            assert not engine.sync_enabled
            raise RuntimeError("backward failed")
    assert engine.sync_enabled
    assert engine.calls == [False]


def test_accumulated_update_matches_full_batch():
    # Real autograd/optimizer test: synchronizing each backward must not zero
    # accumulated gradients or move the optimizer step inside the micro loop.
    engine = make_engine("npu", "fsdp2")
    weight = torch.nn.Parameter(torch.tensor([0.5, -0.3]))
    reference = torch.nn.Parameter(weight.detach().clone())
    inputs = torch.tensor([[1.0, 2.0], [3.0, -1.0], [-2.0, 0.5]])
    targets = torch.tensor([1.0, -1.0, 2.0])
    optimizer = torch.optim.AdamW([weight], lr=0.01)
    reference_optimizer = torch.optim.AdamW([reference], lr=0.01)
    initial = weight.detach().clone()
    for index in range(len(inputs)):
        with engine._gradient_sync_context(is_last_micro_batch=index == len(inputs) - 1):
            assert engine.sync_enabled
            loss = (inputs[index] @ weight - targets[index]).square() / len(inputs)
            loss.backward()
        torch.testing.assert_close(weight, initial)
    ((inputs @ reference - targets).square().mean()).backward()
    torch.testing.assert_close(weight.grad, reference.grad)
    optimizer.step()
    reference_optimizer.step()
    torch.testing.assert_close(weight, reference)
