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

"""Exercise the optional trainer hook without importing Ray/NPU registration."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    class Base:
        def _compute_old_log_prob(self, batch, metrics):
            self.recomputed = True
            return batch

    # Compile the entire production subclass; isolate its external superclass
    # and queue transport, but execute the real diagnostic writer.
    tree = ast.parse((ROOT / "verl_omni/trainer/omni/ray_omni_trainer.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OmniPPOTrainerSync")
    namespace = {"PPOTrainerSync": Base, "register_trainer": lambda name: lambda cls: cls}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "ray_omni_trainer.py", "exec"), namespace)
    instance = namespace["OmniPPOTrainerSync"]()
    instance.config = OmegaConf.create(
        {
            "trainer": {"consistency_debug_dir": str(tmp_path)},
            "actor_rollout_ref": {"rollout": {"calculate_log_probs": True}},
            "algorithm": {"rollout_correction": {"bypass_mode": False}},
        }
    )
    instance.global_steps = 2
    instance.recomputed = False
    payload = {
        "old_log_probs": torch.tensor([[-1.0, -2.0]]),
        "rollout_log_probs": torch.tensor([[-1.2, -2.1]]),
        "response_mask": torch.ones(1, 2),
        "responses": torch.tensor([[1, 2]]),
    }
    instance.queue_calls = []

    def get(**kwargs):
        assert instance.recomputed
        instance.queue_calls.append(kwargs)
        return SimpleNamespace(to_padded_tensor=lambda: payload)

    monkeypatch.setitem(sys.modules, "transfer_queue", SimpleNamespace(kv_batch_get=get))
    spec = importlib.util.spec_from_file_location("consistency_writer", ROOT / "verl_omni/utils/consistency_debug.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "verl_omni.utils.consistency_debug", module)
    return instance


def test_enabled_hook_records_pairs_after_recompute(trainer, tmp_path):
    batch = SimpleNamespace(keys=["sample"], partition_id="train")
    metrics = {}
    assert trainer._compute_old_log_prob(batch, metrics) is batch
    assert metrics["consistency/logprob_mae"] == pytest.approx(0.15)
    assert trainer.queue_calls == [
        {
            "keys": ["sample"],
            "partition_id": "train",
            "select_fields": ["old_log_probs", "rollout_log_probs", "response_mask", "responses"],
        }
    ]
    assert (tmp_path / "step_000002.pt").exists()


def test_disabled_hook_does_not_read_queue(trainer):
    trainer.config.trainer.consistency_debug_dir = None
    batch = object()
    metrics = {}
    assert trainer._compute_old_log_prob(batch, metrics) is batch
    assert trainer.recomputed
    assert not trainer.queue_calls
    assert metrics == {}


@pytest.mark.parametrize("bypass,rollout_logprobs", [(True, True), (False, False)])
def test_invalid_comparison_configuration_fails(trainer, bypass, rollout_logprobs):
    trainer.config.algorithm.rollout_correction.bypass_mode = bypass
    trainer.config.actor_rollout_ref.rollout.calculate_log_probs = rollout_logprobs
    with pytest.raises(ValueError, match="independent actor"):
        trainer._compute_old_log_prob(object(), {})
    assert not trainer.recomputed
