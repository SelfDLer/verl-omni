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

from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

from verl.checkpoint_engine import CheckpointEngineManager

from npu_test.skip_initial_weight_sync import (
    OmniPPOTrainerSkipInitialSleep,
    OmniPPOTrainerSkipInitialSleepPearsonOnly,
    OmniPPOTrainerSkipInitialWeightSync,
    OmniPPOTrainerSkipInitialWeightSyncPearsonOnly,
)


def test_skip_initial_weight_sync_keeps_checkpoint_loaded_rollout_awake():
    trainer = object.__new__(OmniPPOTrainerSkipInitialWeightSync)
    manager = Mock()
    original_sleep = CheckpointEngineManager.sleep_replicas
    trainer._setup = Mock(side_effect=lambda: CheckpointEngineManager.sleep_replicas(manager))

    trainer.init()

    trainer._setup.assert_called_once_with()
    assert manager.mock_calls == []
    assert CheckpointEngineManager.sleep_replicas is original_sleep


def test_skip_initial_weight_sync_sets_version_without_reloading_weights():
    trainer = object.__new__(OmniPPOTrainerSkipInitialWeightSync)
    trainer.global_steps = 0
    server = Mock()
    server.set_global_steps.remote.return_value = "set-version-ref"
    trainer.checkpoint_manager = Mock(replicas=[Mock(servers=[server])])

    with patch("npu_test.skip_initial_weight_sync.ray.get") as ray_get:
        trainer.on_init_end()

    server.set_global_steps.remote.assert_called_once_with(0)
    ray_get.assert_called_once_with(["set-version-ref"])
    trainer.checkpoint_manager.update_weights.assert_not_called()


def _awake_reload_trainer(*, free_cache_engine):
    trainer = object.__new__(OmniPPOTrainerSkipInitialSleep)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(free_cache_engine=free_cache_engine),
        )
    )
    trainer.global_steps = 0
    trainer.checkpoint_manager = Mock()
    trainer._setup = Mock(side_effect=lambda: CheckpointEngineManager.sleep_replicas(trainer.checkpoint_manager))
    return trainer


def test_skip_initial_sleep_still_reloads_actor_weights():
    trainer = _awake_reload_trainer(free_cache_engine=False)
    original_sleep = CheckpointEngineManager.sleep_replicas

    trainer.init()

    trainer._setup.assert_called_once_with()
    trainer.checkpoint_manager.sleep_replicas.assert_not_called()
    trainer.checkpoint_manager.update_weights.assert_called_once_with(0)
    assert CheckpointEngineManager.sleep_replicas is original_sleep


def test_skip_initial_sleep_rejects_implicit_wake_of_awake_rollout():
    trainer = _awake_reload_trainer(free_cache_engine=True)

    with pytest.raises(ValueError, match="free_cache_engine=false"):
        trainer.init()

    trainer._setup.assert_not_called()
    trainer.checkpoint_manager.update_weights.assert_not_called()


@pytest.mark.parametrize(
    "trainer_cls",
    (OmniPPOTrainerSkipInitialWeightSyncPearsonOnly, OmniPPOTrainerSkipInitialSleepPearsonOnly),
)
def test_pearson_only_diagnostic_sleeps_rollout_before_actor_log_prob(trainer_cls):
    trainer = object.__new__(trainer_cls)
    server = Mock()
    server.wait_for_requests_to_drain.remote.return_value = "drain-ref"
    server.collective_rpc.remote.return_value = "sleep-ref"
    server.clear_kv_cache.remote.return_value = "clear-ref"
    trainer.checkpoint_manager = Mock(replicas=[Mock(servers=[server])])

    with patch("npu_test.skip_initial_weight_sync.ray.get") as ray_get:
        trainer.on_sample_end()

    assert ray_get.call_args_list == [call(["drain-ref"]), call(["sleep-ref"]), call(["clear-ref"])]
    server.wait_for_requests_to_drain.remote.assert_called_once_with()
    server.collective_rpc.remote.assert_called_once_with("sleep", kwargs={"level": 1})
    server.clear_kv_cache.remote.assert_called_once_with()


def test_pearson_only_diagnostic_skips_post_step_weight_sync():
    trainer = object.__new__(OmniPPOTrainerSkipInitialSleepPearsonOnly)
    trainer.checkpoint_manager = Mock()

    trainer.on_step_end()

    trainer.checkpoint_manager.update_weights.assert_not_called()


def test_pearson_only_diagnostic_skips_actor_update():
    trainer = object.__new__(OmniPPOTrainerSkipInitialSleepPearsonOnly)
    batch = object()

    assert trainer._update_actor(batch, metrics={}) is batch
