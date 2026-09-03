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

from unittest.mock import Mock, patch

from verl.checkpoint_engine import CheckpointEngineManager

from npu_test.skip_initial_weight_sync import OmniPPOTrainerSkipInitialWeightSync


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
