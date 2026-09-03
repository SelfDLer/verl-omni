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

"""Diagnostic V1 trainer that keeps vLLM-Omni's checkpoint-loaded state."""

import logging
from unittest.mock import patch

import ray
from verl.checkpoint_engine import CheckpointEngineManager
from verl.trainer.ppo.v1.trainer_base import register_trainer

from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync

logger = logging.getLogger(__name__)


def _skip_initial_sleep(_manager):
    """Leave the freshly initialized rollout allocations mapped on device."""
    logger.warning("NPU parity diagnostic: skipping initial rollout sleep")


def _set_rollout_global_steps(manager, global_steps):
    """Set rollout version metadata without transferring or reloading weights."""
    object_refs = []
    for replica in manager.replicas:
        object_refs.extend(server.set_global_steps.remote(global_steps) for server in replica.servers)
    if not object_refs:
        raise RuntimeError("No rollout servers found while setting initial global_steps")
    ray.get(object_refs)


@register_trainer("omni_sync_skip_initial_weight_sync")
class OmniPPOTrainerSkipInitialWeightSync(OmniPPOTrainerSync):
    """Generate once from the rollout's untouched checkpoint-loaded state."""

    def init(self):
        # PPOTrainer._setup() normally sleeps rollout before _load_checkpoint(),
        # then on_init_end() reloads actor weights. Waking that partially
        # discarded state without a reload is invalid on vLLM-Ascend, so bypass
        # both halves of the initial sleep/reload lifecycle for this one case.
        with patch.object(CheckpointEngineManager, "sleep_replicas", _skip_initial_sleep):
            super().init()

    def on_init_end(self):
        logger.warning(
            "NPU parity diagnostic: skipping initial actor-to-rollout weight sync; "
            "using checkpoint-loaded rollout state unchanged and marking it as global_steps=%s",
            self.global_steps,
        )
        _set_rollout_global_steps(self.checkpoint_manager, self.global_steps)
