"""Diagnostic V1 trainer that keeps vLLM-Omni's checkpoint-loaded state."""

import logging
from unittest.mock import patch

from verl.checkpoint_engine import CheckpointEngineManager
from verl.trainer.ppo.v1.trainer_base import register_trainer

from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync

logger = logging.getLogger(__name__)


def _skip_initial_sleep(_manager):
    """Leave the freshly initialized rollout allocations mapped on device."""
    logger.warning("NPU parity diagnostic: skipping initial rollout sleep")


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
            "using checkpoint-loaded rollout state unchanged"
        )
