"""Diagnostic V1 trainer that keeps vLLM-Omni's checkpoint-loaded weights."""

import logging

from verl.trainer.ppo.v1.trainer_base import register_trainer

from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync

logger = logging.getLogger(__name__)


@register_trainer("omni_sync_skip_initial_weight_sync")
class OmniPPOTrainerSkipInitialWeightSync(OmniPPOTrainerSync):
    """Wake rollout replicas without replacing their initial checkpoint weights."""

    def on_init_end(self):
        logger.warning(
            "NPU parity diagnostic: skipping initial actor-to-rollout weight sync; "
            "waking checkpoint-loaded rollout replicas unchanged"
        )
        self.checkpoint_manager.wake_up_replicas()
