from unittest.mock import Mock

from npu_test.skip_initial_weight_sync import OmniPPOTrainerSkipInitialWeightSync


def test_skip_initial_weight_sync_only_wakes_checkpoint_loaded_rollout():
    trainer = object.__new__(OmniPPOTrainerSkipInitialWeightSync)
    trainer.checkpoint_manager = Mock()

    trainer.on_init_end()

    trainer.checkpoint_manager.wake_up_replicas.assert_called_once_with()
    trainer.checkpoint_manager.update_weights.assert_not_called()
