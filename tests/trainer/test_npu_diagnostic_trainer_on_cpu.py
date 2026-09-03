from unittest.mock import Mock

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
