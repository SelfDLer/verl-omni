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

from pathlib import Path


def test_avqa_npu_launcher_wires_v1_multimodal_training():
    launcher_dir = Path(__file__).parents[2] / "examples/gspo_trainer/qwen3_omni"
    avqa_launcher = (launcher_dir / "run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh").read_text(encoding="utf-8")

    required_settings = (
        "python3 -m verl_omni.trainer.main_omni",
        "data.custom_cls.name=QwenOmniRLHFDataset",
        "++data.mm_processor_kwargs.sampling_rate=16000",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.rollout.name=vllm_omni",
        "engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe",
        "reward.reward_manager.source=register",
        "reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py",
    )
    assert all(setting in avqa_launcher for setting in required_settings)
    assert "models.transformers" not in avqa_launcher


def test_nextqa_npu_launcher_wires_video_gspo_training():
    launcher_dir = Path(__file__).parents[2] / "examples/gspo_trainer/qwen3_omni"
    launcher = (launcher_dir / "run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh").read_text(encoding="utf-8")

    required_settings = (
        "python3 -m verl_omni.trainer.main_omni",
        "data.custom_cls.name=QwenOmniRLHFDataset",
        "USE_AUDIO_IN_VIDEO=${USE_AUDIO_IN_VIDEO:-true}",
        "FILTER_OVERLONG_PROMPTS_WORKERS=${FILTER_OVERLONG_PROMPTS_WORKERS:-8}",
        "data.filter_overlong_prompts_workers=${FILTER_OVERLONG_PROMPTS_WORKERS}",
        "++data.mm_processor_kwargs.use_audio_in_video=${USE_AUDIO_IN_VIDEO}",
        "++data.mm_processor_kwargs.sampling_rate=16000",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.actor.policy_loss.loss_mode=gspo",
        "actor_rollout_ref.rollout.name=vllm_omni",
        "engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe",
        "reward.reward_manager.source=register",
        "reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py",
        "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}",
    )
    assert all(setting in launcher for setting in required_settings)
    required_environment = (
        "export VERL_USE_EXTERNAL_MODULES=${VERL_USE_EXTERNAL_MODULES:-verl_omni}",
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}",
        "export TQ_NUM_THREADS=${TQ_NUM_THREADS:-8}",
        "export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}",
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}",
        "export FORCE_QWENVL_VIDEO_READER=${FORCE_QWENVL_VIDEO_READER:-torchcodec}",
    )
    assert all(setting in launcher for setting in required_environment)
    assert "data.filter_overlong_prompts_workers=64" not in launcher
    assert "models.transformers" not in launcher


def test_nextqa_npu_pearson_matrix_caps_samples_before_filtering():
    repo_root = Path(__file__).parents[2]
    matrix = (repo_root / "npu_test" / "run_nextqa_pearson_matrix.sh").read_text(encoding="utf-8")

    required_settings = (
        "DIAG_TRAIN_MAX_SAMPLES=${DIAG_TRAIN_MAX_SAMPLES:-64}",
        "DIAG_VAL_MAX_SAMPLES=${DIAG_VAL_MAX_SAMPLES:-8}",
        '"data.train_max_samples=${DIAG_TRAIN_MAX_SAMPLES}"',
        '"data.val_max_samples=${DIAG_VAL_MAX_SAMPLES}"',
    )
    assert all(setting in matrix for setting in required_settings)


def test_nextqa_npu_pearson_matrix_has_weight_and_batch_ablation_cases():
    repo_root = Path(__file__).parents[2]
    matrix = (repo_root / "npu_test" / "run_nextqa_pearson_matrix.sh").read_text(encoding="utf-8")

    required_settings = (
        "CASES=(baseline no_audio no_init_sync tp4 low_concurrency batch_invariant)",
        'external_modules="${BASE_EXTERNAL_MODULES},npu_test.skip_initial_weight_sync"',
        '"trainer.v1.trainer_mode=omni_sync_skip_initial_weight_sync"',
        '"trainer.resume_mode=disable"',
        '"actor_rollout_ref.rollout.max_num_seqs=8"',
        '"actor_rollout_ref.rollout.full_determinism=true"',
    )
    assert all(setting in matrix for setting in required_settings)
