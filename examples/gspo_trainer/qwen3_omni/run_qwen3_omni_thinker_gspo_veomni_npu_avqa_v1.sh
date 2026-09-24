#!/usr/bin/env bash
# AVQA GSPO on Ascend A3: VeOmni Thinker training + vLLM-Omni rollout.
# Source CANN/ATB first in the pinned docker/Dockerfile.a3.npu runtime.
# Defaults run two updates; pass Hydra overrides as trailing arguments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CPATH=/usr/include${CPATH:+:$CPATH}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-Omni-30B-A3B-Instruct"}
export TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/avqa_r1_6k/train.parquet"}
export VAL_FILE=${VAL_FILE:-"$HOME/data/avqa_r1_6k/validation.parquet"}
export NUM_NPUS=${NUM_NPUS:-${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-16}}
export NNODES=${NNODES:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-2}
for name in NUM_NPUS NNODES ROLLOUT_TP; do
    if [[ ! ${!name} =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer, got '${!name}'." >&2
        exit 2
    fi
done
if (( NUM_NPUS % ROLLOUT_TP != 0 )); then
    echo "NUM_NPUS must be divisible by ROLLOUT_TP." >&2
    exit 2
fi
WORLD_SIZE_NPU=$((NUM_NPUS * NNODES))
LOG_FILE=${LOG_FILE:-"${SCRIPT_DIR}/../../../logs/gspo_avqa_veomni_npu_$(date +%Y%m%d_%H%M%S).log"}
mkdir -p "$(dirname "${LOG_FILE}")"

# Reuse the VeOmni actor/ref groups, NPU settings and frozen encoder defaults.
# Per-device micro-batches are one sample; the batch below controls accumulation.
bash "${SCRIPT_DIR}/run_qwen3_omni_thinker_gspo_veomni.sh" \
    data.train_batch_size="${WORLD_SIZE_NPU}" \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.shuffle=true \
    data.seed=42 \
    data.val_max_samples=16 \
    data.validation_shuffle=false \
    data.filter_overlong_prompts_workers=8 \
    data.truncation=error \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets \
    data.custom_cls.name=QwenOmniRLHFDataset \
    ++data.mm_processor_kwargs.sampling_rate=16000 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${WORLD_SIZE_NPU}" \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length=4160 \
    actor_rollout_ref.rollout.agent.num_workers="$((WORLD_SIZE_NPU / ROLLOUT_TP))" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py \
    reward.custom_reward_function.name=compute_score \
    trainer.balance_batch=true \
    trainer.project_name=qwen3_omni_avqa \
    trainer.experiment_name=gspo_avqa_veomni_npu \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_epochs=10 \
    trainer.total_training_steps=2 \
    "$@" \
    2>&1 | tee "${LOG_FILE}"
