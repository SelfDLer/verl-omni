#!/usr/bin/env bash
# Full-depth Thinker GSPO training on NExT-QA, with all Thinker parameters trainable.
# Source CANN/ATB first. MODEL_PATH must point to the original full checkpoint.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-16}
export NNODES=${NNODES:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-2}
export MODEL_PATH=${MODEL_PATH:-"$HOME/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"}

for name in N_GPUS_PER_NODE NNODES ROLLOUT_TP; do
    if [[ ! ${!name} =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer, got '${!name}'." >&2
        exit 2
    fi
done
WORLD_SIZE_NPU=$((N_GPUS_PER_NODE * NNODES))
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((WORLD_SIZE_NPU * 4))}
ROLLOUT_N=${ROLLOUT_N:-8}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-500}
for name in TRAIN_BATCH_SIZE ROLLOUT_N TOTAL_TRAINING_STEPS; do
    if [[ ! ${!name} =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer, got '${!name}'." >&2
        exit 2
    fi
done
if (( TRAIN_BATCH_SIZE % WORLD_SIZE_NPU != 0 || ROLLOUT_N < 2 )); then
    echo "TRAIN_BATCH_SIZE must be divisible by the NPU count; GRPO needs ROLLOUT_N >= 2." >&2
    exit 2
fi
LEARNING_RATE=${LEARNING_RATE:-1e-6}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-video2_nextqa_fullparam}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/../../../outputs/${EXPERIMENT_NAME}"}

exec bash "${SCRIPT_DIR}/run_qwen3_omni_nextqa_video2_npu.sh" \
    data.train_max_samples=-1 \
    data.val_max_samples=-1 \
    data.shuffle=true \
    data.validation_shuffle=false \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.model.hf_config_path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.lora.merge=false \
    actor_rollout_ref.model.target_modules=null \
    actor_rollout_ref.model.exclude_modules=null \
    actor_rollout_ref.actor.freeze_vision_tower=false \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.do_sample=true \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.logprobs_mode=raw_logprobs \
    algorithm.rollout_correction.bypass_mode=false \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.rollout_correction.rollout_rs=null \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
    trainer.validation_data_dir="${OUTPUT_DIR}/validation" \
    trainer.log_val_generations=8 \
    trainer.resume_mode=auto \
    trainer.val_before_train=true \
    trainer.test_freq=25 \
    trainer.save_freq=25 \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.total_epochs=10 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    "$@"
