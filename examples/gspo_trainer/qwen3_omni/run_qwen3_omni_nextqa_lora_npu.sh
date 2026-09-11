#!/usr/bin/env bash
# Full-depth attention LoRA; share media, sampling and memory controls with full training.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-video2_nextqa_lora}
export LEARNING_RATE=${LEARNING_RATE:-3e-6}

exec bash "${SCRIPT_DIR}/run_qwen3_omni_nextqa_npu.sh" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.model.exclude_modules=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*" \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    "$@"
