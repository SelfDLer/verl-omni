#!/usr/bin/env bash
# Depth-reduced, full-parameter NPU actor/rollout consistency debug.
# Reuse the training recipe so model, media and reward behavior stay identical.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export NNODES=${NNODES:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-2}

SOURCE_MODEL_PATH=${MODEL_PATH:-"$HOME/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"}
DEBUG_CONFIG=${DEBUG_CONFIG:-"${SCRIPT_DIR}/qwen3_omni_4layer.json"}
DEBUG_MODEL_PATH=${DEBUG_MODEL_PATH:-"${SOURCE_MODEL_PATH}-debug4"}
DEBUG_OUTPUT_DIR=${DEBUG_OUTPUT_DIR:-"${SCRIPT_DIR}/../../../outputs/nextqa_consistency"}

# Config/help inspection must not trigger checkpoint export or model loading.
PREPARE_MODEL=true
for arg in "$@"; do
    case "$arg" in --cfg|--cfg=*|--help|-h) PREPARE_MODEL=false ;; esac
done
if [[ "$PREPARE_MODEL" == true ]]; then
    python3 "${SCRIPT_DIR}/prepare_debug_model.py" \
        --source "$SOURCE_MODEL_PATH" --config "$DEBUG_CONFIG" --output "$DEBUG_MODEL_PATH"
fi
export MODEL_PATH="$DEBUG_MODEL_PATH"

exec bash "${SCRIPT_DIR}/../qwen3_omni/run_qwen3_omni_nextqa_video2_npu.sh" \
    data.train_max_samples=32 \
    data.val_max_samples=4 \
    data.shuffle=false \
    data.validation_shuffle=false \
    data.filter_overlong_prompts_workers=1 \
    data.dataloader_num_workers=0 \
    data.max_response_length=128 \
    actor_rollout_ref.model.hf_config_path="${DEBUG_MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${DEBUG_MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.lora.merge=false \
    actor_rollout_ref.model.target_modules=null \
    actor_rollout_ref.model.exclude_modules=null \
    actor_rollout_ref.actor.freeze_vision_tower=false \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.logprobs_mode=raw_logprobs \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.max_num_seqs=2 \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.cudagraph_capture_sizes='[1,2]' \
    algorithm.rollout_correction.bypass_mode=false \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.rollout_correction.rollout_rs=null \
    ++trainer.consistency_debug_dir="${DEBUG_OUTPUT_DIR}" \
    trainer.logger='["console"]' \
    trainer.experiment_name=video2_npu_fullparam_consistency \
    trainer.resume_mode=disable \
    trainer.val_before_train=false \
    trainer.total_epochs=2 \
    trainer.total_training_steps=2 \
    trainer.test_freq=2 \
    trainer.save_freq=-1 \
    "$@"
