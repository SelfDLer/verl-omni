#!/usr/bin/env bash
# Ascend NPU: two updates exercise actor/ref, backward, weight reload and checkpoint save.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export MODEL_PATH=${MODEL_PATH:-"${HOME}/models/tiny-random/Qwen3-Omni-VeOmni"}
export TRAIN_FILE=${TRAIN_FILE:-"${HOME}/data/gsm8k/train.parquet"}
export VAL_FILE=${VAL_FILE:-"${HOME}/data/gsm8k/test.parquet"}
export NUM_NPUS=${NUM_NPUS:-2}
export NNODES=1
# The tiny model has two KV heads; use TP=1 so other NPU counts work too.
export ROLLOUT_TP=${ROLLOUT_TP:-1}
if [[ ! ${NUM_NPUS} =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_NPUS must be a positive integer, got '${NUM_NPUS}'." >&2
    exit 2
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    python3 tests/special_e2e/build_qwen3_omni_tiny_random.py --output-dir "${MODEL_PATH}"
fi

bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_veomni.sh \
    data.train_batch_size="$((NUM_NPUS * 2))" \
    data.max_prompt_length=256 \
    data.max_response_length=32 \
    data.val_max_samples=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$((NUM_NPUS * 2))" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=2 \
    "$@"
