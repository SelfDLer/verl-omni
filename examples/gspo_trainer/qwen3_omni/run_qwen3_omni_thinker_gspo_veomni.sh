#!/usr/bin/env bash
# Ascend NPU: Qwen3-Omni Thinker GSPO with VeOmni actor/ref and vLLM-Omni rollout.
# Run in docker/Dockerfile.a3.npu's runtime, with CANN/ATB sourced first.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
export VERL_USE_EXTERNAL_MODULES=verl_omni
export DEVICE_NAME=npu
export VLLM_ASCEND_ENABLE_NZ=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false

NUM_NPUS=${NUM_NPUS:-8}
NNODES=${NNODES:-1}
ROLLOUT_TP=${ROLLOUT_TP:-2}
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

MODEL_PATH=${MODEL_PATH:-"$HOME/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/gsm8k/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/gsm8k/test.parquet"}
# Start with SDPA + padded micro-batches on Ascend A3.
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-false}

exec python3 -m verl_omni.trainer.main_omni \
    actor@actor_rollout_ref.actor=omni_veomni_actor \
    ref@actor_rollout_ref.ref=omni_veomni_ref \
    actor_rollout_ref.actor._target_=verl_omni.workers.config.omni.OmniVeOmniActorConfig \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
    actor_rollout_ref.model.use_fused_kernels=false \
    actor_rollout_ref.model.use_liger=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.freeze_vision_tower=true \
    actor_rollout_ref.actor.veomni.freeze_audio_tower=true \
    actor_rollout_ref.actor.veomni.attn_implementation="${ATTN_IMPLEMENTATION}" \
    actor_rollout_ref.actor.veomni.param_offload=true \
    actor_rollout_ref.actor.veomni.optimizer_offload=true \
    actor_rollout_ref.actor.veomni.use_torch_compile=false \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0 \
    actor_rollout_ref.ref.veomni.use_torch_compile=false \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    trainer.val_before_train=false \
    trainer.logger='[console]' \
    trainer.project_name=gspo \
    trainer.experiment_name=qwen3_omni_thinker_veomni_npu \
    trainer.device=npu \
    trainer.n_gpus_per_node="${NUM_NPUS}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    "$@"
