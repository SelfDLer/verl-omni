#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO on MMK12: VeOmni FSDP2/EP + vLLM-Omni on Ascend.
# Run from the repository root after activating the CANN/ATB environment.
# Install the NPU stack and VeOmni 0.1.12; see this directory's README.md.
# MODEL_PATH, TRAIN_FILE, VAL_FILE and REWARD_FUNCTION_PATH follow the GPU recipe.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_TIME=${RUN_TIME:-$(date +%Y%m%d_%H%M%S)}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:-2}
export NUM_GPUS=${NUM_GPUS:-16}
export NNODES=${NNODES:-1}
export ACTOR_EP=${ACTOR_EP:-8}
export ROLLOUT_TP=${ROLLOUT_TP:-2}
export ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
export MOE_IMPL=${MOE_IMPL:-fused_npu}

for name in NUM_GPUS NNODES ACTOR_EP ROLLOUT_TP; do
    if [[ ! ${!name} =~ ^[1-9][0-9]*$ ]]; then
        echo "${name} must be a positive integer, got '${!name}'." >&2
        exit 1
    fi
done
if (( NUM_GPUS * NNODES % ACTOR_EP != 0 || NUM_GPUS % ROLLOUT_TP != 0 )); then
    echo "ACTOR_EP must divide the total NPU count; ROLLOUT_TP must divide NUM_GPUS." >&2
    exit 1
fi

bash "${SCRIPT_DIR}/run_qwen3_omni_thinker_gspo_veomni.sh" \
    trainer.device=npu \
    trainer.experiment_name=qwen3_omni_thinker_veomni_mmk12_npu \
    data.max_prompt_length=4096 \
    actor_rollout_ref.rollout.prompt_length=4096 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.compilation_config.cudagraph_mode=PIECEWISE \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0 \
    actor_rollout_ref.actor.veomni.use_torch_compile=false \
    actor_rollout_ref.ref.veomni.use_torch_compile=false \
    actor_rollout_ref.actor.veomni.cross_entropy_loss_implementation=npu \
    actor_rollout_ref.actor.veomni.rms_norm_implementation=npu \
    actor_rollout_ref.actor.veomni.swiglu_mlp_implementation=eager \
    actor_rollout_ref.actor.veomni.rotary_pos_emb_implementation=npu \
    actor_rollout_ref.actor.veomni.load_balancing_loss_implementation=eager \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ENABLE_NZ='${VLLM_ASCEND_ENABLE_NZ}'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.MULTI_STREAM_MEMORY_REUSE='${MULTI_STREAM_MEMORY_REUSE}'" \
    "$@" \
    2>&1 | tee "run_qwen3omni_npu_mmk12_veomni_${RUN_TIME}.log"
