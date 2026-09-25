#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}"
RUN_NAME="${RUN_NAME:-avqa_veomni_compare_$(date +%Y%m%d_%H%M%S)}"
export LOG_FILE="${REPO_ROOT}/debug/${RUN_NAME}.log"

# 沿用服务器已设置的 ASCEND_HOME_PATH。
# CANN/ATB 脚本加载完成后再开启 nounset。
source "${ASCEND_HOME_PATH}/set_env.sh"
source "${ASCEND_HOME_PATH}/../nnal/atb/set_env.sh"
set -euo pipefail

# Optional worker probe with explicit Ray propagation.
PROBE_ARGS=()
if [[ "${VEOMNI_MEMORY_PROBE:-0}" == "1" ]]; then
    : "${VEOMNI_MEMORY_PROBE_DIR:?Set an absolute server output path}"
    [[ "${VEOMNI_MEMORY_PROBE_DIR}" == /* ]] || exit 2
    PROBE_ARGS+=(
        "++ray_kwargs.ray_init.runtime_env.env_vars.VEOMNI_MEMORY_PROBE=\"1\""
        "++ray_kwargs.ray_init.runtime_env.env_vars.VEOMNI_MEMORY_PROBE_DIR=\"${VEOMNI_MEMORY_PROBE_DIR}\""
        "++ray_kwargs.ray_init.runtime_env.env_vars.VEOMNI_MEMORY_PROBE_RANKS=\"${VEOMNI_MEMORY_PROBE_RANKS:-all}\""
        "++ray_kwargs.ray_init.runtime_env.env_vars.VEOMNI_MEMORY_PROBE_MAX_CALLS=\"${VEOMNI_MEMORY_PROBE_MAX_CALLS:-3}\""
    )
fi

bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_veomni_npu_avqa_v1.sh \
    data.train_batch_size=16 \
    data.train_max_samples=-1 \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    data.val_max_samples=32 \
    data.validation_shuffle=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.prompt_length=8192 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=true \
    trainer.default_local_dir="${REPO_ROOT}/debug/checkpoints/${RUN_NAME}" \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.resume_mode=disable \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.veomni.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.veomni.moe_implementation=fused_npu \
    actor_rollout_ref.actor.veomni.enable_fsdp_offload=true \
    actor_rollout_ref.actor.veomni.param_offload=true \
    actor_rollout_ref.actor.veomni.optimizer_offload=true \
    trainer.total_training_steps=3 \
    trainer.val_before_train=false \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.logger='[console]' \
    global_profiler.tool=npu \
    'global_profiler.steps=[2]' \
    global_profiler.save_path="./outputs/profile/${RUN_NAME}" \
    actor_rollout_ref.actor.profiler.enable=true \
    actor_rollout_ref.actor.profiler.all_ranks=true \
    'actor_rollout_ref.actor.profiler.ranks=[0]' \
    actor_rollout_ref.actor.profiler.tool_config.npu.discrete=true \
    actor_rollout_ref.actor.profiler.tool_config.npu.level=level1 \
    'actor_rollout_ref.actor.profiler.tool_config.npu.contents=[npu,cpu,shapes,memory]' \
    actor_rollout_ref.actor.profiler.tool_config.npu.analysis=false \
    "${PROBE_ARGS[@]}" \
    "$@"
