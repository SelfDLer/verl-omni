#!/usr/bin/env bash
# Fixed actor inputs: fresh-process baseline capture, then candidate replay.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}"
: "${RS_AB_ROOT:?Set a new absolute server directory; fixtures stay on the server}"
[[ "${RS_AB_ROOT}" == /* && ! -e "${RS_AB_ROOT}" ]] || {
    echo 'RS_AB_ROOT must be absolute and must not already exist.' >&2; exit 2;
}
UPDATES="${RS_AB_UPDATES:-3}"
NATIVE="${RS_AB_NATIVE_OFFLOAD:-true}"
[[ "${UPDATES}" =~ ^[0-9]+$ && "${UPDATES}" -ge 2 ]] || exit 2
[[ "${NATIVE}" == true || "${NATIVE}" == false ]] || exit 2
# The audited scenario is one host, 16 ranks, SP=EP=1. Do not silently reshard fixtures.
[[ "${NNODES:-1}" == 1 && "${NUM_NPUS:-16}" == 16 ]] || exit 2
export NNODES=1 NUM_NPUS=16
mkdir -p "${RS_AB_ROOT}"
export VEOMNI_MEMORY_PROBE=0
for arm in off on; do
    mode=capture
    flag=0
    [[ "${arm}" == off ]] || { mode=replay; flag=1; }
    RUN_NAME="rs_ab_${arm}_$(date +%Y%m%d_%H%M%S)" \
    bash codex_test/veomni_memory/run_baseline.sh "$@" \
        trainer.total_training_steps="${UPDATES}" \
        trainer.resume_mode=disable \
        actor_rollout_ref.actor.veomni.enable_fsdp_offload="${NATIVE}" \
        actor_rollout_ref.actor.veomni.param_offload=true \
        actor_rollout_ref.actor.veomni.optimizer_offload=true \
        actor_rollout_ref.actor.veomni.ulysses_parallel_size=1 \
        actor_rollout_ref.actor.veomni.expert_parallel_size=1 \
        actor_rollout_ref.actor.profiler.enable=false \
        'global_profiler.steps=[]' \
        "++ray_kwargs.ray_init.runtime_env.env_vars.VEOMNI_MEMORY_PROBE=\"0\"" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.VEOMNI_LIMIT_RS_INFLIGHT=\"${flag}\"" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.VEOMNI_RS_AB_MODE=\"${mode}\"" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.VEOMNI_RS_AB_FIXTURES=\"${RS_AB_ROOT}/fixtures\"" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.VEOMNI_RS_AB_OUTPUT=\"${RS_AB_ROOT}/${arm}\"" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.VEOMNI_RS_AB_UPDATES=\"${UPDATES}\""
done
python codex_test/veomni_memory/compare_rs_ab.py \
    --off "${RS_AB_ROOT}/off" --on "${RS_AB_ROOT}/on" \
    --ranks 16 --updates "${UPDATES}" --warmup 1 --output "${RS_AB_ROOT}/brief.json"
