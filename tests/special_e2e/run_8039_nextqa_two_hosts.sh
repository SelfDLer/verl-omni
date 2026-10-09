#!/usr/bin/env bash
# Real NExT-QA V1 validation on an existing one/two-host Ascend Ray cluster.
# No model/processor mocks. See repro_8039_two_hosts.md before running.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the intended head IP:port}"
: "${MODEL_PATH:?Set MODEL_PATH to the full Qwen3-Omni checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to NExT-QA train.parquet}"
: "${VAL_FILE:?Set VAL_FILE to NExT-QA validation.parquet}"

export NNODES=${NNODES:-2}
export N_GPUS_PER_NODE=16
export ROLLOUT_TP=4
export TRAIN_BATCH_SIZE=64
export ROLLOUT_N=8
export MODEL_PATH TRAIN_FILE VAL_FILE RAY_ADDRESS
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-8}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-32}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-8039-nextqa-${NNODES}hosts-$(date +%Y%m%d-%H%M%S)}
export OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/outputs/debug/${EXPERIMENT_NAME}"}
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

if [[ "$NNODES" != 1 && "$NNODES" != 2 ]]; then
    echo "NNODES must be 1 or 2." >&2
    exit 2
fi
for name in AGENT_NUM_WORKERS VAL_MAX_SAMPLES; do
    if [[ ! ${!name} =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer." >&2
        exit 2
    fi
done

command=(bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh
    "++ray_kwargs.ray_init.address=${RAY_ADDRESS}"
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=verl_omni"
    trainer.val_only=true trainer.val_before_train=true
    trainer.resume_mode=disable trainer.save_freq=-1
    data.train_max_samples=32
    "data.val_max_samples=${VAL_MAX_SAMPLES}" "data.val_batch_size=${VAL_MAX_SAMPLES}"
    data.validation_shuffle=false data.filter_overlong_prompts_workers=8
    "actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS}"
    actor_rollout_ref.rollout.val_kwargs.n=1
    'trainer.logger=[console]'
)
# Forward opt-in tracing to actors on both hosts, including an existing cluster.
if [[ -n ${VERL_OMNI_VIDEO_TRACE_DIR:-} ]]; then
    command+=(
        "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VIDEO_TRACE_DIR='${VERL_OMNI_VIDEO_TRACE_DIR}'"
        "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_VIDEO_TRACE_MAX_REQUESTS='${VERL_OMNI_VIDEO_TRACE_MAX_REQUESTS:-32}'"
    )
fi
# Extra Hydra overrides are intentionally last, as in the underlying recipe.
command+=("$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

[[ -d "$MODEL_PATH" && -f "$TRAIN_FILE" && -f "$VAL_FILE" ]] || {
    echo "Model directory and both parquet files must exist on the head and at the same paths on the worker." >&2
    exit 2
}
mkdir -p "$OUTPUT_DIR"
python3 - <<'PY' | tee "$OUTPUT_DIR/preflight.log"
import importlib.metadata as metadata
import os

import ray

ray.init(address=os.environ["RAY_ADDRESS"])
try:
    nodes = [node for node in ray.nodes() if node["Alive"]]
    expected = int(os.environ["NNODES"])
    if len(nodes) != expected:
        raise RuntimeError(f"Expected exactly {expected} live nodes; found {len(nodes)}")
    if len({node["NodeManagerAddress"] for node in nodes}) != expected:
        raise RuntimeError("Expected distinct host IPs, not multiple raylets on one machine")
    for node in nodes:
        print(node["NodeID"], node["NodeManagerAddress"], node["Resources"], flush=True)
        if node["Resources"].get("NPU", 0) != 16:
            raise RuntimeError("Each node must advertise 16 NPU resources")
    for name in ("verl", "verl-omni", "vllm", "vllm-omni", "vllm-ascend", "ray", "transformers"):
        dist = metadata.distribution(name)
        print(f"{name}={dist.version}", flush=True)
finally:
    ray.shutdown()
PY

# The inner recipe also tees its standard log; this outer log is per run.
"${command[@]}" 2>&1 | tee "$OUTPUT_DIR/driver.log"
