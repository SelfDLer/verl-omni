#!/usr/bin/env bash
# VeOmni counterpart of the full-depth NExT-QA FSDP2 recipe.
# Source CANN/ATB first; see README.md for comparison and validation limits.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export TRAINING_BACKEND=veomni
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_omni_nextqa_fullparam_veomni}
RUN_TIME=${RUN_TIME:-$(date +%Y%m%d_%H%M%S)}
export LOG_FILE=${LOG_FILE:-run_qwen3omni_npu_nextqa_veomni_${RUN_TIME}.log}
bash "${SCRIPT_DIR}/run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh" "$@"
