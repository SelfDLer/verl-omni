#!/usr/bin/env bash
# VeOmni counterpart of the full-parameter AVQA FSDP2 recipe.
# The dataset launcher sources CANN/ATB; see README.md for validation limits.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export TRAINING_BACKEND=veomni
RUN_TIME=${RUN_TIME:-$(date +%Y%m%d_%H%M%S)}
export LOG_FILE=${LOG_FILE:-run_qwen3omni_npu_avqa_veomni_${RUN_TIME}.log}
bash "${SCRIPT_DIR}/run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh" \
    trainer.experiment_name=gspo_avqa_npu_veomni \
    "$@"
