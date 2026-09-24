#!/usr/bin/env bash
# Ascend NPU VeOmni smoke: requires docker/Dockerfile.a3.npu's pinned runtime.
# Usage: NUM_NPUS=2 bash tests/npu_smoke/run_npu_smoke_omni_veomni.sh [Hydra overrides...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export NUM_NPUS=${NUM_NPUS:-2}
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
LOG_DIR=${LOG_DIR:-"${REPO_ROOT}/logs/npu_smoke/veomni_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "${LOG_DIR}"

echo "Qwen3-Omni VeOmni smoke on ${NUM_NPUS} NPUs; log: ${LOG_DIR}/veomni.log"
bash tests/special_e2e/run_qwen3_omni_veomni_smoke.sh "$@" 2>&1 | tee "${LOG_DIR}/veomni.log"
