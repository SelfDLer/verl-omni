#!/usr/bin/env bash
# One-step NPU ablations for rollout/actor probability parity in Qwen3-Omni.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
TRAIN_SCRIPT="${REPO_ROOT}/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "Training script not found: ${TRAIN_SCRIPT}" >&2
    exit 2
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR=${RESULT_DIR:-"${SCRIPT_DIR}/results/${TIMESTAMP}"}
DIAG_TOTAL_STEPS=${DIAG_TOTAL_STEPS:-1}
DIAG_TRAIN_BATCH_SIZE=${DIAG_TRAIN_BATCH_SIZE:-8}
DIAG_MAX_RESPONSE_LENGTH=${DIAG_MAX_RESPONSE_LENGTH:-1024}
BASE_ROLLOUT_TP=${ROLLOUT_TP:-2}

mkdir -p "${RESULT_DIR}"

if (( $# > 0 )); then
    CASES=("$@")
else
    CASES=(baseline no_audio eager no_rmpad)
fi

{
    echo "timestamp=$(date -Iseconds)"
    echo "repo_root=${REPO_ROOT}"
    echo "commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "cases=${CASES[*]}"
    echo "diag_total_steps=${DIAG_TOTAL_STEPS}"
    echo "diag_train_batch_size=${DIAG_TRAIN_BATCH_SIZE}"
    echo "diag_max_response_length=${DIAG_MAX_RESPONSE_LENGTH}"
    echo "base_rollout_tp=${BASE_ROLLOUT_TP}"
    python3 --version
    python3 -m pip show torch transformers verl vllm vllm-omni vllm-ascend 2>/dev/null || true
} >"${RESULT_DIR}/environment.txt" 2>&1

COMMON_OVERRIDES=(
    "data.train_batch_size=${DIAG_TRAIN_BATCH_SIZE}"
    "data.max_response_length=${DIAG_MAX_RESPONSE_LENGTH}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${DIAG_TRAIN_BATCH_SIZE}"
    "trainer.val_before_train=false"
    "trainer.save_freq=-1"
    "trainer.test_freq=-1"
    "trainer.total_epochs=1"
    "trainer.total_training_steps=${DIAG_TOTAL_STEPS}"
    'trainer.logger=["console"]'
)

overall_status=0

for case_name in "${CASES[@]}"; do
    use_audio=true
    rollout_tp=${BASE_ROLLOUT_TP}
    case_overrides=()

    case "${case_name}" in
        baseline)
            ;;
        no_audio)
            use_audio=false
            ;;
        eager)
            case_overrides+=("actor_rollout_ref.rollout.enforce_eager=true")
            ;;
        no_rmpad)
            case_overrides+=("actor_rollout_ref.model.use_remove_padding=false")
            ;;
        tp4)
            rollout_tp=4
            ;;
        no_audio_eager)
            use_audio=false
            case_overrides+=("actor_rollout_ref.rollout.enforce_eager=true")
            ;;
        *)
            echo "Unknown case '${case_name}'. Valid cases: baseline no_audio eager no_rmpad tp4 no_audio_eager" >&2
            overall_status=2
            continue
            ;;
    esac

    case_dir="${RESULT_DIR}/${case_name}"
    mkdir -p "${case_dir}"
    log_file="${case_dir}/run.log"

    echo "===== ${case_name}: audio=${use_audio}, rollout_tp=${rollout_tp}, overrides=${case_overrides[*]:-none} ====="
    set +e
    (
        cd "${REPO_ROOT}"
        USE_AUDIO_IN_VIDEO=${use_audio} \
        ROLLOUT_TP=${rollout_tp} \
        TOTAL_TRAINING_STEPS=${DIAG_TOTAL_STEPS} \
        LOG_FILE="${log_file}" \
        bash "${TRAIN_SCRIPT}" \
            "${COMMON_OVERRIDES[@]}" \
            "trainer.experiment_name=pearson_diag_${case_name}" \
            "${case_overrides[@]}"
    )
    case_status=$?
    set -e
    echo "${case_status}" >"${case_dir}/exit_code.txt"
    if (( case_status != 0 )); then
        echo "Case '${case_name}' failed with exit code ${case_status}; continuing with the remaining cases." >&2
        overall_status=1
    fi
done

python3 "${SCRIPT_DIR}/summarize_pearson.py" "${RESULT_DIR}" --csv "${RESULT_DIR}/summary.csv" || overall_status=1
echo "NPU diagnostic artifacts: ${RESULT_DIR}"
exit "${overall_status}"
