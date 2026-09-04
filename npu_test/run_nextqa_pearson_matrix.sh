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
DIAG_TRAIN_MAX_SAMPLES=${DIAG_TRAIN_MAX_SAMPLES:-64}
DIAG_VAL_MAX_SAMPLES=${DIAG_VAL_MAX_SAMPLES:-8}
DIAG_MAX_RESPONSE_LENGTH=${DIAG_MAX_RESPONSE_LENGTH:-1024}
BASE_ROLLOUT_TP=${ROLLOUT_TP:-2}
BASE_EXTERNAL_MODULES=${VERL_USE_EXTERNAL_MODULES:-verl_omni}

mkdir -p "${RESULT_DIR}"

if (( $# > 0 )); then
    CASES=("$@")
else
    CASES=(baseline no_audio no_init_sync tp4 low_concurrency batch_invariant)
fi

{
    echo "timestamp=$(date -Iseconds)"
    echo "repo_root=${REPO_ROOT}"
    echo "commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "cases=${CASES[*]}"
    echo "diag_total_steps=${DIAG_TOTAL_STEPS}"
    echo "diag_train_batch_size=${DIAG_TRAIN_BATCH_SIZE}"
    echo "diag_train_max_samples=${DIAG_TRAIN_MAX_SAMPLES}"
    echo "diag_val_max_samples=${DIAG_VAL_MAX_SAMPLES}"
    echo "diag_max_response_length=${DIAG_MAX_RESPONSE_LENGTH}"
    echo "base_rollout_tp=${BASE_ROLLOUT_TP}"
    python3 --version
    python3 -m pip show torch transformers verl vllm vllm-omni vllm-ascend 2>/dev/null || true
} >"${RESULT_DIR}/environment.txt" 2>&1

COMMON_OVERRIDES=(
    "data.train_batch_size=${DIAG_TRAIN_BATCH_SIZE}"
    "data.train_max_samples=${DIAG_TRAIN_MAX_SAMPLES}"
    "data.val_max_samples=${DIAG_VAL_MAX_SAMPLES}"
    "data.max_response_length=${DIAG_MAX_RESPONSE_LENGTH}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${DIAG_TRAIN_BATCH_SIZE}"
    "trainer.val_before_train=false"
    "trainer.save_freq=-1"
    "trainer.test_freq=-1"
    "trainer.total_epochs=1"
    "trainer.total_training_steps=${DIAG_TOTAL_STEPS}"
    "trainer.resume_mode=disable"
    'trainer.logger=["console"]'
)

overall_status=0

for case_name in "${CASES[@]}"; do
    use_audio=true
    rollout_tp=${BASE_ROLLOUT_TP}
    external_modules=${BASE_EXTERNAL_MODULES}
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
        no_init_sync)
            external_modules="${BASE_EXTERNAL_MODULES},npu_test.skip_initial_weight_sync"
            case_overrides+=("trainer.v1.trainer_mode=omni_sync_skip_initial_weight_sync")
            ;;
        no_init_sync_eager)
            external_modules="${BASE_EXTERNAL_MODULES},npu_test.skip_initial_weight_sync"
            case_overrides+=("trainer.v1.trainer_mode=omni_sync_skip_initial_weight_sync")
            case_overrides+=("actor_rollout_ref.rollout.enforce_eager=true")
            ;;
        no_sleep_no_reload)
            external_modules="${BASE_EXTERNAL_MODULES},npu_test.skip_initial_weight_sync"
            case_overrides+=("trainer.v1.trainer_mode=omni_sync_skip_initial_weight_sync_pearson_only")
            case_overrides+=("actor_rollout_ref.rollout.free_cache_engine=false")
            case_overrides+=("trainer.total_training_steps=1")
            ;;
        no_sleep_no_reload_eager)
            external_modules="${BASE_EXTERNAL_MODULES},npu_test.skip_initial_weight_sync"
            case_overrides+=("trainer.v1.trainer_mode=omni_sync_skip_initial_weight_sync_pearson_only")
            case_overrides+=("actor_rollout_ref.rollout.free_cache_engine=false")
            case_overrides+=("trainer.total_training_steps=1")
            case_overrides+=("actor_rollout_ref.rollout.enforce_eager=true")
            ;;
        no_sleep_reload)
            external_modules="${BASE_EXTERNAL_MODULES},npu_test.skip_initial_weight_sync"
            case_overrides+=("trainer.v1.trainer_mode=omni_sync_skip_initial_sleep_pearson_only")
            case_overrides+=("actor_rollout_ref.rollout.free_cache_engine=false")
            case_overrides+=("trainer.total_training_steps=1")
            ;;
        no_sleep_reload_eager)
            external_modules="${BASE_EXTERNAL_MODULES},npu_test.skip_initial_weight_sync"
            case_overrides+=("trainer.v1.trainer_mode=omni_sync_skip_initial_sleep_pearson_only")
            case_overrides+=("actor_rollout_ref.rollout.free_cache_engine=false")
            case_overrides+=("trainer.total_training_steps=1")
            case_overrides+=("actor_rollout_ref.rollout.enforce_eager=true")
            ;;
        tp4)
            rollout_tp=4
            ;;
        low_concurrency)
            case_overrides+=("actor_rollout_ref.rollout.max_num_seqs=8")
            ;;
        batch_invariant)
            case_overrides+=("actor_rollout_ref.rollout.full_determinism=true")
            ;;
        no_audio_eager)
            use_audio=false
            case_overrides+=("actor_rollout_ref.rollout.enforce_eager=true")
            ;;
        *)
            echo "Unknown case '${case_name}'." >&2
            echo "Valid cases: baseline no_audio eager no_rmpad no_init_sync no_init_sync_eager" >&2
            echo "             no_sleep_no_reload no_sleep_no_reload_eager no_sleep_reload" >&2
            echo "             no_sleep_reload_eager tp4 low_concurrency batch_invariant" >&2
            echo "             no_audio_eager" >&2
            overall_status=2
            continue
            ;;
    esac

    # Explicitly propagate external modules and repository path to Ray workers.
    case_overrides+=(
        "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=\"${external_modules}\""
        "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=\"${REPO_ROOT}:${PYTHONPATH:-}\""
    )

    case_dir="${RESULT_DIR}/${case_name}"
    mkdir -p "${case_dir}"
    log_file="${case_dir}/run.log"

    echo "[RUNNING] ${case_name}: audio=${use_audio}, rollout_tp=${rollout_tp}"
    echo "          log=${log_file}"

    set +e
    (
        cd "${REPO_ROOT}"
        USE_AUDIO_IN_VIDEO=${use_audio} \
        ROLLOUT_TP=${rollout_tp} \
        VERL_USE_EXTERNAL_MODULES=${external_modules} \
        TOTAL_TRAINING_STEPS=${DIAG_TOTAL_STEPS} \
        LOG_FILE=/dev/null \
        PYTHONUNBUFFERED=1 \
        bash "${TRAIN_SCRIPT}" \
            "${COMMON_OVERRIDES[@]}" \
            "trainer.experiment_name=pearson_diag_${case_name}" \
            "${case_overrides[@]}"
    ) >"${log_file}" 2>&1
    case_status=$?
    set -e

    echo "${case_status}" >"${case_dir}/exit_code.txt"

    if (( case_status == 0 )); then
        echo "[PASSED]  ${case_name}"
    else
        echo "[FAILED]  ${case_name}, exit_code=${case_status}"
        echo "          see ${log_file}"
        overall_status=1
    fi
done

python3 "${SCRIPT_DIR}/summarize_pearson.py" "${RESULT_DIR}" --csv "${RESULT_DIR}/summary.csv" || overall_status=1
echo "NPU diagnostic artifacts: ${RESULT_DIR}"
exit "${overall_status}"
