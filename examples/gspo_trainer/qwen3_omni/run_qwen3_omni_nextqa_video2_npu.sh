#!/usr/bin/env bash
# NExT-QA video + soundtrack -> text, Qwen3-Omni Thinker GSPO/LoRA on Ascend.
# Source the installed CANN/ATB environment before running this script.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO_ROOT"
export VERL_USE_EXTERNAL_MODULES=verl_omni
export VLLM_ASCEND_ENABLE_NZ=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false

MODEL_PATH=${MODEL_PATH:-"$HOME/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/nextqa/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/nextqa/validation.parquet"}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-16}
NNODES=${NNODES:-1}
ROLLOUT_TP=${ROLLOUT_TP:-2}

for name in N_GPUS_PER_NODE NNODES ROLLOUT_TP; do
    if [[ ! ${!name} =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer, got '${!name}'." >&2
        exit 2
    fi
done
if (( N_GPUS_PER_NODE % ROLLOUT_TP != 0 )); then
    echo "N_GPUS_PER_NODE must be divisible by ROLLOUT_TP (rollout replicas stay within a node)." >&2
    exit 2
fi

exec python3 -m verl_omni.trainer.main_omni \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=$((N_GPUS_PER_NODE * NNODES * 2)) \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    data.truncation=error \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=4 \
    data.seed=42 \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.nextqa_rl_dataset \
    data.custom_cls.name=NextQARLHFDataset \
    ++data.mm_processor_kwargs.use_audio_in_video=false \
    ++data.mm_processor_kwargs.sampling_rate=16000 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.model.exclude_modules=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((N_GPUS_PER_NODE * NNODES)) \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.cudagraph_capture_sizes='[1,2,4,8]' \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.agent.num_workers=$((N_GPUS_PER_NODE * NNODES / ROLLOUT_TP)) \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="${REPO_ROOT}/verl_omni/utils/reward_score/choice_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.device=npu \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=qwen3_omni_nextqa \
    trainer.experiment_name=video2_gspo_lora_npu \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=true \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=200 \
    "$@"
