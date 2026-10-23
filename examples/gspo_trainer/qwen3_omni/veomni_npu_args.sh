#!/usr/bin/env bash
# Sourced by the AVQA/NExT-QA recipes after validating their NPU topology.
# Keep data, rollout, rewards and training budgets in the dataset recipe.
ACTOR_EP=${ACTOR_EP:-1}
if [[ ! $ACTOR_EP =~ ^[1-9][0-9]*$ ]] || (( WORLD_SIZE_NPU % ACTOR_EP != 0 )); then
    echo "ACTOR_EP must be positive and divide the total NPU count." >&2
    exit 2
fi
export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:-2}
BACKEND_ARGS=(
    model_engine=veomni
    actor_rollout_ref.actor._target_=verl_omni.workers.config.omni.OmniVeOmniActorConfig
    actor_rollout_ref.model.use_remove_padding=true
    actor_rollout_ref.model.use_fused_kernels=true
    actor_rollout_ref.rollout.layered_summon=false
    actor_rollout_ref.actor.veomni.expert_parallel_size="${ACTOR_EP}"
    actor_rollout_ref.actor.veomni.ulysses_parallel_size=1
    actor_rollout_ref.actor.veomni.attn_implementation=flash_attention_2
    actor_rollout_ref.actor.veomni.moe_implementation=fused_npu
    actor_rollout_ref.actor.veomni.cross_entropy_loss_implementation=npu
    actor_rollout_ref.actor.veomni.rms_norm_implementation=npu
    actor_rollout_ref.actor.veomni.swiglu_mlp_implementation=eager
    actor_rollout_ref.actor.veomni.rotary_pos_emb_implementation=npu
    actor_rollout_ref.actor.veomni.load_balancing_loss_implementation=eager
    actor_rollout_ref.actor.veomni.param_offload=true
    actor_rollout_ref.actor.veomni.optimizer_offload=true
    actor_rollout_ref.actor.veomni.use_torch_compile=false
    actor_rollout_ref.actor.veomni.entropy_from_logits_with_chunking=true
    ++actor_rollout_ref.actor.veomni.entropy_from_logits_chunk_size=2048
    actor_rollout_ref.ref.veomni.expert_parallel_size="${ACTOR_EP}"
    actor_rollout_ref.ref.veomni.ulysses_parallel_size=1
    actor_rollout_ref.ref.veomni.param_offload=true
    actor_rollout_ref.ref.veomni.use_torch_compile=false
    "++ray_kwargs.ray_init.runtime_env.env_vars.MULTI_STREAM_MEMORY_REUSE='${MULTI_STREAM_MEMORY_REUSE}'"
)
for op in attn moe cross_entropy_loss rms_norm swiglu_mlp rotary_pos_emb load_balancing_loss; do
    BACKEND_ARGS+=("actor_rollout_ref.ref.veomni.${op}_implementation=\${actor_rollout_ref.actor.veomni.${op}_implementation}")
done
