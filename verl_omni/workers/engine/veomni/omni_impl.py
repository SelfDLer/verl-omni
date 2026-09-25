# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-Omni Thinker engine, extending verl's pinned VeOmni LM engine."""

import os

import torch
from veomni.arguments import MixedPrecisionConfig
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.auto import build_foundation_model
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.veomni.transformer_impl import (
    OmniSequenceShardCollator,
    VeOmniEngineWithLMHead,
    _build_ops_implementation_config,
)
from verl.workers.engine.veomni.utils import load_safetensors_index


def _packed_attention_kwargs(offsets: torch.Tensor, pad_size: int) -> dict:
    """Derive sample boundaries from token offsets, never from multimodal RoPE."""
    offsets = offsets.to(dtype=torch.int32)
    if pad_size:
        offsets = torch.cat((offsets, offsets[-1:] + pad_size))
    max_length = int(offsets.diff().max().item())
    return {
        "cu_seq_lens_q": offsets,
        "cu_seq_lens_k": offsets,
        "max_length_q": max_length,
        "max_length_k": max_length,
    }


@EngineRegistry.register(model_type="omni_model", backend=["veomni"], device=["cuda", "npu"])
class OmniVeOmniEngine(VeOmniEngineWithLMHead):
    """Use VeOmni's native Thinker model and verl's RL loss and weight export."""

    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config, **kwargs):
        if os.environ.get("MODELING_BACKEND", "veomni") != "veomni" or engine_config.force_use_huggingface:
            raise ValueError("Qwen3-Omni VeOmni requires MODELING_BACKEND=veomni and force_use_huggingface=false.")
        if model_config.architecture != "Qwen3OmniMoeForConditionalGeneration" or model_config.model_stage != "thinker":
            raise ValueError("The omni VeOmni engine currently supports Qwen3-Omni Thinker only.")
        if model_config.lora_rank or model_config.lora.get("rank", 0) or model_config.lora_adapter_path:
            raise ValueError("Qwen3-Omni VeOmni requires full-parameter training; LoRA is not supported.")
        if model_config.use_fused_kernels or model_config.use_liger:
            raise ValueError("Qwen3-Omni VeOmni requires use_fused_kernels=false and use_liger=false.")
        if engine_config.router_replay.mode != "disabled":
            raise ValueError("Qwen3-Omni VeOmni does not yet support router replay.")
        if engine_config.expert_parallel_size > 1 and engine_config.moe_implementation == "eager":
            raise ValueError("VeOmni expert parallelism requires a fused MoE implementation for the target device.")
        if engine_config.ulysses_parallel_size > 1 and not model_config.use_remove_padding:
            raise ValueError("VeOmni sequence parallelism requires model.use_remove_padding=true.")
        if model_config.use_remove_padding and "flash_attention" not in engine_config.attn_implementation:
            raise ValueError("Packed Qwen3-Omni inputs require a varlen flash_attention implementation.")
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config, **kwargs)

    def _build_model_optimizer(self):
        from verl_omni.pipelines.model_base import OmniModelBase

        self.model_adapter_cls = OmniModelBase.get_class_by_name(
            self.model_config.architecture,
            self.model_config.model_stage,
            getattr(self.model_config, "external_lib", None),
        )
        # Adapted from verl's VeOmniEngine: configure trainability before sharding.
        # Keep VeOmni's native forward (including its NPU wrapper) and split modules.
        precision = MixedPrecisionConfig(enable=self.engine_config.mixed_precision)
        module = build_foundation_model(
            config_path=self._get_model_config_path(),
            weights_path=self.model_config.local_path,
            torch_dtype="float32" if precision.enable else "bfloat16",
            ops_implementation=_build_ops_implementation_config(self.engine_config),
            init_device=self.engine_config.init_device,
        )
        module = self.model_adapter_cls.configure_veomni_model(module, self.model_config)
        for name, frozen in (
            ("visual", self.engine_config.freeze_vision_tower),
            ("audio_tower", self.engine_config.freeze_audio_tower),
        ):
            if frozen:
                getattr(module.thinker, name).requires_grad_(False)
        module.config.use_cache = False
        module.config.thinker_config.use_cache = False
        module = build_parallelize_model(
            module,
            init_device=self.engine_config.init_device,
            weights_path=self.model_config.local_path,
            enable_full_shard=self.engine_config.enable_full_shard,
            mixed_precision=precision,
            enable_gradient_checkpointing=self.model_config.enable_gradient_checkpointing,
            enable_fsdp_offload=self.engine_config.enable_fsdp_offload,
            basic_modules=sorted(set(module._no_split_modules) | set(self.engine_config.basic_modules)),
            enable_reentrant=self.engine_config.enable_reentrant,
            enable_forward_prefetch=self.engine_config.forward_prefetch,
            broadcast_model_weights_from_rank0=True,
            fqn_to_index_mapping=load_safetensors_index(self.model_config.local_path),
        )
        if self.engine_config.enable_fsdp_offload:
            # FSDP owns parameter/gradient placement; disable manual stage offload.
            self._is_offload_param = False
            self._is_offload_optimizer = False
            self._uses_fsdp2_cpu_offload_policy = True
            # Keep parameter shards on CPU and place only buffers on the compute device.
            compute_device = get_device_id()
            for submodule in module.modules():
                for name, buffer in submodule.named_buffers(recurse=False):
                    if buffer is not None:
                        submodule._buffers[name] = buffer.to(compute_device)
        self.module = module
        self.optimizer = None if self.engine_config.forward_only else self._build_optimizer(module)
        self.lr_scheduler = None if self.optimizer is None else self._build_lr_scheduler(self.optimizer)
        self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
            self.model_config.enable_activation_offload,
            self.model_config.enable_gradient_checkpointing,
            self.engine_config.activation_gpu_limit,
        )
        if os.environ.get("VEOMNI_LIMIT_RS_INFLIGHT", "0") != "0":
            from .rs_limiter import install as install_rs_limiter

            install_rs_limiter(self)
        if os.environ.get("VEOMNI_MEMORY_PROBE") == "1":
            from codex_test.veomni_memory.worker_probe import install

            install(self)

    def _apply_veomni_input_transforms(self, model_inputs, micro_batch):
        packed = tu.get_non_tensor_data(micro_batch, "use_remove_padding", default=True)
        if packed:
            full_ids = micro_batch["input_ids"].values().unsqueeze(0)
            global_length = model_inputs["input_ids"].shape[-1] * self.ulysses_sequence_parallel_size
            pad_size = global_length - full_ids.shape[-1]
            full_ids = torch.nn.functional.pad(full_ids, (0, pad_size), value=-1)
            model_inputs.update(_packed_attention_kwargs(micro_batch["input_ids"].offsets(), pad_size))
        else:
            full_ids = model_inputs["input_ids"]
        adapted_inputs = self.model_adapter_cls.prepare_veomni_inputs(model_inputs, full_ids, self.module.config)
        if not isinstance(adapted_inputs, dict):
            raise TypeError(
                f"OmniModelBase.prepare_veomni_inputs must return a dict, got {type(adapted_inputs).__name__}."
            )
        if adapted_inputs is not model_inputs:
            # The pinned parent's transform hook mutates its input dictionary.
            model_inputs.clear()
            model_inputs.update(adapted_inputs)
        if model_inputs.get("input_features") is not None:
            # VeOmni always computes in bf16 (mixed_precision controls fp32
            # master parameters); the audio conv does not cast HF float inputs.
            model_inputs["input_features"] = model_inputs["input_features"].to(torch.bfloat16)
        # The inherited forward_step passes use_cache=False explicitly.

        if self.use_ulysses_sp:
            collator = OmniSequenceShardCollator()
            # Token IDs are already sliced by verl; masks and grids remain global.
            for key, scale in (
                ("pixel_values", self.module.config.thinker_config.vision_config.spatial_merge_size**2),
                ("pixel_values_videos", self.module.config.thinker_config.vision_config.spatial_merge_size**2),
                ("input_features", 1),
            ):
                if model_inputs.get(key) is not None:
                    model_inputs[key] = collator.sp_slice(
                        collator.sp_padding(model_inputs[key], dim=0, pad_scale=scale), dim=0
                    )
            model_inputs["position_ids"] = collator.sp_slice(model_inputs["position_ids"], dim=-1)

    def get_per_tensor_param(self, **kwargs):
        """Export HF expert names through verl, using the rollout's bf16 dtype."""
        params, metadata = super().get_per_tensor_param(**kwargs)
        return (
            (name, tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor) for name, tensor in params
        ), metadata
