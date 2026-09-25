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

"""Opt-in host backpressure for the audited VeOmni / NPU FSDP2 path."""

import functools
import hashlib
import importlib
import os
import time
from pathlib import Path


class RSLimiter:
    """Wait before the original post_backward can allocate another RS input."""

    def __init__(self):
        self.wait_ns = 0
        self.wait_calls = 0
        self.groups = []

    def wrap(self, group, check_eager=lambda: None):
        original = group.post_backward

        @functools.wraps(original)
        def post_backward(*args, **kwargs):
            check_eager()
            has_grad = group.reduce_grads and any(
                hasattr(p, "_unsharded_param")
                and (p.unsharded_accumulated_grad is not None or p.unsharded_param.grad is not None)
                for p in group.fsdp_params
            )
            # Do not keep an extra reference to ReduceScatterState across original().
            previous = group.comm_ctx.reduce_scatter_state
            event = previous.event if has_grad and previous is not None else None
            del previous
            if event is not None:
                start = time.perf_counter_ns()
                event.synchronize()
                self.wait_ns += time.perf_counter_ns() - start
                self.wait_calls += 1
            # Original code keeps its wait_event, clears the old input reference,
            # and issues the next collective. Never erase allocator stream records.
            return original(*args, **kwargs)

        group.post_backward = post_backward
        self.groups.append(group)


def audited_groups(engine):
    """Fail closed before touching private APIs on an unaudited backend/version."""
    import torch
    import torch_npu
    from torch.distributed.fsdp import FSDPModule

    if type(engine).__module__ != "verl_omni.workers.engine.veomni.omni_impl" or not torch.npu.is_available():
        raise RuntimeError("RS candidate supports only OmniVeOmniEngine on NPU")
    if torch.__version__.split("+")[0] != "2.10.0" or torch_npu.__version__.split("+")[0] != "2.10.0.post6":
        raise RuntimeError("RS candidate requires torch 2.10.0 / torch-npu 2.10.0.post6")
    sources = [
        (
            "torch.distributed.fsdp._fully_shard._fsdp_param_group",
            "0f81daf147e30914f9f96bd5e654c3bc3df7a5f4a0db37819ed76d3f00858cb6",
        ),
        (
            "torch.distributed.fsdp._fully_shard._fsdp_collectives",
            "f8527a45d7d8ae029e8b2ce78b9fd6fc909b95e4b3cf1be7cfa9ba3da722be31",
        ),
        (
            "torch_npu.distributed.fsdp._add_fsdp_patch",
            "afd6c1f10cb9c328352c1542235a06158aa7e632613ce4c7b9e87175ca0b6550",
        ),
        (
            "verl.workers.engine.veomni.transformer_impl",
            "65b02a6cdff46dadedf9f24fa30ee1a4e31b1facb7a51619d0028d18fc9dfe53",
        ),
    ]
    for name, expected in sources:
        module = importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"RS candidate source mismatch: {name}")
    patch = importlib.import_module("torch_npu.distributed.fsdp._add_fsdp_patch")
    if patch._FSDP_ENHANCE_PATCH_APPLIED:
        raise RuntimeError("RS candidate does not support the NPU FSDP enhance/cache patch")
    group_module = importlib.import_module("torch.distributed.fsdp._fully_shard._fsdp_param_group")
    groups = []
    for module in engine.module.modules():
        if isinstance(module, FSDPModule):
            group = module._get_fsdp_state()._fsdp_param_group
            if group is not None and all(group is not g for g in groups):
                if (
                    group.device.type != "npu"
                    or type(group._reduce_scatter_comm) is not group_module.DefaultReduceScatter
                    or getattr(group.post_backward, "__func__", None) is not group_module.FSDPParamGroup.post_backward
                    or group._all_reduce_hook is not None
                    or type(group.mesh_info) is not group_module.FSDPMeshInfo
                ):
                    raise RuntimeError("RS candidate requires unmodified 1-D FSDP2 groups on NPU")
                groups.append(group)
    if not groups:
        raise RuntimeError("RS candidate found no FSDP2 groups")
    return groups


def install(engine):
    """Install only on this engine's groups; unset/0 leaves all methods intact."""
    flag = os.environ.get("VEOMNI_LIMIT_RS_INFLIGHT", "0")
    if flag == "0":
        return None
    if flag != "1":
        raise ValueError("VEOMNI_LIMIT_RS_INFLIGHT must be 0 or 1")
    if engine.engine_config.forward_only:
        return None
    if hasattr(engine, "_veomni_rs_limiter"):
        return engine._veomni_rs_limiter
    groups = audited_groups(engine)
    from torch.distributed.fsdp._fully_shard._fsdp_common import compiled_autograd_enabled

    def check_eager():
        if compiled_autograd_enabled():
            raise RuntimeError("RS candidate requires eager autograd")

    check_eager()
    limiter = RSLimiter()
    for group in groups:
        limiter.wrap(group, check_eager)
    engine._veomni_rs_limiter = limiter
    return limiter
