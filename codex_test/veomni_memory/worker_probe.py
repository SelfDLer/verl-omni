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

"""Opt-in, worker-local metadata probe; never reads tensor values or synchronizes.

Writes one initialization record and bounded train-call records. Host timestamps
are host-only. This deliberately does not instrument every FSDP group invocation.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path


def source(obj):
    try:
        return {
            "module": obj.__module__,
            "qualname": obj.__qualname__,
            "path": inspect.getsourcefile(obj),
            "line": inspect.getsourcelines(obj)[1],
        }
    except (TypeError, OSError, AttributeError):
        return {"unavailable": type(obj).__name__}


def plain(value):
    # Never stringify tensors: __repr__ can cause a device read/synchronization.
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple) and len(value) <= 128:
        return [plain(v) for v in value]
    return {"type": type(value).__name__, "value": "not read"}


def storage(tensor, seen):
    """Count underlying LOCAL storage exactly once, including cross-role aliases."""
    if hasattr(tensor, "_local_tensor"):
        tensor = tensor._local_tensor
    item = {
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "logical_local_bytes": tensor.numel() * tensor.element_size(),
    }
    try:
        s = tensor.untyped_storage()
        key = (str(tensor.device), s.data_ptr(), s.nbytes())
        item["storage_bytes"] = s.nbytes()
        item["unique_bytes"] = 0 if key in seen else s.nbytes()
        item["alias_of_prior_storage"] = key in seen
        seen.add(key)
    except (RuntimeError, NotImplementedError):
        item["storage_bytes"] = item["unique_bytes"] = None
    return item


def snapshot(engine):
    import torch

    seen, totals, examples = set(), Counter(), []
    counts = Counter()

    def add(role, name, tensor):
        if not isinstance(tensor, torch.Tensor):
            return
        entry = storage(tensor, seen)
        counts[role] += 1
        if entry["unique_bytes"] is not None:
            totals[f"{role}|{entry['device']}|{entry['dtype']}"] += entry["unique_bytes"]
        else:
            counts["storage_unavailable"] += 1
        if counts[role] <= 3:
            examples.append({"role": role, "name": name, **entry})

    for name, param in engine.module.named_parameters():
        add("parameter", name, param)
        if param.grad is not None:
            add("gradient", name, param.grad)
    for name, buf in engine.module.named_buffers():
        add("buffer", name, buf)
    opt = engine.optimizer
    initialized = 0
    if opt is not None:
        for state in opt.state.values():
            if state:
                initialized += 1
            for key, value in state.items():
                add("optimizer_" + str(key), str(key), value)
    return {
        "counts": dict(counts),
        "unique_storage_bytes_by_role_device_dtype": dict(totals),
        "alias_accounting": "shared storage charged to first encountered role; do not sum logical bytes",
        "examples": examples,
        "optimizer_state_entries": initialized,
        "optimizer_state_status": "absent" if opt is None else "initialized" if initialized else "not_initialized",
        "note": "resident shards/state only; transient AG/RS buffers require group/profile evidence",
    }


def group_inventory(engine):
    groups = []
    try:
        from torch.distributed.fsdp import FSDPModule

        for name, module in engine.module.named_modules():
            if not isinstance(module, FSDPModule):
                continue
            state = module._get_fsdp_state()
            group = state._fsdp_param_group
            if group is None:
                continue
            params = group.fsdp_params
            seen = set()
            physical = sum(storage(p.sharded_param, seen)["unique_bytes"] or 0 for p in params)
            sample = params[0] if params else None
            pg = group._reduce_scatter_process_group
            shard_size = pg.size() if pg is not None else 1
            padded_grad_numel = sum(
                math.prod(p.padded_sharded_param_size) for p in params if p.sharded_param.requires_grad
            )
            groups.append(
                {
                    "module": name,
                    "parameter_count": len(params),
                    "local_storage_bytes": physical,
                    "trainable_parameters": sum(p.sharded_param.requires_grad for p in params),
                    "param_dtype": str(getattr(sample, "param_dtype", None)),
                    "reduce_dtype": str(getattr(sample, "reduce_dtype", None)),
                    "offload_to_cpu": getattr(sample, "offload_to_cpu", None),
                    "pin_memory": getattr(sample, "pin_memory", None),
                    "use_mem_cache": getattr(module, "_use_mem_cache", None),
                    "rs_use_mem_cache": getattr(group._reduce_scatter_comm, "_use_mem_cache", None),
                    "rs_world_size": shard_size,
                    "trainable_padded_local_numel": padded_grad_numel,
                    "rs_fp32_input_bytes_upper_bound": 4 * padded_grad_numel * shard_size,
                    "rs_fp32_output_bytes_upper_bound": 4 * padded_grad_numel,
                    "budget_note": "fp32 conditional upper bound; participating gradients may be fewer",
                    "manual_forward_prefetch_count": len(state._states_to_forward_prefetch),
                    "manual_backward_prefetch_count": len(state._states_to_backward_prefetch),
                }
            )
    except (AttributeError, ImportError, RuntimeError) as exc:
        groups.append({"capability_gap": str(exc)})
    return groups


def install(engine):
    if getattr(engine, "_memory_probe_installed", False):
        return
    import torch.distributed as dist
    from verl.utils import tensordict_utils as tu

    rank = dist.get_rank()
    ranks = os.environ.get("VEOMNI_MEMORY_PROBE_RANKS", "all")
    if ranks != "all" and str(rank) not in ranks.split(","):
        return
    out = os.environ.get("VEOMNI_MEMORY_PROBE_DIR")
    if not out or not Path(out).is_absolute():
        raise ValueError("VEOMNI_MEMORY_PROBE_DIR must be an absolute server path")
    engine._memory_probe_installed = True
    limit = int(os.environ.get("VEOMNI_MEMORY_PROBE_MAX_CALLS", "3"))
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".gitignore").write_text("*\n", encoding="utf-8")
    path = directory / f"worker_rank{rank}_pid{os.getpid()}.jsonl"
    context = {"call": 0, "record": None}

    def emit(record):
        with path.open("a", encoding="utf-8") as dst:
            dst.write(
                json.dumps(
                    {
                        "rank": rank,
                        "pid": os.getpid(),
                        "world_size": dist.get_world_size(),
                        "host_time_ns": time.time_ns(),
                        **record,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    config_keys = (
        "strategy",
        "fsdp_size",
        "ulysses_parallel_size",
        "expert_parallel_size",
        "mixed_precision",
        "forward_prefetch",
        "enable_fsdp_offload",
        "param_offload",
        "optimizer_offload",
        "forward_only",
        "micro_batch_size_per_gpu",
        "use_dynamic_bsz",
        "max_token_len_per_gpu",
        "attn_implementation",
        "moe_implementation",
        "freeze_vision_tower",
        "freeze_audio_tower",
        "enable_reentrant",
    )
    methods = {
        name: source(getattr(engine, name))
        for name in (
            "to",
            "train_mode",
            "eval_mode",
            "forward_backward_batch",
            "forward_step",
            "optimizer_step",
            "get_per_tensor_param",
        )
    }
    for mod, attr in (
        ("torch.distributed.fsdp._fully_shard._fsdp_param_group", "foreach_reduce"),
        ("torch.distributed.fsdp._fully_shard._fsdp_collectives", "foreach_reduce"),
        ("veomni.distributed.torch_parallelize", "fully_shard"),
    ):
        try:
            methods[mod + "." + attr] = source(getattr(importlib.import_module(mod), attr))
        except (ImportError, AttributeError) as exc:
            methods[mod] = {"error": str(exc)}
    try:
        mod = importlib.import_module("torch.distributed.fsdp._fully_shard._fsdp_param_group")
        methods["actual_finalize_backward"] = source(mod.FSDPParamGroup.finalize_backward)
        methods["actual_reduce_scatter_allocate"] = source(mod.DefaultReduceScatter.allocate)
    except (ImportError, AttributeError) as exc:
        methods["fsdp_method_gap"] = str(exc)
    emit(
        {
            "event": "installed",
            "config": {k: plain(getattr(engine.engine_config, k, None)) for k in config_keys},
            "worker_imports": {
                name: {
                    "version": plain(getattr(sys.modules.get(name), "__version__", None)),
                    "path": getattr(sys.modules.get(name), "__file__", None),
                }
                for name in ("torch", "torch_npu", "verl", "veomni")
            },
            "model_flags": {
                k: plain(getattr(engine.model_config, k, None))
                for k in ("enable_gradient_checkpointing", "enable_activation_offload", "use_remove_padding")
            },
            "mro": [f"{c.__module__}.{c.__qualname__}" for c in type(engine).__mro__],
            "methods": methods,
            "flags": {
                k: getattr(engine, k, None)
                for k in ("_is_offload_param", "_is_offload_optimizer", "_uses_fsdp2_cpu_offload_policy")
            },
            "dp_size": engine.get_data_parallel_size(),
            "dp_rank": engine.get_data_parallel_rank(),
            "storage": snapshot(engine),
            "groups": group_inventory(engine),
        }
    )

    original_forward = engine.forward_step

    @functools.wraps(original_forward)
    def forward(micro_batch, *args, **kwargs):
        rec = context["record"]
        if rec is not None:
            rec["micro_batches"].append({"local_samples": len(micro_batch), "host_enter_ns": time.perf_counter_ns()})
        return original_forward(micro_batch, *args, **kwargs)

    engine.forward_step = forward

    original_optimizer = engine.optimizer_step

    @functools.wraps(original_optimizer)
    def optimizer(*args, **kwargs):
        rec = context["record"]
        if rec is not None:
            rec["before_optimizer"] = snapshot(engine)
            rec["groups_after_backward"] = group_inventory(engine)
            start = time.perf_counter_ns()
        result = original_optimizer(*args, **kwargs)
        if rec is not None:
            rec["optimizer_host_duration_ns"] = time.perf_counter_ns() - start
            rec["after_optimizer"] = snapshot(engine)
        return result

    engine.optimizer_step = optimizer

    original_train = engine.train_batch

    @functools.wraps(original_train)
    def train(data, *args, **kwargs):
        context["call"] += 1
        if context["call"] > limit:
            return original_train(data, *args, **kwargs)
        rec = {
            "event": "train_batch",
            "local_train_call_index": context["call"],
            "global_step": None,
            "local_samples": len(data),
            "micro_batches": [],
            "metadata": {
                k: plain(tu.get_non_tensor_data(data, k, default=None))
                for k in (
                    "global_steps",
                    "global_step",
                    "global_batch_size",
                    "micro_batch_size_per_gpu",
                    "sp_size",
                    "use_dynamic_bsz",
                    "mini_batch_size",
                    "force_group_size",
                )
            },
        }
        context["record"] = rec
        start = time.perf_counter_ns()
        try:
            return original_train(data, *args, **kwargs)
        finally:
            rec["host_duration_ns"] = time.perf_counter_ns() - start
            rec["actual_micro_batch_count"] = len(rec["micro_batches"])
            context["record"] = None
            emit(rec)

    engine.train_batch = train


def main():
    import argparse

    p = argparse.ArgumentParser(description="Verify worker-local probe handshakes after an OPTIONAL run")
    p.add_argument("--worker-dir", type=Path, required=True)
    p.add_argument("--expected-ranks", type=int, required=True)
    a = p.parse_args()
    installed, trained, counts = set(), set(), {}
    for path in a.worker_dir.glob("worker_rank*_pid*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["event"] == "installed":
                installed.add(rec["rank"])
            elif rec["event"] == "train_batch":
                trained.add(rec["rank"])
                counts.setdefault(str(rec["rank"]), []).append(rec["actual_micro_batch_count"])
    expected = set(range(a.expected_ranks))
    result = {
        "installed_ranks": sorted(installed),
        "trained_ranks": sorted(trained),
        "micro_batches": counts,
        "missing_installed": sorted(expected - installed),
        "missing_trained": sorted(expected - trained),
    }
    (a.worker_dir / "worker_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if installed == trained == expected else 2)


if __name__ == "__main__":
    main()
