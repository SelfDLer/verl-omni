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
"""Probe uneven FSDP2 bias shards with torchrun, without a large model or Ray server."""

import argparse
import hashlib
import inspect
import json
import math
import os
import runpy
import socket
from collections import Counter
from datetime import timedelta
from pathlib import Path


def shard_span(length, size, rank):
    """Return the contiguous Shard(0) offset, actual length and padded length."""
    if length < 1 or size < 1 or not 0 <= rank < size:
        raise ValueError("invalid length, shard size or rank")
    padded = (length + size - 1) // size
    start = min(rank * padded, length)
    return start, min(padded, length - start), padded


def compare_values(actual, expected, start=0):
    """Compare every element, keeping bounded examples with global coordinates."""
    if len(actual) != len(expected):
        return {"status": "shape_mismatch", "actual_size": len(actual), "expected_size": len(expected)}
    bad = [i for i, (a, b) in enumerate(zip(actual, expected, strict=True)) if a != b or not math.isfinite(a)]
    return {
        "status": "different" if bad else "equal",
        "numel": len(expected),
        "different": len(bad),
        "first_global": start + bad[0] if bad else None,
        "last_global": start + bad[-1] if bad else None,
        "examples": [{"global": start + i, "actual": actual[i], "expected": expected[i]} for i in bad[:6]],
    }


def source_info(function):
    """Identify the installed implementation used by the probe."""
    source = inspect.getsource(function)
    return {"file": inspect.getsourcefile(function), "sha256": hashlib.sha256(source.encode()).hexdigest()}


def compact_stage(stage, checks):
    """Keep full counts and bounded failure details for cross-rank reporting."""
    failures = [{"parameter": name, **check} for name, check in checks.items() if check["status"] != "equal"]
    return {
        "stage": stage,
        "counts": dict(Counter(check["status"] for check in checks.values())),
        "failures": failures[:3],
        "failures_omitted": max(0, len(failures) - 3),
    }


def format_summary(reports):
    """Produce a shareable summary capped at 64 KiB."""
    lines = [
        "8039 FSDP2 bias probe v1 (complete vectors, exact numeric comparison)",
        "This isolates small tensors; passing does not clear the full actor/rollout lifecycle.",
        "export_unobserved has no intermediate tensor reads between load and export.",
        "Other cases synchronize for observations and can change timing.",
    ]
    stages = {}
    failures = []
    for report in sorted(reports, key=lambda value: value["rank"]):
        for row in report["stages"]:
            stages.setdefault(row["stage"], Counter()).update(row["counts"])
            if any(status != "equal" and count for status, count in row["counts"].items()):
                failures.append((report["rank"], row))
    lines.extend(f"{stage}: {dict(counts)}" for stage, counts in stages.items())
    if not failures:
        lines.append("All collected checks passed. No reproduction in these cases; do not infer a production fix.")
    for rank, row in failures[:64]:
        lines.append(f"rank={rank} {row['stage']}: {json.dumps(row, ensure_ascii=True, separators=(',', ':'))}")
    if len(failures) > 64:
        lines.append(f"Additional failing rank/stages omitted: {len(failures) - 64}; see rank JSONL files.")
    result = "\n".join(lines) + "\n"
    if len(result.encode()) > 64 * 1024:
        result = result.encode()[: 63 * 1024].decode(errors="ignore") + "\nTRUNCATED: see rank JSONL files.\n"
    return result


def run(args):
    """Run independent export, observed lifecycle and collective controls."""
    import torch
    import torch.distributed as dist

    if args.device == "npu":
        import torch_npu

    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor, Shard

    local_rank = int(os.environ["LOCAL_RANK"])
    accelerator = getattr(torch, args.device)
    accelerator.set_device(local_rank)
    device = torch.device(args.device, local_rank)
    dist.init_process_group("hccl" if args.device == "npu" else "nccl", timeout=timedelta(seconds=args.timeout))
    rank, world = dist.get_rank(), dist.get_world_size()
    size = args.fsdp_size or world
    if size > world or world % size:
        raise ValueError("--fsdp-size must divide WORLD_SIZE")
    mesh = init_device_mesh(
        args.device,
        (world,) if size == world else (world // size, size),
        mesh_dim_names=("fsdp",) if size == world else ("ddp", "fsdp"),
    )
    shard_mesh = mesh if mesh.ndim == 1 else mesh["fsdp"]
    shard_rank = shard_mesh.get_local_rank()
    dtype = getattr(torch, args.dtype)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"bias-probe-{socket.gethostname()}-rank{rank}.jsonl"
    # Separate directories per invocation make stale records explicit instead of merging runs.
    stream = path.open("x", encoding="utf-8")
    report = {"rank": rank, "host": socket.gethostname(), "stages": []}

    def emit(event, **values):
        stream.write(json.dumps({"event": event, "rank": rank, **values}, ensure_ascii=True) + "\n")
        stream.flush()

    def record(stage, tensors, local=False):
        # Module.to('cpu', non_blocking=True) may not have completed its host copy.
        accelerator.synchronize()
        checks = {}
        for name, tensor in tensors.items():
            expected = reference[name]
            start, count, padded = shard_span(expected.numel(), size, shard_rank)
            if local:
                if not isinstance(tensor, DTensor):
                    raise TypeError(f"expected DTensor for {name}, got {type(tensor)}")
                tensor = tensor.to_local()
                target = expected[start : start + count]
            else:
                target = expected
            check = compare_values(
                tensor.detach().float().cpu().reshape(-1).tolist(), target.float().tolist(), start if local else 0
            )
            check.update(device=str(tensor.device), shape=list(tensor.shape))
            if local:
                check.update(shard_rank=shard_rank, global_start=start, local_length=count, padded_length=padded)
            checks[name] = check
        emit("check", stage=stage, parameters=checks)
        report["stages"].append(compact_stage(stage, checks))

    try:
        if args.loader == "verl":
            from verl.utils.fsdp_utils import (
                fsdp2_load_full_state_dict,
                load_fsdp_model_to_gpu,
                offload_fsdp_model_to_cpu,
            )

            def load(model, state):
                fsdp2_load_full_state_dict(model, state, mesh, None)

            to_device = load_fsdp_model_to_gpu
            to_cpu = offload_fsdp_model_to_cpu
            functions = [fsdp2_load_full_state_dict, to_device, to_cpu, set_model_state_dict, fully_shard]
        else:

            def load(model, state):
                if rank == 0:
                    model.to(device, non_blocking=True)
                else:
                    model.to_empty(device=device)
                set_model_state_dict(
                    model, state, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
                )

            def to_device(model):
                model.to(device, non_blocking=True)

            def to_cpu(model):
                model.to("cpu", non_blocking=True)
                accelerator.empty_cache()

            functions = [set_model_state_dict, fully_shard]

        reference, checkpoint_keys = {}, {}
        if args.checkpoint:
            audit = runpy.run_path(Path(__file__).with_name("audit_8039_vision_bias.py"))
            reader = audit["CheckpointBias"](args.checkpoint)
        for index in range(args.blocks):
            name = f"layers.b{index:02d}.bias"
            if args.checkpoint:
                item = reader.read(f"blocks.{index}.mlp.linear_fc1.bias")
                if len(item["values"]) != 4304:
                    raise ValueError(f"expected 4304-element fc1 bias, got {len(item['values'])}")
                reference[name] = torch.tensor(item["values"], dtype=dtype)
                checkpoint_keys[name] = item["key"]
            else:
                reference[name] = ((torch.arange(4304) % 127 - 63).float() / 16 - index / 16).to(dtype)
        for length in (4096, 4320):
            reference[f"layers.control{length}.bias"] = ((torch.arange(length) % 127 - 63).float() / 16).to(dtype)
        runtime = {
            "world_size": world,
            "fsdp_size": size,
            "shard_rank": shard_rank,
            "device": str(device),
            "torch": torch.__version__,
            "torch_npu": getattr(torch, "npu", None) is not None,
            "loader": args.loader,
            "dtype": args.dtype,
            "checkpoint": str(args.checkpoint),
            "checkpoint_keys": checkpoint_keys,
            "sources": {f.__name__: source_info(f) for f in functions},
        }
        if args.device == "npu":
            runtime["torch_npu"] = torch_npu.__version__
        emit("runtime", **runtime)

        class Bias(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.bias = torch.nn.Parameter(value)

            def forward(self, zero):
                return self.bias + zero

        class Bank(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleDict(
                    {
                        name.split(".")[1]: Bias(
                            value.clone() if shard_rank == 0 else torch.empty_like(value, device="meta")
                        )
                        for name, value in reference.items()
                    }
                )

            def forward(self, zero):
                return {f"layers.{name}.bias": layer(zero) for name, layer in self.layers.items()}

        def loaded_bank():
            model = Bank()
            state = model.state_dict()
            policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
            for layer in model.layers.values():
                fully_shard(layer, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
            fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
            load(model, state)
            return model

        def gather(state):
            return {name: value.to(device, non_blocking=True).full_tensor() for name, value in state.items()}

        with torch.no_grad():
            # Run the export sequence first, without diagnostic barriers/reads inside it.
            emit("begin", case="export_unobserved")
            model = loaded_bank()
            # Actor param_offload stores the module on CPU before the export loads it again.
            to_cpu(model)
            to_device(model)
            saved = model.state_dict()
            to_cpu(model)
            record("export_unobserved.result", gather(saved))
            del saved, model

            # Independent fresh model so added observations cannot repair the first case.
            emit("begin", case="lifecycle_observed")
            model = loaded_bank()
            record("load.local", model.state_dict(), local=True)
            record("load.full_tensor", gather(model.state_dict()))
            to_cpu(model)
            record("initial_offload.local", model.state_dict(), local=True)
            to_device(model)
            record("reload.local", model.state_dict(), local=True)
            saved = model.state_dict()
            to_cpu(model)
            record("export.offloaded_local", model.state_dict(), local=True)
            record("export.retained_local", saved, local=True)
            record("export.retained_full_tensor", gather(saved))
            record("export.fresh_full_tensor", gather(model.state_dict()))
            del saved
            to_device(model)
            zero = torch.zeros((), device=device, dtype=dtype)
            # FSDP's padded all-gather buffer differs from DTensor.full_tensor().
            record("forward.result", model(zero))
            record("forward.resharded_local", model.state_dict(), local=True)
            del model

            # Known-correct local slices separate collectives from model loading/offload.
            emit("begin", case="collective_controls")
            manual, dtensors = {}, {}
            for name, expected in reference.items():
                start, count, padded = shard_span(expected.numel(), size, shard_rank)
                local = expected[start : start + count].to(device)
                dtensors[name] = DTensor.from_local(
                    local, shard_mesh, (Shard(0),), shape=expected.shape, stride=expected.stride(), run_check=False
                ).full_tensor()
                padded_input = torch.zeros(padded, dtype=dtype, device=device)
                padded_input[:count].copy_(local)
                output = torch.empty(size * padded, dtype=dtype, device=device)
                dist.all_gather_into_tensor(output, padded_input, group=shard_mesh.get_group())
                manual[name] = output[: expected.numel()]
            record("control.dtensor_full_tensor", dtensors)
            record("control.padded_all_gather", manual)

        emit("complete")
        reports = [None] * world
        dist.all_gather_object(reports, report)
        if rank == 0:
            summary = format_summary(reports)
            (args.output / "bias_probe_summary.txt").write_text(summary, encoding="utf-8")
            (args.output / "bias_probe_summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
            print(summary, flush=True)
        dist.destroy_process_group()
    except Exception as error:
        emit("error", type=type(error).__name__, message=str(error))
        raise
    finally:
        stream.close()


def main():
    """Parse standalone torchrun options without importing Torch for --help."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory for this invocation")
    parser.add_argument("--checkpoint", type=Path, help="Read only visual fc1 biases; otherwise use synthetic values")
    parser.add_argument("--device", choices=("npu", "cuda"), default="npu")
    parser.add_argument("--loader", choices=("verl", "torch"), default="verl")
    parser.add_argument("--fsdp-size", type=int, default=0, help="0 uses WORLD_SIZE; 16 permits a two-node control")
    parser.add_argument("--blocks", type=int, default=27)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--timeout", type=int, default=300, help="Process group timeout in seconds")
    args = parser.parse_args()
    if not 1 <= args.blocks <= 27 or args.fsdp_size < 0 or args.timeout < 1:
        parser.error("require 1 <= blocks <= 27, fsdp-size >= 0 and timeout > 0")
    run(args)


if __name__ == "__main__":
    main()
