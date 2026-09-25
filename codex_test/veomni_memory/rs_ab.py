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

"""Server-only fixed actor-input capture/replay and bounded summary metrics.

Fixtures contain training tensors and stay on the server. Load only fixtures
created by this tool in your own trusted directory (torch.load weights_only=False).
No per-layer device synchronization or diagnostic collective is added.
"""

import dataclasses
import functools
import hashlib
import json
import os
import random
import time
from pathlib import Path


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _config_hash(engine):
    from omegaconf import OmegaConf

    def plain(config):
        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True)
        if dataclasses.is_dataclass(config):
            # Runtime processors/tokenizers are not configuration values. Their
            # repr may contain addresses; deep-copying them also adds memory load.
            return {
                field.name: plain(getattr(config, field.name))
                for field in dataclasses.fields(config)
                if field.name not in ("tokenizer", "processor")
            }
        if isinstance(config, dict):
            return {key: plain(value) for key, value in config.items()}
        if isinstance(config, list | tuple):
            return [plain(value) for value in config]
        if hasattr(config, "to_dict"):
            return plain(config.to_dict())
        return config

    config = {name: plain(getattr(engine, name)) for name in ("model_config", "engine_config", "optimizer_config")}
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def _parameter_hash(module):
    """Stream local shard bytes to a hash; never build a full-model state_dict."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        tensor = getattr(tensor, "_local_tensor", tensor).detach()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        array = tensor.cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(array))
    return digest.hexdigest()


class BackwardMetrics:
    def __init__(self, engine):
        import torch

        from verl_omni.workers.engine.veomni.rs_limiter import audited_groups

        self.engine = engine
        self.npu = torch.npu
        self.limiter = getattr(engine, "_veomni_rs_limiter", None)
        self.records = []
        self.current = None
        self.originals = []
        groups = self.limiter.groups if self.limiter else audited_groups(engine)
        for group in groups:
            original = group.post_backward

            def sampled(*args, _original=original, **kwargs):
                self.sample()
                try:
                    return _original(*args, **kwargs)
                finally:
                    self.sample()

            self.originals.append((group, original))
            group.post_backward = sampled
        self.forward = engine.forward_step

        @functools.wraps(self.forward)
        def forward(*args, **kwargs):
            loss, meta = self.forward(*args, **kwargs)
            if not kwargs.get("forward_only", False):
                return _MeasuredLoss(loss, self), meta
            return loss, meta

        engine.forward_step = forward

    def sample(self):
        if self.current is not None:
            stats = self.npu.memory_stats()
            gap = stats["active_bytes.all.current"] - stats["allocated_bytes.all.current"]
            self.current["gap_sampled_max_bytes"] = max(self.current["gap_sampled_max_bytes"], gap)

    def backward(self, loss, *args, **kwargs):
        if self.current is not None or len(self.records) >= 128:
            raise RuntimeError("AB metrics supports at most 128 non-reentrant loss.backward calls per update")
        before = self.npu.memory_stats()
        self.npu.reset_peak_memory_stats()
        wait_ns = self.limiter.wait_ns if self.limiter else 0
        wait_calls = self.limiter.wait_calls if self.limiter else 0
        start_event = self.npu.Event(enable_timing=True)
        end_event = self.npu.Event(enable_timing=True)
        rec = {"backward": len(self.records), "gap_sampled_max_bytes": 0}
        self.current = rec
        self.sample()
        start_event.record()
        start = time.perf_counter_ns()
        try:
            result = loss.backward(*args, **kwargs)
        finally:
            rec["backward_host_ms"] = (time.perf_counter_ns() - start) / 1e6
            end_event.record()
            self.sample()
            after = self.npu.memory_stats()
            rec.update(
                alloc_retries=after["num_alloc_retries"] - before["num_alloc_retries"],
                active_peak_bytes=after["active_bytes.all.peak"],
                allocated_peak_bytes=after["allocated_bytes.all.peak"],
                limiter_wait_ms=((self.limiter.wait_ns if self.limiter else 0) - wait_ns) / 1e6,
                limiter_wait_calls=(self.limiter.wait_calls if self.limiter else 0) - wait_calls,
            )
            self.records.append((rec, start_event, end_event))
            self.current = None
        return result

    def close(self):
        self.engine.forward_step = self.forward
        for group, original in self.originals:
            group.post_backward = original

    def completed_records(self):
        # Caller has drained the device once at the update boundary, in both arms.
        for rec, start, end in self.records:
            rec["backward_device_ms"] = start.elapsed_time(end)
        return [rec for rec, _, _ in self.records]


class _MeasuredLoss:
    """Pinned VeOmni only calls .backward() on this forward_step return value."""

    def __init__(self, loss, metrics):
        self.loss = loss
        self.metrics = metrics

    def backward(self, *args, **kwargs):
        return self.metrics.backward(self.loss, *args, **kwargs)


def _numeric(value):
    import numpy as np

    if isinstance(value, int | float | np.number):
        return float(value)
    if isinstance(value, list | tuple):
        return [_numeric(v) for v in value]
    return None


def run_update(actor, data):
    """Called only by the explicit codex_test hook inside update_actor."""
    import numpy as np
    import torch
    import torch.distributed as dist
    from verl.utils import tensordict_utils as tu

    mode = os.environ["VEOMNI_RS_AB_MODE"]
    if mode not in ("capture", "replay"):
        raise ValueError("VEOMNI_RS_AB_MODE must be capture or replay")
    engine = actor.engine
    if type(engine).__module__ != "verl_omni.workers.engine.veomni.omni_impl":
        raise RuntimeError("RS AB is restricted to OmniVeOmniEngine")
    if os.environ.get("VEOMNI_MEMORY_PROBE", "0") != "0":
        raise RuntimeError("Disable the older memory probe for fixed-input AB")
    fixture_dir = Path(os.environ["VEOMNI_RS_AB_FIXTURES"])
    out_dir = Path(os.environ["VEOMNI_RS_AB_OUTPUT"])
    if not fixture_dir.is_absolute() or not out_dir.is_absolute():
        raise ValueError("AB directories must be absolute server paths")
    for directory in (fixture_dir, out_dir):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitignore").write_text("*\n", encoding="utf-8")
    rank = dist.get_rank()
    update = getattr(engine, "_rs_ab_update", 0)
    if update >= int(os.environ.get("VEOMNI_RS_AB_UPDATES", "3")):
        raise RuntimeError("AB update limit exceeded; match trainer.total_training_steps")
    fixture = fixture_dir / f"rank{rank}_update{update}.pt"
    output = out_dir / f"rank{rank}_update{update}.json"
    if output.exists():
        raise FileExistsError(output)
    config_hash = _config_hash(engine)
    initial_hash = _parameter_hash(engine.module) if update == 0 else None
    if update == 0 and any(
        opt.state for opt in getattr(engine.optimizer, "optimizers_dict", {"one": engine.optimizer}).values()
    ):
        raise RuntimeError("AB requires a fresh optimizer; do not resume a training checkpoint")
    if mode == "capture":
        if fixture.exists():
            raise FileExistsError(fixture)
        frozen_data = data.cpu().clone()
        payload = {
            "data": frozen_data,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "npu_rng": torch.npu.get_rng_state(),
            "config_hash": config_hash,
            "initial_hash": initial_hash,
            "rank": rank,
            "world_size": dist.get_world_size(),
        }
        torch.save(payload, fixture)
        del frozen_data, payload
    # Both arms use the same serialization round trip, including nested values.
    payload = torch.load(fixture, map_location="cpu", weights_only=False)
    if (
        payload["rank"] != rank
        or payload["world_size"] != dist.get_world_size()
        or payload["config_hash"] != config_hash
        or payload["initial_hash"] != initial_hash
    ):
        raise RuntimeError("AB fixture topology/config/initial local parameters differ")
    data = payload["data"]
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    torch.npu.set_rng_state(payload["npu_rng"])
    fixture_hash = file_hash(fixture)
    torch.npu.synchronize()  # common update boundary; never a per-backward drain
    metrics = BackwardMetrics(engine)
    stats_before = torch.npu.memory_stats()
    start = time.perf_counter_ns()
    try:
        result = actor.train_mini_batch(data=data)
        result = result.cpu() if result is not None else None
        host_ms = (time.perf_counter_ns() - start) / 1e6
        drain_start = time.perf_counter_ns()
        torch.npu.synchronize()
        drain_ms = (time.perf_counter_ns() - drain_start) / 1e6
        update_ms = (time.perf_counter_ns() - start) / 1e6
    finally:
        metrics.close()
    engine._rs_ab_update = update + 1
    reported = tu.get_non_tensor_data(result, "metrics", default={}) if result is not None else {}
    summary = {
        "rank": rank,
        "world_size": dist.get_world_size(),
        "update": update,
        "limit_rs": metrics.limiter is not None,
        "native_offload": engine.engine_config.enable_fsdp_offload,
        "fixture_sha256": fixture_hash,
        "config_sha256": config_hash,
        "initial_shard_sha256": initial_hash,
        "update_actor_host_ms": host_ms,
        "update_actor_drained_ms": update_ms,
        "update_end_drain_ms": drain_ms,
        "update_alloc_retries": torch.npu.memory_stats()["num_alloc_retries"] - stats_before["num_alloc_retries"],
        "backwards": metrics.completed_records(),
        "loss": _numeric(reported.get("loss")),
        "grad_norm": _numeric(reported.get("grad_norm")),
    }
    output.write_text(json.dumps(summary, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    return result
