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
"""Opt-in worker observations outside compiled model forward functions."""

import functools
import importlib.metadata
import inspect
import os
import socket
from collections import OrderedDict

from verl_omni.utils import video_trace as trace
from verl_omni.utils.video_trace_frontend import feature_records

_TRACE_DEFAULTS = {
    "VERL_OMNI_VIDEO_TRACE_DIR": "",
    "VERL_OMNI_VIDEO_TRACE_STAGE": "boundary",
    "VERL_OMNI_VIDEO_TRACE_MODE": "metadata",
    "VERL_OMNI_VIDEO_TRACE_DEVICE_SAMPLE": "0",
    "VERL_OMNI_VIDEO_TRACE_VISION": "0",
    "VERL_OMNI_VIDEO_TRACE_NUMERIC_SAMPLES": "1024",
    "VERL_OMNI_VIDEO_TRACE_MAX_BYTES": "8388608",
    "VERL_OMNI_VIDEO_TRACE_MAX_TOKENS": "32768",
}


def enabled():
    return trace.enabled() and os.environ.get("VERL_OMNI_VIDEO_TRACE_STAGE") == "worker"


def snapshot(value):
    """Device reads require opt-in; vision mode also retains bounded numeric samples."""
    if os.environ.get("VERL_OMNI_VIDEO_TRACE_VISION", "0") == "1":
        from verl_omni.utils.video_trace_numeric import snapshot as numeric_snapshot

        return numeric_snapshot(value)
    if value is None or os.environ.get("VERL_OMNI_VIDEO_TRACE_DEVICE_SAMPLE", "0") != "1":
        return value
    if type(value).__module__.split(".")[0] != "torch" or str(value.device) == "cpu":
        return value
    import numpy as np
    import torch

    result = trace._array(value, "metadata")
    count = value.numel()
    indices = np.linspace(0, count - 1, min(count, 256), dtype=np.int64) if count else np.array([], dtype=np.int64)
    if not count:
        raw = b""
    else:
        coordinates = (
            tuple(torch.tensor(axis.copy(), device=value.device) for axis in np.unravel_index(indices, value.shape))
            if value.ndim
            else ()
        )
        sample = value.detach()[coordinates] if value.ndim else value.detach()
        raw = sample.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    result.update(
        digest_status="sample",
        sample_count=len(indices),
        sample_scheme="linspace-flat-c-order-v1",
        fingerprint=trace._Bytes(raw, "sample_sha256"),
        device_read=True,
    )
    return result


def _global_id(request):
    info = getattr(request, "model_intermediate_buffer", None)
    if not isinstance(info, dict) or "global_request_id" not in info:
        info = getattr(request, "additional_information", None)
        if info is not None and not isinstance(info, dict):
            from vllm_omni.engine.serialization import deserialize_additional_information

            info = deserialize_additional_information(info)
    ids = info.get("global_request_id") if isinstance(info, dict) else None
    return ids[0] if isinstance(ids, list | tuple) and len(ids) == 1 and isinstance(ids[0], str) else None


class WorkerObserver:
    def __init__(self, worker):
        self.runner = worker.model_runner
        self.pending = OrderedDict()
        self.requests = {}
        self.active = []
        self.hooks = {}
        self.entered_hooks = set()
        self.unmatched_count = 0
        self.rank = getattr(worker, "rank", None)
        self.local_rank = getattr(worker, "local_rank", None)
        self.runtime = {}
        self.vision = None
        for package in ("vllm", "vllm-omni", "vllm-ascend", "torch", "torch-npu"):
            try:
                self.runtime[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self.runtime[package] = None
        for config_name, keys in (
            ("model_config", ("model", "seed", "dtype", "revision", "enforce_eager", "quantization")),
            (
                "parallel_config",
                ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "enable_expert_parallel"),
            ),
            ("cache_config", ("enable_prefix_caching",)),
            ("compilation_config", ("mode", "backend", "cudagraph_mode")),
        ):
            config = getattr(self.runner, config_name, None)
            if config is None:
                config = getattr(getattr(self.runner, "vllm_config", None), config_name, None)
            self.runtime[config_name] = {key: str(getattr(config, key, None)) for key in keys}
        self.runtime["runner_class"] = f"{type(self.runner).__module__}.{type(self.runner).__qualname__}"

    def register(self, request_id, context):
        if not enabled() or not context.get("selected"):
            return
        if len(self.pending) >= 64:
            self.pending.popitem(last=False)
            trace.event("trace.worker_registration_evicted", force=True)
        self.pending[request_id] = dict(context)
        self.entered_hooks.clear()
        self.unmatched_count = 0

    def hook_enter(self, name):
        if name not in self.entered_hooks:
            self.entered_hooks.add(name)
            trace.event(
                "trace.worker_hook_enter",
                force=True,
                hook=name,
                worker_rank=self.rank,
                pending_request_ids=list(self.pending)[:8],
            )

    def emit(self, entry, name, **fields):
        limit = 1024 if self.vision is not None else 256
        if entry["events"] >= limit:
            if entry["events"] == limit:
                with trace.scope("worker.limit", incoming=entry["context"]):
                    trace.event("trace.worker_limit", core_request_id=entry["id"], worker_rank=self.rank)
            entry["events"] += 1
            return
        entry["events"] += 1
        with trace.scope(
            "worker.observation",
            incoming=entry["context"],
            core_request_id=entry["id"],
            record_boundaries=self.vision is None,
        ):
            trace.event(name, core_request_id=entry["id"], worker_rank=self.rank, local_rank=self.local_rank, **fields)

    def receive(self, arguments, result=None):
        scheduler = arguments["scheduler_output"]
        for request_id in getattr(scheduler, "finished_req_ids", ()):
            entry = self.requests.pop(request_id, None)
            if entry:
                self.emit(entry, "worker.finished")
        for request in scheduler.scheduled_new_reqs:
            if not self.pending:
                break
            external_id = _global_id(request)
            context = self.pending.pop(external_id, None)
            # Exact matching only. Never strip a guessed UUID suffix.
            if context is None:
                context = self.pending.pop(request.req_id, None)
            if context is None:
                if self.pending and self.unmatched_count < 64:
                    trace.event(
                        "trace.worker_unmatched",
                        force=True,
                        core_request_id=request.req_id,
                        global_id=external_id,
                        pending_request_ids=list(self.pending)[:8],
                        worker_rank=self.rank,
                    )
                elif self.unmatched_count == 64:
                    trace.event("trace.worker_unmatched_limit", force=True, worker_rank=self.rank)
                self.unmatched_count += 1
                continue
            entry = {"id": request.req_id, "context": context, "events": 0, "chunks": 0}
            self.requests[request.req_id] = entry
            self.emit(
                entry,
                "worker.receive",
                token_fields={"prompt_token_snapshot": request.prompt_token_ids},
                multimodal_features=feature_records(request.mm_features),
                sampling_params=trace.sampling(request.sampling_params),
                num_computed_tokens=request.num_computed_tokens,
                runtime=self.runtime,
                global_id=external_id,
            )

    def spans(self, scheduler):
        offset = 0
        spans = []
        for request_id in self.runner.input_batch.req_ids:
            count = scheduler.num_scheduled_tokens[request_id]
            state = self.runner.requests[request_id]
            entry = self.requests.get(request_id)
            start = state.num_computed_tokens
            if entry and start < len(state.prompt_token_ids or ()) and entry["chunks"] < 32:
                entry["chunks"] += 1
                spans.append((entry, state, offset, count, start))
            elif entry and entry["chunks"] == 32:
                self.emit(entry, "trace.worker_chunk_limit")
                entry["chunks"] += 1
            offset += count
        return spans

    def cache(self, arguments, result, write):
        identifier = arguments["mm_hash"]
        value = arguments["output"] if write else result
        for entry, state, _, count, start in self.active:
            if entry["events"] >= (1024 if self.vision is not None else 256):
                self.emit(entry, "trace.worker_limit")
                continue
            for index, feature in enumerate(state.mm_features):
                if feature.identifier == identifier:
                    self.emit(
                        entry,
                        "worker.encoder.write" if write else "worker.encoder.read",
                        feature_index=index,
                        modality=feature.modality,
                        identifier=identifier,
                        chunk_start=start,
                        chunk_tokens=count,
                        cache_present=value is not None,
                        embedding=snapshot(value),
                    )
                    if self.vision is not None and feature.modality == "video":
                        self.vision.safe(self.vision.cache_parts, entry, index, value, write)

    def gather(self, arguments, result):
        embeddings, mask = result
        for entry, _, offset, count, start in self.active:
            if entry["events"] >= (1024 if self.vision is not None else 256):
                self.emit(entry, "trace.worker_limit")
                continue
            self.emit(
                entry,
                "worker.gather",
                chunk_start=start,
                chunk_tokens=count,
                mask=snapshot(mask[offset : offset + count]),
                batch_embedding_count=len(embeddings),
                batch_offset=offset,
                shift_computed_tokens=arguments.get("shift_computed_tokens", 0),
            )

    def merged(self, arguments, result):
        input_ids, embeds, positions = result[:3]
        for entry, state, offset, count, start in self.active:
            if entry["events"] >= (1024 if self.vision is not None else 256):
                self.emit(entry, "trace.worker_limit")
                continue
            if embeds is not None and len(embeds.shape) != 2:
                self.emit(entry, "trace.worker_layout_unknown", shape=list(embeds.shape))
                continue
            if (embeds is not None and embeds.shape[0] < offset + count) or (
                positions is not None and positions.shape[-1] < offset + count
            ):
                self.emit(entry, "trace.worker_layout_unknown", batch_offset=offset, chunk_tokens=count)
                continue
            self.emit(
                entry,
                "worker.model_input",
                chunk_start=start,
                chunk_tokens=count,
                input_ids=snapshot(input_ids[offset : offset + count]) if input_ids is not None else None,
                embedding=snapshot(embeds[offset : offset + count]) if embeds is not None else None,
                positions=snapshot(positions[..., offset : offset + count]) if positions is not None else None,
                multimodal_features=[
                    {k: v for k, v in item.items() if k != "data"} for item in feature_records(state.mm_features)
                ],
                batch_offset=offset,
            )

    def safe(self, function, *args):
        try:
            function(*args)
        except Exception as exc:
            trace.event(
                "trace.observation_error", force=True, boundary="worker", operation=function.__name__, error=str(exc)
            )

    def attach(self, name, before=None, after=None, preprocess=False):
        try:
            original = getattr(self.runner, name)
            signature = inspect.signature(original)
            required = (
                {"mm_hash", "output"}
                if name == "_cache_encoder_output"
                else {"mm_hash"}
                if name == "_get_encoder_output_from_cache"
                else {"scheduler_output"}
            )
            if not required.issubset(signature.parameters) or inspect.iscoroutinefunction(original):
                raise TypeError(f"Unsupported signature: {signature}")

            @functools.wraps(original)
            def observed(*args, **kwargs):
                if not enabled() or not (self.pending or self.requests or self.active):
                    return original(*args, **kwargs)
                self.safe(self.hook_enter, name)
                try:
                    arguments = signature.bind(*args, **kwargs).arguments
                except Exception as exc:
                    trace.event(
                        "trace.observation_error",
                        force=True,
                        boundary="worker." + name,
                        operation="signature_bind",
                        error=str(exc),
                    )
                    return original(*args, **kwargs)
                previous = self.active
                if preprocess:
                    self.active = []
                    self.safe(lambda: self.active.extend(self.spans(arguments["scheduler_output"])))
                    if self.vision is not None:
                        self.vision.safe(self.vision.prepare)
                try:
                    if before:
                        self.safe(before, arguments)
                    result = original(*args, **kwargs)
                    if after:
                        self.safe(after, arguments, result)
                    if preprocess and self.vision is not None:
                        self.vision.safe(self.vision.coverage)
                    return result
                finally:
                    if preprocess:
                        self.active = previous

            setattr(self.runner, name, observed)
            self.hooks[name] = {"status": "installed", "signature": str(signature)}
            trace.event(
                "trace.install", force=True, boundary="worker." + name, status="installed", worker_rank=self.rank
            )
        except Exception as exc:
            self.hooks[name] = {"status": "unavailable", "error": str(exc)}
            trace.event("trace.install", force=True, boundary="worker." + name, status="unavailable", error=str(exc))


def _install_vision(observer):
    if observer.vision is not None or os.environ.get("VERL_OMNI_VIDEO_TRACE_VISION", "0") != "1":
        return
    from verl_omni.utils.video_trace_vision import VisionObserver

    observer.vision = VisionObserver(observer)
    try:
        observer.vision.install()
        unavailable = [key for key, value in observer.vision.hooks.items() if value.get("status") != "installed"]
        observer.hooks["vision"] = {
            "status": "partial" if unavailable else "installed",
            "hook_count": len(observer.vision.hooks),
            "unavailable": unavailable,
        }
    except Exception as exc:
        observer.hooks["vision"] = {"status": "unavailable", "error": str(exc)}
        observer.vision.hooks["vision"] = observer.hooks["vision"]


def install(worker):
    if not enabled() or getattr(worker, "_video_observer", None) is not None:
        return
    try:
        observer = WorkerObserver(worker)
        worker._video_observer = observer
        observer.attach("_update_states", before=observer.receive)
        observer.attach("_preprocess", after=observer.merged, preprocess=True)
        observer.attach("_gather_mm_embeddings", after=observer.gather)
        observer.attach("_cache_encoder_output", after=lambda args, result: observer.cache(args, result, True))
        observer.attach(
            "_get_encoder_output_from_cache", after=lambda args, result: observer.cache(args, result, False)
        )
        _install_vision(observer)
    except Exception as exc:
        trace.event("trace.install", force=True, boundary="worker", status="unavailable", error=str(exc))


def register_worker(worker, request_id, context, trace_options=None):
    """Return worker-side evidence even when its trace environment is missing."""
    if trace_options is not None and context.get("selected"):
        for name in _TRACE_DEFAULTS:
            value = trace_options.get(name)
            if isinstance(value, str):
                os.environ[name] = value
    reply = {
        "video_trace_ack": True,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "worker_rank": getattr(worker, "rank", None),
        "registered_request_id": request_id,
        "trace_dir": os.environ.get("VERL_OMNI_VIDEO_TRACE_DIR", ""),
        "trace_stage": os.environ.get("VERL_OMNI_VIDEO_TRACE_STAGE", "boundary"),
        "device_sample": os.environ.get("VERL_OMNI_VIDEO_TRACE_DEVICE_SAMPLE", "0"),
        "vision": os.environ.get("VERL_OMNI_VIDEO_TRACE_VISION", "0"),
        "worker_cwd": os.getcwd(),
        "status": "disabled",
    }
    if not enabled():
        return reply
    install(worker)
    observer = getattr(worker, "_video_observer", None)
    if observer is None:
        return {**reply, "status": "unavailable"}
    _install_vision(observer)
    observer.register(request_id, context)
    return {
        **reply,
        "status": "registered" if request_id in observer.pending else "not_selected",
        "hooks": observer.hooks,
        "trace_file": str(trace._sink.path) if trace._sink is not None else None,
    }


async def register(server, request_id):
    """Send scalar diagnostic context separately from the inference request."""
    context = trace.export_context()
    if not enabled() or not context.get("selected"):
        return
    try:
        options = {name: os.environ.get(name, default) for name, default in _TRACE_DEFAULTS.items()}
        replies = await server.collective_rpc(
            method="register_video_trace", timeout=10.0, args=(request_id, context), kwargs={"trace_options": options}
        )
        trace.event(
            "trace.worker_registration", status="returned", registered_request_id=request_id, worker_replies=replies
        )
    except Exception as exc:
        trace.event("trace.worker_registration", status="failed", error=str(exc))


def install_client(server):
    """Register the rewritten Omni ID immediately before engine admission."""
    if not enabled():
        return
    boundary = "frontend.worker_admission"
    try:
        engine = server.engine.engine
        original = engine.add_request_async
        if getattr(original, "_video_trace_admission", False):
            return
        signature = inspect.signature(original)
        if not inspect.iscoroutinefunction(original) or "request_id" not in signature.parameters:
            raise TypeError(f"Unsupported add_request_async signature: {signature}")

        @functools.wraps(original)
        async def observed(*args, **kwargs):
            if enabled() and trace.export_context().get("selected"):
                try:
                    actual_id = signature.bind(*args, **kwargs).arguments["request_id"]
                    if not isinstance(actual_id, str) or not actual_id:
                        raise TypeError("Engine admission request_id must be a nonempty string")
                except Exception as exc:
                    trace.event("trace.observation_error", boundary=boundary, error=str(exc))
                else:
                    await register(server, actual_id)
            return await original(*args, **kwargs)

        observed._video_trace_admission = True
        engine.add_request_async = observed
        trace.event("trace.install", force=True, boundary=boundary, status="installed", signature=str(signature))
    except Exception as exc:
        trace.event("trace.install", force=True, boundary=boundary, status="unavailable", error=str(exc))
