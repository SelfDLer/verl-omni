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
"""Opt-in, synchronous payload fingerprints for diagnosing video request mixing.

Only standard-library imports at module load: agent registration imports this
before torch/vLLM are necessarily available. No payload references are retained.
"""

import contextlib
import contextvars
import functools
import hashlib
import importlib
import inspect
import json
import logging
import os
import socket
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path

_context = contextvars.ContextVar("verl_omni_video_trace", default=None)
_lock = threading.Lock()
_count = 0
_sequence = 0
_warned = False
_session = uuid.uuid4().hex[:12]
_logger = logging.getLogger(__name__)


def enabled():
    return bool(os.environ.get("VERL_OMNI_VIDEO_TRACE_DIR"))


def _warn(exc):
    global _warned
    if not _warned:
        _warned = True
        _logger.warning("Video trace incomplete: %s: %s", type(exc).__name__, exc)


def snapshot(value):
    """Freeze full array bytes now, including dtype/shape; never log raw pixels."""
    if value is None or isinstance(value, str | bool | int | float):
        return value
    module = type(value).__module__.split(".")[0]
    if module == "torch":
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            return {"sha256": hashlib.sha256(raw).hexdigest(), "shape": list(value.shape), "dtype": str(value.dtype)}
    if module == "numpy":
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return {"unsupported": "numpy object array"}
            return {
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if isinstance(value, np.generic):
            return snapshot(value.item())
    if isinstance(value, Mapping):
        return {str(key): snapshot(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [snapshot(item) for item in value]
    # vLLM's MultiModalKwargsItem contains field wrappers, not bare tensors.
    if type(value).__name__ == "MultiModalFieldElem":
        return snapshot(value.data)
    return {"unsupported": f"{type(value).__module__}.{type(value).__qualname__}"}


def emit(event, *, force=False, **fields):
    """Trace failures must not replace the application's result or exception."""
    global _sequence
    ctx = _context.get()
    if not enabled() or (not force and not ctx):
        return
    try:
        frozen = snapshot(fields)
        host = socket.gethostname()
        with _lock:
            _sequence += 1
            record = {
                "event": event,
                "host": host,
                "pid": os.getpid(),
                "session": _session,
                "seq": _sequence,
                "time_ns": time.time_ns(),
                **(ctx or {}),
                **frozen,
            }
            directory = Path(os.environ["VERL_OMNI_VIDEO_TRACE_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"video-{host}-{os.getpid()}-{_session}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
    except Exception as exc:
        _warn(exc)


@contextlib.contextmanager
def request_scope(request_id, **fields):
    global _count
    if not enabled():
        yield
        return
    parent = _context.get()
    ctx = parent
    if parent is None:
        try:
            limit = int(os.environ.get("VERL_OMNI_VIDEO_TRACE_MAX_REQUESTS", "32"))
            if limit < 0:
                raise ValueError("VERL_OMNI_VIDEO_TRACE_MAX_REQUESTS must be >= 0")
            with _lock:
                _count += 1
                selected = limit == 0 or _count <= limit
            ctx = {"request_id": str(request_id), "trace_id": uuid.uuid4().hex, **fields} if selected else {}
        except Exception as exc:
            _warn(exc)
            ctx = {}
    elif parent:
        ctx = {**parent, **fields}
    token = _context.set(ctx)
    try:
        yield
    finally:
        _context.reset(token)


def trace_prompt(event, prompt, **fields):
    if not enabled() or not _context.get():
        return
    try:
        if not isinstance(prompt, Mapping):
            emit(event, prompt_type=type(prompt).__name__, **fields)
            return
        ids = prompt.get("prompt_token_ids", prompt.get("prompt_ids"))
        token_digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        data = prompt.get("multi_modal_data") or {}
        emit(
            event,
            prompt_ids_sha256=token_digest,
            video=data.get("video"),
            multi_modal_uuids=prompt.get("multi_modal_uuids"),
            mm_processor_kwargs=prompt.get("mm_processor_kwargs"),
            **fields,
        )
    except Exception as exc:
        _warn(exc)


def trace_generate(event):
    """Observe the real client/server generate method without changing arguments."""

    def decorate(func):
        signature = inspect.signature(func)

        @functools.wraps(func)
        async def wrapped(*args, **kwargs):
            if not enabled():
                return await func(*args, **kwargs)
            try:
                bound = signature.bind(*args, **kwargs).arguments
                owner = bound.get("self")
                identity = {key: getattr(owner, key, None) for key in ("replica_rank", "node_rank")}
            except Exception as exc:
                _warn(exc)
                return await func(*args, **kwargs)
            with request_scope(bound.get("request_id"), **identity):
                trace_prompt(
                    event,
                    {
                        "prompt_ids": bound.get("prompt_ids"),
                        "multi_modal_data": {"video": bound.get("video_data")},
                        "mm_processor_kwargs": bound.get("mm_processor_kwargs"),
                    },
                )
                try:
                    result = await func(*args, **kwargs)
                except BaseException as exc:
                    emit(event + ".error", error_type=type(exc).__name__)
                    raise
                emit(event + ".done")
                return result

        wrapped._video_trace_installed = True
        return wrapped

    return decorate


def _video_kwargs(value):
    return {"video": value.get("video")} if isinstance(value, Mapping) else value


def _observe(event, bound, result=None, *, after=False):
    """Select small, comparable fields from version-specific frontend objects."""
    phase = ".after" if after else ".before"
    if event in ("engine.build", "engine.scope_uuids", "frontend.process_tokens"):
        prompt = bound.get("prompt", bound.get("parsed_content"))
        extra = {key: bound[key] for key in ("stage_id", "replica_id") if key in bound}
        trace_prompt(event + phase, prompt, **extra)
    if after and event == "engine.build":
        request = getattr(result, "prompt", None)
        features = getattr(request, "mm_features", None)
        emit(
            "engine.submission",
            core_request_id=getattr(request, "request_id", None),
            external_req_id=getattr(request, "external_req_id", None),
            video_features=[
                {key: getattr(item, key, None) for key in ("identifier", "mm_hash", "data")}
                for item in (features or [])
                if getattr(item, "modality", None) == "video"
            ],
        )
    elif after and event == "frontend.process_tokens" and isinstance(result, Mapping):
        emit("frontend.processed", mm_kwargs=_video_kwargs(result.get("mm_kwargs")), mm_hashes=result.get("mm_hashes"))
    elif after and event == "frontend.hf_fresh":
        emit(event, mm_kwargs=_video_kwargs(result))
    elif event == "frontend.cache_merge":
        if after:
            emit(event + phase, mm_kwargs=_video_kwargs(result[0]))
        else:
            emit(
                event + phase,
                mm_hashes=bound.get("mm_hashes"),
                mm_is_cached=bound.get("mm_is_cached"),
                missing_mm_kwargs=_video_kwargs(bound.get("mm_missing_kwargs")),
            )
    elif after and event == "agent.request_id":
        emit(event, engine_request_id=result, agent_request_id=bound.get("request_id"))


def _install_sync_hook(owner, name, event):
    descriptor = inspect.getattr_static(owner, name)
    original = getattr(owner, name)
    if getattr(original, "_video_trace_installed", False):
        return
    signature = inspect.signature(original)
    required = {
        "engine.build": {"request_id", "prompt"},
        "engine.scope_uuids": {"prompt", "stage_id", "replica_id"},
        "frontend.process_tokens": {"parsed_content"},
        "frontend.hf_fresh": {"hf_inputs", "config_by_key"},
        "frontend.cache_merge": {"mm_hashes", "mm_is_cached", "mm_missing_kwargs"},
        "agent.request_id": {"request_id"},
    }[event]
    if not required.issubset(signature.parameters):
        raise TypeError(f"Unsupported signature for {event}: {signature}")

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if not enabled():
            return original(*args, **kwargs)
        try:
            bound = signature.bind(*args, **kwargs).arguments
        except Exception as exc:
            _warn(exc)
            return original(*args, **kwargs)
        scope = (
            request_scope(bound.get("request_id"), engine_request_id=bound.get("request_id"))
            if event == "engine.build"
            else contextlib.nullcontext()
        )
        with scope:

            def observe(after=False, result=None):
                if _context.get():
                    try:
                        _observe(event, bound, result, after=after)
                    except Exception as exc:
                        _warn(exc)

            observe()
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                emit(event + ".error", error_type=type(exc).__name__)
                raise
            observe(after=True, result=result)
            return result

    wrapped._video_trace_installed = True
    setattr(owner, name, staticmethod(wrapped) if isinstance(descriptor, staticmethod) else wrapped)


def install_frontend_hooks():
    """Install only in the enabled server process, before AsyncOmni starts."""
    if not enabled():
        return
    specs = (
        ("vllm_omni.engine.async_omni_engine", "AsyncOmniEngine", "_build_add_request_message", "engine.build"),
        (
            "vllm_omni.engine.async_omni_engine",
            "AsyncOmniEngine",
            "_ensure_stage_replica_mm_uuids",
            "engine.scope_uuids",
        ),
        ("vllm_omni.inputs.preprocess", "OmniInputPreprocessor", "_process_tokens", "frontend.process_tokens"),
        ("vllm.multimodal.inputs", "MultiModalKwargsItems", "from_hf_inputs", "frontend.hf_fresh"),
        ("vllm.multimodal.processing.processor", "BaseMultiModalProcessor", "_merge_mm_kwargs", "frontend.cache_merge"),
    )
    for module, cls, method, event in specs:
        try:
            _install_sync_hook(getattr(importlib.import_module(module), cls), method, event)
            emit("hook.install", force=True, hook=event, status="installed")
        except Exception as exc:
            emit("hook.install", force=True, hook=event, status="unavailable", error_type=type(exc).__name__)
            _logger.warning("Video trace hook %s unavailable: %s", event, exc)


def install_agent_hooks():
    """Include upstream single_turn_agent, and bridge its regenerated request ID."""
    if not enabled():
        return
    try:
        module = importlib.import_module("verl.workers.rollout.llm_server")
        client = module.LLMServerClient
        if not getattr(client.generate, "_video_trace_installed", False):
            client.generate = trace_generate("agent.send")(client.generate)
        _install_sync_hook(client, "_vllm_request_id", "agent.request_id")
        emit("hook.install", force=True, hook="agent", status="installed")
    except Exception as exc:
        emit("hook.install", force=True, hook="agent", status="unavailable", error_type=type(exc).__name__)
        _logger.warning("Video trace agent hook unavailable: %s", exc)
