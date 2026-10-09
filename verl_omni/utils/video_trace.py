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
"""Explicit observations with CPU-only snapshots and bounded background output.

Metadata is the default. Optional byte snapshots are copied at observation time;
only immutable bytes and plain metadata may reach the writer. No tensor transfers
or dependency patches are performed here.
"""

import atexit
import contextlib
import contextvars
import hashlib
import itertools
import json
import logging
import math
import os
import queue
import re
import socket
import sys
import threading
import time
import uuid
from array import array
from collections.abc import Mapping
from pathlib import Path

_context = contextvars.ContextVar("omni_video_diagnostic", default=None)
_sink = None
_sink_lock = threading.Lock()
_sink_pid = os.getpid()
_request_count = 0
_logger = logging.getLogger(__name__)
_CONTEXT_KEYS = (
    "trace_id",
    "call_id",
    "uid",
    "session_id",
    "sample_index",
    "sample_key",
    "video_id",
    "question_id",
    "agent_request_id",
    "selected",
)
_SAMPLING_KEYS = (
    "max_tokens",
    "min_tokens",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "stop",
    "stop_token_ids",
    "ignore_eos",
    "detokenize",
    "skip_special_tokens",
    "include_stop_str_in_output",
    "repetition_penalty",
)


def enabled():
    """Enable only when a destination was explicitly supplied to this process."""
    return bool(os.environ.get("VERL_OMNI_VIDEO_TRACE_DIR"))


class _Bytes:
    def __init__(self, raw, field):
        self.raw = raw
        self.field = field


class _TokenIds(_Bytes):
    pass


class _Text:
    def __init__(self, value):
        self.characters = len(value)
        self.value = value[:65536]


def describe_text(value):
    """Summarize format markers without assigning an answer score."""
    pairs = list(itertools.islice(re.finditer(r"<answer>\s*(.*?)\s*</answer>", value, re.DOTALL), 4))
    return {
        "characters": len(value),
        "empty_visible_text": not value.strip(),
        "answer_open_count": value.count("<answer>"),
        "answer_close_count": value.count("</answer>"),
        "has_answer_pair": bool(pairs),
        "answer_payloads": [match.group(1).strip()[:64] for match in pairs],
        "think_open_count": value.count("<think>"),
        "think_close_count": value.count("</think>"),
        "head": value[:512],
        "tail": value[-512:],
    }


def _token_snapshot(ids):
    if type(ids) not in (list, tuple):
        return {"status": "unavailable" if ids is None else "unsupported_type"}
    limit = int(os.environ.get("VERL_OMNI_VIDEO_TRACE_MAX_TOKENS", "32768"))
    if limit <= 0 or len(ids) > limit:
        return {"status": "disabled" if limit <= 0 else "skipped_token_limit", "count": len(ids)}
    # Never let array() coerce a tensor scalar and synchronize an accelerator.
    if any(type(item) is not int for item in ids):
        return {"status": "unsupported_element_type", "count": len(ids)}
    packed = array("q", ids)
    if sys.byteorder != "little":
        packed.byteswap()
    return _TokenIds(packed.tobytes(), "sha256")


def sampling(value):
    """Read only the scalar/list settings relevant to format and termination."""
    try:
        return {
            key: value.get(key) if isinstance(value, Mapping) else getattr(value, key, None) for key in _SAMPLING_KEYS
        }
    except Exception as exc:
        return {"observation_error": type(exc).__name__}


def _finalize(value):
    if isinstance(value, _TokenIds):
        ids = array("q")
        ids.frombytes(value.raw)
        if sys.byteorder != "little":
            ids.byteswap()
        return {
            "status": "complete",
            "count": len(ids),
            "encoding": "int64-le",
            "sha256": hashlib.sha256(value.raw).hexdigest(),
            "ids": ids.tolist(),
        }
    if isinstance(value, _Text):
        complete = value.characters == len(value.value)
        return {
            **describe_text(value.value),
            "characters": value.characters,
            "complete": complete,
            "sha256": hashlib.sha256(value.value.encode("utf-8")).hexdigest() if complete else None,
        }
    if isinstance(value, _Bytes):
        return {value.field: hashlib.sha256(value.raw).hexdigest()}
    if isinstance(value, dict):
        has_bytes = isinstance(value.get("fingerprint"), _Bytes)
        result = {key: _finalize(item) for key, item in value.items() if not (has_bytes and key == "fingerprint")}
        if has_bytes:
            result.update(_finalize(value["fingerprint"]))
        return result
    if isinstance(value, list):
        return [_finalize(item) for item in value]
    return value


def _byte_count(value):
    if isinstance(value, _Bytes):
        return len(value.raw)
    if isinstance(value, _Text):
        return len(value.value) * 4
    if isinstance(value, dict):
        return sum(_byte_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(_byte_count(item) for item in value)
    return 0


class _Writer:
    def __init__(self, directory):
        self.pid = os.getpid()
        self.host = socket.gethostname()
        self.run = uuid.uuid4().hex
        self.path = Path(directory) / f"video-v2-{self.host}-{self.pid}-{self.run}.jsonl"
        self.queue = queue.Queue(maxsize=128)
        self.lock = threading.Lock()
        self.pending_bytes = 0
        self.byte_limit = 64 * 1024 * 1024
        self.dropped = 0
        self.sequence = 0
        self.error = None
        self.thread = threading.Thread(target=self._run, name="video-trace-writer", daemon=True)
        self.thread.start()

    def put(self, record):
        size = _byte_count(record)
        # This lock protects counters only, never hashing or filesystem calls.
        with self.lock:
            self.sequence += 1
            if self.error or self.queue.full() or self.pending_bytes + size > self.byte_limit:
                self.dropped += 1
                return
            record.update(
                schema=2,
                host=self.host,
                pid=self.pid,
                writer_id=self.run,
                seq=self.sequence,
                dropped_events=self.dropped,
            )
            self.pending_bytes += size
            self.queue.put_nowait((record, size))

    def _write(self, stream, record):
        stream.write(json.dumps(_finalize(record), ensure_ascii=True, sort_keys=True) + "\n")
        stream.flush()

    def _run(self):
        stream = None
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    self.queue.task_done()
                    return
                record, size = item
                try:
                    if not self.error:
                        if stream is None:
                            self.path.parent.mkdir(parents=True, exist_ok=True)
                            stream = self.path.open("a", encoding="utf-8")
                        self._write(stream, record)
                        with self.lock:
                            dropped = self.dropped
                        if dropped:
                            self._write(
                                stream,
                                {
                                    "schema": 2,
                                    "event": "trace.health",
                                    "host": self.host,
                                    "pid": self.pid,
                                    "writer_id": self.run,
                                    "dropped_events": dropped,
                                },
                            )
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    _logger.warning("Video trace writer failed; generation continues: %s", self.error)
                finally:
                    with self.lock:
                        self.pending_bytes -= size
                    self.queue.task_done()
        finally:
            if stream is not None:
                stream.close()


def _writer():
    global _sink, _sink_pid, _sink_lock
    # A forked child must not use its parent's queue/lock or writer thread.
    if _sink_pid != os.getpid():
        _sink, _sink_lock, _sink_pid = None, threading.Lock(), os.getpid()
    if _sink is None:
        with _sink_lock:
            if _sink is None:
                _sink = _Writer(os.environ["VERL_OMNI_VIDEO_TRACE_DIR"])
    return _sink


def flush(timeout=1.0):
    """Bounded drain for tests/shutdown, never called by request observations."""
    if _sink is None or _sink.pid != os.getpid():
        return True
    deadline = time.monotonic() + timeout
    while _sink.queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.005)
    return not _sink.queue.unfinished_tasks and _sink.error is None


atexit.register(flush)


def _array(value, mode):
    module = type(value).__module__.split(".")[0]
    if module not in ("numpy", "torch") or not hasattr(value, "shape"):
        return None
    import numpy as np

    is_torch = module == "torch"
    if not is_torch and not isinstance(value, np.ndarray):
        return None
    shape = list(value.shape)
    device = str(value.device) if is_torch else "cpu"
    result = {
        "shape": shape,
        "dtype": str(value.dtype).removeprefix("torch."),
        "device": device,
        "object_id": id(value),
    }
    result["stride" if is_torch else "strides_bytes"] = list(value.stride() if is_torch else value.strides)
    if mode == "metadata":
        result["digest_status"] = "metadata_only"
        return result
    if device != "cpu":
        result["digest_status"] = "skipped_non_cpu"
        return result
    if not is_torch and value.dtype.hasobject:
        result["digest_status"] = "skipped_object_array"
        return result
    count = math.prod(shape)
    itemsize = value.element_size() if is_torch else value.dtype.itemsize
    if mode == "full" and count * itemsize > int(os.environ.get("VERL_OMNI_VIDEO_TRACE_MAX_BYTES", "8388608")):
        result["digest_status"] = "skipped_byte_limit"
        return result
    if mode == "sample" and count:
        indices = np.linspace(0, count - 1, min(count, 256), dtype=np.int64)
        coordinates = np.unravel_index(indices, tuple(shape)) if shape else ()
        if is_torch:
            import torch

            coordinates = tuple(torch.from_numpy(index.copy()) for index in coordinates)
            selected = value.detach()[coordinates] if shape else value.detach()
        else:
            selected = value[coordinates] if shape else value
        result.update(sample_count=len(indices), sample_scheme="linspace-flat-c-order-v1")
    else:
        selected = value.detach() if is_torch else value
    if is_torch:
        import torch

        raw = selected.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    else:
        raw = selected.tobytes(order="C")
    result["digest_status"] = mode
    result["fingerprint"] = _Bytes(raw, "sha256" if mode == "full" else "sample_sha256")
    return result


def _freeze(value, mode, depth=0):
    if isinstance(value, _Bytes | _Text):
        return value
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:2048]
    if depth > 8:
        return {"omitted": "depth_limit"}
    array = _array(value, mode)
    if array is not None:
        return array
    if type(value).__module__.split(".")[0] == "numpy" and hasattr(value, "item"):
        return _freeze(value.item(), mode, depth + 1)
    if isinstance(value, Mapping):
        result = {str(key): _freeze(item, mode, depth + 1) for key, item in itertools.islice(value.items(), 128)}
        if len(value) > 128:
            result["omitted_fields"] = len(value) - 128
        return result
    if isinstance(value, list | tuple):
        result = [_freeze(item, mode, depth + 1) for item in value[:128]]
        if len(value) > 128:
            result.append({"omitted_items": len(value) - 128})
        return result
    if type(value).__name__ == "MultiModalFieldElem":
        return _freeze(value.data, mode, depth + 1)
    return {"unsupported": f"{type(value).__module__}.{type(value).__qualname__}"}


def event(name, *, force=False, token_fields=None, text_fields=None, **fields):
    """Freeze now and enqueue without waiting for file I/O or digest computation."""
    ctx = _context.get()
    if not enabled() or (not force and (not ctx or not ctx.get("selected"))):
        return
    try:
        sink = _writer()
        if sink.queue.full() or sink.error:
            with sink.lock:
                sink.dropped += 1
            return
        start = time.perf_counter_ns()
        for key, ids in (token_fields or {}).items():
            fields[key] = _token_snapshot(ids)
        for key, value in (text_fields or {}).items():
            fields[key] = _Text(value) if isinstance(value, str) else {"complete": False, "status": "unavailable"}
        mode = os.environ.get("VERL_OMNI_VIDEO_TRACE_MODE", "metadata")
        if mode not in ("metadata", "sample", "full"):
            mode = "metadata"
        frozen = _freeze(fields, mode)
        sink.put(
            {
                "event": name,
                "time_ns": time.time_ns(),
                "mode": mode,
                **(ctx or {}),
                **frozen,
                "capture_ns": time.perf_counter_ns() - start,
            }
        )
    except Exception:
        # Never invoke a potentially blocking logging handler on this thread.
        if _sink is not None:
            _sink.dropped += 1


def export_context():
    """Copy scalar correlation fields for explicit propagation through Ray RPC."""
    return {key: value for key, value in (_context.get() or {}).items() if key in _CONTEXT_KEYS}


@contextlib.contextmanager
def scope(name, *, incoming=None, **identity):
    """Track a task without retaining payloads or suppressing its exceptions."""
    global _request_count
    if not enabled():
        yield
        return
    parent = _context.get()
    ctx = dict(parent or {})
    if not parent:
        if incoming:
            ctx = {
                key: value
                for key, value in incoming.items()
                if key in _CONTEXT_KEYS and isinstance(value, str | int | bool | type(None))
            }
        if "selected" not in ctx:
            _request_count += 1
            try:
                limit = int(os.environ.get("VERL_OMNI_VIDEO_TRACE_MAX_REQUESTS", "32"))
            except ValueError:
                limit = 32
            ctx["selected"] = limit == 0 or 0 < _request_count <= limit
        ctx.setdefault("trace_id", uuid.uuid4().hex)
    for key, value in identity.items():
        if type(value).__module__.split(".")[0] == "numpy" and hasattr(value, "item"):
            value = value.item()
        ctx[key] = value if isinstance(value, str | int | float | bool | type(None)) else type(value).__name__
    token = _context.set(ctx)
    try:
        event(name + ".begin")
        yield
    except BaseException as exc:
        event(name + ".error", error_type=type(exc).__name__, error=str(exc)[:2048])
        raise
    finally:
        event(name + ".end")
        _context.reset(token)


def prompt(name, value, **fields):
    """Snapshot CPU prompt tokens for offline decoding; video mode is independent."""
    if not enabled() or not (_context.get() or {}).get("selected"):
        return
    try:
        ids = value.get("prompt_token_ids", value.get("prompt_ids"))
        data = value.get("multi_modal_data") or {}
        event(
            name,
            token_fields={"prompt_token_snapshot": ids},
            prompt_tokens=len(ids) if ids is not None else None,
            video=data.get("video"),
            mm_uuids=value.get("multi_modal_uuids"),
            mm_processor_kwargs=value.get("mm_processor_kwargs"),
            **fields,
        )
    except Exception:
        if _sink is not None:
            _sink.dropped += 1


def output(name, value):
    """Distinguish empty engine tokens and empty final agent response masks."""
    if not enabled() or not (_context.get() or {}).get("selected"):
        return
    try:
        lengths = {}
        for key in ("token_ids", "prompt_ids", "response_ids", "response_mask", "log_probs", "logprobs"):
            item = getattr(value, key, None)
            lengths[key + "_count"] = len(item) if item is not None else None
        event(
            name,
            token_fields={
                "response_token_snapshot": getattr(value, "token_ids", getattr(value, "response_ids", None)),
                "prompt_token_snapshot": getattr(value, "prompt_ids", None),
            },
            text_fields={"engine_text": getattr(value, "text", None)},
            **lengths,
            finish_reason=getattr(value, "finish_reason", None),
            stop_reason=getattr(value, "stop_reason", None),
            num_preempted=getattr(value, "num_preempted", None),
        )
    except Exception:
        if _sink is not None:
            _sink.dropped += 1


def messages(name, values):
    """Record roles/text parts without traversing images, video frames or audio."""
    if not enabled() or not (_context.get() or {}).get("selected"):
        return
    try:
        is_array = type(values).__module__ == "numpy" and type(values).__name__ == "ndarray"
        if type(values) not in (list, tuple) and not is_array:
            event(name, status="unsupported_messages")
            return
        event(
            name + ".layout",
            message_count=len(values),
            omitted_messages=max(0, len(values) - 32),
            roles=[item.get("role") if isinstance(item, Mapping) else None for item in values[:32]],
        )
        for index, message in enumerate(itertools.islice(values, 32)):
            role, content = message.get("role"), message.get("content")
            if isinstance(content, str):
                event(name, message_index=index, part_index=0, role=role, text_fields={"text": content})
            elif isinstance(content, list | tuple):
                for part_index, part in enumerate(content[:32]):
                    if isinstance(part, Mapping) and part.get("type") == "text":
                        event(
                            name,
                            message_index=index,
                            part_index=part_index,
                            role=role,
                            text_fields={"text": part.get("text")},
                        )
                if len(content) > 32:
                    event(name, message_index=index, role=role, status="content_limit", count=len(content))
            else:
                event(name, message_index=index, role=role, status="unsupported_content")
        if len(values) > 32:
            event(name, status="message_limit", count=len(values))
    except Exception as exc:
        event("trace.observation_error", boundary=name, error_type=type(exc).__name__)


def agent_config(agent):
    """Keep tokenizer/template provenance without calling the tokenizer."""
    if not enabled() or not (_context.get() or {}).get("selected"):
        return
    try:
        tokenizer, processor = getattr(agent, "tokenizer", None), getattr(agent, "processor", None)
        event(
            "agent.template.config",
            tokenizer_name=getattr(tokenizer, "name_or_path", None),
            tokenizer_class=type(tokenizer).__name__,
            processor_class=type(processor).__name__,
            builder_class=type(getattr(agent, "continuous_token_builder", None)).__name__,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            prompt_length=getattr(agent, "prompt_length", None),
            response_length=getattr(agent, "response_length", None),
            text_fields={
                "tokenizer_template": getattr(tokenizer, "chat_template", None),
                "processor_template": getattr(processor, "chat_template", None),
            },
        )
    except Exception as exc:
        event("trace.observation_error", boundary="agent.template.config", error_type=type(exc).__name__)


def merged_output(result, mask, logprobs):
    if not enabled() or not (_context.get() or {}).get("selected"):
        return
    try:
        event(
            "agent.merge.after",
            token_fields={"response_token_snapshot": result.token_ids[-len(mask) :] if mask else []},
            merged_tokens=len(result.token_ids),
            response_mask_count=len(mask),
            logprob_count=len(logprobs) if logprobs is not None else None,
        )
    except Exception as exc:
        event("trace.observation_error", boundary="agent.merge.after", error_type=type(exc).__name__)
