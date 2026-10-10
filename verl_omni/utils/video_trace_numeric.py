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
"""Bounded numeric snapshots; accelerator reads require explicit opt-in."""

import contextlib
import math
import os

from verl_omni.utils import video_trace as trace


def enabled():
    return (
        trace.enabled()
        and os.environ.get("VERL_OMNI_VIDEO_TRACE_STAGE") == "worker"
        and os.environ.get("VERL_OMNI_VIDEO_TRACE_VISION", "0") == "1"
    )


def snapshot(value, limit=None, full_cpu=False, row_indices=None):
    import numpy as np

    info = trace._array(value, "metadata")
    if info is None:
        return value
    device = info["device"] != "cpu"
    if device and os.environ.get("VERL_OMNI_VIDEO_TRACE_DEVICE_SAMPLE", "0") != "1":
        return {**info, "digest_status": "skipped_non_cpu"}
    shape = tuple(value.shape)
    if row_indices is not None:
        if len(shape) != 2 or len(row_indices) > 32768 or any(i < 0 or i >= shape[0] for i in row_indices):
            raise ValueError("unsupported logical row selection")
        shape = (len(row_indices), shape[1])
        info["shape"] = list(shape)
        info["row_selection"] = "prompt_video_token_positions"
    count = math.prod(shape)
    limit = min(4096, max(1, int(limit or os.environ.get("VERL_OMNI_VIDEO_TRACE_NUMERIC_SAMPLES", "1024"))))
    indices = np.linspace(0, count - 1, min(count, limit), dtype=np.int64) if count else np.array([], dtype=np.int64)
    coords = np.unravel_index(indices, shape) if count and value.ndim else ()
    if row_indices is not None and coords:
        coords = (np.asarray(row_indices, dtype=np.int64)[coords[0]], *coords[1:])
    if trace._is_torch_tensor(value):
        import torch

        with trace.device_sample(value) if device else contextlib.nullcontext():
            coords = tuple(torch.tensor(axis.copy(), device=value.device) for axis in coords)
            selected = value.detach()[coords] if coords else value.detach()
            if not count:
                selected = selected.reshape(-1)[:0]
            raw = selected.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    else:
        if value.dtype.hasobject:
            return {**info, "digest_status": "skipped_object_array"}
        selected = value[coords] if coords else value
        if not count:
            selected = selected.reshape(-1)[:0]
        raw = selected.tobytes(order="C")
    info.update(
        digest_status="sample",
        sample_count=len(indices),
        sample_scheme="linspace-flat-c-order-v1",
        fingerprint=trace._NumericSample(raw, "sample_sha256"),
        device_read=device,
    )
    if full_cpu and not device and row_indices is None:
        full = trace._array(value, "full")
        info["full_cpu"] = full
    return info


def tree(value, depth=0, full_cpu=False):
    if depth > 5:
        return {"omitted": "numeric_depth_limit"}
    if trace._array(value, "metadata") is not None:
        return snapshot(value, full_cpu=full_cpu)
    if type(value).__name__ == "MultiModalFieldElem":
        return tree(value.data, depth + 1, full_cpu)
    if isinstance(value, dict) or hasattr(value, "items"):
        from itertools import islice

        result = {str(key): tree(item, depth + 1, full_cpu) for key, item in islice(value.items(), 32)}
        if len(value) > 32:
            result["omitted"] = "numeric_field_limit"
        return result
    if isinstance(value, list | tuple):
        result = [tree(item, depth + 1, full_cpu) for item in value[:32]]
        if len(value) > 32:
            result.append({"omitted": "numeric_item_limit"})
        return result
    return value
