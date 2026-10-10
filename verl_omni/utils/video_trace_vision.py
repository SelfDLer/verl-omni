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
"""Selected-request vision probes for the pinned Qwen3 Omni eager encoder path.

No replay, RNG changes, global patches or replacement model computations. Batch
membership is obtained from the runner and checked against the model's grids.
Unsupported layouts are recorded as gaps rather than matched by tensor shape.
"""

import functools
import hashlib
import inspect
import math
import sys
from collections import Counter

from verl_omni.utils import video_trace_numeric as numeric


def grid(value):
    if hasattr(value, "data") and type(value).__name__ == "MultiModalFieldElem":
        value = value.data
    if hasattr(value, "shape"):
        if math.prod(value.shape) > 384:
            raise ValueError("grid exceeds 128 items")
        if str(getattr(value, "device", "cpu")) != "cpu":
            import os

            if os.environ.get("VERL_OMNI_VIDEO_TRACE_DEVICE_SAMPLE", "0") != "1":
                raise ValueError("device grid reads disabled")
        value = value.tolist()
    if len(value) == 3 and all(isinstance(item, int) for item in value):
        value = [value]
    if (
        not value
        or len(value) > 128
        or any(len(row) != 3 or any(not isinstance(item, int) or item <= 0 for item in row) for row in value)
    ):
        raise ValueError("unsupported video grid")
    return value


class VisionObserver:
    def __init__(self, observer):
        self.observer = observer
        self.members = []
        self.cursor = 0
        self.current = []
        self.visual_active = False
        self.pending_consume = []
        self.hooks = {}
        self.owner = None
        self.visual = None
        self.source_cache = {}

    def fail(self, checkpoint, error):
        for entry, *_ in self.observer.active:
            self.observer.emit(entry, "worker.vision.gap", checkpoint=checkpoint, reason=str(error)[:512])

    def safe(self, function, *args):
        try:
            return function(*args)
        except Exception as exc:
            self.fail(function.__name__, exc)
            return None

    def emit(self, entry, feature_index, checkpoint, value, **fields):
        counts = entry.setdefault("vision_counts", Counter())
        key = f"{feature_index}:{checkpoint}"
        counts[key] += 1
        # Preserve repeated observations and their ordinal; never choose one
        # silently when the same feature is encoded repeatedly.
        if counts[key] > 4 or entry["events"] >= 1024:
            if counts[key] == 5:
                self.observer.emit(entry, "worker.vision.gap", checkpoint=checkpoint, reason="checkpoint_limit")
            return
        self.observer.emit(
            entry,
            "worker.vision.tensor",
            feature_index=feature_index,
            modality="video",
            checkpoint=checkpoint,
            occurrence=counts[key],
            value=numeric.tree(value),
            **fields,
        )

    def prepare(self):
        self.members, self.cursor = [], 0
        self.pending_consume = list(self.observer.active)
        for entry, *_ in self.observer.active:
            if entry.get("vision_manifest"):
                continue
            entry["vision_manifest"] = True
            config = {}
            if self.visual is not None:
                for key in (
                    "spatial_merge_size",
                    "temporal_patch_size",
                    "deepstack_visual_indexes",
                    "attn_backend",
                    "apply_vit_abs_pos_embed",
                    "hidden_size",
                    "tp_size",
                    "training",
                ):
                    value = getattr(self.visual, key, None)
                    config[key] = (
                        value if isinstance(value, str | int | bool | list | tuple | type(None)) else str(value)
                    )
            for start in range(0, max(1, len(self.hooks)), 32):
                self.observer.emit(
                    entry,
                    "worker.vision.manifest",
                    config=config,
                    hooks=dict(list(self.hooks.items())[start : start + 32]),
                )
            if self.visual is None:
                continue
            # Probe actual loaded vision parameters and buffers at request time,
            # including every block/merger, not just checkpoint filenames.
            for kind, iterator in (
                ("parameter", self.visual.named_parameters()),
                ("buffer", self.visual.named_buffers()),
            ):
                batch = {}
                for index, (name, value) in enumerate(iterator):
                    if index >= 768:
                        self.fail("weights", f"{kind} limit: 768")
                        break
                    try:
                        batch[name] = numeric.snapshot(value, limit=64)
                    except Exception as exc:
                        batch[name] = {"omitted": str(exc)[:256]}
                    if len(batch) == 8:
                        self.observer.emit(entry, "worker.vision.weights", kind=kind, values=batch)
                        batch = {}
                if batch:
                    self.observer.emit(entry, "worker.vision.weights", kind=kind, values=batch)

    def batch(self, arguments, result):
        hashes, kwargs, refs = result
        if not (len(hashes) == len(kwargs) == len(refs)) or len(hashes) > 128:
            raise ValueError("unsupported encoder batch membership")
        self.members, self.cursor = [], 0
        for identifier, (modality, data), (request_id, _) in zip(hashes, kwargs, refs, strict=True):
            if modality != "video":
                continue
            state = self.observer.runner.requests[request_id]
            indices = [i for i, item in enumerate(state.mm_features) if item.identifier == identifier]
            if len(indices) != 1:
                raise ValueError("ambiguous video feature identifier within request")
            grids = grid(data["video_grid_thw"])
            if len(grids) != 1:
                raise ValueError("expected one video grid per feature")
            member = {
                "request_id": request_id,
                "identifier": identifier,
                "feature_index": indices[0],
                "grid": grids[0],
                "entry": self.observer.requests.get(request_id),
            }
            self.members.append(member)
            if member["entry"]:
                self.emit(
                    member["entry"],
                    indices[0],
                    "source",
                    numeric.tree(data, full_cpu=True),
                    grid=grids[0],
                    encoder_item_index=len(self.members) - 1,
                )

    def video_begin(self, arguments):
        value = arguments["video_input"]
        grids = grid(value["video_grid_thw"])
        candidates = self.members[self.cursor : self.cursor + len(grids)]
        self.cursor += len(grids)
        if len(candidates) != len(grids) or [item["grid"] for item in candidates] != grids:
            raise ValueError("video batch grid/order does not match scheduled features")
        self.current = candidates
        self.tensor("video.input", value.get("pixel_values_videos", value.get("video_embeds")))

    def video_end(self, arguments, result):
        if not isinstance(result, list | tuple) or len(result) != len(self.current):
            raise ValueError("video outputs not split one-to-one by scheduled feature")
        for member, value in zip(self.current, result, strict=True):
            if member["entry"]:
                self.emit(member["entry"], member["feature_index"], "video.output", value)

    def tensor(self, checkpoint, value, **fields):
        if not self.current:
            return
        if isinstance(value, tuple):
            value = value[0]
        if not hasattr(value, "shape") or not value.shape:
            raise ValueError(f"{checkpoint}: missing tensor or unsupported layout")
        raw = [math.prod(member["grid"]) for member in self.current]
        merge = self.visual.spatial_merge_size**2
        merged = [count // merge for count in raw]
        if value.shape[0] == sum(raw):
            sizes = raw
        elif value.shape[0] == sum(merged) and all(count % merge == 0 for count in raw):
            sizes = merged
        else:
            raise ValueError(f"{checkpoint}: token layout {list(value.shape)} inconsistent with grids")
        offset = 0
        for member, count in zip(self.current, sizes, strict=True):
            if member["entry"]:
                self.emit(
                    member["entry"],
                    member["feature_index"],
                    checkpoint,
                    value[offset : offset + count],
                    batch_shape=list(value.shape),
                    batch_offset=offset,
                    batch_members=[
                        {key: item[key] for key in ("request_id", "feature_index", "grid")} for item in self.current
                    ],
                    **fields,
                )
            offset += count

    def visual_begin(self, arguments):
        grids = grid(arguments["grid_thw"])
        if grids != [member["grid"] for member in self.current]:
            raise ValueError("visual grid differs from video batch; sharded/reordered layout not inferred")
        self.visual_active = True
        self.tensor("visual.input", arguments["x"])

    def block_begin(self, checkpoint, arguments):
        value = next(iter(arguments.values()))
        self.tensor(checkpoint + ".input", value)
        # Actual attention boundaries/rotary inputs for each block are captured
        # once per block. Sequence lengths refer to the whole encoder batch.
        if checkpoint.startswith("blocks.") and checkpoint.count(".") == 1:
            for key in ("rotary_pos_emb_cos", "rotary_pos_emb_sin"):
                if arguments.get(key) is not None:
                    self.tensor(checkpoint + "." + key, arguments[key])
            for member in self.current:
                if member["entry"]:
                    self.emit(
                        member["entry"],
                        member["feature_index"],
                        checkpoint + ".attention_layout",
                        {key: arguments.get(key) for key in ("cu_seqlens", "sequence_lengths", "max_seqlen")},
                        comparison_scope="batch",
                    )

    def kernel_begin(self, checkpoint, arguments):
        for key in ("query", "key", "value"):
            value = arguments.get(key)
            if value is None or len(value.shape) != 4 or value.shape[0] != 1:
                raise ValueError("attention kernel layout is not [1, tokens, heads, head_dim]")
            self.tensor(checkpoint + "." + key, value[0])

    def kernel_end(self, checkpoint, result):
        if len(result.shape) != 4 or result.shape[0] != 1:
            raise ValueError("attention output layout is not [1, tokens, heads, head_dim]")
        self.tensor(checkpoint + ".output", result[0])

    def deepstack(self, arguments, result=None, consume=False):
        spans = self.pending_consume if consume else self.observer.active
        if not spans:
            return
        if consume:
            values = getattr(result, "tensors", result)
            if values is None:
                values = {}
            if not hasattr(values, "items"):
                raise ValueError("unsupported deepstack result")
            values = list(values.items())
        else:
            value = arguments["deepstack_input_embeds"]
            values = [(str(i), value[i]) for i in range(min(value.shape[0], 8))]
            if value.shape[0] > 8:
                self.fail("deepstack", "level limit: 8")
        for entry, _, offset, count, start in spans:
            for name, value in values:
                if value.shape[0] < offset + count:
                    raise ValueError("deepstack token interval exceeds tensor")
                self.emit(
                    entry,
                    None,
                    ("deepstack.consume." if consume else "deepstack.set.") + str(name),
                    value[offset : offset + count],
                    chunk_start=start,
                    chunk_tokens=count,
                )
        if consume:
            self.pending_consume = []

    def cache_parts(self, entry, feature_index, value, write):
        if self.visual is None or value is None or len(value.shape) != 2:
            return
        levels = getattr(self.visual, "deepstack_visual_indexes", None) or []
        config = getattr(self.owner, "config", None)
        width = getattr(getattr(config, "vision_config", None), "out_hidden_size", None)
        if not width or value.shape[1] != width * (len(levels) + 1):
            self.observer.emit(
                entry, "worker.vision.gap", checkpoint="cache.parts", reason="unknown feature channel layout"
            )
            return
        for index in range(len(levels) + 1):
            self.emit(
                entry,
                feature_index,
                f"cache.{'write' if write else 'read'}.part.{index}",
                value[:, index * width : (index + 1) * width],
                branch="main" if index == 0 else f"deepstack.block.{levels[index - 1]}",
            )

    def merge(self, arguments, result=None):
        for entry, state, offset, count, start in self.observer.active:
            if result is None:
                self.emit(
                    entry,
                    None,
                    "merge.input_ids",
                    numeric.snapshot(arguments["input_ids"][offset : offset + count], limit=4096),
                    chunk_start=start,
                    chunk_tokens=count,
                    token_fields={"expected_token_snapshot": state.prompt_token_ids[start : start + count]},
                )
                mask = arguments.get("is_multimodal")
                if mask is not None:
                    self.emit(
                        entry,
                        None,
                        "merge.mask",
                        numeric.snapshot(mask[offset : offset + count], limit=4096),
                        chunk_start=start,
                        chunk_tokens=count,
                    )
                self.observer.emit(
                    entry,
                    "worker.vision.merge_layout",
                    embedding_items=[
                        {"shape": list(value.shape), "modality": getattr(value, "modality", None)}
                        for value in (arguments.get("multimodal_embeddings") or [])[:32]
                    ],
                )
            else:
                video_token = getattr(getattr(self.owner, "config", None), "video_token_id", None)
                if video_token is None:
                    raise ValueError("video token ID missing from model config")
                ids = state.prompt_token_ids[start : start + count]
                for index, feature in enumerate(state.mm_features):
                    if feature.modality != "video":
                        continue
                    begin, end = feature.mm_position.offset, feature.mm_position.offset + feature.mm_position.length
                    positions = [
                        offset + i for i, token in enumerate(ids) if token == video_token and begin <= start + i < end
                    ]
                    if positions:
                        self.emit(
                            entry,
                            index,
                            "merge.video",
                            numeric.snapshot(result, row_indices=positions),
                            chunk_start=start,
                            chunk_tokens=count,
                            alignment="prompt_video_token_positions; validate merge.input_ids against expected tokens",
                        )

    def coverage(self):
        for entry, *_ in self.observer.active:
            counts = list(entry.get("vision_counts", {}).items())
            for start in range(0, max(1, len(counts)), 64):
                self.observer.emit(
                    entry,
                    "worker.vision.coverage",
                    observed=dict(counts[start : start + 64]),
                    note="Installed hooks are not proof of execution; absent stages remain unknown.",
                )

    def attach(self, owner, method, name, before=None, after=None, context=None):
        try:
            original = getattr(owner, method)
            signature = inspect.signature(original)
            if inspect.iscoroutinefunction(original):
                raise TypeError("async vision methods are unsupported")
            if getattr(owner, "_compiled_call_impl", None) is not None or hasattr(
                original, "_torchdynamo_orig_callable"
            ):
                raise TypeError("compiled method not instrumented; use coverage to detect missing stages")
            function = getattr(original, "__func__", original)
            if function not in self.source_cache:
                try:
                    self.source_cache[function] = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
                except (OSError, TypeError):
                    self.source_cache[function] = None
            source_hash = self.source_cache[function]

            @functools.wraps(original)
            def observed(*args, **kwargs):
                active = self.observer.active or (name == "deepstack.consume" and self.pending_consume)
                if not numeric.enabled() or not active:
                    return original(*args, **kwargs)
                torch = sys.modules.get("torch")
                if torch is not None and getattr(torch, "compiler", None) is not None and torch.compiler.is_compiling():
                    return original(*args, **kwargs)
                if context == "internal" and not self.visual_active:
                    return original(*args, **kwargs)
                previous, previous_visual = self.current, self.visual_active
                if context == "video":
                    self.current, self.visual_active = [], False
                if context == "visual":
                    self.visual_active = False
                try:
                    arguments = signature.bind(*args, **kwargs).arguments
                except Exception as exc:
                    self.fail(name, exc)
                    self.current, self.visual_active = previous, previous_visual
                    return original(*args, **kwargs)
                try:
                    if before:
                        self.safe(before, arguments)
                    result = original(*args, **kwargs)
                    if after and (context != "visual" or self.visual_active):
                        self.safe(after, arguments, result)
                    return result
                finally:
                    if context in ("video", "visual"):
                        self.current, self.visual_active = previous, previous_visual

            setattr(owner, method, observed)
            self.hooks[name] = {
                "status": "installed",
                "class": f"{type(owner).__module__}.{type(owner).__qualname__}",
                "signature": str(signature),
                "source_sha256": source_hash,
            }
        except Exception as exc:
            self.hooks[name] = {"status": "unavailable", "reason": str(exc)[:512]}

    def install(self):
        candidates, seen = [self.observer.runner.model], set()
        for _ in range(8):
            if not candidates:
                break
            item = candidates.pop(0)
            if id(item) in seen:
                continue
            seen.add(id(item))
            if hasattr(item, "visual") and hasattr(item, "_process_video_input"):
                self.owner, self.visual = item, item.visual
                break
            candidates.extend(
                getattr(item, key) for key in ("thinker", "model", "_orig_mod") if getattr(item, key, None) is not None
            )
        if self.visual is None:
            raise ValueError("Qwen video processor/visual instance not found")
        self.attach(self.observer.runner, "_batch_mm_inputs_from_scheduler", "batch", after=self.batch)
        self.attach(self.owner, "_process_video_input", "video", self.video_begin, self.video_end, "video")
        self.attach(
            self.visual, "forward", "visual", self.visual_begin, lambda a, r: self.tensor("visual.output", r), "visual"
        )
        modules = [("patch_embed", self.visual.patch_embed), ("merger", self.visual.merger)]
        blocks = list(self.visual.blocks)
        if len(blocks) > 64:
            raise ValueError("vision block limit: 64")
        for index, block in enumerate(blocks):
            name = f"blocks.{index:02d}"
            modules.append((name, block))
            modules.extend(
                (name + "." + key, getattr(block, key))
                for key in ("norm1", "attn", "norm2", "mlp")
                if hasattr(block, key)
            )
            for parent, children in (("attn", ("qkv", "proj")), ("mlp", ("linear_fc1", "linear_fc2"))):
                owner = getattr(block, parent, None)
                modules.extend(
                    (name + "." + parent + "." + key, getattr(owner, key))
                    for key in children
                    if owner is not None and hasattr(owner, key)
                )
            attention = getattr(getattr(block, "attn", None), "attn", None)
            if attention is not None:
                label = name + ".attn.kernel"
                self.attach(
                    attention,
                    "forward",
                    label,
                    before=lambda args, n=label: self.kernel_begin(n, args),
                    after=lambda args, result, n=label: self.kernel_end(n, result),
                    context="internal",
                )
        modules.extend(
            (f"merger_list.{index}", item) for index, item in enumerate(getattr(self.visual, "merger_list", []))
        )
        for name, module in modules:
            self.attach(
                module,
                "forward",
                name,
                before=lambda args, n=name: self.block_begin(n, args)
                if n.startswith("blocks.") and n.count(".") == 1
                else None,
                after=lambda args, result, n=name: self.tensor(n + ".output", result),
                context="internal",
            )
        self.attach(
            self.visual,
            "fast_pos_embed_interpolate",
            "position",
            after=lambda a, r: self.tensor("position.output", r),
            context="internal",
        )
        self.attach(self.owner, "_set_deepstack_input_embeds", "deepstack.set", after=self.deepstack)
        self.attach(
            self.owner,
            "_get_deepstack_input_embeds",
            "deepstack.consume",
            after=lambda a, r: self.deepstack(a, r, True),
        )
        self.attach(self.owner, "embed_input_ids", "merge", before=self.merge, after=self.merge)
