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
"""Standard-library numerical analysis of bounded observations, never model code."""

import base64
import hashlib
import math
import re
import struct
from collections import defaultdict


def decode(snapshot):
    count = snapshot.get("sample_count")
    if not isinstance(count, int) or not 0 <= count <= 4096:
        raise ValueError("missing or invalid sample count")
    payload = snapshot.get("sample_data_b64")
    if not isinstance(payload, str) or len(payload) > 65536:
        raise ValueError("numeric sample bytes unavailable")
    raw = base64.b64decode(payload, validate=True)
    if hashlib.sha256(raw).hexdigest() != snapshot.get("sample_sha256"):
        raise ValueError("numeric sample digest mismatch")
    byteorder = snapshot.get("sample_byteorder")
    if byteorder not in ("little", "big"):
        raise ValueError("sample byte order unavailable")
    prefix = "<" if byteorder == "little" else ">"
    dtype = snapshot["dtype"].removeprefix("torch.")
    formats = {
        "float16": "e",
        "float32": "f",
        "float64": "d",
        "int8": "b",
        "uint8": "B",
        "int16": "h",
        "uint16": "H",
        "int32": "i",
        "uint32": "I",
        "int64": "q",
        "uint64": "Q",
        "bool": "?",
    }
    if dtype == "bfloat16":
        return [
            struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in struct.unpack(prefix + "H" * count, raw)
        ]
    if dtype not in formats:
        raise ValueError(f"unsupported numeric dtype: {dtype}")
    return list(struct.unpack(prefix + formats[dtype] * count, raw))


def numeric_difference(a, b):
    try:
        if a.get("shape") is None or a.get("sample_scheme") != "linspace-flat-c-order-v1":
            raise ValueError("numeric sample coordinates unavailable")
        for key in ("shape", "sample_scheme", "sample_count"):
            if a.get(key) != b.get(key):
                raise ValueError(f"numeric samples not aligned: {key}")
        x, y = decode(a), decode(b)
    except (ValueError, KeyError, TypeError, struct.error) as exc:
        return {"status": "unknown", "reason": str(exc)}
    pairs = [(float(v), float(w)) for v, w in zip(x, y, strict=True) if math.isfinite(v) and math.isfinite(w)]
    result = {
        "status": "sample_metrics",
        "sample_count": len(x),
        "finite_pairs": len(pairs),
        "single_nonfinite": sum(not math.isfinite(v) for v in x),
        "multi_nonfinite": sum(not math.isfinite(v) for v in y),
        "different_elements": sum(v != w for v, w in zip(x, y, strict=True)),
    }
    if pairs:
        errors = [abs(v - w) for v, w in pairs]
        nx, ny = (math.sqrt(sum(v[i] ** 2 for v in pairs)) for i in (0, 1))
        l2 = math.sqrt(sum(error**2 for error in errors))
        result.update(
            max_abs=max(errors),
            mean_abs=sum(errors) / len(errors),
            relative_l2=l2 / nx if nx else None,
            cosine=sum(v * w for v, w in pairs) / (nx * ny) if nx and ny else None,
            abs_exceeds={str(t): sum(error > t for error in errors) for t in (0.001, 0.01, 0.1)},
            single_min=min(v for v, _ in pairs),
            single_max=max(v for v, _ in pairs),
            multi_min=min(w for _, w in pairs),
            multi_max=max(w for _, w in pairs),
        )
    return result


def vision_stages(left, right, trees):
    groups = []
    for side in (left, right):
        grouped = defaultdict(list)
        for row in side["records"]:
            event = row.get("event")
            if event == "worker.vision.tensor":
                key = (
                    row.get("worker_rank"),
                    row.get("feature_index"),
                    row.get("checkpoint"),
                    row.get("occurrence"),
                    row.get("chunk_start"),
                    row.get("chunk_tokens"),
                )
                grouped[key].append((row.get("value"), row))
            elif event == "worker.vision.weights":
                for name, value in row.get("values", {}).items():
                    key = (
                        row.get("worker_rank"),
                        None,
                        "weight." + row.get("kind", "unknown") + "." + name,
                        1,
                        None,
                        None,
                    )
                    grouped[key].append((value, row))
        groups.append(grouped)
    stages = []
    for key in sorted(groups[0].keys() | groups[1].keys(), key=lambda key: (checkpoint_order(key[2]), repr(key))):
        a, b = (group.get(key, []) for group in groups)
        if len(a) != 1 or len(b) != 1:
            result = {"status": "unknown", "reason": "missing_or_repeated_vision_observation"}
        elif key[0] is None or key[2] is None:
            result = {"status": "unknown", "reason": "missing_vision_identity"}
        elif any(len(side.get("routes", [])) != 1 for side in (left, right)):
            result = {"status": "unknown", "reason": "engine_attempts_not_unambiguously_aligned"}
        elif a[0][1].get("comparison_scope") == "batch" or b[0][1].get("comparison_scope") == "batch":
            result = {"status": "unknown", "reason": "batch_context_only: inspect raw attention layout"}
        else:
            result = trees(a[0][0], b[0][0], require_arrays=True)
        stages.append(
            {
                "stage": (
                    f"worker.vision [{key[2]}, rank={key[0]}, feature={key[1]}, "
                    f"occurrence={key[3]}, interval={key[4:]}]"
                ),
                "checkpoint": key[2],
                "worker_rank": key[0],
                "feature_index": key[1],
                "components": {"value": result},
                "sources": {
                    label: [row.get("_source") for _, row in values] for label, values in (("single", a), ("multi", b))
                },
            }
        )
    return stages


def checkpoint_order(name):
    names = ["source", "video.input", "visual.input", "patch_embed.output", "position.output"]
    if name in names:
        return (0, names.index(name), "")
    block = re.match(r"blocks\.(\d+)\.(.+)", name or "")
    if block:
        parts = [
            "input",
            "rotary_pos_emb_cos",
            "rotary_pos_emb_sin",
            "attention_layout",
            "norm1.output",
            "attn.qkv.output",
            "attn.kernel.query",
            "attn.kernel.key",
            "attn.kernel.value",
            "attn.kernel.output",
            "attn.proj.output",
            "attn.output",
            "norm2.output",
            "mlp.linear_fc1.output",
            "mlp.linear_fc2.output",
            "mlp.output",
            "output",
        ]
        return (1, int(block[1]), parts.index(block[2]) if block[2] in parts else 99)
    for index, prefix in enumerate(
        (
            "merger.",
            "merger_list.",
            "visual.output",
            "video.output",
            "cache.write",
            "cache.read",
            "merge.",
            "deepstack.set",
            "deepstack.consume",
            "weight.",
        )
    ):
        if (name or "").startswith(prefix):
            return (index + 2, 0, name)
    return (99, 0, name or "")


def vision_coverage(side):
    ranks = defaultdict(lambda: {"hooks": {}, "observed": [], "gaps": [], "weights": 0, "configs": []})
    for row in side["records"]:
        event, rank = row.get("event", ""), row.get("worker_rank")
        if not event.startswith("worker.vision."):
            continue
        value = ranks[str(rank)]
        if event == "worker.vision.manifest":
            value["hooks"].update(row.get("hooks", {}))
            if row.get("config") not in value["configs"]:
                value["configs"].append(row.get("config"))
        elif event == "worker.vision.gap":
            value["gaps"].append({key: row.get(key) for key in ("checkpoint", "reason", "_source")})
        elif event == "worker.vision.weights":
            value["weights"] += len(row.get("values", {}))
        elif event == "worker.vision.tensor":
            if row.get("checkpoint") not in value["observed"]:
                value["observed"].append(row.get("checkpoint"))
    for value in ranks.values():
        expected = {
            "source",
            "video.input",
            "video.output",
            "visual.input",
            "visual.output",
            "patch_embed.output",
            "merger.output",
            "merge.input_ids",
            "merge.mask",
            "merge.video",
        }
        expected.update(name + ".output" for name in value["hooks"] if name.startswith(("blocks.", "merger_list.")))
        value["missing_checkpoints"] = sorted(expected - set(value["observed"]))
        if any(config and config.get("deepstack_visual_indexes") for config in value["configs"]):
            for prefix in ("deepstack.set.", "deepstack.consume."):
                if not any(name.startswith(prefix) for name in value["observed"]):
                    value["missing_checkpoints"].append(prefix + "*")
    return dict(ranks)


def vision_flow(side, trees):
    groups = defaultdict(lambda: defaultdict(list))
    token_checks = defaultdict(list)
    for row in side["records"]:
        event, rank, feature = row.get("event"), row.get("worker_rank"), row.get("feature_index")
        if event == "worker.vision.tensor":
            name, value = row.get("checkpoint"), row.get("value")
            if name == "merge.input_ids":
                expected = row.get("expected_token_snapshot", {})
                try:
                    ids = decode(value)
                    token_checks[rank].append(expected.get("status") == "complete" and ids == expected.get("ids"))
                except (ValueError, TypeError, KeyError, struct.error):
                    token_checks[rank].append(False)
            if name == "source":
                value = (value or {}).get("pixel_values_videos")
            groups[(rank, feature)][name].append(value)
        elif event in ("worker.encoder.write", "worker.encoder.read") and row.get("modality") == "video":
            groups[(rank, feature)][event.removeprefix("worker.")].append(row.get("embedding"))
    edges = [
        ("source", "video.input"),
        ("video.input", "visual.input"),
        ("visual.output", "video.output"),
        ("video.output", "encoder.write"),
        ("encoder.write", "encoder.read"),
        ("cache.read.part.0", "merge.video"),
    ]
    results = []
    for (rank, feature), values in sorted(groups.items(), key=lambda item: repr(item[0])):
        selected_edges = edges
        if feature is None:
            selected_edges = [
                (name, "deepstack.consume.deepstack_input_embeds_" + name.removeprefix("deepstack.set."))
                for name in values
                if name.startswith("deepstack.set.")
            ]
        for start, end in selected_edges:
            a, b = values[start], values[end]
            if len(a) != 1 or len(b) != 1 or a[0] is None or b[0] is None:
                result = {"status": "unknown", "reason": "missing_or_repeated_flow_observation"}
            elif end == "merge.video" and (not token_checks[rank] or not all(token_checks[rank])):
                result = {"status": "unknown", "reason": "actual_merge_tokens_not_fully_verified"}
            elif a[0].get("shape") != b[0].get("shape"):
                result = {"status": "unknown", "reason": "flow_shape_or_chunk_layout_differs"}
            else:
                result = trees(a[0], b[0], require_arrays=True)
            results.append({"worker_rank": rank, "feature_index": feature, "edge": f"{start} -> {end}", **result})
    return results


def short_metrics(value):
    metrics = value.get("numeric", {})
    available = [item for item in metrics.values() if item.get("status") == "sample_metrics"]
    if not available:
        return "numeric=unavailable"
    parts = []
    for key in ("max_abs", "mean_abs", "relative_l2", "different_elements", "single_nonfinite", "multi_nonfinite"):
        values = [item[key] for item in available if item.get(key) is not None]
        parts.append(f"{key}={max(values):.5g}" if values else f"{key}=unknown")
    return " ".join(parts)


def vision_summary(report):
    stages = report.get("vision_stages", [])
    coverage = report.get("vision_coverage", {})
    if not stages and not any(coverage.values()):
        return []
    lines = ["", "[vision diagnosis: sampled numeric errors, not a corruption verdict]"]
    lines.append(
        "Block metrics are maxima across recorded tensors/ranks; per-tensor metrics remain in comparison.json."
    )
    for side, ranks in coverage.items():
        for rank, value in ranks.items():
            unavailable = {
                key: item.get("reason") for key, item in value["hooks"].items() if item.get("status") != "installed"
            }
            lines.append(
                f"{side} rank={rank}: weights={value['weights']} observed={len(value['observed'])} "
                f"missing={value['missing_checkpoints']} unavailable={unavailable}"
            )
            for gap in value["gaps"][:8]:
                lines.append(f"  gap {gap['checkpoint']}: {gap['reason']}")
            if len(value["gaps"]) > 8:
                lines.append(f"  omitted_gaps={len(value['gaps']) - 8}; see comparison.json")
    for rank in sorted(set(coverage.get("single", {})) | set(coverage.get("multi", {}))):
        a, b = (coverage.get(side, {}).get(rank) for side in ("single", "multi"))
        if not a or not b:
            lines.append(f"rank={rank}: runtime comparison unknown (missing manifest)")
            continue
        lines.append(f"rank={rank}: vision_config={'equal' if a['configs'] == b['configs'] else 'different'}")
        changed = []
        unknown = []
        for name in a["hooks"].keys() | b["hooks"].keys():
            x, y = (value["hooks"].get(name, {}) for value in (a, b))
            if not x.get("source_sha256") or not y.get("source_sha256"):
                unknown.append(name)
            elif any(x.get(key) != y.get(key) for key in ("class", "signature", "source_sha256")):
                changed.append(name)
        lines.append(f"rank={rank}: code_changed={sorted(changed)} code_unknown={sorted(unknown)}")
    first = next(
        (
            row
            for row in stages
            if not row["checkpoint"].startswith("weight.") and row["components"]["value"]["status"] == "different"
        ),
        None,
    )
    lines.append("First observed vision difference: " + (first["stage"] if first else "none observed; check coverage"))
    weights = [row for row in stages if row["checkpoint"].startswith("weight.")]
    for status in ("equal", "sample_equal", "different", "unknown"):
        lines.append(
            f"weight comparisons {status}: {sum(row['components']['value']['status'] == status for row in weights)}"
        )
    changed = [row for row in weights if row["components"]["value"]["status"] == "different"]
    for row in changed[:64]:
        result = row["components"]["value"]
        lines.append(f"{row['stage']}: {result['status']} {short_metrics(result)}")
    if len(changed) > 64:
        lines.append(f"omitted_weight_difference_details={len(changed) - 64}; see comparison.json")
    unknown = [row for row in weights if row["components"]["value"]["status"] == "unknown"]
    for row in unknown[:8]:
        result = row["components"]["value"]
        reason = str(result.get("reason") or result.get("gaps") or "unspecified")
        lines.append(f"{row['stage']}: unknown reason={reason[:512]}")
    if len(unknown) > 8:
        lines.append(f"omitted_weight_unknown_details={len(unknown) - 8}; see comparison.json")
    groups = defaultdict(list)
    for row in stages:
        if not row["checkpoint"].startswith("weight."):
            groups[row["checkpoint"]].append(row)
    blocks = defaultdict(list)
    for name, rows in groups.items():
        match = re.match(r"blocks\.(\d+)\.", name)
        if match:
            blocks[int(match[1])].extend(rows)
    first_block = next(
        (
            index
            for index in sorted(blocks)
            if any(row["components"]["value"]["status"] == "different" for row in blocks[index])
        ),
        None,
    )
    for index, rows in sorted(blocks.items()):
        first_change = next(
            (row["checkpoint"] for row in rows if row["components"]["value"]["status"] == "different"), None
        )
        counts = {
            status: sum(row["components"]["value"]["status"] == status for row in rows)
            for status in ("equal", "sample_equal", "different", "unknown")
        }
        numeric = {
            str(i) + path: value
            for i, row in enumerate(rows)
            for path, value in row["components"]["value"].get("numeric", {}).items()
        }
        lines.append(
            f"block {index:02d}: first_difference={first_change} counts={counts} {short_metrics({'numeric': numeric})}"
        )
    for name in sorted(groups, key=checkpoint_order):
        match = re.match(r"blocks\.(\d+)\.", name)
        if match and int(match[1]) != first_block:
            continue
        rows = groups[name]
        statuses = defaultdict(list)
        for row in rows:
            statuses[row["components"]["value"]["status"]].append(row["worker_rank"])
        numeric = {
            str(i) + path: value
            for i, row in enumerate(rows)
            for path, value in row["components"]["value"].get("numeric", {}).items()
        }
        lines.append(f"{name}: {dict(statuses)} {short_metrics({'numeric': numeric})}")
    for side, checks in report.get("vision_flow", {}).items():
        lines.append(f"[{side} within-run video flow; video.input -> visual.input includes dtype conversion]")
        for item in checks:
            lines.append(
                f"rank={item['worker_rank']} feature={item['feature_index']} {item['edge']}: "
                f"{item['status']} {short_metrics(item)} {item.get('reason', '')}"
            )
    lines.append(
        "Batch attention layouts are context-only. Numeric errors describe sampled elements; weights are sampled too."
    )
    return lines
