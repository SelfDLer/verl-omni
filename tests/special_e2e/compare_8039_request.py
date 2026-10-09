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
"""Compare one dataset question across two runs using raw video-v2 observations."""

import argparse
import hashlib
import json
import sys
from array import array
from collections import defaultdict
from pathlib import Path

_STAGES = (
    ("agent.source.message", ("text",)),
    ("agent.template.message", ("text",)),
    ("agent.prompt", ("prompt",)),
    ("agent.dispatch", ("prompt", "video", "processor", "sampling")),
    ("server.receive", ("prompt", "video", "processor", "sampling")),
    ("strategy.submit", ("prompt", "video", "processor", "sampling")),
    ("frontend.build.before", ("prompt", "video", "processor")),
    ("frontend.input.before", ("prompt", "video", "processor")),
    ("frontend.tokens.before", ("prompt", "video", "processor")),
    ("frontend.tokens.after", ("prompt", "video", "processor")),
    ("frontend.tokens.result", ("prompt",)),
    ("frontend.tokens.features", ("features",)),
    ("frontend.input.after", ("prompt", "video", "processor")),
    ("frontend.input.result", ("prompt",)),
    ("frontend.input.features", ("features",)),
    ("frontend.build.after", ("prompt", "video", "processor")),
    ("frontend.build.result", ("prompt",)),
    ("frontend.build.features", ("features",)),
    ("strategy.result", ("response",)),
    ("server.result", ("response",)),
    ("agent.result", ("response",)),
    ("agent.merge.before", ("response",)),
    ("agent.merge.after", ("response",)),
    ("agent.output", ("response",)),
)
_IGNORED = {"object_id", "stride", "strides_bytes", "device", "identifier", "mm_hash"}
_OMITTED = {"omitted", "omitted_items", "omitted_fields", "unsupported"}


def load_request(paths, sample_key, trace_id=None):
    files = set()
    for value in paths:
        path = Path(value)
        files.update(p.resolve() for p in (path.glob("video-v2-*.jsonl") if path.is_dir() else [path]))
    if not files:
        raise ValueError("No video-v2-*.jsonl files found")
    groups, writers, installs, warnings = defaultdict(list), {}, [], []
    worker_diagnostics = []
    worker_inventory = defaultdict(lambda: defaultdict(int))
    seen = {}
    for path in sorted(files):
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                source = f"{path}:{line_number}"
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected a JSON object")
                except ValueError as exc:
                    warnings.append(f"{source}: {exc}")
                    continue
                writer = (row.get("host"), row.get("pid"), row.get("writer_id"))
                dropped = row.get("dropped_events", 0)
                if type(dropped) is not int or dropped < 0:
                    warnings.append(f"{source}: invalid dropped_events")
                else:
                    writers[writer] = max(writers.get(writer, 0), dropped)
                if row.get("event") == "trace.install":
                    installs.append({**row, "_source": source})
                if row.get("event", "").startswith("worker.") or (
                    row.get("event") == "trace.install" and row.get("boundary", "").startswith("worker")
                ):
                    worker_inventory[str(path)][row["event"]] += 1
                if row.get("event") in {
                    "trace.worker_unmatched",
                    "trace.observation_error",
                    "trace.worker_registration_evicted",
                }:
                    worker_diagnostics.append({**row, "_source": source})
                if row.get("event") in {
                    "trace.observation_error",
                    "trace.worker_limit",
                    "trace.worker_chunk_limit",
                    "trace.worker_layout_unknown",
                    "trace.worker_registration_evicted",
                }:
                    warnings.append(f"{source}: {row['event']}: {row.get('error', '')}")
                if row.get("sample_key") is None or str(row["sample_key"]) != sample_key:
                    continue
                if not row.get("trace_id"):
                    warnings.append(f"{source}: matching sample has no trace_id")
                    continue
                if row.get("writer_id") is not None and row.get("seq") is not None:
                    identity = (*writer, row["seq"])
                    if identity in seen:
                        if seen[identity] != row:
                            raise ValueError(f"Conflicting records for writer/sequence at {source}")
                        continue
                    seen[identity] = row
                groups[row["trace_id"]].append({**row, "_source": source})
    if trace_id is None and len(groups) != 1:
        raise ValueError(
            f"sample_key={sample_key!r}: expected one trace; found {len(groups)}: {sorted(groups)}. "
            "Collect both hosts, check the trace selection limit, or select a trace ID explicitly."
        )
    trace_id = trace_id or next(iter(groups))
    if trace_id not in groups:
        raise ValueError(f"trace_id={trace_id!r} not found for sample_key={sample_key!r}")
    records = groups[trace_id]
    identities = {tuple(row.get(key) for key in ("sample_index", "video_id", "question_id")) for row in records}
    if len(identities) != 1:
        raise ValueError(f"Conflicting sample identities inside trace {trace_id}")
    attempts = [row for row in records if row.get("event") == "server.receive"]
    return {
        "trace_id": trace_id,
        "identity": {key: records[0].get(key) for key in ("sample_key", "sample_index", "video_id", "question_id")},
        "routes": [
            {key: row.get(key) for key in ("host", "pid", "replica_rank", "request_id", "call_id", "seq")}
            for row in attempts
        ],
        "records": records,
        "warnings": warnings,
        "dropped_events": sum(writers.values()),
        "frontend_installation": installs,
        "worker_diagnostics": worker_diagnostics,
        "input_files": [str(path) for path in sorted(files)],
        "worker_file_inventory": {path: dict(counts) for path, counts in worker_inventory.items()},
    }


def _snapshot_ids(snapshot):
    if not isinstance(snapshot, dict) or snapshot.get("status") != "complete":
        return None, (snapshot or {}).get("status", "missing_snapshot") if isinstance(
            snapshot, dict
        ) else "missing_snapshot"
    ids = snapshot.get("ids")
    if not isinstance(ids, list) or len(ids) != snapshot.get("count") or any(type(v) is not int for v in ids):
        return None, "invalid_token_ids"
    try:
        packed = array("q", ids)
    except (OverflowError, ValueError):
        return None, "invalid_token_ids"
    if sys.byteorder != "little":
        packed.byteswap()
    if snapshot.get("encoding") != "int64-le" or hashlib.sha256(packed.tobytes()).hexdigest() != snapshot.get("sha256"):
        return None, "invalid_token_digest"
    return ids, None


def _tokens(left, right, field, tokenizer):
    a, a_error = _snapshot_ids(left.get(field))
    b, b_error = _snapshot_ids(right.get(field))
    if a_error or b_error:
        return {"status": "unknown", "single": a_error or "complete", "multi": b_error or "complete"}
    result = {"status": "equal" if a == b else "different", "single_count": len(a), "multi_count": len(b)}
    if a != b:
        index = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
        result["first_token_difference"] = index
        for label, ids in (("single", a), ("multi", b)):
            window = ids[max(0, index - 6) : index + 7]
            result[label + "_ids_near_difference"] = window
            if tokenizer is not None:
                try:
                    result[label + "_text_near_difference"] = tokenizer.decode(window, skip_special_tokens=False)
                except Exception as exc:
                    result[label + "_decode_error"] = str(exc)
    return result


def _flatten(value, path, values, unknown, arrays):
    if isinstance(value, dict) and "digest_status" in value:
        values[path + ".shape"] = value.get("shape")
        values[path + ".dtype"] = value.get("dtype")
        arrays[path] = value
    elif isinstance(value, dict) and "complete" in value and "characters" in value:
        if value.get("complete") and value.get("sha256"):
            values[path] = {"characters": value["characters"], "sha256": value["sha256"]}
        else:
            unknown[path] = "incomplete_text"
    elif isinstance(value, dict):
        for key in sorted(value):
            if key in _IGNORED:
                continue
            if key in _OMITTED:
                unknown[path] = f"{key}: {value[key]}"
            else:
                _flatten(value[key], f"{path}.{key}", values, unknown, arrays)
    elif isinstance(value, list):
        if any(isinstance(item, dict) and "omitted_items" in item for item in value):
            unknown[path] = "omitted_items"
            return
        values[path + ".length"] = len(value)
        for index, item in enumerate(value):
            _flatten(item, f"{path}[{index}]", values, unknown, arrays)
    else:
        values[path] = value


def _payload(row, component):
    if component == "multimodal":
        return row.get("multimodal_features"), None if "multimodal_features" in row else "missing_field"
    if component == "features":
        if "video_features" in row:
            features = row["video_features"]
            if not isinstance(features, list) or not features:
                return None, "no_video_features"
            if any(not isinstance(item, dict) or item.get("data") is None for item in features):
                return None, "cache_omitted_or_missing_feature_data"
            return [item["data"] for item in features], None
        value = row.get("video_kwargs")
        return value, "cache_omitted_or_missing_feature_data" if value is None else None
    field = {"video": "video", "audio": "audio", "processor": "mm_processor_kwargs", "sampling": "sampling_params"}[
        component
    ]
    if field not in row:
        return None, "missing_field"
    value = row[field]
    if component == "video" and value is None:
        return None, "cache_omitted_or_missing_video"
    return value, None


def _trees(a, b, *, require_arrays=False):
    av, au, aa, bv, bu, ba = {}, {}, {}, {}, {}, {}
    _flatten(a, "$", av, au, aa)
    _flatten(b, "$", bv, bu, ba)
    if require_arrays:
        if not aa:
            au["$"] = "no_array_snapshot"
        if not ba:
            bu["$"] = "no_array_snapshot"
        for values, unknown in ((av, au), (bv, bu)):
            for path, value in values.items():
                if value is None and path.endswith((".data", ".pixel_values_videos")):
                    unknown[path] = "cache_omitted_or_missing_data"
    differences = []
    gaps = {"single": au, "multi": bu}
    sampled = False
    for path in sorted(av.keys() | bv.keys()):
        # Truncated/unsupported subtrees cannot establish structural differences.
        if any(path == p or path.startswith((p + ".", p + "[")) for p in au.keys() | bu.keys()):
            continue
        if path not in av or path not in bv or av[path] != bv[path]:
            differences.append({"path": path, "single": av.get(path, "<missing>"), "multi": bv.get(path, "<missing>")})
    for path in sorted(aa.keys() | ba.keys()):
        x, y = aa.get(path, {}), ba.get(path, {})
        mode = x.get("digest_status")
        if mode not in ("full", "sample") or mode != y.get("digest_status"):
            au[path + ".content"] = x.get("digest_status", "missing_array")
            bu[path + ".content"] = y.get("digest_status", "missing_array")
            continue
        if mode == "sample" and (x.get("sample_scheme"), x.get("sample_count")) != (
            y.get("sample_scheme"),
            y.get("sample_count"),
        ):
            au[path + ".content"] = bu[path + ".content"] = "incompatible_sample_scheme"
            continue
        field = "sha256" if mode == "full" else "sample_sha256"
        if not x.get(field) or not y.get(field):
            au[path + ".content"] = bu[path + ".content"] = "missing_digest"
            continue
        sampled |= mode == "sample"
        if x[field] != y[field]:
            differences.append({"path": path + "." + field, "single": x[field], "multi": y[field]})
    status = "different" if differences else "unknown" if au or bu else "sample_equal" if sampled else "equal"
    return {"status": status, "differences": differences, "gaps": gaps}


def _compare_stage(left, right, stage, components, tokenizer):
    a = [row for row in left["records"] if row.get("event") == stage]
    b = [row for row in right["records"] if row.get("event") == stage]
    references = {label: [row["_source"] for row in rows] for label, rows in (("single", a), ("multi", b))}
    is_message = components == ("text",)
    reason = None
    if not a or not b:
        reason = "missing_event"
    elif is_message and any(row.get("status") or not isinstance(row.get("text"), dict) for row in a + b):
        reason = "incomplete_messages"
    elif is_message and any(
        len({(row.get("message_index"), row.get("part_index")) for row in rows}) != len(rows) for rows in (a, b)
    ):
        reason = "repeated_messages_not_aligned"
    elif not is_message and (len(a) != 1 or len(b) != 1):
        reason = "repeated_event_not_aligned"
    elif stage.startswith(("server.", "strategy.", "frontend.")) and (
        len(left["routes"]) != 1 or len(right["routes"]) != 1
    ):
        reason = "engine_attempts_not_unambiguously_aligned"
    result = {}
    for component in components:
        if reason:
            result[component] = {"status": "unknown", "reason": reason}
        elif component in ("prompt", "response"):
            result[component] = _tokens(a[0], b[0], component + "_token_snapshot", tokenizer)
        elif component == "text":

            def messages(rows):
                return [
                    {key: row.get(key) for key in ("role", "message_index", "part_index", "text")}
                    for row in sorted(rows, key=lambda r: (r.get("message_index", -1), r.get("part_index", -1)))
                ]

            result[component] = _trees(messages(a), messages(b))
        else:
            x, xe = _payload(a[0], component)
            y, ye = _payload(b[0], component)
            result[component] = (
                {"status": "unknown", "single": xe, "multi": ye}
                if xe or ye
                else _trees(
                    x,
                    y,
                    require_arrays=component in ("video", "features", "multimodal")
                    or (component == "audio" and x is not None and y is not None),
                )
            )
    return {"stage": stage, "components": result, "sources": references}


def _worker_stages(left, right, tokenizer):
    specs = (
        ("worker.receive", ("prompt", "multimodal_features", "sampling_params")),
        ("worker.encoder.write", ("embedding",)),
        ("worker.encoder.read", ("embedding",)),
        ("worker.gather", ("mask",)),
        ("worker.model_input", ("embedding", "positions")),
    )
    result = []
    for event, fields in specs:
        is_cache = event in {"worker.encoder.write", "worker.encoder.read"}
        groups = []
        for side in (left, right):
            grouped = defaultdict(list)
            for row in side["records"]:
                if row.get("event") == event:
                    key = (
                        row.get("worker_rank"),
                        None if is_cache else row.get("chunk_start"),
                        None if is_cache else row.get("chunk_tokens"),
                        row.get("feature_index"),
                    )
                    grouped[key].append(row)
            groups.append(grouped)
        keys = groups[0].keys() | groups[1].keys()
        for key in sorted(keys or {(None, None, None, None)}, key=repr):
            a, b = (group.get(key, []) for group in groups)
            sources = {"single": [r["_source"] for r in a], "multi": [r["_source"] for r in b]}
            # Cache methods return the whole feature, independently of prefill
            # chunk boundaries. Collapse repeated reads only when all recorded
            # content fingerprints agree; never choose a changing cache value.
            if is_cache:

                def collapse(rows):
                    if (
                        rows
                        and rows[0].get("embedding") is not None
                        and all(
                            _trees(rows[0]["embedding"], row.get("embedding"), require_arrays=True)["status"]
                            in ("equal", "sample_equal")
                            for row in rows[1:]
                        )
                    ):
                        return rows[:1]
                    return rows

                a, b = collapse(a), collapse(b)
            comparisons = {}
            reason = None
            if not a or not b:
                missing = "both" if not a and not b else "single" if not a else "multi"
                if not any(groups):
                    reason = "worker_event_not_recorded_in_either_run"
                elif any(group and key not in group for group in groups):
                    reason = "worker_rank_or_token_interval_not_aligned"
                else:
                    reason = "worker_event_missing_" + missing
            elif key[0] is None:
                reason = "worker_rank_not_recorded"
            elif len(a) != 1 or len(b) != 1:
                reason = "repeated_worker_event_not_aligned"
            elif any(len(side["routes"]) != 1 for side in (left, right)):
                reason = "engine_attempts_not_unambiguously_aligned"
            for field in fields:
                if reason:
                    comparison = {
                        "status": "unknown",
                        "reason": reason,
                        "record_counts": {"single": len(a), "multi": len(b)},
                    }
                elif field == "prompt":
                    comparison = _tokens(a[0], b[0], "prompt_token_snapshot", tokenizer)
                elif a[0].get(field) is None or b[0].get(field) is None:
                    comparison = {"status": "unknown", "reason": "missing_worker_payload"}
                else:
                    comparison = _trees(a[0][field], b[0][field], require_arrays=field != "sampling_params")
                comparisons[field] = comparison
            result.append(
                {
                    "stage": f"{event} [rank={key[0]}, start={key[1]}, tokens={key[2]}, feature={key[3]}]",
                    "components": comparisons,
                    "sources": sources,
                }
            )
    return result


def _worker_coverage(side):
    counts = defaultdict(int)
    for row in side["records"]:
        event = row.get("event", "")
        if event in {
            "worker.receive",
            "worker.encoder.write",
            "worker.encoder.read",
            "worker.gather",
            "worker.model_input",
        }:
            counts[event] += 1
    installations = [
        row for row in side.get("frontend_installation", []) if row.get("boundary", "").startswith("worker")
    ]
    registrations = [row for row in side["records"] if row.get("event") == "trace.worker_registration"]

    def acknowledgements(value):
        if isinstance(value, dict):
            if value.get("video_trace_ack") is True:
                yield value
            else:
                for item in value.values():
                    yield from acknowledgements(item)
        elif isinstance(value, list):
            for item in value:
                yield from acknowledgements(item)

    replies = [ack for row in registrations for ack in acknowledgements(row.get("worker_replies"))]
    supplied_names = {Path(path).name for path in side.get("input_files", [])}
    missing_files = [
        reply["trace_file"]
        for reply in replies
        if reply.get("trace_file") and Path(reply["trace_file"]).name not in supplied_names
    ]
    core_ids = {
        row["core_request_id"]
        for row in side["records"]
        if row.get("event", "").startswith("frontend.") and row.get("core_request_id")
    }
    unmatched = [
        row
        for row in side.get("worker_diagnostics", [])
        if row.get("event") == "trace.worker_unmatched" and row.get("core_request_id") in core_ids
    ]
    return {
        "event_counts": dict(counts),
        "registration": registrations,
        "worker_acknowledgements": replies,
        "expected_worker_files_not_supplied": missing_files,
        "unmatched_target_requests": unmatched,
        "worker_file_inventory": side.get("worker_file_inventory", {}),
        "installation": installations,
        "diagnostics_from_input_files": side.get("worker_diagnostics", []),
        "status": "worker_records_present" if counts else "no_worker_records_for_selected_trace",
        "note": "Installations and unscoped diagnostics describe input files, not necessarily this request's workers. "
        "Registration status=sent only confirms the RPC returned, "
        "not that observation was enabled or the request matched.",
    }


def _unknown_reason(value):
    if value.get("reason"):
        return value["reason"]
    gaps = value.get("gaps", {})
    reasons = sorted({str(reason) for side in gaps.values() for reason in side.values()})
    if reasons:
        return ", ".join(reasons)
    return (
        "; ".join(f"{side}={value[side]}" for side in ("single", "multi") if side in value) or "insufficient_evidence"
    )


def compare(left, right, tokenizer=None):
    if left["identity"] != right["identity"]:
        raise ValueError(f"Sample identity differs: {left['identity']} vs {right['identity']}")
    stages = []
    for stage, fields in _STAGES:
        if "video" in fields and any(
            row.get("event") == stage and "audio" in row for side in (left, right) for row in side["records"]
        ):
            fields = (*fields, "audio")
        if (
            stage.startswith("frontend.")
            and stage.endswith(".features")
            and any(
                row.get("event") == stage and "multimodal_features" in row
                for side in (left, right)
                for row in side["records"]
            )
        ):
            fields = (*fields, "multimodal")
        stages.append(_compare_stage(left, right, stage, fields, tokenizer))
    deep = any(row.get("event", "").startswith("worker.") for side in (left, right) for row in side["records"])
    # Registration without worker records must still expose missing coverage.
    deep |= any(row.get("event") == "trace.worker_registration" for side in (left, right) for row in side["records"])
    if deep:
        index = next(i for i, row in enumerate(stages) if row["stage"] == "strategy.result")
        stages[index:index] = _worker_stages(left, right, tokenizer)
    first, previous, gaps = {}, {}, defaultdict(list)
    for row in stages:
        for component, result in row["components"].items():
            status = result["status"]
            if status == "different" and component not in first:
                first[component] = {
                    "stage": row["stage"],
                    "previous_observed_match": previous.get(component),
                    "unresolved_stages_before": list(gaps[component]),
                    "comparison": result,
                }
            if status in ("equal", "sample_equal"):
                previous[component] = {"stage": row["stage"], "status": status}
            elif status == "unknown":
                gaps[component].append(row["stage"])
    return {
        "sample": left["identity"],
        "single": {k: v for k, v in left.items() if k != "records"},
        "multi": {k: v for k, v in right.items() if k != "records"},
        "first_observed_difference": next(
            (
                {"stage": row["stage"], "component": key}
                for row in stages
                for key, value in row["components"].items()
                if value["status"] == "different"
            ),
            None,
        ),
        "first_difference_by_component": first,
        "stages": stages,
        "worker_coverage": {label: _worker_coverage(side) for label, side in (("single", left), ("multi", right))}
        if deep
        else {},
        "worker_runtime": {
            label: [
                {key: row.get(key) for key in ("host", "pid", "worker_rank", "runtime", "_source")}
                for row in side["records"]
                if row.get("event") == "worker.receive"
            ]
            for label, side in (("single", left), ("multi", right))
        },
        "limitations": [
            "Stage order follows the logical request path, not cross-host wall clocks.",
            "A difference is an observation, not proof of corruption or its cause; sampling can change responses.",
            "sample_equal checks at most 256 elements per array; metadata-only/skipped content remains unknown.",
            "Missing/cache-omitted data and multiple engine attempts are not treated as equal or corrupt.",
            "Cache identifiers, addresses and strides are preserved in raw records, not compared as content.",
            "Worker mode observes recovered video/audio features, encoder cache reads/writes and model inputs; "
            "absent events remain unknown.",
            "Worker device contents require DEVICE_SAMPLE=1; this introduces device reads and can affect scheduling.",
            "Worker chunks are aligned by rank and exact token interval; different chunking is not a content mismatch.",
            "EngineCore internals, encoder internals, runtime weights and logits are not covered; "
            "model input observation does not prove attention used video.",
        ],
    }


def markdown(report):
    first = report["first_observed_difference"]
    unknown = sum(result["status"] == "unknown" for row in report["stages"] for result in row["components"].values())
    lines = [
        f"Sample: `{report['sample']['sample_key']}`",
        "",
        "First observed difference: " + (f"`{first['stage']}` / `{first['component']}`" if first else "none observed"),
        "",
        f"Unknown comparisons: {unknown}. A missing observation is not evidence of equality.",
        "",
        "| Stage | Comparison |",
        "| --- | --- |",
    ]
    for row in report["stages"]:
        cells = (
            "; ".join(
                f"{key}: **{value['status']}**"
                + (f" ({_unknown_reason(value)})" if value["status"] == "unknown" else "")
                for key, value in row["components"].items()
            )
            .replace("|", "\\|")
            .replace("\n", " ")
        )
        lines.append(f"| `{row['stage']}` | {cells} |")
    for side in ("single", "multi"):
        data = report[side]
        lines.extend(["", f"{side}: trace `{data['trace_id']}`, dropped events: {data['dropped_events']}"])
        lines.append("")
        for route in data["routes"]:
            lines.append(
                f"- host `{route['host']}`, pid `{route['pid']}`, replica `{route['replica_rank']}`, "
                f"request `{route['request_id']}`"
            )
        lines.extend(f"- Warning: {item}" for item in data["warnings"])
        coverage = report.get("worker_coverage", {}).get(side)
        if coverage:
            lines.append(f"- Worker coverage: `{coverage['status']}`; events: `{coverage['event_counts']}`")
            if not coverage["event_counts"]:
                lines.append(
                    "- No worker payloads are available for this trace. "
                    "This is a collection/correlation gap, not an embedding comparison result."
                )
            for path in coverage["expected_worker_files_not_supplied"]:
                lines.append(
                    f"- Expected worker file is absent from analysis inputs: `{path}`. "
                    "Check file collection and writer errors."
                )
            if coverage["unmatched_target_requests"]:
                lines.append(
                    "- The exact frontend core request ID appears in worker_unmatched: "
                    "the request reached a worker but trace context was not associated."
                )
            lines.append(
                f"- Files containing any worker events/installations: `{len(coverage['worker_file_inventory'])}`"
            )
            lines.append(
                f"- Worker registration statuses: `{[r.get('status') for r in coverage['registration']]}`; "
                f"worker hook installation records in input files: `{len(coverage['installation'])}`"
            )
            if coverage["registration"] and not coverage["worker_acknowledgements"]:
                lines.append(
                    "- No worker acknowledgement was recorded. Legacy status=sent does not prove tracing was enabled."
                )
            for reply in coverage["worker_acknowledgements"]:
                lines.append(
                    f"- Worker acknowledgement: host `{reply.get('host')}`, rank `{reply.get('worker_rank')}`, "
                    f"status `{reply.get('status')}`, stage `{reply.get('trace_stage')}`, "
                    f"device_sample `{reply.get('device_sample')}`, directory `{reply.get('trace_dir')}`"
                )
                for name, hook in reply.get("hooks", {}).items():
                    if hook.get("status") != "installed":
                        lines.append(f"- Worker hook `{name}`: {hook}")
            for row in coverage["installation"]:
                if row.get("status") != "installed":
                    lines.append(
                        f"- Worker hook unavailable: `{row.get('boundary')}`: {row.get('error')} ({row.get('_source')})"
                    )
        for worker in report.get("worker_runtime", {}).get(side, []):
            seed = (worker.get("runtime") or {}).get("model_config", {}).get("seed")
            lines.append(
                f"- Worker rank `{worker['worker_rank']}`, host `{worker['host']}`, effective model seed `{seed}`"
            )
    lines.extend(
        ["", "equal = recorded values match; sample_equal = sampled values match; unknown = insufficient evidence."]
    )
    lines.extend(["", "Details and source file/line references are in comparison.json.", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", nargs="+", required=True, type=Path, help="Single-node raw trace files/directories")
    parser.add_argument(
        "--multi", nargs="+", required=True, type=Path, help="Raw trace files/directories from both hosts"
    )
    parser.add_argument("--sample-key", required=True, help="extra_info.problem_id; not a random request ID")
    parser.add_argument("--single-trace-id", help="Select one invocation if the sample occurs more than once")
    parser.add_argument("--multi-trace-id", help="Select one invocation if the sample occurs more than once")
    parser.add_argument("--tokenizer", help="Optional local tokenizer directory for token difference excerpts")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    try:
        single = load_request(args.single, args.sample_key, args.single_trace_id)
        multi = load_request(args.multi, args.sample_key, args.multi_trace_id)
        report = compare(single, multi, tokenizer)
    except ValueError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "comparison.md").write_text(markdown(report), encoding="utf-8")
    for label, data in (("single", single), ("multi", multi)):
        (args.output_dir / f"{label}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data["records"]), encoding="utf-8"
        )
    print(markdown(report))


if __name__ == "__main__":
    main()
