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
"""Locate format changes in video traces; never recompute or relax reward scores."""

import argparse
import hashlib
import json
import os
import runpy
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path

_DESCRIBE = runpy.run_path(Path(__file__).resolve().parents[2] / "verl_omni/utils/video_trace.py")["describe_text"]
_REQUIRED = (
    "agent.prompt",
    "agent.dispatch",
    "server.receive",
    "strategy.submit",
    "strategy.result",
    "server.result",
    "agent.result",
    "agent.merge.before",
    "agent.merge.after",
    "agent.output",
)
_PROMPTS = (
    "agent.prompt",
    "agent.dispatch",
    "server.receive",
    "strategy.submit",
    "frontend.build.before",
    "frontend.input.before",
    "frontend.tokens.before",
    "frontend.tokens.result",
    "frontend.input.result",
    "frontend.build.result",
)
_RESPONSES = (
    "strategy.result",
    "server.result",
    "agent.result",
    "agent.merge.before",
    "agent.merge.after",
    "agent.output",
)


def _text(value):
    return {**_DESCRIBE(value), "complete": True, "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()}


def _snapshot(snapshot, tokenizer, cache):
    if not isinstance(snapshot, dict) or snapshot.get("status") != "complete":
        return snapshot or {"status": "unavailable"}
    result = {key: value for key, value in snapshot.items() if key != "ids"}
    ids = snapshot.get("ids")
    if not isinstance(ids, list) or len(ids) != snapshot.get("count") or any(type(item) is not int for item in ids):
        return {**result, "status": "invalid_snapshot"}
    packed = array("q", ids)
    if sys.byteorder != "little":
        packed.byteswap()
    digest = hashlib.sha256(packed.tobytes()).hexdigest()
    if snapshot.get("encoding") != "int64-le" or digest != snapshot.get("sha256"):
        return {**result, "status": "invalid_digest"}
    result.update(head_ids=ids[:8], tail_ids=ids[-8:])
    if tokenizer is None:
        return {**result, "decode_status": "tokenizer_not_supplied"}
    if digest not in cache:
        try:
            cache[digest] = {
                "decode_status": "decoded",
                "raw": _text(tokenizer.decode(ids, skip_special_tokens=False)),
                "visible": _text(tokenizer.decode(ids, skip_special_tokens=True)),
            }
        except Exception as exc:
            cache[digest] = {"decode_status": "failed", "decode_error": f"{type(exc).__name__}: {exc}"}
    return {**result, **cache[digest]}


def _view(record, tokenizer, cache):
    view = {
        key: record[key]
        for key in (
            "event",
            "host",
            "pid",
            "seq",
            "request_id",
            "call_id",
            "replica_rank",
            "node_rank",
            "role",
            "message_index",
            "part_index",
            "text",
            "engine_text",
            "finish_reason",
            "stop_reason",
            "sampling_params",
            "max_tokens",
            "response_mask_count",
            "video",
            "mm_uuids",
            "mm_processor_kwargs",
            "roles",
            "message_count",
            "omitted_messages",
            "status",
            "tokenizer_name",
            "tokenizer_class",
            "processor_class",
            "builder_class",
            "eos_token_id",
            "pad_token_id",
            "prompt_length",
            "response_length",
            "tokenizer_template",
            "processor_template",
        )
        if key in record
    }
    for field in ("prompt_token_snapshot", "response_token_snapshot"):
        if field in record:
            view[field] = _snapshot(record[field], tokenizer, cache)
    return view


def _decoded(view, field, kind="visible"):
    value = view.get(field, {}).get(kind)
    return value if value and value.get("complete") else None


def _compare_chain(views, names, field, *, prompt=False):
    """Compare unique observations only; report an interval when stages are absent."""
    findings, transitions = [], []
    previous = None
    for name in names:
        candidates = [view for view in views if view["event"] == name]
        if len(candidates) != 1:
            continue
        current = candidates[0]
        snapshot = current.get(field, {})
        text = _decoded(current, field, "raw" if prompt else "visible")
        if previous and snapshot.get("status") == previous.get(field, {}).get("status") == "complete":
            transitions.append(
                {
                    "from": previous["event"],
                    "to": name,
                    "tokens_equal": snapshot["sha256"] == previous[field]["sha256"],
                }
            )
        if text:
            before = _decoded(previous, field, "raw" if prompt else "visible") if previous else None
            has_format = lambda item: (
                item["answer_open_count"] > 0 and item["answer_close_count"] > 0 if prompt else item["has_answer_pair"]
            )
            if before and has_format(before) and not has_format(text):
                findings.append(
                    {
                        "code": "prompt_answer_markers_lost" if prompt else "answer_tags_lost",
                        "from": previous["event"],
                        "to": name,
                    }
                )
        # Undecoded stages break the format comparison, rather than implying tag loss.
        previous = current
    return findings, transitions


def _validation_outputs(path):
    outputs = {}
    if path is None:
        return outputs
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            parts = str(row.get("uid", "")).rsplit("_", 2)
            if len(parts) != 3 or not parts[2].isdigit() or not isinstance(row.get("output"), str):
                continue
            key, index = (parts[0], parts[1]), int(parts[2])
            if key not in outputs or outputs[key][0] < index:
                outputs[key] = (index, _text(row["output"]))
    return outputs


def analyze(paths, tokenizer=None, validation=None):
    grouped, cache = defaultdict(list), {}
    writers, warnings, installs = {}, [], []
    files = set()
    for path in paths:
        path = Path(path)
        files.update(path.glob("video-v2-*.jsonl") if path.is_dir() else [path])
    if not files:
        raise ValueError("No video-v2 trace files found")
    for path in sorted({path.resolve() for path in files}):
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("expected a JSON object")
                except ValueError as exc:
                    warnings.append(f"{path}:{line_number}: {exc}")
                    continue
                writer = (record.get("host"), record.get("pid"), record.get("writer_id"))
                writers[writer] = max(writers.get(writer, 0), record.get("dropped_events", 0))
                if record.get("event") == "trace.install":
                    installs.append({key: record.get(key) for key in ("host", "pid", "boundary", "status", "error")})
                if record.get("trace_id"):
                    grouped[record["trace_id"]].append(record)
    validation_outputs = _validation_outputs(validation)
    reports, replicas, totals = [], {}, Counter()
    for trace_id, records in grouped.items():
        first = records[0]
        views = [_view(record, tokenizer, cache) for record in records]
        report = {
            "trace_id": trace_id,
            **{key: first.get(key) for key in ("uid", "session_id", "sample_key", "sample_index", "video_id")},
            "missing_events": [name for name in _REQUIRED if not any(view["event"] == name for view in views)],
            "findings": [],
            "transitions": [],
            "observations": views,
        }
        attempts = defaultdict(list)
        for view in views:
            if view.get("request_id"):
                attempts[(view.get("host"), view.get("pid"), view["request_id"])].append(view)
        report["engine_attempts"] = len(attempts)
        for (host, pid, request_id), attempt in attempts.items():
            received = next((view for view in attempt if view["event"] == "server.receive"), {})
            rank = received.get("replica_rank")
            route = {"host": host, "pid": pid, "replica_rank": rank, "request_id": request_id}
            replica = replicas.setdefault(
                (host, pid, rank),
                {"host": host, "pid": pid, "replica_rank": rank, "requests": 0, "findings": Counter()},
            )
            replica["requests"] += 1
            findings, transitions = _compare_chain(attempt, _PROMPTS, "prompt_token_snapshot", prompt=True)
            report["transitions"].extend({**item, **route, "kind": "prompt"} for item in transitions)
            submits = [view for view in attempt if view["event"] == "strategy.submit"]
            maximum = submits[0].get("max_tokens") if len(submits) == 1 else None
            for view in attempt:
                if view["event"] != "strategy.result":
                    continue
                snapshot = view.get("response_token_snapshot", {})
                visible = _decoded(view, "response_token_snapshot")
                details = {
                    "stage": view["event"],
                    "finish_reason": view.get("finish_reason"),
                    "tokens": snapshot.get("count"),
                    "max_tokens": maximum,
                }
                if snapshot.get("count") == 0 and snapshot.get("status") == "complete":
                    findings.append({"code": "empty_engine_tokens", **details})
                if visible and visible["empty_visible_text"]:
                    findings.append({"code": "engine_visible_empty", **details})
                if visible and not visible["has_answer_pair"]:
                    findings.append({"code": "engine_missing_answer_tags", **details})
                if view.get("finish_reason") == "length":
                    findings.append({"code": "engine_length_stop", **details})
            for finding in findings:
                replica["findings"][finding["code"]] += 1
                report["findings"].append({**finding, **route})

        # Multiple resume requests have different token budgets/prompts by design.
        # Do not compare a single partial engine result with the combined response.
        output_stages = _RESPONSES if len(attempts) == 1 else _RESPONSES[2:]
        findings, transitions = _compare_chain(views, output_stages, "response_token_snapshot")
        report["findings"].extend(findings)
        report["transitions"].extend({**item, "kind": "response"} for item in transitions)
        if len(attempts) == 1:
            findings, transitions = _compare_chain(views, _PROMPTS[:3], "prompt_token_snapshot", prompt=True)
            report["findings"].extend(findings)
            report["transitions"].extend({**item, "kind": "prompt_boundary"} for item in transitions)
        systems = [
            view.get("text", {})
            for view in views
            if view["event"] == "agent.source.message" and view.get("role") == "system"
        ]
        prompts = [view for view in views if view["event"] == "agent.prompt"]
        if systems and all(item.get("complete") for item in systems) and len(prompts) == 1:
            initial = _decoded(prompts[0], "prompt_token_snapshot", "raw")
            expected = any(item.get("answer_open_count") for item in systems) and any(
                item.get("answer_close_count") for item in systems
            )
            if not expected:
                report["findings"].append(
                    {"code": "source_system_has_no_answer_markers", "stage": "agent.source.message"}
                )
            if expected and initial and not (initial["answer_open_count"] and initial["answer_close_count"]):
                report["findings"].append(
                    {"code": "prompt_answer_markers_lost", "from": "agent.source.message", "to": "agent.prompt"}
                )
        finals = [view for view in views if view["event"] == "agent.output"]
        for final in finals:
            raw = _decoded(final, "response_token_snapshot", "raw")
            visible = _decoded(final, "response_token_snapshot")
            if raw and visible and raw["has_answer_pair"] and not visible["has_answer_pair"]:
                report["findings"].append({"code": "special_token_filter_removed_answer_tags", "stage": "agent.output"})
        key = (str(first.get("uid")), str(first.get("session_id")))
        if key in validation_outputs:
            saved = validation_outputs[key][1]
            report["validation_output"] = saved
            visible = _decoded(finals[0], "response_token_snapshot") if len(finals) == 1 else None
            if visible and visible["sha256"] != saved["sha256"]:
                report["findings"].append(
                    {"code": "validation_text_changed", "from": "agent.output", "to": "validation.output"}
                )
        elif validation is not None:
            report["validation_output"] = {"status": "uid_session_not_found"}
        report["undecoded_snapshots"] = sum(
            view.get(field, {}).get("decode_status") != "decoded"
            for view in views
            for field in (
                ["prompt_token_snapshot"]
                if view["event"] in _PROMPTS
                else ["response_token_snapshot"]
                if view["event"] in _RESPONSES
                else []
            )
        )
        if len(attempts) == 1:
            (host, pid, request_id), attempt = next(iter(attempts.items()))
            received = next((view for view in attempt if view["event"] == "server.receive"), {})
            rank = received.get("replica_rank")
            for finding in report["findings"]:
                if "request_id" not in finding:
                    finding.update(host=host, pid=pid, replica_rank=rank, request_id=request_id)
                    replicas[(host, pid, rank)]["findings"][finding["code"]] += 1
        totals.update(finding["code"] for finding in report["findings"])
        reports.append(report)
    incomplete = sum(bool(item["missing_events"] or item["undecoded_snapshots"]) for item in reports)
    return {
        "files": len(files),
        "traces": len(reports),
        "findings": dict(totals),
        "warnings": warnings,
        "dropped_events": sum(writers.values()),
        "incomplete_traces": incomplete,
        "coverage_complete": bool(reports) and not (incomplete or warnings or any(writers.values())),
        "replicas": [{**item, "findings": dict(item["findings"])} for item in replicas.values()],
        "frontend_installation": installs,
        "requests": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Trace files or directories from both hosts")
    parser.add_argument("--tokenizer", default=os.environ.get("MODEL_PATH"), help="Local model/tokenizer directory")
    parser.add_argument("--validation", type=Path, help="Optional validation JSONL, joined by uid and session")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = analyze(args.paths, tokenizer=tokenizer, validation=args.validation)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
