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
"""Summarize existing compare outputs without loading models or changing traces."""

import argparse
import json
import runpy
from collections import Counter, defaultdict
from pathlib import Path

MAX_OUTPUT_BYTES = 64 * 1024
DIAGNOSTICS = runpy.run_path(Path(__file__).with_name("video_trace_diagnostics.py"))
WORKER_DATA_EVENTS = {
    "worker.receive",
    "worker.encoder.write",
    "worker.encoder.read",
    "worker.gather",
    "worker.model_input",
}


def compact(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def signature(value):
    if not isinstance(value, dict):
        return None
    mode = value.get("digest_status")
    digest = value.get("sha256" if mode == "full" else "sample_sha256")
    if mode not in ("full", "sample") or not digest or value.get("shape") is None or not value.get("dtype"):
        return None
    if mode == "sample" and (not value.get("sample_scheme") or value.get("sample_count") is None):
        return None
    return compact(
        {key: value.get(key) for key in ("shape", "dtype", "digest_status")}
        | {"digest": digest}
        | ({key: value[key] for key in ("sample_scheme", "sample_count")} if mode == "sample" else {})
    )


def read_workers(path):
    rows, traces, samples = [], set(), set()
    attempts = {}
    counts = Counter()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path.name}:{number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{number}: expected an object")
            traces.add(row.get("trace_id"))
            samples.add(row.get("sample_key"))
            event = row.get("event", "")
            if not event.startswith("worker."):
                continue
            counts[event] += 1
            # Scope begin/end/error records have no worker_rank. Keep their
            # counts, but validate identities only for supported data events.
            if event not in WORKER_DATA_EVENTS:
                continue
            missing = [key for key in ("worker_rank", "host", "pid", "core_request_id") if row.get(key) is None]
            if missing:
                raise ValueError(f"{path.name}:{number}: {event}: missing worker identity fields: {', '.join(missing)}")
            rank = row.get("worker_rank")
            identity = {key: row[key] for key in ("host", "pid", "core_request_id")}
            if rank in attempts and attempts[rank][0] != identity:
                previous, previous_line = attempts[rank]
                raise ValueError(
                    f"{path.name}:{number}: {event}: conflicting worker identity for rank={rank}; "
                    f"line {previous_line}={compact(previous)}, line {number}={compact(identity)}"
                )
            attempts.setdefault(rank, (identity, number))
            if event == "worker.receive":
                rows.append(
                    {key: row.get(key) for key in ("event", "worker_rank", "runtime")}
                    | {
                        "features": [
                            {key: item.get(key) for key in ("feature_index", "modality", "position")}
                            for item in row.get("multimodal_features", [])
                        ]
                    }
                )
            elif event in ("worker.encoder.write", "worker.encoder.read"):
                rows.append(
                    {key: row.get(key) for key in ("event", "worker_rank", "feature_index", "modality", "identifier")}
                    | {"signature": signature(row.get("embedding"))}
                )
    if len(traces) != 1 or None in traces or len(samples) != 1 or None in samples:
        raise ValueError(f"{path.name}: expected one selected trace and sample; use compare's single.jsonl/multi.jsonl")
    return rows, dict(counts), next(iter(traces)), next(iter(samples))


def cache_status(writes, reads):
    if not writes or not reads or any(row["signature"] is None or row["identifier"] is None for row in writes + reads):
        return "unknown: missing event, digest or cache identifier"
    if len({row["identifier"] for row in writes + reads}) != 1:
        return "unknown: multiple cache identifiers"
    ws, rs = ({row["signature"] for row in rows} for rows in (writes, reads))
    if len(ws) != 1 or len(rs) != 1:
        return "changing_observations: repeated writes/reads disagree"
    a, b = (json.loads(next(iter(values))) for values in (ws, rs))
    if (a["shape"], a["dtype"]) != (b["shape"], b["dtype"]):
        return "different: shape/dtype"
    if any(a.get(key) != b.get(key) for key in ("digest_status", "sample_scheme", "sample_count")):
        return "unknown: incompatible digest schemes"
    if a["digest"] != b["digest"]:
        return "different: content digest"
    return "sample_equal" if a["digest_status"] == "sample" else "equal"


def summarize(directory):
    report = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    sides = {side: read_workers(directory / f"{side}.jsonl") for side in ("single", "multi")}
    if sides["single"][3] != sides["multi"][3] or report.get("sample", {}).get("sample_key") != sides["single"][3]:
        raise ValueError("Sample keys differ between comparison.json and selected traces")
    for side, (_, _, trace_id, _) in sides.items():
        if report.get(side, {}).get("trace_id") != trace_id:
            raise ValueError(f"{side}: comparison.json and selected trace IDs differ")
    lines = [
        "8039 worker summary v1",
        "sample=" + compact(report.get("sample")),
        "first_difference=" + compact(report.get("first_observed_difference")),
        "Digest equality is exact or sampled, never a floating-point tolerance check.",
        "Cross-rank groups describe recorded bytes; rank equality is not assumed by the model architecture.",
        "Hash-only observations cannot measure error size or prove the cause of accuracy loss.",
    ]
    lines.extend(DIAGNOSTICS["vision_summary"](report))
    aliases = {}

    def label(value):
        if value is None:
            return "unknown"
        if value not in aliases:
            aliases[value] = f"D{len(aliases) + 1}"
        return aliases[value]

    for side, (rows, counts, trace_id, _) in sides.items():
        lines.extend(["", f"[{side}] trace={trace_id}", "events=" + compact(counts)])
        groups = defaultdict(list)
        runtime_groups = defaultdict(list)
        for row in rows:
            rank = row["worker_rank"]
            if row["event"] == "worker.receive":
                lines.append(f"rank={rank} features=" + compact(row["features"]))
                runtime_groups[compact(row["runtime"])].append(rank)
            else:
                groups[(rank, row["feature_index"])].append(row)
        for runtime, ranks in runtime_groups.items():
            lines.append(f"runtime ranks={ranks}: {runtime}")
        rank_groups = defaultdict(lambda: defaultdict(set))
        for (rank, feature), events in sorted(groups.items(), key=lambda item: repr(item[0])):
            writes = [row for row in events if row["event"] == "worker.encoder.write"]
            reads = [row for row in events if row["event"] == "worker.encoder.read"]
            modalities = sorted({str(row["modality"]) for row in events})
            parts = []
            for name, entries in (("write", writes), ("read", reads)):
                labels = sorted({label(row["signature"]) for row in entries})
                parts.append(f"{name}={','.join(labels) or 'missing'}(n={len(entries)})")
                for entry in entries:
                    rank_groups[(feature, entry["modality"], name)][label(entry["signature"])].add(rank)
            lines.append(
                f"rank={rank} feature={feature} modality={modalities} "
                + " ".join(parts)
                + " write_vs_read="
                + cache_status(writes, reads)
            )
        for (feature, modality, event), values in sorted(rank_groups.items(), key=lambda item: repr(item[0])):
            lines.append(
                f"rank_groups feature={feature} modality={modality} {event}: "
                + compact({key: sorted(ranks) for key, ranks in values.items()})
            )
    lines.extend(["", "[digest legend: shared across both runs]"])
    lines.extend(f"{name}={value}" for value, name in aliases.items())
    lines.extend(["", "[cross-run worker differences and unknowns]"])
    for stage in report.get("stages", []):
        if not stage["stage"].startswith("worker."):
            continue
        for component, result in stage["components"].items():
            if result["status"] in ("equal", "sample_equal"):
                continue
            differences = result.get("differences", [])
            lines.append(
                f"{stage['stage']} {component}: "
                + compact(
                    {key: result[key] for key in ("status", "reason", "gaps") if key in result}
                    | {"differences": differences[:4], "omitted_differences": max(0, len(differences) - 4)}
                )
                + " "
                + DIAGNOSTICS["short_metrics"](result)
            )
    output = "\n".join(lines) + "\n"
    if len(output.encode("utf-8")) > MAX_OUTPUT_BYTES:
        marker = "\nTRUNCATED: summary exceeded 64 KiB; omitted records are not evidence of equality.\n"
        output = output.encode("utf-8")[: MAX_OUTPUT_BYTES - len(marker.encode())].decode("utf-8", errors="ignore")
        output = output.rsplit("\n", 1)[0] + marker
    return output


def live_health(paths, sample_key):
    groups = defaultdict(list)
    files = set()
    invalid = 0
    for path in paths:
        files.update(path.rglob("video-v2-*.jsonl") if path.is_dir() else [path])
    for path in sorted(files):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    invalid += 1  # A running writer may not have finished its last line.
                    continue
                if not isinstance(row, dict) or str(row.get("sample_key")) != sample_key:
                    continue
                if not row.get("event", "").startswith("worker.vision."):
                    continue
                key = (row.get("trace_id"), row.get("host"), row.get("pid"), row.get("worker_rank"))
                kept = {
                    name: row[name]
                    for name in ("event", "worker_rank", "checkpoint", "hooks", "config", "reason")
                    if name in row
                }
                if row["event"] == "worker.vision.weights":
                    kept["values"] = dict.fromkeys(row.get("values", {}))
                groups[key].append(kept)
    lines = [f"Vision capture health: sample={sample_key}, files={len(files)}, invalid_or_partial_lines={invalid}"]
    if not groups:
        lines.append("NO_VISION_EVENTS: target not reached, vision not enabled, worker hooks missing, or files absent.")
    for (trace_id, host, pid, rank), rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        coverage = DIAGNOSTICS["vision_coverage"]({"records": rows})[str(rank)]
        failed = [key for key, value in coverage["hooks"].items() if value.get("status") != "installed"]
        lines.append(
            f"trace={trace_id} {host}:{pid} rank={rank} "
            f"observed={len(coverage['observed'])} weights={coverage['weights']}"
        )
        lines.append(f"  missing={coverage['missing_checkpoints']} unavailable={failed}")
        lines.extend(f"  gap {item['checkpoint']}: {item['reason']}" for item in coverage["gaps"][:4])
    lines.append(
        "During an active request, missing stages may still arrive. Coverage does not establish numeric equality."
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, nargs="?", help="Compare output directory containing comparison.json and both JSONLs"
    )
    parser.add_argument("--raw", type=Path, nargs="+", help="Inspect live raw trace directories/files without compare")
    parser.add_argument("--sample-key", help="Required with --raw")
    args = parser.parse_args()
    if args.raw:
        if args.directory or not args.sample_key:
            parser.error("--raw requires --sample-key and cannot be combined with a compare directory")
        try:
            print(live_health(args.raw, args.sample_key), end="")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        return
    if args.directory is None:
        parser.error("provide a compare directory or --raw with --sample-key")
    try:
        output = summarize(args.directory)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    destination = args.directory / "worker_summary.txt"
    destination.write_text(output, encoding="utf-8")
    print(output, end="")
    print(f"Saved {len(output.encode('utf-8'))} bytes to {destination}")


if __name__ == "__main__":
    main()
