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
from collections import Counter, defaultdict
from pathlib import Path

MAX_OUTPUT_BYTES = 64 * 1024


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
    attempts = defaultdict(set)
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
            rank = row.get("worker_rank")
            attempts[rank].add((row.get("host"), row.get("pid"), row.get("core_request_id")))
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
    if any(rank is None or len(ids) != 1 or any(None in item for item in ids) for rank, ids in attempts.items()):
        raise ValueError(f"{path.name}: worker rank/request identity is missing or ambiguous")
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
            )
    output = "\n".join(lines) + "\n"
    if len(output.encode("utf-8")) > MAX_OUTPUT_BYTES:
        marker = "\nTRUNCATED: summary exceeded 64 KiB; omitted records are not evidence of equality.\n"
        output = output.encode("utf-8")[: MAX_OUTPUT_BYTES - len(marker.encode())].decode("utf-8", errors="ignore")
        output = output.rsplit("\n", 1)[0] + marker
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Compare output directory containing comparison.json and both JSONLs"
    )
    args = parser.parse_args()
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
