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

"""Read small per-update records; export a bounded, path-free AB summary."""

import argparse
import json
import math
import statistics
from pathlib import Path


def load_records(directory, ranks, updates):
    records = {}
    for path in directory.glob("rank*_update*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = record["rank"], record["update"]
        if key in records or record["world_size"] != ranks:
            raise ValueError("Duplicate rank/update or world-size mismatch")
        records[key] = record
    if set(records) != {(r, u) for r in range(ranks) for u in range(updates)}:
        raise ValueError("Missing/extra rank/update records; do not compare partial runs")
    return records


def numeric_close(a, b, rtol, atol):
    if isinstance(a, list) and isinstance(b, list):
        return bool(a) and len(a) == len(b) and all(numeric_close(x, y, rtol, atol) for x, y in zip(a, b, strict=True))
    return (
        isinstance(a, int | float)
        and isinstance(b, int | float)
        and math.isfinite(a)
        and math.isfinite(b)
        and math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    )


def aggregate(records, warmup):
    selected = [r for (_, u), r in sorted(records.items()) if u >= warmup]
    backwards = [b for r in selected for b in r["backwards"]]
    if not backwards:
        raise ValueError("No measured backwards")
    critical_updates = [
        max(r["update_actor_drained_ms"] for r in selected if r["update"] == u)
        for u in sorted({r["update"] for r in selected})
    ]
    return {
        "rank_backwards": len(backwards),
        "backward_retries_sum": sum(b["alloc_retries"] for b in backwards),
        "backward_retries_max": max(b["alloc_retries"] for b in backwards),
        "update_retries_sum": sum(r["update_alloc_retries"] for r in selected),
        "active_peak_max_MiB": round(max(b["active_peak_bytes"] for b in backwards) / 2**20, 2),
        "allocated_peak_max_MiB": round(max(b["allocated_peak_bytes"] for b in backwards) / 2**20, 2),
        "gap_sampled_max_MiB": round(max(b["gap_sampled_max_bytes"] for b in backwards) / 2**20, 2),
        **{
            key + "_max": round(max(b[key] for b in backwards), 3)
            for key in ("backward_host_ms", "backward_device_ms", "limiter_wait_ms")
        },
        "limiter_wait_ms_sum": round(sum(b["limiter_wait_ms"] for b in backwards), 3),
        "limiter_wait_calls_sum": sum(b["limiter_wait_calls"] for b in backwards),
        "update_actor_rankmax_ms": [round(v, 3) for v in critical_updates],
        "update_actor_rankmax_median_ms": round(statistics.median(critical_updates), 3),
    }


def compare(a, b, warmup, rtol=1e-5, atol=1e-6):
    if not a or set(a) != set(b):
        raise ValueError("AB keys differ or no records")
    mismatches = []
    for key in sorted(a):
        left, right = a[key], b[key]
        if left["limit_rs"] or not right["limit_rs"]:
            raise ValueError("Expected A=off and B=on on every rank/update")
        for field in ("fixture_sha256", "config_sha256", "initial_shard_sha256", "native_offload"):
            if left[field] != right[field]:
                raise ValueError(f"Fixed-input comparison mismatch: {field}, rank/update={key}")
        if not left["backwards"] or len(left["backwards"]) != len(right["backwards"]):
            raise ValueError("Missing/different backward count")
        for field in ("loss", "grad_norm"):
            if not numeric_close(left[field], right[field], rtol, atol):
                mismatches.append([*key, field])
    return {
        "fixed_inputs_match": True,
        "numerical_metrics_close": not mismatches,
        "numerical_mismatch_count": len(mismatches),
        "numerical_mismatch_examples": mismatches[:8],
        "rtol": rtol,
        "atol": atol,
        "warmup_updates_excluded": warmup,
        "off": aggregate(a, warmup),
        "on": aggregate(b, warmup),
        "scope": "gap is sampled; peaks are independent; timings include common AB instrumentation",
        "criterion": "less pending reuse/retries/stalls; Reserved/HBM need not decrease; no automatic speedup verdict",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=16)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.warmup < args.updates:
        parser.error("Require 0 <= warmup < updates")
    summary = compare(
        load_records(args.off, args.ranks, args.updates),
        load_records(args.on, args.ranks, args.updates),
        args.warmup,
        args.rtol,
        args.atol,
    )
    content = json.dumps(summary, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    if len(content.encode()) > 8192:
        raise ValueError("Summary exceeds 8 KiB; keep the full records on the server")
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        target.write(content)
    print(content, end="")
    return 0 if summary["numerical_metrics_close"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
