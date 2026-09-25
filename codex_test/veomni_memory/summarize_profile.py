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

"""Read-only profile inventory and conservative, bounded summaries; no torch imports.

Unknown schemas are reported, never guessed into cross-rank evidence. See README
for the explicit schema-map contract and which first-round formats are supported.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import gzip
import heapq
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from decimal import Decimal
from pathlib import Path

MAX_OBJECT = 4 * 1024 * 1024
UNITS = {"ns": 1, "us": 1000, "µs": 1000, "ms": 1000000, "s": 1000000000}
FIELDS = (
    "name",
    "kind",
    "rank",
    "device",
    "pid",
    "stream",
    "step",
    "micro_batch",
    "phase",
    "start",
    "end",
    "duration",
    "self_time",
    "time_unit",
    "run",
    "clock",
    "clock_verified",
    "group",
    "members",
    "collective",
    "collective_seq",
    "sequence_verified",
    "bytes",
    "wait",
    "transfer",
    "wait_unit",
    "transfer_unit",
    "allocated",
    "active",
    "reserved",
    "memory_unit",
    "allocator",
    "address",
    "action",
    "correlation_id",
)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def ns(value, unit):
    if value is None or value == "":
        return None
    if unit not in UNITS:
        raise ValueError(f"unknown time unit {unit!r}")
    # Do not pass epoch timestamps through float. Reject already lossy large floats.
    if isinstance(value, float) and abs(value) * UNITS[unit] >= 2**53:
        raise ValueError("large floating timestamp may already have lost precision")
    v = Decimal(str(value)) * UNITS[unit]
    if not v.is_finite() or v != v.to_integral_value():
        raise ValueError("time is not an integral number of nanoseconds")
    n = int(v)
    if not -(2**63) < n < 2**63:
        raise ValueError("timestamp outside signed int64")
    return n


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def readonly(path):
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def open_text(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return path.open(encoding="utf-8-sig", newline="")


def trace_items(path):
    """Stream Chrome traceEvents or a root array, retaining at most one bounded event.

    Only the array prefix is scanned. B/E stacks and arbitrary nested communication
    JSON are intentionally not interpreted by this adapter.
    """
    decoder = json.JSONDecoder(parse_float=Decimal)
    with open_text(path) as src:
        buf = ""
        while True:
            chunk = src.read(65536)
            if not chunk:
                raise ValueError("no traceEvents/root array; needs a communication/schema adapter")
            buf += chunk
            if buf.lstrip().startswith("["):
                buf = buf[buf.index("[") + 1 :]
                break
            match = re.search(r'"traceEvents"\s*:\s*\[', buf)
            if match:
                buf = buf[match.end() :]
                break
            if len(buf) > MAX_OBJECT:
                raise ValueError("trace array prefix exceeds 4 MiB; use inventory first")
        while True:
            buf = buf.lstrip()
            if buf.startswith(","):
                buf = buf[1:].lstrip()
            if buf.startswith("]"):
                return
            try:
                item, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                chunk = src.read(65536)
                if not chunk:
                    raise ValueError("truncated trace array") from None
                buf += chunk
                if len(buf) > MAX_OBJECT:
                    raise ValueError("single event exceeds 4 MiB; skipped remainder") from None
                continue
            buf = buf[end:]
            if isinstance(item, dict):
                yield item


def classify(name, category=""):
    name = str(name).lower()
    cat = str(category).lower()
    # Host collective calls must not be mixed with device collective execution.
    if cat in ("cpu_op", "operator", "python_function", "runtime", "runtime_api"):
        return "cpu"
    if any(
        s in name for s in ("reducescatter", "reduce_scatter", "allreduce", "all_reduce", "allgather", "all_gather")
    ):
        return "collective_candidate"
    if any(s in name for s in ("memcpy", "h2d", "d2h")):
        return "copy"
    if "kernel" in cat or cat in ("npu", "aicore", "aicpu"):
        return "device"
    return "unknown"


def automatic_map(columns):
    """Only unambiguous, unit-bearing columns; bare timestamps require a map."""
    lower = {str(c).lower(): c for c in columns}
    mapping = {k: lower[k] for k in FIELDS if k in lower}
    for dest, aliases in {
        "name": ("name", "op_name", "op name", "operator name"),
        "rank": ("rank", "rank_id", "rankid"),
        "device": ("device", "device_id", "deviceid"),
        "pid": ("pid", "process_id"),
        "stream": ("stream", "stream_id", "streamid"),
    }.items():
        for alias in aliases:
            if alias in lower:
                mapping[dest] = lower[alias]
                break
    unit = None
    for suffix in UNITS:
        for dest, names in {
            "start": ("start", "start_time", "timestamp", "ts"),
            "end": ("end", "end_time"),
            "duration": ("duration", "dur"),
        }.items():
            for name in names:
                for candidate in (f"{name}_{suffix}", f"{name}({suffix})", f"{name} ({suffix})"):
                    if candidate in lower:
                        if unit is not None and unit != suffix:
                            return {"columns": {}, "reason": "mixed time units require explicit map"}
                        mapping[dest], unit = lower[candidate], suffix
    constants = {"time_unit": unit} if unit else {}
    return {"columns": mapping, "constants": constants}


def mapped_row(row, spec):
    out = {k: row.get(v) for k, v in spec.get("columns", {}).items() if k in FIELDS}
    out.update({k: v for k, v in spec.get("constants", {}).items() if k in FIELDS})
    for key in ("clock_verified", "sequence_verified"):
        out[key] = spec.get("constants", {}).get(key) is True
    return out


def normalize(row, source):
    event = {k: row.get(k) for k in FIELDS}
    event["source"] = source
    event["kind"] = event["kind"] or classify(event["name"])
    event["start_ns"] = ns(row.get("start"), row.get("time_unit"))
    event["duration_ns"] = ns(row.get("duration"), row.get("time_unit"))
    event["end_ns"] = ns(row.get("end"), row.get("time_unit"))
    if event["end_ns"] is None and event["start_ns"] is not None and event["duration_ns"] is not None:
        event["end_ns"] = event["start_ns"] + event["duration_ns"]
    if event["duration_ns"] is None and event["start_ns"] is not None and event["end_ns"] is not None:
        event["duration_ns"] = event["end_ns"] - event["start_ns"]
    if event["duration_ns"] is not None and event["duration_ns"] < 0:
        raise ValueError("negative duration")
    if event["start_ns"] is None:
        raise ValueError("missing start time/unit; memory lifetime is not CPU duration")
    # SQLite tables can use different clocks even inside ONE database. Until
    # calibrated, scope to the source/table (one Chrome trace is one source).
    event["scope"] = (
        f"{event['run']}|{event['clock']}"
        if event["clock_verified"] is True and event["run"] and event["clock"]
        else f"source:{source}"
    )
    for key in ("rank", "step", "micro_batch"):
        if event[key] is not None:
            event[key] = str(event[key])
    for key in ("clock_verified", "sequence_verified"):
        # Verification must be explicitly supplied by a reviewed schema-map,
        # not inferred from a profiler's unrelated columns named similarly.
        event[key] = event[key] is True
    return event


def collective_key(event):
    required = ("run", "clock", "group", "members", "collective", "collective_seq", "step", "bytes", "rank")
    if event["kind"] != "collective":
        return None, "not verified device collective"
    if event["clock_verified"] is not True or event["sequence_verified"] is not True:
        return None, "clock/sequence correspondence not verified"
    missing = [k for k in required if event[k] is None or event[k] == ""]
    if missing:
        return None, "missing " + ",".join(missing)
    members = event["members"]
    if isinstance(members, str):
        try:
            members = json.loads(members)
        except ValueError:
            return None, "members must be an explicit JSON rank list"
    if not isinstance(members, list) or not members or len(members) != len(set(map(str, members))):
        return None, "invalid members"
    members = sorted(map(str, members))
    if event["rank"] not in members:
        return None, "rank outside group members"
    event["members"] = members
    key = {k: event[k] for k in required if k != "rank"}
    key.update(micro_batch=event["micro_batch"], phase=event["phase"])
    return dumps(key), None


class Analyzer:
    def __init__(self, args):
        self.args = args
        self.inventory = []
        self.gaps = Counter()
        self.counts = Counter()
        self.maps = json.loads(args.schema_map.read_text(encoding="utf-8")) if args.schema_map else {"sources": []}
        self.conn = sqlite3.connect(args.output_dir / "local_events.sqlite")
        self.conn.execute(
            "CREATE TABLE events(id INTEGER PRIMARY KEY, source TEXT, scope TEXT, rank TEXT, "
            "step TEXT, kind TEXT, name TEXT, start INTEGER, end INTEGER, dur INTEGER, "
            "match_key TEXT, reason TEXT, payload TEXT)"
        )

    def spec(self, source, table=None):
        matches = [
            s for s in self.maps.get("sources", []) if fnmatch.fnmatch(source, s["glob"]) and s.get("table") == table
        ]
        if len(matches) > 1:
            raise ValueError(f"ambiguous schema-map for {source}::{table}")
        return matches[0] if matches else None

    def add(self, row, source):
        if len(dumps(row).encode("utf-8")) > 16000:
            self.gaps[f"{source}: oversized mapped row (>16KB), excluded"] += 1
            return
        try:
            e = normalize(row, source)
        except (ValueError, TypeError, ArithmeticError) as exc:
            self.gaps[f"{source}: {exc}"] += 1
            return
        if self.args.step is not None and e["step"] != self.args.step:
            self.counts["excluded_step_unknown" if e["step"] is None else "excluded_step_other"] += 1
            return
        if self.args.ranks is not None and e["rank"] not in self.args.ranks:
            self.counts["excluded_rank_unknown" if e["rank"] is None else "excluded_rank_other"] += 1
            return
        key, reason = collective_key(e)
        self.conn.execute(
            "INSERT INTO events VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                source,
                e["scope"],
                e["rank"],
                e["step"],
                e["kind"],
                str(e["name"]),
                e["start_ns"],
                e["end_ns"],
                e["duration_ns"],
                key,
                reason,
                dumps(e),
            ),
        )
        self.counts["accepted"] += 1
        if self.counts["accepted"] % 10000 == 0:
            self.conn.commit()

    def sqlite(self, path, entry):
        with closing(readonly(path)) as conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
            entry["tables"] = []
            for (table,) in tables[:200]:
                columns = conn.execute(f"PRAGMA table_info({quote(table)})").fetchall()
                names = [c[1] for c in columns]
                meta = {"name": table, "columns": [{"name": c[1], "type": c[2]} for c in columns[:100]]}
                entry["tables"].append(meta)
                if len(columns) > 100:
                    meta["columns_truncated"] = len(columns) - 100
                spec = self.spec(entry["path"], table) or automatic_map(names)
                meta["mapped_fields"] = list(spec.get("columns", {}))
                if self.args.inventory_only:
                    continue
                mapping = spec.get("columns", {})
                if "start" not in mapping or not ("time_unit" in mapping or spec.get("constants", {}).get("time_unit")):
                    self.gaps[f"{entry['path']}::{table}: timestamp/unit adapter missing"] += 1
                    continue
                if not set(mapping.values()) <= set(names):
                    self.gaps[f"{entry['path']}::{table}: mapped columns absent"] += 1
                    continue
                selected = list(dict.fromkeys(mapping.values()))
                cursor = conn.execute(f"SELECT {','.join(map(quote, selected))} FROM {quote(table)}")
                for values in cursor:
                    self.add(mapped_row(dict(zip(selected, values, strict=False)), spec), entry["path"] + "::" + table)
            if len(tables) > 200:
                self.gaps[f"{entry['path']}: tables truncated after 200"] += len(tables) - 200

    def csv(self, path, entry):
        with open_text(path) as src:
            rows = csv.DictReader(src)
            columns = rows.fieldnames or []
            entry["columns"] = columns[:100]
            if self.args.inventory_only:
                return
            spec = self.spec(entry["path"]) or automatic_map(columns)
            if "start" not in spec.get("columns", {}):
                self.gaps[f"{entry['path']}: CSV needs schema-map; allocation lifetime not CPU duration"] += 1
                return
            for row in rows:
                self.add(mapped_row(row, spec), entry["path"])

    def trace(self, path, entry):
        if self.args.inventory_only:
            # Read just one event and only report field names; never expose arbitrary args.
            items = trace_items(path)
            try:
                first = next(items, {})
                entry["first_event_fields"] = list(first)
                entry["first_args_fields"] = list(first.get("args", {}))[:100]
            except ValueError:
                # Communication JSON and other nested JSON: bounded structural
                # inventory only. No whole-file loads and no invented timing semantics.
                with open_text(path) as src:
                    prefix = src.read(65536)
                entry["json_key_sample"] = list(dict.fromkeys(re.findall(r'"([^"\\]{1,100})"\s*:', prefix)))[:80]
                entry["capability_gap"] = "nested JSON adapter needed; key sample is structure, not correspondence"
            finally:
                items.close()
            return
        spec = self.spec(entry["path"])
        for item in trace_items(path):
            if item.get("ph") not in ("X", "C", "i", "I"):
                self.counts["unsupported_trace_phase_" + str(item.get("ph"))] += 1
                continue
            name, cat = item.get("name", ""), item.get("cat", "")
            row = {
                "name": name,
                "start": item.get("ts"),
                "duration": item.get("dur"),
                "time_unit": "us",
                "pid": item.get("pid"),
                "stream": item.get("tid"),
                "kind": "memory" if item.get("ph") == "C" else classify(name, cat),
            }
            flat = dict(item)
            flat.update({f"args.{k}": v for k, v in item.get("args", {}).items()})
            # Explicit rank metadata only; pid/device never substituted for rank.
            for k in ("rank", "step", "device"):
                row[k] = item.get("args", {}).get(k)
            if spec:
                row.update(mapped_row(flat, spec))
            self.add(row, entry["path"])

    def scan(self):
        files = 0
        candidates = []
        for path in self.args.profile_root.rglob("*"):
            if not path.is_file() or path.is_relative_to(self.args.output_dir):
                continue
            files += 1
            entry = {"path": path.relative_to(self.args.profile_root).as_posix(), "bytes": path.stat().st_size}
            try:
                with path.open("rb") as src:
                    header = src.read(16)
                if header == b"SQLite format 3\x00":
                    fmt, priority = "sqlite", 0
                elif path.name.endswith((".csv", ".csv.gz")):
                    fmt, priority = "csv", 1
                elif path.name.endswith((".json", ".json.gz")):
                    fmt, priority = "json", 2
                else:
                    fmt, priority = "raw_or_unsupported", 3
            except OSError as exc:
                fmt, priority = "unreadable", 3
                entry["capability_gap"] = str(exc)[:300]
            entry["format"] = fmt
            candidate = (-priority, -files, path, entry)
            if len(candidates) < 1000:
                heapq.heappush(candidates, candidate)
            elif candidate > candidates[0]:
                heapq.heapreplace(candidates, candidate)
        self.counts["inventory_files_omitted"] = files - len(candidates)
        # Existing parsed databases first; retain them even when a raw capture
        # contains thousands of small files. No unbounded file list in memory.
        for _, _, path, entry in sorted(candidates, key=lambda v: (-v[0], v[3]["path"])):
            self.inventory.append(entry)
            try:
                if entry["format"] == "sqlite":
                    self.sqlite(path, entry)
                elif entry["format"] == "csv":
                    self.csv(path, entry)
                elif entry["format"] == "json":
                    self.trace(path, entry)
            except (OSError, ValueError, sqlite3.Error, UnicodeError, EOFError) as exc:
                entry["capability_gap"] = str(exc)[:500]
                self.gaps[f"{entry['path']}: {str(exc)[:300]}"] += 1
        self.counts["discovered_files"] = files
        self.conn.commit()
        self.conn.execute("CREATE INDEX event_match ON events(match_key)")
        self.conn.execute("CREATE INDEX event_window ON events(scope,rank,start)")

    def rows(self, query, args=()):
        return [json.loads(r[0]) for r in self.conn.execute(query, args)]

    def report(self):
        k = self.args.top_k
        groups, late = [], Counter()
        group_count = 0
        query = (
            "SELECT match_key,MAX(start)-MIN(start) skew FROM events WHERE match_key IS NOT NULL "
            "GROUP BY match_key ORDER BY skew DESC"
        )
        for key, skew in self.conn.execute(query):
            events = self.rows("SELECT payload FROM events WHERE match_key=? ORDER BY rank", (key,))
            expected = set(events[0]["members"])
            ranks = [e["rank"] for e in events]
            if len(ranks) != len(set(ranks)) or set(ranks) != expected:
                self.gaps["collective group incomplete or duplicate rank; rejected cross-rank matching"] += 1
                continue
            group_count += 1
            last = max(e["start_ns"] for e in events)
            latest = [e["rank"] for e in events if e["start_ns"] == last]
            late.update(latest)
            if len(groups) < k:
                groups.append(
                    {
                        "key": json.loads(key),
                        "arrival_skew_ns": skew,
                        "latest_ranks": latest,
                        "events": events,
                        "meaning": "device start skew; NOT proven host submission skew",
                    }
                )
        top = self.rows(
            "SELECT payload FROM events WHERE kind='cpu' AND name LIKE '%aten::empty%' ORDER BY dur DESC LIMIT ?", (k,)
        )
        uncertain = self.rows(
            "SELECT payload FROM events WHERE kind IN ('collective','collective_candidate') "
            "AND match_key IS NULL ORDER BY dur DESC LIMIT ?",
            (k,),
        )
        # Same-file/same-calibrated-clock windows, anchored at long host allocations.
        windows = []
        for anchor in top[: min(5, k)]:
            lo, hi = anchor["start_ns"], anchor["end_ns"] or anchor["start_ns"]
            neighbours = self.rows(
                "SELECT payload FROM events WHERE scope=? AND rank IS ? AND start BETWEEN ? AND ? "
                "AND kind IN ('collective','collective_candidate','copy','memory','device','cpu') "
                "ORDER BY start LIMIT 120",
                (anchor["scope"], anchor["rank"], lo - 100000000, hi + 100000000),
            )
            previous = self.rows(
                "SELECT payload FROM events WHERE scope=? AND rank IS ? AND end<=? "
                "AND kind IN ('device','collective','collective_candidate','copy') "
                "ORDER BY end DESC LIMIT 5",
                (anchor["scope"], anchor["rank"], hi),
            )
            following = self.rows(
                "SELECT payload FROM events WHERE scope=? AND rank IS ? AND start>=? "
                "AND kind IN ('collective','collective_candidate') ORDER BY start LIMIT 3",
                (anchor["scope"], anchor["rank"], hi),
            )
            windows.append(
                {
                    "anchor": anchor,
                    "nearby_first_120": neighbours,
                    "preceding_completions": previous,
                    "following_device_collectives": following,
                    "limitation": (
                        "temporal association only; free/allocator and submission links require explicit fields"
                    ),
                }
            )
        memory = self.rows("SELECT payload FROM events WHERE kind='memory' ORDER BY start LIMIT ?", (k,))
        memory_peaks, previous_memory, drops = [], {}, []
        pending_frees, free_pairs = {}, []
        for (payload,) in self.conn.execute("SELECT payload FROM events WHERE kind='memory' ORDER BY start"):
            e = json.loads(payload)
            if e["address"] is not None and e["allocator"] is not None and e["device"] is not None:
                addr_key = (e["scope"], e["rank"], str(e["device"]), str(e["pid"]), e["allocator"], str(e["address"]))
                if e["action"] == "FREE_REQUESTED":
                    pending_frees[addr_key] = e
                elif e["action"] == "ALLOC":
                    pending_frees.pop(addr_key, None)
                elif e["action"] == "FREE_COMPLETED":
                    req = pending_frees.pop(addr_key, None)
                    if req:
                        free_pairs.append(
                            {"request": req, "completion": e, "delay_ns": e["start_ns"] - req["start_ns"]}
                        )
                        free_pairs.sort(key=lambda x: x["delay_ns"], reverse=True)
                        del free_pairs[k:]
            if not e["allocator"] or not e["memory_unit"] or e["device"] is None:
                self.counts["memory_samples_missing_allocator_unit_or_device"] += 1
                continue
            if any(e[f] is None for f in ("active", "allocated", "reserved")):
                self.counts["memory_samples_missing_counter"] += 1
                continue
            try:
                scale = {"B": 1, "KB": 1000, "MB": 1000000, "KiB": 1024, "MiB": 1048576}[e["memory_unit"]]
                gap = (Decimal(str(e["active"])) - Decimal(str(e["allocated"]))) * scale
                active = Decimal(str(e["active"])) * scale
            except (ArithmeticError, KeyError):
                self.counts["invalid_memory_counter"] += 1
                continue
            scope = (e["scope"], e["rank"], str(e["device"]), str(e["pid"]), e["allocator"], e["memory_unit"])
            item = {
                "active_minus_allocated_bytes": str(gap),
                "sample": e,
                "interpretation": "pending reuse clue; not attributed to RS or gradients",
            }
            memory_peaks.append(item)
            memory_peaks.sort(key=lambda x: Decimal(x["active_minus_allocated_bytes"]), reverse=True)
            del memory_peaks[k:]
            previous = previous_memory.get(scope)
            if previous and active < previous[0]:
                drops.append({"active_drop_bytes": str(previous[0] - active), "before": previous[1], "after": e})
                drops.sort(key=lambda x: Decimal(x["active_drop_bytes"]), reverse=True)
                del drops[k:]
            previous_memory[scope] = (active, e)
        counterexamples = self.rows(
            "SELECT payload FROM events WHERE kind='cpu' AND name LIKE '%aten::empty%' "
            "AND dur IS NOT NULL ORDER BY dur ASC LIMIT ?",
            (min(k, 10),),
        )
        result = {
            "format_version": 1,
            "filters": {"step": self.args.step, "ranks": self.args.ranks},
            "counts": dict(self.counts),
            "inventory": self.inventory,
            "capability_gaps": dict(self.gaps.most_common(300)),
            "matched_collective_count": group_count,
            "latest_rank_counts_all_matched": dict(late),
            "collectives_top": groups,
            "uncertain_collectives_top": uncertain,
            "long_cpu_empty": top,
            "short_cpu_empty_counterexamples": counterexamples,
            "memory_samples_first": memory,
            "memory_gap_top": memory_peaks,
            "memory_active_drops_top": drops,
            "free_request_completion_pairs_top": free_pairs,
            "allocation_windows": windows,
            "evidence_boundary": [
                "No root cause established by temporal correlation alone.",
                "Missing wait/transfer/self_time/allocator/free fields are null, not zero.",
                "CPU X duration is inclusive; memory lifetime and device kernel duration are separate.",
                "Sync and Wait are never added. Reserved is not a leak metric.",
                "Unknown step/rank rows are excluded when the corresponding filter is supplied.",
                "Rank rotation covers all completely matched collectives, not only Top-K.",
                "Memory differences require explicit allocator and byte units; no buffer ownership attribution.",
                "FREE pairs require explicit action/address/allocator/device; absence cannot prove delayed frees.",
            ],
        }
        # Bound return artifacts. The full intermediate SQLite stays on the server.
        if len(dumps(result).encode()) > 3500000:
            result["allocation_windows"] = []
            result["capability_gaps"]["return budget: windows retained only in local_events.sqlite"] = 1
        if len(dumps(result).encode()) > 3500000:
            result["inventory"] = result["inventory"][:100]
            result["capability_gaps"]["return budget: inventory truncated to 100 files"] = 1
        for key in (
            "collectives_top",
            "memory_gap_top",
            "memory_active_drops_top",
            "free_request_completion_pairs_top",
        ):
            while result[key] and len(dumps(result).encode()) > 3500000:
                result[key].pop()
                result["capability_gaps"]["return budget: Top-K records truncated"] = 1
        out = self.args.output_dir
        (out / "summary.json").write_text(dumps(result), encoding="utf-8")
        export_csv(out / "cpu_empty_top.csv", top)
        export_csv(out / "collectives_top.csv", [e for g in groups for e in g["events"]])
        export_csv(out / "uncertain_top.csv", uncertain)
        lines = [
            "# Profiling 首轮摘要",
            "",
            f"文件数：{files_count(self.counts)}；接受事件：{self.counts['accepted']}；完整匹配通信组：{group_count}。",
            "",
            "跨 rank 比较仅启用于已确认 run、公共时钟、通信成员和全局序号的记录。",
            "CPU aten::empty 区间只表示主机 inclusive time；没有将内存存活期替代为调用耗时。",
            "",
            f"最晚 rank 计数（全部完整匹配组）：`{dict(late)}`。",
            "",
            "## 能力缺口",
            "",
        ]
        lines.extend(f"- {reason}（{count}）" for reason, count in self.gaps.most_common(60))
        if not self.counts["accepted"]:
            lines.append("- 尚无可解释的事件。先依据 summary.json 的文件/schema 清单添加适配；无需重跑训练。")
        lines.extend(
            [
                "",
                "## 回传文件",
                "",
                "summary.md、summary.json、cpu_empty_top.csv、collectives_top.csv、uncertain_top.csv。",
                "local_events.sqlite 是服务器本地中间结果，不回传。原始 profiling 未修改。",
                "",
                "## 边界",
                "",
                *[f"- {v}" for v in result["evidence_boundary"]],
            ]
        )
        (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.conn.close()
        return result


def files_count(counts):
    return counts.get("discovered_files", 0)


def export_csv(path, events):
    columns = (
        "source",
        "rank",
        "device",
        "pid",
        "stream",
        "step",
        "micro_batch",
        "phase",
        "kind",
        "name",
        "start",
        "end",
        "duration",
        "time_unit",
        "start_ns",
        "end_ns",
        "duration_ns",
        "self_time",
        "wait",
        "wait_unit",
        "transfer",
        "transfer_unit",
        "group",
        "collective_seq",
        "bytes",
    )
    with path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, columns, extrasaction="ignore")
        writer.writeheader()
        for event in events:
            writer.writerow({k: event.get(k) for k in columns})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--step", help="exact step; unknown step rows are excluded, never assumed")
    p.add_argument("--ranks", help="comma-separated global ranks; unknown rank rows are excluded")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument("--schema-map", type=Path)
    args = p.parse_args()
    args.profile_root, args.output_dir = args.profile_root.resolve(), args.output_dir.resolve()
    if not args.profile_root.is_dir():
        p.error("--profile-root must be an existing directory containing ONE run")
    if not 1 <= args.top_k <= 100:
        p.error("--top-k must be 1..100")
    if args.output_dir == args.profile_root or args.profile_root.is_relative_to(args.output_dir):
        p.error("output directory must not equal or contain the input directory")
    if (args.output_dir / "local_events.sqlite").exists():
        p.error("output already contains a summary database; choose a NEW output directory")
    args.ranks = args.ranks.split(",") if args.ranks else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / ".gitignore").write_text("*\n", encoding="utf-8")
    analyzer = Analyzer(args)
    analyzer.scan()
    result = analyzer.report()
    print(f"Summary: {args.output_dir / 'summary.md'}; events={result['counts'].get('accepted', 0)}")


if __name__ == "__main__":
    main()
