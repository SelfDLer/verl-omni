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

"""Synthetic CPU-only correctness tests. No model, torch, Ray, or NPU imports."""

import argparse
import gzip
import hashlib
import importlib.util
import json
import shutil
import sqlite3
import unittest
import uuid
from contextlib import closing, contextmanager
from decimal import Decimal
from pathlib import Path

MODULE = Path(__file__).with_name("summarize_profile.py")
SPEC = importlib.util.spec_from_file_location("summary", MODULE)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


@contextmanager
def fixture_directory():
    base = Path(__file__).resolve().parent / "results" / "cpu_tests"
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        resolved = path.resolve()
        if resolved.is_relative_to(base.resolve()) and resolved != base.resolve():
            shutil.rmtree(resolved)


class SummaryTests(unittest.TestCase):
    def test_exact_epoch_ns_and_decimal_microseconds(self):
        self.assertEqual(summary.ns("1758800000123456789", "ns"), 1758800000123456789)
        self.assertEqual(summary.ns(Decimal("1758800000123456.789"), "us"), 1758800000123456789)
        self.assertEqual(summary.ns("24.7635", "s"), 24763500000)
        with self.assertRaises(ValueError):
            summary.ns(1.7588000001234568e18, "ns")
        with self.assertRaises(ValueError):
            summary.ns(1, None)

    def test_missing_is_not_zero(self):
        e = summary.normalize({"start": 1, "time_unit": "ns"}, "trace.json")
        self.assertIsNone(e["duration_ns"])
        self.assertIsNone(e["wait"])
        self.assertIsNone(e["active"])

    def event(self, **changes):
        row = dict(
            start=100,
            duration=10,
            time_unit="ns",
            kind="collective",
            name="ReduceScatter",
            run="synthetic",
            clock="same-calibrated-clock",
            clock_verified=True,
            sequence_verified=True,
            group="dp",
            members=[0, 1],
            collective="ReduceScatter",
            collective_seq=4,
            step=2,
            bytes=4096,
            rank=0,
            phase="backward",
        )
        row.update(changes)
        return summary.normalize(row, "one.db::events")

    def test_name_or_local_task_id_not_sufficient(self):
        e = self.event(collective_seq=None)
        e["taskId"] = 42
        key, reason = summary.collective_key(e)
        self.assertIsNone(key)
        self.assertIn("collective_seq", reason)

    def test_clock_and_members_and_sequence(self):
        key, _ = summary.collective_key(self.event())
        other, _ = summary.collective_key(self.event(rank=1, start=300))
        self.assertEqual(key, other)
        for change in (dict(clock="another"), dict(collective_seq=5), dict(step=3), dict(group="ep")):
            self.assertNotEqual(key, summary.collective_key(self.event(**change))[0])
        self.assertIsNone(summary.collective_key(self.event(clock_verified=False))[0])
        self.assertIsNone(summary.collective_key(self.event(rank=8))[0])

    def test_host_collective_not_device_collective(self):
        self.assertEqual(summary.classify("all_reduce", "cpu_op"), "cpu")
        self.assertEqual(summary.classify("hcom_allReduce__612", ""), "collective_candidate")

    def test_unit_columns_and_unknown_schema(self):
        self.assertEqual(summary.automatic_map(["start_ns", "duration_ns"])["constants"]["time_unit"], "ns")
        self.assertEqual(summary.automatic_map(["timestamp", "duration"])["constants"], {})
        self.assertEqual(summary.automatic_map(["start_ns", "duration_us"])["columns"], {})
        self.assertNotIn("start", summary.automatic_map(["Allocation Time(us)", "Release Time(us)"])["columns"])

    def test_large_stream_and_gzip(self):
        with fixture_directory() as temp:
            path = Path(temp) / "trace.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as dst:
                dst.write('{"traceEvents":[')
                for i in range(2000):
                    if i:
                        dst.write(",")
                    dst.write('{"name":"aten::empty","ph":"X","cat":"cpu_op","ts":1758800000123456.789,"dur":0.001}')
                dst.write("]}")
            rows = summary.trace_items(path)
            count = 0
            for row in rows:
                self.assertEqual(summary.ns(row["ts"], "us"), 1758800000123456789)
                count += 1
            self.assertEqual(count, 2000)

    def analyze(self, root, records, **kwargs):
        root = Path(root)
        inp, out = root / "input", root / "output"
        inp.mkdir()
        out.mkdir()
        db = inp / "profile.db"
        with closing(sqlite3.connect(db)) as conn, conn:
            conn.execute("CREATE TABLE events(name TEXT,start_ns INTEGER,duration_ns INTEGER,rank INTEGER,seq INTEGER)")
            conn.executemany("INSERT INTO events VALUES (?,?,?,?,?)", records)
            conn.execute("CREATE TABLE unknown(memory_lifetime INTEGER, taskId INTEGER)")
        original = hashlib.sha256(db.read_bytes()).hexdigest()
        spec = {
            "sources": [
                {
                    "glob": "profile.db",
                    "table": "events",
                    "columns": {
                        "name": "name",
                        "start": "start_ns",
                        "duration": "duration_ns",
                        "rank": "rank",
                        "collective_seq": "seq",
                    },
                    "constants": {
                        "kind": "collective",
                        "time_unit": "ns",
                        "step": 2,
                        "phase": "backward",
                        "run": "fixture",
                        "clock": "fixture-global",
                        "clock_verified": True,
                        "sequence_verified": True,
                        "group": "dp",
                        "members": [0, 1],
                        "collective": "ReduceScatter",
                        "bytes": 4096,
                    },
                }
            ]
        }
        mapping = root / "map.json"
        mapping.write_text(json.dumps(spec))
        args = argparse.Namespace(
            profile_root=inp, output_dir=out, schema_map=mapping, inventory_only=False, step=None, ranks=None, top_k=2
        )
        for k, v in kwargs.items():
            setattr(args, k, v)
        analyzer = summary.Analyzer(args)
        analyzer.scan()
        result = analyzer.report()
        self.assertEqual(original, hashlib.sha256(db.read_bytes()).hexdigest())
        return result

    def test_sqlite_matches_repeated_ops_and_late_rank_rotation(self):
        with fixture_directory() as temp:
            result = self.analyze(
                temp,
                [
                    ("same_name", 100, 20, 0, 1),
                    ("same_name", 110, 10, 1, 1),
                    ("same_name", 230, 10, 0, 2),
                    ("same_name", 200, 40, 1, 2),
                ],
            )
            self.assertEqual(result["matched_collective_count"], 2)
            self.assertEqual(result["latest_rank_counts_all_matched"], {"0": 1, "1": 1})
            self.assertEqual(result["collectives_top"][0]["arrival_skew_ns"], 30)
            self.assertTrue(any("timestamp/unit adapter missing" in s for s in result["capability_gaps"]))

    def test_duplicate_rank_is_not_matched(self):
        with fixture_directory() as temp:
            result = self.analyze(temp, [("x", 100, 10, 0, 1), ("x", 120, 10, 0, 1), ("x", 150, 10, 1, 1)])
            self.assertEqual(result["matched_collective_count"], 0)

    def test_incomplete_rank_filter_does_not_claim_full_skew(self):
        with fixture_directory() as temp:
            result = self.analyze(temp, [("x", 100, 10, 0, 1), ("x", 150, 10, 1, 1)], ranks=["0"])
            self.assertEqual(result["matched_collective_count"], 0)
            self.assertEqual(result["counts"]["excluded_rank_other"], 1)

    def test_explicit_step_filter(self):
        with fixture_directory() as temp:
            result = self.analyze(temp, [("x", 100, 10, 0, 1)], step="3")
            self.assertEqual(result["counts"]["excluded_step_other"], 1)

    def test_sqlite_ro_rejects_writes(self):
        with fixture_directory() as temp:
            path = Path(temp) / "original.db"
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute("CREATE TABLE x(a)")
            with closing(summary.readonly(path)) as conn:
                with self.assertRaises(sqlite3.OperationalError):
                    conn.execute("INSERT INTO x VALUES(1)")

    def test_different_tables_not_assumed_same_clock(self):
        a = summary.normalize(dict(start=100, time_unit="ns"), "one.db::host")
        b = summary.normalize(dict(start=100, time_unit="ns"), "one.db::device")
        self.assertNotEqual(a["scope"], b["scope"])

    def test_memory_units_frees_and_cpu_lifetime_are_separate(self):
        with fixture_directory() as temp:
            inp, out = temp / "input", temp / "output"
            inp.mkdir()
            out.mkdir()
            args = argparse.Namespace(
                profile_root=inp, output_dir=out, schema_map=None, inventory_only=False, step=None, ranks=None, top_k=5
            )
            analyzer = summary.Analyzer(args)
            for start, active in ((100, 55), (200, 9)):
                analyzer.add(
                    dict(
                        start=start,
                        time_unit="ns",
                        kind="memory",
                        name="counters",
                        rank=0,
                        device=0,
                        pid=1,
                        allocator="PTA",
                        memory_unit="MiB",
                        active=active,
                        allocated=8,
                        reserved=56,
                    ),
                    "memory.csv",
                )
            for start, action in ((110, "FREE_REQUESTED"), (190, "FREE_COMPLETED")):
                analyzer.add(
                    dict(
                        start=start,
                        time_unit="ns",
                        kind="memory",
                        name="free",
                        rank=0,
                        device=0,
                        pid=1,
                        allocator="PTA",
                        address="0x123",
                        action=action,
                    ),
                    "memory.csv",
                )
            analyzer.add(
                dict(start=100, duration=2000, time_unit="ns", kind="memory_lifetime", name="aten::empty"),
                "allocation_table",
            )
            analyzer.scan()
            result = analyzer.report()
            self.assertEqual(result["long_cpu_empty"], [])
            self.assertEqual(result["memory_gap_top"][0]["active_minus_allocated_bytes"], str(47 * 1048576))
            self.assertEqual(result["memory_active_drops_top"][0]["active_drop_bytes"], str(46 * 1048576))
            self.assertEqual(result["free_request_completion_pairs_top"][0]["delay_ns"], 80)

    def test_nested_communication_json_inventory_is_bounded(self):
        with fixture_directory() as temp:
            inp, out = temp / "input", temp / "output"
            inp.mkdir()
            out.mkdir()
            (inp / "communication.json").write_text('{"step2":{"rank0":{"Wait Time(ms)":12}}}')
            args = argparse.Namespace(
                profile_root=inp, output_dir=out, schema_map=None, inventory_only=True, step=None, ranks=None, top_k=5
            )
            analyzer = summary.Analyzer(args)
            analyzer.scan()
            result = analyzer.report()
            self.assertEqual(result["matched_collective_count"], 0)
            self.assertIn("Wait Time(ms)", result["inventory"][0]["json_key_sample"])

    def test_local_storage_alias_is_counted_once(self):
        spec = importlib.util.spec_from_file_location("worker_probe", MODULE.with_name("worker_probe.py"))
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)

        class LocalStorage:
            def data_ptr(self):
                return 123

            def nbytes(self):
                return 1024

        class LocalTensor:
            device, dtype, shape = "cpu", "float32", (100,)

            def numel(self):
                return 100

            def element_size(self):
                return 4

            def untyped_storage(self):
                return LocalStorage()

        class DistributedTensor:
            _local_tensor = LocalTensor()

            def numel(self):
                raise AssertionError("global numel must not be used")

        seen = set()
        first = probe.storage(DistributedTensor(), seen)
        second = probe.storage(DistributedTensor(), seen)
        self.assertEqual(first["logical_local_bytes"], 400)
        self.assertEqual(first["unique_bytes"], 1024)
        self.assertEqual(second["unique_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
