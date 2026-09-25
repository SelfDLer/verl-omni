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

"""Control-order tests only: no claim to emulate an NPU allocator or HCCL."""

import importlib.util
import json
import unittest
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from codex_test.veomni_memory.compare_rs_ab import compare, load_records, numeric_close
from codex_test.veomni_memory.rs_ab import BackwardMetrics, _config_hash

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "rs_limiter_test_target", ROOT / "verl_omni/workers/engine/veomni/rs_limiter.py"
)
limiter_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(limiter_module)


class Buffer:
    pass


class LimiterTests(unittest.TestCase):
    def group(self, *, grad=True, reduce=True, previous=True):
        log = []
        buffer = Buffer()
        ref = weakref.ref(buffer)
        event = NS(synchronize=lambda: log.append("host_wait"))
        state = NS(event=event, input=buffer) if previous else None
        group = NS(
            comm_ctx=NS(reduce_scatter_state=state),
            reduce_grads=reduce,
            fsdp_params=[
                NS(_unsharded_param=True, unsharded_param=NS(grad=1 if grad else None), unsharded_accumulated_grad=None)
            ],
        )

        def original(value=None):
            log.append("original")
            if reduce and grad:
                log.append("device_wait")
                group.comm_ctx.reduce_scatter_state = None
                self.assertIsNone(ref(), "wrapper must not retain the old RS input")
                log.append("allocate_and_collective")
            return value

        group.post_backward = original
        return group, log, event

    def test_default_off_does_not_import_torch_or_mutate_engine(self):
        engine = NS()
        with patch.dict("os.environ", {}, clear=True), patch.object(limiter_module, "audited_groups") as audit:
            self.assertIsNone(limiter_module.install(engine))
            audit.assert_not_called()
            self.assertEqual(vars(engine), {})

    def test_explicit_off_and_invalid_flag(self):
        with patch.dict("os.environ", {"VEOMNI_LIMIT_RS_INFLIGHT": "0"}):
            self.assertIsNone(limiter_module.install(NS()))
        with patch.dict("os.environ", {"VEOMNI_LIMIT_RS_INFLIGHT": "yes"}):
            with self.assertRaises(ValueError):
                limiter_module.install(NS())

    def test_reference_engine_is_not_patched(self):
        with patch.dict("os.environ", {"VEOMNI_LIMIT_RS_INFLIGHT": "1"}):
            with patch.object(limiter_module, "audited_groups") as audit:
                self.assertIsNone(limiter_module.install(NS(engine_config=NS(forward_only=True))))
                audit.assert_not_called()

    def test_wait_before_reference_release_and_next_allocation(self):
        group, log, _ = self.group()
        limiter = limiter_module.RSLimiter()
        limiter.wrap(group)
        self.assertEqual(group.post_backward("result"), "result")
        self.assertEqual(log, ["host_wait", "original", "device_wait", "allocate_and_collective"])
        self.assertEqual(limiter.wait_calls, 1)

    def test_no_previous_event(self):
        group, log, _ = self.group(previous=False)
        limiter_module.RSLimiter().wrap(group)
        group.post_backward()
        self.assertNotIn("host_wait", log)

    def test_no_sync_and_unused_group_do_not_wait(self):
        for options in ({"reduce": False}, {"grad": False}):
            group, log, _ = self.group(**options)
            limiter_module.RSLimiter().wrap(group)
            group.post_backward()
            self.assertEqual(log, ["original"])

    def test_accumulated_gradient_still_waits(self):
        group, log, _ = self.group(grad=False)
        group.fsdp_params[0].unsharded_accumulated_grad = object()
        limiter_module.RSLimiter().wrap(group)
        group.post_backward()
        self.assertEqual(log[0], "host_wait")

    def test_failed_wait_never_releases_or_issues_next_collective(self):
        group, log, event = self.group()
        old_state = group.comm_ctx.reduce_scatter_state

        def fail():
            raise RuntimeError("device wait failed")

        event.synchronize = fail
        limiter_module.RSLimiter().wrap(group)
        with self.assertRaisesRegex(RuntimeError, "device wait failed"):
            group.post_backward()
        self.assertIs(group.comm_ctx.reduce_scatter_state, old_state)
        self.assertEqual(log, [])

    def test_wrap_is_instance_local(self):
        target, _, _ = self.group()
        other, log, _ = self.group()
        limiter_module.RSLimiter().wrap(target)
        other.post_backward()
        self.assertNotIn("host_wait", log)


class MetricsTests(unittest.TestCase):
    def test_config_hash_ignores_runtime_object_addresses_but_keeps_training_flags(self):
        @dataclass
        class Config:
            offload: bool
            processor: object

        engine = NS(model_config=Config(True, object()), engine_config={}, optimizer_config={})
        with patch.dict("sys.modules", {"omegaconf": NS(OmegaConf=NS(is_config=lambda _: False))}):
            initial = _config_hash(engine)
            engine.model_config.processor = object()
            self.assertEqual(_config_hash(engine), initial)
            engine.model_config.offload = False
            self.assertNotEqual(_config_hash(engine), initial)

    def test_retry_delta_peak_and_gap_are_distinct(self):
        stats = {
            "num_alloc_retries": 5,
            "active_bytes.all.current": 100,
            "allocated_bytes.all.current": 60,
            "active_bytes.all.peak": 100,
            "allocated_bytes.all.peak": 60,
        }
        metric = object.__new__(BackwardMetrics)
        metric.records, metric.current, metric.limiter = [], None, NS(wait_ns=100, wait_calls=2)
        metric.npu = NS(
            memory_stats=lambda: dict(stats),
            reset_peak_memory_stats=lambda: None,
            Event=lambda **_: NS(record=lambda: None, elapsed_time=lambda other: 12.0),
        )

        def backward():
            stats.update(
                {
                    "num_alloc_retries": 7,
                    "active_bytes.all.current": 130,
                    "allocated_bytes.all.current": 100,
                    "active_bytes.all.peak": 150,
                    "allocated_bytes.all.peak": 120,
                }
            )
            metric.limiter.wait_ns += 2_000_000
            metric.limiter.wait_calls += 1
            return "ok"

        self.assertEqual(metric.backward(NS(backward=backward)), "ok")
        record = metric.completed_records()[0]
        self.assertEqual(record["alloc_retries"], 2)
        self.assertEqual(record["gap_sampled_max_bytes"], 40)  # not 150 - 120
        self.assertEqual(record["limiter_wait_ms"], 2)
        self.assertEqual(record["limiter_wait_calls"], 1)
        self.assertEqual(record["backward_device_ms"], 12)

    def test_failure_clears_measurement_scope(self):
        metric = object.__new__(BackwardMetrics)
        metric.records, metric.current, metric.limiter = [], None, None
        metric.npu = NS(
            memory_stats=lambda: dict.fromkeys(
                (
                    "num_alloc_retries",
                    "active_bytes.all.current",
                    "allocated_bytes.all.current",
                    "active_bytes.all.peak",
                    "allocated_bytes.all.peak",
                ),
                0,
            ),
            reset_peak_memory_stats=lambda: None,
            Event=lambda **_: NS(record=lambda: None),
        )
        with self.assertRaisesRegex(RuntimeError, "backward failed"):
            metric.backward(NS(backward=lambda: (_ for _ in ()).throw(RuntimeError("backward failed"))))
        self.assertIsNone(metric.current)


class ComparisonTests(unittest.TestCase):
    def test_pair_aggregates_critical_rank_time_and_detects_numerical_change(self):
        a, b = {}, {}
        for rank in range(2):
            for update in range(2):
                backward = dict.fromkeys(
                    (
                        "alloc_retries",
                        "active_peak_bytes",
                        "allocated_peak_bytes",
                        "gap_sampled_max_bytes",
                        "backward_host_ms",
                        "backward_device_ms",
                        "limiter_wait_ms",
                        "limiter_wait_calls",
                    ),
                    0,
                )
                record = {
                    "rank": rank,
                    "update": update,
                    "world_size": 2,
                    "fixture_sha256": "fixed",
                    "config_sha256": "config",
                    "initial_shard_sha256": "initial",
                    "native_offload": True,
                    "backwards": [backward],
                    "loss": [1.0],
                    "grad_norm": [2.0],
                    "update_alloc_retries": 0,
                    "update_actor_drained_ms": 10 + rank * 10,
                }
                a[rank, update] = {**record, "limit_rs": False}
                b[rank, update] = {**record, "limit_rs": True}
        summary = compare(a, b, 1)
        self.assertTrue(summary["numerical_metrics_close"])
        self.assertEqual(summary["on"]["update_actor_rankmax_ms"], [20])
        self.assertEqual(summary["on"]["rank_backwards"], 2)
        self.assertLess(len(json.dumps(summary).encode()), 8192)
        b[0, 1]["loss"] = [1.1]
        self.assertFalse(compare(a, b, 1)["numerical_metrics_close"])

    def test_numeric_missing_nan_shape_mismatch_are_not_success(self):
        for a, b in ((None, None), (float("nan"), float("nan")), ([1], [1, 2]), ([], [])):
            self.assertFalse(numeric_close(a, b, 1e-5, 1e-6))
        self.assertTrue(numeric_close([1.0], [1.000001], 1e-5, 1e-6))

    def test_missing_rank_is_an_error(self):
        file = NS(read_text=lambda **_: json.dumps({"rank": 0, "update": 0, "world_size": 2}))
        directory = NS(glob=lambda _: [file])
        with self.assertRaisesRegex(ValueError, "Missing"):
            load_records(directory, 2, 1)

    def test_wrong_fixture_fails_before_performance_comparison(self):
        common = {
            "fixture_sha256": "fixed",
            "config_sha256": "config",
            "initial_shard_sha256": "initial",
            "native_offload": True,
        }
        a = {(0, 0): {**common, "limit_rs": False}}
        b = {(0, 0): {**common, "limit_rs": True, "fixture_sha256": "different"}}
        with self.assertRaisesRegex(ValueError, "fixture_sha256"):
            compare(a, b, 0)


if __name__ == "__main__":
    unittest.main()
