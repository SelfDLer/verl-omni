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

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "special_e2e/probe_8039_fsdp_bias.py"
PROBE = runpy.run_path(SCRIPT)


@pytest.mark.parametrize("length,size", [(4304, 16), (4304, 32), (4096, 32), (4320, 32), (3, 8)])
def test_shards_cover_the_vector_without_gaps_or_overlap(length, size):
    spans = [PROBE["shard_span"](length, size, rank) for rank in range(size)]
    assert [index for start, count, _ in spans for index in range(start, start + count)] == list(range(length))
    assert len({padded for _, _, padded in spans}) == 1


def test_observed_bias_boundary_matches_last_uneven_shard_hypothesis():
    assert PROBE["shard_span"](4304, 16, 15) == (4035, 269, 269)
    assert PROBE["shard_span"](4304, 32, 31) == (4185, 119, 135)
    # Observations bracket this boundary; they do not prove every element is bad.
    assert 4183 < 4185 <= 4200
    assert 4185 - 3 * 1076 == 957


@pytest.mark.parametrize("length,size,rank", [(0, 32, 0), (4304, 0, 0), (4304, 32, 32), (4304, 32, -1)])
def test_invalid_shard_layout_is_rejected(length, size, rank):
    with pytest.raises(ValueError):
        PROBE["shard_span"](length, size, rank)


def test_complete_comparison_detects_unsampled_first_element_of_bad_shard():
    expected = [1.0] * 119
    actual = expected.copy()
    actual[0] = -2.0
    result = PROBE["compare_values"](actual, expected, 4185)
    assert result["status"] == "different"
    assert result["different"] == 1
    assert result["first_global"] == result["last_global"] == 4185
    assert result["examples"] == [{"global": 4185, "actual": -2.0, "expected": 1.0}]


def test_comparison_bounds_examples_but_preserves_full_count_and_last_position():
    result = PROBE["compare_values"]([2.0] * 119, [1.0] * 119, 4185)
    assert result["different"] == 119
    assert result["last_global"] == 4303
    assert len(result["examples"]) == 6


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_not_a_pass(value):
    assert PROBE["compare_values"]([value], [value])["status"] == "different"


def test_missing_elements_are_not_equality():
    assert PROBE["compare_values"]([], [1.0])["status"] == "shape_mismatch"
    assert PROBE["compare_values"]([1.0], [1.0])["status"] == "equal"


def test_compact_summary_preserves_failure_count_and_stage():
    check = PROBE["compare_values"]([0.0], [1.0], 4185)
    stage = PROBE["compact_stage"]("load.local", {f"block{i}": check for i in range(27)})
    assert stage["counts"] == {"different": 27}
    assert len(stage["failures"]) == 3
    assert stage["failures_omitted"] == 24
    output = PROBE["format_summary"]([{"rank": 31, "stages": [stage]}])
    assert "rank=31 load.local" in output
    assert '"first_global":4185' in output
    assert "All collected checks passed" not in output


def test_summary_is_bounded_for_many_failures():
    check = PROBE["compare_values"]([2.0] * 119, [1.0] * 119, 4185)
    stage = PROBE["compact_stage"]("load.local", {f"block{i}": check for i in range(27)})
    reports = [{"rank": rank, "stages": [stage] * 10} for rank in range(32)]
    assert len(PROBE["format_summary"](reports).encode()) <= 64 * 1024


def test_help_does_not_require_torch_or_initialize_devices():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=True)
    assert "--fsdp-size" in result.stdout
    assert "--loader" in result.stdout


def test_summary_preserves_last_rank_local_failure_before_repeated_gather_failures():
    check = PROBE["compare_values"]([2.0] * 119, [1.0] * 119, 4185)
    reports = []
    for rank in range(32):
        gathered = PROBE["compact_stage"]("export_unobserved.result", {"bias": {**check, "device": f"npu:{rank % 16}"}})
        local = PROBE["compact_stage"]("initial_offload.local", {"bias": check if rank == 31 else {"status": "equal"}})
        reports.append({"rank": rank, "stages": [gathered, local]})
    output = PROBE["format_summary"](reports)
    assert "initial_offload.local: {'equal': 31, 'different': 1} failing_ranks=[31]" in output
    assert output.index("rank=31 initial_offload.local:") < output.index("rank=0 export_unobserved.result:")
    assert output.count('"stage":"export_unobserved.result"') == 1
    assert "TRUNCATED" not in output


def test_summary_does_not_merge_distinct_failure_values_or_pass_missing_data():
    reports = []
    for rank in range(2):
        check = PROBE["compare_values"]([rank], [3])
        reports.append({"rank": rank, "stages": [PROBE["compact_stage"]("export", {"bias": check})]})
    output = PROBE["format_summary"](reports)
    assert "distinct bounded failure records=2" in output
    assert '"actual":0' in output and '"actual":1' in output
    assert "All collected checks passed" not in PROBE["format_summary"]([])
    assert "All collected checks passed" not in PROBE["format_summary"](
        [
            {
                "rank": 0,
                "stages": [PROBE["compact_stage"]("load", {"bias": {"status": "equal"}})],
                "gaps": ["guard absent"],
            }
        ]
    )


def test_offline_summary_cli_preserves_existing_json_without_torch(tmp_path):
    check = PROBE["compare_values"]([2.0], [1.0], 4185)
    path = tmp_path / "bias_probe_summary.json"
    original = json.dumps([{"rank": 31, "stages": [PROBE["compact_stage"]("initial_offload.local", {"bias": check})]}])
    path.write_text(original, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--summarize", str(tmp_path)], capture_output=True, text=True, check=True
    )
    assert "rank=31 initial_offload.local:" in result.stdout
    assert path.read_text(encoding="utf-8") == original
    assert "first_global" in (tmp_path / "bias_probe_summary_v2.txt").read_text(encoding="utf-8")


class DeferredCopyDevice:
    """A deferred host copy makes CPU repadding order observable without Torch."""

    def __init__(self):
        self.pending = []

    def synchronize(self):
        for copy in self.pending:
            copy()
        self.pending.clear()

    def empty_cache(self):
        pass


class LocalTensor:
    def __init__(self, values, device="cpu"):
        self.values = values
        self.device = SimpleNamespace(type=device)

    def size(self):
        return (len(self.values),)


class RepackingParam:
    def __init__(self):
        self.sharded_param = SimpleNamespace(_local_tensor=LocalTensor([7.0], device="npu"))
        self.padded_sharded_param_size = (2,)

    def reset_sharded_param(self):
        current = self.sharded_param._local_tensor
        # The new storage is a CPU copy, not the destination of the pending DMA.
        self.sharded_param._local_tensor = LocalTensor(current.values.copy())


class MovingModel:
    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.param = RepackingParam()

    def to(self, device, non_blocking):
        assert device == "cpu"
        source = self.param.sharded_param._local_tensor.values.copy()
        destination = LocalTensor([-999.0])

        def copy():
            destination.values[:] = source

        if non_blocking:
            self.accelerator.pending.append(copy)
        else:
            copy()
        self.param.sharded_param._local_tensor = destination
        self.param.reset_sharded_param()


@pytest.mark.parametrize("mode", PROBE["OFFLOAD_MODES"])
def test_variants_distinguish_waiting_before_repad_from_waiting_after_offload(mode):
    accelerator = DeferredCopyDevice()
    model = MovingModel(accelerator)

    def upstream(current):
        current.to("cpu", non_blocking=True)

    original = RepackingParam.reset_sharded_param
    result = PROBE["offload_variant"](model, mode, upstream, accelerator, (RepackingParam,))
    accelerator.synchronize()
    expected = 7.0 if mode in ("blocking", "sync_before_repad") else -999.0
    assert model.param.sharded_param._local_tensor.values == [expected]
    assert result["cpu_uneven_waits"] == int(mode == "sync_before_repad")
    assert RepackingParam.reset_sharded_param is original


def test_reset_guard_restores_method_after_an_exception():
    original = RepackingParam.reset_sharded_param
    with pytest.raises(RuntimeError), PROBE["wait_before_cpu_repad"]((RepackingParam,), lambda: None):
        assert RepackingParam.reset_sharded_param is not original
        raise RuntimeError("offload failed")
    assert RepackingParam.reset_sharded_param is original


@pytest.mark.parametrize("device,size", [("npu", 1), ("meta", 1), ("cpu", 2)])
def test_reset_guard_only_waits_for_uneven_cpu_shards(device, size):
    param = RepackingParam()
    param.sharded_param._local_tensor = LocalTensor([7.0] * size, device=device)
    calls = []
    with PROBE["wait_before_cpu_repad"]((RepackingParam,), lambda: calls.append("sync")) as stats:
        param.reset_sharded_param()
    assert calls == []
    assert stats["cpu_uneven_waits"] == 0
