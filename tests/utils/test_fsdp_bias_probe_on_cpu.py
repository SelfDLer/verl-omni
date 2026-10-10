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

import runpy
import subprocess
import sys
from pathlib import Path

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
