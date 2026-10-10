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

import copy
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "special_e2e/summarize_8039_worker.py"
SUMMARY = runpy.run_path(SCRIPT)
COMPARE = runpy.run_path(SCRIPT.with_name("compare_8039_request.py"))


def snapshot(content):
    return {
        "shape": [100, 16],
        "dtype": "bfloat16",
        "digest_status": "sample",
        "sample_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "sample_count": 256,
        "sample_scheme": "linspace-flat-c-order-v1",
    }


def fixture(directory):
    sides = {}
    for side in ("single", "multi"):
        rows = []
        for rank in (0, 1):
            common = {
                "trace_id": side,
                "sample_key": "6806999702_8",
                "host": "node",
                "pid": rank + 10,
                "worker_rank": rank,
                "core_request_id": side + "-core",
                "_source": {"file": side + ".jsonl", "line": len(rows) + 1},
            }
            rows.append(
                common
                | {
                    "event": "worker.receive",
                    "multimodal_features": [{"feature_index": 0, "modality": "audio", "data": "x" * 600_000}],
                    "runtime": {"model_config": {"seed": "42"}},
                }
            )
            for event in ("worker.encoder.write", "worker.encoder.read"):
                rows.append(
                    common
                    | {
                        "event": event,
                        "feature_index": 0,
                        "modality": "audio",
                        "identifier": side + "-audio",
                        "embedding": snapshot(side),
                    }
                )
        sides[side] = {"records": rows, "routes": [{}]}
        write_rows(directory, side, rows)
    report = {
        "sample": {"sample_key": "6806999702_8"},
        "single": {"trace_id": "single"},
        "multi": {"trace_id": "multi"},
        "stages": COMPARE["_worker_stages"](sides["single"], sides["multi"], None),
    }
    (directory / "comparison.json").write_text(json.dumps(report), encoding="utf-8")
    return sides


def write_rows(directory, side, rows):
    (directory / f"{side}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_large_inputs_produce_small_summary_with_actual_modality_and_cache_consistency(tmp_path):
    fixture(tmp_path)
    assert (tmp_path / "single.jsonl").stat().st_size > 1_000_000
    output = SUMMARY["summarize"](tmp_path)
    assert len(output.encode()) < 10_000
    assert "modality=['audio']" in output
    assert "write_vs_read=sample_equal" in output
    assert '"D1":[0,1]' in output and '"D2":[0,1]' in output
    assert "$.sample_sha256" in output
    assert "x" * 100 not in output
    result = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "worker_summary.txt").read_text(encoding="utf-8") == output


def test_changing_cache_reads_are_not_collapsed_to_equality(tmp_path):
    sides = fixture(tmp_path)
    rows = sides["single"]["records"]
    changed = copy.deepcopy(rows[2])
    changed["embedding"] = snapshot("changed")
    rows.append(changed)
    write_rows(tmp_path, "single", rows)
    assert "changing_observations" in SUMMARY["summarize"](tmp_path)


@pytest.mark.parametrize("field,value", [("dtype", "float32"), ("shape", [200, 16]), ("sample_sha256", "other")])
def test_cache_read_mismatch_is_reported(tmp_path, field, value):
    sides = fixture(tmp_path)
    sides["single"]["records"][2]["embedding"][field] = value
    write_rows(tmp_path, "single", sides["single"]["records"])
    assert "write_vs_read=different:" in SUMMARY["summarize"](tmp_path)


@pytest.mark.parametrize("change", ["missing_digest", "scheme", "identifier", "missing_read"])
def test_incomplete_or_incompatible_cache_observations_remain_unknown(tmp_path, change):
    sides = fixture(tmp_path)
    rows = sides["single"]["records"]
    if change == "missing_digest":
        rows[2]["embedding"].pop("sample_sha256")
    elif change == "scheme":
        rows[2]["embedding"]["sample_scheme"] = "different-scheme"
    elif change == "identifier":
        rows[2]["identifier"] = "another-cache-entry"
    else:
        del rows[2]
    write_rows(tmp_path, "single", rows)
    assert "write_vs_read=unknown:" in SUMMARY["summarize"](tmp_path)


@pytest.mark.parametrize("change", ["trace_id", "sample_key", "pid"])
def test_mixed_requests_or_workers_fail_instead_of_merging(tmp_path, change):
    sides = fixture(tmp_path)
    sides["single"]["records"][0][change] = "other"
    write_rows(tmp_path, "single", sides["single"]["records"])
    with pytest.raises(ValueError):
        SUMMARY["summarize"](tmp_path)


def test_output_limit_is_explicit_and_below_upload_limit(tmp_path):
    sides = fixture(tmp_path)
    sides["single"]["records"][0]["runtime"] = {"large": "x" * 100_000}
    write_rows(tmp_path, "single", sides["single"]["records"])
    output = SUMMARY["summarize"](tmp_path)
    assert len(output.encode()) <= 64 * 1024
    assert "TRUNCATED" in output


def test_weight_differences_are_shown_before_bounded_unknown_examples(tmp_path):
    fixture(tmp_path)
    report_path = tmp_path / "comparison.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stages = []
    unsupported = {"unsupported": "vllm.model_executor.parameter.ModelWeightParameter"}
    for index in range(464 + 27):
        missing = index < 464
        checkpoint = f"weight.parameter.{'missing' if missing else 'changed'}.{index}"
        result = COMPARE["_trees"](
            unsupported if missing else snapshot("single"),
            unsupported if missing else snapshot("multi"),
            require_arrays=True,
        )
        stages.append({"checkpoint": checkpoint, "stage": checkpoint, "components": {"value": result}})
    report["vision_stages"] = stages
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output = SUMMARY["summarize"](tmp_path)
    assert "weight comparisons different: 27" in output
    assert "weight comparisons unknown: 464" in output
    for row in stages[464:]:
        assert f"{row['stage']}: different" in output
    assert output.index("weight.parameter.changed.") < output.index("weight.parameter.missing.")
    assert "omitted_weight_unknown_details=456" in output
    assert "unsupported: vllm.model_executor.parameter.ModelWeightParameter" in output
    assert len(output.encode()) < 64 * 1024


@pytest.mark.parametrize("event", sorted(SUMMARY["WORKER_DATA_EVENTS"]))
def test_data_events_still_require_identity_with_actionable_errors(tmp_path, event):
    sides = fixture(tmp_path)
    row = sides["single"]["records"][0]
    row["event"] = event
    del row["worker_rank"]
    write_rows(tmp_path, "single", sides["single"]["records"])
    with pytest.raises(ValueError, match=f"single.jsonl:1: {event}: missing worker identity fields: worker_rank"):
        SUMMARY["summarize"](tmp_path)


def test_conflicting_requests_report_both_identities_and_lines(tmp_path):
    sides = fixture(tmp_path)
    sides["single"]["records"][1]["core_request_id"] = "another-request"
    write_rows(tmp_path, "single", sides["single"]["records"])
    with pytest.raises(ValueError, match="conflicting worker identity for rank=0") as error:
        SUMMARY["summarize"](tmp_path)
    assert "line 1=" in str(error.value) and "line 2=" in str(error.value)
    assert "single-core" in str(error.value) and "another-request" in str(error.value)


def test_scope_records_are_counted_without_inventing_worker_identity(tmp_path):
    sides = fixture(tmp_path)
    data = sides["single"]["records"]
    for event in (
        "worker.observation.begin",
        "worker.observation.end",
        "worker.observation.error",
        "worker.limit.begin",
    ):
        data.append({"trace_id": "single", "sample_key": "6806999702_8", "event": event})
    write_rows(tmp_path, "single", data)
    output = SUMMARY["summarize"](tmp_path)
    assert '"worker.observation.begin":1' in output
    assert '"worker.observation.error":1' in output
    assert not any(line.startswith("rank=None") for line in output.splitlines())
    assert "write_vs_read=sample_equal" in output
