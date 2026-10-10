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

import base64
import hashlib
import json
import runpy
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "special_e2e/audit_8039_vision_bias.py"
AUDIT = runpy.run_path(SCRIPT)


def snapshot(values, width=1076, dtype="bfloat16"):
    raw = b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", value))[0] >> 16) for value in values)
    return {
        "shape": [width],
        "dtype": dtype,
        "digest_status": "sample",
        "sample_count": len(values),
        "sample_scheme": "linspace-flat-c-order-v1",
        "sample_byteorder": "little",
        "sample_data_b64": base64.b64encode(raw).decode(),
        "sample_sha256": hashlib.sha256(raw).hexdigest(),
    }


def write_checkpoint(directory, biases, indexed=True, dtype="BF16"):
    directory.mkdir(exist_ok=True)
    header, payload = {}, bytearray()
    for name, values in biases.items():
        start = len(payload)
        for value in values:
            if dtype == "BF16":
                payload.extend(struct.pack("<H", struct.unpack("<I", struct.pack("<f", value))[0] >> 16))
            else:
                payload.extend(struct.pack("<" + {"F16": "e", "F32": "f"}[dtype], value))
        header[name] = {"dtype": dtype, "shape": [len(values)], "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header).encode()
    path = directory / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    if indexed:
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: path.name for name in biases}}), encoding="utf-8"
        )


def fixture(directory, blocks=2):
    rows, biases = {"single": [], "multi": []}, {}
    indices = np.linspace(0, 1075, 64, dtype=np.int64)
    for rank in range(4):
        for side in rows:
            common = {
                "host": side,
                "pid": rank + 10,
                "core_request_id": side,
                "worker_rank": rank,
                "trace_id": side,
                "sample_key": "6806999702_8",
            }
            rows[side].append(
                common
                | {
                    "event": "worker.receive",
                    "runtime": {
                        "parallel_config": {
                            "tensor_parallel_size": "4",
                            "pipeline_parallel_size": "1",
                            "data_parallel_size": "1",
                        }
                    },
                }
            )
            for block in range(blocks):
                name = f"blocks.{block}.mlp.linear_fc1.bias"
                full = [float((i % 128) / 16 + block / 8) for i in range(4304)]
                biases["thinker.visual." + name] = full
                values = [full[rank * 1076 + int(i)] for i in indices]
                if side == "multi" and rank == 3:
                    values[-7:] = [value + 2 for value in values[-7:]]
                rows[side].append(
                    common | {"event": "worker.vision.weights", "kind": "parameter", "values": {name: snapshot(values)}}
                )
    (directory / "comparison.json").write_text(
        json.dumps(
            {"sample": {"sample_key": "6806999702_8"}, "single": {"trace_id": "single"}, "multi": {"trace_id": "multi"}}
        ),
        encoding="utf-8",
    )
    for side, records in rows.items():
        write_rows(directory, side, records)
    checkpoint = directory / "checkpoint"
    write_checkpoint(checkpoint, biases)
    return checkpoint, rows, biases


def write_rows(directory, side, records):
    (directory / f"{side}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def test_all_27_biases_have_tail_coordinates_and_checkpoint_attribution(tmp_path):
    checkpoint, _, _ = fixture(tmp_path, blocks=27)
    result = AUDIT["audit"](tmp_path, checkpoint)
    changed = [row for row in result["biases"] if row["status"] == "different"]
    assert len(result["biases"]) == 108 and len(changed) == 27
    for row in changed:
        assert row["rank"] == 3 and row["different_count"] == 7
        assert row["slots"] == list(range(57, 64)) and row["sampled_suffix"]
        assert row["local_indices"] == np.linspace(0, 1075, 64, dtype=np.int64)[-7:].tolist()
        assert row["checkpoint"]["single"]["status"] == "sample_equal"
        assert row["checkpoint"]["multi"]["different_count"] == 7
        assert row["checkpoint"]["multi"]["global_start"] == 3228
        assert row["checkpoint"]["multi"]["examples"][-1]["global"] == 4303
    output = AUDIT["summary"](result)
    for block in range(27):
        assert f"blocks.{block}.mlp.linear_fc1.bias rank=3: different" in output
    assert "layers=27" in output and "TRUNCATED" not in output
    assert len(output.encode()) <= 64 * 1024
    run = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), "--checkpoint", str(checkpoint)], text=True, capture_output=True
    )
    assert run.returncode == 0, run.stderr
    assert (tmp_path / "vision_bias_audit.txt").read_text(encoding="utf-8") == output


@pytest.mark.parametrize("width", [1, 17, 1076, 1088, 1152, 4304, 1_000_000])
def test_indices_match_numpy_sampler(width):
    for count in (1, min(64, width), min(1024, width)):
        value = snapshot([0.0] * count, width=width)
        assert AUDIT["sample_indices"](value) == np.linspace(0, width - 1, count, dtype=np.int64).tolist()


def test_checkpoint_is_optional_and_cross_run_differences_remain_available(tmp_path):
    fixture(tmp_path)
    result = AUDIT["audit"](tmp_path)
    row = next(row for row in result["biases"] if row["status"] == "different")
    assert row["different_count"] == 7
    assert row["checkpoint"]["multi"] == {"status": "unknown", "reason": "checkpoint not supplied"}


def test_single_is_not_assumed_to_be_the_checkpoint_reference(tmp_path):
    checkpoint, rows, _ = fixture(tmp_path)
    rows["single"][-1]["values"], rows["multi"][-1]["values"] = (
        rows["multi"][-1]["values"],
        rows["single"][-1]["values"],
    )
    for side, records in rows.items():
        write_rows(tmp_path, side, records)
    row = AUDIT["audit"](tmp_path, checkpoint)["biases"][-1]
    assert row["checkpoint"]["single"]["status"] == "different"
    assert row["checkpoint"]["multi"]["status"] == "sample_equal"


@pytest.mark.parametrize("change", ["repeated", "missing", "digest", "unsupported", "different_shape"])
def test_insufficient_observations_do_not_become_equal(tmp_path, change):
    checkpoint, rows, _ = fixture(tmp_path)
    target = rows["multi"][-1]
    value = next(iter(target["values"].values()))
    if change == "repeated":
        rows["multi"].append(target)
    elif change == "missing":
        rows["multi"].pop()
    elif change == "digest":
        value["sample_sha256"] = "bad"
    elif change == "unsupported":
        target["values"] = {name: {"unsupported": "parameter"} for name in target["values"]}
    else:
        value["shape"] = [1088]
    write_rows(tmp_path, "multi", rows["multi"])
    result = AUDIT["audit"](tmp_path, checkpoint)
    assert result["biases"][-1]["status"] == "unknown"


@pytest.mark.parametrize("field", ["sample_key", "trace_id", "core_request_id", "pid"])
def test_mixed_requests_and_workers_are_rejected(tmp_path, field):
    checkpoint, rows, _ = fixture(tmp_path)
    rows["multi"][-1][field] = "another"
    write_rows(tmp_path, "multi", rows["multi"])
    with pytest.raises(ValueError):
        AUDIT["audit"](tmp_path, checkpoint)


@pytest.mark.parametrize("change", ["padding", "dtype", "ambiguous_name", "missing_runtime"])
def test_unknown_checkpoint_layout_does_not_hide_cross_run_difference(tmp_path, change):
    checkpoint, rows, biases = fixture(tmp_path)
    if change == "padding":
        write_checkpoint(checkpoint, {name: value[:-1] for name, value in biases.items()})
    elif change == "dtype":
        write_checkpoint(checkpoint, biases, dtype="F32")
    elif change == "ambiguous_name":
        write_checkpoint(checkpoint, biases | {"other." + name: value for name, value in biases.items()})
    else:
        rows["multi"] = [row for row in rows["multi"] if row["event"] != "worker.receive"]
        write_rows(tmp_path, "multi", rows["multi"])
    result = AUDIT["audit"](tmp_path, checkpoint)
    row = result["biases"][-1]
    assert row["status"] == "different" and row["checkpoint"]["multi"]["status"] == "unknown"


@pytest.mark.parametrize("dtype", ["BF16", "F16", "F32"])
def test_unindexed_safetensors_reader_is_independent_of_torch(tmp_path, dtype):
    name = "blocks.0.mlp.linear_fc1.bias"
    write_checkpoint(tmp_path, {"thinker.visual." + name: [-2.0, 0.0, 0.125, 3.5]}, indexed=False, dtype=dtype)
    assert AUDIT["CheckpointBias"](tmp_path).read(name)["values"] == [-2.0, 0.0, 0.125, 3.5]


def test_truncated_checkpoint_payload_is_reported_unknown(tmp_path):
    checkpoint, _, _ = fixture(tmp_path)
    path = checkpoint / "model.safetensors"
    path.write_bytes(path.read_bytes()[:-1])
    row = AUDIT["audit"](tmp_path, checkpoint)["biases"][-1]
    assert row["status"] == "different"
    assert row["checkpoint"]["multi"]["reason"] == "invalid checkpoint bias data offsets"
