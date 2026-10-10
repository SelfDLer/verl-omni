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
"""Audit recorded vision fc1 biases against local checkpoint slices without Torch."""

import argparse
import json
import math
import re
import runpy
import struct
from collections import Counter, defaultdict
from pathlib import Path

DIAGNOSTICS = runpy.run_path(Path(__file__).with_name("video_trace_diagnostics.py"))
BIAS = re.compile(r"blocks\.(\d+)\.mlp\.linear_fc1\.bias\Z")
MAX_OUTPUT_BYTES = 64 * 1024


def sample_indices(snapshot):
    """Recover coordinates from the bounded linspace sampling scheme."""
    shape, count = snapshot.get("shape"), snapshot.get("sample_count")
    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or type(shape[0]) is not int
        or not 0 < shape[0] <= 1_000_000
        or type(count) is not int
        or not 0 < count <= min(shape[0], 4096)
        or snapshot.get("sample_scheme") != "linspace-flat-c-order-v1"
        or snapshot.get("digest_status") != "sample"
    ):
        raise ValueError("unsupported bias shape or sampling scheme")
    if count == 1:
        return [0]
    # np.linspace(..., dtype=int64) multiplies by a float64 step, then floors.
    step = (shape[0] - 1) / (count - 1)
    return [math.floor(i * step) for i in range(count - 1)] + [shape[0] - 1]


class CheckpointBias:
    """Read only small 1-D bias payloads using the safetensors header offsets."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.headers = {}
        self.cache = {}
        index = self.directory / "model.safetensors.index.json"
        self.weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"] if index.exists() else None

    def _header(self, filename):
        path = (self.directory / filename).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("checkpoint shard is outside checkpoint directory")
        if path not in self.headers:
            with path.open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                if not 2 <= size <= 64 * 1024 * 1024 or size + 8 > path.stat().st_size:
                    raise ValueError("invalid safetensors header length")
                self.headers[path] = (8 + size, json.loads(stream.read(size)))
        return path, *self.headers[path]

    def read(self, name):
        """Return one uniquely named visual fc1 bias; reject unsupported layouts."""
        if name in self.cache:
            return self.cache[name]
        suffix = "visual." + name

        def matches(key):
            return key == suffix or key.endswith("." + suffix)

        if self.weight_map is not None:
            candidates = [(key, filename) for key, filename in self.weight_map.items() if matches(key)]
        else:
            candidates = []
            for path in sorted(self.directory.glob("*.safetensors")):
                _, _, header = self._header(path.name)
                candidates.extend((key, path.name) for key in header if matches(key))
        if len(candidates) != 1:
            raise ValueError(f"expected one checkpoint bias for {name}; found {len(candidates)}")
        key, filename = candidates[0]
        path, base, header = self._header(filename)
        entry = header[key]
        shape, dtype = entry["shape"], entry["dtype"]
        formats = {"BF16": ("bfloat16", "H"), "F16": ("float16", "e"), "F32": ("float32", "f")}
        if dtype not in formats or len(shape) != 1 or type(shape[0]) is not int or not 0 < shape[0] <= 1_000_000:
            raise ValueError(f"unsupported checkpoint bias: shape={shape}, dtype={dtype}")
        dtype_name, fmt = formats[dtype]
        start, end = entry["data_offsets"]
        if not 0 <= start <= end or end - start != shape[0] * struct.calcsize(fmt) or base + end > path.stat().st_size:
            raise ValueError("invalid checkpoint bias data offsets")
        with path.open("rb") as stream:
            stream.seek(base + start)
            values = list(struct.unpack("<" + fmt * shape[0], stream.read(end - start)))
        if dtype == "BF16":
            values = [struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in values]
        self.cache[name] = {"key": key, "file": str(path), "dtype": dtype_name, "values": values}
        return self.cache[name]


def read_side(path, trace_id, sample_key):
    """Validate selected trace identity and retain only bias observations/runtime."""
    weights, runtimes, identities = defaultdict(list), {}, {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not trace_id or not sample_key or row.get("trace_id") != trace_id or row.get("sample_key") != sample_key:
                raise ValueError(f"{path.name}:{number}: trace/sample differs from comparison.json")
            event = row.get("event")
            if event not in ("worker.receive", "worker.vision.weights"):
                continue
            rank = row.get("worker_rank")
            identity = tuple(row.get(key) for key in ("host", "pid", "core_request_id"))
            if type(rank) is not int or None in identity:
                raise ValueError(f"{path.name}:{number}: missing worker identity")
            if rank in identities and identities[rank] != identity:
                raise ValueError(f"{path.name}:{number}: ambiguous worker/engine attempt for rank={rank}")
            identities[rank] = identity
            if event == "worker.receive":
                runtime = row.get("runtime", {})
                if rank in runtimes and runtimes[rank] != runtime:
                    raise ValueError(f"{path.name}:{number}: conflicting runtime for rank={rank}")
                runtimes[rank] = runtime
            elif row.get("kind") == "parameter":
                for name, value in row.get("values", {}).items():
                    if BIAS.fullmatch(name):
                        weights[rank, name].append(
                            {"value": value, "source": row.get("_source") or {"file": str(path), "line": number}}
                        )
    return weights, runtimes


def checkpoint_comparison(checkpoint, name, rank, snapshot, values, indices, runtime):
    """Compare using the ordinary contiguous column-parallel bias mapping only."""
    if checkpoint is None:
        return {"status": "unknown", "reason": "checkpoint not supplied"}
    try:
        parallel = runtime["parallel_config"]
        tp = int(parallel["tensor_parallel_size"])
        if int(parallel["pipeline_parallel_size"]) != 1 or int(parallel["data_parallel_size"]) != 1:
            raise ValueError("worker rank cannot be mapped to TP rank with this PP/DP layout")
        if not 0 <= rank < tp:
            raise ValueError("worker rank is outside TP group")
        saved = checkpoint.read(name)
        width = snapshot["shape"][0]
        if len(saved["values"]) != width * tp:
            raise ValueError(
                f"layout not inferred: checkpoint_length={len(saved['values'])}, local_length={width}, tp={tp}"
            )
        if snapshot["dtype"].removeprefix("torch.") != saved["dtype"]:
            raise ValueError(
                f"dtype mismatch: checkpoint={saved['dtype']}, runtime={snapshot['dtype']}; no cast assumed"
            )
        expected = [saved["values"][rank * width + i] for i in indices]
        bad = [i for i, (a, b) in enumerate(zip(values, expected, strict=True)) if a != b]
        return {
            "status": "sample_equal" if not bad else "different",
            "mapping": "contiguous_column_parallel_assumption",
            "checkpoint_key": saved["key"],
            "checkpoint_file": saved["file"],
            "global_start": rank * width,
            "different_count": len(bad),
            "slots": bad[:64],
            "omitted_slots": max(0, len(bad) - 64),
            "examples": [
                {
                    "local": indices[i],
                    "global": rank * width + indices[i],
                    "observed": values[i],
                    "checkpoint": expected[i],
                }
                for i in bad[:8]
            ],
        }
    except (ValueError, KeyError, TypeError, OSError, struct.error) as exc:
        return {"status": "unknown", "reason": str(exc)}


def audit(directory, checkpoint_dir=None):
    """Build an offline report from compare's selected single/multi JSONL files."""
    report = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    sample_key = report.get("sample", {}).get("sample_key")
    sides = {
        side: read_side(directory / f"{side}.jsonl", report.get(side, {}).get("trace_id"), sample_key)
        for side in ("single", "multi")
    }
    checkpoint = CheckpointBias(checkpoint_dir) if checkpoint_dir else None
    result = {"sample_key": sample_key, "checkpoint": str(checkpoint_dir) if checkpoint_dir else None, "biases": []}
    keys = sides["single"][0].keys() | sides["multi"][0].keys()
    for rank, name in sorted(keys, key=lambda key: (int(BIAS.fullmatch(key[1])[1]), key[0])):
        row = {"name": name, "rank": rank, "sources": {}, "checkpoint": {}}
        result["biases"].append(row)
        decoded, snapshots, positions = {}, {}, {}
        try:
            for side, (weights, runtimes) in sides.items():
                records = weights.get((rank, name), [])
                row["sources"][side] = [item["source"] for item in records]
                if len(records) != 1:
                    raise ValueError(f"{side}: missing or repeated bias observation ({len(records)})")
                snapshot = records[0]["value"]
                if not isinstance(snapshot, dict) or "digest_status" not in snapshot:
                    raise ValueError(f"{side}: no array snapshot: {snapshot}")
                snapshots[side] = snapshot
                positions[side] = sample_indices(snapshot)
                decoded[side] = DIAGNOSTICS["decode"](snapshot)
                row["checkpoint"][side] = checkpoint_comparison(
                    checkpoint, name, rank, snapshot, decoded[side], positions[side], runtimes.get(rank, {})
                )
            a, b = snapshots["single"], snapshots["multi"]
            if any(a.get(key) != b.get(key) for key in ("shape", "dtype", "sample_count", "sample_scheme")):
                raise ValueError("single/multi bias samples have different layouts")
            x, y, indices = decoded["single"], decoded["multi"], positions["single"]
            bad = [i for i, (v, w) in enumerate(zip(x, y, strict=True)) if v != w]
            row.update(
                status="different" if bad else "sample_equal",
                shape=a["shape"],
                dtype=a["dtype"],
                sample_count=len(x),
                different_count=len(bad),
                slots=bad[:64],
                local_indices=[indices[i] for i in bad[:64]],
                omitted_slots=max(0, len(bad) - 64),
                sampled_suffix=bool(bad) and bad == list(range(bad[0], len(x))),
                examples=[{"local": indices[i], "single": x[i], "multi": y[i]} for i in bad[:8]],
            )
        except (ValueError, KeyError, TypeError, struct.error) as exc:
            row.update(status="unknown", reason=str(exc))
    return result


def summary(report):
    """Render a bounded uploadable report with changed coordinates and references."""
    rows = report["biases"]
    lines = [
        f"8039 vision fc1 bias audit v1 sample={report['sample_key']}",
        f"checkpoint={report['checkpoint']}",
        "Checkpoint mapping assumes ordinary contiguous column parallelism with PP=DP=1.",
        "Matches refer only to sampled numeric values, not complete tensors or bitwise equality.",
        "A checkpoint mismatch can reflect actor updates; it does not identify the faulty lifecycle step.",
        "sampled_suffix refers only to sampled positions, not all tensor elements.",
        f"cross_run_counts={dict(Counter(row['status'] for row in rows))}",
    ]
    for side in ("single", "multi"):
        counts = Counter(row["checkpoint"].get(side, {}).get("status", "unknown") for row in rows)
        lines.append(f"{side}_vs_checkpoint={dict(counts)}")
    patterns = defaultdict(list)
    for row in rows:
        if row["status"] == "different":
            patterns[row["rank"], tuple(row["shape"]), tuple(row["local_indices"]), row["omitted_slots"]].append(
                row["name"]
            )
    for (rank, shape, indices, omitted), names in patterns.items():
        lines.append(f"pattern rank={rank} shape={shape} layers={len(names)} local_indices={indices} omitted={omitted}")
    interesting = [
        row
        for row in rows
        if row["status"] != "sample_equal" or any(x.get("status") != "sample_equal" for x in row["checkpoint"].values())
    ]
    interesting.sort(key=lambda row: row["status"] != "different")
    for row in interesting[:128]:
        lines.append(
            f"{row['name']} rank={row['rank']}: {row['status']} shape={row.get('shape')} "
            f"different={row.get('different_count')}/{row.get('sample_count')} "
            f"sampled_suffix={row.get('sampled_suffix')}"
        )
        if row.get("reason"):
            lines.append("  reason=" + row["reason"][:512])
        elif row.get("different_count"):
            lines.append(f"  slots={row['slots']} local_indices={row['local_indices']} omitted={row['omitted_slots']}")
            lines.append("  values=" + json.dumps(row["examples"], separators=(",", ":")))
        for side, check in row["checkpoint"].items():
            lines.append(
                f"  {side}_vs_checkpoint: {check['status']} different={check.get('different_count')} "
                f"global_start={check.get('global_start')}"
            )
            if check.get("reason"):
                lines.append("    reason=" + check["reason"][:512])
            if check.get("examples"):
                lines.append("    values=" + json.dumps(check["examples"], separators=(",", ":")))
    if not rows:
        lines.append("NO_BIAS_OBSERVATIONS: absence is not evidence of equality.")
    if len(interesting) > 128:
        lines.append(f"omitted_bias_details={len(interesting) - 128}; see vision_bias_audit.json")
    raw = ("\n".join(lines) + "\n").encode()
    if len(raw) > MAX_OUTPUT_BYTES:
        marker = b"\nTRUNCATED: see vision_bias_audit.json; omitted data is not evidence of equality.\n"
        raw = raw[: MAX_OUTPUT_BYTES - len(marker)] + marker
    return raw.decode("utf-8", errors="ignore")


def main():
    """Audit already captured bias values without starting Ray or model inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Compare output directory containing comparison.json and selected JSONL"
    )
    parser.add_argument("--checkpoint", type=Path, help="Optional local safetensors checkpoint directory")
    args = parser.parse_args()
    try:
        result = audit(args.directory, args.checkpoint)
    except (ValueError, OSError, KeyError, TypeError, struct.error) as exc:
        parser.error(str(exc))
    (args.directory / "vision_bias_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = summary(result)
    (args.directory / "vision_bias_audit.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
