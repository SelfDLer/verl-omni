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

"""Materialize a depth-only Qwen3-Omni checkpoint shared by actor and rollout.

The JSON is an overlay on the source model's complete config, not a replacement
with library defaults. Weight tensors are copied unchanged, shard by shard.
No model is instantiated and no accelerator memory is allocated.
"""

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

TEXT_LAYER = re.compile(r"^thinker\.model\.layers\.(\d+)\.")


def reduced_config(source, patch):
    """Change only text depth and disable inference-only audio output."""
    expected = {"enable_audio_output", "thinker_config"}
    if set(patch) != expected or patch["enable_audio_output"] is not False:
        raise ValueError("Debug config must only disable audio output and set Thinker text depth.")
    thinker_patch = patch["thinker_config"]
    if set(thinker_patch) != {"text_config"} or set(thinker_patch["text_config"]) != {"num_hidden_layers"}:
        raise ValueError("Only thinker_config.text_config.num_hidden_layers may change.")
    if source.get("model_type") != "qwen3_omni_moe":
        raise ValueError("Expected a Qwen3-Omni MoE source checkpoint.")
    count = thinker_patch["text_config"]["num_hidden_layers"]
    original = source["thinker_config"]["text_config"]["num_hidden_layers"]
    deepstack = len(source["thinker_config"]["vision_config"].get("deepstack_visual_indexes", []))
    if type(count) is not int or not max(1, deepstack) <= count <= original:
        raise ValueError(f"Text depth must be in [{max(1, deepstack)}, {original}] to retain all DeepStack injections.")
    text = source["thinker_config"]["text_config"]
    dense_layers = text.get("mlp_only_layers") or []
    stride = text.get("decoder_sparse_step", 1)

    def layer_kinds(depth):
        return {
            "moe" if text.get("num_experts", 0) > 0 and i not in dense_layers and (i + 1) % stride == 0 else "dense"
            for i in range(depth)
        }

    if layer_kinds(count) != layer_kinds(original):
        raise ValueError("The retained prefix must exercise the source model's dense/MoE layer types; increase depth.")
    result = deepcopy(source)
    result["enable_audio_output"] = False
    result["thinker_config"]["text_config"]["num_hidden_layers"] = count
    return result


def keep_weight(name, depth):
    """Keep the complete encoder/projector stack and a prefix of text layers."""
    if not name.startswith("thinker."):
        return False
    match = TEXT_LAYER.match(name)
    return match is None or int(match[1]) < depth


def prepare(source_dir, config_path, output_dir):
    """Write a separate checkpoint, or reuse a matching completed export."""
    source_dir, config_path, output_dir = (
        Path(p).expanduser().resolve() for p in (source_dir, config_path, output_dir)
    )
    if output_dir.is_relative_to(source_dir) or source_dir.is_relative_to(output_dir):
        raise ValueError("Source and debug checkpoint directories must be separate, non-nested paths.")
    config_bytes = (source_dir / "config.json").read_bytes()
    patch = json.loads(config_path.read_text(encoding="utf-8"))
    config = reduced_config(json.loads(config_bytes), patch)
    depth = config["thinker_config"]["text_config"]["num_hidden_layers"]
    index_file = source_dir / "model.safetensors.index.json"
    if index_file.exists():
        weight_map = json.loads(index_file.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    elif (source_dir / "model.safetensors").is_file():
        shards = ["model.safetensors"]
        with safe_open(source_dir / shards[0], framework="pt", device="cpu") as reader:
            weight_map = dict.fromkeys(reader.keys(), shards[0])
    else:
        raise FileNotFoundError("A local model.safetensors or model.safetensors.index.json checkpoint is required.")
    for name in shards:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(f"Unsupported shard path: {name}")
    assets = [
        p
        for p in sorted(source_dir.iterdir())
        if p.is_file()
        and p.suffix in {".json", ".txt", ".model", ".jinja", ".tiktoken"}
        and p.name not in {"config.json", "model.safetensors.index.json", "debug_manifest.json"}
    ]
    identity = {
        "source": str(source_dir),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "patch": patch,
        "files": {
            name: [p.stat().st_size, p.stat().st_mtime_ns]
            for name in [*shards, *(p.name for p in assets), *([index_file.name] if index_file.exists() else [])]
            for p in [source_dir / name]
        },
    }
    manifest_path = output_dir / "debug_manifest.json"
    if output_dir.exists():
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            intact = all(
                (output_dir / name).is_file() and (output_dir / name).stat().st_size == size
                for name, size in manifest.get("output_sizes", {}).items()
            )
            config_matches = (output_dir / "config.json").is_file() and json.loads(
                (output_dir / "config.json").read_text(encoding="utf-8")
            ) == config
            if manifest["identity"] == identity and intact and manifest.get("output_sizes") and config_matches:
                print(f"Reusing debug checkpoint: {output_dir}")
                return output_dir
        raise FileExistsError(f"Refusing to replace {output_dir}; use a new output directory.")

    selected = {name: shard for name, shard in weight_map.items() if keep_weight(name, depth)}
    layers = {int(match[1]) for name in selected if (match := TEXT_LAYER.match(name))}
    if layers != set(range(depth)):
        raise ValueError(f"Checkpoint text layer keys do not cover layers 0..{depth - 1}: {sorted(layers)}")
    if not any(name.startswith("thinker.visual.") for name in selected):
        raise ValueError("Source checkpoint has no Thinker vision weights.")
    if not any(name.startswith("thinker.audio_tower.") for name in selected):
        raise ValueError("Source checkpoint has no Thinker audio weights.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    # Incomplete exports are deliberately left under their unique staging name
    # on failure. They are never mistaken for a reusable checkpoint.
    total_size = 0
    for shard in sorted(set(selected.values())):
        with safe_open(source_dir / shard, framework="pt", device="cpu") as reader:
            tensors = {name: reader.get_tensor(name) for name, filename in selected.items() if filename == shard}
            total_size += sum(t.numel() * t.element_size() for t in tensors.values())
            save_file(tensors, staging / shard, metadata={"format": "pt"})
            del tensors
        print(f"Exported retained tensors from {shard}", flush=True)
    for asset in assets:
        shutil.copy2(asset, staging / asset.name)
    (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (staging / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": selected}, indent=2) + "\n", encoding="utf-8"
    )
    output_sizes = {p.name: p.stat().st_size for p in staging.iterdir()}
    manifest = {
        "identity": identity,
        "output_sizes": output_sizes,
        "retained_weight_bytes": total_size,
        "retained_tensors": len(selected),
        "text_layers": depth,
    }
    (staging / "debug_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    staging.rename(output_dir)
    print(f"Prepared {depth}-layer debug checkpoint: {output_dir} ({total_size / 2**30:.2f} GiB of weights)")
    return output_dir


def main():
    """Prepare a local checkpoint using the depth-only debug config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", default=str(Path(__file__).with_name("qwen3_omni_4layer.json")))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.source, args.config, args.output)


if __name__ == "__main__":
    main()
