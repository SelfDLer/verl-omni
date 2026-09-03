#!/usr/bin/env python3
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

"""Create a prefix-layer Qwen3-Omni checkpoint for parity diagnostics.

The reduced checkpoint keeps layers ``[0, num_layers)`` from the Thinker text
decoder. Audio, vision, Talker, code2wav, embeddings, and output-head weights
are left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "config.json"
INDEX_FILENAME = "model.safetensors.index.json"
UNSHARDED_WEIGHTS_FILENAME = "model.safetensors"
MANIFEST_FILENAME = "layer_reduction_manifest.json"
LAYER_KEY_PATTERN = r"^thinker\.model\.layers\.(\d+)\."
LAYER_KEY_RE = re.compile(LAYER_KEY_PATTERN)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, sort_keys=True)
        file.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _thinker_text_config(config: dict[str, Any]) -> dict[str, Any]:
    try:
        text_config = config["thinker_config"]["text_config"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "config.json does not contain thinker_config.text_config; is this a Qwen3-Omni checkpoint?"
        ) from error
    if not isinstance(text_config, dict):
        raise ValueError("thinker_config.text_config must be a JSON object")
    return text_config


def _original_layer_count(config: dict[str, Any]) -> int:
    text_config = _thinker_text_config(config)
    value = text_config.get("num_hidden_layers")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("thinker_config.text_config.num_hidden_layers must be a positive integer")
    return value


def _set_layer_count(config: dict[str, Any], num_layers: int) -> None:
    text_config = _thinker_text_config(config)
    original_layers = _original_layer_count(config)
    text_config["num_hidden_layers"] = num_layers

    # Some Transformers configs materialize a per-layer attention schedule.
    # Qwen3-Omni currently does not, but truncate it when present so the
    # reduced config remains internally consistent across versions.
    layer_types = text_config.get("layer_types")
    if layer_types is not None:
        if not isinstance(layer_types, list) or len(layer_types) != original_layers:
            raise ValueError("thinker_config.text_config.layer_types must contain one entry per original hidden layer")
        text_config["layer_types"] = layer_types[:num_layers]


def _layer_index(weight_name: str) -> int | None:
    match = LAYER_KEY_RE.match(weight_name)
    return int(match.group(1)) if match else None


def _validate_layer_keys(weight_names: list[str], original_layers: int) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for weight_name in weight_names:
        layer_index = _layer_index(weight_name)
        if layer_index is not None:
            counts[layer_index] += 1

    expected = set(range(original_layers))
    discovered = set(counts)
    if discovered != expected:
        missing = sorted(expected - discovered)
        unexpected = sorted(discovered - expected)
        raise ValueError(
            "Thinker decoder layer keys do not match config.json: "
            f"missing={missing}, unexpected={unexpected}, pattern={LAYER_KEY_PATTERN!r}"
        )
    return dict(counts)


def _keep_weight(weight_name: str, num_layers: int) -> bool:
    layer_index = _layer_index(weight_name)
    return layer_index is None or layer_index < num_layers


def _discover_weights(source: Path) -> tuple[dict[str, Any] | None, dict[str, str]]:
    index_path = source / INDEX_FILENAME
    unsharded_path = source / UNSHARDED_WEIGHTS_FILENAME
    if index_path.is_file():
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index_path} has no non-empty weight_map")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()):
            raise ValueError(f"{index_path} weight_map must map strings to strings")
        return index, weight_map
    if unsharded_path.is_file():
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError("Install safetensors to inspect checkpoint weights") from error
        with safe_open(unsharded_path, framework="pt", device="cpu") as weights:
            return None, {key: UNSHARDED_WEIGHTS_FILENAME for key in weights.keys()}
    raise FileNotFoundError(f"Expected {INDEX_FILENAME} or {UNSHARDED_WEIGHTS_FILENAME} under {source}")


def _copy_or_link(source: Path, destination: Path, link: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(source, destination)
            return "linked"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copied"


def _copy_auxiliary_files(
    source: Path,
    destination: Path,
    excluded_relative_paths: set[Path],
    link_unchanged: bool,
) -> dict[str, int]:
    counts = {"linked": 0, "copied": 0}
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative_path = source_file.relative_to(source)
        if relative_path in excluded_relative_paths:
            continue
        action = _copy_or_link(source_file, destination / relative_path, link_unchanged)
        counts[action] += 1
    return counts


def _tensor_size(tensor: Any) -> int:
    return tensor.numel() * tensor.element_size()


def _process_weight_shard(
    source_path: Path,
    destination_path: Path,
    all_keys: list[str],
    kept_keys: list[str],
    link_unchanged: bool,
) -> tuple[str, int, int]:
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("Install torch and safetensors to reduce checkpoint weights") from error

    if kept_keys == all_keys:
        action = _copy_or_link(source_path, destination_path, link_unchanged)
        total_size = 0
        total_parameters = 0
        with safe_open(source_path, framework="pt", device="cpu") as weights:
            for key in kept_keys:
                tensor = weights.get_tensor(key)
                total_size += _tensor_size(tensor)
                total_parameters += tensor.numel()
        return action, total_size, total_parameters

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with safe_open(source_path, framework="pt", device="cpu") as weights:
        metadata = weights.metadata()
        tensors = {key: weights.get_tensor(key) for key in kept_keys}
        total_size = sum(_tensor_size(tensor) for tensor in tensors.values())
        total_parameters = sum(tensor.numel() for tensor in tensors.values())
        save_file(tensors, destination_path, metadata=metadata)
    return "rewritten", total_size, total_parameters


def _format_gib(value: int) -> str:
    return f"{value / (1024**3):.2f} GiB"


def _build_plan(
    source: Path,
    num_layers: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, str], int, dict[int, int]]:
    config_path = source / CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}")
    config = _read_json(config_path)
    original_layers = _original_layer_count(config)
    if not 1 <= num_layers <= original_layers:
        raise ValueError(f"--num-layers must be between 1 and {original_layers}, got {num_layers}")
    if num_layers == original_layers:
        raise ValueError(f"--num-layers equals the original layer count ({original_layers}); nothing would be reduced")

    index, weight_map = _discover_weights(source)
    layer_key_counts = _validate_layer_keys(list(weight_map), original_layers)
    return config, index, weight_map, original_layers, layer_key_counts


def reduce_checkpoint(
    source: Path,
    output: Path,
    num_layers: int,
    *,
    link_unchanged: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build a reduced checkpoint and return its manifest."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Source checkpoint is not a directory: {source}")
    if output == source or source in output.parents:
        raise ValueError("Output must not be the source directory or a child of it")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    config, index, weight_map, original_layers, layer_key_counts = _build_plan(source, num_layers)
    kept_weight_map = {
        weight_name: shard_name
        for weight_name, shard_name in weight_map.items()
        if _keep_weight(weight_name, num_layers)
    }
    source_shards: dict[str, list[str]] = defaultdict(list)
    kept_shards: dict[str, list[str]] = defaultdict(list)
    for weight_name, shard_name in weight_map.items():
        source_shards[shard_name].append(weight_name)
    for weight_name, shard_name in kept_weight_map.items():
        kept_shards[shard_name].append(weight_name)

    manifest: dict[str, Any] = {
        "format_version": 1,
        "source_checkpoint": str(source),
        "source_config_sha256": _sha256(source / CONFIG_FILENAME),
        "layer_key_pattern": LAYER_KEY_PATTERN,
        "original_num_hidden_layers": original_layers,
        "num_hidden_layers": num_layers,
        "kept_layer_indices": list(range(num_layers)),
        "removed_layer_indices": list(range(num_layers, original_layers)),
        "layer_weight_counts": {str(key): value for key, value in sorted(layer_key_counts.items())},
        "source_weight_count": len(weight_map),
        "kept_weight_count": len(kept_weight_map),
        "removed_weight_count": len(weight_map) - len(kept_weight_map),
        "source_shard_count": len(source_shards),
        "output_shard_count": len(kept_shards),
        "link_unchanged_requested": link_unchanged,
    }
    if dry_run:
        return manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        excluded_paths = {Path(CONFIG_FILENAME), Path(MANIFEST_FILENAME)}
        if index is not None:
            excluded_paths.add(Path(INDEX_FILENAME))
        excluded_paths.update(Path(shard_name) for shard_name in source_shards)
        auxiliary_counts = _copy_auxiliary_files(
            source,
            temporary_output,
            excluded_paths,
            link_unchanged,
        )

        shard_actions: dict[str, str] = {}
        output_tensor_bytes = 0
        output_parameter_count = 0
        for shard_number, (shard_name, kept_keys) in enumerate(sorted(kept_shards.items()), start=1):
            source_path = source / shard_name
            if not source_path.is_file():
                raise FileNotFoundError(f"Weight index references missing shard: {source_path}")
            print(f"[{shard_number}/{len(kept_shards)}] processing {shard_name}", flush=True)
            action, tensor_bytes, parameter_count = _process_weight_shard(
                source_path,
                temporary_output / shard_name,
                source_shards[shard_name],
                kept_keys,
                link_unchanged,
            )
            shard_actions[shard_name] = action
            output_tensor_bytes += tensor_bytes
            output_parameter_count += parameter_count

        _set_layer_count(config, num_layers)
        _write_json(temporary_output / CONFIG_FILENAME, config)
        if index is not None:
            output_index = dict(index)
            output_index["weight_map"] = kept_weight_map
            metadata = dict(output_index.get("metadata") or {})
            metadata["total_size"] = output_tensor_bytes
            output_index["metadata"] = metadata
            _write_json(temporary_output / INDEX_FILENAME, output_index)

        manifest.update(
            {
                "output_tensor_bytes": output_tensor_bytes,
                "output_tensor_size": _format_gib(output_tensor_bytes),
                "output_parameter_count": output_parameter_count,
                "shard_actions": shard_actions,
                "auxiliary_file_actions": auxiliary_counts,
            }
        )
        _write_json(temporary_output / MANIFEST_FILENAME, manifest)
        temporary_output.rename(output)
    except BaseException:
        shutil.rmtree(temporary_output, ignore_errors=True)
        raise
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Keep the first N Qwen3-Omni Thinker text layers and produce a self-contained safetensors checkpoint."
        )
    )
    parser.add_argument("--source", type=Path, required=True, help="Source Qwen3-Omni checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="New reduced checkpoint directory")
    parser.add_argument("--num-layers", type=int, required=True, help="Number of Thinker text layers to keep")
    parser.add_argument(
        "--copy-unchanged",
        action="store_true",
        help="Copy unchanged files instead of hard-linking them when possible",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/index and print the reduction plan without writing files",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        manifest = reduce_checkpoint(
            args.source,
            args.output,
            args.num_layers,
            link_unchanged=not args.copy_unchanged,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, FileExistsError, NotADirectoryError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(
            f"Reduced checkpoint created at {args.output.resolve()} "
            f"({manifest['num_hidden_layers']}/{manifest['original_num_hidden_layers']} Thinker layers, "
            f"{manifest['output_tensor_size']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
