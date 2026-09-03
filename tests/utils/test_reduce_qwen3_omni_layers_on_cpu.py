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
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from npu_test.reduce_qwen3_omni_layers import reduce_checkpoint


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _build_checkpoint(path: Path) -> dict[str, str]:
    path.mkdir()
    _write_json(
        path / "config.json",
        {
            "model_type": "qwen3_omni_moe",
            "thinker_config": {
                "text_config": {
                    "num_hidden_layers": 3,
                    "layer_types": ["full_attention", "full_attention", "full_attention"],
                }
            },
        },
    )
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")

    shards = {
        "model-00001-of-00003.safetensors": {
            "thinker.model.embed_tokens.weight": torch.arange(4, dtype=torch.float32),
        },
        "model-00002-of-00003.safetensors": {
            "thinker.model.layers.0.self_attn.q_proj.weight": torch.full((2,), 10.0),
            "thinker.model.layers.1.self_attn.q_proj.weight": torch.full((2,), 11.0),
        },
        "model-00003-of-00003.safetensors": {
            "thinker.model.layers.2.self_attn.q_proj.weight": torch.full((2,), 12.0),
            "thinker.audio_tower.layers.2.fc1.weight": torch.full((2,), 20.0),
            "talker.model.layers.2.self_attn.q_proj.weight": torch.full((2,), 30.0),
        },
    }
    weight_map = {}
    for shard_name, tensors in shards.items():
        save_file(tensors, path / shard_name, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(tensors, shard_name))
    _write_json(
        path / "model.safetensors.index.json",
        {"metadata": {"total_size": 123456}, "weight_map": weight_map},
    )
    return weight_map


def _output_tensors(path: Path) -> dict[str, torch.Tensor]:
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    tensors = {}
    for shard_name in set(index["weight_map"].values()):
        with safe_open(path / shard_name, framework="pt", device="cpu") as shard:
            tensors.update({key: shard.get_tensor(key) for key in shard.keys()})
    return tensors


def test_reduce_checkpoint_keeps_only_thinker_layer_prefix(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "reduced"
    source_weight_map = _build_checkpoint(source)

    manifest = reduce_checkpoint(source, output, 1, link_unchanged=False)

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["thinker_config"]["text_config"]["num_hidden_layers"] == 1
    assert config["thinker_config"]["text_config"]["layer_types"] == ["full_attention"]

    tensors = _output_tensors(output)
    assert "thinker.model.layers.0.self_attn.q_proj.weight" in tensors
    assert "thinker.model.layers.1.self_attn.q_proj.weight" not in tensors
    assert "thinker.model.layers.2.self_attn.q_proj.weight" not in tensors
    assert "thinker.audio_tower.layers.2.fc1.weight" in tensors
    assert "talker.model.layers.2.self_attn.q_proj.weight" in tensors
    assert "thinker.model.embed_tokens.weight" in tensors
    assert (output / "tokenizer.json").is_file()

    output_index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert output_index["metadata"]["total_size"] == sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values()
    )
    assert manifest["removed_weight_count"] == 2
    assert manifest["shard_actions"]["model-00001-of-00003.safetensors"] == "copied"
    assert json.loads((output / "layer_reduction_manifest.json").read_text(encoding="utf-8")) == manifest

    # The source checkpoint is never modified.
    source_index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert source_index["weight_map"] == source_weight_map


def test_reduce_checkpoint_dry_run_does_not_create_output(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "reduced"
    _build_checkpoint(source)

    manifest = reduce_checkpoint(source, output, 2, dry_run=True)

    assert manifest["num_hidden_layers"] == 2
    assert manifest["kept_layer_indices"] == [0, 1]
    assert manifest["removed_layer_indices"] == [2]
    assert not output.exists()


def test_reduce_checkpoint_rejects_incomplete_layer_key_set(tmp_path: Path):
    source = tmp_path / "source"
    _build_checkpoint(source)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["weight_map"]["thinker.model.layers.1.self_attn.q_proj.weight"]
    _write_json(index_path, index)

    with pytest.raises(ValueError, match=r"missing=\[1\]"):
        reduce_checkpoint(source, tmp_path / "reduced", 1, dry_run=True)
