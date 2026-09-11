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

"""Depth-only checkpoint exports and masked consistency measurements."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = _load("debug_export", "examples/gspo_trainer/debug/prepare_debug_model.py")
diagnostics = _load("consistency_debug", "verl_omni/utils/consistency_debug.py")
PATCH = ROOT / "examples/gspo_trainer/debug/qwen3_omni_4layer.json"


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    config = {
        "model_type": "qwen3_omni_moe",
        "enable_audio_output": True,
        "thinker_config": {
            "text_config": {"num_hidden_layers": 8, "hidden_size": 16, "num_experts": 8},
            "vision_config": {"depth": 12, "deepstack_visual_indexes": [3, 7, 11]},
            "audio_config": {"encoder_layers": 6},
        },
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "chat_template.json").write_text('{"chat_template": "original"}', encoding="utf-8")
    weights = {f"thinker.model.layers.{i}.mlp.experts.weight": torch.full((2, 3), float(i)) for i in range(8)}
    weights.update(
        {
            key: torch.ones(2, 3)
            for key in (
                "thinker.visual.weight",
                "thinker.audio_tower.weight",
                "thinker.model.embed_tokens.weight",
                "thinker.model.norm.weight",
                "thinker.lm_head.weight",
                "talker.weight",
                "code2wav.weight",
            )
        }
    )
    save_file(weights, directory / "model.safetensors")
    return directory, config, weights


def test_export_keeps_weights_and_architecture_except_depth(source, tmp_path):
    directory, original, weights = source
    output = exporter.prepare(directory, PATCH, tmp_path / "reduced")
    config = json.loads((output / "config.json").read_text())
    assert config["thinker_config"]["text_config"] == {
        **original["thinker_config"]["text_config"],
        "num_hidden_layers": 4,
    }
    assert config["thinker_config"]["vision_config"] == original["thinker_config"]["vision_config"]
    assert config["thinker_config"]["audio_config"] == original["thinker_config"]["audio_config"]
    assert config["enable_audio_output"] is False
    assert json.loads((directory / "config.json").read_text()) == original
    with safe_open(output / "model.safetensors", framework="pt") as reader:
        assert set(reader.keys()) == {name for name in weights if exporter.keep_weight(name, 4)}
        for key in reader.keys():
            torch.testing.assert_close(reader.get_tensor(key), weights[key], rtol=0, atol=0)
    assert (output / "chat_template.json").read_bytes() == (directory / "chat_template.json").read_bytes()
    assert exporter.prepare(directory, PATCH, output) == output


def test_sharded_export_skips_entire_removed_shard(source, tmp_path):
    directory, _, weights = source
    selected = {k: v for k, v in weights.items() if exporter.keep_weight(k, 4)}
    removed = {k: v for k, v in weights.items() if k not in selected}
    save_file(selected, directory / "retained.safetensors")
    save_file(removed, directory / "removed.safetensors")
    weight_map = {k: "retained.safetensors" if k in selected else "removed.safetensors" for k in weights}
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    output = exporter.prepare(directory, PATCH, tmp_path / "reduced")
    assert not (output / "removed.safetensors").exists()
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(selected)


@pytest.mark.parametrize("depth", [0, 1, 2, 9, True])
def test_reduction_preserves_all_deepstack_injections(source, depth):
    _, config, _ = source
    patch = json.loads(PATCH.read_text())
    patch["thinker_config"]["text_config"]["num_hidden_layers"] = depth
    with pytest.raises(ValueError):
        exporter.reduced_config(config, patch)


def test_disallow_unrelated_architecture_changes(source):
    _, config, _ = source
    patch = json.loads(PATCH.read_text())
    patch["thinker_config"]["text_config"]["hidden_size"] = 8
    with pytest.raises(ValueError, match="Only"):
        exporter.reduced_config(config, patch)


def test_stale_export_is_not_silently_reused(source, tmp_path):
    directory, _, _ = source
    output = exporter.prepare(directory, PATCH, tmp_path / "reduced")
    (directory / "chat_template.json").write_text('{"chat_template": "changed"}')
    with pytest.raises(FileExistsError):
        exporter.prepare(directory, PATCH, output)


def test_nested_output_is_rejected(source):
    directory, _, _ = source
    with pytest.raises(ValueError, match="non-nested"):
        exporter.prepare(directory, PATCH, directory / "debug")


def test_logprob_metrics_mask_padding_and_save_pairs(tmp_path):
    data = {
        "old_log_probs": torch.tensor([[-1.0, -2.0, float("nan")]]),
        "rollout_log_probs": torch.tensor([[-1.1, -2.3, 0.0]]),
        "response_mask": torch.tensor([[1, 1, 0]]),
        "responses": torch.tensor([[42, 43, 0]]),
    }
    metrics = diagnostics.save_consistency_batch(data, tmp_path, 2)
    assert metrics["consistency/valid"] == 1
    assert metrics["consistency/response_tokens"] == 2
    assert metrics["consistency/logprob_mae"] == pytest.approx(0.2)
    assert metrics["consistency/logprob_max_abs"] == pytest.approx(0.3)
    assert metrics["consistency/logprob_signed_mean"] == pytest.approx(0.2)
    saved = torch.load(tmp_path / "step_000002.pt", weights_only=True)
    torch.testing.assert_close(saved["responses"], data["responses"])
    assert saved["step"] == 2


@pytest.mark.parametrize("mask", [[[0, 0]], [[1, 1]]])
def test_empty_or_nonfinite_pairs_cannot_pass(tmp_path, mask):
    data = {
        "old_log_probs": torch.tensor([[float("nan"), -2.0]]),
        "rollout_log_probs": torch.tensor([[-1.0, -2.0]]),
        "response_mask": torch.tensor(mask),
        "responses": torch.tensor([[42, 43]]),
    }
    metrics = diagnostics.save_consistency_batch(data, tmp_path, 1)
    assert metrics["consistency/valid"] == 0


def test_misaligned_pairs_raise(tmp_path):
    data = {key: torch.ones(1, 2) for key in ("old_log_probs", "rollout_log_probs", "response_mask")}
    data["responses"] = torch.ones(1, 3)
    with pytest.raises(ValueError, match="aligned"):
        diagnostics.save_consistency_batch(data, tmp_path, 1)


def test_export_loads_with_real_transformers_without_missing_weights(tmp_path):
    from transformers.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration

    tiny = _load("tiny_omni_fixture", "tests/special_e2e/build_qwen3_omni_tiny_random.py")
    config = tiny._build_tiny_config(vocab_size=256)
    config.thinker_config.text_config.num_hidden_layers = 6
    original = Qwen3OmniMoeForConditionalGeneration(config)
    source = tmp_path / "original"
    original.save_pretrained(source, max_shard_size="2MB")
    output = exporter.prepare(source, PATCH, tmp_path / "reduced")
    reduced, info = Qwen3OmniMoeForConditionalGeneration.from_pretrained(output, output_loading_info=True)
    assert len(reduced.thinker.model.layers) == 4
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    assert not hasattr(reduced, "talker")
    assert all(parameter.requires_grad for parameter in reduced.parameters())
    original_weights = original.state_dict()
    for name, value in reduced.state_dict().items():
        torch.testing.assert_close(value, original_weights[name], rtol=0, atol=0)
