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

"""Real tiny VeOmni forwards on CPU; requires the optional pinned VeOmni package."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def native_model():
    from dataclasses import fields

    pytest.importorskip("veomni")
    from veomni.arguments import OpsImplementationConfig
    from veomni.models.auto import build_foundation_model
    from veomni.models.transformers.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig

    builder = load_file("veomni_tiny_builder", ROOT / "tests/special_e2e/build_qwen3_omni_tiny_random.py")
    compat = load_file("veomni_compat_test", ROOT / "verl_omni/workers/engine/veomni/omni_compat.py")
    config = Qwen3OmniMoeConfig(**builder._build_tiny_config(128).to_dict())
    config.architectures = ["Qwen3OmniMoeForConditionalGeneration"]
    config.thinker_config.image_token_id = 120
    config.thinker_config.video_token_id = 121
    config.thinker_config.audio_token_id = 122
    ops = OpsImplementationConfig(
        **{
            field.name: "sdpa" if field.name == "attn_implementation" else "eager"
            for field in fields(OpsImplementationConfig)
            if field.name.endswith("_implementation")
        }
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = build_foundation_model(
            config_path=config, torch_dtype="float32", init_device="cpu", ops_implementation=ops
        )
    compat.configure_omni_compat(model)
    compat.configure_omni_compat(model)  # Setup must be idempotent.
    return model.train()


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_native_text_backward(native_model, batch_size):
    ids = torch.tensor([[1, 2, 3, 4]]).expand(batch_size, -1)
    output = native_model(
        input_ids=ids,
        position_ids=torch.arange(4).view(1, 1, 4).expand(4, batch_size, -1),
        attention_mask=torch.ones_like(ids),
        image_mask=ids == 120,
        video_mask=ids == 121,
        audio_mask=ids == 122,
        use_cache=False,
    )
    assert output.logits.shape == (batch_size, 4, 128)
    output.logits.square().mean().backward()
    assert torch.isfinite(native_model.thinker.lm_head.weight.grad).all()
    native_model.zero_grad(set_to_none=True)


@pytest.mark.parametrize("modality", ["image", "video", "audio"])
def test_native_multimodal_backward(native_model, modality):
    inputs = load_file("veomni_inputs_test", ROOT / "verl_omni/workers/engine/veomni/omni_inputs.py")
    token = {"image": 120, "video": 121, "audio": 122}[modality]
    ids = torch.tensor([[1, token, 3, 4]])
    mm = {}
    if modality in ("image", "video"):
        pixel_key = "pixel_values" if modality == "image" else "pixel_values_videos"
        mm[pixel_key] = torch.randn(4, 3 * 2 * 16 * 16)
        mm[f"{modality}_grid_thw"] = torch.tensor([[1, 2, 2]])
    else:
        mm["input_features"] = torch.randn(1, 128, 8)
        mm["feature_attention_mask"] = torch.ones(1, 8, dtype=torch.long)
    inputs.prepare_omni_inputs(mm, ids, native_model.config.thinker_config)
    output = native_model(
        input_ids=ids,
        position_ids=torch.arange(4).view(1, 1, 4).expand(4, 1, -1),
        attention_mask=torch.ones_like(ids),
        use_cache=False,
        **mm,
    )
    output.logits.square().mean().backward()
    tower = native_model.thinker.audio_tower if modality == "audio" else native_model.thinker.visual
    gradients = [p.grad for p in tower.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    native_model.zero_grad(set_to_none=True)


def test_sdpa_audio_clips_do_not_cross_attend(native_model):
    features = torch.randn(24, 128)
    with torch.no_grad():
        separate = torch.cat(
            [
                native_model.thinker.get_audio_features(features[:8], audio_feature_lengths=torch.tensor([8])),
                native_model.thinker.get_audio_features(features[8:], audio_feature_lengths=torch.tensor([16])),
            ]
        )
        combined = native_model.thinker.get_audio_features(features, audio_feature_lengths=torch.tensor([8, 16]))
    torch.testing.assert_close(combined, separate, atol=1e-5, rtol=1e-4)


def test_native_text_logprobs_match_hf_thinker(native_model):
    from copy import deepcopy

    from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

    reference = Qwen3OmniMoeThinkerForConditionalGeneration._from_config(
        deepcopy(native_model.config.thinker_config), attn_implementation="sdpa"
    )
    reference.load_state_dict(native_model.thinker.state_dict(), strict=True)
    ids = torch.tensor([[1, 2, 3, 4]])
    kwargs = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "position_ids": torch.arange(4).view(1, 1, 4).expand(4, 1, -1),
        "use_cache": False,
    }
    with torch.no_grad():
        expected = reference(**kwargs).logits.log_softmax(-1)
        actual = native_model(**kwargs, image_mask=ids == 120, video_mask=ids == 121, audio_mask=ids == 122)
        actual = actual.logits.log_softmax(-1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.fixture
def real_engine_class(monkeypatch):
    from copy import deepcopy

    pytest.importorskip("verl")
    from verl.workers.engine.base import EngineRegistry

    registry = deepcopy(EngineRegistry._engines)
    registry.get("omni_model", {}).pop("veomni", None)
    monkeypatch.setattr(EngineRegistry, "_engines", registry)
    # Isolate only the package initializers that register vLLM pipelines. The
    # engine, config classes, TensorDict and verl parent implementation are real.
    for name in (
        "verl_omni",
        "verl_omni.workers",
        "verl_omni.workers.engine",
        "verl_omni.workers.engine.veomni",
    ):
        package = ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        package.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
        monkeypatch.setitem(sys.modules, name, package)
    module = load_file(
        "verl_omni.workers.engine.veomni.omni_impl", ROOT / "verl_omni/workers/engine/veomni/omni_impl.py"
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.OmniVeOmniEngine


@pytest.mark.parametrize("packed", [False, True])
def test_real_verl_input_and_logprob_contract(native_model, real_engine_class, packed):
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl.workers.config import VeOmniEngineConfig

    engine = object.__new__(real_engine_class)
    engine.module = native_model
    engine.engine_config = VeOmniEngineConfig(attn_implementation="sdpa", use_torch_compile=False)
    engine.pad_to_length = False
    engine.use_ulysses_sp = False
    engine.ulysses_sequence_parallel_size = 1
    engine._router_replay = None
    sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8])]
    ids = torch.nested.nested_tensor(sequences, layout=torch.jagged)
    positions = torch.nested.nested_tensor(
        [torch.arange(len(seq)).expand(4, -1) for seq in sequences], layout=torch.jagged
    )
    batch = TensorDict({"input_ids": ids, "position_ids": positions}, batch_size=[2])
    tu.assign_non_tensor(batch, temperature=0.8, use_remove_padding=packed, use_fused_kernels=False)
    model_inputs, output_args = engine.prepare_model_inputs(batch)
    assert model_inputs["image_mask"].shape == model_inputs["input_ids"].shape
    if packed:
        # CPU SDPA cannot consume the packed FA boundaries; input preparation is
        # still exercised against the real parent to check token alignment.
        assert model_inputs["cu_seq_lens_q"].tolist() == [0, 3, 8]
        assert model_inputs["position_ids"].shape == (4, 1, 8)
        return
    with torch.no_grad():
        output = native_model(**model_inputs)
        result = engine.prepare_model_outputs(output, output_args, batch, None)
        selected = output.logits.float().div(0.8).log_softmax(-1)
        # Compare only within-sequence next-token probabilities; terminal-token
        # slots are subsequently excluded by the RL response/loss mask.
        for index, sequence in enumerate(sequences):
            expected = selected[index, : len(sequence) - 1].gather(-1, sequence[1:, None]).squeeze(-1)
            torch.testing.assert_close(result["log_probs"].unbind()[index][:-1], expected)


def test_sp_keeps_global_masks_and_slices_encoder_inputs(native_model, real_engine_class, monkeypatch):
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl.workers.engine.veomni.transformer_impl import OmniSequenceShardCollator

    collator = OmniSequenceShardCollator()
    collator.sp_size, collator.sp_rank = 2, 1
    monkeypatch.setattr(sys.modules[real_engine_class.__module__], "OmniSequenceShardCollator", lambda: collator)
    engine = object.__new__(real_engine_class)
    engine.module = native_model
    engine.use_ulysses_sp = True
    engine.ulysses_sequence_parallel_size = 2
    ids = torch.nested.nested_tensor([torch.tensor([120, 120, 120, 122, 3])], layout=torch.jagged)
    batch = TensorDict({"input_ids": ids}, batch_size=[1])
    tu.assign_non_tensor(batch, use_remove_padding=True)
    features = torch.arange(128 * 5).reshape(1, 128, 5).float()
    pixels = torch.randn(12, 1536)
    inputs = {
        "input_ids": torch.tensor([[122, 3, 0]]),
        "position_ids": torch.arange(6).view(1, 1, 6).expand(4, 1, -1),
        "pixel_values": pixels,
        "input_features": features,
        "feature_attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    engine._apply_veomni_input_transforms(inputs, batch)
    assert inputs["image_mask"].shape == (1, 6)
    assert inputs["image_mask"].sum() == 3
    assert inputs["audio_mask"].tolist() == [[False, False, False, True, False, False]]
    assert inputs["cu_seq_lens_q"].tolist() == [0, 5, 6]
    assert inputs["position_ids"].shape == (4, 1, 3)
    torch.testing.assert_close(inputs["pixel_values"], torch.cat((pixels[8:], torch.zeros(4, 1536))))
    expected_audio = torch.cat((features[0, :, 3:].T, torch.zeros(1, 128))).to(torch.bfloat16)
    torch.testing.assert_close(inputs["input_features"], expected_audio)
    assert inputs["audio_feature_lengths"].tolist() == [5]
