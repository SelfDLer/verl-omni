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
import struct
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tests.utils import test_video_trace_worker_on_cpu as helpers

trace = helpers.trace
frontend = helpers.frontend
load = helpers.load


@pytest.fixture
def numeric(trace, frontend, monkeypatch):
    module = load("numeric_test", "verl_omni/utils/video_trace_numeric.py")
    monkeypatch.setitem(sys.modules, "verl_omni.utils.video_trace_numeric", module)
    monkeypatch.setattr(sys.modules["verl_omni.utils"], "video_trace_numeric", module, raising=False)
    return module


@pytest.fixture
def worker_module(trace, frontend, numeric, monkeypatch):
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_STAGE", "worker")
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_MODE", "sample")
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_VISION", "1")
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_DEVICE_SAMPLE", "1")
    monkeypatch.setitem(sys.modules, "verl_omni.utils.video_trace_frontend", frontend)
    vision = load("vision_test", "verl_omni/utils/video_trace_vision.py")
    monkeypatch.setitem(sys.modules, "verl_omni.utils.video_trace_vision", vision)
    return load("vision_worker_test", "verl_omni/utils/video_trace_worker.py")


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Scale(Module):
    def __init__(self, scale=1):
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class Block(Module):
    def __init__(self):
        self.norm1, self.norm2, self.attn, self.mlp = Scale(), Scale(), Scale(), Scale()

    def forward(self, x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, max_seqlen, sequence_lengths):
        return x + self.mlp(self.norm2(x + self.attn(self.norm1(x))))


class Merger(Module):
    def forward(self, x):
        return x.reshape(-1, 4, x.shape[-1]).mean(axis=1)


class Visual(Module):
    spatial_merge_size = 2
    deepstack_visual_indexes = [0]
    training = False

    def __init__(self):
        self.patch_embed = Scale()
        self.blocks = [Block(), Block()]
        self.merger = Merger()
        self.merger_list = [Merger()]

    def fast_pos_embed_interpolate(self, grid_thw):
        return np.zeros((sum(np.prod(row) for row in grid_thw), 4), dtype=np.float32)

    def forward(self, x, grid_thw):
        x = self.patch_embed(x) + self.fast_pos_embed_interpolate(grid_thw)
        rotary = np.ones((x.shape[0], 2), dtype=np.float32)
        early = None
        for block in self.blocks:
            x = block(x, np.array([0, len(x)], dtype=np.int32), rotary, rotary, np.array(len(x)), None)
            if early is None:
                early = x
        return np.concatenate([self.merger(x), self.merger_list[0](early)], axis=1)

    def named_parameters(self):
        yield "patch.weight", np.ones((4, 4), dtype=np.float32)
        for index, block in enumerate(self.blocks):
            yield f"blocks.{index}.attn.weight", np.array(block.attn.scale, dtype=np.float32)

    def named_buffers(self):
        yield "position", np.ones(4, dtype=np.float32)


class Model:
    def __init__(self):
        self.visual = Visual()
        self.config = NS(vision_config=NS(out_hidden_size=4))
        self.config.video_token_id = 11
        self.deepstack = None

    def _process_video_input(self, video_input):
        grids = video_input["video_grid_thw"]
        output = self.visual(video_input["pixel_values_videos"], grids)
        boundaries = np.cumsum(np.prod(grids, axis=1) // 4)[:-1]
        return tuple(np.split(output, boundaries))

    def _set_deepstack_input_embeds(self, deepstack_input_embeds):
        self.deepstack = deepstack_input_embeds

    def _get_deepstack_input_embeds(self, num_tokens):
        return NS(tensors={"deepstack_input_embeds_0": self.deepstack[0, :num_tokens]})

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        result = np.zeros((len(input_ids), 4), dtype=np.float32)
        result[input_ids == 11] = np.concatenate([value[:, :4] for value in multimodal_embeddings])
        self._set_deepstack_input_embeds(np.ones((1, len(input_ids), 4), dtype=np.float32))
        return result


class Runner(helpers.Runner):
    def __init__(self):
        super().__init__()
        self.model = Model()

    def _batch_mm_inputs_from_scheduler(self, scheduler_output):
        items = [(r, f) for r in self.requests.values() for f in r.mm_features]
        return (
            [f.identifier for _, f in items],
            [(f.modality, f.data) for _, f in items],
            [(r.req_id, f.mm_position) for r, f in items],
        )

    def _preprocess(self, scheduler_output):
        hashes, inputs, _ = self._batch_mm_inputs_from_scheduler(scheduler_output)
        data = {
            "pixel_values_videos": np.concatenate([d["pixel_values_videos"] for _, d in inputs]),
            "video_grid_thw": np.stack([d["video_grid_thw"] for _, d in inputs]),
        }
        outputs = self.model._process_video_input(data)
        for key, value in zip(hashes, outputs, strict=True):
            self._cache_encoder_output(key, value)
        self._gather_mm_embeddings(scheduler_output)
        ids = np.array([token for request in self.requests.values() for token in request.prompt_token_ids])
        embeds = self.model.embed_input_ids(ids, list(self.cache.values()), is_multimodal=ids == 11)
        self.result = (None, embeds, np.arange(len(ids)), None, {}, None)
        return self.result


def setup(worker_module, trace):
    runner = Runner()
    worker = NS(model_runner=runner, rank=0, local_rank=0)
    worker_module.install(worker)
    with trace.scope("agent", sample_key="target", sample_index=0, video_id="v", question_id=0):
        context = trace.export_context()
    worker._video_observer.register("external", context)
    reqs = []
    for key, global_id, value in (("other", "other", 99), ("selected", "external", 1)):
        feature = NS(
            identifier=key,
            modality="video",
            mm_position=NS(offset=0, length=2, is_embed=None),
            data={
                "video_grid_thw": np.array([1, 2, 4]),
                "pixel_values_videos": np.full((8, 4), value, dtype=np.float32),
            },
        )
        request = helpers.request(key, global_id, [feature])
        request.prompt_token_ids = [11, 11, 12]
        reqs.append(request)
    scheduler = NS(scheduled_new_reqs=reqs, num_scheduled_tokens={r.req_id: 3 for r in reqs}, finished_req_ids=[])
    runner._update_states(scheduler)
    return worker, scheduler


def test_real_observation_pipeline_captures_selected_video_layers_weights_and_deepstack(worker_module, trace, tmp_path):
    worker, scheduler = setup(worker_module, trace)
    runner = worker.model_runner
    plain = Runner()
    plain._update_states(scheduler)
    plain._preprocess(scheduler)
    assert runner._preprocess(scheduler) is runner.result
    runner.model._get_deepstack_input_embeds(6)
    np.testing.assert_array_equal(runner.cache["selected"], plain.cache["selected"])
    records = helpers.rows(trace)
    assert not [row for row in records if row["event"] == "worker.vision.gap"]
    captured = {row["checkpoint"]: row for row in records if row["event"] == "worker.vision.tensor"}
    assert {
        "source",
        "visual.input",
        "blocks.00.attn.output",
        "blocks.01.output",
        "merger_list.0.output",
        "video.output",
        "cache.read.part.1",
        "deepstack.consume.deepstack_input_embeds_0",
    } <= captured.keys()
    diagnostics = load("numeric_diagnostics_test", "tests/special_e2e/video_trace_diagnostics.py")
    values = diagnostics.decode(captured["visual.input"]["value"])
    assert set(values) == {1.0}  # The neighboring video's value is 99.
    assert captured["visual.input"]["batch_offset"] == 8
    assert captured["source"]["value"]["pixel_values_videos"]["full_cpu"]["sha256"]
    compare = load("vision_compare_test", "tests/special_e2e/compare_8039_request.py")
    selected = compare.load_request([trace._sink.path], "target")
    selected["routes"] = [{}]
    report = compare.compare(selected, selected)
    assert all(
        row["components"]["value"]["status"] == "sample_equal"
        for row in report["vision_stages"]
        if "attention_layout" not in row["checkpoint"]
    )
    assert report["vision_coverage"]["single"]["0"]["missing_checkpoints"] == []
    for side in ("single", "multi"):
        (tmp_path / f"{side}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in selected["records"]), encoding="utf-8"
        )
    (tmp_path / "comparison.json").write_text(json.dumps(report), encoding="utf-8")
    summary = load("vision_summary_test", "tests/special_e2e/summarize_8039_worker.py").summarize(tmp_path)
    assert "max_abs=0" in summary
    assert all(
        row["components"]["value"]["status"] == "unknown"
        for row in report["vision_stages"]
        if "attention_layout" in row["checkpoint"]
    )
    assert len(summary.encode()) <= 65536


def test_numeric_failure_cannot_change_original_result(worker_module, trace, numeric, monkeypatch):
    worker, scheduler = setup(worker_module, trace)
    monkeypatch.setattr(numeric, "snapshot", lambda *a, **k: (_ for _ in ()).throw(ValueError("probe failure")))
    assert worker.model_runner._preprocess(scheduler) is worker.model_runner.result
    assert any(row["event"] == "worker.vision.gap" for row in helpers.rows(trace))


def test_same_grid_video_swap_is_exposed_by_within_run_input_checks(worker_module, trace, monkeypatch):
    original = Model._process_video_input

    def swapped(self, video_input):
        pixels = video_input["pixel_values_videos"]
        changed = {**video_input, "pixel_values_videos": np.concatenate([pixels[8:], pixels[:8]])}
        return original(self, changed)

    monkeypatch.setattr(Model, "_process_video_input", swapped)
    worker, scheduler = setup(worker_module, trace)
    worker.model_runner._preprocess(scheduler)
    helpers.rows(trace)
    compare = load("swapped_vision_compare", "tests/special_e2e/compare_8039_request.py")
    selected = compare.load_request([trace._sink.path], "target")
    selected["routes"] = [{}]
    report = compare.compare(selected, selected)
    mismatch = next(row for row in report["vision_flow"]["single"] if row["edge"] == "video.input -> visual.input")
    assert mismatch["status"] == "different"
    assert mismatch["numeric"]["$"]["max_abs"] == 98


def test_batch_mapping_mismatch_is_a_gap_not_guessed_alignment(worker_module, trace, monkeypatch):
    worker, scheduler = setup(worker_module, trace)
    vision = worker._video_observer.vision
    original = vision.tensor

    def corrupt_membership(checkpoint, *args, **kwargs):
        if checkpoint == "video.input":
            vision.current[0]["grid"] = [2, 2, 2]
        return original(checkpoint, *args, **kwargs)

    monkeypatch.setattr(vision, "tensor", corrupt_membership)
    assert worker.model_runner._preprocess(scheduler) is worker.model_runner.result
    records = helpers.rows(trace)
    assert any(row.get("reason", "").startswith("visual grid differs") for row in records)
    assert not any(row.get("checkpoint") == "visual.output" for row in records)


def test_numeric_samples_preserve_noncontiguous_coordinates_and_values(numeric, trace):
    values = np.arange(4096, dtype=np.float32).reshape(64, 64)[:, ::2]
    observed = trace._finalize(numeric.snapshot(values, limit=256))
    diagnostics = load("numeric_decode_test", "tests/special_e2e/video_trace_diagnostics.py")
    expected = values.flatten()[np.linspace(0, values.size - 1, 256, dtype=np.int64)]
    np.testing.assert_array_equal(diagnostics.decode(observed), expected)
    changed = trace._finalize(numeric.snapshot(values + 0.5, limit=256))
    difference = diagnostics.numeric_difference(observed, changed)
    assert difference["max_abs"] == 0.5 and difference["different_elements"] == 256


def test_changed_attention_is_localized_with_weights_and_consistent_cache_flow(worker_module, trace):
    worker, scheduler = setup(worker_module, trace)
    worker.model_runner._preprocess(scheduler)
    worker.model_runner.model._get_deepstack_input_embeds(6)
    left_id = next(row["trace_id"] for row in helpers.rows(trace) if row["event"] == "worker.receive")
    other, scheduler = setup(worker_module, trace)
    other.model_runner.model.visual.blocks[1].attn.scale = 1.1
    other.model_runner._preprocess(scheduler)
    other.model_runner.model._get_deepstack_input_embeds(6)
    right_id = [row["trace_id"] for row in helpers.rows(trace) if row["event"] == "worker.receive"][-1]
    compare = load("changed_vision_compare", "tests/special_e2e/compare_8039_request.py")
    left, right = (compare.load_request([trace._sink.path], "target", identity) for identity in (left_id, right_id))
    left["routes"] = right["routes"] = [{}]
    report = compare.compare(left, right)
    changed = [row for row in report["vision_stages"] if row["components"]["value"]["status"] == "different"]
    assert changed[0]["checkpoint"] == "blocks.01.attn.output"
    assert any(row["checkpoint"] == "weight.parameter.blocks.1.attn.weight" for row in changed)
    assert changed[0]["components"]["value"]["numeric"]["$"]["max_abs"] == pytest.approx(0.3)
    for side in ("single", "multi"):
        assert all(row["status"] == "sample_equal" for row in report["vision_flow"][side])


def test_observer_is_instance_local_and_original_exceptions_propagate(worker_module, trace, monkeypatch):
    untouched = Runner()
    original = untouched.model.visual.forward
    worker, scheduler = setup(worker_module, trace)
    assert untouched.model.visual.forward == original
    error = RuntimeError("model failed")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(worker.model_runner.model.visual.blocks[0].attn, "forward", fail)
    with pytest.raises(RuntimeError) as caught:
        worker.model_runner._preprocess(scheduler)
    assert caught.value is error
    assert worker._video_observer.active == []
    assert worker._video_observer.vision.current == []
    assert not worker._video_observer.vision.visual_active


def test_vision_can_be_enabled_on_preexisting_worker(worker_module, trace, monkeypatch):
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_VISION", "0")
    worker = NS(model_runner=Runner(), rank=0, local_rank=0)
    worker_module.install(worker)
    assert worker._video_observer.vision is None
    with trace.scope("agent", sample_key="target"):
        context = trace.export_context()
    reply = worker_module.register_worker(worker, "request", context, {"VERL_OMNI_VIDEO_TRACE_VISION": "1"})
    assert reply["hooks"]["vision"]["status"] == "installed"
    original = worker.model_runner.model.visual.forward
    worker_module.register_worker(worker, "request", context)
    assert worker.model_runner.model.visual.forward == original


def test_live_health_works_before_compare_and_handles_partial_lines(worker_module, trace, tmp_path):
    worker, scheduler = setup(worker_module, trace)
    worker.model_runner._preprocess(scheduler)
    worker.model_runner.model._get_deepstack_input_embeds(6)
    helpers.rows(trace)
    summary = load("live_vision_summary", "tests/special_e2e/summarize_8039_worker.py")
    output = summary.live_health([trace._sink.path], "target")
    assert "missing=[] unavailable=[]" in output
    assert "weights=4" in output
    assert "NO_VISION_EVENTS" in summary.live_health([trace._sink.path], "other-sample")
    partial = tmp_path / "partial.jsonl"
    partial.write_text('{"event":', encoding="utf-8")
    assert "invalid_or_partial_lines=1" in summary.live_health([trace._sink.path, partial], "target")


def test_installed_but_bypassed_layers_are_reported_missing(worker_module, trace, monkeypatch):
    def fused(self, x, grid_thw):
        return np.zeros((x.shape[0] // 4, 8), dtype=np.float32)

    monkeypatch.setattr(Visual, "forward", fused)
    worker, scheduler = setup(worker_module, trace)
    worker.model_runner._preprocess(scheduler)
    worker.model_runner.model._get_deepstack_input_embeds(6)
    helpers.rows(trace)
    summary = load("bypassed_vision_summary", "tests/special_e2e/summarize_8039_worker.py")
    output = summary.live_health([trace._sink.path], "target")
    assert "blocks.00.output" in output and "patch_embed.output" in output
    assert "unavailable=[]" in output  # Installation succeeded; execution did not enter these layers.


def test_bfloat16_numeric_decode_and_nonfinite_values():
    diagnostic = load("bf16_numeric_diagnostic", "tests/special_e2e/video_trace_diagnostics.py")
    raw = struct.pack("<4H", 0x3F80, 0x4000, 0x7F80, 0x7FC0)
    value = {
        "sample_count": 4,
        "sample_data_b64": base64.b64encode(raw).decode(),
        "sample_sha256": hashlib.sha256(raw).hexdigest(),
        "sample_byteorder": "little",
        "dtype": "bfloat16",
        "shape": [4],
        "sample_scheme": "linspace-flat-c-order-v1",
    }
    result = diagnostic.numeric_difference(value, value)
    assert result["single_nonfinite"] == 2 and result["finite_pairs"] == 2
    assert diagnostic.decode(value)[:2] == [1.0, 2.0]
    value["sample_sha256"] = "tampered"
    assert diagnostic.numeric_difference(value, value)["status"] == "unknown"


def test_logical_rows_match_cache_sampling_without_copying_full_embeddings(numeric, trace):
    original = np.arange(8000, dtype=np.float32).reshape(1000, 8)
    rows = list(range(3, 900, 7))
    selected = trace._finalize(numeric.snapshot(original, row_indices=rows))
    compact = trace._finalize(numeric.snapshot(original[rows]))
    assert selected["sample_sha256"] == compact["sample_sha256"]
    empty = trace._finalize(numeric.snapshot(original, row_indices=[]))
    assert empty["sample_count"] == 0 and empty["sample_data_b64"] == ""


def test_accelerator_reads_require_opt_in_and_transfer_only_bounded_samples(numeric, trace, monkeypatch):
    transfers = []

    class Tensor:
        __module__ = "torch"

        def __init__(self, value):
            self.value = value
            self.device = "npu:3"
            self.shape, self.dtype, self.ndim = value.shape, value.dtype, value.ndim

        def stride(self):
            return tuple(step // self.value.itemsize for step in self.value.strides)

        def detach(self):
            return self

        def __getitem__(self, key):
            return Tensor(self.value[key])

        def contiguous(self):
            return Tensor(np.ascontiguousarray(self.value))

        def reshape(self, *shape):
            return Tensor(self.value.reshape(*shape))

        def view(self, dtype):
            return Tensor(self.value.view(dtype))

        def cpu(self):
            transfers.append(self.value.nbytes)
            return self

        def numpy(self):
            return self.value

    monkeypatch.setitem(sys.modules, "torch", NS(tensor=lambda x, device: np.asarray(x), uint8=np.uint8))
    value = Tensor(np.arange(100_000, dtype=np.float32).reshape(1000, 100)[:, ::2])
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_DEVICE_SAMPLE", "0")
    assert numeric.snapshot(value)["digest_status"] == "skipped_non_cpu"
    assert transfers == []
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_DEVICE_SAMPLE", "1")
    result = trace._finalize(numeric.snapshot(value, limit=99999, full_cpu=True))
    assert result["sample_count"] == 4096 and transfers == [4096 * 4]
    assert "full_cpu" not in result
