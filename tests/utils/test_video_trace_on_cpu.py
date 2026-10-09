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
"""Trace correctness tests; these do not simulate or reproduce issue #8039."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def trace(monkeypatch, tmp_path):
    # Avoid package registration and its optional Ray/model dependencies.
    path = Path(__file__).resolve().parents[2] / "verl_omni/utils/video_trace.py"
    spec = importlib.util.spec_from_file_location("video_trace_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_MAX_REQUESTS", "0")
    return module


def records(tmp_path):
    return [json.loads(line) for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines()]


def test_full_content_digest_is_frozen_and_layout_independent(trace, tmp_path):
    frames = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    expected = trace.snapshot(frames)
    assert trace.snapshot(np.asfortranarray(frames)) == expected
    with trace.request_scope("a"):
        trace.trace_prompt("entry", {"multi_modal_data": {"video": [(frames, {"fps": 2})]}})
        frames[-1, -1, -1] = 99
        trace.trace_prompt("later", {"multi_modal_data": {"video": [(frames, {"fps": 2})]}})
    first, second = records(tmp_path)
    assert first["video"][0][0] == expected
    assert second["video"][0][0]["sha256"] != expected["sha256"]
    assert first["video"][0][1] == {"fps": 2}


def test_disabled_never_hashes_or_writes(trace, monkeypatch, tmp_path):
    monkeypatch.delenv("VERL_OMNI_VIDEO_TRACE_DIR")

    def fail(*args, **kwargs):
        pytest.fail("Disabled tracing touched a payload")

    monkeypatch.setattr(trace, "snapshot", fail)

    @trace.trace_generate("server.receive")
    async def generate(request_id, *, video_data):
        return video_data

    video = object()
    assert asyncio.run(generate("r", video_data=video)) is video
    assert not list(tmp_path.iterdir())


def test_concurrent_requests_and_regenerated_ids_stay_associated(trace, tmp_path):
    class Client:
        def _vllm_request_id(self, request_id):
            return "engine-" + request_id

        @trace.trace_generate("agent.send")
        async def generate(self, request_id, *, video_data):
            await asyncio.sleep(0)
            self._vllm_request_id(request_id)
            await asyncio.sleep(0)
            trace.emit("inside", marker=request_id)
            if request_id == "bad":
                raise ValueError("original exception")
            return video_data

    trace._install_sync_hook(Client, "_vllm_request_id", "agent.request_id")
    wrapped = Client._vllm_request_id
    trace._install_sync_hook(Client, "_vllm_request_id", "agent.request_id")
    assert Client._vllm_request_id is wrapped

    async def run():
        client = Client()
        output = await asyncio.gather(
            *(client.generate(key, video_data=[np.array([index])]) for index, key in enumerate(("a", "bad", "b"))),
            return_exceptions=True,
        )
        assert isinstance(output[1], ValueError)
        assert str(output[1]) == "original exception"
        assert trace._context.get() is None

    asyncio.run(run())
    rows = records(tmp_path)
    assert {r["request_id"] for r in rows} == {"a", "b", "bad"}
    for key in ("a", "bad", "b"):
        group = [r for r in rows if r["request_id"] == key]
        assert len({r["trace_id"] for r in group}) == 1
        assert next(r for r in group if r["event"] == "inside")["marker"] == key
        mapping = next(r for r in group if r["event"] == "agent.request_id")
        assert mapping["engine_request_id"] == "engine-" + key
    assert any(r["event"] == "agent.send.error" for r in rows)


def test_frontend_staticmethod_cache_hit_and_engine_identity(trace, tmp_path):
    class MultiModalFieldElem:
        def __init__(self, data):
            self.data = data

    fresh = {"video": [{"pixel_values_videos": MultiModalFieldElem(np.ones((2, 4), dtype=np.float32))}]}
    merged = {"video": [None, fresh["video"][0]]}

    class Kwargs:
        @staticmethod
        def from_hf_inputs(hf_inputs, config_by_key):
            return hf_inputs

    class Processor:
        def _merge_mm_kwargs(self, cache, mm_hashes, mm_is_cached, mm_missing_kwargs, mm_missing_prompt_updates):
            return merged, {}

    submission = SimpleNamespace(
        prompt=SimpleNamespace(
            mm_features=[SimpleNamespace(modality="video", identifier="id-a", mm_hash="hash-a", data=None)]
        )
    )

    class Engine:
        def _build_add_request_message(self, request_id, prompt):
            assert Kwargs.from_hf_inputs(fresh, {}) is fresh
            result = Processor()._merge_mm_kwargs(None, {"video": ["a", "b"]}, {"video": [True, False]}, fresh, {})
            assert result[0] is merged
            return submission

    trace._install_sync_hook(Kwargs, "from_hf_inputs", "frontend.hf_fresh")
    trace._install_sync_hook(Processor, "_merge_mm_kwargs", "frontend.cache_merge")
    trace._install_sync_hook(Engine, "_build_add_request_message", "engine.build")
    prompt = {"prompt_token_ids": [1, 2], "multi_modal_data": {"video": [(np.zeros((1, 2)), {"fps": 1})]}}
    with trace.request_scope("server-id", replica_rank=3):
        assert Engine()._build_add_request_message("inner-id", prompt) is submission
    assert trace._context.get() is None
    rows = records(tmp_path)
    assert all(row["request_id"] == "server-id" and row["engine_request_id"] == "inner-id" for row in rows)
    assert all(row["replica_rank"] == 3 for row in rows)
    before = next(r for r in rows if r["event"] == "frontend.cache_merge.before")
    assert before["mm_is_cached"] == {"video": [True, False]}
    after = next(r for r in rows if r["event"] == "frontend.cache_merge.after")
    assert after["mm_kwargs"]["video"][0] is None
    hf = next(r for r in rows if r["event"] == "frontend.hf_fresh")
    assert hf["mm_kwargs"]["video"][0]["pixel_values_videos"] == trace.snapshot(np.ones((2, 4), dtype=np.float32))
    assert prompt["prompt_token_ids"] == [1, 2]
    assert "multi_modal_uuids" not in prompt


def test_trace_io_failure_does_not_change_result_or_exception(trace, monkeypatch, tmp_path):
    invalid_dir = tmp_path / "file"
    invalid_dir.write_text("not a directory")
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_DIR", str(invalid_dir))
    output = object()

    @trace.trace_generate("server.receive")
    async def generate(request_id):
        if request_id == "bad":
            raise RuntimeError("engine failed")
        return output

    assert asyncio.run(generate("ok")) is output
    with pytest.raises(RuntimeError, match="engine failed"):
        asyncio.run(generate("bad"))
    assert trace._context.get() is None


def test_request_limit_cannot_be_bypassed_by_nested_hooks(trace, monkeypatch, tmp_path):
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_MAX_REQUESTS", "1")
    for request_id in ("first", "second"):
        with trace.request_scope(request_id):
            with trace.request_scope("inner", engine_request_id="inner"):
                trace.emit("nested")
    assert [r["request_id"] for r in records(tmp_path)] == ["first"]


def test_missing_dependency_hooks_are_reported(trace, monkeypatch, tmp_path):
    def unavailable(name):
        raise ImportError(name)

    monkeypatch.setattr(trace.importlib, "import_module", unavailable)
    trace.install_frontend_hooks()
    trace.install_agent_hooks()
    rows = records(tmp_path)
    assert len(rows) == 6
    assert all(row["event"] == "hook.install" and row["status"] == "unavailable" for row in rows)


def test_uuid_scoping_and_preprocessor_mutation_are_visible(trace, tmp_path):
    class Frontend:
        def _ensure_stage_replica_mm_uuids(self, prompt, *, stage_id, replica_id):
            prompt["multi_modal_uuids"] = {"video": [f"stage{stage_id}:rep{replica_id}:video"]}

        def _process_tokens(self, parsed_content, tokenization_kwargs=None):
            # Deliberate corruption to verify observation, not a model of #8039.
            parsed_content["multi_modal_data"]["video"][0][0][0] = 99
            return {"mm_kwargs": {"video": [None]}, "mm_hashes": {"video": ["hash"]}}

    trace._install_sync_hook(Frontend, "_ensure_stage_replica_mm_uuids", "engine.scope_uuids")
    trace._install_sync_hook(Frontend, "_process_tokens", "frontend.process_tokens")
    prompt = {"multi_modal_data": {"video": [(np.array([1, 2]), {"fps": 1})]}}
    with trace.request_scope("a"):
        frontend = Frontend()
        assert frontend._ensure_stage_replica_mm_uuids(prompt, stage_id=0, replica_id=3) is None
        frontend._process_tokens(prompt)
    by_event = {row["event"]: row for row in records(tmp_path)}
    scoped = by_event["engine.scope_uuids.after"]
    assert scoped["multi_modal_uuids"] == {"video": ["stage0:rep3:video"]}
    assert scoped["replica_id"] == 3
    before = by_event["frontend.process_tokens.before"]["video"][0][0]
    after = by_event["frontend.process_tokens.after"]["video"][0][0]
    assert before["sha256"] != after["sha256"]


def test_incompatible_signature_is_rejected_without_patching(trace):
    class Engine:
        def build(self, renamed_id, renamed_prompt):
            return renamed_prompt

    original = Engine.build
    with pytest.raises(TypeError, match="Unsupported signature"):
        trace._install_sync_hook(Engine, "build", "engine.build")
    assert Engine.build is original


def test_bfloat16_digest_uses_bytes(trace):
    torch = pytest.importorskip("torch")
    tensor = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).T
    first = trace.snapshot(tensor)
    assert first == trace.snapshot(tensor.contiguous())
    tensor[0, 0] = 42
    assert trace.snapshot(tensor)["sha256"] != first["sha256"]
    assert trace.snapshot(torch.tensor(1.0))["shape"] == []
