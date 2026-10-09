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
"""Regression tests for diagnostic interference, not a reproduction of #8039."""

import ast
import asyncio
import contextvars
import hashlib
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trace(monkeypatch, tmp_path):
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_MODE", "metadata")
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_STAGE", "boundary")
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_MAX_REQUESTS", "0")
    module = load("diagnostic_test", "verl_omni/utils/video_trace.py")
    yield module
    module.flush(2)
    if module._sink is not None:
        module._sink.queue.put_nowait(None)
        module._sink.thread.join(2)


def rows(trace):
    assert trace.flush(2)
    if trace._sink is None:
        return []
    return [json.loads(line) for line in trace._sink.path.read_text(encoding="utf-8").splitlines()]


def test_disabled_does_not_start_thread_read_data_or_create_directory(trace, monkeypatch, tmp_path):
    monkeypatch.delenv("VERL_OMNI_VIDEO_TRACE_DIR")
    with trace.scope("request", request_id="a"):
        trace.prompt("input", {"multi_modal_data": {"video": [object()]}})
        trace.event("forced", force=True)
        trace.event("tokens", token_fields={"response_token_snapshot": object()})
        trace.messages("messages", object())
    assert trace._sink is None
    assert not list(tmp_path.iterdir())


def test_token_snapshots_are_complete_immutable_and_not_limited_to_128_items(trace, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = trace._Writer._write

    def blocked(self, stream, record):
        entered.set()
        assert release.wait(3)
        original(self, stream, record)

    monkeypatch.setattr(trace._Writer, "_write", blocked)
    ids = list(range(500))
    expected = list(ids)
    try:
        with trace.scope("request"):
            trace.prompt("input", {"prompt_token_ids": ids})
            assert entered.wait(1)
            ids[:] = [999]
    finally:
        release.set()
    snapshot = next(row for row in rows(trace) if row["event"] == "input")["prompt_token_snapshot"]
    assert snapshot["status"] == "complete"
    assert snapshot["ids"] == expected
    assert snapshot["count"] == 500


def test_token_snapshots_refuse_tensor_scalars_and_enforce_limit(trace, monkeypatch):
    class DeviceScalar:
        def __index__(self):
            pytest.fail("token observation synchronized a device")

    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_MAX_TOKENS", "2")
    with trace.scope("request"):
        trace.event("device", token_fields={"tokens": [DeviceScalar()]})
        trace.event("large", token_fields={"tokens": [1, 2, 3]})
    records = {row["event"]: row for row in rows(trace)}
    assert records["device"]["tokens"]["status"] == "unsupported_element_type"
    assert records["large"]["tokens"]["status"] == "skipped_token_limit"
    assert "ids" not in records["large"]["tokens"]


def test_message_observation_skips_media_and_reports_text_truncation(trace):
    class Media:
        def __repr__(self):
            pytest.fail("message observer inspected media")

    with trace.scope("request"):
        trace.messages(
            "source",
            [
                {"role": "system", "content": "Return <answer>A</answer>"},
                {
                    "role": "user",
                    "content": [{"type": "video", "video": Media()}, {"type": "text", "text": "question"}],
                },
            ],
        )
        trace.event("long", text_fields={"text": "x" * 65536 + "<answer>B</answer>"})
    records = rows(trace)
    messages = [row for row in records if row["event"] == "source"]
    assert len(messages) == 2
    assert messages[0]["text"]["has_answer_pair"]
    assert messages[1]["part_index"] == 1
    long = next(row for row in records if row["event"] == "long")["text"]
    assert not long["complete"]
    assert long["sha256"] is None
    assert long["characters"] > 65536


def test_message_observation_never_consumes_an_iterator_or_calls_tokenizer(trace):
    def source():
        pytest.fail("observation consumed the source iterator")
        yield

    class Tokenizer:
        name_or_path = "checkpoint"
        chat_template = "template {{ messages }}"
        eos_token_id = 151645

        def decode(self, *args, **kwargs):
            pytest.fail("inference observation invoked the tokenizer")

    with trace.scope("request"):
        trace.messages("source", source())
        trace.agent_config(SimpleNamespace(tokenizer=Tokenizer()))
    records = rows(trace)
    assert next(row for row in records if row["event"] == "source")["status"] == "unsupported_messages"
    config = next(row for row in records if row["event"] == "agent.template.config")
    assert config["tokenizer_name"] == "checkpoint"
    assert config["tokenizer_template"]["complete"]


def test_metadata_never_reads_tensor_contents(trace):
    class Tensor:
        __module__ = "torch"
        shape = (12, 3, 256, 352)
        dtype = "torch.bfloat16"
        device = "npu:0"

        def stride(self):
            return (270336, 90112, 352, 1)

        def detach(self):
            pytest.fail("metadata touched tensor contents")

        def cpu(self):
            pytest.fail("diagnostics initiated a device transfer")

    with trace.scope("request"):
        trace.event("input", video=Tensor())
    record = next(r for r in rows(trace) if r["event"] == "input")
    assert record["video"]["digest_status"] == "metadata_only"
    assert record["video"]["shape"] == [12, 3, 256, 352]


@pytest.mark.parametrize("mode", ["sample", "full"])
def test_device_data_is_never_read_even_with_hashing(trace, monkeypatch, mode):
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_MODE", mode)

    class Tensor:
        __module__ = "torch"
        shape = (10,)
        dtype = "torch.float32"
        device = "cuda:0"

        def stride(self):
            return (1,)

        def detach(self):
            pytest.fail("device tensor read")

    with trace.scope("request"):
        trace.event("input", video=Tensor())
    assert next(r for r in rows(trace) if r["event"] == "input")["video"]["digest_status"] == "skipped_non_cpu"


def test_full_snapshot_copies_bytes_before_background_write(trace, monkeypatch):
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_MODE", "full")
    entered, release = threading.Event(), threading.Event()
    original = trace._Writer._write

    def slow_write(self, stream, record):
        entered.set()
        assert release.wait(3)
        original(self, stream, record)

    monkeypatch.setattr(trace._Writer, "_write", slow_write)
    video = np.arange(96, dtype=np.float32).reshape(3, 4, 8).transpose(0, 2, 1)
    expected = hashlib.sha256(video.tobytes(order="C")).hexdigest()
    try:
        with trace.scope("request"):
            trace.event("input", video=video)
            assert entered.wait(1)
            video[:] = 0
    finally:
        release.set()
    assert next(r for r in rows(trace) if r["event"] == "input")["video"]["sha256"] == expected


def test_sample_is_bounded_and_not_labeled_as_full_digest(trace, monkeypatch):
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_MODE", "sample")
    value = np.arange(10000, dtype=np.int32).reshape(100, 100).T
    with trace.scope("request"):
        trace.event("input", video=value)
    fingerprint = next(r for r in rows(trace) if r["event"] == "input")["video"]
    indices = np.linspace(0, value.size - 1, 256, dtype=np.int64)
    expected = value[tuple(np.unravel_index(indices, value.shape))].tobytes()
    assert fingerprint["sample_sha256"] == hashlib.sha256(expected).hexdigest()
    assert fingerprint["sample_count"] == 256
    assert "sha256" not in fingerprint


def test_full_digest_byte_limit_is_explicit(trace, monkeypatch):
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_MODE", "full")
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_MAX_BYTES", "4")
    with trace.scope("request"):
        trace.event("input", video=np.zeros(5, dtype=np.uint8))
    assert next(r for r in rows(trace) if r["event"] == "input")["video"]["digest_status"] == "skipped_byte_limit"


def test_blocked_filesystem_and_full_queue_do_not_block_event_loop(trace, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = trace._Writer._write

    def blocked_write(self, stream, record):
        entered.set()
        assert release.wait(3)
        original(self, stream, record)

    monkeypatch.setattr(trace._Writer, "_write", blocked_write)
    with trace.scope("initial"):
        trace.event("input")
    assert entered.wait(1)

    async def run():
        async def request(key):
            with trace.scope("request", request_id=key):
                for _ in range(300):
                    trace.event("step")
                await asyncio.sleep(0)
                return key

        return await asyncio.wait_for(asyncio.gather(request("a"), request("b")), timeout=1)

    try:
        assert asyncio.run(run()) == ["a", "b"]
        assert not release.is_set()
        assert trace._sink.dropped > 0
        assert trace._sink.queue.qsize() <= 128
    finally:
        release.set()
    assert any(row["dropped_events"] > 0 for row in rows(trace))


def test_write_error_does_not_change_request_exception(trace, monkeypatch):
    error = RuntimeError("original engine failure")

    def fail(self, stream, record):
        raise OSError("shared filesystem unavailable")

    monkeypatch.setattr(trace._Writer, "_write", fail)
    with pytest.raises(RuntimeError) as result:
        with trace.scope("request"):
            trace.event("input")
            raise error
    assert result.value is error
    trace.flush(1)
    assert "shared filesystem unavailable" in trace._sink.error
    assert trace._context.get() is None


def test_concurrent_contexts_and_cancellation(trace):
    async def run():
        async def request(key):
            with trace.scope("request", uid=key):
                await asyncio.sleep(0)
                trace.event("point", marker=key)
                if key == "cancel":
                    raise asyncio.CancelledError()
                return trace.export_context()

        results = await asyncio.gather(request("a"), request("cancel"), return_exceptions=True)
        assert isinstance(results[1], asyncio.CancelledError)
        assert trace._context.get() is None

    asyncio.run(run())
    for row in rows(trace):
        if row["event"] == "point":
            assert row["uid"] == row["marker"]


def test_remote_context_preserves_sampling_decision(trace, monkeypatch):
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_MAX_REQUESTS", "1")
    with trace.scope("agent", uid="first"):
        first = trace.export_context()
    with trace.scope("agent", uid="second"):
        second = trace.export_context()
    with trace.scope("server", incoming=first, request_id="engine-a"):
        trace.event("server.receive")
    with trace.scope("server", incoming=second, request_id="engine-b"):
        trace.event("server.receive")
    server_rows = [row for row in rows(trace) if row["event"] == "server.receive"]
    assert len(server_rows) == 1
    assert server_rows[0]["uid"] == "first"
    assert server_rows[0]["trace_id"] == first["trace_id"]


def test_numpy_sample_identity_and_metadata_field_named_fingerprint(trace):
    with trace.scope("agent", sample_index=np.int64(17)):
        trace.event("input", metadata={"fingerprint": "dataset supplied value"})
    record = next(row for row in rows(trace) if row["event"] == "input")
    assert record["sample_index"] == 17
    assert record["metadata"]["fingerprint"] == "dataset supplied value"


def test_byte_queue_budget_is_enforced(trace):
    writer = trace._writer()
    writer.byte_limit = 4
    writer.put({"event": "oversized", "value": trace._Bytes(b"12345", "sha256")})
    assert writer.dropped == 1
    assert writer.pending_bytes == 0
    assert writer.queue.qsize() == 0


@pytest.mark.parametrize("fail", [False, True])
def test_real_server_generate_forwards_args_and_preserves_results(trace, fail):
    # Exercise the repository's actual method without importing Ray/torch/vLLM.
    path = ROOT / "verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "vLLMOmniHttpServer")
    method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "generate")
    unit = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = {"trace": trace}
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
    result = SimpleNamespace(token_ids=[], log_probs=[], stop_reason="completed")
    error = RuntimeError("engine failed")
    received = []

    class Strategy:
        async def generate(self, **kwargs):
            received.append(kwargs)
            if fail:
                raise error
            return result

    server = SimpleNamespace(_generate_strategy=Strategy(), replica_rank=3, node_rank=1)
    video, params = [np.zeros((1, 2))], {"temperature": 1}
    call = namespace["generate"](
        server,
        [1],
        params,
        "engine-request",
        video_data=video,
        video_trace_context={"trace_id": "from-agent", "uid": "sample", "selected": True},
    )
    if fail:
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(call)
        assert caught.value is error
    else:
        assert asyncio.run(call) is result
    assert received[0]["video_data"] is video
    assert received[0]["sampling_params"] is params
    assert "video_trace_context" not in received[0]
    assert all(row["trace_id"] == "from-agent" for row in rows(trace))


@pytest.fixture
def frontend(trace, monkeypatch):
    utils = ModuleType("verl_omni.utils")
    utils.video_trace = trace
    monkeypatch.setitem(sys.modules, "verl_omni.utils", utils)
    return load("frontend_test", "verl_omni/utils/video_trace_frontend.py")


def test_frontend_is_opt_in_and_only_modifies_one_instance(trace, frontend, monkeypatch):
    class Engine:
        def _build_add_request_message(self, request_id, prompt):
            return prompt

    engine, other = Engine(), Engine()
    original = Engine._build_add_request_message
    frontend.install(SimpleNamespace(engine=engine))
    assert "_build_add_request_message" not in engine.__dict__
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_STAGE", "frontend")
    frontend.install(SimpleNamespace(engine=engine))
    assert Engine._build_add_request_message is original
    assert "_build_add_request_message" not in other.__dict__
    wrapped = engine._build_add_request_message
    frontend.install(SimpleNamespace(engine=engine))
    assert engine._build_add_request_message is wrapped
    prompt = {"prompt_token_ids": [1], "multi_modal_data": {"video": [(np.zeros(1), {})]}}
    with trace.scope("server", request_id="request"):
        assert engine._build_add_request_message("request", prompt) is prompt
    events = rows(trace)
    assert any(row["event"] == "frontend.build.before" for row in events)
    assert any(row.get("status") == "unavailable" for row in events)


def test_frontend_preserves_exceptions(trace, frontend):
    error = ValueError("original processor exception")

    class Processor:
        def _process_tokens(self, parsed_content):
            raise error

    processor = Processor()
    frontend._attach(processor, "_process_tokens", "frontend.tokens", {"parsed_content"})
    with trace.scope("server"), pytest.raises(ValueError) as result:
        processor._process_tokens({})
    assert result.value is error


@pytest.mark.parametrize("as_mapping", [False, True])
def test_frontend_records_returned_core_prompt_tokens(trace, frontend, as_mapping):
    returned = (
        {"prompt_token_ids": [7, 8, 9], "prompt": "text alongside tokens"}
        if as_mapping
        else SimpleNamespace(prompt_token_ids=[7, 8, 9])
    )

    class Processor:
        def process_inputs(self, request_id, prompt):
            return returned

    processor = Processor()
    frontend._attach(processor, "process_inputs", "frontend.input", {"request_id", "prompt"})
    with trace.scope("server", request_id="r"):
        assert processor.process_inputs("r", {"prompt_token_ids": [1, 2]}) is returned
    records = {row["event"]: row for row in rows(trace)}
    assert records["frontend.input.before"]["prompt_token_snapshot"]["ids"] == [1, 2]
    assert records["frontend.input.result"]["prompt_token_snapshot"]["ids"] == [7, 8, 9]


def test_real_ar_strategy_observes_original_completion_without_changing_generation(trace):
    path = ROOT / "verl_omni/workers/rollout/vllm_rollout/vllm_omni_ar_strategy.py"
    cls = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "ARStrategy"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name in ("run_generation", "process_output")
    ]
    unit = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods],
        type_ignores=[],
    )
    namespace = {"trace": trace, "TokenOutput": SimpleNamespace}
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
    completion = SimpleNamespace(token_ids=[11, 12], text="B", finish_reason="stop")
    final = SimpleNamespace(outputs=[completion])
    received = []
    params = SimpleNamespace(max_tokens=1024, temperature=1.0, logprobs=None)
    prompt = {"prompt_token_ids": [1, 2, 3]}

    class Engine:
        def generate(self, **kwargs):
            received.append(kwargs)

            async def output():
                yield final

            return output()

    async def collect(generator):
        async for item in generator:
            return item

    strategy = SimpleNamespace(
        server=SimpleNamespace(engine=Engine(), global_steps=0),
        _rollout_output_modalities=None,
        _collect_last_output=collect,
        _map_stop_reason=lambda value: "completed",
        _extract_num_preempted=lambda _: 0,
    )

    async def run():
        with trace.scope("server", request_id="r"):
            result = await namespace["run_generation"](strategy, prompt, params, "r", None, 0)
            assert result is final
            return namespace["process_output"](strategy, result, params, {})

    output = asyncio.run(run())
    assert received[0]["prompt"] is prompt
    assert received[0]["sampling_params_list"] is params
    assert output.token_ids is completion.token_ids
    assert completion.text == "B"
    events = {row["event"]: row for row in rows(trace)}
    assert events["strategy.submit"]["sampling_params"]["temperature"] == 1.0
    assert events["strategy.result"]["response_token_snapshot"]["ids"] == [11, 12]
    assert events["strategy.result"]["engine_text"]["head"] == "B"


def test_agent_proxy_rpc_identity_empty_result_and_merge(trace, monkeypatch):
    utils = ModuleType("verl_omni.utils")
    utils.video_trace = trace
    base_module = ModuleType("verl.experimental.agent_loop.agent_loop")
    base_module.register = lambda name: lambda cls: cls
    single_module = ModuleType("verl.experimental.agent_loop.single_turn_agent_loop")
    received = []
    final = SimpleNamespace(prompt_ids=[1], response_ids=[], response_mask=[])

    class Client:
        async def generate(self, request_id, **kwargs):
            received.append(kwargs)
            incoming = kwargs.pop("video_trace_context")

            async def remote():
                with trace.scope("server", incoming=incoming, request_id="rewritten-id"):
                    trace.event("server.receive")
                    return SimpleNamespace(token_ids=[], log_probs=[], stop_reason="completed")

            return await asyncio.create_task(remote(), context=contextvars.Context())

    class SingleTurn:
        def __init__(self, server_manager):
            self.server_manager = server_manager

        async def run(self, sampling_params, **kwargs):
            prompt_ids = await self.ct_build_initial_tokens(kwargs.get("raw_prompt", []))
            result = await self.server_manager.generate("agent-id", prompt_ids=prompt_ids, video_data=[np.zeros(1)])
            await self.ct_merge_assistant_token([1], result.token_ids, [])
            return final

        async def ct_merge_assistant_token(self, *args):
            return SimpleNamespace(token_ids=[1]), [], None

        async def ct_build_initial_tokens(self, messages):
            return [1]

    single_module.SingleTurnAgentLoop = SingleTurn
    monkeypatch.setitem(sys.modules, "verl_omni.utils", utils)
    monkeypatch.setitem(sys.modules, base_module.__name__, base_module)
    monkeypatch.setitem(sys.modules, single_module.__name__, single_module)
    agent_module = load("agent_test", "verl_omni/agent_loop/video_trace_agent_loop.py")
    agent = agent_module.VideoTraceSingleTurnAgentLoop(server_manager=Client())
    assert (
        asyncio.run(
            agent.run(
                {},
                uid="dataset-uid",
                session_id=0,
                raw_prompt=[{"role": "system", "content": "Use <answer>A</answer>"}],
                extra_info={"problem_id": "video-1-question-2"},
            )
        )
        is final
    )
    events = rows(trace)
    assert {row["trace_id"] for row in events} == {events[0]["trace_id"]}
    assert all(row["uid"] == "dataset-uid" for row in events)
    assert all(row["sample_key"] == "video-1-question-2" for row in events)
    assert next(row for row in events if row["event"] == "agent.source.message")["text"]["has_answer_pair"]
    assert next(row for row in events if row["event"] == "agent.prompt")["prompt_token_snapshot"]["ids"] == [1]
    assert next(row for row in events if row["event"] == "agent.result")["token_ids_count"] == 0
    assert next(row for row in events if row["event"] == "agent.merge.after")["response_mask_count"] == 0
    assert next(row for row in events if row["event"] == "agent.output")["response_ids_count"] == 0


def test_torch_cpu_bfloat16_and_scalar(trace, monkeypatch):
    torch = pytest.importorskip("torch")
    for mode in ("sample", "full"):
        monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_MODE", mode)
        for tensor in (torch.arange(24, dtype=torch.bfloat16).reshape(4, 6).T, torch.tensor(1.0)):
            snapshot = trace._array(tensor, mode)
            assert snapshot["digest_status"] == mode
            assert isinstance(snapshot["fingerprint"].raw, bytes)
