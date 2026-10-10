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

import ast
import asyncio
import json
import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tests.utils import test_video_trace_on_cpu as helpers

frontend = helpers.frontend
trace = helpers.trace
load = helpers.load
rows = helpers.rows


@pytest.fixture
def worker_module(trace, frontend, monkeypatch):
    monkeypatch.setitem(sys.modules, "verl_omni.utils.video_trace_frontend", frontend)
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_STAGE", "worker")
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_MODE", "sample")
    monkeypatch.delenv("VERL_OMNI_VIDEO_TRACE_DEVICE_SAMPLE", raising=False)
    return load("worker_test", "verl_omni/utils/video_trace_worker.py")


def feature(modality, index):
    return NS(
        modality=modality,
        identifier=f"cache-{index}",
        data={"pixels": np.arange(4) + index},
        mm_position=NS(offset=index, length=1, is_embed=None),
    )


def test_startup_options_enable_memory_before_request_registration(worker_module, trace, monkeypatch, tmp_path):
    monkeypatch.delenv("VERL_OMNI_VIDEO_TRACE_DIR")
    worker = NS(model_runner=Runner(), rank=0, wake_up=lambda tags=None: tags)
    worker_module.configure_worker(
        {
            "VERL_OMNI_VIDEO_TRACE_DIR": str(tmp_path),
            "VERL_OMNI_VIDEO_TRACE_STAGE": "worker",
            "UNRELATED_VIDEO_TRACE_TEST_OPTION": "must not propagate",
        }
    )
    worker_module.install(worker)
    assert "UNRELATED_VIDEO_TRACE_TEST_OPTION" not in os.environ
    assert not worker._video_observer.pending
    assert worker.wake_up(tags=["weights"]) == ["weights"]
    rows = [json.loads(line) for line in trace._memory_recorder.path.read_text().splitlines()]
    assert rows[-2]["event"] == "wake_up.before"
    assert rows[-2]["samples_started"] == 0


class Runner:
    def __init__(self):
        self.requests = {}
        self.input_batch = NS(req_ids=[])
        self.cache = {}
        self.calls = []
        self.result = (None, np.arange(18).reshape(6, 3), np.arange(6).reshape(1, 6), None, {}, None)
        self.model_config = NS(seed=44, model="qwen", dtype="bf16")

    def _update_states(self, scheduler_output):
        for request in scheduler_output.scheduled_new_reqs:
            self.requests[request.req_id] = request
        self.input_batch.req_ids = list(scheduler_output.num_scheduled_tokens)

    def _cache_encoder_output(self, mm_hash, output):
        self.calls.append(("write", mm_hash))
        self.cache[mm_hash] = output

    def _get_encoder_output_from_cache(self, mm_hash):
        self.calls.append(("read", mm_hash))
        return self.cache.get(mm_hash)

    def _gather_mm_embeddings(self, scheduler_output):
        self.calls.append(("gather",))
        for request in self.requests.values():
            for item in request.mm_features:
                self._get_encoder_output_from_cache(item.identifier)
        return list(self.cache.values()), np.array([False, True, False, True, True, False])

    def _preprocess(self, scheduler_output):
        for request in self.requests.values():
            for item in request.mm_features:
                if item.identifier not in self.cache:
                    self._cache_encoder_output(item.identifier, np.ones((1, 3)))
        self._gather_mm_embeddings(scheduler_output)
        return self.result


def request(req_id, global_id, features):
    return NS(
        req_id=req_id,
        prompt_token_ids=[10, 11, 12],
        num_computed_tokens=0,
        mm_features=features,
        sampling_params=NS(temperature=1, seed=None),
        additional_information={"global_request_id": [global_id]},
    )


def setup(worker_module, trace):
    runner = Runner()
    worker = NS(model_runner=runner, rank=0, local_rank=0)
    worker_module.install(worker)
    with trace.scope("agent", sample_key="6806999702_8", sample_index=8, video_id="6806999702", question_id=8):
        context = trace.export_context()
    worker._video_observer.register("external", context)
    reqs = [
        request("other-internal", "other", [feature("video", 9)]),
        request("random-internal", "external", [feature("audio", 0), feature("video", 1)]),
    ]
    scheduler = NS(scheduled_new_reqs=reqs, num_scheduled_tokens={r.req_id: 3 for r in reqs}, finished_req_ids=[])
    return worker, scheduler, context


def test_exact_global_id_selected_after_batch_reordering_and_original_results(worker_module, trace):
    worker, scheduler, context = setup(worker_module, trace)
    runner = worker.model_runner
    plain = Runner()
    plain._update_states(scheduler)
    plain._preprocess(scheduler)
    runner._update_states(scheduler)
    assert runner._preprocess(scheduler) is runner.result
    assert runner.calls == plain.calls
    records = [r for r in rows(trace) if r["event"].startswith("worker.")]
    assert {r["trace_id"] for r in records} == {context["trace_id"]}
    assert {r["core_request_id"] for r in records} == {"random-internal"}
    received = next(r for r in records if r["event"] == "worker.receive")
    assert [f["modality"] for f in received["multimodal_features"]] == ["audio", "video"]
    assert received["runtime"]["model_config"]["seed"] == "44"
    merged = next(r for r in records if r["event"] == "worker.model_input")
    assert merged["batch_offset"] == 3
    assert merged["embedding"]["shape"] == [3, 3]
    expected = trace._finalize(trace._array(runner.result[1][3:6], "sample"))
    assert merged["embedding"]["sample_sha256"] == expected["sample_sha256"]
    assert {r["modality"] for r in records if r["event"] == "worker.encoder.read"} == {"audio", "video"}


def test_real_worker_events_can_be_compared_and_summarized(worker_module, trace, tmp_path):
    worker, scheduler, _ = setup(worker_module, trace)
    worker.model_runner._update_states(scheduler)
    worker.model_runner._preprocess(scheduler)
    observed = rows(trace)
    boundaries = [row for row in observed if row["event"] == "worker.observation.begin"]
    assert boundaries and all("worker_rank" not in row for row in boundaries)
    compare = load("worker_compare_test", "tests/special_e2e/compare_8039_request.py")
    summary = load("worker_summary_test", "tests/special_e2e/summarize_8039_worker.py")
    selected = compare.load_request([trace._sink.path], "6806999702_8")
    report = compare.compare(selected, selected)
    (tmp_path / "comparison.json").write_text(json.dumps(report), encoding="utf-8")
    for side in ("single", "multi"):
        (tmp_path / f"{side}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in selected["records"]), encoding="utf-8"
        )
    output = summary.summarize(tmp_path)
    assert "write_vs_read=sample_equal" in output
    assert "modality=['video']" in output
    assert "modality=['audio']" in output
    assert "worker.observation.begin" in output


def test_cache_hit_observed_without_claiming_encoder_ran(worker_module, trace):
    worker, scheduler, _ = setup(worker_module, trace)
    runner = worker.model_runner
    runner.cache = {f.identifier: np.ones((1, 3)) for r in scheduler.scheduled_new_reqs for f in r.mm_features}
    runner._update_states(scheduler)
    runner._preprocess(scheduler)
    records = rows(trace)
    assert any(r["event"] == "worker.encoder.read" for r in records)
    assert not any(r["event"] == "worker.encoder.write" for r in records)


def test_capture_budget_stops_device_sampling_but_not_inference(worker_module, trace, monkeypatch):
    worker, scheduler, _ = setup(worker_module, trace)
    worker.model_runner._update_states(scheduler)
    worker._video_observer.requests["random-internal"]["events"] = 256
    sampled = []
    monkeypatch.setattr(worker_module, "snapshot", lambda value: sampled.append(value))
    assert worker.model_runner._preprocess(scheduler) is worker.model_runner.result
    assert not sampled
    assert sum(r["event"] == "trace.worker_limit" for r in rows(trace)) == 1


def test_no_guessed_uuid_matching_and_finished_cleanup(worker_module, trace):
    worker, scheduler, _ = setup(worker_module, trace)
    scheduler.scheduled_new_reqs[1].additional_information = None
    scheduler.scheduled_new_reqs[1].req_id = "external-deadbeef"
    worker.model_runner._update_states(scheduler)
    assert not worker._video_observer.requests
    assert any(r["event"] == "trace.worker_unmatched" for r in rows(trace))
    scheduler.scheduled_new_reqs[1].additional_information = {"global_request_id": ["external"]}
    worker.model_runner._update_states(scheduler)
    scheduler.finished_req_ids = ["external-deadbeef"]
    scheduler.scheduled_new_reqs = []
    worker.model_runner._update_states(scheduler)
    assert not worker._video_observer.requests


def test_nonempty_mismatched_global_id_is_observable_and_bounded(worker_module, trace):
    worker, scheduler, _ = setup(worker_module, trace)
    target = scheduler.scheduled_new_reqs[1]
    target.additional_information = {"global_request_id": ["different-external"]}
    scheduler.scheduled_new_reqs = [target]
    for _ in range(70):
        worker.model_runner._update_states(scheduler)
    records = rows(trace)
    unmatched = [r for r in records if r["event"] == "trace.worker_unmatched"]
    assert len(unmatched) == 64
    assert unmatched[0]["global_id"] == "different-external"
    assert unmatched[0]["pending_request_ids"] == ["external"]
    assert unmatched[0]["core_request_id"] == "random-internal"
    assert sum(r["event"] == "trace.worker_hook_enter" for r in records) == 1
    assert sum(r["event"] == "trace.worker_unmatched_limit" for r in records) == 1
    assert not worker._video_observer.requests
    assert worker.model_runner.requests["random-internal"] is target


def test_observation_failure_preserves_result_and_business_exception(worker_module, trace, monkeypatch):
    worker, scheduler, _ = setup(worker_module, trace)
    worker.model_runner._update_states(scheduler)
    monkeypatch.setattr(worker_module, "snapshot", lambda _: (_ for _ in ()).throw(RuntimeError("capture failed")))
    assert worker.model_runner._preprocess(scheduler) is worker.model_runner.result
    assert not worker._video_observer.active
    scheduler.num_scheduled_tokens = {}
    # Failure in the actual implementation is not suppressed by observation.
    broken = Runner()

    def business(scheduler_output):
        raise ValueError("business failed")

    broken._preprocess = business
    instance = NS(model_runner=broken)
    worker_module.install(instance)
    with pytest.raises(ValueError, match="business failed"):
        broken._preprocess(scheduler)
    assert not instance._video_observer.active


def test_boundary_mode_no_patch_no_rpc_no_device_read(worker_module, trace, monkeypatch):
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_STAGE", "boundary")
    worker = NS(model_runner=Runner())
    before = worker.model_runner._preprocess
    worker_module.install(worker)
    assert worker.model_runner._preprocess == before
    with trace.scope("agent"):
        asyncio.run(worker_module.register(object(), "r"))

    class DeviceTensor:
        __module__ = "torch"
        device = "npu:0"

        def numel(self):
            raise AssertionError("must not read device data")

    tensor = DeviceTensor()
    assert worker_module.snapshot(tensor) is tensor


def test_registration_failure_does_not_change_request_or_cancel_generation(worker_module, trace):
    class Server:
        async def collective_rpc(self, **kwargs):
            assert kwargs["timeout"] == 10
            assert kwargs["args"][0] == "external"
            raise TimeoutError("unsupported worker")

    with trace.scope("agent"):
        asyncio.run(worker_module.register(Server(), "external"))
    assert next(r for r in rows(trace) if r["event"] == "trace.worker_registration")["status"] == "failed"


def test_worker_acknowledgement_reports_disabled_and_unavailable_hooks(worker_module, trace, monkeypatch):
    worker = NS(model_runner=Runner(), rank=0)
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_STAGE", "boundary")
    reply = worker_module.register_worker(worker, "external", {"selected": True})
    assert reply["status"] == "disabled"
    assert not hasattr(worker, "_video_observer")
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_STAGE", "worker")
    worker.model_runner._preprocess = lambda unexpected_signature: None
    reply = worker_module.register_worker(worker, "external", {"selected": True})
    assert reply["status"] == "registered"
    assert reply["hooks"]["_preprocess"]["status"] == "unavailable"
    assert reply["hooks"]["_update_states"]["status"] == "installed"


def test_registration_configures_worker_when_ray_did_not_forward_environment(worker_module, trace, monkeypatch):
    options = {name: os.environ.get(name, default) for name, default in worker_module._TRACE_DEFAULTS.items()}
    options["UNRELATED_GENERATION_SETTING"] = "must-not-be-applied"
    monkeypatch.delenv("VERL_OMNI_VIDEO_TRACE_DIR")
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_STAGE", "boundary")
    worker = NS(model_runner=Runner(), rank=0)
    reply = worker_module.register_worker(worker, "external", {"selected": True}, options)
    assert reply["status"] == "registered"
    assert reply["trace_stage"] == "worker"
    assert reply["device_sample"] == "0"
    assert reply["trace_file"]
    assert "UNRELATED_GENERATION_SETTING" not in os.environ
    assert worker._video_observer.pending["external"]["selected"]


def test_real_server_collective_rpc_preserves_worker_acknowledgements():
    path = helpers.ROOT / "verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py"
    method = next(
        n
        for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "collective_rpc"
    )
    unit = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
    replies = [[{"video_trace_ack": True, "status": "registered"}]]

    class Engine:
        async def collective_rpc(self, **kwargs):
            assert kwargs["stage_ids"] == [0]
            return replies

    server = NS(engine=Engine(), _generate_strategy=NS(collective_rpc_stage_ids=lambda _: [0]))
    assert asyncio.run(namespace["collective_rpc"](server, "register_video_trace")) is replies


def test_registration_retains_worker_replies_in_server_trace(worker_module, trace):
    class Server:
        async def collective_rpc(self, **kwargs):
            return [[{"video_trace_ack": True, "status": "disabled", "trace_stage": "boundary"}]]

    with trace.scope("agent"):
        asyncio.run(worker_module.register(Server(), "external"))
    record = next(r for r in rows(trace) if r["event"] == "trace.worker_registration")
    assert record["status"] == "returned"
    assert record["worker_replies"][0][0]["status"] == "disabled"


@pytest.mark.parametrize("omni_suffix", ["", "-omni1234"])
def test_actual_ar_generation_keeps_params_and_output_with_worker_registration(
    worker_module, trace, monkeypatch, omni_suffix
):
    monkeypatch.setitem(sys.modules, "verl_omni.utils.video_trace_worker", worker_module)
    path = helpers.ROOT / "verl_omni/workers/rollout/vllm_rollout/vllm_omni_ar_strategy.py"
    cls = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "ARStrategy"
    )
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_generation")
    unit = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = {"trace": trace, "os": os}
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
    calls = []
    completion = NS(token_ids=[10, 20])
    prompt, params = {"prompt_token_ids": [1, 2]}, NS(max_tokens=1024)
    worker = NS(model_runner=Runner(), rank=0, local_rank=0)
    worker_module.install(worker)

    class AdmissionEngine:
        async def add_request_async(self, request_id, prompt, sampling_params_list):
            calls.append("admit")
            assert prompt is expected_prompt
            assert sampling_params_list is params
            # AsyncOmni rewrites the external ID, and InputProcessor may
            # append a second suffix. global_request_id retains the first.
            core_id = request_id + "-core5678"
            req = request(core_id, request_id, [feature("video", 0)])
            scheduler = NS(scheduled_new_reqs=[req], num_scheduled_tokens={core_id: 3}, finished_req_ids=[])
            worker.model_runner._update_states(scheduler)
            worker.model_runner._preprocess(scheduler)

    expected_prompt = prompt

    class Engine:
        engine = AdmissionEngine()

        def generate(self, **kwargs):
            calls.append("generate")
            assert kwargs["prompt"] is prompt
            assert kwargs["sampling_params_list"] is params
            assert kwargs["request_id"] == "external"

            async def outputs():
                await self.engine.add_request_async(
                    request_id=kwargs["request_id"] + omni_suffix,
                    prompt=kwargs["prompt"],
                    sampling_params_list=kwargs["sampling_params_list"],
                )
                yield completion

            return outputs()

    class Server:
        engine = Engine()

        async def collective_rpc(self, **kwargs):
            calls.append("register")
            assert kwargs["args"][1]["sample_key"] == "6806999702_8"
            return [worker_module.register_worker(worker, *kwargs["args"], **kwargs.get("kwargs", {}))]

    async def collect(generator):
        async for item in generator:
            return item

    strategy = NS(server=Server(), _rollout_output_modalities=None, _collect_last_output=collect)
    worker_module.install_client(strategy.server)
    with trace.scope("agent", sample_key="6806999702_8"):
        assert asyncio.run(namespace["run_generation"](strategy, prompt, params, "external", None, 0)) is completion
    records = rows(trace)
    received = next(r for r in records if r["event"] == "worker.receive")
    assert received["global_id"] == "external" + omni_suffix
    assert received["core_request_id"] == "external" + omni_suffix + "-core5678"
    assert any(r["event"] == "worker.model_input" for r in records)
    registered = next(r for r in records if r["event"] == "trace.worker_registration")
    assert registered["registered_request_id"] == "external" + omni_suffix
    assert calls == ["generate", "register", "admit"]


def test_admission_hook_is_instance_local_and_preserves_concurrent_context(worker_module, trace):
    admitted, registered = [], []

    class Admission:
        async def add_request_async(self, request_id, payload):
            await asyncio.sleep(0)
            admitted.append((request_id, payload))
            if request_id == "business-error":
                raise ValueError("original admission error")
            return payload

    class Server:
        def __init__(self):
            self.engine = NS(engine=Admission())

        async def collective_rpc(self, **kwargs):
            await asyncio.sleep(0)
            request_id, context = kwargs["args"]
            registered.append((request_id, context["sample_key"]))
            return []

    server, other = Server(), Server()
    original_other = other.engine.engine.add_request_async
    worker_module.install_client(server)
    installed = server.engine.engine.add_request_async
    worker_module.install_client(server)
    assert installed is server.engine.engine.add_request_async
    assert other.engine.engine.add_request_async == original_other

    async def run(request_id, key, selected):
        payload = object()
        with trace.scope("agent", sample_key=key, selected=selected):
            assert await installed(request_id, payload) is payload

    async def together():
        await asyncio.gather(
            run("actual-1", "sample-1", True), run("actual-2", "sample-2", True), run("unselected", "sample-3", False)
        )

    asyncio.run(together())
    assert set(registered) == {("actual-1", "sample-1"), ("actual-2", "sample-2")}
    assert len(admitted) == 3
    with pytest.raises(ValueError, match="original admission error"):
        asyncio.run(run("business-error", "sample-1", True))


def test_admission_registration_failure_preserves_business_output(worker_module, trace):
    payload = object()

    class Admission:
        async def add_request_async(self, request_id):
            assert request_id == "actual-internal-id"
            return payload

    class Server:
        engine = NS(engine=Admission())

        async def collective_rpc(self, **kwargs):
            raise RuntimeError("diagnostic rpc unavailable")

    server = Server()
    worker_module.install_client(server)
    with trace.scope("agent"):
        assert asyncio.run(server.engine.engine.add_request_async("actual-internal-id")) is payload
    assert next(r for r in rows(trace) if r["event"] == "trace.worker_registration")["status"] == "failed"


def test_unsupported_admission_reports_unavailable_without_replacing_method(worker_module, trace):
    admission = NS(add_request_async=lambda request_id: request_id)
    original = admission.add_request_async
    worker_module.install_client(NS(engine=NS(engine=admission)))
    assert admission.add_request_async is original
    record = next(r for r in rows(trace) if r.get("boundary") == "frontend.worker_admission")
    assert record["status"] == "unavailable"
