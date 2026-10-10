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
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.utils import test_video_trace_on_cpu as helpers

ROOT, load, trace = helpers.ROOT, helpers.load, helpers.trace


def records(trace):
    return [json.loads(line) for line in trace._memory_recorder.path.read_text().splitlines()]


@pytest.fixture
def memory(trace, monkeypatch):
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_STAGE", "worker")
    npu = NS(
        is_initialized=lambda: True,
        current_device=lambda: 3,
        memory_allocated=lambda device: 100,
        memory_reserved=lambda device: 200,
        max_memory_allocated=lambda device: 150,
        max_memory_reserved=lambda device: 250,
        mem_get_info=lambda device: (500, 1000),
    )
    # This fake exposes only metadata queries: allocation/sync/cache calls fail.
    monkeypatch.setitem(sys.modules, "torch", NS(npu=npu))
    return trace


def test_lifecycle_counters_synchronous_records_and_original_results(memory):
    sentinel = object()
    worker = NS(rank=2, device="npu:3", sleep=lambda level=1: sentinel)

    def wake_up(tags=None):
        assert records(memory)[-1]["event"] == "wake_up.before"
        assert records(memory)[-1]["samples_started"] == 0
        assert tags == ["weights"]
        return sentinel

    worker.wake_up = wake_up
    memory.install_memory(worker)
    original_wrapper = worker.wake_up
    memory.install_memory(worker)
    assert worker.wake_up is original_wrapper
    assert worker.sleep(level=1) is sentinel
    assert worker.wake_up(tags=["weights"]) is sentinel
    value = NS(shape=(2304, 8192), device="npu:3", dtype="bfloat16")
    with memory.device_sample(value):
        assert records(memory)[-1]["event"] == "sample.first.before"
    memory.memory_event("preprocess.exit")
    row = records(memory)[-1]
    assert row["samples_started"] == row["samples_completed"] == 1
    assert row["last_sample"]["shape"] == [2304, 8192]
    assert row["wake_calls"] == 1
    assert row["memory"]["memory_reserved"] == 200
    assert row["memory"]["device_index"] == 3
    assert memory._sink is None


def test_errors_preserved_and_write_failure_does_not_break_worker(memory, monkeypatch, capsys):
    error = RuntimeError("original failure")

    def fail(*args, **kwargs):
        raise error

    worker = NS(rank=0, wake_up=fail)
    memory.install_memory(worker)
    with pytest.raises(RuntimeError) as caught:
        worker.wake_up()
    assert caught.value is error
    assert records(memory)[-1]["event"] == "wake_up.error"
    with pytest.raises(RuntimeError) as caught:
        with memory.device_sample(NS(shape=(2,), device="npu:0", dtype="float32")):
            raise error
    assert caught.value is error
    assert records(memory)[-1]["samples_completed"] == 0
    monkeypatch.setattr(Path, "open", fail)
    with pytest.raises(RuntimeError) as caught:
        worker.wake_up()
    assert caught.value is error
    assert '"write_error": "RuntimeError"' in capsys.readouterr().err


def test_disabled_or_uninitialized_never_initializes_device(memory, monkeypatch, tmp_path):
    monkeypatch.delenv("VERL_OMNI_VIDEO_TRACE_DIR")
    worker = NS(wake_up=lambda: 12)
    original = worker.wake_up
    memory.install_memory(worker)
    assert worker.wake_up is original
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv("VERL_OMNI_VIDEO_TRACE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", NS(npu=NS(is_initialized=lambda: False)))
    memory.install_memory(worker)
    assert worker.wake_up() == 12
    assert records(memory)[-1]["memory"] == {"status": "npu_not_initialized"}


def test_native_exit_leaves_before_record_and_summary_needs_no_request(tmp_path):
    script = """
import importlib.util, os
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('trace', os.environ['TRACE_MODULE'])
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)
worker = SimpleNamespace(rank=0, wake_up=lambda: os._exit(17))
trace.install_memory(worker)
worker.wake_up()
"""
    env = dict(
        os.environ,
        VERL_OMNI_VIDEO_TRACE_DIR=str(tmp_path),
        VERL_OMNI_VIDEO_TRACE_STAGE="worker",
        TRACE_MODULE=str(ROOT / "verl_omni/utils/video_trace.py"),
    )
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, timeout=20)
    assert result.returncode == 17
    assert b"wake_up.before" in result.stderr
    module = load("memory_summary_test", "tests/special_e2e/summarize_8039_worker.py")
    summary = module.memory_health([tmp_path])
    assert "INCOMPLETE_WAKE calls=1 samples_started=0" in summary
    assert "No traced device sampling preceded" in summary
    assert "NO_MEMORY_FILES" in module.memory_health([])


def test_summary_distinguishes_later_wake_and_partial_line(memory):
    worker = NS(wake_up=lambda: None)
    memory.install_memory(worker)
    worker.wake_up()
    with memory.device_sample(NS(shape=(2,), device="npu:0", dtype="float32")):
        pass
    memory._memory_recorder.wake_calls += 1
    memory.memory_event("wake_up.before")
    with memory._memory_recorder.path.open("a") as stream:
        stream.write('{"partial":')
    module = load("memory_summary_test", "tests/special_e2e/summarize_8039_worker.py")
    summary = module.memory_health([memory._memory_recorder.path])
    assert "INCOMPLETE_WAKE calls=2 samples_started=1" in summary
    assert "invalid_or_partial_lines=1" in summary
    assert "No traced device sampling preceded" not in summary
