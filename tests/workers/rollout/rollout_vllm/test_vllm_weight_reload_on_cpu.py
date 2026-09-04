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

from types import SimpleNamespace
from unittest.mock import sentinel

import pytest
from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

from verl_omni.workers.rollout.vllm_rollout import npu_utils as npu_utils_module
from verl_omni.workers.rollout.vllm_rollout import utils as utils_module


def _make_worker(model, model_config):
    return SimpleNamespace(
        device="cpu",
        _pending_lora_peft_config=None,
        _get_zmq_handle=lambda: "ipc:///tmp/test.sock",
        _get_standard_weight_model_and_config=lambda: (model, model_config),
    )


def _patch_standard_reload_dependencies(monkeypatch, events):
    patch_module = __import__("verl.utils.vllm.patch", fromlist=["patch_vllm_moe_model_weight_loader"])
    loader_utils = __import__("vllm.model_executor.model_loader.utils", fromlist=["process_weights_after_loading"])
    monkeypatch.setattr(
        patch_module,
        "patch_vllm_moe_model_weight_loader",
        lambda model: events.append("patch_moe_loader"),
    )
    monkeypatch.setattr(
        loader_utils,
        "process_weights_after_loading",
        lambda model, config, device: events.append("process_after_loading"),
    )


def test_standard_vllm_bucket_reload_order_on_cpu(monkeypatch):
    events = []
    buckets = [sentinel.bucket_0, sentinel.bucket_1]
    model_config = sentinel.model_config
    model = SimpleNamespace(load_weights=lambda weights: events.append(("load", weights)))

    class FakeReceiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            for bucket in buckets:
                on_bucket_received(bucket)

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", FakeReceiver)
    _patch_standard_reload_dependencies(monkeypatch, events)
    monkeypatch.setattr(npu_utils_module, "_is_npu_platform", lambda: False)

    utils_module.vLLMOmniColocateWorkerExtension.update_weights_from_ipc(
        _make_worker(model, model_config)
    )

    assert events == [
        "patch_moe_loader",
        ("load", sentinel.bucket_0),
        ("load", sentinel.bucket_1),
        "process_after_loading",
    ]


def test_npu_reload_restores_moe_layout_before_receive(monkeypatch):
    events = []
    model = SimpleNamespace(load_weights=lambda weights: events.append(("load", weights)))
    model_config = SimpleNamespace(hf_text_config=SimpleNamespace(hidden_size=4096))

    class FakeReceiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received(sentinel.bucket)

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", FakeReceiver)
    _patch_standard_reload_dependencies(monkeypatch, events)
    monkeypatch.setattr(npu_utils_module, "_is_npu_platform", lambda: True)
    monkeypatch.setattr(
        npu_utils_module,
        "restore_moe_param_layout",
        lambda model, hidden_size: events.append(("restore_moe_layout", hidden_size)),
    )

    utils_module.vLLMOmniColocateWorkerExtension.update_weights_from_ipc(
        _make_worker(model, model_config)
    )

    assert events == [
        "patch_moe_loader",
        ("restore_moe_layout", 4096),
        ("load", sentinel.bucket),
        "process_after_loading",
    ]


@pytest.mark.parametrize("failure_site", ["receive", "load"])
def test_standard_vllm_reload_propagates_failures(monkeypatch, failure_site):
    events = []
    original_error = RuntimeError(f"{failure_site} failed")

    def load_weights(weights):
        events.append("load")
        if failure_site == "load":
            raise original_error

    model = SimpleNamespace(load_weights=load_weights)

    class FakeReceiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            events.append("receive")
            if failure_site == "receive":
                raise original_error
            on_bucket_received(sentinel.bucket)

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", FakeReceiver)
    _patch_standard_reload_dependencies(monkeypatch, events)
    monkeypatch.setattr(npu_utils_module, "_is_npu_platform", lambda: False)

    with pytest.raises(RuntimeError) as exc_info:
        utils_module.vLLMOmniColocateWorkerExtension.update_weights_from_ipc(
            _make_worker(model, sentinel.model_config)
        )

    assert exc_info.value is original_error
    assert "process_after_loading" not in events
