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
"""Optional instance-local observers, installed after the real engine starts."""

import functools
import inspect
import os
from collections.abc import Mapping

from verl_omni.utils import video_trace as trace


def _observe(name, arguments, result=None, after=False):
    value = arguments.get("prompt", arguments.get("parsed_content"))
    trace.prompt(
        name + (".after" if after else ".before"),
        value or {},
        stage_id=arguments.get("stage_id"),
        replica_id=arguments.get("replica_id"),
    )
    if not after:
        return
    if isinstance(result, Mapping):
        request = result if "prompt_token_ids" in result else result.get("prompt", result)
    else:
        request = result if getattr(result, "prompt_token_ids", None) is not None else getattr(result, "prompt", result)
    result_prompt = (
        request if isinstance(request, Mapping) else {"prompt_token_ids": getattr(request, "prompt_token_ids", None)}
    )
    trace.prompt(name + ".result", result_prompt)
    if isinstance(result, Mapping):
        kwargs = result.get("mm_kwargs")
        trace.event(
            name + ".features",
            video_kwargs=kwargs.get("video") if isinstance(kwargs, Mapping) else None,
            mm_hashes=result.get("mm_hashes"),
        )
    else:
        features = getattr(request, "mm_features", None)
        if features is not None:
            trace.event(
                name + ".features",
                core_request_id=getattr(request, "request_id", None),
                video_features=[
                    {key: getattr(item, key, None) for key in ("identifier", "mm_hash", "data")}
                    for item in features
                    if getattr(item, "modality", None) == "video"
                ],
            )


def _attach(owner, method_name, event_name, required):
    original = getattr(owner, method_name)
    if getattr(original, "_video_trace_v2", False):
        return
    signature = inspect.signature(original)
    if inspect.iscoroutinefunction(original) or not required.issubset(signature.parameters):
        raise TypeError(f"Unsupported signature for {method_name}: {signature}")

    @functools.wraps(original)
    def observed(*args, **kwargs):
        if not trace.enabled() or not trace.export_context().get("selected"):
            return original(*args, **kwargs)
        try:
            bound = signature.bind(*args, **kwargs).arguments
        except Exception:
            return original(*args, **kwargs)

        def observe(result=None, after=False):
            try:
                _observe(event_name, bound, result, after)
            except Exception as exc:
                trace.event("trace.observation_error", boundary=event_name, error_type=type(exc).__name__)

        observe()
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            trace.event(event_name + ".error", error_type=type(exc).__name__)
            raise
        observe(result, after=True)
        return result

    observed._video_trace_v2 = True
    # Assign the bound-method closure to this object, never to its class.
    setattr(owner, method_name, observed)


def install(engine_client):
    """Opt into frontend observations without modifying global vLLM classes."""
    if not trace.enabled() or os.environ.get("VERL_OMNI_VIDEO_TRACE_STAGE", "boundary") != "frontend":
        return
    engine = getattr(engine_client, "engine", None)
    processor = getattr(engine, "input_processor", None)
    preprocessor = getattr(processor, "input_preprocessor", None)
    specs = (
        (engine, "_build_add_request_message", "frontend.build", {"request_id", "prompt"}),
        (engine, "_ensure_stage_replica_mm_uuids", "frontend.uuids", {"prompt", "stage_id", "replica_id"}),
        (processor, "process_inputs", "frontend.input", {"request_id", "prompt"}),
        (preprocessor, "_process_tokens", "frontend.tokens", {"parsed_content"}),
    )
    for owner, method, event_name, required in specs:
        try:
            _attach(owner, method, event_name, required)
            trace.event("trace.install", force=True, boundary=event_name, status="installed")
        except Exception as exc:
            trace.event("trace.install", force=True, boundary=event_name, status="unavailable", error=str(exc))
