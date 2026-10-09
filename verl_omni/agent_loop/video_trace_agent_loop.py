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
"""Explicitly selected diagnostic agent; never patches upstream client classes."""

from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

from verl_omni.utils import video_trace as trace


class _ObservedClient:
    def __init__(self, delegate):
        self.delegate = delegate

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    async def generate(self, request_id, **kwargs):
        if not trace.enabled():
            return await self.delegate.generate(request_id=request_id, **kwargs)
        with trace.scope("agent.call", agent_request_id=request_id, call_id=uuid4().hex):
            trace.prompt(
                "agent.dispatch",
                {
                    "prompt_ids": kwargs.get("prompt_ids"),
                    "multi_modal_data": {"video": kwargs.get("video_data"), "audio": kwargs.get("audio_data")},
                    "mm_processor_kwargs": kwargs.get("mm_processor_kwargs"),
                },
                sampling_params=trace.sampling(kwargs.get("sampling_params")),
            )
            # Current verl clients forward extra kwargs through each resume RPC.
            # This is correlation metadata only; the server consumes it locally.
            result = await self.delegate.generate(
                request_id=request_id,
                video_trace_context=trace.export_context(),
                **kwargs,
            )
            trace.output("agent.result", result)
            return result


@register("video_trace_single_turn_agent")
class VideoTraceSingleTurnAgentLoop(SingleTurnAgentLoop):
    """Run the upstream thinker loop unchanged, observing its explicit boundaries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if trace.enabled():
            self.server_manager = _ObservedClient(self.server_manager)

    async def run(self, sampling_params, **kwargs):
        """Keep dataset uid/session attached through dispatch and token merging."""
        extra = kwargs.get("extra_info")
        extra = extra if isinstance(extra, dict) else {}
        with trace.scope(
            "agent",
            uid=kwargs.get("uid"),
            session_id=kwargs.get("session_id"),
            sample_index=extra.get("index", kwargs.get("index")),
            sample_key=extra.get("problem_id"),
            video_id=extra.get("video_id"),
            question_id=extra.get("qid"),
        ):
            trace.agent_config(self)
            trace.messages("agent.source.message", kwargs.get("raw_prompt", []))
            result = await super().run(sampling_params, **kwargs)
            trace.output("agent.output", result)
            return result

    async def ct_build_initial_tokens(self, messages, *args, **kwargs):
        trace.messages("agent.template.message", messages)
        result = await super().ct_build_initial_tokens(messages, *args, **kwargs)
        trace.prompt("agent.prompt", {"prompt_ids": result})
        return result

    async def ct_merge_assistant_token(
        self, runtime_token_ids, assistant_token_ids, response_mask, response_logprobs=None, assistant_logprobs=None
    ):
        """Observe alignment before truncation without changing the merge result."""
        trace.event(
            "agent.merge.before",
            token_fields={"response_token_snapshot": assistant_token_ids},
            runtime_tokens=len(runtime_token_ids),
            assistant_tokens=len(assistant_token_ids),
            response_mask_count=len(response_mask),
        )
        result = await super().ct_merge_assistant_token(
            runtime_token_ids,
            assistant_token_ids,
            response_mask,
            response_logprobs,
            assistant_logprobs,
        )
        merge_result, mask, logprobs = result
        trace.merged_output(merge_result, mask, logprobs)
        return result
