# Copyright 2026 verl-omni contributors
# SPDX-License-Identifier: Apache-2.0
"""Weight-free frontend probe for vllm-project/vllm-omni#8039.

Run with --help; see repro_vllm_omni_8039.md for scope and interpretation.
Imports of the inference stack are deferred so --self-test works without it.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.metadata
import json
import os
import queue
import sys
import traceback
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace


class PayloadMismatch(RuntimeError):
    """A measured payload invariant failed, rather than an environment setup."""


def tensor_hash(tensor):
    import torch

    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{tensor.dtype}:{tuple(tensor.shape)}".encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def raw_hash(prompt):
    frames, metadata = prompt["multi_modal_data"]["video"][0]
    return hashlib.sha256((tensor_hash(frames) + json.dumps(metadata, sort_keys=True)).encode()).hexdigest()


def make_prompt(index, frames, size):
    import numpy as np
    import torch

    generator = torch.Generator().manual_seed(8039 + index)
    video = torch.randint(0, 256, (frames, 3, size, size), dtype=torch.uint8, generator=generator)
    # Same shape/metadata and different bytes catch content-key collisions.
    metadata = {
        "fps": 2.0,
        "duration": frames / 2.0,
        "total_num_frames": frames,
        "frames_indices": list(range(frames)),
        "video_backend": "synthetic",
        "do_sample_frames": False,
    }
    audio = (0.1 * np.sin(2 * np.pi * (220 + index * 11) * np.arange(16000) / 16000)).astype(np.float32)
    return {
        "multi_modal_data": {"video": [(video, metadata)], "audio": [audio]},
        "mm_processor_kwargs": {"use_audio_in_video": False, "sampling_rate": 16000},
    }


def check_records(records, baseline, expected_ids):
    """Return concrete violations, including missing/duplicate requests."""
    errors = []
    seen = set()
    for record in records:
        request_id = record["request_id"]
        if request_id in seen:
            errors.append(f"duplicate request: {request_id}")
        seen.add(request_id)
        expected = baseline[record["sample"]]
        for field in ("entry_hash", "preprocess_hash"):
            if record[field] != expected["entry_hash"]:
                errors.append(f"{request_id}: {field} differs from submitted video")
        if not record["video_hash"] or record["video_hash"] != expected["video_hash"]:
            errors.append(f"{request_id}: pixel_values_videos differs from uncached serial reference")
        if record["prompt_hash"] != expected["prompt_hash"]:
            errors.append(f"{request_id}: processed prompt tokens differ from reference")
    if seen != set(expected_ids):
        missing, extra = set(expected_ids) - seen, seen - set(expected_ids)
        errors.append(f"request coverage mismatch: missing={missing}, extra={extra}")
    return errors


def self_test():
    reference = {0: {"entry_hash": "raw0", "video_hash": "px0", "prompt_hash": "tokens0"}}
    healthy = {
        "request_id": "r0",
        "sample": 0,
        "entry_hash": "raw0",
        "preprocess_hash": "raw0",
        "video_hash": "px0",
        "prompt_hash": "tokens0",
    }
    assert not check_records([healthy], reference, ["r0"])
    for field in ("entry_hash", "preprocess_hash", "video_hash", "prompt_hash"):
        assert check_records([dict(healthy, **{field: "other-request"})], reference, ["r0"])
    assert check_records([], reference, ["r0"])
    assert check_records([healthy, healthy], reference, ["r0"])
    print("PASS: detector accepts intact records and rejects corruption, missing and duplicate requests")


class Receiver:
    """Real P1 LRU cache, without an EngineCore or model executor."""

    def __init__(self, cache_gb):
        import ray
        from vllm.multimodal.cache import MultiModalReceiverCache

        self.node_id = ray.get_runtime_context().get_node_id()
        self.cache = MultiModalReceiverCache(
            SimpleNamespace(get_multimodal_config=lambda: SimpleNamespace(mm_processor_cache_gb=cache_gb))
        )

    def consume(self, request):
        video_hashes = []
        features = request.mm_features or []
        omitted = sum(feature.data is None for feature in features)
        for feature in self.cache.get_and_update_features(features):
            if feature.modality == "video":
                video_hashes.append(tensor_hash(feature.data["pixel_values_videos"].data))
        if len(video_hashes) != 1:
            raise RuntimeError(f"Expected one video feature, got {len(video_hashes)}")
        return {
            "video_hash": video_hashes[0],
            "receiver_node": self.node_id,
            "omitted_items": omitted,
            "prompt_hash": hashlib.sha256(json.dumps(request.prompt_token_ids).encode()).hexdigest(),
        }


class Frontend:
    """Run production stage-0 submission; replace only engine startup/dispatch."""

    def __init__(self, model, cache_gb, receivers):
        import torch
        from transformers import AutoProcessor
        from vllm import SamplingParams
        from vllm_omni.engine.arg_utils import OmniEngineArgs
        from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
        from vllm_omni.engine.stage_init_utils import build_stage0_input_processor
        from vllm_omni.engine.stage_pool import StagePool

        torch.set_num_threads(1)
        args = OmniEngineArgs(
            model=model,
            model_stage="thinker",
            model_arch="Qwen3OmniMoeForConditionalGeneration",
            max_model_len=8192,
            max_num_batched_tokens=8192,
            max_num_seqs=32,
            enforce_eager=True,
            enable_prefix_caching=False,
            mm_processor_cache_gb=cache_gb,
            mm_processor_cache_type="lru",
            limit_mm_per_prompt={"video": 1, "audio": 1},
        )
        config = args.create_engine_config()
        # No __init__: it would start the orchestrator and load model weights.
        engine = object.__new__(AsyncOmniEngine)
        engine.model = model
        engine.stage_metadata = [SimpleNamespace(stage_type="llm")]
        engine.stage_pools = [StagePool(0, [SimpleNamespace(stage_type="llm") for _ in receivers])]
        engine.input_processor = build_stage0_input_processor(config)
        engine.supported_tasks = ("generate",)
        engine.default_sampling_params_list = [SamplingParams(max_tokens=1, temperature=0)]
        engine.prompt_transform_func = None
        engine.prompt_expand_func = None
        engine.request_queue = SimpleNamespace(sync_q=queue.Queue())
        self.engine = engine
        self.receivers = receivers
        self.observed = {}
        self.current_request = ContextVar("request_id")
        preprocessor = engine.input_processor.input_preprocessor
        original = preprocessor._process_tokens

        def observe(parsed_content, *args, **kwargs):
            self.observed[self.current_request.get()] = raw_hash(parsed_content)
            return original(parsed_content, *args, **kwargs)

        preprocessor._process_tokens = observe
        processor = AutoProcessor.from_pretrained(model)
        text = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": "unused"},
                        {"type": "audio", "audio": "unused"},
                        {"type": "text", "text": "Describe the video and audio."},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.prompt_ids = processor.tokenizer.encode(text, add_special_tokens=False)

    async def wave(self, payloads, wave_id, concurrency, rotation=0):
        semaphore = asyncio.Semaphore(concurrency)

        async def submit(sample, payload):
            async with semaphore:
                request_id = f"{wave_id}-{sample}"
                prompt = copy.deepcopy(payload)
                prompt["prompt_token_ids"] = list(self.prompt_ids)
                entry_hash = raw_hash(prompt)
                self.current_request.set(request_id)
                await self.engine.add_request_async(request_id, prompt)
                message = self.engine.request_queue.sync_q.get_nowait()
                if message.request_id != request_id:
                    raise RuntimeError(f"Submission queue mixed {request_id} and {message.request_id}")
                pool = self.engine.stage_pools[0]
                replica = pool.get_bound_replica_id(request_id)
                replica = 0 if replica is None else replica
                video_features = [f for f in message.prompt.mm_features or [] if f.modality == "video"]
                if len(video_features) != 1:
                    raise PayloadMismatch(f"{request_id}: expected one video before P1")
                video = video_features[0]
                p0_video_hash = None if video.data is None else tensor_hash(video.data["pixel_values_videos"].data)
                try:
                    result = await self.receivers[replica].consume.remote(message.prompt)
                    return dict(
                        result,
                        request_id=request_id,
                        sample=sample,
                        entry_hash=entry_hash,
                        preprocess_hash=self.observed.pop(request_id),
                        replica=replica,
                        mm_identifier=video.identifier,
                        mm_hash=video.mm_hash,
                        p0_video_hash=p0_video_hash,
                    )
                finally:
                    pool.release_binding(request_id)

        order = [(i + rotation) % len(payloads) for i in range(len(payloads))]
        return await asyncio.gather(*(submit(i, payloads[i]) for i in order))


def run(args, report):
    import ray
    from ray.cluster_utils import Cluster
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # Keep the installed CUDA/Ascend platform importable; no model is instantiated.
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    cluster = Cluster()
    actors = []
    try:
        for _ in range(args.nodes):
            cluster.add_node(
                num_cpus=2,
                num_gpus=0,
                object_store_memory=128 * 1024 * 1024,
                include_dashboard=False,
            )
        ray.init(address=cluster.address, log_to_driver=True)
        nodes = sorted(node["NodeID"] for node in ray.nodes() if node["Alive"])
        if len(nodes) != args.nodes:
            raise RuntimeError(f"Expected {args.nodes} raylets, found {nodes}")
        head = ray.get_runtime_context().get_node_id()
        nodes = [head] + [node for node in nodes if node != head]
        report["node_ids"] = nodes
        print(f"Ray nodes: {nodes}", flush=True)
        payloads = [make_prompt(i, args.frames, args.size) for i in range(args.requests)]
        raw_hashes = [raw_hash(payload) for payload in payloads]
        if len(set(raw_hashes)) != args.requests:
            raise RuntimeError("Synthetic videos are not distinct")
        receiver_cls = ray.remote(num_cpus=0, max_restarts=0)(Receiver)
        frontend_cls = ray.remote(num_cpus=1, max_restarts=0)(Frontend)

        def setup(cache_gb):
            receivers = []
            for replica in range(args.replicas):
                receiver = receiver_cls.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[replica % len(nodes)], soft=False)
                ).remote(max(cache_gb, 0.01))
                actors.append(receiver)
                receivers.append(receiver)
            frontend = frontend_cls.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(head, soft=False)
            ).remote(args.model, cache_gb, receivers)
            actors.append(frontend)
            return frontend

        def cleanup():
            for actor in actors:
                ray.kill(actor)
            actors.clear()

        baseline_actor = setup(0)
        reference = ray.get(baseline_actor.wave.remote(payloads, "reference", 1), timeout=args.timeout)
        report["reference"] = reference
        baseline = {row["sample"]: row for row in reference}
        if len(baseline) != args.requests or len({row["video_hash"] for row in reference}) != args.requests:
            raise PayloadMismatch("Serial reference is incomplete or already collapses distinct videos")
        for sample, expected_raw in enumerate(raw_hashes):
            if baseline[sample]["entry_hash"] != expected_raw or baseline[sample]["preprocess_hash"] != expected_raw:
                raise PayloadMismatch("Serial reference input changed before preprocessing")
        cleanup()
        # Fresh P0/P1 caches for each concurrency setting; retain both across waves.
        for concurrency in dict.fromkeys((1, args.concurrency)):
            frontend = setup(args.cache_gb)
            for wave in range(args.waves):
                wave_id = f"c{concurrency}-w{wave}"
                records = ray.get(frontend.wave.remote(payloads, wave_id, concurrency, wave), timeout=args.timeout)
                errors = check_records(records, baseline, [f"{wave_id}-{i}" for i in range(args.requests)])
                receiver_nodes = {row["receiver_node"] for row in records}
                if receiver_nodes != set(nodes):
                    errors.append(f"Not all nodes received requests: {receiver_nodes}")
                report["cases"].append({"wave": wave_id, "records": records, "errors": errors})
                omitted = sum(row["omitted_items"] for row in records)
                print(f"{wave_id}: requests={len(records)} errors={len(errors)} omitted_items={omitted}", flush=True)
            cleanup()
        return 1 if any(case["errors"] for case in report["cases"]) else 0
    finally:
        for actor in actors:
            ray.kill(actor)
        ray.shutdown()
        cluster.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Qwen3-Omni directory or HF ID; only processor/config files are read")
    parser.add_argument("--nodes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--replicas", type=int, default=2, help="P1 cache actors, distributed over the raylets")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--size", type=int, default=112)
    parser.add_argument("--cache-gb", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=600, help="Per-wave timeout, including processor startup")
    parser.add_argument("--report", type=Path, default=Path("outputs/repro-8039.json"))
    parser.add_argument("--self-test", action="store_true", help="Test the detector only, using standard Python")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.model:
        parser.error("--model is required")
    if min(args.requests, args.concurrency, args.waves, args.frames, args.size) < 1:
        parser.error("request/concurrency/wave/frame/size counts must be positive")
    if not args.nodes <= args.replicas <= args.requests:
        parser.error("require nodes <= replicas <= requests")
    if args.cache_gb < 0 or args.timeout <= 0:
        parser.error("cache-gb must be nonnegative and timeout must be positive")
    report = {"arguments": vars(args) | {"report": str(args.report)}, "cases": [], "versions": {}}
    try:
        for package in ("ray", "torch", "transformers", "vllm", "vllm-omni"):
            report["versions"][package] = importlib.metadata.version(package)
        direct_url = importlib.metadata.distribution("vllm-omni").read_text("direct_url.json")
        if direct_url:
            report["vllm_omni_commit"] = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        status = run(args, report)
        report["result"] = "MISMATCH" if status else "NOT_REPRODUCED"
    except PayloadMismatch:
        status = 1
        report["result"] = "MISMATCH"
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr)
    except Exception:
        status = 2
        report["result"] = "ENVIRONMENT_OR_RUNTIME_ERROR"
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['result']}: {args.report}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
