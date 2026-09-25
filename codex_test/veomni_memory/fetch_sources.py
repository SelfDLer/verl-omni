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

"""Explicit, optional public-source download for static audit; never installs packages."""

import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main():
    verl = (REPO / ".github/verl_pin.txt").read_text().strip()
    veomni = (REPO / ".github/veomni_pin.txt").read_text().strip()
    specs = [
        (
            "verl",
            "verl-project/verl",
            verl,
            [
                "verl/workers/engine/veomni/transformer_impl.py",
                "verl/workers/engine/base.py",
                "verl/workers/engine/veomni/utils.py",
                "verl/workers/engine_workers.py",
                "verl/workers/config/engine.py",
                "verl/workers/config/actor.py",
                "verl/trainer/config/actor/veomni_actor.yaml",
                "verl/trainer/config/engine/veomni.yaml",
                "verl/trainer/config/actor/actor.yaml",
                "verl/trainer/main_ppo.py",
                "verl/trainer/ppo/ray_trainer.py",
                "verl/trainer/constants_ppo.py",
                "verl/utils/profiler/config.py",
                "verl/utils/profiler/profile.py",
                "verl/utils/profiler/mstx_profile.py",
                "verl/workers/engine/fsdp/transformer_impl.py",
                "verl/workers/engine/utils.py",
                "verl/trainer/ppo/v1/trainer_base.py",
                "verl/trainer/config/rollout/rollout.yaml",
            ],
        ),
        (
            "veomni",
            "ByteDance-Seed/VeOmni",
            veomni,
            [
                "veomni/distributed/torch_parallelize.py",
                "veomni/distributed/parallel_state.py",
                "veomni/arguments/arguments_types.py",
                "veomni/arguments/__init__.py",
                "veomni/__init__.py",
                "veomni/models/transformers/qwen3_omni_moe/__init__.py",
                "veomni/models/transformers/qwen3_omni_moe/generated/patched_modeling_qwen3_omni_moe_gpu.py",
                "veomni/ops/platform/npu/hccl_premul_sum.py",
                "veomni/optim/optimizer.py",
            ],
        ),
        (
            "torch",
            "pytorch/pytorch",
            "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0",
            [
                "torch/distributed/fsdp/_fully_shard/_fsdp_param.py",
                "torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py",
                "torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py",
                "torch/distributed/fsdp/_fully_shard/_fsdp_state.py",
                "torch/distributed/fsdp/_fully_shard/_fsdp_api.py",
            ],
        ),
        # Reference snapshot of the v2.10.0 branch; NOT identified as the post6 wheel.
        (
            "npu",
            "Ascend/pytorch",
            "85f239d69ebb205ac18d968ffd03fa1d7974dab3",
            [
                "torch_npu/csrc/distributed/ProcessGroupHCCL.cpp",
                "torch_npu/csrc/core/npu/NPUCachingAllocator.cpp",
                "torch_npu/distributed/fsdp/_add_fsdp_patch.py",
                "torch_npu/distributed/fsdp/__init__.py",
            ],
        ),
    ]

    def fetch(task):
        label, repo, revision, path = task
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
        result = dict(label=label, repository=repo, revision=revision, path=path, url=url)
        try:
            data = urllib.request.urlopen(url, timeout=40).read()
            dest = ROOT / "_sources" / label / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            result.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        except Exception as exc:
            result["error"] = str(exc)
        return result

    tasks = [(label, repo, rev, p) for label, repo, rev, paths in specs for p in paths]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fetch, tasks))
    (ROOT / "_sources").mkdir(exist_ok=True)
    (ROOT / "_sources/manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        print(result["label"], result["path"], result.get("error", result.get("bytes")))


if __name__ == "__main__":
    main()
