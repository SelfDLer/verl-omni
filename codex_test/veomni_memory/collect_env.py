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

"""Small, allowlisted environment report. Run inside the server training environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = {"torch": "torch", "torch_npu": "torch-npu", "verl": "verl", "veomni": "veomni"}
ENV_KEYS = (
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "CANN_VERSION",
    "DEVICE_NAME",
    "MODELING_BACKEND",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TASK_QUEUE_ENABLE",
    "MULTI_STREAM_MEMORY_REUSE",
    "PYTORCH_NPU_ALLOC_CONF",
    "NUM_NPUS",
    "NNODES",
    "ROLLOUT_TP",
    "ATTN_IMPLEMENTATION",
    "USE_REMOVE_PADDING",
)


def command(args, cwd=None):
    try:
        p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=20, errors="replace")
        return {"returncode": p.returncode, "stdout": p.stdout[:12000], "stderr": p.stderr[:2000]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def git_info(path):
    return {
        key: command(["git", *args], cwd=path)
        for key, args in {
            "root": ["rev-parse", "--show-toplevel"],
            "head": ["rev-parse", "HEAD"],
            "branch": ["branch", "--show-current"],
            "status": ["status", "--short", "--untracked-files=no"],
        }.items()
    }


def source_info(obj, excerpt=False):
    result = {"module": getattr(obj, "__module__", None), "qualname": getattr(obj, "__qualname__", None)}
    try:
        path = Path(inspect.getsourcefile(obj))
        result.update(
            path=str(path), line=inspect.getsourcelines(obj)[1], sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        if excerpt:
            code = inspect.getsource(obj)
            result["source_excerpt"] = code[:24000]
            result["source_excerpt_truncated"] = len(code) > 24000
    except Exception as exc:
        result["error"] = str(exc)
    return result


def collect(repo):
    result = {
        "scope": "standalone process; NOT Ray worker state",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "repository": git_info(repo),
        "environment_allowlist": {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
        "packages": {},
        "pins": {},
        "sources": {},
        "device": {},
    }
    for name in ("verl", "veomni"):
        p = repo / f".github/{name}_pin.txt"
        result["pins"][name] = p.read_text().strip() if p.exists() else None
    for name, dist_name in PACKAGES.items():
        item = result["packages"][name] = {}
        try:
            dist = metadata.distribution(dist_name)
            item.update(distribution_version=dist.version, distribution_root=str(dist.locate_file("")))
            direct = dist.read_text("direct_url.json")
            if direct:
                # URLs can contain credentials: retain revision metadata, never the URL.
                d = json.loads(direct)
                item["install_metadata"] = {k: d[k] for k in ("vcs_info", "dir_info", "archive_info") if k in d}
        except Exception as exc:
            item["metadata_error"] = str(exc)
        try:
            module = importlib.import_module(name)
            item.update(
                import_version=str(getattr(module, "__version__", "unknown")),
                import_path=getattr(module, "__file__", None),
            )
            if item["import_path"]:
                item["checkout"] = git_info(Path(item["import_path"]).parent)
            if name == "torch_npu":
                version = importlib.import_module("torch_npu.version")
                item["git_version"] = getattr(version, "git_version", None)
            elif name == "torch":
                item["git_version"] = getattr(module.version, "git_version", None)
        except Exception as exc:
            item["import_error"] = f"{type(exc).__name__}: {exc}"
    for mod, attrs in {
        "torch.distributed.fsdp._fully_shard._fsdp_collectives": ["foreach_reduce"],
        "torch.distributed.fsdp._fully_shard._fsdp_param_group": ["FSDPParamGroup"],
        "torch_npu.distributed.fsdp._add_fsdp_patch": ["_apply_fsdp_patch"],
        "veomni.distributed.torch_parallelize": ["build_parallelize_model"],
        "verl.workers.engine.veomni.transformer_impl": ["VeOmniEngine"],
    }.items():
        try:
            module = importlib.import_module(mod)
            for attr in attrs:
                if hasattr(module, attr):
                    result["sources"][f"{mod}.{attr}"] = source_info(
                        getattr(module, attr), excerpt=attr in ("foreach_reduce", "_apply_fsdp_patch")
                    )
            if hasattr(module, "FSDPParamGroup"):
                result["sources"][mod + ".FSDPParamGroup.finalize_backward"] = source_info(
                    module.FSDPParamGroup.finalize_backward, excerpt=True
                )
            path = Path(module.__file__)
            result["sources"][mod] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        except Exception as exc:
            result["sources"][mod] = {"error": str(exc)}
    try:
        import torch

        result["device"]["npu_available"] = hasattr(torch, "npu") and torch.npu.is_available()
        if result["device"]["npu_available"]:
            result["device"]["count"] = torch.npu.device_count()
            result["device"]["name"] = torch.npu.get_device_name(0)
    except Exception as exc:
        result["device"]["error"] = str(exc)
    result["npu_smi"] = command(["npu-smi", "info"])
    result["cann_version_files"] = {}
    cann = os.environ.get("ASCEND_HOME_PATH")
    if cann:
        for rel in ("version.cfg", "version.info", "../ascend_toolkit_install.info"):
            path = Path(cann) / rel
            if path.is_file():
                result["cann_version_files"][str(path)] = path.read_text(errors="replace")[:4000]
    result["worker_state"] = "not observed; optional worker_probe is required for final config/batches/storage"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / ".gitignore").write_text("*\n", encoding="utf-8")
    path = args.output_dir / "environment.json"
    path.write_text(json.dumps(collect(args.repo_root.resolve()), ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
