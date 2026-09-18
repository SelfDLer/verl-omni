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
"""Run in the training shell and repository: python3 scripts/audit_npu_environment.py.

Read-only, standard library only. Does not import torch/vLLM, initialize an NPU,
install packages, or print remote URLs/credentials. Emulates python -m's cwd
lookup so a repository shadowing site-packages is visible in the report.
"""

import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

PACKAGES = {
    "verl": "verl",
    "verl-omni": "verl_omni",
    "vllm": "vllm",
    "vllm-ascend": "vllm_ascend",
    "vllm-omni": "vllm_omni",
    "torch": "torch",
    "torch-npu": "torch_npu",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "transformers": "transformers",
    "qwen-omni-utils": "qwen_omni_utils",
    "qwen-vl-utils": "qwen_vl_utils",
    "torchcodec": "torchcodec",
    "ray": "ray",
    "tensordict": "tensordict",
    "TransferQueue": "transfer_queue",
    "triton-ascend": "triton",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "fastapi": "fastapi",
}
CORE = {"verl", "verl-omni", "vllm", "vllm-ascend", "vllm-omni"}
DEP_NAMES = {name.lower().replace("_", "-") for name in PACKAGES}


def git_state(path):
    """Find the source checkout and read its revision without updating the index."""
    for parent in (path, *path.parents):
        if (parent / ".git").exists():

            def run(*args, directory=parent):
                result = subprocess.run(
                    ["git", "--no-optional-locks", "-C", str(directory), *args],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                return result.stdout.strip()

            try:
                return {
                    "root": str(parent),
                    "head": run("rev-parse", "HEAD"),
                    "tracked_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
                }
            except (OSError, subprocess.SubprocessError) as exc:
                return {"root": str(parent), "error": type(exc).__name__}
    return None


def inspect_package(name, module):
    """Read installation metadata and locate source without importing the package."""
    info = {}
    try:
        dist = metadata.distribution(name)
        info.update(version=dist.version, metadata_root=str(dist.locate_file("")))
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        if direct:
            vcs = direct.get("vcs_info", {})
            info["installation"] = {
                "commit_id": vcs.get("commit_id"),
                "editable": direct.get("dir_info", {}).get("editable", False),
            }
        if name in CORE:
            info["relevant_requires_dist"] = []
            for requirement in dist.requires or []:
                match = re.match(r"[A-Za-z0-9_.-]+", requirement)
                if match and match.group().lower().replace("_", "-") in DEP_NAMES:
                    safe = requirement
                    if "@" in requirement:
                        safe = requirement.split("@", 1)[0] + "@ <direct reference>"
                    info["relevant_requires_dist"].append(safe)
    except metadata.PackageNotFoundError:
        info["version"] = None
    except (OSError, ValueError) as exc:
        info["metadata_error"] = type(exc).__name__
    try:
        spec = importlib.util.find_spec(module)  # Top-level name; no package import.
        if spec and spec.origin:
            origin = Path(spec.origin).resolve()
            info["module_file"] = str(origin)
            if name in CORE:
                info["source_git"] = git_state(origin.parent)
            if name == "verl":
                loop = origin.parent / "experimental/agent_loop/agent_loop.py"
                if loop.is_file():
                    raw = loop.read_bytes()
                    lines = raw.decode("utf-8").splitlines()
                    info["agent_loop"] = {
                        "file": str(loop),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "reward_assignment_lines": [i for i, line in enumerate(lines, 1) if "rm_scores[-1] =" in line],
                        "rope_kwargs_hook_present": any("get_rope_index_kwargs" in line for line in lines),
                    }
    except (ImportError, OSError, ValueError) as exc:
        info["source_error"] = type(exc).__name__
    return info


def main():
    """Print the active training environment as JSON using only the standard library."""
    sys.path.insert(0, str(Path.cwd()))
    selected_env = (
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_OPP_PATH",
        "VLLM_USE_V1",
        "VLLM_BATCH_INVARIANT",
        "VERL_DATAPROTO_SERIALIZATION_METHOD",
    )
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "environment": {key: os.getenv(key) for key in selected_env},
        "packages": {name: inspect_package(name, module) for name, module in PACKAGES.items()},
    }
    candidates = [Path("/usr/local/Ascend/ascend-toolkit/latest"), Path("/usr/local/Ascend/ascend-toolkit")]
    candidates += [Path(os.environ[key]) for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME") if os.getenv(key)]
    report["cann_version_files"] = {}
    for base in candidates:
        for relative in (
            "version.cfg",
            "version.info",
            "aarch64-linux/ascend_toolkit_install.info",
            "x86_64-linux/ascend_toolkit_install.info",
        ):
            path = base / relative
            try:
                if path.is_file():
                    report["cann_version_files"][str(path)] = path.read_text(errors="replace")[:4096]
            except OSError:
                pass
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
