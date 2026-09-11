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

"""Masked actor/rollout log-probability diagnostics for a debug run."""

import json
from pathlib import Path

import torch


def save_consistency_batch(data, output_dir, step):
    """Save paired response tokens/logprobs and report errors in log space.

    These are sampled-token differences, not full-vocabulary KL. Empty and
    nonfinite responses are reported explicitly instead of looking like a pass.
    """
    names = ("old_log_probs", "rollout_log_probs", "response_mask", "responses")
    tensors = {name: data[name].detach().cpu() for name in names}
    actor, rollout = (tensors[name].float() for name in names[:2])
    mask = tensors["response_mask"].bool()
    if any(value.shape != mask.shape for value in tensors.values()):
        raise ValueError("Consistency diagnostics require aligned response token, mask and log-prob shapes.")
    finite = torch.isfinite(actor) & torch.isfinite(rollout)
    valid = mask & finite
    metrics = {
        "consistency/response_tokens": int(mask.sum()),
        "consistency/nonfinite_tokens": int((mask & ~finite).sum()),
        "consistency/valid": int(bool(mask.any()) and bool(finite[mask].all())),
    }
    if valid.any():
        diff = actor[valid] - rollout[valid]
        absolute = diff.abs()
        metrics.update(
            {
                "consistency/logprob_mae": absolute.mean().item(),
                "consistency/logprob_max_abs": absolute.max().item(),
                "consistency/logprob_p95_abs": torch.quantile(absolute, 0.95).item(),
                "consistency/logprob_signed_mean": diff.mean().item(),
            }
        )
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"step_{step:06d}"
    torch.save({"step": step, **tensors}, prefix.with_suffix(".pt"))
    prefix.with_suffix(".json").write_text(json.dumps({"step": step, **metrics}, indent=2) + "\n", encoding="utf-8")
    return metrics
