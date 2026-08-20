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
"""Binary multiple-choice reward for TinyLLaVA-Video-R1 NextQA."""

import re
from typing import Any

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_OPTION_RE = re.compile(r"[A-E]")


def extract_answer(text: Any) -> str:
    """Return the first answer payload, matching Relax's task4 extraction."""
    match = _ANSWER_TAG_RE.search(str(text or ""))
    return match.group(1).strip() if match else ""


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict[str, float]:
    """Reward an exactly tagged correct option, matching Relax's task4 scorer."""
    del kwargs
    prediction = extract_answer(solution_str)
    target = extract_answer(ground_truth)
    format_score = float(_OPTION_RE.fullmatch(prediction) is not None)
    accuracy = float(bool(target) and prediction == target)
    return {
        "score": accuracy,
        "accuracy": accuracy,
        "format": format_score,
    }
