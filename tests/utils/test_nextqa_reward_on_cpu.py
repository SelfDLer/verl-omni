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

from verl_omni.utils.reward_score.nextqa_reward import compute_score, extract_answer


def test_extract_answer_matches_relax_payload_extraction():
    assert extract_answer("<think>reasoning</think><answer> C </answer>") == "C"
    assert extract_answer("C") == ""
    assert extract_answer("<answer>choice C</answer>") == "choice C"
    assert extract_answer("<answer>F</answer>") == "F"


def test_compute_score_uses_binary_accuracy_and_reports_format():
    assert compute_score("<think>x</think><answer>B</answer>", "<answer>B</answer>") == {
        "score": 1.0,
        "accuracy": 1.0,
        "format": 1.0,
    }
    assert compute_score("<answer>A</answer>", "<answer>B</answer>") == {
        "score": 0.0,
        "accuracy": 0.0,
        "format": 1.0,
    }
    assert compute_score("B", "<answer>B</answer>") == {
        "score": 0.0,
        "accuracy": 0.0,
        "format": 0.0,
    }
    assert compute_score("<answer>b</answer>", "<answer>B</answer>")["score"] == 0.0
