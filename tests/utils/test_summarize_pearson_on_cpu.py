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

import csv
import sys

import pytest

from npu_test import summarize_pearson


def test_summary_keeps_probability_spread_and_log_ppl_diagnostics(tmp_path, monkeypatch, capsys):
    case_dir = tmp_path / "no_sleep_reload"
    case_dir.mkdir()
    log = case_dir / "run.log"
    log.write_text(
        " ".join(
            (
                "training/rollout_actor_probs_pearson_corr:0.97",
                "training/rollout_probs_diff_mean=0.001",
                "training/rollout_probs_diff_max=0.05",
                "training/rollout_probs_diff_std=0.004",
                "rollout_corr/log_ppl_diff=-0.012",
            )
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "summary.csv"
    monkeypatch.setattr(sys, "argv", ["summarize_pearson.py", str(tmp_path), "--csv", str(csv_path)])

    assert summarize_pearson.main() == 0

    output = capsys.readouterr().out
    assert "prob_diff_std_last" in output
    assert "log_ppl_diff_last" in output
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        row = next(csv.DictReader(csv_file))
    assert float(row["prob_diff_std_last"]) == pytest.approx(0.004)
    assert float(row["log_ppl_diff_last"]) == pytest.approx(-0.012)
