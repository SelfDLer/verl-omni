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

import hashlib
import json
import runpy
import struct
from pathlib import Path

import pytest

REPORT = runpy.run_path(Path(__file__).resolve().parents[1] / "special_e2e/analyze_8039_trace.py")


class Tokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(token) if token else ("" if skip_special_tokens else "<eos>") for token in ids)


def snapshot(text):
    ids = [ord(char) for char in text]
    raw = struct.pack(f"<{len(ids)}q", *ids)
    return {
        "status": "complete",
        "count": len(ids),
        "ids": ids,
        "encoding": "int64-le",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def records(*, engine="<answer>B</answer>", merged=None, prompt="Use <answer>A</answer>", request="r", trace="t"):
    merged = engine if merged is None else merged
    result = []
    for event in REPORT["_REQUIRED"]:
        server = event.startswith(("server.", "strategy."))
        row = {
            "trace_id": trace,
            "uid": "uid",
            "session_id": 0,
            "event": event,
            "call_id": "call",
            "host": "worker" if server else "head",
            "pid": 2 if server else 1,
            "writer_id": "writer",
            "seq": len(result),
            "dropped_events": 0,
        }
        if server:
            row.update(request_id=request, replica_rank=3)
        if event in REPORT["_PROMPTS"]:
            row["prompt_token_snapshot"] = snapshot(prompt)
        else:
            row["response_token_snapshot"] = snapshot(
                merged if event in ("agent.merge.after", "agent.output") else engine
            )
        if event == "strategy.submit":
            row["max_tokens"] = 1024
        if event == "strategy.result":
            row["finish_reason"] = "stop"
        result.append(row)
    result.append(
        {
            "trace_id": trace,
            "uid": "uid",
            "session_id": 0,
            "event": "agent.source.message",
            "role": "system",
            "text": REPORT["_text"]("Use <answer>A</answer>"),
        }
    )
    return result


def write_records(tmp_path, rows):
    path = tmp_path / "video-v2-test.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def analyze(tmp_path, rows, **kwargs):
    return REPORT["analyze"]([write_records(tmp_path, rows)], tokenizer=Tokenizer(), **kwargs)


def codes(report):
    return {item["code"] for item in report["requests"][0]["findings"]}


def test_format_loss_at_merge_is_located_without_rescoring(tmp_path):
    report = analyze(tmp_path, records(merged="B"))
    assert report["incomplete_traces"] == 0
    finding = next(item for item in report["requests"][0]["findings"] if item["code"] == "answer_tags_lost")
    assert finding["from"] == "agent.merge.before"
    assert finding["to"] == "agent.merge.after"
    assert "engine_missing_answer_tags" not in codes(report)
    assert report["replicas"][0]["host"] == "worker"
    assert report["replicas"][0]["replica_rank"] == 3
    assert report["replicas"][0]["findings"] == {"answer_tags_lost": 1}
    assert "accuracy" not in json.dumps(report)


def test_engine_missing_format_is_not_attributed_to_agent(tmp_path):
    report = analyze(tmp_path, records(engine="The answer is B."))
    assert codes(report) == {"engine_missing_answer_tags"}
    assert report["replicas"][0]["findings"] == {"engine_missing_answer_tags": 1}


@pytest.mark.parametrize(("engine", "expected"), [("", True), ("\x00", False)])
def test_empty_tokens_and_special_token_only_output_are_distinct(tmp_path, engine, expected):
    report = analyze(tmp_path, records(engine=engine))
    assert ("empty_engine_tokens" in codes(report)) is expected
    assert "engine_visible_empty" in codes(report)


def test_length_stop_and_actual_budget_are_preserved(tmp_path):
    rows = records(engine="unfinished reasoning")
    next(row for row in rows if row["event"] == "strategy.result")["finish_reason"] = "length"
    report = analyze(tmp_path, rows)
    finding = next(item for item in report["requests"][0]["findings"] if item["code"] == "engine_length_stop")
    assert finding["max_tokens"] == 1024
    assert finding["tokens"] == len("unfinished reasoning")


def test_prompt_format_instruction_loss_and_transport_loss(tmp_path):
    report = analyze(tmp_path, records(prompt="Question without system instruction"))
    assert any(
        item.get("from") == "agent.source.message" and item.get("to") == "agent.prompt"
        for item in report["requests"][0]["findings"]
    )
    rows = records()
    next(row for row in rows if row["event"] == "server.receive")["prompt_token_snapshot"] = snapshot("Question only")
    report = analyze(tmp_path, rows)
    assert any(
        item.get("from") == "agent.dispatch" and item.get("to") == "server.receive"
        for item in report["requests"][0]["findings"]
    )


def test_invalid_digest_and_no_tokenizer_never_imply_missing_tags(tmp_path):
    rows = records()
    next(row for row in rows if row["event"] == "strategy.result")["response_token_snapshot"]["ids"][0] = 99
    report = analyze(tmp_path, rows)
    assert report["incomplete_traces"] == 1
    assert "engine_missing_answer_tags" not in codes(report)
    report = REPORT["analyze"]([write_records(tmp_path, records(engine="B"))])
    assert report["incomplete_traces"] == 1
    assert "engine_missing_answer_tags" not in codes(report)


def test_resume_attempts_are_not_compared_as_a_single_completion(tmp_path):
    rows = records()
    rows.extend(row for row in records(engine="partial", request="resume") if row.get("request_id"))
    report = analyze(tmp_path, rows)
    assert report["requests"][0]["engine_attempts"] == 2
    assert "answer_tags_lost" not in codes(report)
    assert report["replicas"][0]["requests"] == 2


def test_validation_text_is_joined_by_uid_session_not_file_order(tmp_path):
    path = tmp_path / "validation.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"uid": "other_0_0", "output": "<answer>B</answer>"},
                {"uid": "uid_0_0", "output": "B"},
            ]
        ),
        encoding="utf-8",
    )
    report = analyze(tmp_path, records(), validation=path)
    assert "validation_text_changed" in codes(report)
    assert report["requests"][0]["validation_output"]["head"] == "B"


def test_missing_and_dropped_events_remain_visible(tmp_path):
    rows = [row for row in records() if row["event"] != "agent.merge.after"]
    rows[0]["dropped_events"] = 5
    path = write_records(tmp_path, rows)
    with path.open("a", encoding="utf-8") as stream:
        stream.write('\n{"incomplete":')
    report = REPORT["analyze"]([path], tokenizer=Tokenizer())
    assert report["dropped_events"] == 5
    assert report["warnings"]
    assert not report["coverage_complete"]
    assert "agent.merge.after" in report["requests"][0]["missing_events"]


def test_unavailable_required_snapshot_does_not_imply_complete_coverage(tmp_path):
    rows = records()
    next(row for row in rows if row["event"] == "strategy.result")["response_token_snapshot"] = {
        "status": "unavailable"
    }
    report = analyze(tmp_path, rows)
    assert not report["coverage_complete"]
    assert "engine_missing_answer_tags" not in codes(report)


def test_special_token_filtering_is_distinguished_from_merge_loss(tmp_path):
    class StrippingTokenizer(Tokenizer):
        def decode(self, ids, skip_special_tokens=False):
            text = super().decode(ids, skip_special_tokens=skip_special_tokens)
            return text.replace("<answer>", "").replace("</answer>", "") if skip_special_tokens else text

    report = REPORT["analyze"]([write_records(tmp_path, records())], tokenizer=StrippingTokenizer())
    assert "special_token_filter_removed_answer_tags" in codes(report)
    assert "answer_tags_lost" not in codes(report)
