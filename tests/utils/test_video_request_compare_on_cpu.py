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

import copy
import hashlib
import json
import runpy
import struct
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "special_e2e/compare_8039_request.py"
COMPARE = runpy.run_path(SCRIPT)


def tokens(ids):
    return {
        "status": "complete",
        "encoding": "int64-le",
        "count": len(ids),
        "ids": ids,
        "sha256": hashlib.sha256(struct.pack(f"<{len(ids)}q", *ids)).hexdigest(),
    }


def pixels(value=b"video", mode="sample"):
    result = {"shape": [2, 3, 4, 4], "dtype": "uint8", "device": "cpu", "object_id": 1, "digest_status": mode}
    if mode == "sample":
        result.update(
            sample_sha256=hashlib.sha256(value).hexdigest(), sample_scheme="linspace-flat-c-order-v1", sample_count=96
        )
    elif mode == "full":
        result.update(sha256=hashlib.sha256(value).hexdigest())
    return result


def request(trace_id="single"):
    records = []
    for stage, components in COMPARE["_STAGES"]:
        row = {
            "event": stage,
            "trace_id": trace_id,
            "sample_key": "video_question",
            "sample_index": 42,
            "video_id": "video",
            "question_id": "question",
            "host": trace_id,
            "pid": 10,
            "writer_id": trace_id,
            "seq": len(records),
        }
        if not stage.startswith("agent."):
            row.update(request_id=trace_id + "-engine", replica_rank=2)
        if "prompt" in components:
            row["prompt_token_snapshot"] = tokens([10, 11, 12])
        if "response" in components:
            row["response_token_snapshot"] = tokens([20, 21])
        if "video" in components:
            row["video"] = [[pixels(), {"frames_indices": [0, 10], "fps": 25, "duration": 3}]]
        if "features" in components:
            row["video_features"] = [{"identifier": trace_id, "data": {"pixel_values_videos": pixels(b"features")}}]
        if "processor" in components:
            row["mm_processor_kwargs"] = {"use_audio_in_video": False, "sampling_rate": 16000}
        if "sampling" in components:
            row["sampling_params"] = {"temperature": 1.0, "seed": None}
        if "text" in components:
            row.update(role="user", message_index=1, part_index=0)
            row["text"] = {"characters": 8, "complete": True, "sha256": hashlib.sha256(b"question").hexdigest()}
        records.append(row)
    return records


def load(tmp_path, name, records):
    path = tmp_path / f"video-v2-{name}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return COMPARE["load_request"]([path], "video_question")


def compare(tmp_path, multi, single=None):
    return COMPARE["compare"](load(tmp_path, "single", single or request()), load(tmp_path, "multi", multi))


def stage(report, name, component):
    return next(row for row in report["stages"] if row["stage"] == name)["components"][component]


def test_same_sample_matches_across_different_routes_ids_and_log_order(tmp_path):
    multi = request("multi")
    for row in multi:
        if "video" in row:
            row["video"][0][0].update(object_id=200, stride=[1, 2, 3, 4])
    report = compare(tmp_path, list(reversed(multi)))
    assert report["first_observed_difference"] is None
    assert stage(report, "agent.dispatch", "video")["status"] == "sample_equal"
    assert stage(report, "frontend.build.features", "features")["status"] == "sample_equal"
    assert stage(report, "agent.prompt", "prompt")["status"] == "equal"
    assert report["single"]["routes"][0]["host"] != report["multi"]["routes"][0]["host"]


def worker_record(records, event, **fields):
    return {**records[0], "event": event, "seq": len(records), "worker_rank": 0, **fields}


def test_worker_embedding_difference_precedes_engine_output_and_keeps_ranks_separate(tmp_path):
    single, multi = request(), request("multi")
    for records, content in ((single, b"good"), (multi, b"bad")):
        records.append(
            worker_record(
                records,
                "worker.model_input",
                chunk_start=0,
                chunk_tokens=10,
                embedding=pixels(content),
                positions=pixels(b"positions"),
            )
        )
    report = compare(tmp_path, multi, single)
    assert report["first_observed_difference"]["stage"].startswith("worker.model_input")
    assert report["first_observed_difference"]["component"] == "embedding"
    assert any(
        row["components"]["prompt"]["status"] == "unknown"
        for row in report["stages"]
        if row["stage"].startswith("worker.receive")
    )
    multi[-1]["worker_rank"] = 1
    report = compare(tmp_path, multi, single)
    assert report["first_observed_difference"] is None


def test_worker_registration_without_coverage_and_different_chunking_are_unknown(tmp_path):
    single, multi = request(), request("multi")
    for records in (single, multi):
        records.append(worker_record(records, "trace.worker_registration", status="sent"))
    report = compare(tmp_path, multi, single)
    assert any(row["stage"].startswith("worker.receive") for row in report["stages"])
    for records, count in ((single, 10), (multi, 5)):
        records.append(
            worker_record(records, "worker.model_input", chunk_start=0, chunk_tokens=count, embedding=pixels())
        )
    report = compare(tmp_path, multi, single)
    assert report["first_observed_difference"] is None
    assert all(
        row["components"]["embedding"]["status"] == "unknown"
        for row in report["stages"]
        if row["stage"].startswith("worker.model_input")
    )


def test_report_distinguishes_missing_capture_from_device_metadata(tmp_path):
    single, multi = request(), request("multi")
    for records in (single, multi):
        records.append(worker_record(records, "trace.worker_registration", status="sent"))
    report = compare(tmp_path, multi, single)
    text = COMPARE["markdown"](report)
    assert "no_worker_records_for_selected_trace" in text
    assert "worker_event_not_recorded_in_either_run" in text
    assert "Legacy status=sent does not prove tracing was enabled" in text
    for records in (single, multi):
        records.append(
            worker_record(
                records,
                "worker.receive",
                prompt_token_snapshot=tokens([1, 2]),
                sampling_params={"seed": None},
                multimodal_features=[{"data": pixels(mode="metadata_only")}],
            )
        )
    report = compare(tmp_path, multi, single)
    received = next(row for row in report["stages"] if row["stage"].startswith("worker.receive"))
    assert received["components"]["prompt"]["status"] == "equal"
    assert received["components"]["sampling_params"]["status"] == "equal"
    assert received["components"]["multimodal_features"]["status"] == "unknown"
    assert "metadata_only" in COMPARE["markdown"](report)


def test_full_encoder_cache_compares_across_chunking_and_repeated_reads(tmp_path):
    single, multi = request(), request("multi")
    single.append(
        worker_record(
            single, "worker.encoder.read", chunk_start=0, chunk_tokens=10, feature_index=0, embedding=pixels()
        )
    )
    for start in (0, 5):
        multi.append(
            worker_record(
                multi, "worker.encoder.read", chunk_start=start, chunk_tokens=5, feature_index=0, embedding=pixels()
            )
        )
    report = compare(tmp_path, multi, single)
    reads = [row for row in report["stages"] if row["stage"].startswith("worker.encoder.read")]
    assert len(reads) == 1
    assert reads[0]["components"]["embedding"]["status"] == "sample_equal"
    assert len(reads[0]["sources"]["multi"]) == 2
    multi[-1]["embedding"] = pixels(b"changed cache")
    report = compare(tmp_path, multi, single)
    row = next(row for row in report["stages"] if row["stage"].startswith("worker.encoder.read"))
    assert row["components"]["embedding"]["status"] == "unknown"
    assert row["components"]["embedding"]["reason"] == "repeated_worker_event_not_aligned"


def test_worker_acknowledgement_shows_disabled_environment_without_worker_files(tmp_path):
    single, multi = request(), request("multi")
    for records in (single, multi):
        records.append(
            worker_record(
                records,
                "trace.worker_registration",
                status="returned",
                worker_replies=[
                    [{"video_trace_ack": True, "status": "disabled", "trace_stage": "boundary", "device_sample": "0"}]
                ],
            )
        )
    report = compare(tmp_path, multi, single)
    assert report["worker_coverage"]["single"]["worker_acknowledgements"][0]["status"] == "disabled"
    assert "status `disabled`, stage `boundary`" in COMPARE["markdown"](report)


def test_missing_admission_hook_is_reported_even_without_registration(tmp_path):
    single, multi = request(), request("multi")
    for records in (single, multi):
        records.append(
            {
                "event": "trace.install",
                "boundary": "frontend.worker_admission",
                "status": "unavailable",
                "error": "unsupported signature",
            }
        )
    report = compare(tmp_path, multi, single)
    assert "single" in report["worker_coverage"]
    assert "Worker admission hook: `unavailable`; unsupported signature" in COMPARE["markdown"](report)


def test_report_identifies_missing_files_and_exact_unmatched_target(tmp_path):
    single, multi = request(), request("multi")
    for records in (single, multi):
        row = next(r for r in records if r["event"] == "frontend.build.features")
        row["core_request_id"] = "exact-core-id"
        records.append(
            worker_record(
                records,
                "trace.worker_registration",
                status="returned",
                worker_replies=[
                    {"video_trace_ack": True, "status": "registered", "trace_file": "/tmp/video-v2-worker.jsonl"}
                ],
            )
        )
        records.append({"event": "trace.worker_unmatched", "core_request_id": "exact-core-id"})
    report = compare(tmp_path, multi, single)
    coverage = report["worker_coverage"]["single"]
    assert coverage["expected_worker_files_not_supplied"] == ["/tmp/video-v2-worker.jsonl"]
    assert len(coverage["unmatched_target_requests"]) == 1
    assert "request reached a worker but trace context was not associated" in COMPARE["markdown"](report)


def test_worker_audio_difference_and_frontend_position_difference_are_visible(tmp_path):
    single, multi = request(), request("multi")
    for records, value in ((single, b"audio1"), (multi, b"audio2")):
        records.append(
            worker_record(
                records,
                "worker.receive",
                prompt_token_snapshot=tokens([1]),
                sampling_params={"seed": None},
                multimodal_features=[{"modality": "audio", "data": pixels(value)}],
            )
        )
    report = compare(tmp_path, multi, single)
    assert report["first_observed_difference"]["component"] == "multimodal_features"
    for records, offset in ((single, 5), (multi, 6)):
        row = next(r for r in records if r["event"] == "frontend.build.features")
        row["multimodal_features"] = [{"modality": "video", "data": pixels(), "position": {"offset": offset}}]
    report = compare(tmp_path, multi, single)
    assert report["first_observed_difference"] == {"stage": "frontend.build.features", "component": "multimodal"}


def test_video_first_changes_inside_frontend_while_prompt_remains_equal(tmp_path):
    multi = request("multi")
    next(row for row in multi if row["event"] == "frontend.tokens.before")["video"][0][0] = pixels(b"other video")
    report = compare(tmp_path, multi)
    assert report["first_observed_difference"] == {"stage": "frontend.tokens.before", "component": "video"}
    first = report["first_difference_by_component"]["video"]
    assert first["previous_observed_match"] == {"stage": "frontend.input.before", "status": "sample_equal"}
    assert stage(report, "frontend.tokens.before", "prompt")["status"] == "equal"
    assert first["comparison"]["differences"][0]["path"].endswith("sample_sha256")


def test_feature_change_is_compared_against_same_stage_not_raw_pixels(tmp_path):
    multi = request("multi")
    next(row for row in multi if row["event"] == "frontend.tokens.features")["video_features"][0]["data"] = {
        "pixel_values_videos": pixels(b"wrong features")
    }
    report = compare(tmp_path, multi)
    assert report["first_observed_difference"] == {"stage": "frontend.tokens.features", "component": "features"}


@pytest.mark.parametrize("mode", ["metadata_only", "skipped_non_cpu", "skipped_byte_limit"])
def test_unread_video_content_never_claims_equality(tmp_path, mode):
    single, multi = request(), request("multi")
    for rows in (single, multi):
        next(row for row in rows if row["event"] == "agent.dispatch")["video"][0][0] = pixels(mode=mode)
    report = compare(tmp_path, multi, single)
    assert stage(report, "agent.dispatch", "video")["status"] == "unknown"


def test_cache_omission_and_truncation_do_not_report_corruption(tmp_path):
    multi = request("multi")
    next(row for row in multi if row["event"] == "frontend.input.features")["video_features"][0]["data"] = None
    next(row for row in multi if row["event"] == "agent.dispatch")["video"] = [{"omitted_items": 100}]
    report = compare(tmp_path, multi)
    assert stage(report, "frontend.input.features", "features")["status"] == "unknown"
    assert stage(report, "agent.dispatch", "video")["status"] == "unknown"
    assert report["first_observed_difference"] is None


def test_full_fingerprints_and_incompatible_modes(tmp_path):
    single, multi = request(), request("multi")
    for rows in (single, multi):
        next(row for row in rows if row["event"] == "agent.dispatch")["video"][0][0] = pixels(mode="full")
    next(row for row in multi if row["event"] == "server.receive")["video"][0][0] = pixels(mode="full")
    report = compare(tmp_path, multi, single)
    assert stage(report, "agent.dispatch", "video")["status"] == "equal"
    assert stage(report, "server.receive", "video")["status"] == "unknown"


def test_message_limit_is_unknown_even_when_both_runs_hit_it(tmp_path):
    single, multi = request(), request("multi")
    for rows in (single, multi):
        next(row for row in rows if row["event"] == "agent.source.message").update(status="message_limit")
    assert stage(compare(tmp_path, multi, single), "agent.source.message", "text")["status"] == "unknown"


def test_missing_stage_is_reported_as_gap_before_observed_difference(tmp_path):
    multi = [row for row in request("multi") if row["event"] != "frontend.input.before"]
    next(row for row in multi if row["event"] == "frontend.tokens.before")["video"][0][0] = pixels(b"changed")
    report = compare(tmp_path, multi)
    first = report["first_difference_by_component"]["video"]
    assert first["previous_observed_match"]["stage"] == "frontend.build.before"
    assert first["unresolved_stages_before"] == ["frontend.input.before"]


def test_token_difference_index_and_invalid_digest(tmp_path):
    multi = request("multi")
    next(row for row in multi if row["event"] == "strategy.result")["response_token_snapshot"] = tokens([20, 99])
    next(row for row in multi if row["event"] == "server.result")["response_token_snapshot"]["ids"] = [88, 99]
    report = compare(tmp_path, multi)
    assert stage(report, "strategy.result", "response")["first_token_difference"] == 1
    assert stage(report, "server.result", "response")["status"] == "unknown"


def test_retries_are_not_implicitly_paired(tmp_path):
    multi = request("multi")
    retry = copy.deepcopy(next(row for row in multi if row["event"] == "server.receive"))
    retry.update(request_id="retry", seq=100)
    multi.append(retry)
    report = compare(tmp_path, multi)
    assert stage(report, "frontend.tokens.before", "video")["status"] == "unknown"
    assert stage(report, "agent.output", "response")["status"] == "equal"


def test_duplicate_sample_invocations_require_explicit_selection(tmp_path):
    rows = request("one") + request("two")
    path = tmp_path / "video-v2-all.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="expected one trace"):
        COMPARE["load_request"]([path], "video_question")
    assert COMPARE["load_request"]([path], "video_question", "two")["trace_id"] == "two"


def test_same_video_with_different_question_or_index_is_not_matched(tmp_path):
    multi = request("multi")
    for row in multi:
        row["question_id"] = "another-question"
    with pytest.raises(ValueError, match="Sample identity differs"):
        compare(tmp_path, multi)


def test_export_keeps_feature_data_and_log_health(tmp_path, monkeypatch, capsys):
    left = load(tmp_path, "left", request())
    right = load(tmp_path, "right", request("multi"))
    right_path = tmp_path / "video-v2-right.jsonl"
    with right_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps({"event": "trace.health", "host": "multi", "pid": 10, "writer_id": "multi", "dropped_events": 3})
            + "\n"
        )
        stream.write("{broken\n")
    output = tmp_path / "comparison"
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--single",
            str(tmp_path / "video-v2-left.jsonl"),
            "--multi",
            str(right_path),
            "--sample-key",
            "video_question",
            "--output-dir",
            str(output),
        ],
    )
    runpy.run_path(SCRIPT, run_name="__main__")
    report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert report["multi"]["dropped_events"] == 3
    assert len(report["multi"]["warnings"]) == 1
    assert "video_features" in (output / "multi.jsonl").read_text(encoding="utf-8")
    assert left["trace_id"] != right["trace_id"]
    assert "sample_equal" in (output / "comparison.md").read_text(encoding="utf-8")
    assert "First observed difference" in capsys.readouterr().out
