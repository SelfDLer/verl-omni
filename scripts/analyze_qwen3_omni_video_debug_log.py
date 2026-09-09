# Copyright 2026 verl-omni contributors
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
"""Analyze Qwen3-Omni actor/rollout multimodal consistency debug logs.

The script understands the one-line JSON records emitted by the temporary
debug probes used when diagnosing Qwen3-Omni video off-policy errors:

* OFFPOLICY_SAMPLE
* ACTOR_VIDEO_DEBUG
* ROLLOUT_INPUT_DEBUG
* VLLM_MM_BEFORE_DEBUG
* VLLM_MM_AFTER_DEBUG

Ray worker prefixes and ANSI color sequences are ignored.  The input may be a
single log, several logs, a Ray log directory, or ``-`` for stdin.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TAGS = (
    "OFFPOLICY_SAMPLE",
    "ACTOR_VIDEO_DEBUG",
    "ROLLOUT_INPUT_DEBUG",
    "VLLM_MM_BEFORE_DEBUG",
    "VLLM_MM_AFTER_DEBUG",
)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
JSON_DECODER = json.JSONDecoder()
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
METRIC_ALIASES = {
    "step": ("training/global_step", "global_step"),
    "reward": ("critic/rewards/mean",),
    "pearson": ("training/rollout_actor_probs_pearson_corr", "rollout_actor_probs_pearson_corr"),
    "log_ppl_abs_diff": ("rollout_corr/log_ppl_abs_diff", "log_ppl_abs_diff"),
    "probs_diff_mean": ("training/rollout_actor_probs_diff_mean", "rollout_actor_probs_diff_mean"),
    "response_length": ("response_length/mean", "response/length/mean"),
    "grad_norm": ("actor/grad_norm",),
}
METRIC_REGEXES = {
    canonical: [
        re.compile(rf"(?<![\w/])['\"]?{re.escape(alias)}['\"]?\s*[:=]\s*({FLOAT_PATTERN})")
        for alias in aliases
    ]
    for canonical, aliases in METRIC_ALIASES.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Log files/directories, or - for stdin")
    parser.add_argument("--top", type=int, default=10, help="Number of worst samples to print")
    parser.add_argument(
        "--offpolicy-threshold",
        type=float,
        default=0.03,
        help="seq_log_ppl_abs_diff threshold used to count abnormal samples",
    )
    parser.add_argument("--json-output", type=Path, help="Also write the complete report as JSON")
    return parser.parse_args()


def iter_input_files(paths: list[str]) -> Iterable[tuple[str, Iterable[str]]]:
    for raw_path in paths:
        if raw_path == "-":
            yield "<stdin>", sys.stdin
            continue

        path = Path(raw_path)
        if path.is_file():
            yield str(path), path.open("r", encoding="utf-8", errors="replace")
            continue
        if path.is_dir():
            for root, _, names in os.walk(path, onerror=lambda _: None):
                for name in names:
                    child = Path(root, name)
                    try:
                        yield str(child), child.open("r", encoding="utf-8", errors="replace")
                    except (OSError, UnicodeError):
                        continue
            continue
        print(f"warning: input does not exist: {path}", file=sys.stderr)


def parse_records(paths: list[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    records: dict[str, list[dict[str, Any]]] = {tag: [] for tag in TAGS}
    records["TRAIN_METRICS"] = []
    meta: dict[str, Any] = {"files": 0, "lines": 0, "tagged_lines": 0, "parse_errors": []}

    for source, lines in iter_input_files(paths):
        meta["files"] += 1
        try:
            for line_number, raw_line in enumerate(lines, 1):
                meta["lines"] += 1
                line = ANSI_RE.sub("", raw_line)
                metric_record: dict[str, Any] = {}
                for canonical, regexes in METRIC_REGEXES.items():
                    for regex in regexes:
                        match = regex.search(line)
                        if match:
                            metric_record[canonical] = float(match.group(1))
                            break
                if metric_record:
                    metric_record["_source"] = source
                    metric_record["_line"] = line_number
                    records["TRAIN_METRICS"].append(metric_record)

                tag = next((candidate for candidate in TAGS if f"[{candidate}]" in line), None)
                if tag is None:
                    continue
                meta["tagged_lines"] += 1
                json_start = line.find("{", line.find(f"[{tag}]"))
                if json_start < 0:
                    meta["parse_errors"].append(f"{source}:{line_number}: no JSON object after [{tag}]")
                    continue
                try:
                    payload, _ = JSON_DECODER.raw_decode(line[json_start:])
                except json.JSONDecodeError as exc:
                    meta["parse_errors"].append(f"{source}:{line_number}: {exc.msg}")
                    continue
                if not isinstance(payload, dict):
                    meta["parse_errors"].append(f"{source}:{line_number}: JSON payload is not an object")
                    continue
                payload["_source"] = source
                payload["_line"] = line_number
                records[tag].append(payload)
        finally:
            if lines is not sys.stdin:
                lines.close()  # type: ignore[attr-defined]
    return records, meta


def analyze_training_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    last_step: float | None = None
    for record in records:
        if record.get("step") is not None:
            last_step = record["step"]
        for key in METRIC_ALIASES:
            if key == "step" or record.get(key) is None:
                continue
            series[key].append(
                {
                    "step": record.get("step", last_step),
                    "value": record[key],
                    "source": record["_source"],
                    "line": record["_line"],
                }
            )

    summaries = {}
    for key, points in series.items():
        values = [point["value"] for point in points]
        summaries[key] = {
            "count": len(values),
            "first": values[0],
            "last": values[-1],
            "change": values[-1] - values[0],
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "min_step": points[values.index(min(values))]["step"],
            "max_step": points[values.index(max(values))]["step"],
        }
    return {"line_count": len(records), "summaries": summaries, "series": dict(series)}


def unwrap(value: Any) -> Any:
    """Unwrap the compact tensor summaries used by the vLLM probe."""
    if isinstance(value, dict):
        for key in ("values", "value", "data"):
            if key in value:
                return unwrap(value[key])
    return value


def numbers(value: Any) -> list[float]:
    value = unwrap(value)
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for item in value:
            result.extend(numbers(item))
        return result
    return []


def first_number(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        values = numbers(record.get(key))
        if values:
            return values[0]
    return None


def first_value(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return unwrap(record[key])
    return None


def normalized(value: Any) -> str:
    value = unwrap(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def almost_equal(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def extract_problem_id(record: dict[str, Any]) -> str:
    for key in ("problem_id", "uid", "sample_id", "id"):
        if record.get(key) is not None:
            return str(record[key])

    extra = record.get("extra_info")
    if isinstance(extra, str):
        try:
            extra = ast.literal_eval(extra)
        except (SyntaxError, ValueError):
            extra = None
    if isinstance(extra, dict):
        for key in ("problem_id", "uid", "sample_id", "id"):
            if extra.get(key) is not None:
                return str(extra[key])
    return f"batch:{record.get('batch_index', '?')}"


def get_mm_field(record: dict[str, Any], *keys: str) -> Any:
    mm_fields = record.get("mm_fields")
    for key in keys:
        if isinstance(mm_fields, dict) and key in mm_fields:
            return unwrap(mm_fields[key])
        if key in record:
            return unwrap(record[key])
    return None


def expected_video_tokens(record: dict[str, Any]) -> list[float]:
    for key in ("expected_video_tokens", "video_token_counts", "expected_video_token_count"):
        result = numbers(get_mm_field(record, key))
        if result:
            return result

    # A fallback for logs that only contain video_grid_thw.  Qwen3-Omni uses
    # merge_size=2 by default, but do not guess it unless the probe logged it.
    grid = numbers(get_mm_field(record, "video_grid_thw"))
    merge_size = first_number(record, "spatial_merge_size", "merge_size")
    if grid and len(grid) % 3 == 0 and merge_size:
        divisor = merge_size * merge_size
        return [grid[i] * grid[i + 1] * grid[i + 2] / divisor for i in range(0, len(grid), 3)]
    return []


def analyze_actor(records: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches = []
    grids = Counter()
    seconds = Counter()
    audio_modes = Counter()

    for record in records:
        expected = expected_video_tokens(record)
        actual = numbers(first_value(record, "actual_video_tokens_per_sample", "actual_video_token_counts"))
        if expected and actual:
            if len(expected) == len(actual):
                bad = [(a, b) for a, b in zip(expected, actual) if not almost_equal(a, b)]
                mismatch = bool(bad)
            else:
                mismatch = not almost_equal(sum(expected), sum(actual))
            if mismatch:
                mismatches.append(
                    {
                        "source": record["_source"],
                        "line": record["_line"],
                        "expected": expected,
                        "actual": actual,
                    }
                )

        grid = get_mm_field(record, "video_grid_thw")
        if grid is not None:
            grids[normalized(grid)] += 1
        second = first_value(record, "second_per_grids", "second_per_grid_ts", "video_second_per_grid")
        if second is not None:
            seconds[normalized(second)] += 1
        audio_modes[str(record.get("use_audio_in_video", "missing"))] += 1

    return {
        "count": len(records),
        "token_mismatches": mismatches,
        "video_grid_thw": dict(grids),
        "second_per_grid": dict(seconds),
        "use_audio_in_video": dict(audio_modes),
    }


def analyze_offpolicy(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    samples = []
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        absolute = first_number(record, "seq_log_ppl_abs_diff", "log_ppl_abs_diff")
        signed = first_number(record, "seq_log_ppl_diff", "log_ppl_diff")
        pearson = first_number(record, "pearson_corr", "pearson")
        item = {
            "problem_id": extract_problem_id(record),
            "batch_index": record.get("batch_index"),
            "seq_log_ppl_abs_diff": absolute,
            "seq_log_ppl_diff": signed,
            "pearson_corr": pearson,
            "token_abs_mean": first_number(record, "token_abs_mean"),
            "token_abs_max": first_number(record, "token_abs_max"),
            "first_position_abs_diff_gt_0.05": record.get("first_position_abs_diff_gt_0.05"),
            "top_positions": record.get("top_positions"),
            "source": record["_source"],
            "line": record["_line"],
        }
        samples.append(item)
        by_problem[item["problem_id"]].append(item)

    ranked = sorted(
        samples,
        key=lambda item: item["seq_log_ppl_abs_diff"] if item["seq_log_ppl_abs_diff"] is not None else -1.0,
        reverse=True,
    )
    values = [item["seq_log_ppl_abs_diff"] for item in samples if item["seq_log_ppl_abs_diff"] is not None]
    first_positions = [
        item["first_position_abs_diff_gt_0.05"]
        for item in samples
        if isinstance(item["first_position_abs_diff_gt_0.05"], (int, float))
    ]
    problem_summary = []
    for problem_id, items in by_problem.items():
        item_values = [item["seq_log_ppl_abs_diff"] for item in items if item["seq_log_ppl_abs_diff"] is not None]
        problem_summary.append(
            {
                "problem_id": problem_id,
                "count": len(items),
                "abnormal_count": sum(value > threshold for value in item_values),
                "mean": statistics.fmean(item_values) if item_values else None,
                "max": max(item_values) if item_values else None,
            }
        )
    problem_summary.sort(key=lambda item: item["max"] if item["max"] is not None else -1.0, reverse=True)

    return {
        "count": len(records),
        "with_abs_diff": len(values),
        "threshold": threshold,
        "abnormal_count": sum(value > threshold for value in values),
        "all_records_above_threshold": bool(values) and all(value > threshold for value in values),
        "first_position_zero_count": sum(position == 0 for position in first_positions),
        "with_first_position": len(first_positions),
        "with_stable_problem_id": sum(not item["problem_id"].startswith("batch:") for item in samples),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "max": max(values) if values else None,
        "worst_samples": ranked,
        "by_problem": problem_summary,
    }


def index_unique(records: list[dict[str, Any]], *keys: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = first_value(record, *keys)
        if value is not None:
            result[str(value)].append(record)
    return result


def analyze_rollout_vllm(
    rollouts: list[dict[str, Any]],
    before_records: list[dict[str, Any]],
    after_records: list[dict[str, Any]],
) -> dict[str, Any]:
    before_by_hash = index_unique(before_records, "prompt_hash_before_update", "prompt_hash", "dedup_prompt_hash")
    after_by_hash = index_unique(after_records, "prompt_hash_after_update", "prompt_hash", "expanded_prompt_hash")
    checks = []
    mismatch_counts = Counter()
    matched_before = matched_after = 0

    for rollout in rollouts:
        issues = []
        dedup_hash = first_value(rollout, "dedup_prompt_hash", "prompt_hash_before_update")
        expanded_hash = first_value(rollout, "expanded_prompt_hash", "prompt_hash_after_update")
        before = before_by_hash.get(str(dedup_hash), [None])[0] if dedup_hash is not None else None
        after = after_by_hash.get(str(expanded_hash), [None])[0] if expanded_hash is not None else None
        matched_before += before is not None
        matched_after += after is not None

        dedup_length = first_number(rollout, "dedup_prompt_length")
        expanded_length = first_number(rollout, "expanded_prompt_length")
        rollout_video_tokens = first_number(rollout, "expanded_video_token_count")

        if before is not None:
            before_length = first_number(before, "prompt_length_before_update", "prompt_length")
            if dedup_length is not None and before_length is not None and not almost_equal(dedup_length, before_length):
                issues.append("before_prompt_length")
        if after is not None:
            after_length = first_number(after, "prompt_length_after_update", "prompt_length")
            after_video_tokens = first_number(after, "expanded_video_tokens", "expanded_video_token_count")
            if (
                expanded_length is not None
                and after_length is not None
                and not almost_equal(expanded_length, after_length)
            ):
                issues.append("after_prompt_length")
            if (
                rollout_video_tokens is not None
                and after_video_tokens is not None
                and not almost_equal(rollout_video_tokens, after_video_tokens)
            ):
                issues.append("expanded_video_tokens")

        expected = expected_video_tokens(before) if before is not None else []
        after_video_tokens = (
            first_number(after, "expanded_video_tokens", "expanded_video_token_count") if after else None
        )
        if expected and after_video_tokens is not None and not almost_equal(sum(expected), after_video_tokens):
            issues.append("expected_vs_expanded_video_tokens")

        for issue in set(issues):
            mismatch_counts[issue] += 1
        if issues:
            checks.append(
                {
                    "source": rollout["_source"],
                    "line": rollout["_line"],
                    "dedup_prompt_hash": dedup_hash,
                    "expanded_prompt_hash": expanded_hash,
                    "issues": sorted(set(issues)),
                }
            )

    before_seconds = Counter()
    before_grids = Counter()
    missing_seconds = 0
    for record in before_records:
        second = get_mm_field(record, "second_per_grid_ts", "video_second_per_grid", "second_per_grids")
        grid = get_mm_field(record, "video_grid_thw")
        if second is None:
            missing_seconds += 1
        else:
            before_seconds[normalized(second)] += 1
        if grid is not None:
            before_grids[normalized(grid)] += 1

    return {
        "rollout_count": len(rollouts),
        "before_count": len(before_records),
        "after_count": len(after_records),
        "rollouts_with_dedup_hash": sum(first_value(r, "dedup_prompt_hash") is not None for r in rollouts),
        "rollouts_with_expanded_hash": sum(first_value(r, "expanded_prompt_hash") is not None for r in rollouts),
        "matched_before_by_hash": matched_before,
        "matched_after_by_hash": matched_after,
        "mismatch_counts": dict(mismatch_counts),
        "mismatches": checks,
        "missing_second_per_grid": missing_seconds,
        "video_grid_thw": dict(before_grids),
        "second_per_grid": dict(before_seconds),
    }


def classify(report: dict[str, Any]) -> list[str]:
    actor = report["actor"]
    cross = report["rollout_vllm"]
    offpolicy = report["offpolicy"]
    metric_summaries = report["training_metrics"]["summaries"]
    conclusions = []
    missing_record_types = [
        name
        for name, count in (
            ("ACTOR_VIDEO_DEBUG", actor["count"]),
            ("ROLLOUT_INPUT_DEBUG", cross["rollout_count"]),
            ("VLLM_MM_BEFORE_DEBUG", cross["before_count"]),
            ("VLLM_MM_AFTER_DEBUG", cross["after_count"]),
        )
        if count == 0
    ]
    hashes_incomplete = (
        cross["rollout_count"] > 0
        and (
            cross["rollouts_with_dedup_hash"] == 0
            or cross["rollouts_with_expanded_hash"] == 0
            or cross["matched_before_by_hash"] == 0
            or cross["matched_after_by_hash"] == 0
        )
    )
    structural_check_complete = not missing_record_types and not hashes_incomplete

    if missing_record_types:
        conclusions.append(
            "结构检查不完整，缺少 " + ", ".join(missing_record_types) + "；不能据此判定 token/grid/hash/长度一致。"
        )
    if cross["rollout_count"] and cross["rollouts_with_dedup_hash"] == 0:
        conclusions.append("ROLLOUT_INPUT_DEBUG 没有 dedup_prompt_hash，无法与 vLLM 展开前记录对齐。")
    if cross["rollouts_with_expanded_hash"] and cross["matched_after_by_hash"] == 0:
        conclusions.append("rollout 有 expanded_prompt_hash，但没有匹配的 vLLM 展开后记录；先补齐 vLLM 日志。")

    if actor["token_mismatches"]:
        conclusions.append("Actor 侧 expected/actual video token 数不一致：优先检查 HF processor、grid_thw 和训练输入拼接。")
    if cross["mismatches"]:
        conclusions.append("rollout 与 vLLM 多模态展开结果不一致：优先检查 placeholder 更新、dedup 和 prompt 展开。")
    if cross["before_count"] and cross["missing_second_per_grid"]:
        conclusions.append("部分 vLLM 请求缺少 second_per_grid_ts：检查视频时间参数是否从 processor 一直传到 vLLM。")

    actor_seconds = set(actor["second_per_grid"])
    vllm_seconds = set(cross["second_per_grid"])
    if actor_seconds and vllm_seconds and actor_seconds != vllm_seconds:
        conclusions.append("Actor 与 vLLM 观察到的 second_per_grid 值集合不同：这是 mRoPE position_ids 不一致的强信号。")
    actor_grids = set(actor["video_grid_thw"])
    vllm_grids = set(cross["video_grid_thw"])
    if actor_grids and vllm_grids and actor_grids != vllm_grids:
        conclusions.append("Actor 与 vLLM 观察到的 video_grid_thw 值集合不同：检查抽帧/resize/processor 参数。")

    has_structural_error = bool(actor["token_mismatches"] or cross["mismatches"])
    if structural_check_complete and not has_structural_error and offpolicy["abnormal_count"]:
        conclusions.append(
            "已记录的 token/grid/hash/长度未发现硬不一致，但仍有 off-policy 异常；"
            "下一组只切换 enforce_eager=true，排查 NPU graph/算子数值路径。"
        )
    if offpolicy["all_records_above_threshold"]:
        conclusions.append(
            "所有 OFFPOLICY_SAMPLE 都超过阈值，说明这些记录很可能是阈值筛选后的异常子集；"
            "其均值不能与全批次 log_ppl_abs_diff 直接比较。"
        )
    if offpolicy["with_first_position"] and offpolicy["first_position_zero_count"] == offpolicy["with_first_position"]:
        conclusions.append(
            "全部已记录异常都从 response 第 0 个 token 开始，优先怀疑 prompt/prefill 多模态上下文、mRoPE 或首步 logits，"
            "而不是长回复误差累积。"
        )
    if offpolicy["count"] and offpolicy["with_stable_problem_id"] == 0:
        conclusions.append("OFFPOLICY_SAMPLE 未包含稳定 problem_id；batch_index 跨 step 不代表同一个样本，暂时不能判断是否固定视频复现。")
    pearson = metric_summaries.get("pearson")
    ppl = metric_summaries.get("log_ppl_abs_diff")
    if pearson and pearson["min"] < 0.995:
        conclusions.append(
            f"全局 Pearson 最低为 {pearson['min']:.6g}（step={pearson['min_step']}）；结合逐样本结果判断是固定问题样本还是全批次偏差。"
        )
    if ppl and ppl["max"] > offpolicy["threshold"]:
        conclusions.append(
            f"全局 log_ppl_abs_diff 最高为 {ppl['max']:.6g}（step={ppl['max_step']}），"
            f"已超过分析阈值 {offpolicy['threshold']:.6g}。"
        )
    if not conclusions:
        conclusions.append("当前日志未发现硬不一致；若仍有 Pearson/PPL 异常，请确认五类日志来自同一次、未 resume 的运行。")
    return conclusions


def build_report(records: dict[str, list[dict[str, Any]]], meta: dict[str, Any], threshold: float) -> dict[str, Any]:
    report = {
        "input": meta,
        "record_counts": {tag: len(values) for tag, values in records.items()},
        "actor": analyze_actor(records["ACTOR_VIDEO_DEBUG"]),
        "offpolicy": analyze_offpolicy(records["OFFPOLICY_SAMPLE"], threshold),
        "training_metrics": analyze_training_metrics(records["TRAIN_METRICS"]),
        "rollout_vllm": analyze_rollout_vllm(
            records["ROLLOUT_INPUT_DEBUG"],
            records["VLLM_MM_BEFORE_DEBUG"],
            records["VLLM_MM_AFTER_DEBUG"],
        ),
    }
    report["conclusions"] = classify(report)
    return report


def fmt_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def print_report(report: dict[str, Any], top: int) -> None:
    meta = report["input"]
    print("=== Parsed records ===")
    print(
        f"files={meta['files']} lines={meta['lines']} tagged={meta['tagged_lines']} "
        f"parse_errors={len(meta['parse_errors'])}"
    )
    for tag, count in report["record_counts"].items():
        print(f"  {tag}: {count}")

    actor = report["actor"]
    cross = report["rollout_vllm"]
    print("\n=== Structural checks ===")
    print(f"actor expected/actual token mismatches: {len(actor['token_mismatches'])}/{actor['count']}")
    print(
        "rollout -> vLLM hash coverage: "
        f"before {cross['matched_before_by_hash']}/{cross['rollouts_with_dedup_hash']}, "
        f"after {cross['matched_after_by_hash']}/{cross['rollouts_with_expanded_hash']}"
    )
    print(f"rollout/vLLM mismatches: {len(cross['mismatches'])} {cross['mismatch_counts']}")
    print(f"vLLM records missing second_per_grid: {cross['missing_second_per_grid']}/{cross['before_count']}")
    print(f"actor use_audio_in_video: {actor['use_audio_in_video']}")
    print(f"actor second_per_grid variants: {actor['second_per_grid']}")
    print(f"vLLM second_per_grid variants: {cross['second_per_grid']}")
    print(f"actor video_grid_thw variants: {actor['video_grid_thw']}")
    print(f"vLLM video_grid_thw variants: {cross['video_grid_thw']}")

    training_metrics = report["training_metrics"]
    print("\n=== Training metric trends ===")
    if not training_metrics["summaries"]:
        print("no recognized training metrics found")
    for name, summary in training_metrics["summaries"].items():
        print(
            f"  {name}: n={summary['count']} first={fmt_number(summary['first'])} "
            f"last={fmt_number(summary['last'])} change={fmt_number(summary['change'])} "
            f"min={fmt_number(summary['min'])}@{summary['min_step']} "
            f"max={fmt_number(summary['max'])}@{summary['max_step']}"
        )

    offpolicy = report["offpolicy"]
    print("\n=== Logged off-policy sample records ===")
    print(
        f"samples={offpolicy['count']} with_abs_diff={offpolicy['with_abs_diff']} "
        f">{offpolicy['threshold']:.4g}={offpolicy['abnormal_count']} "
        f"mean={fmt_number(offpolicy['mean'])} median={fmt_number(offpolicy['median'])} "
        f"max={fmt_number(offpolicy['max'])}"
    )
    if offpolicy["all_records_above_threshold"]:
        print("note: every record exceeds the threshold; this is probably a filtered subset, not the full batch")
    print(
        "first mismatch at response position 0: "
        f"{offpolicy['first_position_zero_count']}/{offpolicy['with_first_position']}; "
        f"records with stable problem_id: {offpolicy['with_stable_problem_id']}/{offpolicy['count']}"
    )
    print(f"\nWorst {min(top, len(offpolicy['worst_samples']))} samples:")
    for item in offpolicy["worst_samples"][:top]:
        print(
            f"  problem={item['problem_id']} batch={item['batch_index']} "
            f"abs={fmt_number(item['seq_log_ppl_abs_diff'])} "
            f"pearson={fmt_number(item['pearson_corr'])} token_max={fmt_number(item['token_abs_max'])} "
            f"first_pos>0.05={item['first_position_abs_diff_gt_0.05']} "
            f"({item['source']}:{item['line']})"
        )
    print(f"\nWorst {min(top, len(offpolicy['by_problem']))} problem IDs:")
    for item in offpolicy["by_problem"][:top]:
        print(
            f"  {item['problem_id']}: count={item['count']} abnormal={item['abnormal_count']} "
            f"mean={fmt_number(item['mean'])} max={fmt_number(item['max'])}"
        )

    if meta["parse_errors"]:
        print("\nFirst parse errors:")
        for error in meta["parse_errors"][:top]:
            print(f"  {error}")

    print("\n=== Suggested interpretation ===")
    for conclusion in report["conclusions"]:
        print(f"- {conclusion}")


def main() -> int:
    args = parse_args()
    records, meta = parse_records(args.paths)
    report = build_report(records, meta, args.offpolicy_threshold)
    print_report(report, max(args.top, 0))
    if args.json_output:
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
