#!/usr/bin/env python3
"""Summarize rollout/actor probability-parity metrics from NPU logs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


METRICS = (
    "training/rollout_actor_probs_pearson_corr",
    "training/rollout_probs_diff_mean",
    "training/rollout_probs_diff_max",
    "training/rollout_probs_diff_std",
    "rollout_corr/log_ppl_diff",
)
NUMBER = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan)"
PATTERNS = {
    metric: re.compile(rf"[\"']?{re.escape(metric)}[\"']?\s*[:=]\s*({NUMBER})", re.IGNORECASE)
    for metric in METRICS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="A log file or a result directory containing run.log files")
    parser.add_argument("--csv", type=Path, help="Optional output CSV path")
    return parser.parse_args()


def find_logs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.log"))


def parse_log(path: Path) -> dict[str, list[float]]:
    values = {metric: [] for metric in METRICS}
    text = path.read_text(encoding="utf-8", errors="replace")
    for metric, pattern in PATTERNS.items():
        values[metric] = [float(match.group(1)) for match in pattern.finditer(text)]
    return values


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.8g}"


def case_name(log: Path, root: Path) -> str:
    if root.is_file():
        return log.stem
    relative = log.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else log.stem


def main() -> int:
    args = parse_args()
    logs = find_logs(args.path)
    if not logs:
        print(f"No .log files found under {args.path}")
        return 1

    rows: list[dict[str, object]] = []
    for log in logs:
        parsed = parse_log(log)
        pearson = finite(parsed[METRICS[0]])
        diff_mean = finite(parsed[METRICS[1]])
        diff_max = finite(parsed[METRICS[2]])
        rows.append(
            {
                "case": case_name(log, args.path),
                "samples": len(pearson),
                "pearson_first": pearson[0] if pearson else None,
                "pearson_last": pearson[-1] if pearson else None,
                "pearson_min": min(pearson) if pearson else None,
                "pearson_max": max(pearson) if pearson else None,
                "prob_diff_mean_last": diff_mean[-1] if diff_mean else None,
                "prob_diff_max_last": diff_max[-1] if diff_max else None,
                "log": str(log),
            }
        )

    headers = (
        "case",
        "samples",
        "pearson_first",
        "pearson_last",
        "pearson_min",
        "pearson_max",
        "prob_diff_mean_last",
        "prob_diff_max_last",
    )
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        cells = [str(row[key]) if key in ("case", "samples") else fmt(row[key]) for key in headers]
        print("| " + " | ".join(cells) + " |")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
