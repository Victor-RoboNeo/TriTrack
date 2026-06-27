#!/usr/bin/env python3
"""
Create a GMT train/test/delta metrics matrix from two CSVs.

Input CSVs are expected to be produced by:
  scripts/statistics/wandb_avg_last_n_metrics.py

Output CSV shape (horizontal metrics, vertical splits):
  split,<metric_1>,<metric_2>,...
  train,<train_mean>,...
  test,<test_mean>,...
  delta_test_minus_train,<test_mean - train_mean>,...
  delta_over_train_percent,(test_mean - train_mean) / train_mean * 100,...

If you pass ``--metrics A B C``, the script uses only those metric names
as columns (missing metrics are left blank).

python scripts/statistics/gmt_train_test_delta_metrics.py \
  --train-csv gmt_train_last100.csv \
  --test-csv gmt_eval_mosaic_last100.csv \
  --out gmt_matrix_test_on_mosaic.csv \
  --short-metric-names \
  --metrics "Metrics/motion/error_joint_pos" "Metrics/motion/error_joint_vel" \
    "Metrics/motion/error_body_pos" "Metrics/motion/error_body_rot" \
    "Metrics/motion/error_body_lin_vel" "Metrics/motion/error_body_ang_vel" \
    "Metrics/motion/error_anchor_pos" "Metrics/motion/error_anchor_rot" \
    "Metrics/motion/error_anchor_lin_vel" "Metrics/motion/error_anchor_ang_vel" \
    "Train/mean_episode_length" "Train/mean_reward"
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from typing import Dict, Optional, Tuple


def _read_metric_csv(path: str) -> Dict[str, float]:
    """
    Parse the CSV written by wandb_avg_last_n_metrics.py.

    That file layout is:
      key,value
      <metadata...>
      <blank line>
      metric,mean_last_n
      <metric>,<value>
    """
    metrics: Dict[str, float] = {}

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        in_metric_section = False
        for row in reader:
            if not row:
                continue
            if len(row) >= 2 and row[0].strip() == "metric":
                in_metric_section = True
                continue
            if not in_metric_section:
                continue
            if len(row) < 2:
                continue
            key = row[0].strip()
            raw_val = row[1].strip()
            if not key or raw_val == "":
                continue
            try:
                metrics[key] = float(raw_val)
            except ValueError:
                # Skip non-numeric values (should be rare / only if CSV got edited).
                continue

    return metrics


def _nan_or_value(x: object) -> str:
    """Format a cell value for CSV.

    - None / NaN => blank
    - strings => returned as-is (needed for values like "-3.61%")
    - numbers => float repr
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, float) and math.isnan(x):
        return ""
    # Fallback for numeric types.
    try:
        return repr(float(x))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(x)


def _short_metric_name(full_name: str) -> str:
    """If metric is like 'Train/mean_reward', return 'mean_reward'."""
    if "/" not in full_name:
        return full_name
    return full_name.rsplit("/", 1)[-1]


def main() -> None:
    p = argparse.ArgumentParser(description="Pivot GMT train/test metric CSVs into a matrix with delta.")
    p.add_argument("--train-csv", required=True, help="CSV from wandb_avg_last_n_metrics.py for training.")
    p.add_argument("--test-csv", required=True, help="CSV from wandb_avg_last_n_metrics.py for testing/eval.")
    p.add_argument("--out", "-o", required=True, help="Output CSV path.")
    p.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional explicit list of metric names to include as columns. If omitted, metrics are auto-discovered.",
    )
    p.add_argument(
        "--mode",
        choices=("intersection", "union"),
        default="intersection",
        help="Whether to keep metrics present in both CSVs (intersection) or either CSV (union).",
    )
    p.add_argument(
        "--metrics-regex",
        default=None,
        help="Optional regex to filter metric columns (matched against the metric name).",
    )
    p.add_argument(
        "--short-metric-names",
        action="store_true",
        help="When writing the CSV header, shorten metric names by keeping only the substring after the last '/'.",
    )
    args = p.parse_args()

    train = _read_metric_csv(args.train_csv)
    test = _read_metric_csv(args.test_csv)

    if not train:
        print(f"[error] No metrics parsed from train CSV: {args.train_csv}", file=sys.stderr)
        sys.exit(1)
    if not test:
        print(f"[error] No metrics parsed from test CSV: {args.test_csv}", file=sys.stderr)
        sys.exit(1)

    if args.metrics is not None and len(args.metrics) > 0:
        requested = list(args.metrics)
        full_available = set(train.keys()) | set(test.keys())
        metrics: list[str] = []
        for req in requested:
            if req in full_available:
                metrics.append(req)
                continue
            # If the user asked for short labels, also allow selecting by the short name.
            if args.short_metric_names:
                short = _short_metric_name(req)
                matches = [k for k in full_available if _short_metric_name(k) == short]
                if len(matches) == 1:
                    metrics.append(matches[0])
                    continue
                if len(matches) > 1:
                    print(
                        f"[warn] Metric short name {short!r} is ambiguous; skipping {req!r}.",
                        file=sys.stderr,
                    )
                    continue
            print(
                f"[warn] Metric not found or not numeric (skipped): {req!r}",
                file=sys.stderr,
            )
        metrics = sorted(set(metrics))
    else:
        metrics_set = set(train.keys())
        if args.mode == "intersection":
            metrics_set &= set(test.keys())
        else:
            metrics_set |= set(test.keys())

        if args.metrics_regex:
            rx = re.compile(args.metrics_regex)
            metrics_set = {m for m in metrics_set if rx.search(m)}
        metrics = sorted(metrics_set)

    if not metrics:
        print("[error] After filtering, there are no metrics to write.", file=sys.stderr)
        sys.exit(1)

    # Header labels: optionally shorten by removing the prefix up to last '/'.
    # If shortening would cause collisions, fall back to full names for the collided metrics.
    if args.short_metric_names:
        short = [_short_metric_name(m) for m in metrics]
        counts: Dict[str, int] = {}
        for s in short:
            counts[s] = counts.get(s, 0) + 1
        output_metric_names = [m if counts[s] > 1 else s for m, s in zip(metrics, short)]
    else:
        output_metric_names = metrics

    # Build a dense matrix: columns are metrics, rows are splits.
    # delta = test - train (blank if either is missing).
    # delta_over_train_percent = delta / train * 100 (blank if train is missing or 0).
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", *output_metric_names])

        def row_for(split: str) -> Tuple[object, ...]:
            if split == "train":
                return tuple(train.get(m) for m in metrics)
            if split == "test":
                return tuple(test.get(m) for m in metrics)
            if split == "delta_test_minus_train":
                out: list[Optional[float]] = []
                for m in metrics:
                    if m not in train or m not in test:
                        out.append(None)
                    else:
                        out.append(test[m] - train[m])
                return tuple(out)
            if split == "delta_over_train_percent":
                perc_cells: list[object] = []
                for m in metrics:
                    if m not in train or m not in test:
                        perc_cells.append(None)
                        continue
                    denom = train[m]
                    if denom == 0:
                        perc_cells.append(None)
                        continue
                    value = (test[m] - train[m]) / denom * 100.0
                    perc_cells.append(f"{value:.2f}%")
                return tuple(perc_cells)
            raise ValueError(f"Unknown split row: {split}")

        for split in ("train", "test", "delta_over_train_percent"):
            vals = row_for(split)
            w.writerow([split, *[_nan_or_value(v) for v in vals]])

    print(
        f"Wrote {args.out}: {len(metrics)} metrics, mode={args.mode}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

