#!/usr/bin/env python3
"""
Aggregate multiple wandb_avg_last_n_metrics.py CSVs into one table.

Each input CSV becomes one row. Use the same explicit ``--metrics`` list as
``gmt_train_test_delta_metrics.py`` (defaults below); delta rows are omitted.

Input CSV layout is the same as for gmt_train_test_delta_metrics.py (see
``_read_metric_csv`` there).

Examples::

    python scripts/statistics/pulse_train_test_multiple_run_metrics.py \\
        --csv-dir csv/ \\
        --out pulse_all_runs.csv \\
        --short-metric-names

    # Same metric names as gmt_train_test_delta_metrics.py example:
    python scripts/statistics/pulse_train_test_multiple_run_metrics.py \
        --csv-dir wandb_exports/PULSE \
        --out wandb_exports/PULSE/stats_24k.csv \
        --short-metric-names \
        --metrics "Metrics/motion/error_joint_pos" "Metrics/motion/error_joint_vel" \
        "Metrics/motion/error_body_pos" "Metrics/motion/error_body_rot" \
        "Metrics/motion/error_body_lin_vel" "Metrics/motion/error_body_ang_vel" \
        "Metrics/motion/error_anchor_pos" "Metrics/motion/error_anchor_rot" \
        "Metrics/motion/error_anchor_lin_vel" "Metrics/motion/error_anchor_ang_vel" \
        "Episode/mdp_termination_success_rate" \
        "Train/mean_episode_length" "Train/mean_reward"

    # Include all dense motion metrics without listing each joint metric.
    python scripts/statistics/pulse_train_test_multiple_run_metrics.py \
        --csv-dir wandb_exports/dense_metrics \
        --out wandb_exports/dense_metrics/all_motion.csv \
        --short-metric-names \
        --metrics "Metrics/motion/*"

    
    python scripts/statistics/pulse_train_test_multiple_run_metrics.py \
        --csv-dir wandb_exports/vr_tracking_rl \
        --out wandb_exports/vr_tracking_rl/stats.csv \
        --short-metric-names \
        --metrics "Metrics/motion/error_keypoint_pos_both_ee" "Metrics/motion/error_keypoint_vel_both_ee" \
        "Metrics/motion/error_keypoint_pos_left_ee" "Metrics/motion/error_keypoint_vel_left_ee" \
        "Metrics/motion/error_keypoint_pos_right_ee" "Metrics/motion/error_keypoint_vel_right_ee" \
        "Metrics/motion/error_joint_pos" "Metrics/motion/error_joint_vel" \
        "Metrics/motion/error_body_pos" "Metrics/motion/error_body_rot" \
        "Metrics/motion/error_body_lin_vel" "Metrics/motion/error_body_ang_vel" \
        "Metrics/motion/error_anchor_pos" "Metrics/motion/error_anchor_rot" \
        "Metrics/motion/error_anchor_lin_vel" "Metrics/motion/error_anchor_ang_vel" \
        "Episode/mdp_termination_success_rate" \
        "Train/mean_episode_length" "Train/mean_reward"
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Same default set as the gmt_train_test_delta_metrics.py docstring example.
DEFAULT_METRICS: Tuple[str, ...] = (
    "Metrics/motion/error_joint_pos",
    "Metrics/motion/error_joint_vel",
    "Metrics/motion/error_body_pos",
    "Metrics/motion/error_body_rot",
    "Metrics/motion/error_body_lin_vel",
    "Metrics/motion/error_body_ang_vel",
    "Metrics/motion/error_anchor_pos",
    "Metrics/motion/error_anchor_rot",
    "Metrics/motion/error_anchor_lin_vel",
    "Metrics/motion/error_anchor_ang_vel",
    "Metrics/motion/error_keypoint_pos_left_ee",
    "Metrics/motion/error_keypoint_vel_left_ee",
    "Metrics/motion/error_keypoint_pos_right_ee",
    "Metrics/motion/error_keypoint_vel_right_ee",
    "Metrics/motion/error_keypoint_pos_both_ee",
    "Metrics/motion/error_keypoint_vel_both_ee",
    "Train/mean_episode_length",
    "Train/mean_reward",
)


def _read_metric_csv(path: Path) -> Dict[str, float]:
    """Parse CSV written by wandb_avg_last_n_metrics.py (same as gmt_train_test_delta_metrics)."""
    metrics: Dict[str, float] = {}

    with path.open("r", newline="") as f:
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
                continue

    return metrics


def _read_metadata_csv(path: Path) -> Dict[str, str]:
    """Parse key,value header section before the metric block."""
    meta: Dict[str, str] = {}
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                break
            if len(row) >= 2 and row[0].strip() == "metric":
                break
            if len(row) < 2:
                continue
            k, v = row[0].strip(), row[1].strip()
            if k == "key" and v == "value":
                continue
            if k:
                meta[k] = v
    return meta


def _nan_or_value(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, float) and math.isnan(x):
        return ""
    try:
        return repr(float(x))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(x)


def _short_metric_name(full_name: str) -> str:
    if "/" not in full_name:
        return full_name
    return full_name.rsplit("/", 1)[-1]


def _row_label(path: Path, label_from: str, meta: Dict[str, str]) -> str:
    if label_from == "filename":
        return path.name
    if label_from == "stem":
        return path.stem
    if label_from in meta:
        return meta[label_from]
    return path.stem


def _expand_requested_metrics(requested: Tuple[str, ...], full_available: set[str]) -> list[str]:
    """Expand wildcard metric selectors like ``prefix/*`` against available keys."""
    expanded: list[str] = []
    seen: set[str] = set()
    for req in requested:
        if req.endswith("*"):
            prefix = req[:-1]
            matches = sorted(k for k in full_available if k.startswith(prefix))
            if not matches:
                print(f"[warn] Metric wildcard matched nothing (skipped): {req!r}", file=sys.stderr)
                continue
            for m in matches:
                if m not in seen:
                    expanded.append(m)
                    seen.add(m)
            continue
        if req not in seen:
            expanded.append(req)
            seen.add(req)
    return expanded


def main() -> None:
    p = argparse.ArgumentParser(
        description="Combine per-run metric CSVs into one row per file (gmt-style columns, no deltas)."
    )
    p.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("csv"),
        help="Directory containing *.csv files from wandb_avg_last_n_metrics.py (default: csv/).",
    )
    p.add_argument("--out", "-o", required=True, help="Output CSV path.")
    p.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help=(
            "Metric column names (full W&B keys), same as gmt_train_test_delta_metrics.py. "
            f"If omitted, uses the default {len(DEFAULT_METRICS)} metrics from that script's example. "
            "Supports wildcard prefix selection, e.g. 'Metrics/motion/*'."
        ),
    )
    p.add_argument(
        "--short-metric-names",
        action="store_true",
        help="Shorten metric headers to the substring after the last '/'.",
    )
    p.add_argument(
        "--label-from",
        default="stem",
        choices=("filename", "stem", "run_name", "run_id", "run_path"),
        help="First column value: file name, stem, or a metadata key from the CSV header.",
    )
    args = p.parse_args()

    csv_dir: Path = args.csv_dir
    if not csv_dir.is_dir():
        print(f"[error] Not a directory: {csv_dir}", file=sys.stderr)
        sys.exit(1)

    paths = sorted(csv_dir.glob("*.csv"))
    if not paths:
        print(f"[error] No *.csv under {csv_dir}", file=sys.stderr)
        sys.exit(1)

    per_file: List[Tuple[Path, Dict[str, float], Dict[str, str]]] = []
    for path in paths:
        meta = _read_metadata_csv(path)
        m = _read_metric_csv(path)
        per_file.append((path, m, meta))

    if args.metrics is not None and len(args.metrics) == 0:
        print("[error] --metrics was passed but no metric names were given.", file=sys.stderr)
        sys.exit(1)

    requested: Tuple[str, ...] = tuple(args.metrics) if args.metrics is not None else DEFAULT_METRICS

    full_available: set[str] = set()
    for _, m, _ in per_file:
        full_available |= set(m.keys())

    requested_expanded = _expand_requested_metrics(requested, full_available)

    metrics: list[str] = []
    for req in requested_expanded:
        if req in full_available:
            metrics.append(req)
            continue
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
        print(f"[warn] Metric not found (skipped): {req!r}", file=sys.stderr)

    if not metrics:
        print("[error] After filtering, there are no metrics to write.", file=sys.stderr)
        sys.exit(1)

    if args.short_metric_names:
        short = [_short_metric_name(m) for m in metrics]
        counts: Dict[str, int] = {}
        for s in short:
            counts[s] = counts.get(s, 0) + 1
        output_metric_names = [m if counts[s] > 1 else s for m, s in zip(metrics, short)]
    else:
        output_metric_names = metrics

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", *output_metric_names])
        for path, m, meta in per_file:
            label = _row_label(path, args.label_from, meta)
            row = [label, *[_nan_or_value(m.get(metric)) for metric in metrics]]
            w.writerow(row)

    print(
        f"Wrote {args.out}: {len(paths)} runs, {len(metrics)} metrics",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
