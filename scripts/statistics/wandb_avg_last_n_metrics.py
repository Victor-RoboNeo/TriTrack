#!/usr/bin/env python3
"""Average selected W&B metrics over the last N logged steps and export to CSV.

Typical use for MOSAIC GMT training (same metric names as residual / mosaic runners):
  Train/mean_reward, Train/mean_episode_length, Episode/*, Loss/*, Perf/*, Policy/*

Eval jobs that use ``--logger wandb`` log the same scalar groups under a separate run
(with its own run id / name).

Requires ``WANDB_API_KEY`` (or ``wandb login``) for private projects.

Examples::

    # 1. training 
    python scripts/statistics/wandb_avg_last_n_metrics.py \
        --run-path hzlhammer/latent_residual_vr_tracking_rl/mc8k4p8z \
        --start-step 10000 \
        --last-n 100 \
        --out ./wandb_exports/vr_tracking_rl/train.csv

    # 2. testing
    python scripts/statistics/wandb_avg_last_n_metrics.py \
        --run-path hzlhammer/latent_residual_vr_tracking_rl/fke2puq2 \
        --last-n 100 \
        --out ./wandb_exports/vr_tracking_rl/test.csv

    # 4. average first N logged rows with _step >= START_STEP (dedicated script)
    python scripts/statistics/wandb_avg_last_n_metrics.py \
        --run-path hzlhammer/Residual_Latent_Distill_2B/vzci0q1k \
        --start-step 12000 \
        --last-n 100 \
        --out ./wandb_exports/keypoint_tracker/train_12k.csv

    # 5. prior metrics
    python scripts/statistics/wandb_avg_last_n_metrics.py \
        --run-path hzlhammer/evaluate_prior_rollout/0qqu2jn1 \
        --last-n 15 \
        --out pulse_prior_last15_19000.csv

    # 6. every run in a project (one CSV per run under --out-dir)
    python scripts/statistics/wandb_avg_last_n_metrics.py \
        --project-path hzlhammer/PULSE_Distill_eval \
        --last-n 100 \
        --out-dir ./wandb_exports/PULSE_Distill \

"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import sys
from typing import Any, cast
from urllib.parse import urlparse

try:
    import wandb
except ModuleNotFoundError:
    print("Install wandb: pip install wandb", file=sys.stderr)
    raise


def normalize_project_path(project_path: str) -> str:
    """Accept ``entity/project`` (2 segments)."""
    s = project_path.strip().rstrip("/")
    chunks = [c for c in s.split("/") if c]
    if len(chunks) != 2:
        raise ValueError(
            f"Project path must be entity/project (2 segments), got: {project_path!r}"
        )
    return f"{chunks[0]}/{chunks[1]}"


def normalize_run_path(run_path: str) -> str:
    """Accept ``entity/project/run_id`` or a ``wandb.ai/.../runs/<id>`` URL."""
    s = run_path.strip().rstrip("/")
    if "wandb.ai" in s or s.startswith("http"):
        u = urlparse(s if "://" in s else f"https://{s}")
        parts = [p for p in u.path.strip("/").split("/") if p]
        if len(parts) >= 4 and parts[2] == "runs":
            return f"{parts[0]}/{parts[1]}/{parts[3]}"
        raise ValueError(
            f"Could not parse run URL (expected .../entity/project/runs/run_id): {run_path!r}"
        )
    chunks = [c for c in s.split("/") if c]
    if len(chunks) != 3:
        raise ValueError(
            f"Run path must be entity/project/run_id (3 segments), got: {run_path!r}"
        )
    return s


def _gmt_metric_regex() -> re.Pattern[str]:
    """Scalar groups used by rsl_rl mosaic / on-policy runners + optional RND."""
    return re.compile(r"^(Train|Episode|Loss|Perf|Policy|Rnd)/")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def load_history_rows(
    run: Any,
    metric_keys: list[str] | None,
) -> list[dict[str, Any]]:
    """All logged steps via ``scan_history`` (exact series; no downsampling)."""
    if metric_keys:
        keys = list(dict.fromkeys(list(metric_keys) + ["_step"]))
        it = run.scan_history(keys=keys)
    else:
        it = run.scan_history()
    return list(it)


def select_metric_names(
    rows: list[dict[str, Any]],
    include_regex: re.Pattern[str] | None,
) -> list[str]:
    """Union of numeric keys across rows, minus wandb internals, with optional regex filter."""
    skip = {"_step", "_runtime", "_timestamp"}
    names: set[str] = set()
    for row in rows:
        for k, v in row.items():
            if k in skip or k.startswith("system/"):
                continue
            if not _is_number(v):
                continue
            if include_regex is not None and not include_regex.search(k):
                continue
            names.add(k)
    return sorted(names)


def _step_value(r: dict[str, Any]) -> float:
    s = r.get("_step", 0)
    if _is_number(s):
        return float(cast(int | float, s))
    return 0.0


def _mean_over_rows(
    slice_rows: list[dict[str, Any]], metric_names: list[str]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in metric_names:
        vals: list[float] = []
        for r in slice_rows:
            v = r.get(name)
            if _is_number(v) and not (isinstance(v, float) and math.isnan(v)):
                vals.append(float(cast(int | float, v)))
        if vals:
            out[name] = float(statistics.mean(vals))
    return out


def mean_last_n(
    rows: list[dict[str, Any]], metric_names: list[str], last_n: int
) -> tuple[int, dict[str, float]]:
    """Sort by ``_step``, take last ``last_n`` rows, return means (skip NaN per metric)."""
    if not rows:
        return 0, {}

    ordered = sorted(rows, key=_step_value)
    tail = ordered[-last_n:] if last_n else ordered
    n_used = len(tail)
    return n_used, _mean_over_rows(tail, metric_names)


def mean_from_start_step(
    rows: list[dict[str, Any]],
    metric_names: list[str],
    start_step: float,
    n: int,
) -> tuple[int, dict[str, float]]:
    """Sort by ``_step``, keep rows with ``_step >= start_step``, then first ``n`` rows."""
    if not rows:
        return 0, {}

    ordered = sorted(rows, key=_step_value)
    subset = [r for r in ordered if _step_value(r) >= start_step]
    head = subset[:n] if n else subset
    n_used = len(head)
    return n_used, _mean_over_rows(head, metric_names)


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description
        or "Average W&B metrics over the last N logged steps and write CSV."
    )
    p.add_argument(
        "--run-path",
        default=None,
        help="entity/project/run_id or https://wandb.ai/entity/project/runs/run_id",
    )
    p.add_argument(
        "--project-path",
        default=None,
        help="entity/project: process every run in that project (writes one CSV per run under --out-dir).",
    )
    p.add_argument(
        "--last-n",
        type=int,
        default=100,
        help="Number of logged rows to average: tail (default script) or first N at/after --start-step.",
    )
    p.add_argument(
        "--start-step",
        type=float,
        default=None,
        help="If set (use wandb_avg_from_step_metrics.py or pass here): average the first --last-n "
        "logged rows with _step >= this value.",
    )
    p.add_argument(
        "--out",
        "-o",
        type=str,
        default=None,
        help="Output CSV path (required with --run-path).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for per-run CSV files (required with --project-path).",
    )
    p.add_argument(
        "--only-state",
        choices=("any", "finished", "running", "crashed"),
        default="any",
        help="When using --project-path, only include runs in this W&B state (default: finished).",
    )
    p.add_argument(
        "--name-regex",
        type=str,
        default=None,
        help="When using --project-path, only include runs whose display name matches this regex.",
    )
    p.add_argument(
        "--gmt-metrics",
        action="store_true",
        help="Only include metrics under Train/, Episode/, Loss/, Perf/, Policy/, Rnd/ (runner scalars).",
    )
    p.add_argument(
        "--include-regex",
        type=str,
        default=None,
        help="If set, only average metrics whose names match this regex (overrides --gmt-metrics).",
    )
    p.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Explicit metric names to fetch (reduces scan size). If omitted, all keys are scanned.",
    )
    return p


def _safe_filename_part(s: str) -> str:
    """Filesystem-safe fragment from run name."""
    out = re.sub(r"[^\w.\-]+", "_", s.strip())
    return out[:200] if len(out) > 200 else out


def process_one_run(args: argparse.Namespace, run_path: str, out_path: str) -> bool:
    """Average metrics for a single run; write CSV. Returns True on success."""
    run_path = normalize_run_path(run_path)
    include_re: re.Pattern[str] | None
    if args.include_regex:
        include_re = re.compile(args.include_regex)
    elif args.gmt_metrics:
        include_re = _gmt_metric_regex()
    else:
        include_re = None

    api = wandb.Api()
    run = api.run(run_path)

    metric_keys = list(args.metrics) if args.metrics else None
    rows = load_history_rows(run, metric_keys)
    if not rows:
        print(f"No history rows for run {run_path}", file=sys.stderr)
        return False

    discovered = select_metric_names(rows, include_re)
    if args.metrics:
        metric_names = [m for m in args.metrics if m in discovered]
        missing = [m for m in args.metrics if m not in discovered]
        if missing:
            print(f"[warn] metrics not found or not numeric (skipped): {missing}", file=sys.stderr)
    else:
        metric_names = discovered
    if not metric_names:
        print(
            "No numeric metric columns left after filtering. "
            "Try dropping --gmt-metrics or adjust --include-regex.",
            file=sys.stderr,
        )
        return False

    if args.start_step is not None:
        n_used, means = mean_from_start_step(
            rows, metric_names, args.start_step, args.last_n
        )
        value_header = "mean_from_start_step"
        empty_msg = "No values to average in the selected window (check --start-step and --last-n)."
    else:
        n_used, means = mean_last_n(rows, metric_names, args.last_n)
        value_header = "mean_last_n"
        empty_msg = "No values to average in the selected tail."

    if not means:
        print(empty_msg, file=sys.stderr)
        return False

    meta = {
        "run_path": run_path,
        "run_name": getattr(run, "name", "") or "",
        "run_id": getattr(run, "id", "") or "",
        "project": getattr(run, "project", "") or "",
        "entity": getattr(run, "entity", "") or "",
        "state": getattr(run, "state", "") or "",
        "avg_mode": "from_start_step" if args.start_step is not None else "last_n",
        "last_n_requested": str(args.last_n),
        "last_n_used": str(n_used),
    }
    if args.start_step is not None:
        meta["start_step"] = str(args.start_step)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in meta.items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["metric", value_header])
        for m in sorted(means.keys()):
            w.writerow([m, means[m]])

    print(
        f"Wrote {out_path} ({n_used} rows averaged, {len(means)} metrics) for {run_path}",
        file=sys.stderr,
    )
    return True


def run(args: argparse.Namespace) -> None:
    """CLI: single run or full project."""
    if args.last_n < 1:
        raise SystemExit("--last-n must be >= 1")
    if bool(args.run_path) == bool(args.project_path):
        raise SystemExit("Provide exactly one of: --run-path, --project-path")
    if args.run_path:
        if not args.out:
            raise SystemExit("--out is required with --run-path")
        if args.out_dir:
            raise SystemExit("--out-dir is only valid with --project-path")
        ok = process_one_run(args, args.run_path, args.out)
        if not ok:
            sys.exit(1)
        return
    # project sweep
    if not args.out_dir:
        raise SystemExit("--out-dir is required with --project-path")
    if args.out:
        raise SystemExit("--out is only valid with --run-path")
    project_path = normalize_project_path(cast(str, args.project_path))
    name_re = re.compile(args.name_regex) if args.name_regex else None

    os.makedirs(args.out_dir, exist_ok=True)
    api = wandb.Api()
    runs = api.runs(project_path)
    ok_count = 0
    fail_count = 0
    skip_count = 0
    for w_run in runs:
        st = getattr(w_run, "state", "") or ""
        if args.only_state != "any" and st != args.only_state:
            skip_count += 1
            continue
        rname = getattr(w_run, "name", "") or ""
        if name_re is not None and not name_re.search(rname):
            skip_count += 1
            continue
        rid = getattr(w_run, "id", "") or "unknown"
        fname = f"{rid}_{_safe_filename_part(rname) or 'run'}.csv"
        out_path = os.path.join(args.out_dir, fname)
        run_path_full = f"{project_path}/{rid}"
        try:
            if process_one_run(args, run_path_full, out_path):
                ok_count += 1
            else:
                fail_count += 1
                print(f"[fail] {run_path_full} (see messages above)", file=sys.stderr)
        except Exception as e:
            fail_count += 1
            print(f"[fail] {run_path_full}: {e}", file=sys.stderr)

    print(
        f"Project {project_path}: wrote {ok_count} CSV(s) under {args.out_dir}; "
        f"skipped {skip_count} run(s) (state/name); failed {fail_count}.",
        file=sys.stderr,
    )
    if fail_count:
        sys.exit(1)


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
