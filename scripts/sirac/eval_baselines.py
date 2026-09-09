#!/usr/bin/env python3
"""Deterministic SIRAC Phase-1 baseline evaluation.

Default path is Isaac-free (dummy LBC) and writes JSON/CSV. This must not be
read as Case A/B/C/D evidence.

When Isaac Lab + the HTD student JIT are available, pass --isaac to run a
short vectorized rollout. Never uses GPU 6/7. Does not modify frozen AnyBody
eval scripts.

    python scripts/sirac/eval_baselines.py --seed 42
    python scripts/sirac/eval_baselines.py --isaac --num-envs 4 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
import sirac_isaacfree  # noqa: E402

sirac_isaacfree.install()
sys.path.insert(0, str(ROOT / "source" / "whole_body_tracking"))

from whole_body_tracking.sirac.metrics import EpisodeMetrics, write_csv, write_json
from whole_body_tracking.sirac.pipeline import Baseline

# Import smoke helper
sys.path.insert(0, str(ROOT / "scripts" / "sirac"))
from smoke_realization import dummy_rollout  # noqa: E402


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--baselines", nargs="*", default=[b.value for b in Baseline])
    p.add_argument("--out", type=str, default=str(ROOT / "results" / "sirac_phase1" / "eval"))
    p.add_argument("--isaac", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--jit", type=str, default=None)
    return p.parse_args()


def _guard_gpu(gpu: int) -> None:
    if gpu in (6, 7):
        raise SystemExit("GPU 6/7 are reserved; pick 0-5")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))


def eval_isaac(args: argparse.Namespace) -> list[EpisodeMetrics]:
    """Placeholder: real Isaac loop belongs here once Kit isolation + JIT exist.

    We refuse to silently skip and claim success. Return a blocked metric row.
    """
    from whole_body_tracking.sirac.lower_body_controller import find_student_jit

    jit = find_student_jit(args.jit)
    note = "isaac_eval_not_wired: kit isolation + HTD student JIT required"
    if jit is None:
        note = "blocked: HTD student JIT not found on disk; dummy LBC is not evidence"
    rows = []
    for name in args.baselines:
        rows.append(
            EpisodeMetrics(
                baseline=name,
                seed=args.seed,
                n_steps=0,
                notes=note,
            )
        )
    return rows


def main() -> int:
    args = _parse()
    _guard_gpu(args.gpu)
    np.random.seed(args.seed)
    out = Path(args.out)
    rows: list[EpisodeMetrics] = []
    if args.isaac:
        rows.extend(eval_isaac(args))
    for name in args.baselines:
        b = Baseline(name)
        rows.append(dummy_rollout(b, n_env=args.num_envs, n_steps=args.steps, seed=args.seed))
    write_json(out / "metrics.json", rows)
    write_csv(out / "metrics.csv", rows)
    summary = {
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "isaac": args.isaac,
        "decision": "no_claim",
        "reason": "dummy or blocked Isaac eval is not Case A-D evidence",
        "rows": [r.baseline for r in rows],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
