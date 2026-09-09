#!/usr/bin/env python3
"""R1a-2a smoke: terminal progress must not leak through reset. No Isaac.

Also calibrates λ_p so |λ_p r_prog| is ~10–30% of canonical per-step scale
on parent active frames.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import numpy as np
import torch

ROOT = Path("/data/home/chenxiangyu/robotics/Anybody")
sys.path.insert(0, str(ROOT / "source" / "rsl_rl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rsl_rl.modules.intent_recovery import (  # noqa: E402
    DEFAULT_PROGRESS_CLIP,
    DT,
    Q50_E,
    Q90_E,
    RecoveryRiskGate,
    recovery_progress_reset_leak,
    recovery_progress_reward,
)
from analyze_p2r_step3 import ALL_TERRAINS, MATRIX, WARMUP, _load_root  # noqa: E402

OUT = Path("/data/home/chenxiangyu/robotics/Anybody/results/p2r_r1a2")
# R1a mid-run: Mean reward 15.59 / episode length 104 ≈ 0.15 per step.
CANONICAL_STEP_SCALE = 0.15
TARGET_FRAC = 0.20  # midpoint of 10–30%


def _t(*xs):
    return torch.tensor(list(xs), dtype=torch.float32)


def test_terminal_zero() -> dict:
    """Failure then reset must not yield positive progress."""
    # User's pathology: R_t=5, post-reset R=0.3, done=1.
    r = recovery_progress_reward(
        active_t=_t(1, 1, 0, 1, 1),
        r_e_t=_t(5.0, 1.5, 0.4, 1.2, 8.0),
        r_e_next=_t(0.3, 1.2, 0.2, 1.5, 2.0),
        done=_t(1, 0, 0, 0, 0),
        clip_p=1.0,
    )
    want = _t(0.0, 0.3, 0.0, -0.3, 1.0)
    ok = bool(torch.allclose(r, want, atol=1e-6))
    leak = recovery_progress_reset_leak(
        active_t=_t(1),
        r_e_t=_t(5.0),
        r_e_next=_t(0.3),
        done=_t(1),
        clip_p=1.0,
    )
    unclipped = 5.0 - 0.3
    return {
        "H_pass": ok and float(r[0]) == 0.0 and float(leak[0]) == 1.0,
        "got": [float(x) for x in r],
        "want": [float(x) for x in want],
        "counterfactual_clipped": float(leak[0]),
        "counterfactual_unclipped": unclipped,
        "note": "done=1 → r_prog=0; unclipped reset leak would be +4.7",
    }


def calibrate_lambda() -> dict:
    """σ(clip ΔR_E | active, live) on parent matrix; pick λ_p for ~20% of 0.15."""
    rows = []
    for ter in ALL_TERRAINS:
        rows.extend(_load_root(MATRIX, "loco", ter, "torso"))
    progs: list[float] = []
    raws: list[float] = []
    n_active = 0
    n_fail_active_last = 0
    leak_unclipped: list[float] = []
    den = max(Q90_E - Q50_E, 1e-6)
    for r in rows:
        e = np.asarray(r["feats"]["E"], dtype=np.float64)
        fail = bool(r["fail"])
        fail_t = r["fail_t"]
        t_end = int(fail_t) if fail and fail_t is not None else int(e.size - 1)
        gate = RecoveryRiskGate(s_enabled=False)
        r_e = np.zeros(e.size, dtype=np.float64)
        active = np.zeros(e.size, dtype=bool)
        for t in range(e.size):
            o = gate.step(
                torch.tensor([float(e[t])], dtype=torch.float32),
                torch.tensor([0.0], dtype=torch.float32),
            )
            r_e[t] = float(o["R_E"].item())
            active[t] = bool(o["active"].item())
        for t in range(max(0, min(t_end, e.size - 1))):
            if t < WARMUP:
                continue
            done = t >= t_end  # last in-episode index has no live successor
            if t == t_end:
                break
            if not active[t]:
                continue
            n_active += 1
            raw = float(r_e[t] - r_e[t + 1])
            raws.append(raw)
            rp = recovery_progress_reward(
                torch.tensor([1.0]),
                torch.tensor([r_e[t]], dtype=torch.float32),
                torch.tensor([r_e[t + 1]], dtype=torch.float32),
                torch.tensor([False]),
            )
            progs.append(float(rp.item()))
        if fail and active[t_end]:
            n_fail_active_last += 1
            # Synthetic healthy reset, matching the Isaac bug.
            r_reset = (0.05 - Q50_E) / den
            leak_unclipped.append(float(r_e[t_end] - r_reset))
    p = np.asarray(progs, dtype=np.float64)
    raw_a = np.asarray(raws, dtype=np.float64)
    sigma_p = float(p.std()) if p.size else float("nan")
    mean_abs_p = float(np.abs(p).mean()) if p.size else float("nan")
    sigma_raw = float(raw_a.std()) if raw_a.size else float("nan")
    # Target |λ r_prog| ~ 20% of canonical scale on the std of clipped progress.
    lam = TARGET_FRAC * CANONICAL_STEP_SCALE / max(sigma_p, 1e-6)
    lam = float(np.clip(lam, 0.2, 2.0))
    frac = lam * sigma_p / CANONICAL_STEP_SCALE if sigma_p == sigma_p else float("nan")
    leak_a = np.asarray(leak_unclipped, dtype=np.float64)
    return {
        "n_episodes": len(rows),
        "n_active_live": int(p.size),
        "n_fail_while_active": n_fail_active_last,
        "sigma_r_prog_clipped": sigma_p,
        "mean_abs_r_prog_clipped": mean_abs_p,
        "sigma_raw_dR": sigma_raw,
        "p_clip_sat": float((np.abs(raw_a) > DEFAULT_PROGRESS_CLIP).mean()) if raw_a.size else float("nan"),
        "canonical_step_scale": CANONICAL_STEP_SCALE,
        "canonical_source": "R1a train Mean reward / Mean episode length ≈ 15.59/104",
        "lambda_p": lam,
        "lambda_p_times_sigma_over_can": frac,
        "mean_unclipped_fail_reset_leak": float(leak_a.mean()) if leak_a.size else float("nan"),
        "median_unclipped_fail_reset_leak": float(np.median(leak_a)) if leak_a.size else float("nan"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"H": test_terminal_zero(), "lambda": calibrate_lambda()}
    report["all_pass"] = bool(report["H"]["H_pass"])
    outp = OUT / "smoke_r1a2.json"
    outp.write_text(json.dumps(report, indent=2))
    print("== R1a-2a terminal smoke ==")
    print(json.dumps(report["H"], indent=2))
    print("== λ_p calibration (parent active frames) ==")
    print(json.dumps(report["lambda"], indent=2))
    print("PASS" if report["all_pass"] else "FAIL")
    print("wrote", outp)
    print("LAMBDA_P", report["lambda"]["lambda_p"])


if __name__ == "__main__":
    main()
