"""Isaac-free smoke test: dummy rollout through Baselines A/B/C.

This is *not* evidence that the realization bottleneck works. It only checks
that the pipeline is wired, batched, and writes metrics JSON/CSV.

    python scripts/sirac/smoke_realization.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
import sirac_isaacfree  # noqa: E402

sirac_isaacfree.install()
sys.path.insert(0, str(ROOT / "source" / "whole_body_tracking"))

from whole_body_tracking.sirac.lower_body_controller import LowerBodyRealizationController
from whole_body_tracking.sirac.mappings import HTD_DEFAULT_LOWER, HTD_NUM_ACTIONS
from whole_body_tracking.sirac.metrics import EpisodeMetrics, action_smoothness, write_csv, write_json
from whole_body_tracking.sirac.pipeline import Baseline, SiracPhase1Controller


def dummy_rollout(baseline: Baseline, n_env: int = 4, n_steps: int = 16, seed: int = 0) -> EpisodeMetrics:
    rng = np.random.default_rng(seed)
    ctl = SiracPhase1Controller(baseline=baseline, lbc=LowerBodyRealizationController())
    ctl.reset()
    actions = []
    t0 = time.perf_counter()
    lat = []
    q29 = np.concatenate([np.tile(HTD_DEFAULT_LOWER, (n_env, 1)), np.zeros((n_env, 14))], axis=1)
    for _ in range(n_steps):
        t1 = time.perf_counter()
        out = ctl.step(
            stage2_q_target_29=q29 + 0.01 * rng.standard_normal(q29.shape),
            pelvis_quat_w=np.tile(np.array([1.0, 0, 0, 0]), (n_env, 1)),
            pelvis_lin_vel_w=np.zeros((n_env, 3)),
            pelvis_ang_vel_w=np.zeros((n_env, 3)),
            torso_pos_w=np.tile(np.array([0.0, 0.0, 0.72]), (n_env, 1)),
            torso_quat_w=np.tile(np.array([1.0, 0, 0, 0]), (n_env, 1)),
            ang_vel_b=np.zeros((n_env, 3)),
            joint_pos_29=q29,
            joint_vel_29=np.zeros((n_env, 29)),
        )
        lat.append((time.perf_counter() - t1) * 1e3)
        if out.lower_action_15 is not None:
            actions.append(out.lower_action_15)
        assert out.q_target_29.shape[-1] == 29
    runtime = time.perf_counter() - t0
    a = np.stack(actions, axis=1) if actions else np.zeros((n_env, n_steps, HTD_NUM_ACTIONS))
    return EpisodeMetrics(
        baseline=baseline.value,
        seed=seed,
        n_steps=n_steps,
        runtime_s=runtime,
        policy_latency_ms=float(np.mean(lat)),
        action_smoothness=action_smoothness(a.reshape(-1, a.shape[-1])),
        notes="dummy_policy" if ctl.lbc.is_dummy else str(ctl.lbc.loaded_path),
        success=float("nan"),
        fell=float("nan"),
    )


def main() -> int:
    out_dir = ROOT / "results" / "sirac_phase1" / "smoke"
    rows = [dummy_rollout(b) for b in Baseline]
    write_json(out_dir / "smoke_metrics.json", rows)
    write_csv(out_dir / "smoke_metrics.csv", rows)
    print(f"wrote {out_dir / 'smoke_metrics.json'}")
    for r in rows:
        print(f"  {r.baseline}: dummy={r.notes} latency_ms={r.policy_latency_ms:.3f}")
    print("SMOKE OK — not an experimental finding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
