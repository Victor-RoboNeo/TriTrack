"""PATH M vs PATH L observation-layer waterfall (CPU). Isaac fills encoder/g_phi/decoder."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from flat_locomani.live_anchor_v2.contract_audit import audit_frame, run_cpu_audit, summarize_audit
from flat_locomani.live_anchor_v2.motion_convert import load_seed_and_index
from flat_locomani.eval_helpers import load_suite
from flat_locomani.reference_suite import family_a, family_b, family_c, family_e, family_f

from .constants import RESULTS, V3_DEV
from .io_util import atomic_write_json, utc_now


def _stats(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def _items():
    if V3_DEV.exists():
        return load_suite(V3_DEV)
    items = []
    items += family_a(2026)[:4]
    items += family_b()[:4]
    items += family_c()[:4]
    items += family_e()[:4]
    items += family_f()[:4]
    # convert TrajectoryMeta to dict
    out = []
    for w, m in items:
        rec = m if isinstance(m, dict) else {
            "traj_id": getattr(m, "traj_id", "x"),
            "family": getattr(m, "family", "unk"),
        }
        out.append((w, rec))
    return out


def run_cpu() -> dict:
    seed0, idxs = load_seed_and_index()
    items = _items()
    # Expand ticks to reach N>=1000 paired samples.
    ident = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    bp = np.asarray(seed0["body_pos_w"])
    if bp.ndim == 3:
        bp = bp[0]

    def _body(name: str) -> np.ndarray:
        return np.asarray(bp[idxs[name]], dtype=np.float64)

    torso = _body("torso_link")
    robot_kp = np.stack(
        [
            _body("torso_link"),
            _body("left_wrist_yaw_link"),
            _body("right_wrist_yaw_link"),
            _body("left_ankle_roll_link"),
            _body("right_ankle_roll_link"),
        ]
    )
    rows = []
    for world, rec in items:
        T = int(world.shape[0])
        # dense ticks
        step = max(1, T // 40)
        for t in range(0, T, step):
            row = audit_frame(
                world,
                t,
                seed0,
                idxs,
                torso,
                ident,
                robot_kp,
                str(rec.get("traj_id")),
                str(rec.get("family")),
            )
            rows.append(row)
            if len(rows) >= 1200:
                break
        if len(rows) >= 1200:
            break
    outdir = RESULTS / "01_p0_contract"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_p = outdir / "p0_path_m_vs_l_obs.csv"
    keys = sorted({k for r in rows for k in r})
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    vis = [float(r["packed_visible_l2"]) for r in rows]
    summary = {
        "n_paired_samples": len(rows),
        "packed_visible_l2": _stats(vis),
        "historical_packed_visible_l2_mean": 2.2087,
        "note": "CPU observation layer only. Encoder / g_phi / decoder filled by P0_PROP_ISAAC.",
        "D_obs_placeholder_equals_packed_visible_l2": True,
        "D_stage2_latent": "REQUIRES_ISAAC",
        "D_gphi_residual": "REQUIRES_ISAAC",
        "D_final_latent": "REQUIRES_ISAAC",
        "D_decoder_action": "REQUIRES_ISAAC",
        "timestamp": utc_now(),
    }
    atomic_write_json(outdir / "p0_propagation_cpu.json", summary)
    print(json_dumps(summary))
    return summary


def json_dumps(obj) -> str:
    import json

    s = json.dumps(obj, indent=2)
    print(s)
    return s


if __name__ == "__main__":
    run_cpu()
