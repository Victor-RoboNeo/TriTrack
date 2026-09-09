#!/usr/bin/env python3
"""Estimate T_H_C from paired WORLD head/chest poses. Does not invent an offset.

Usage:
  python scripts/calibrate_head_to_chest.py --paired path/to/pairs.jsonl --out results/.../configs/head_to_chest_calibration.yaml

pairs.jsonl records: {"head": {"pos":[x,y,z], "quat_wxyz":[w,x,y,z]}, "chest": {...}}

Without paired data this script refuses to write a validated calibration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/data/home/chenxiangyu/humantracker_3pt_ood")
sys.path.insert(0, "/data/home/chenxiangyu/victor/TriTrack")

from flat_nominal_competence_repair_v2.canonicalize import (  # noqa: E402
    _as_pose,
    invert_se3,
    compose_se3,
    geodesic,
)


def _avg_quat(qs: np.ndarray) -> np.ndarray:
    m = qs.T @ qs
    w, v = np.linalg.eigh(m)
    q = v[:, int(np.argmax(w))]
    if q[0] < 0:
        q = -q
    return q / max(float(np.linalg.norm(q)), 1e-12)


def estimate(pairs: list[dict]) -> dict:
    ts, qs = [], []
    for rec in pairs:
        h_p, h_q = _as_pose(rec["head"])
        c_p, c_q = _as_pose(rec["chest"])
        t_wh_inv, q_wh_inv = invert_se3(h_p, h_q)
        t_hc, q_hc = compose_se3(t_wh_inv, q_wh_inv, c_p, c_q)
        ts.append(t_hc)
        qs.append(q_hc)
    T = np.stack(ts)
    Q = np.stack(qs)
    t_med = np.median(T, axis=0)
    q_avg = _avg_quat(Q)
    pos_err = np.linalg.norm(T - t_med, axis=-1)
    ori_err = np.array([geodesic(q, q_avg) for q in Q])
    return {
        "translation_xyz": [float(x) for x in t_med],
        "quaternion_wxyz": [float(x) for x in q_avg],
        "n": len(pairs),
        "pos_mad_m": float(np.median(pos_err)),
        "ori_mad_rad": float(np.median(ori_err)),
        "status": "ESTIMATED_FROM_PAIRED_DATA",
        "synthetic_test_only": False,
        "convention": "T_W_C = T_W_H @ T_H_C ; quat wxyz",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.paired:
        out.write_text(
            """# NOT a validated real-human T_H_C.
# Case C: no paired head/chest data in this repo.
status: NOT_VALIDATED
synthetic_test_only: true
translation_xyz: [0.0, 0.0, -0.22]
quaternion_wxyz: [1.0, 0.0, 0.0, 0.0]
convention: "T_W_C = T_W_H @ T_H_C ; quaternion wxyz ; WORLD frames"
note: "Synthetic test-only. Do not use for real-human eval. Training uses canonical chest."
HEAD_MODE_INTERFACE: IMPLEMENTED
HEAD_MODE_REAL_HUMAN_CALIBRATION: NOT_VALIDATED
"""
        )
        print("wrote NOT_VALIDATED placeholder", out)
        return
    recs = [json.loads(l) for l in Path(args.paired).read_text().splitlines() if l.strip()]
    if len(recs) < 10:
        raise SystemExit("need >=10 paired poses")
    est = estimate(recs)
    lines = [
        f"status: {est['status']}",
        "synthetic_test_only: false",
        f"translation_xyz: {est['translation_xyz']}",
        f"quaternion_wxyz: {est['quaternion_wxyz']}",
        'convention: "T_W_C = T_W_H @ T_H_C ; quaternion wxyz ; WORLD frames"',
        f"n: {est['n']}",
        f"pos_mad_m: {est['pos_mad_m']}",
        f"ori_mad_rad: {est['ori_mad_rad']}",
        "HEAD_MODE_INTERFACE: IMPLEMENTED",
        "HEAD_MODE_REAL_HUMAN_CALIBRATION: ESTIMATED",
        "",
    ]
    out.write_text("\n".join(lines))
    print(json.dumps(est, indent=2))


if __name__ == "__main__":
    main()
