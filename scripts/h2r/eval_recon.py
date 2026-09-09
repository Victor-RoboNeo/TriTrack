#!/usr/bin/env python3
"""Reconstruction + morphology invariance. CPU/GPU. No Isaac required."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h2r.adapter import align_pair, analytic_map, calibrate, naive_affine_map, pairwise_d, to_u
from h2r.bvh_fk import extract_hlr
from h2r.constants import OUT, OUT_HZ, P1_ROOT, SCALE_SWEEP, SCALE_SWEEP_EVAL, TASKS
from h2r.g1_io import load_g1_tlr
from h2r.mlp import ResidualMLP
from h2r.overlay_p1 import match_p1, p1_stem
from h2r.pair_index import build_index


def mpjpe(a, b):
    t = min(len(a), len(b))
    return np.linalg.norm(a[:t] - b[:t], axis=-1).mean()


def pair_err(a, b):
    t = min(len(a), len(b))
    return np.abs(pairwise_d(a[:t]) - pairwise_d(b[:t])).mean()


def dir_err(a, b):
    t = min(len(a), len(b))
    def vecs(p):
        return np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 1] - p[:, 2]], 1)
    va, vb = vecs(a[:t]), vecs(b[:t])
    na = np.linalg.norm(va, axis=-1, keepdims=True).clip(1e-8)
    nb = np.linalg.norm(vb, axis=-1, keepdims=True).clip(1e-8)
    c = ((va / na) * (vb / nb)).sum(-1).clip(-1, 1)
    return np.degrees(np.arccos(c)).mean()


def scale_human(hlr, alpha, origin):
    return origin[None, None] + alpha * (hlr - origin[None, None])


def eval_clip(hlr, fps_h, tlr, fps_g, mlp=None):
    h50, g50, info = align_pair(hlr, fps_h, tlr, fps_g)
    hc = calibrate(h50, OUT_HZ)
    gc = calibrate(g50, OUT_HZ)
    uh = to_u(h50, hc)
    pred_a = analytic_map(uh, gc, gc.s)
    if mlp is not None:
        with torch.no_grad():
            uh_l = mlp(torch.from_numpy(uh.astype(np.float32))).numpy()
        pred = analytic_map(uh_l, gc, gc.s)
    else:
        pred = pred_a
    pred_n = naive_affine_map(h50, hc, gc, gc.s)

    def pack(pred):
        pr = pred - pred[:, 0:1]
        gr = g50[: len(pred)] - g50[: len(pred), 0:1]
        return {
            "mpjpe": float(mpjpe(pred, g50)),
            "mpjpe_rel": float(np.linalg.norm(pr - gr, axis=-1).mean()),
            "pair": float(pair_err(pred, g50)),
            "dir_deg": float(dir_err(pred, g50)),
        }

    inv = {}
    base = analytic_map(to_u(h50, hc), gc, gc.s)
    for a in SCALE_SWEEP_EVAL:
        hs = scale_human(h50, a, hc.origin)
        hcs = calibrate(hs, OUT_HZ)
        ps = analytic_map(to_u(hs, hcs), gc, gc.s)
        inv[str(a)] = {
            "output_mpjpe": float(mpjpe(ps, base)),
            "pair_drift": float(pair_err(ps, base)),
            "s_h": hcs.s,
        }
    return {
        "align": info,
        "s_h": hc.s,
        "s_g": gc.s,
        "analytic": pack(pred_a),
        "naive": pack(pred_n),
        "learned": None if mlp is None else pack(pred),
        "invariance": inv,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp", type=str, default="")
    ap.add_argument("--out", type=str, default=str(OUT / "invariance" / "p1_recon.json"))
    args = ap.parse_args()
    mlp = None
    if args.mlp:
        mlp = ResidualMLP()
        blob = torch.load(args.mlp, map_location="cpu")
        mlp.load_state_dict(blob["state_dict"] if "state_dict" in blob else blob)
        mlp.eval()
    index = build_index()
    report = {}
    for task in TASKS:
        report[task] = []
        for p1 in sorted((P1_ROOT / task).glob("*.npz")):
            rec = match_p1(p1, index)
            if rec is None:
                report[task].append({"p1": p1.name, "ok": False, "reason": "no_match"})
                continue
            try:
                hlr, fps_h = extract_hlr(rec["bvh"])
                tlr, fps_g = load_g1_tlr(p1)
                row = eval_clip(hlr, fps_h, tlr, fps_g, mlp)
                row.update({"p1": p1.name, "ok": True, "filename": rec["filename"]})
                report[task].append(row)
                print(f"[{task}] {p1.name} analytic_mpjpe={row['analytic']['mpjpe']:.4f}", flush=True)
            except Exception as e:
                report[task].append({"p1": p1.name, "ok": False, "reason": str(e)})
    agg = {}
    for mode in ("analytic", "naive"):
        xs = [r[mode]["mpjpe"] for t in TASKS for r in report[t] if r.get("ok")]
        rs = [r[mode]["mpjpe_rel"] for t in TASKS for r in report[t] if r.get("ok")]
        ps = [r[mode]["pair"] for t in TASKS for r in report[t] if r.get("ok")]
        agg[mode] = {
            "mpjpe_mean": float(np.mean(xs)),
            "mpjpe_rel_mean": float(np.mean(rs)),
            "pair_mean": float(np.mean(ps)),
            "n": len(xs),
        } if xs else {}
    inv_d = []
    for t in TASKS:
        for r in report[t]:
            if r.get("ok"):
                inv_d.append(r["invariance"]["1.3"]["output_mpjpe"])
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"agg": agg, "scale_1.3_drift_mean": float(np.mean(inv_d) if inv_d else np.nan), "tasks": report}, indent=2, default=str))
    print("wrote", dest, agg)


if __name__ == "__main__":
    main()
