#!/usr/bin/env python3
"""P1 overlay clips: SOMA HLR -> analytic (or MLP) G1-compatible torso/wrists."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h2r.adapter import align_pair, analytic_map, calibrate, naive_affine_map, to_u
from h2r.bvh_fk import extract_hlr
from h2r.constants import OUT, OUT_HZ, P1_ROOT, SCALE_SWEEP, TASKS
from h2r.g1_io import load_g1_tlr, overlay_npz
from h2r.mlp import ResidualMLP
from h2r.pair_index import build_index


def p1_stem(path: Path) -> str:
    name = path.name
    if name.startswith("bones_seed_g1_"):
        return name[len("bones_seed_g1_") : -4]
    # 00_bones_seed_g1_XXX.npz
    i = name.find("bones_seed_g1_")
    if i >= 0:
        return name[i + len("bones_seed_g1_") : -4]
    return path.stem


def match_p1(p1: Path, index: list[dict]) -> dict | None:
    stem = p1_stem(p1)
    cands = [r for r in index if r["filename"].startswith(stem + "__") or r["filename"] == stem]
    cands = [r for r in cands if r["bvh_ok"] and r["npz_ok"]]
    if not cands:
        return None
    src = np.load(str(p1))
    pos = src["body_pos_w"]
    best, best_err = None, 1e9
    for r in cands:
        try:
            g, _ = load_g1_tlr(r["npz"])
        except Exception:
            continue
        t = min(pos.shape[0], g.shape[0])
        err = float(np.mean(np.abs(pos[:t, [9, 28, 29]] - g[:t])))
        pref = 0 if "_M" not in r["filename"] else 1
        key = (pref, err)
        if best is None or key < best_err:
            best, best_err = r, key
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("analytic", "naive", "learned"), default="analytic")
    ap.add_argument("--mlp", type=str, default="")
    ap.add_argument("--s_g1", type=float, default=0.0, help="0 = per-clip G1 calib.s")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    tag = args.tag or args.mode
    index = build_index()
    man = {}
    s_g1_vals = []
    for task in TASKS:
        clips = sorted((P1_ROOT / task).glob("*.npz"))
        man[task] = []
        for p1 in clips:
            rec = match_p1(p1, index)
            row = {"p1": p1.name, "ok": False}
            if rec is None:
                row["reason"] = "no_meta_match"
                man[task].append(row)
                continue
            try:
                hlr, fps_h = extract_hlr(rec["bvh"])
                tlr, fps_g = load_g1_tlr(p1)
                h50, g50, info = align_pair(hlr, fps_h, tlr, fps_g)
                hc = calibrate(h50, OUT_HZ)
                gc = calibrate(g50, OUT_HZ)
                s_g1 = args.s_g1 if args.s_g1 > 0 else gc.s
                s_g1_vals.append(s_g1)
                uh = to_u(h50, hc)
                if args.mode == "naive":
                    pred = naive_affine_map(h50, hc, gc, s_g1)
                elif args.mode == "learned":
                    mlp = ResidualMLP()
                    blob = torch.load(args.mlp, map_location="cpu")
                    mlp.load_state_dict(blob["state_dict"] if "state_dict" in blob else blob)
                    mlp.eval()
                    with torch.no_grad():
                        u = torch.from_numpy(uh.astype(np.float32))
                        uh2 = mlp(u).numpy()
                    pred = analytic_map(uh2, gc, s_g1)
                else:
                    pred = analytic_map(uh, gc, s_g1)
                dst = OUT / "end_to_end" / "clips" / tag / task / p1.name
                overlay_npz(p1, dst, pred.astype(np.float32))
                err = np.linalg.norm((pred[: g50.shape[0]] - g50[: pred.shape[0]]).reshape(min(len(pred), len(g50)), -1), axis=-1)
                row.update(
                    {
                        "ok": True,
                        "filename": rec["filename"],
                        "bvh": rec["bvh"],
                        "s_h": hc.s,
                        "s_g1": s_g1,
                        "align": info,
                        "mpjpe_m": float(err.mean()),
                    }
                )
            except Exception as e:
                row["reason"] = f"{type(e).__name__}: {e}"
            man[task].append(row)
            print(f"[{task}] {p1.name} {row.get('ok')} {row.get('mpjpe_m', row.get('reason'))}", flush=True)
    dest = OUT / "manifests" / f"overlay_{tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"mode": args.mode, "s_g1_used": s_g1_vals[:5], "tasks": man}, indent=2, default=str))
    print("wrote", dest)


if __name__ == "__main__":
    main()
