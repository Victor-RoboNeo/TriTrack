#!/usr/bin/env python3
"""Build SOMA↔G1 paired 50Hz dimensionless 3-point dataset. CPU-parallel. No Isaac."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h2r.adapter import align_pair, calibrate, to_u
from h2r.bvh_fk import extract_hlr
from h2r.constants import DUR_TOL, MIN_FRAMES, OUT, OUT_HZ
from h2r.g1_io import load_g1_tlr
from h2r.pair_index import build_index

OUT.mkdir(parents=True, exist_ok=True)


def _one(rec: dict) -> dict:
    if not rec["bvh_ok"] or not rec["npz_ok"]:
        return {"ok": False, "reason": "missing_file", "filename": rec["filename"]}
    try:
        hlr, fps_h = extract_hlr(rec["bvh"])
        tlr, fps_g = load_g1_tlr(rec["npz"])
        h50, g50, info = align_pair(hlr, fps_h, tlr, fps_g, DUR_TOL)
        if h50.shape[0] < MIN_FRAMES:
            return {"ok": False, "reason": "short", "filename": rec["filename"], **info}
        hc = calibrate(h50, OUT_HZ)
        gc = calibrate(g50, OUT_HZ)
        uh = to_u(h50, hc)
        ug = to_u(g50, gc)
        if not (np.isfinite(uh).all() and np.isfinite(ug).all()):
            return {"ok": False, "reason": "nan", "filename": rec["filename"]}
        payload = {
            "ok": True,
            "filename": rec["filename"],
            "split": rec["split"],
            "actor_uid": rec["actor_uid"],
            "take_name": rec["take_name"],
            "package": rec["package"],
            "T": int(h50.shape[0]),
            "fps": OUT_HZ,
            "s_h": hc.s,
            "s_g": gc.s,
            "align": info,
            "u_h": uh.astype(np.float32),
            "u_g": ug.astype(np.float32),
            "rest_h": np.stack([hc.rest_l, hc.rest_r]).astype(np.float32),
            "rest_g": np.stack([gc.rest_l, gc.rest_r]).astype(np.float32),
        }
        return payload
    except Exception as e:
        return {
            "ok": False,
            "reason": f"{type(e).__name__}: {e}",
            "filename": rec["filename"],
            "trace": traceback.format_exc()[-400:],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() or 8, 8))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--p1_stems", type=str, default="")
    ap.add_argument("--tag", type=str, default="full")
    args = ap.parse_args()
    idx = [r for r in build_index() if r["bvh_ok"] and r["npz_ok"]]
    if args.p1_stems:
        stems = set(Path(args.p1_stems).read_text().split())
        idx = [r for r in idx if any(r["filename"].startswith(s + "__") or r["filename"] == s for s in stems)]
    if args.limit:
        idx = idx[: args.limit]
    dest = OUT / "paired_dataset" / args.tag
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[h2r] pair candidates={len(idx)} workers={args.workers} tag={args.tag}", flush=True)
    ok_n = fail_n = 0
    reasons: dict[str, int] = {}
    shards = {s: {"u_h": [], "u_g": [], "meta": []} for s in ("train", "val", "test")}
    s_g, s_h = [], []
    with Pool(args.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_one, idx, chunksize=8), 1):
            if not rec.get("ok"):
                fail_n += 1
                why = str(rec.get("reason", "fail")).split(":")[0]
                reasons[why] = reasons.get(why, 0) + 1
            else:
                ok_n += 1
                sp = rec["split"]
                shards[sp]["u_h"].append(rec["u_h"])
                shards[sp]["u_g"].append(rec["u_g"])
                shards[sp]["meta"].append(
                    {
                        "filename": rec["filename"],
                        "T": rec["T"],
                        "s_h": rec["s_h"],
                        "s_g": rec["s_g"],
                        "actor_uid": rec["actor_uid"],
                        "take_name": rec["take_name"],
                        "package": rec["package"],
                        "align": rec["align"],
                    }
                )
                s_g.append(rec["s_g"])
                s_h.append(rec["s_h"])
            if i % 500 == 0 or i == len(idx):
                print(f"[h2r] {i}/{len(idx)} ok={ok_n} fail={fail_n} {reasons}", flush=True)
    stats = {
        "n_candidates": len(idx),
        "n_ok": ok_n,
        "n_fail": fail_n,
        "align_rate": ok_n / max(len(idx), 1),
        "reasons": reasons,
        "s_g1_median": float(np.median(s_g)) if s_g else None,
        "s_h_median": float(np.median(s_h)) if s_h else None,
        "s_g1_mean": float(np.mean(s_g)) if s_g else None,
        "n_frames": {k: int(sum(m["T"] for m in shards[k]["meta"])) for k in shards},
        "n_clips": {k: len(shards[k]["meta"]) for k in shards},
        "hz": OUT_HZ,
        "split": "take_name hash 80/10/10",
        "representation": "u_human -> u_G1 dimensionless heading/gravity/scale",
    }
    (dest / "stats.json").write_text(json.dumps(stats, indent=2))
    for sp, blob in shards.items():
        if not blob["meta"]:
            continue
        np.savez_compressed(
            dest / f"{sp}.npz",
            u_h=np.concatenate(blob["u_h"], 0),
            u_g=np.concatenate(blob["u_g"], 0),
            clip_ptr=np.cumsum([0] + [m["T"] for m in blob["meta"]]).astype(np.int64),
            meta=np.array([json.dumps(m) for m in blob["meta"]]),
        )
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
