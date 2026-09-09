#!/usr/bin/env python3
"""UCR-1: pack RE-trigger oracle dumps into clip-level train/val/test JSON.

No task ID at runtime. task/λ stay metadata. Same clip never crosses splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

TASKS = ("loco", "stoop", "reach", "carry")
DEPLOY_BINS = (2.5, 5.0, 7.5, 10.0)
SWEEP = (1.0, 2.5, 5.0, 7.5, 10.0)
SPLIT_SEED = 7


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _assign_clips(clips: list[str], task: str) -> dict[str, str]:
    """6/2/2 on 10 clips. Deterministic. At least one val and one test when n>=3."""
    clips = sorted(set(clips))
    n = len(clips)
    h = int(hashlib.md5(f"ucr1|{task}|{SPLIT_SEED}".encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(h)
    order = rng.permutation(n)
    n_test = 2 if n >= 5 else max(1, n // 5)
    n_val = 2 if n >= 5 else max(1, n // 5)
    if n_test + n_val >= n:
        n_test = max(1, n // 5)
        n_val = max(1, n // 5)
        if n_test + n_val >= n:
            n_test, n_val = 1, 1
    n_train = n - n_test - n_val
    out = {}
    for k, idx in enumerate(order):
        if k < n_train:
            out[clips[idx]] = "train"
        elif k < n_train + n_val:
            out[clips[idx]] = "val"
        else:
            out[clips[idx]] = "test"
    return out


def _row_from(i: int, blob, snaps: list | None) -> dict:
    rec = np.asarray(blob["rec_obs"][i], dtype=np.float32)
    z = np.asarray(blob["z_nom"][i], dtype=np.float32)
    d = np.asarray(blob["d_star"][i], dtype=np.float32)
    dn = float(np.linalg.norm(d))
    if dn > 1e-8:
        d = d / dn
    j_ang = np.asarray(blob["j_by_angle"][i], dtype=np.float64)
    sweep = np.asarray(blob["sweep_deg"], dtype=np.float32) if "sweep_deg" in blob.files else np.asarray(SWEEP)
    util_all = {f"{float(th):g}": float(j_ang[k]) for k, th in enumerate(sweep)}
    deploy_j = []
    for th in DEPLOY_BINS:
        key = f"{float(th):g}"
        if key not in util_all:
            key = f"{float(th):.1f}"
        deploy_j.append(float(util_all.get(key, util_all.get(str(th), 1e9))))
    deploy_j = np.asarray(deploy_j, dtype=np.float64)
    theta_idx = int(np.argmin(deploy_j))
    theta_ora_all = float(sweep[int(np.argmin(j_ang))]) if j_ang.size else float("nan")
    task = str(np.asarray(blob["task"]).astype(str)[i])
    clip = str(np.asarray(blob["clip"]).astype(str)[i])
    seed = int(blob["seed"][i])
    util_deploy = {str(th): float(deploy_j[k]) for k, th in enumerate(DEPLOY_BINS)}
    snap = snaps[i] if snaps is not None and i < len(snaps) else None
    return {
        "task_source": task,
        "task": task,
        "episode_id": f"{task}|plane|{seed}|{clip}",
        "terrain_label": "plane",
        "seed": seed,
        "clip": clip,
        "t0": int(blob["t"][i]),
        "lam": float(blob["lam"][i]),
        "trigger_channel": "E",
        "rec_obs": rec.tolist(),
        "z_nom": z.tolist(),
        "d_oracle": d.tolist(),
        "utility_by_angle": util_deploy,
        "utility_by_angle_all": util_all,
        "theta_idx": theta_idx,
        "theta_ora": float(blob["theta_ora"][i]),
        "theta_ora_all": theta_ora_all,
        "oracle_best_is_1deg": bool(abs(theta_ora_all - 1.0) < 1e-6),
        "j_p": float(blob["j_p"][i]),
        "j_r": float(blob["j_r"][i]),
        "j_ora": float(blob["j_ora"][i]),
        "i_rm3": float(blob["i_rm3"][i]),
        "i_ora": float(blob["i_ora"][i]),
        "re": float(blob["re"][i]),
        "rs": float(blob["rs"][i]),
        "e0": float(blob["e0"][i]),
        "sr_ep": int(blob["sr_ep"][i]) if "sr_ep" in blob.files else 1,
        "snap": snap,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    raw = Path(args.raw)
    out = Path(args.out)
    splits_dir = out / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    missing = []
    for task in TASKS:
        for p in sorted((raw / task).glob("lam_*/plane/clones.npz")):
            blob = np.load(p, allow_pickle=True)
            n = int(blob["j_p"].shape[0])
            print(f"[ucr1-pack] {p} n={n}", flush=True)
            if n == 0:
                continue
            if "rec_obs" not in blob.files:
                missing.append(str(p))
                continue
            snaps_p = p.with_name("snaps.pt")
            snaps = torch.load(snaps_p, map_location="cpu", weights_only=False) if snaps_p.is_file() else None
            for i in range(n):
                rows.append(_row_from(i, blob, snaps))
    if missing:
        raise SystemExit(f"clones missing rec_obs (re-run with --ucr1_dump): {missing}")
    if not rows:
        raise SystemExit(f"no oracle rows in {raw}")

    clips_by_task = defaultdict(set)
    for r in rows:
        clips_by_task[r["task_source"]].add(r["clip"])
    clip_split: dict[str, dict[str, str]] = {}
    for task in TASKS:
        clip_split[task] = _assign_clips(sorted(clips_by_task.get(task, [])), task)

    buckets = {"train": [], "val": [], "test": []}
    for r in rows:
        sp = clip_split[r["task_source"]][r["clip"]]
        r["split"] = sp
        buckets[sp].append(r)

    def _count(rs):
        by_task = {t: sum(1 for r in rs if r["task_source"] == t) for t in TASKS}
        by_lam = {}
        for r in rs:
            k = f"{r['task_source']}|lam_{int(r['lam'])}"
            by_lam[k] = by_lam.get(k, 0) + 1
        n1 = sum(1 for r in rs if r["oracle_best_is_1deg"])
        return {
            "n": len(rs),
            "n_clips": len({(r["task_source"], r["clip"]) for r in rs}),
            "n_episodes": len({r["episode_id"] for r in rs}),
            "by_task": by_task,
            "by_lam": by_lam,
            "frac_oracle_best_1deg": (n1 / len(rs)) if rs else float("nan"),
            "n_oracle_best_1deg": n1,
        }

    summary = {
        "split_seed": SPLIT_SEED,
        "unit": "task|clip (all seeds/λ of a clip stay in one split)",
        "deploy_bins_deg": list(DEPLOY_BINS),
        "sweep_deg": list(SWEEP),
        "note_1deg": "1° is oracle diagnostic only; magnitude labels use {2.5,5,7.5,10}",
        "n_all": _count(rows),
        "n_split": {k: _count(v) for k, v in buckets.items()},
        "clip_split": clip_split,
    }
    (splits_dir / "split.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    test_clips = {t: [c for c, sp in clip_split[t].items() if sp == "test"] for t in TASKS}
    (splits_dir / "test_clips.json").write_text(json.dumps(test_clips, indent=2))

    for name, rs in buckets.items():
        json_rows = [{k: v for k, v in r.items() if k != "snap"} for r in rs]
        (splits_dir / f"{name}.json").write_text(json.dumps(json_rows, default=_json_default))
        torch.save([r["snap"] for r in rs if r.get("snap") is not None], splits_dir / f"{name}_snaps.pt")
        print(f"[ucr1-pack] {name} n={len(rs)} by_task={summary['n_split'][name]['by_task']}", flush=True)

    (out / "oracle_data" / "counts.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "oracle_data" / "counts.json").write_text(json.dumps(summary["n_all"], indent=2, default=_json_default))
    print(json.dumps(summary["n_all"], indent=2, default=_json_default), flush=True)
    print(f"[ucr1-pack] 1° oracle-best fraction={summary['n_all']['frac_oracle_best_1deg']:.3f}", flush=True)


if __name__ == "__main__":
    main()
