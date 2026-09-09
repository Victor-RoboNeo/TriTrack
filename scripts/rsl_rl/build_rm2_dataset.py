#!/usr/bin/env python3
"""Phase R-M2 — merge Loco Step-5 + Stoop Probe-B oracles. No PPO. No Isaac.

task_source / terrain_label / trigger_channel / utilities are metadata.
Runtime tensors are rec_obs (487) + z_nom + d_oracle only.
Split is episode-level (terrain|seed|clip), 70/15/15, never random-state.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

TERRAINS = ("plane", "light_rough", "slope", "steps")
SWEEP_KEYS = ("1.0", "2.5", "5.0", "7.5", "10.0")
THETA_BINS = (2.5, 5.0, 7.5, 10.0)
THETA_KEYS = ("2.5", "5.0", "7.5", "10.0")
IN_DIM = 487


def _ep_key(task: str, r: dict) -> str:
    return f"{task}|{r['terrain']}|{r['seed']}|{r['clip']}"


def _state_key(task: str, r: dict) -> str:
    ter = r.get("terrain_label") or r.get("terrain")
    return f"{task}|{ter}|{r['seed']}|{r['clip']}|{r['t0']}"


def _sweep(r: dict) -> dict[str, float]:
    raw = r.get("A_sweep_cm") or {}
    out = {}
    for k in SWEEP_KEYS:
        v = raw.get(k, raw.get(str(float(k))))
        if v is None:
            continue
        out[k] = float(v)
    return out


def _theta_label(sweep: dict[str, float]) -> tuple[float, float, int]:
    """argmin I among {2.5,5,7.5,10}. 1° is recorded but not a class."""
    vals = []
    for k, th in zip(THETA_KEYS, THETA_BINS):
        if k not in sweep:
            continue
        vals.append((float(sweep[k]), th, THETA_KEYS.index(k)))
    if not vals:
        return float("nan"), float("nan"), -1
    vals.sort(key=lambda x: x[0])
    i_star, th_star, idx = vals[0]
    return th_star, i_star, idx


def _unify(r: dict, task: str) -> dict | None:
    rec = r.get("rec_obs")
    d = r.get("d_star")
    z = r.get("z_nom")
    if rec is None or d is None or z is None:
        return None
    rec = np.asarray(rec, dtype=np.float32).reshape(-1)
    d = np.asarray(d, dtype=np.float32).reshape(-1)
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    if rec.size != IN_DIM or d.size != 16 or z.size != 16:
        return None
    dn = float(np.linalg.norm(d))
    if dn < 1e-8:
        return None
    d = (d / dn).astype(np.float32)
    sweep = _sweep(r)
    if len(sweep) < 4:
        return None
    th, i_th, idx = _theta_label(sweep)
    ch = r.get("trigger_channel")
    if not ch:
        ch = "E" if task == "loco" else "?"
    return {
        "task_source": task,
        "episode_id": _ep_key(task, r),
        "terrain_label": str(r["terrain"]),
        "seed": int(r["seed"]),
        "clip": str(r["clip"]),
        "t0": int(r["t0"]),
        "trigger_channel": str(ch),
        "rec_obs": rec.tolist(),
        "z_nom": z.tolist(),
        "d_oracle": d.tolist(),
        "utility_by_angle": sweep,
        "best_angle": float(th) if th == th else None,
        "best_utility": float(i_th) if i_th == i_th else None,
        "theta_idx": int(idx),
        "I_oracle_cm": float(r.get("I_oracle_cm", i_th)),
        "E0_cm": float(r.get("E0_cm", float("nan"))),
        "R_E": float(r.get("R_E", float("nan"))),
        "R_S": float(r.get("R_S")) if r.get("R_S") is not None else None,
    }


def _load_states(root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.exists():
        return rows
    for ter in TERRAINS:
        p = root / ter / "states.json"
        if p.exists():
            rows.extend(json.loads(p.read_text()))
    return rows


def _split_episodes(rows: list[dict], seed: int) -> dict[str, set[str]]:
    rng = np.random.default_rng(seed)
    splits = {"train": set(), "val": set(), "test": set()}
    for task in ("loco", "stoop"):
        by_ter: dict[str, list[str]] = {t: [] for t in TERRAINS}
        seen: set[str] = set()
        for r in rows:
            if r["task_source"] != task:
                continue
            k = r["episode_id"]
            if k in seen:
                continue
            seen.add(k)
            by_ter[r["terrain_label"]].append(k)
        for ter in TERRAINS:
            eps = list(by_ter[ter])
            rng.shuffle(eps)
            n = len(eps)
            if n == 0:
                continue
            n_train = int(round(0.70 * n))
            n_val = int(round(0.15 * n))
            if n_train + n_val >= n:
                n_val = max(0, n - n_train - 1)
            n_test = n - n_train - n_val
            splits["train"].update(eps[:n_train])
            splits["val"].update(eps[n_train : n_train + n_val])
            splits["test"].update(eps[n_train + n_val :])
            print(
                f"[rm2-data] {task}/{ter}: episodes {n} → "
                f"train {n_train} val {n_val} test {n_test}",
                flush=True,
            )
    return splits


def _count_block(rows: list[dict]) -> dict:
    ch = Counter(r["trigger_channel"] for r in rows)
    task = Counter(r["task_source"] for r in rows)
    ter = Counter(r["terrain_label"] for r in rows)
    th = Counter(r["best_angle"] for r in rows if r.get("best_angle") is not None)
    return {
        "n": len(rows),
        "n_episodes": len({r["episode_id"] for r in rows}),
        "by_task": dict(task),
        "by_channel": dict(ch),
        "by_terrain": dict(ter),
        "by_best_angle": {str(k): int(v) for k, v in sorted(th.items(), key=lambda x: (x[0] is None, x[0]))},
        "stoop_s_first": sum(1 for r in rows if r["task_source"] == "stoop" and r["trigger_channel"] == "S"),
        "stoop_e_first": sum(1 for r in rows if r["task_source"] == "stoop" and r["trigger_channel"] == "E"),
        "stoop_both": sum(1 for r in rows if r["task_source"] == "stoop" and r["trigger_channel"] == "both"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loco_root", type=str, default="/data/home/chenxiangyu/robotics/Anybody/results/p2r_step5")
    ap.add_argument(
        "--stoop_roots",
        type=str,
        default=(
            "/data/home/chenxiangyu/robotics/Anybody/results/p2r_rm_stoop_oracle,"
            "/data/home/chenxiangyu/robotics/Anybody/results/rm_intent_conditioned_recovery/data/stoop_expand"
        ),
    )
    ap.add_argument(
        "--out",
        type=str,
        default="/data/home/chenxiangyu/robotics/Anybody/results/rm_intent_conditioned_recovery/data",
    )
    ap.add_argument("--split_seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "merged").mkdir(parents=True, exist_ok=True)
    (out / "loco").mkdir(parents=True, exist_ok=True)
    (out / "stoop").mkdir(parents=True, exist_ok=True)

    loco_raw = _load_states(Path(args.loco_root))
    stoop_raw: list[dict] = []
    for root in [p.strip() for p in args.stoop_roots.split(",") if p.strip()]:
        got = _load_states(Path(root))
        print(f"[rm2-data] stoop source {root} n={len(got)}", flush=True)
        stoop_raw.extend(got)

    unified: list[dict] = []
    seen: set[str] = set()
    dropped = 0
    for task, raw in (("loco", loco_raw), ("stoop", stoop_raw)):
        for r in raw:
            rec = _unify(r, task)
            if rec is None:
                dropped += 1
                continue
            k = _state_key(task, rec)
            if k in seen:
                continue
            seen.add(k)
            unified.append(rec)
    print(f"[rm2-data] unified n={len(unified)} dropped={dropped}", flush=True)

    splits = _split_episodes(unified, seed=int(args.split_seed))
    buckets = {k: [r for r in unified if r["episode_id"] in splits[k]] for k in ("train", "val", "test")}

    def _dump(path: Path, rows: list[dict]) -> None:
        path.write_text(json.dumps(rows))

    _dump(out / "loco" / "records.json", [r for r in unified if r["task_source"] == "loco"])
    _dump(out / "stoop" / "records.json", [r for r in unified if r["task_source"] == "stoop"])
    for name, rs in buckets.items():
        _dump(out / "merged" / f"{name}.json", rs)
    _dump(out / "merged" / "all.json", unified)

    manifest = {
        "unit": "task|terrain|seed|clip",
        "split_seed": int(args.split_seed),
        "in_dim": IN_DIM,
        "theta_bins": list(THETA_BINS),
        "runtime_inputs": ["rec_obs", "z_nom"],
        "forbidden_runtime_inputs": ["task_source", "terrain_label", "trigger_channel", "utility_by_angle", "future"],
        "sources": {
            "loco": args.loco_root,
            "stoop": [p.strip() for p in args.stoop_roots.split(",") if p.strip()],
        },
        "n_all": _count_block(unified),
        "n_split": {k: _count_block(v) for k, v in buckets.items()},
        "n_test_slices": {
            "loco": _count_block([r for r in buckets["test"] if r["task_source"] == "loco"]),
            "stoop": _count_block([r for r in buckets["test"] if r["task_source"] == "stoop"]),
            "stoop_s_first": _count_block(
                [r for r in buckets["test"] if r["task_source"] == "stoop" and r["trigger_channel"] == "S"]
            ),
            "stoop_e_first": _count_block(
                [r for r in buckets["test"] if r["task_source"] == "stoop" and r["trigger_channel"] == "E"]
            ),
        },
        "episodes": {k: sorted(splits[k]) for k in splits},
        "n_episodes": {k: len(splits[k]) for k in splits},
    }
    (out / "merged" / "split.json").write_text(json.dumps(manifest, indent=2))
    (out / "loco" / "manifest.json").write_text(json.dumps({"n": manifest["n_all"]["by_task"].get("loco", 0), "source": args.loco_root}, indent=2))
    (out / "stoop" / "manifest.json").write_text(
        json.dumps({"n": manifest["n_all"]["by_task"].get("stoop", 0), "sources": manifest["sources"]["stoop"]}, indent=2)
    )
    print(json.dumps(manifest["n_all"], indent=2), flush=True)
    print(json.dumps(manifest["n_split"], indent=2), flush=True)
    print(f"[rm2-data] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
