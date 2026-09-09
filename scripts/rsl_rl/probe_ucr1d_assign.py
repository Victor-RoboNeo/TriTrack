#!/usr/bin/env python3
"""UCR-1D: offline kNN assignments on unified recovery observation.

No Isaac. No new controller. No task ID at retrieval time for pooled kNN.
Same-task / per-source kNN are diagnostic only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

TASKS = ("loco", "stoop", "reach", "carry")
DEPLOY = (2.5, 5.0, 7.5, 10.0)
TOL_M = 0.0025  # 0.25 cm


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _load(split_dir: Path, name: str) -> list[dict]:
    return json.loads((split_dir / f"{name}.json").read_text())


def _pack(rows: list[dict]) -> dict:
    x = np.asarray([r["rec_obs"] for r in rows], dtype=np.float32)
    d = np.asarray([r["d_oracle"] for r in rows], dtype=np.float32)
    dn = np.linalg.norm(d, axis=-1, keepdims=True)
    d = d / np.clip(dn, 1e-8, None)
    z = np.asarray([r["z_nom"] for r in rows], dtype=np.float32)
    th = np.asarray([DEPLOY[int(r.get("theta_idx", 1))] for r in rows], dtype=np.float32)
    task = np.asarray([r["task_source"] for r in rows])
    clip = np.asarray([r["clip"] for r in rows])
    seed = np.asarray([int(r["seed"]) for r in rows], dtype=np.int32)
    t = np.asarray([int(r["t0"]) for r in rows], dtype=np.int32)
    lam = np.asarray([float(r["lam"]) for r in rows], dtype=np.float32)
    j_ora = np.asarray([r["j_ora"] for r in rows], dtype=np.float32)
    i_ora = np.asarray([r["i_ora"] for r in rows], dtype=np.float32)
    i_rm3 = np.asarray([r["i_rm3"] for r in rows], dtype=np.float32)
    return {
        "x": x, "d": d, "z": z, "th": th, "task": task, "clip": clip, "seed": seed,
        "t": t, "lam": lam, "j_ora": j_ora, "i_ora": i_ora, "i_rm3": i_rm3, "rows": rows,
    }


def _zscore(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def _nn(query: np.ndarray, gallery: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Exact L2 kNN. query [Nq,D], gallery [Ng,D] → idx [Nq,k], dist [Nq,k]."""
    # (q-g)^2 = q^2 + g^2 - 2 q g
    q2 = (query * query).sum(-1, keepdims=True)
    g2 = (gallery * gallery).sum(-1)
    d2 = q2 + g2[None, :] - 2.0 * (query @ gallery.T)
    np.maximum(d2, 0.0, out=d2)
    if k >= gallery.shape[0]:
        idx = np.argsort(d2, axis=1)
        dist = np.take_along_axis(d2, idx, axis=1)
        return idx, np.sqrt(dist)
    idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    dist = np.take_along_axis(d2, idx, axis=1)
    order = np.argsort(dist, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    dist = np.take_along_axis(dist, order, axis=1)
    return idx, np.sqrt(dist)


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8, None)
    bn = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8, None)
    return (an * bn).sum(-1)


def _mag_ambiguity(rows: list[dict]) -> dict:
    n_eq = []
    n_eq_deploy = []
    for r in rows:
        util_all = r.get("utility_by_angle_all") or {}
        j_ora = float(r["j_ora"])
        vals = []
        for k, v in util_all.items():
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        vals = np.asarray(vals, dtype=np.float64)
        n_eq.append(int(np.sum(vals <= j_ora + TOL_M)) if vals.size else 1)
        dep = np.asarray([float((r.get("utility_by_angle") or {}).get(str(th), 1e9)) for th in DEPLOY])
        n_eq_deploy.append(int(np.sum(dep <= j_ora + TOL_M)))
    n_eq = np.asarray(n_eq)
    n_eq_deploy = np.asarray(n_eq_deploy)
    return {
        "n": int(len(rows)),
        "mean_n_eq_sweep": float(n_eq.mean()) if n_eq.size else float("nan"),
        "P_n_eq_ge_2_sweep": float((n_eq >= 2).mean()) if n_eq.size else float("nan"),
        "mean_n_eq_deploy": float(n_eq_deploy.mean()) if n_eq_deploy.size else float("nan"),
        "P_n_eq_ge_2_deploy": float((n_eq_deploy >= 2).mean()) if n_eq_deploy.size else float("nan"),
        "hist_sweep": {str(i): int((n_eq == i).sum()) for i in range(0, 6)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1, help="Executed neighbor is k=1; k>1 only for label-disagreement stats.")
    args = ap.parse_args()
    splits = Path(args.splits)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_rows = _load(splits, "train")
    test_rows = _load(splits, "test")
    tr = _pack(train_rows)
    te = _pack(test_rows)
    mu = tr["x"].mean(0)
    sd = tr["x"].std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    xtr = _zscore(tr["x"], mu, sd)
    xte = _zscore(te["x"], mu, sd)

    k_exec = 1
    k_diag = max(int(args.k), 5)

    idx_p, dist_p = _nn(xte, xtr, k=k_diag)
    nn1 = idx_p[:, 0]
    d_pooled = tr["d"][nn1]
    th_pooled = tr["th"][nn1]
    src_pooled = tr["task"][nn1]
    cos_own = _cos(te["d"], d_pooled)

    # same-task
    d_same = np.zeros_like(te["d"])
    th_same = np.zeros(len(test_rows), dtype=np.float32)
    dist_same = np.zeros(len(test_rows), dtype=np.float32)
    cos_same = np.zeros(len(test_rows), dtype=np.float32)
    idx_same = np.zeros(len(test_rows), dtype=np.int32)
    for t in TASKS:
        m_te = te["task"] == t
        m_tr = tr["task"] == t
        if not m_te.any() or not m_tr.any():
            continue
        gidx = np.flatnonzero(m_tr)
        ii, dd = _nn(xte[m_te], xtr[m_tr], k=k_diag)
        loc = gidx[ii[:, 0]]
        d_same[m_te] = tr["d"][loc]
        th_same[m_te] = tr["th"][loc]
        dist_same[m_te] = dd[:, 0]
        cos_same[m_te] = _cos(te["d"][m_te], tr["d"][loc])
        idx_same[m_te] = loc

    # per-source
    d_src = {s: np.zeros_like(te["d"]) for s in TASKS}
    th_src = {s: np.zeros(len(test_rows), dtype=np.float32) for s in TASKS}
    dist_src = {s: np.full(len(test_rows), np.nan, dtype=np.float32) for s in TASKS}
    cos_src = {s: np.full(len(test_rows), np.nan, dtype=np.float32) for s in TASKS}
    for s in TASKS:
        m_tr = tr["task"] == s
        gidx = np.flatnonzero(m_tr)
        if gidx.size == 0:
            continue
        ii, dd = _nn(xte, xtr[m_tr], k=1)
        loc = gidx[ii[:, 0]]
        d_src[s] = tr["d"][loc]
        th_src[s] = tr["th"][loc]
        dist_src[s] = dd[:, 0]
        cos_src[s] = _cos(te["d"], tr["d"][loc])

    # k=5 neighbor d* disagreement (pooled)
    k5 = min(5, idx_p.shape[1])
    neigh_d = tr["d"][idx_p[:, :k5]]  # [N,k,16]
    pair_cos = []
    for i in range(k5):
        for j in range(i + 1, k5):
            pair_cos.append(_cos(neigh_d[:, i], neigh_d[:, j]))
    pair_cos = np.stack(pair_cos, 0) if pair_cos else np.zeros((1, len(test_rows)))
    mean_pair_cos = pair_cos.mean(0)

    mag_all = _mag_ambiguity(test_rows)
    mag_by = {t: _mag_ambiguity([r for r in test_rows if r["task_source"] == t]) for t in TASKS}

    src_conf = {}
    for t in TASKS:
        m = te["task"] == t
        c = Counter(src_pooled[m].tolist()) if m.any() else {}
        src_conf[t] = {s: int(c.get(s, 0)) for s in TASKS}
        src_conf[t]["n"] = int(m.sum())
        src_conf[t]["frac_other"] = float(1.0 - c.get(t, 0) / m.sum()) if m.any() else float("nan")

    np.savez_compressed(
        out / "assignments.npz",
        rec_obs=te["x"],
        z_nom=te["z"],
        d_oracle=te["d"],
        d_pooled=d_pooled,
        th_pooled=th_pooled,
        dist_pooled=dist_p[:, 0].astype(np.float32),
        src_pooled=src_pooled,
        cos_pooled_vs_own=cos_own.astype(np.float32),
        d_same=d_same,
        th_same=th_same,
        dist_same=dist_same,
        cos_same_vs_own=cos_same.astype(np.float32),
        **{f"d_from_{s}": d_src[s] for s in TASKS},
        **{f"th_from_{s}": th_src[s] for s in TASKS},
        **{f"dist_from_{s}": dist_src[s] for s in TASKS},
        **{f"cos_from_{s}": cos_src[s] for s in TASKS},
        task=te["task"],
        clip=te["clip"],
        seed=te["seed"],
        t=te["t"],
        lam=te["lam"],
        j_ora=te["j_ora"],
        i_ora=te["i_ora"],
        i_rm3=te["i_rm3"],
        mean_pair_cos_k5=mean_pair_cos.astype(np.float32),
        mu=mu.astype(np.float32),
        sd=sd.astype(np.float32),
    )

    offline = {
        "k_exec": k_exec,
        "k_diag": k5,
        "metric": "z-scored Euclidean on rec_obs 487 (train mu/sd); no task ID for pooled",
        "n_train": {t: int((tr['task'] == t).sum()) for t in TASKS},
        "n_test": {t: int((te['task'] == t).sum()) for t in TASKS},
        "pooled_nn_source": src_conf,
        "cos_neighbor_d_vs_own_oracle": {
            "pooled": {
                "median": float(np.median(cos_own)),
                "mean": float(cos_own.mean()),
                "P_gt_0": float((cos_own > 0).mean()),
                "P_gt_0.5": float((cos_own > 0.5).mean()),
            },
            "same_task": {
                "median": float(np.median(cos_same)),
                "mean": float(cos_same.mean()),
                "P_gt_0": float((cos_same > 0).mean()),
                "P_gt_0.5": float((cos_same > 0.5).mean()),
            },
            "by_target": {
                t: {
                    "pooled_median": float(np.median(cos_own[te["task"] == t])) if (te["task"] == t).any() else None,
                    "same_median": float(np.median(cos_same[te["task"] == t])) if (te["task"] == t).any() else None,
                }
                for t in TASKS
            },
        },
        "k5_neighbor_d_pairwise_cosine": {
            "median": float(np.median(mean_pair_cos)),
            "mean": float(mean_pair_cos.mean()),
            "P_lt_0": float((mean_pair_cos < 0).mean()),
            "note": "Low pairwise cosine among nearby o_R states ⇒ label d* is not locally unique.",
        },
        "magnitude_ambiguity_0.25cm": {
            "all": mag_all,
            "by_task": mag_by,
            "note": "Count of sweep/deploy magnitudes with J <= J* + 0.25cm. Direction-set needs FD clones.",
        },
        "stoop_pooled_nn_sources": src_conf["stoop"],
        "note": "Assignments only. Clone I is measured by eval --ucr1d_probe.",
    }
    (out / "offline.json").write_text(json.dumps(offline, indent=2, default=_json_default))
    print(json.dumps(offline["pooled_nn_source"], indent=2), flush=True)
    print(json.dumps(offline["cos_neighbor_d_vs_own_oracle"], indent=2), flush=True)
    print(json.dumps(offline["magnitude_ambiguity_0.25cm"], indent=2, default=_json_default), flush=True)
    print(f"[ucr1d] wrote {out / 'assignments.npz'}", flush=True)


if __name__ == "__main__":
    main()
