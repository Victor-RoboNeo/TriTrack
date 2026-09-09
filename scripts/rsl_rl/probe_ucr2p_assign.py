#!/usr/bin/env python3
"""UCR-2P: process-conditioned kNN assignments. No Isaac. No closed-loop.

Join UCR-1 oracle labels with causal 500 ms process dumps.
Neighbor (d*, θ*) is transferred; GO is clone utility, not cosine.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TASKS = ("loco", "stoop", "reach", "carry")
DEPLOY = (2.5, 5.0, 7.5, 10.0)
TOL_T = 5


def _load_rows(split_dir: Path, name: str) -> list[dict]:
    return json.loads((split_dir / f"{name}.json").read_text())


def _pack(rows: list[dict]) -> dict:
    x = np.asarray([r["rec_obs"] for r in rows], dtype=np.float32)
    d = np.asarray([r["d_oracle"] for r in rows], dtype=np.float32)
    dn = np.linalg.norm(d, axis=-1, keepdims=True)
    d = d / np.clip(dn, 1e-8, None)
    th = np.asarray([DEPLOY[int(r.get("theta_idx", 1))] for r in rows], dtype=np.float32)
    task = np.asarray([r["task_source"] for r in rows])
    clip = np.asarray([r["clip"] for r in rows])
    seed = np.asarray([int(r["seed"]) for r in rows], dtype=np.int32)
    t = np.asarray([int(r["t0"]) for r in rows], dtype=np.int32)
    lam = np.asarray([float(r["lam"]) for r in rows], dtype=np.float32)
    return {"x": x, "d": d, "th": th, "task": task, "clip": clip, "seed": seed, "t": t, "lam": lam, "n": len(rows)}


def _zscore(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def _nn(query: np.ndarray, gallery: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
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


def _load_process(dump_root: Path) -> dict:
    """key (task, clip, seed, lam) -> list of (t, process[H,D], rec_obs)."""
    out: dict[tuple, list] = {}
    for p in sorted(dump_root.glob("*/lam_*/plane/process.npz")):
        b = np.load(p, allow_pickle=True)
        if "skipped" in b.files and bool(b["skipped"]):
            continue
        n = int(b["n"]) if "n" in b.files else int(b["t"].shape[0])
        if n <= 0 or "process" not in b.files:
            continue
        proc = np.asarray(b["process"], dtype=np.float32)
        clips = np.asarray(b["clip"]).astype(str)
        seeds = np.asarray(b["seed"]).astype(int)
        ts = np.asarray(b["t"]).astype(int)
        tasks = np.asarray(b["task"]).astype(str)
        lams = np.asarray(b["lam"]).astype(np.float32)
        rec = np.asarray(b["rec_obs"], dtype=np.float32) if "rec_obs" in b.files else None
        for i in range(min(n, proc.shape[0])):
            key = (tasks[i], clips[i], int(seeds[i]), float(lams[i]))
            rec_i = rec[i] if rec is not None and i < rec.shape[0] else None
            out.setdefault(key, []).append((int(ts[i]), proc[i], rec_i))
    for key in out:
        out[key].sort(key=lambda x: x[0])
    return out


def _join_one(index: dict, task: str, clip: str, seed: int, t: int, lam: float):
    key = (task, clip, int(seed), float(lam))
    cands = index.get(key)
    if not cands:
        # lam float noise
        for k, v in index.items():
            if k[0] == task and k[1] == clip and k[2] == int(seed) and abs(k[3] - float(lam)) < 1e-3:
                cands = v
                break
    if not cands:
        return None
    best = min(cands, key=lambda x: abs(x[0] - int(t)))
    if abs(best[0] - int(t)) > TOL_T:
        return None
    return best[1]


def _summary(p: np.ndarray, nlast: int) -> np.ndarray:
    w = np.asarray(p, dtype=np.float32)
    if w.ndim != 2 or w.shape[0] == 0:
        return np.zeros(32, dtype=np.float32)
    w = w[-nlast:]
    e = w[:, 0]
    re = w[:, 4]
    v = w[:, 15:18]
    c = w[:, 24:27]
    zn = w[:, 27:43]
    parts = [
        np.asarray([e[-1] - e[0], float(e.mean()), float(e.std()) if e.size > 1 else 0.0], dtype=np.float32),
        np.asarray([re[-1] - re[0], float(re.max())], dtype=np.float32),
        (v[-1] - v[0]).astype(np.float32),
        np.asarray(
            [
                float(c[:, 0].mean()),
                float(c[:, 1].mean()),
                float(np.abs(np.diff(c[:, 2])).sum()) if c.shape[0] > 1 else 0.0,
            ],
            dtype=np.float32,
        ),
        (zn[-1] - zn[0]).astype(np.float32),
        np.asarray([w[-1, 21] - w[0, 21], w[-1, 22] - w[0, 22], w[-1, 23]], dtype=np.float32),
    ]
    return np.concatenate(parts, 0)


def _pca_fit(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(0)
    xc = x - mu
    # economy SVD
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    k = max(1, min(k, vt.shape[0], x.shape[0] - 1 if x.shape[0] > 1 else 1))
    return vt[:k], mu


def _pca_apply(x: np.ndarray, vt: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (x - mu) @ vt.T


def _block_z(tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = tr.mean(0)
    sd = tr.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return _zscore(tr, mu, sd), _zscore(te, mu, sd)


def _feat_static(tr, te):
    return _block_z(tr["x"], te["x"])


def _window(proc: np.ndarray, nlast: int) -> np.ndarray:
    w = proc[-nlast:]
    if w.shape[0] < nlast:
        pad = np.zeros((nlast - w.shape[0], w.shape[1]), dtype=np.float32)
        w = np.concatenate([pad, w], 0)
    return w.reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca", type=int, default=64)
    args = ap.parse_args()
    splits = Path(args.splits)
    dump = Path(args.dump)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index = _load_process(dump)
    print(f"[ucr2p] process keys={len(index)}", flush=True)

    train_rows = _load_rows(splits, "train")
    test_rows = _load_rows(splits, "test")
    tr = _pack(train_rows)
    te = _pack(test_rows)

    def attach(pack, rows):
        proc = []
        keep = []
        for i, r in enumerate(rows):
            p = _join_one(index, r["task_source"], r["clip"], int(r["seed"]), int(r["t0"]), float(r["lam"]))
            if p is None:
                continue
            proc.append(np.asarray(p, dtype=np.float32))
            keep.append(i)
        return np.stack(proc, 0) if proc else np.zeros((0, 25, 48), dtype=np.float32), np.asarray(keep, dtype=np.int64)

    ptr, ktr = attach(tr, train_rows)
    pte, kte = attach(te, test_rows)
    print(f"[ucr2p] joined train {len(ktr)}/{tr['n']} test {len(kte)}/{te['n']}", flush=True)
    meta = {
        "n_process_keys": len(index),
        "n_train_joined": int(len(ktr)),
        "n_test_joined": int(len(kte)),
        "n_train": int(tr["n"]),
        "n_test": int(te["n"]),
        "tol_t": TOL_T,
    }
    if len(ktr) < 20 or len(kte) < 10:
        (out / "offline.json").write_text(json.dumps({"error": "too_few_joined", **meta}, indent=2))
        print("[ucr2p] HOLD too few joined rows", flush=True)
        return

    def sub(pack, idx):
        return {k: pack[k][idx] if isinstance(pack[k], np.ndarray) else pack[k] for k in pack if k != "n"}

    trj = sub(tr, ktr)
    tej = sub(te, kte)
    trj["proc"] = ptr
    tej["proc"] = pte

    xtr_s, xte_s = _feat_static(trj, tej)
    flat_tr_200 = np.stack([_window(p, 10) for p in ptr], 0)
    flat_te_200 = np.stack([_window(p, 10) for p in pte], 0)
    flat_tr_500 = np.stack([_window(p, 25) for p in ptr], 0)
    flat_te_500 = np.stack([_window(p, 25) for p in pte], 0)
    sum_tr_200 = np.stack([_summary(p, 10) for p in ptr], 0)
    sum_te_200 = np.stack([_summary(p, 10) for p in pte], 0)
    sum_tr_500 = np.stack([_summary(p, 25) for p in ptr], 0)
    sum_te_500 = np.stack([_summary(p, 25) for p in pte], 0)

    f200_tr, f200_te = _block_z(flat_tr_200, flat_te_200)
    s200_tr, s200_te = _block_z(sum_tr_200, sum_te_200)
    s500_tr, s500_te = _block_z(sum_tr_500, sum_te_500)
    vt, mu = _pca_fit(flat_tr_500, int(args.pca))
    p500_tr = _pca_apply(flat_tr_500, vt, mu)
    p500_te = _pca_apply(flat_te_500, vt, mu)
    p500_tr, p500_te = _block_z(p500_tr, p500_te)

    feats = {
        "static": (xtr_s, xte_s),
        "p200": (np.concatenate([xtr_s, f200_tr, s200_tr], 1), np.concatenate([xte_s, f200_te, s200_te], 1)),
        "p500": (np.concatenate([xtr_s, p500_tr, s500_tr], 1), np.concatenate([xte_s, p500_te, s500_te], 1)),
        "p200_sum": (np.concatenate([xtr_s, s200_tr], 1), np.concatenate([xte_s, s200_te], 1)),
        "p500_sum": (np.concatenate([xtr_s, s500_tr], 1), np.concatenate([xte_s, s500_te], 1)),
    }

    def knn_pack(xtr, xte):
        idx, dist = _nn(xte, xtr, k=1)
        nn = idx[:, 0]
        return trj["d"][nn], trj["th"][nn], trj["task"][nn], dist[:, 0], nn

    d_p200, th_p200, src_p200, dist_p200, _ = knn_pack(*feats["p200"])
    d_p500, th_p500, src_p500, dist_p500, _ = knn_pack(*feats["p500"])
    d_p200_sum, th_p200_sum, _, _, _ = knn_pack(*feats["p200_sum"])
    d_p500_sum, th_p500_sum, _, _, _ = knn_pack(*feats["p500_sum"])

    def same_task(xtr, xte):
        d = np.zeros_like(tej["d"])
        th = np.zeros(len(kte), dtype=np.float32)
        for t in TASKS:
            m_te = tej["task"] == t
            m_tr = trj["task"] == t
            if not m_te.any() or not m_tr.any():
                continue
            gidx = np.flatnonzero(m_tr)
            ii, _dd = _nn(xte[m_te], xtr[m_tr], k=1)
            loc = gidx[ii[:, 0]]
            d[m_te] = trj["d"][loc]
            th[m_te] = trj["th"][loc]
        return d, th

    d_p200_same, th_p200_same = same_task(*feats["p200"])
    d_p500_same, th_p500_same = same_task(*feats["p500"])

    # Map joined test rows back to full test index for probe matching (clip,seed,t,task,lam).
    np.savez_compressed(
        out / "assignments.npz",
        clip=tej["clip"],
        seed=tej["seed"],
        t=tej["t"],
        task=tej["task"],
        lam=tej["lam"],
        d_p200=d_p200,
        th_p200=th_p200,
        d_p500=d_p500,
        th_p500=th_p500,
        d_p200_same=d_p200_same,
        th_p200_same=th_p200_same,
        d_p500_same=d_p500_same,
        th_p500_same=th_p500_same,
        d_p200_sum=d_p200_sum,
        th_p200_sum=th_p200_sum,
        d_p500_sum=d_p500_sum,
        th_p500_sum=th_p500_sum,
        src_p200=src_p200,
        src_p500=src_p500,
        dist_p200=dist_p200.astype(np.float32),
        dist_p500=dist_p500.astype(np.float32),
        join_idx=kte,
    )
    by_task = {}
    for t in TASKS:
        m = tej["task"] == t
        by_task[t] = {
            "n": int(m.sum()),
            "src_p500": {s: int((src_p500[m] == s).sum()) for s in TASKS},
        }
    offline = {
        **meta,
        "feat_dim": {k: int(v[0].shape[1]) for k, v in feats.items()},
        "pca_k": int(vt.shape[0]),
        "by_task": by_task,
        "no_transformer": True,
        "no_gru_yet": True,
        "note": "GO is clone I, not cosine. static kNN already measured in UCR-1D; not re-cloned.",
    }
    (out / "offline.json").write_text(json.dumps(offline, indent=2, default=str))
    print(json.dumps({"joined_test": int(len(kte)), "by_task": {t: by_task[t]["n"] for t in TASKS}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
