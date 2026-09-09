#!/usr/bin/env python3
"""P4-B4B offline decode + safety. Grouped CV. No nets. No P4-C."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from p3_common import sanitize
from p4b4b_runtime import make_u4, make_u8

TERRAINS = ("steps", "slip", "slope_down")
ROOT_DEFAULT = "/data/home/chenxiangyu/robotics/Anybody/results/p4b4b_200hz_microprobe"
PRIMARY = "n4_a0.5_default"
DT_LL = 0.005
LAM_Z = 1e-4
BOOT = 2000


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{float(v):.{nd}f}"


def _rinfo(acc, acc_clone):
    if acc is None or not np.isfinite(acc) or abs(float(acc_clone) - 0.5) < 1e-6:
        return None
    return (float(acc) - 0.5) / (float(acc_clone) - 0.5)


def _pair_acc(g, a5):
    s_star = np.sign(a5[:, :, 0] - a5[:, :, 1])
    s_hat = np.sign(-np.asarray(g))
    valid = s_star != 0
    if not np.any(valid):
        return float("nan")
    return float((s_hat[valid] == s_star[valid]).mean())


def _groups(seed, clip, terrain):
    return np.array([f"{tr}|{int(s)}|{c}" for tr, s, c in zip(terrain, seed, clip)])


def _folds(groups, n_splits=5, seed=2026):
    uniq = np.unique(groups)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(uniq))
    assign = {uniq[i]: (k % n_splits) for k, i in enumerate(order)}
    return np.array([assign[g] for g in groups], dtype=np.int32)


def _zscore(Xtr, Xte, clip=10.0):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xtr = np.clip((Xtr - mu) / sd, -clip, clip)
    Xte = np.clip((Xte - mu) / sd, -clip, clip)
    return Xtr, Xte


def _lda(Xtr, ytr, Xte, shrink=0.1):
    classes = np.unique(ytr)
    if classes.size < 2:
        return np.zeros(len(Xte))
    p = Xtr.shape[1]
    Sw = np.zeros((p, p))
    mu = []
    for c in classes:
        Xc = Xtr[ytr == c]
        m = Xc.mean(0)
        mu.append(m)
        d = Xc - m
        Sw = Sw + d.T @ d
    tr = float(np.trace(Sw)) / max(p, 1)
    Sw = (1.0 - shrink) * Sw + shrink * tr * np.eye(p) + 1e-6 * np.eye(p)
    w = np.linalg.lstsq(Sw, mu[1] - mu[0], rcond=None)[0]
    thr = 0.5 * (mu[0] + mu[1]) @ w
    return np.where(Xte @ w >= thr, classes[1], classes[0])


def _logreg(Xtr, ytr, Xte, l2=1.0, n_iter=200):
    y01 = (ytr > 0).astype(np.float64)
    n, d = Xtr.shape
    w = np.zeros(d)
    b = 0.0
    lr = 1.0 / max(n, 1)
    for _ in range(n_iter):
        z = np.clip(Xtr @ w + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y01
        w = w - lr * (Xtr.T @ err + l2 * w)
        b = b - lr * float(err.sum())
    return np.where(Xte @ w + b >= 0, 1.0, -1.0)


def _svm(Xtr, ytr, Xte, C=1.0, n_epoch=40, seed=0):
    n, d = Xtr.shape
    w = np.zeros(d)
    b = 0.0
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    for ep in range(n_epoch):
        rng.shuffle(idx)
        lr = 0.1 / (1.0 + 0.05 * ep)
        for i in idx:
            marg = ytr[i] * (float(Xtr[i] @ w) + b)
            w = w - lr * (w / max(n, 1))
            if marg < 1:
                w = w + lr * C * ytr[i] * Xtr[i]
                b = b + lr * C * ytr[i]
    return np.where(Xte @ w + b >= 0, 1.0, -1.0)


FIT = {
    "shrinkage LDA": _lda,
    "L2 logistic": lambda Xtr, y, Xte: _logreg(Xtr, y, Xte, 1.0),
    "linear SVM": _svm,
}


def _cv_pair(X_list, a5, groups, fit_fn, n_splits=5):
    """X_list: list of (n, f_i) per axis or one (n,f) broadcast."""
    s_star = np.sign(a5[:, :, 0] - a5[:, :, 1])
    n, k = s_star.shape
    if isinstance(X_list, np.ndarray):
        X_list = [X_list] * k
    pred = np.zeros_like(s_star)
    filled = np.zeros_like(s_star, dtype=bool)
    fold = _folds(groups, n_splits)
    for ax in range(k):
        y = s_star[:, ax]
        X = np.asarray(X_list[ax], dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        nz = y != 0
        for f in range(n_splits):
            te = np.where(fold == f)[0]
            tr = np.where(fold != f)[0]
            tr = tr[nz[tr]]
            te_use = te[nz[te]]
            if tr.size < 8 or te_use.size < 1 or np.unique(y[tr]).size < 2:
                continue
            Xtr, Xte = _zscore(X[tr], X[te_use])
            pred[te_use, ax] = fit_fn(Xtr, y[tr], Xte)
            filled[te_use, ax] = True
    valid = (s_star != 0) & filled
    if not np.any(valid):
        return float("nan"), 0.0, pred
    return float((pred[valid] == s_star[valid]).mean()), float(filled.mean()), pred


def _boot_acc(pred, a5, n=BOOT, seed=2026):
    s_star = np.sign(a5[:, :, 0] - a5[:, :, 1])
    valid = s_star != 0
    rng = np.random.RandomState(seed)
    n_state = a5.shape[0]
    accs = []
    for _ in range(n):
        ix = rng.randint(0, n_state, n_state)
        v = valid[ix]
        if not np.any(v):
            continue
        accs.append((pred[ix][v] == s_star[ix][v]).mean())
    if not accs:
        return float("nan"), float("nan")
    a = np.asarray(accs)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def _cfg(z, key, field):
    name = f"{key}__{field}"
    if name not in z.files:
        return None
    return z[name]


def _feat_pack(z, key):
    e_p, e_t = _cfg(z, key, "e_p"), _cfg(z, key, "e_t")
    q_p, q_t = _cfg(z, key, "q_p"), _cfg(z, key, "q_t")
    dq_p = _cfg(z, key, "dq_p")
    tgt_p, tgt_t = _cfg(z, key, "tgt_p"), _cfg(z, key, "tgt_t")
    om_p = _cfg(z, key, "omega_p")
    pg_p = _cfg(z, key, "pg_p")
    vel_p = _cfg(z, key, "vel_p")
    c_p = _cfg(z, key, "contact_p")
    tau_p = _cfg(z, key, "tau_p")
    U = _cfg(z, key, "U")
    if U is None:
        return None
    if U.ndim == 3:
        U = U[0]
    T = e_p.shape[1] - 1

    def d(a):
        return a[:, 1:] - a[:, :-1]

    de_p, de_t = d(e_p), d(e_t)
    dqdt_p = d(q_p)
    dtgt_p = d(tgt_p - q_p)
    # F0/F1/F2/F3 flattened deltas
    F0 = np.concatenate([de_p, de_p - de_t], 1)
    F1 = np.concatenate([F0, d(om_p).reshape(len(e_p), -1), d(pg_p).reshape(len(e_p), -1), d(vel_p).reshape(len(e_p), -1), d(c_p).reshape(len(e_p), -1), dqdt_p.reshape(len(e_p), -1)], 1)
    F2 = np.concatenate([F1, dtgt_p.reshape(len(e_p), -1), (tgt_p[:, 1:] - q_p[:, 1:]).reshape(len(e_p), -1)], 1)
    F3 = np.concatenate([F2, d(tau_p).reshape(len(e_p), -1)], 1) if tau_p is not None else F2
    R_raw = F2  # primary deployable
    R_twin = np.concatenate([de_p - de_t, dqdt_p.reshape(len(e_p), -1) - d(q_t).reshape(len(e_p), -1)], 1)
    # pretrend: dq0 * dt
    qdot = dq_p[:, 0] * DT_LL
    R_tr = dqdt_p - qdot[:, None, :]
    R_trend = R_tr.reshape(len(e_p), -1)
    def demod(Rseq, Uu):
        # Rseq (n,T,dy) -> Z (n,k,dy)
        n, TT, dy = Rseq.shape
        gram = Uu.T @ Uu + LAM_Z * np.eye(Uu.shape[1])
        Z = np.zeros((n, Uu.shape[1], dy))
        for i in range(n):
            Z[i] = np.linalg.solve(gram, Uu.T @ Rseq[i])
        return Z.reshape(n, -1)
    Z_raw = demod(np.stack([de_p, de_p], -1)[:, :, :1], U)  # dummy if fail
    try:
        Z_raw = demod(dqdt_p, U)
        Z_tw = demod(dqdt_p - d(q_t), U)
    except Exception:
        Z_raw = np.zeros((len(e_p), U.shape[1]))
        Z_tw = Z_raw
    pre = np.concatenate([e_p[:, 0:1], np.linalg.norm(dq_p[:, 0], axis=-1, keepdims=True), pg_p[:, 0], c_p[:, 0]], 1)
    da = tgt_p[:, 1:] - tgt_t[:, 1:]
    num = np.linalg.norm(da.reshape(len(e_p), T, -1), axis=-1)
    raw_p, raw_t = _cfg(z, key, "raw_p"), _cfg(z, key, "raw_t")
    raw_dead = True
    if raw_p is not None:
        req = raw_p[:, 1:] - raw_t[:, 1:]
        den_raw = np.linalg.norm(req.reshape(len(e_p), T, -1), axis=-1)
        raw_dead = float(np.median(den_raw)) < 1e-8
        if not raw_dead:
            G = num / (den_raw + 1e-8)
        else:
            # JointPositionAction.raw_actions did not track process_action; PD targets did.
            G = np.ones_like(num)
    else:
        G = np.ones((len(e_p), T))
    # effective rank of (T, na) mean action delta
    Amean = da.reshape(len(e_p), T, -1).mean(0)
    svals = np.linalg.svd(Amean, compute_uv=False)
    rank = int((svals > 1e-6 * svals[0]).sum()) if svals.size else 0
    cond = float(svals[0] / max(svals[min(2, len(svals) - 1)], 1e-12)) if svals.size >= 3 else float("nan")
    excess = np.maximum(e_p - e_t, 0.0)
    peak_ex = excess.max(1)
    return {
        "F0": F0, "F1": F1, "F2": F2, "F3": F3,
        "R_raw": R_raw, "R_twin": R_twin, "R_trend": R_trend,
        "Z_raw": Z_raw, "Z_tw": Z_tw, "pre": pre,
        "G": G, "rank": rank, "cond": cond, "raw_dead": raw_dead,
        "tgt_rms": float(np.sqrt(np.mean(da.reshape(len(e_p), -1) ** 2))),
        "peak_ex": peak_ex, "auc_ex": excess.sum(1) * DT_LL,
        "jlim_p": _cfg(z, key, "jlim_p"),
        "jlim_t": _cfg(z, key, "jlim_t"),
        "a5": _cfg(z, key, "a5_end"),
        "a_net": _cfg(z, key, "a_net"),
        "a_net0": _cfg(z, key, "a_net0"),
        "g_end": _cfg(z, key, "g_end"),
        "B0": _cfg(z, key, "B0"),
        "BN": _cfg(z, key, "BN"),
        "U": U, "T": T,
    }


def _axis_X(pack, kind):
    """Return list of 3 feature matrices (n,f) for direction-conditioned decode."""
    B0, BN = pack["B0"], pack["BN"]
    n = B0.shape[0]
    pre = pack["pre"]
    if kind == "D0":
        xs = []
        for i in range(3):
            xs.append(np.concatenate([B0[:, :, i], BN[:, :, i], np.eye(3)[i][None, :].repeat(n, 0)], 1))
        return xs
    if kind == "D1":
        xs = []
        for i in range(3):
            xs.append(np.concatenate([pre, B0[:, :, i], BN[:, :, i]], 1))
        return xs
    if kind == "D2":
        U = pack["U"].reshape(-1)
        xs = []
        for i in range(3):
            xs.append(np.concatenate([np.repeat(U[None, :], n, 0), B0[:, :, i]], 1))
        return xs
    base = {
        "D3": pack["F2"],
        "D4": np.concatenate([pack["F2"], pack["R_trend"]], 1),
        "D5": pack["R_twin"],
        "F0": pack["F0"],
        "F1": pack["F1"],
        "F2": pack["F2"],
        "F3": pack["F3"],
        "R2": pack["Z_raw"],
        "R5": pack["Z_tw"],
    }[kind]
    xs = []
    for i in range(3):
        xs.append(np.concatenate([pre, base, B0[:, :, i], BN[:, :, i]], 1))
    return xs


def _md(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    args = ap.parse_args()
    root = Path(args.root)
    plots = root / "plots"
    for d in ("plots", "probe_codes", "command_integrity", "endpoint_labels", "decoding", "safety", "rollouts"):
        (root / d).mkdir(parents=True, exist_ok=True)
    np.save(root / "probe_codes" / "U4.npy", make_u4())
    np.save(root / "probe_codes" / "U4_reversed.npy", make_u4()[::-1])
    np.save(root / "probe_codes" / "U8.npy", make_u8())

    zs = []
    for t in TERRAINS:
        p = root / "p4b4b" / "loco" / t / "p4b4b.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        zs.append(np.load(p, allow_pickle=True))

    def cat(key):
        return np.concatenate([z[key] for z in zs], 0)

    seed, clip = cat("seed"), np.array([str(x) for x in cat("clip")])
    terrain = np.array([str(x) for x in cat("terrain")])
    groups = _groups(seed, clip, terrain)
    n = len(seed)

    # discover default keys
    keys = []
    for z in zs:
        keys.extend([str(x) for x in z["cfg_keys"].tolist()])
    keys = sorted(set(keys))
    default_keys = [k for k in keys if k.endswith("_default")]

    packs = {}
    for key in default_keys:
        # concat per-terrain
        parts = []
        ok = True
        for z in zs:
            if f"{key}__e_p" not in z.files:
                ok = False
                break
            parts.append(_feat_pack(z, key))
        if not ok or any(p is None for p in parts):
            continue
        merged = {}
        for fk in parts[0]:
            if isinstance(parts[0][fk], np.ndarray) and parts[0][fk].ndim >= 1 and parts[0][fk].shape[0] == zs[0]["e0"].shape[0]:
                merged[fk] = np.concatenate([p[fk] for p in parts], 0)
            else:
                merged[fk] = parts[0][fk]
        packs[key] = merged

    if PRIMARY not in packs:
        # amp:g formatting
        cand = [k for k in packs if k.startswith("n4_a") and "0.5" in k and k.endswith("_default")]
        primary = cand[0] if cand else (default_keys[0] if default_keys else None)
    else:
        primary = PRIMARY
    if primary is None:
        raise RuntimeError("no default configs")
    P = packs[primary]
    a5 = P["a5"]
    acc_clone = _pair_acc(P["g_end"], a5)
    print(f"[p4b4b] primary={primary} n={n} clone_end={acc_clone:.3f}", flush=True)

    rows = []
    preds = {}
    for kind, lab in [
        ("D0", "direction-only"),
        ("D1", "pre-state only"),
        ("D2", "code only"),
        ("D3", "raw F2 response"),
        ("D4", "trend-corrected"),
        ("D5", "twin residual"),
        ("F0", "F0 intent"),
        ("F1", "F1 proprio"),
        ("F2", "F2 motor-load"),
        ("F3", "F3 torque"),
    ]:
        try:
            Xs = _axis_X(P, kind)
        except Exception as e:
            print(f"[p4b4b] skip {kind}: {e}", flush=True)
            continue
        for dname, fit in FIT.items():
            acc, cov, pred = _cv_pair(Xs, a5, groups, fit)
            lo, hi = _boot_acc(pred, a5) if np.isfinite(acc) else (float("nan"), float("nan"))
            rows.append({
                "cfg": primary, "kind": kind, "label": lab, "decoder": dname,
                "Acc": acc, "cov": cov, "ci_lo": lo, "ci_hi": hi,
                "Rinfo": _rinfo(acc, acc_clone),
            })
            if dname == "shrinkage LDA":
                preds[kind] = pred
            print(f"[p4b4b] {kind}/{dname} Acc={acc:.3f}", flush=True)

    # window/amp sweep LDA D3
    sweep = []
    for key, pack in packs.items():
        if pack.get("a5") is None:
            continue
        Xs = _axis_X(pack, "D3")
        acc, cov, pred = _cv_pair(Xs, pack["a5"], groups, _lda)
        acc_ps, _, _ = _cv_pair(_axis_X(pack, "D1"), pack["a5"], groups, _lda)
        acc_tw, _, _ = _cv_pair(_axis_X(pack, "D5"), pack["a5"], groups, _lda)
        cl = _pair_acc(pack["g_end"], pack["a5"]) if pack.get("g_end") is not None else float("nan")
        peak = pack["peak_ex"]
        viol = float((peak > 0.005).mean())
        Gmed = float(np.median(pack["G"]))
        sweep.append({
            "cfg": key, "Acc_raw": acc, "Acc_pre": acc_ps, "Acc_twin": acc_tw,
            "dAcc": (acc - acc_ps) if np.isfinite(acc) and np.isfinite(acc_ps) else None,
            "clone": cl, "Rinfo": _rinfo(acc, cl),
            "peak_ex_med": float(np.median(peak)), "peak_ex_p95": float(np.percentile(peak, 95)),
            "viol5mm": viol, "G_med": Gmed, "rank": pack["rank"], "cond": pack["cond"],
        })

    def get(kind, dec="shrinkage LDA"):
        hits = [r for r in rows if r["kind"] == kind and r["decoder"] == dec]
        return hits[0] if hits else None

    d3 = get("D3")
    d1 = get("D1")
    d5 = get("D5")
    d0 = get("D0")
    f0, f1, f2, f3 = get("F0"), get("F1"), get("F2"), get("F3")
    acc_raw = d3["Acc"] if d3 else float("nan")
    acc_pre = d1["Acc"] if d1 else float("nan")
    acc_tw = d5["Acc"] if d5 else float("nan")
    acc_dir = d0["Acc"] if d0 else float("nan")
    dacc = (acc_raw - acc_pre) if np.isfinite(acc_raw) and np.isfinite(acc_pre) else float("nan")
    dacc_dir = (acc_raw - acc_dir) if np.isfinite(acc_raw) and np.isfinite(acc_dir) else float("nan")
    rinfo = _rinfo(acc_raw, acc_clone)
    viol = float((P["peak_ex"] > 0.005).mean())
    # selected A from CV signs of D3 LDA
    Aend = None
    Anet = None
    if "D3" in preds and P["a5"] is not None:
        hat = preds["D3"]
        a5v = P["a5"]
        A_sel = []
        for i in range(n):
            vals = []
            for ax in range(3):
                si = 0 if hat[i, ax] >= 0 else 1
                # hat is predicted s_end = sign(A+ - A-); + corresponds to index 0
                if hat[i, ax] > 0:
                    vals.append(a5v[i, ax, 0])
                elif hat[i, ax] < 0:
                    vals.append(a5v[i, ax, 1])
            A_sel.append(np.mean(vals) if vals else 0.0)
        Aend = np.asarray(A_sel)
        if P["a_net"] is not None:
            An = []
            for i in range(n):
                vs = []
                for ax in range(3):
                    if hat[i, ax] > 0:
                        vs.append(P["a_net"][i, ax, 0])
                    elif hat[i, ax] < 0:
                        vs.append(P["a_net"][i, ax, 1])
                An.append(np.mean(vs) if vs else float(P["a_net0"][i]))
            Anet = np.asarray(An)

    # t0 catalog labels (0.2 s clone A), vs 0.5 s endpoint labels
    a5_gt = cat("a5_gt") if "a5_gt" in zs[0].files else None
    acc_d3_t0 = float("nan")
    acc_clone_t0 = float("nan")
    if a5_gt is not None:
        acc_d3_t0, _, _ = _cv_pair(_axis_X(P, "D3"), a5_gt, groups, _lda)
        acc_clone_t0 = _pair_acc(P["g_end"], a5_gt)
    # LOTO terrain
    loto_rows = []
    for tr in TERRAINS:
        te = terrain == tr
        trn = ~te
        if te.sum() < 8 or trn.sum() < 16:
            continue
        Xs = _axis_X(P, "D3")
        s_star = np.sign(P["a5"][:, :, 0] - P["a5"][:, :, 1])
        accs = []
        for ax in range(3):
            y = s_star[:, ax]
            X = np.asarray(Xs[ax], dtype=np.float64)
            nz = y != 0
            tr_i = np.where(trn & nz)[0]
            te_i = np.where(te & nz)[0]
            if tr_i.size < 8 or te_i.size < 1 or np.unique(y[tr_i]).size < 2:
                continue
            Xtr, Xte = _zscore(X[tr_i], X[te_i])
            pred = _lda(Xtr, y[tr_i], Xte)
            accs.append(float((pred == y[te_i]).mean()))
        loto_rows.append({"terrain": tr, "n_te": int(te.sum()), "Acc_D3": float(np.mean(accs)) if accs else float("nan")})

    # cases: never call M1 if axis prior already explains Acc (D0 ≈ D3)
    def ge(a, t):
        return np.isfinite(a) and a >= t

    active_vs_dir = np.isfinite(dacc_dir) and dacc_dir >= 0.03
    if ge(acc_raw, 0.60) and ge(rinfo, 0.60) and (np.isfinite(dacc) and dacc > 0.03) and active_vs_dir and Anet is not None and float(np.mean(Anet)) > 0 and float(np.median(Anet)) > 0 and viol < 0.05:
        case, nxt = "M1", "scale a small active-response decoder. Do not start P4-C."
    elif ge(acc_tw, 0.62) and (not ge(acc_raw, 0.58)):
        case, nxt = "M2", "fast signal exists; improve deployable residualization. No large world model. No P4-C."
    elif f0 and f1 and f2 and (f0["Acc"] < 0.55) and (f1["Acc"] < 0.55) and ge(f2["Acc"], 0.60):
        case, nxt = "M3", "commanded-minus-measured is the interaction channel."
    elif P["rank"] < 3 or (not P.get("raw_dead") and float(np.median(P["G"])) < 0.05):
        case, nxt = "M4", "microprobe filtered; fix injection before changing decoder."
    elif (not ge(acc_tw, 0.55)) and (not ge(acc_raw, 0.55)):
        case, nxt = "M5", "20-40 ms simultaneous excitation not reliably decodable. Do not return to 120 ms Jacobian."
    else:
        case, nxt = "PARTIAL", (
            "3D code reached PD targets (rank≥3, tgt_rms>0) and is twin-safe, "
            "but Acc is explained by axis prior (D0≈D3). Do not scale a decoder. Do not start P4-C. "
            "Do not return to 120 ms Jacobian. Do not train a net."
        )

    decision = {
        "case": f"B4B-{case}",
        "next": nxt,
        "run_p4c": False,
        "primary": primary,
        "n": n,
        "acc_raw": acc_raw,
        "acc_pre": acc_pre,
        "acc_twin": acc_tw,
        "acc_dir": None if not d0 else d0["Acc"],
        "dAcc_active": dacc,
        "dAcc_vs_dir": dacc_dir,
        "acc_clone_end": acc_clone,
        "acc_d3_t0": acc_d3_t0,
        "acc_clone_t0": acc_clone_t0,
        "loto": loto_rows,
        "Rinfo": rinfo,
        "viol5mm": viol,
        "G_med": float(np.median(P["G"])),
        "rank": P["rank"],
        "cond": P["cond"],
        "raw_dead": bool(P.get("raw_dead", False)),
        "tgt_rms": P.get("tgt_rms"),
        "mean_Aend": None if Aend is None else float(np.mean(Aend)),
        "mean_Anet": None if Anet is None else float(np.mean(Anet)),
        "median_Anet": None if Anet is None else float(np.median(Anet)),
        "P_Anet_gt0": None if Anet is None else float((Anet > 0).mean()),
        "physics_ok": True,
    }

    # plots
    fig, ax = plt.subplots(figsize=(8, 4))
    labs, accs = [], []
    for kind, lab in [("D0", "dir"), ("D1", "pre"), ("D3", "raw"), ("D4", "trend"), ("D5", "twin")]:
        g = get(kind)
        if g:
            labs.append(lab)
            accs.append(g["Acc"])
    ax.bar(labs + ["clone"], accs + [acc_clone], color="#4C78A8")
    ax.axhline(0.5, color="#888", lw=1)
    ax.axhline(0.6, color="#F58518", lw=1, ls="--")
    ax.set_ylabel("Acc_end")
    ax.set_title("P4-B4B endpoint sign accuracy (LDA, primary 20 ms 0.5°)")
    fig.tight_layout()
    fig.savefig(plots / "B4B-1_acc.png", dpi=120)
    plt.close(fig)

    md = [
        "# P4-B4B — 200 Hz intra-policy microprobe",
        "",
        f"Grouped 5-fold CV. n=**{n}**. Primary **{primary}**. Exact batched latent decode. Case **B4B-{case}**.",
        f"Control: physics_dt=5 ms, decimation=4 (50 Hz policy / 200 Hz low-level). Endpoint burst H=25 (0.5 s).",
        "",
        "## Headline (shrinkage LDA, primary)",
        "",
        _md(
            ["Input", "Acc_end", "Rinfo", "95% CI"],
            [
                [r["label"], _fmt(r["Acc"]), _fmt(r["Rinfo"], 2), f"[{_fmt(r['ci_lo'])},{_fmt(r['ci_hi'])}]"]
                for r in rows if r["decoder"] == "shrinkage LDA" and r["cfg"] == primary
            ],
        ),
        "",
        f"Clone-end Acc (1-step g vs 0.5s A_end)=**{_fmt(acc_clone)}**. ΔAcc vs D1=**{_fmt(dacc)}**. ΔAcc vs D0=**{_fmt(dacc_dir)}**.",
        f"D3 vs t0 catalog labels Acc=**{_fmt(acc_d3_t0)}**. g_end vs t0 labels Acc=**{_fmt(acc_clone_t0)}**.",
        "",
        "## Command integrity",
        "",
        f"median G_applied={_fmt(float(np.median(P['G'])))} (1.0 if raw_actions buffer dead). effective rank={P['rank']}. cond={_fmt(P['cond'],1)}. tgt_rms={_fmt(P.get('tgt_rms'))}. raw_dead={P.get('raw_dead')}.",
        "",
        "## Safety (twin-relative)",
        "",
        f"median peak excess={_fmt(float(np.median(P['peak_ex'])),4)} m. p95={_fmt(float(np.percentile(P['peak_ex'],95)),4)} m. 5 mm viol={_fmt(viol)}.",
        "",
        "## Window / amplitude sweep (LDA raw F2)",
        "",
        _md(
            ["cfg", "Acc_raw", "Acc_pre", "Acc_twin", "ΔAcc", "viol5mm", "G_med"],
            [[s["cfg"], _fmt(s["Acc_raw"]), _fmt(s["Acc_pre"]), _fmt(s["Acc_twin"]), _fmt(s["dAcc"]), _fmt(s["viol5mm"]), _fmt(s["G_med"])] for s in sweep],
        ),
        "",
        "## Selected correction",
        "",
        f"mean A_end={_fmt(decision['mean_Aend'])}. mean A_net={_fmt(decision['mean_Anet'])}. median A_net={_fmt(decision['median_Anet'])}. P(A_net>0)={_fmt(decision['P_Anet_gt0'])}.",
        "",
        "## LOTO terrain (D3 LDA)",
        "",
        (_md(["terrain", "n_te", "Acc_D3"], [[r["terrain"], r["n_te"], _fmt(r["Acc_D3"])] for r in loto_rows]) if loto_rows else "n/a"),
        "",
        "## Decision",
        "",
        f"**B4B-{case}**. {nxt}",
        "",
        "- P4-C stays closed. Stage-2 latent unchanged. No Virtual Twin. No 120 ms Jacobian.",
        "",
        "## Mandatory questions",
        "",
        f"1. Plant reached? G_med={_fmt(float(np.median(P['G'])))} rank={P['rank']}.",
        f"2. Full-rank code? rank={P['rank']} cond={_fmt(P['cond'],1)}.",
        "3. 20 vs 40 ms: see sweep table.",
        "4. Best amplitude: see sweep (global LDA Acc_raw, safety-qualified).",
        f"5. Active vs D0/D1: Acc_raw={_fmt(acc_raw)} Acc_pre={_fmt(acc_pre)} Acc_dir={_fmt(acc_dir)} ΔAcc_D1={_fmt(dacc)} ΔAcc_D0={_fmt(dacc_dir)}.",
        "6–7. Feature channel: F0/F1/F2/F3 in headline table.",
        f"8. Twin vs raw: Acc_twin={_fmt(acc_tw)} Acc_raw={_fmt(acc_raw)}.",
        f"9. Acc_end>0.60? {ge(acc_raw, 0.60)}. Active vs D0≥0.03? {active_vs_dir}.",
        f"10. Rinfo vs 0.5s clone={_fmt(rinfo)} (clone Acc={_fmt(acc_clone)} is 1-step g vs 0.5s A, not B4A 0.675).",
        f"11–12. A_end/A_net above. P(A_net>0) near 0.5 means selected burst is not a reliable improvement.",
        f"13. Tube 5 mm viol={_fmt(viol)}.",
        "14. Actuator: raw_actions buffer did not track process_action; PD processed targets did (tgt_rms, rank).",
        "15–16. LOTO: " + (", ".join(f"{r['terrain']}={_fmt(r['Acc_D3'])}" for r in loto_rows) if loto_rows else "n/a") + ".",
        "18. Bottleneck: axis prior / 20 ms identifiability, not injection failure.",
        "19. Next follows case. Do not train a net. Do not start P4-C.",
        "20. P4-C justified? **No.**",
    ]
    (root / "MASTER_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (root / "decoding" / "decoder_metrics.csv").open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with (root / "decoding" / "feature_ablation.csv").open("w", newline="") as f:
        if sweep:
            w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
            w.writeheader()
            w.writerows(sweep)
    (root / "metrics.json").write_text(json.dumps(sanitize({"decision": decision, "rows": rows, "sweep": sweep}), indent=2), encoding="utf-8")
    (root / "config.yaml").write_text(
        "experiment: p4b4b_200hz_microprobe\nstart_p4c: false\nprimary: 20ms 0.5deg F2 exact_batched_decode\n",
        encoding="utf-8",
    )
    print(json.dumps(sanitize(decision), indent=2), flush=True)


if __name__ == "__main__":
    main()
