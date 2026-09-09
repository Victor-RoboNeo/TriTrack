#!/usr/bin/env python3
"""P4-B3 Oracle-Twin Residual + Lag/FIR + LDA. Offline on P4-B2 dumps. No P4-C."""
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

TERRAINS = ("steps", "slip", "slope_down")
P4B2_DEFAULT = "/data/home/chenxiangyu/robotics/Anybody/results/p4b2_twin_referenced_id"
ROOT_DEFAULT = "/data/home/chenxiangyu/robotics/Anybody/results/p4b3_oracle_residual_id"
ACC_CLONE = 0.658
MEAN_A_CLONE = 0.020
LAMBDAS = (1e-4, 1e-3, 1e-2)
LAGS = (0, 1, 2, 3)
FIRS = (0, 1, 2, 3)
NS = (6, 8, 10)


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{float(v):.{nd}f}"


def _rinfo(acc, acc_raw=ACC_CLONE):
    if acc is None or not np.isfinite(acc) or abs(acc_raw - 0.5) < 1e-6:
        return None
    return (float(acc) - 0.5) / (float(acc_raw) - 0.5)


def _pair_acc(g, a5, mask=None):
    s_star = np.sign(a5[:, :, 0] - a5[:, :, 1])
    s_hat = np.sign(-np.asarray(g))
    valid = s_star != 0
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool).reshape(-1, 1)
    if not np.any(valid):
        return float("nan")
    return float((s_hat[valid] == s_star[valid]).mean())


def _axis_A(g, a5):
    g = np.asarray(g, dtype=np.float64)
    a5 = np.asarray(a5, dtype=np.float64)
    n = g.shape[0]
    out = np.zeros(n)
    for i in range(n):
        j = int(np.argmax(np.abs(g[i])))
        si = 0 if (-g[i, j]) >= 0 else 1
        out[i] = a5[i, j, si]
    return out


def _ridge_J(R, U, lam):
    """R (T,dy), U (T,k) → J (dy,k) with R ≈ U J^T."""
    R = np.asarray(R, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    if R.ndim == 1:
        R = R.reshape(-1, 1)
    T, k = U.shape
    dy = R.shape[1]
    if T < 1:
        return np.zeros((dy, k))
    gram = U.T @ U + float(lam) * np.eye(k)
    Jt = np.linalg.solve(gram, U.T @ R)
    return Jt.T


def _g_from_J(J):
    g = np.asarray(J[0], dtype=np.float64)
    if not np.isfinite(g).all() or float(np.linalg.norm(g)) < 1e-10:
        return g, False
    return g, True


def _align_lag(R, U, lag):
    R = np.asarray(R, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    if lag <= 0:
        return R, U
    if R.shape[0] <= lag:
        return R[:0], U[:0]
    return R[lag:], U[:-lag]


def _fir_XU(R, U, L):
    """r_t = sum_{ℓ=0}^L J_ℓ u_{t-ℓ}. Returns X (T, k*(L+1)), R_use (T, dy)."""
    R = np.asarray(R, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    if R.ndim == 1:
        R = R.reshape(-1, 1)
    N, k = U.shape
    if N <= L:
        return R[:0], U[:0]
    rows, ys = [], []
    for t in range(L, N):
        rows.append(np.concatenate([U[t - ell] for ell in range(L + 1)], axis=0))
        ys.append(R[t])
    return np.stack(ys, axis=0), np.stack(rows, axis=0)


def _fir_g(Jbig, k, L):
    """Sum impulse taps on channel 0. Jbig (dy, k*(L+1))."""
    g = np.zeros(k, dtype=np.float64)
    row = np.asarray(Jbig[0], dtype=np.float64)
    for ell in range(L + 1):
        g = g + row[ell * k : (ell + 1) * k]
    return g


def _lda_2class(Xtr, ytr, Xte):
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    ytr = np.asarray(ytr, dtype=np.float64)
    classes = np.unique(ytr)
    if classes.size < 2:
        return np.zeros(len(Xte))
    mu, Sw = [], np.zeros((Xtr.shape[1], Xtr.shape[1]))
    for c in classes:
        Xc = Xtr[ytr == c]
        m = Xc.mean(axis=0)
        mu.append(m)
        d = Xc - m
        Sw = Sw + d.T @ d
    Sw = Sw + 1e-6 * np.eye(Xtr.shape[1])
    w = np.linalg.solve(Sw, mu[1] - mu[0])
    thr = 0.5 * (mu[0] + mu[1]) @ w
    pred = np.where(Xte @ w >= thr, classes[1], classes[0])
    return pred


def _lda_cv_pair(X, a5, n_splits=5, seed=2026):
    """X (n, f). Binary s* per axis, 5-fold. Returns pair-style Acc."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    s_star = np.sign(a5[:, :, 0] - a5[:, :, 1])
    pred = np.zeros_like(s_star)
    filled = np.zeros_like(s_star, dtype=bool)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = [idx[f::n_splits] for f in range(n_splits)]
    for k in range(s_star.shape[1]):
        y = s_star[:, k]
        nz = y != 0
        for f, te in enumerate(folds):
            tr = np.setdiff1d(idx, te, assume_unique=False)
            tr = tr[nz[tr]]
            te_use = te[nz[te]]
            if tr.size < 8 or te_use.size < 1 or np.unique(y[tr]).size < 2:
                continue
            hat = _lda_2class(X[tr], y[tr], X[te_use])
            pred[te_use, k] = hat
            filled[te_use, k] = True
    valid = (s_star != 0) & filled
    if not np.any(valid):
        return float("nan"), 0.0
    acc = float((pred[valid] == s_star[valid]).mean())
    cov = float(filled.mean())
    return acc, cov


def _val_mask(n, seed=2026):
    rng = np.random.RandomState(seed)
    m = np.zeros(n, dtype=bool)
    m[rng.choice(n, size=max(1, n // 5), replace=False)] = True
    return m


def _fit_states(R_all, U, a5, mode, lag, L, lam):
    """R_all (n,N) scalar residual. Returns g (n,3), valid (n,)."""
    n, N = R_all.shape
    k = U.shape[1]
    g = np.zeros((n, k))
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        R = R_all[i]
        if mode == "ls":
            Ru, Uu = _align_lag(R.reshape(-1, 1), U, lag)
            if Ru.shape[0] < 2:
                continue
            J = _ridge_J(Ru, Uu, lam)
            gi, ok = _g_from_J(J)
        elif mode == "inc":
            dR = np.diff(R).reshape(-1, 1)
            Ru, Uu = _align_lag(dR, U[:-1], lag)
            if Ru.shape[0] < 2:
                continue
            J = _ridge_J(Ru, Uu, lam)
            gi, ok = _g_from_J(J)
        elif mode == "fir":
            Ru, Xu = _fir_XU(R.reshape(-1, 1), U, L)
            if Ru.shape[0] < 2:
                continue
            Jbig = _ridge_J(Ru, Xu, lam)
            gi = _fir_g(Jbig, k, L)
            ok = bool(np.isfinite(gi).all() and float(np.linalg.norm(gi)) > 1e-10)
        else:
            raise ValueError(mode)
        g[i] = gi
        valid[i] = ok
    return g, valid


def _metrics(g, a5, valid, name, N, extra=None):
    acc = _pair_acc(g, a5, mask=valid)
    acc_eff = float(valid.mean()) * (acc if np.isfinite(acc) else 0.5) + (1.0 - float(valid.mean())) * 0.5
    A = _axis_A(g, a5)
    A_v = A[valid] if valid.any() else A
    row = {
        "method": name,
        "N": int(N),
        "coverage": float(valid.mean()),
        "Acc": acc,
        "Acc_effective": acc_eff,
        "Rinfo": _rinfo(acc),
        "mean_A": float(np.mean(A_v)) if A_v.size else None,
        "median_A": float(np.median(A_v)) if A_v.size else None,
        "P_Agt0": float((A_v > 0).mean()) if A_v.size else None,
        "n": int(g.shape[0]),
        "n_valid": int(valid.sum()),
    }
    if extra:
        row.update(extra)
    return row


def _decide(rows):
    def get(name):
        hits = [r for r in rows if r["method"] == name]
        return hits[0] if hits else None

    r0 = get("Twin residual LS N=6 lag0")
    best_lag = max(
        (r for r in rows if r["method"].startswith("Twin residual LS N=6 lag")),
        key=lambda r: r["Acc"] if r["Acc"] is not None and np.isfinite(r["Acc"]) else -1,
        default=None,
    )
    best_fir = max(
        (r for r in rows if "FIR" in r["method"] and "N=6" in r["method"]),
        key=lambda r: r["Acc"] if r["Acc"] is not None and np.isfinite(r["Acc"]) else -1,
        default=None,
    )
    lda = get("LDA residual E N=6")
    acc0 = r0["Acc"] if r0 else 0.5
    acc_lag = best_lag["Acc"] if best_lag else acc0
    acc_fir = best_fir["Acc"] if best_fir else acc0
    lda_acc = lda["Acc"] if lda else float("nan")

    def ge60(a):
        return a is not None and np.isfinite(a) and a >= 0.60

    def le55(a):
        return a is None or not np.isfinite(a) or a <= 0.55

    if ge60(acc0):
        case, nxt, bottleneck = (
            "R3-A",
            "P4-B4 Virtual Stage2 Twin (nominal-drift predictor). Do not start P4-C.",
            "nominal drift estimation",
        )
    elif (not ge60(acc0)) and (ge60(acc_lag) or ge60(acc_fir)):
        case, nxt, bottleneck = (
            "R3-C",
            "lag-aware coded regression / short FIR impulse response. Do not train virtual twin yet.",
            "response latency",
        )
    elif le55(acc0) and le55(acc_lag) and le55(acc_fir):
        if np.isfinite(lda_acc) and lda_acc >= 0.65:
            case, nxt, bottleneck = (
                "R3-B-est",
                "signal is in residual trajectory but LS/FIR miss it; change estimator, not twin.",
                "estimator formulation",
            )
        elif np.isfinite(lda_acc) and lda_acc <= 0.55:
            case, nxt, bottleneck = (
                "R3-B",
                "do not train virtual twin. Locally time-varying J or pulse recursive ID, or abandon this coded ID.",
                "local stationarity / residual trajectory has no clone-label signal",
            )
        else:
            case, nxt, bottleneck = (
                "R3-B",
                "do not train virtual twin. Time-varying J or shorter pulse ID.",
                "nominal drift is not the bottleneck",
            )
    else:
        case, nxt, bottleneck = ("PARTIAL", "stop and report; do not start P4-C or P4-B4", "mixed")
    return {
        "case": case,
        "next": nxt,
        "bottleneck": bottleneck,
        "run_p4c": False,
        "run_p4b4_virtual_twin": case == "R3-A",
        "acc_residual_lag0": acc0,
        "acc_best_lag": None if best_lag is None else best_lag["Acc"],
        "best_lag_method": None if best_lag is None else best_lag["method"],
        "acc_best_fir": None if best_fir is None else best_fir["Acc"],
        "best_fir_method": None if best_fir is None else best_fir["method"],
        "acc_lda": lda_acc,
        "keep_stage2_latent": True,
        "probe_intent_safe_frozen": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4b2", default=P4B2_DEFAULT)
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--jstat", default="", help="optional p4b3j npz root")
    args = ap.parse_args()
    p4b2 = Path(args.p4b2)
    root = Path(args.root)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    by = {}
    for t in TERRAINS:
        p = p4b2 / "p4b2" / "loco" / t / "p4b2.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        by[t] = np.load(p, allow_pickle=True)

    def cat(key):
        return np.concatenate([by[t][key] for t in TERRAINS], axis=0)

    a5 = cat("a5_gt").astype(np.float64)
    n = int(a5.shape[0])
    U = {int(N): cat(f"U_{N}") if f"U_{N}" in by["steps"].files else by["steps"][f"U_{N}"] for N in NS}
    # U is shared; take from steps
    U = {int(N): np.asarray(by["steps"][f"U_{N}"], dtype=np.float64) for N in NS}

    val = _val_mask(n)
    rows = [
        {
            "method": "Clone 1-step",
            "N": 1,
            "lag/FIR": "—",
            "coverage": 1.0,
            "Acc": ACC_CLONE,
            "Acc_effective": ACC_CLONE,
            "Rinfo": 1.0,
            "mean_A": MEAN_A_CLONE,
        },
        {
            "method": "Raw coded E1",
            "N": 6,
            "lag/FIR": 0,
            "coverage": 1.0,
            "Acc": _pair_acc(cat("g_E1_6"), a5),
            "Rinfo": _rinfo(_pair_acc(cat("g_E1_6"), a5)),
            "mean_A": float(np.mean(cat("A_E1_6"))),
        },
        {
            "method": "Raw coded E2",
            "N": 6,
            "lag/FIR": 0,
            "coverage": 1.0,
            "Acc": _pair_acc(cat("g_E2_6"), a5),
            "Rinfo": _rinfo(_pair_acc(cat("g_E2_6"), a5)),
            "mean_A": float(np.mean(cat("A_E2_6"))),
        },
    ]
    rows[1]["Acc_effective"] = rows[1]["Acc"]
    rows[2]["Acc_effective"] = rows[2]["Acc"]

    # pick lambda on val, residual LS N=6 lag0
    R6 = cat("E_probe_6") - cat("E_twin_6")
    acc_lams = {}
    for lam in LAMBDAS:
        g, v = _fit_states(R6, U[6], a5, "ls", 0, 0, lam)
        acc_lams[lam] = _pair_acc(g[val], a5[val], mask=v[val])
    lam_star = max(acc_lams, key=lambda x: -1 if not np.isfinite(acc_lams[x]) else acc_lams[x])
    print(f"[p4b3] val Acc by lambda lag0 N=6: {acc_lams} → λ={lam_star}", flush=True)

    all_g = {}
    for N in NS:
        R = cat(f"E_probe_{N}") - cat(f"E_twin_{N}")
        UU = U[int(N)]
        for lag in LAGS:
            if N - lag < 3:
                continue
            g, v = _fit_states(R, UU, a5, "ls", lag, 0, lam_star)
            name = f"Twin residual LS N={N} lag{lag}"
            row = _metrics(g, a5, v, name, N, {"lag/FIR": f"lag{lag}", "lambda": lam_star, "mode": "ls"})
            rows.append(row)
            all_g[name] = (g, v)
            g2, v2 = _fit_states(R, UU, a5, "inc", lag, 0, lam_star)
            name2 = f"Twin residual inc N={N} lag{lag}"
            rows.append(_metrics(g2, a5, v2, name2, N, {"lag/FIR": f"lag{lag}", "lambda": lam_star, "mode": "inc"}))
        for L in FIRS:
            g, v = _fit_states(R, UU, a5, "fir", 0, L, lam_star)
            name = f"Twin residual FIR N={N} L={L}"
            row = _metrics(g, a5, v, name, N, {"lag/FIR": f"FIR{L}", "lambda": lam_star, "mode": "fir"})
            rows.append(row)
            all_g[name] = (g, v)

    # LDA / matched-filter diagnostics on N=6 residual
    X_e = R6
    acc_lda, cov_lda = _lda_cv_pair(X_e, a5)
    rows.append({
        "method": "LDA residual E N=6",
        "N": 6,
        "lag/FIR": "LDA",
        "coverage": cov_lda,
        "Acc": acc_lda,
        "Acc_effective": acc_lda,
        "Rinfo": _rinfo(acc_lda),
        "mean_A": None,
        "note": "offline CV diagnostic, not a method",
    })
    # matched filter U^T r  (3-D) — almost LS
    feat = np.einsum("nk,tk->nt", R6, U[6]) if False else R6 @ U[6]
    acc_mf, cov_mf = _lda_cv_pair(feat, a5)
    rows.append({
        "method": "LDA U^T r N=6",
        "N": 6,
        "lag/FIR": "LDA-3D",
        "coverage": cov_mf,
        "Acc": acc_mf,
        "Acc_effective": acc_mf,
        "Rinfo": _rinfo(acc_mf),
        "mean_A": None,
        "note": "offline CV on matched-filter features",
    })
    # also raw probe E without residual (control)
    acc_raw, _ = _lda_cv_pair(cat("E_probe_6"), a5)
    rows.append({
        "method": "LDA raw probe E N=6 (no residual)",
        "N": 6,
        "lag/FIR": "LDA",
        "coverage": 1.0,
        "Acc": acc_raw,
        "Acc_effective": acc_raw,
        "Rinfo": _rinfo(acc_raw),
        "mean_A": None,
    })

    jstat = None
    jroot = Path(args.jstat) if args.jstat else root
    jfiles = [jroot / "p4b3j" / "loco" / t / "p4b3j.npz" for t in TERRAINS]
    if all(p.exists() for p in jfiles):
        jz = [np.load(p, allow_pickle=True) for p in jfiles]
        g0 = np.concatenate([z["g_0"] for z in jz], 0)
        g2 = np.concatenate([z["g_2"] for z in jz], 0)
        g4 = np.concatenate([z["g_4"] for z in jz], 0)

        def coss(a, b):
            na = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
            return np.sum(a * b, axis=1) / na

        c02, c04 = coss(g0, g2), coss(g0, g4)
        jstat = {
            "n": int(g0.shape[0]),
            "cos_0_2_median": float(np.median(c02)),
            "cos_0_2_mean": float(np.mean(c02)),
            "cos_0_4_median": float(np.median(c04)),
            "cos_0_4_mean": float(np.mean(c04)),
            "angle_0_2_med_deg": float(np.median(np.degrees(np.arccos(np.clip(c02, -1, 1))))),
            "angle_0_4_med_deg": float(np.median(np.degrees(np.arccos(np.clip(c04, -1, 1))))),
        }

    decision = _decide(rows)
    decision["lambda"] = lam_star
    decision["lambda_val_acc"] = {str(k): acc_lams[k] for k in acc_lams}
    decision["j_stationarity"] = jstat
    decision["y_twin_note"] = (
        "Y_twin not dumped in P4-B2; residual uses e_I = Y[:,0] = E_I. "
        "E1/scalar g is the e_I row, so oracle residual LS ≡ scalar residual LS."
    )

    keys = ["method", "N", "lag/FIR", "coverage", "Acc", "Acc_effective", "Rinfo", "mean_A"]
    with (root / "estimator_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    (root / "metrics.json").write_text(json.dumps(sanitize({"n": n, "rows": rows, "decision": decision}), indent=2), encoding="utf-8")

    # plots
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for mode, col, mk in (("ls", "#1f4e79", "o"), ("inc", "#c44e52", "s")):
        xs, ys = [], []
        for lag in LAGS:
            r = next((x for x in rows if x.get("mode") == mode and x["N"] == 6 and x.get("lag/FIR") == f"lag{lag}"), None)
            if r and r["Acc"] is not None:
                xs.append(lag)
                ys.append(r["Acc"])
        if xs:
            ax.plot(xs, ys, mk + "-", color=col, label=mode)
    ax.axhline(ACC_CLONE, color="0.3", ls=":", label="clone")
    ax.axhline(0.60, color="#2a9d8f", ls="--")
    ax.axhline(0.521, color="0.6", ls=":", label="raw E1")
    ax.set_xlabel("lag (steps)")
    ax.set_ylabel("sign Acc")
    ax.set_title("B3-1 Residual Acc vs lag (N=6)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "B3_1_acc_vs_lag.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for N, col in ((6, "#1f4e79"), (8, "#c44e52"), (10, "#2a9d8f")):
        xs, ys = [], []
        for L in FIRS:
            r = next((x for x in rows if x.get("mode") == "fir" and x["N"] == N and x.get("lag/FIR") == f"FIR{L}"), None)
            if r and r["Acc"] is not None:
                xs.append(L)
                ys.append(r["Acc"])
        if xs:
            ax.plot(xs, ys, "o-", color=col, label=f"N={N}")
    ax.axhline(0.60, color="0.5", ls="--")
    ax.axhline(ACC_CLONE, color="0.3", ls=":")
    ax.set_xlabel("FIR order L")
    ax.set_ylabel("sign Acc")
    ax.set_title("B3-2 Residual FIR Acc")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "B3_2_acc_vs_FIR.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    show = [
        "Clone 1-step",
        "Raw coded E1",
        "Raw coded E2",
        "Twin residual LS N=6 lag0",
        "Twin residual LS N=6 lag1",
        "Twin residual LS N=6 lag2",
        "Twin residual FIR N=6 L=2",
        "Twin residual FIR N=8 L=2",
        "LDA residual E N=6",
    ]
    labs, accs = [], []
    for name in show:
        r = next((x for x in rows if x["method"] == name), None)
        if r:
            labs.append(name.replace("Twin residual ", "").replace(" N=6", ""))
            accs.append(r["Acc"])
    ax.barh(range(len(labs)), accs, color="#1f4e79")
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=8)
    ax.axvline(0.60, color="#2a9d8f", ls="--")
    ax.axvline(ACC_CLONE, color="0.3", ls=":")
    ax.set_xlabel("Acc")
    ax.set_title("B3-3 Headline comparison")
    fig.tight_layout()
    fig.savefig(plots / "B3_3_headline.png", dpi=140)
    plt.close(fig)

    def line(r):
        return (
            f"| {r.get('method')} | {r.get('N')} | {r.get('lag/FIR', '—')} | "
            f"{_fmt(r.get('Acc'))} | {_fmt(r.get('Rinfo'))} | {_fmt(r.get('mean_A'), 4)} |"
        )

    headline = [
        next(r for r in rows if r["method"] == "Clone 1-step"),
        next(r for r in rows if r["method"] == "Raw coded E1"),
        next(r for r in rows if r["method"] == "Raw coded E2"),
        next(r for r in rows if r["method"] == "Twin residual LS N=6 lag0"),
        next(r for r in rows if r["method"] == "Twin residual LS N=6 lag1"),
        next(r for r in rows if r["method"] == "Twin residual LS N=6 lag2"),
        next(r for r in rows if r["method"] == "Twin residual FIR N=6 L=2"),
        next(r for r in rows if r["method"] == "Twin residual FIR N=8 L=2"),
        next(r for r in rows if r["method"] == "LDA residual E N=6"),
    ]
    q_r0 = next(r for r in rows if r["method"] == "Twin residual LS N=6 lag0")
    lines = [
        "# P4-B3 — Oracle-Twin Residualized Online ID",
        "",
        "Offline on P4-B2 dumps. Residual \(r=E_I^{probe}-E_I^{twin}\) (P4-B2 did not store full \(Y^{twin}\); \(g\) uses the \(e_I\) row, so vector-LS ≡ scalar residual).",
        "No new tube-safety proof. 1° coded probe remains intent-safe (P4-B2 frozen).",
        "No P4-C. No Virtual Twin unless Case R3-A.",
        "",
        f"Pooled n=**{n}**. λ*={lam_star:g} (val Acc N=6 lag0). Case **{decision['case']}**.",
        "",
        "## Headline table",
        "",
        "| Estimator | N | lag/FIR | Acc | Rinfo | mean A |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in headline:
        lines.append(line(r))
    lines += [
        "",
        "## Full residual / lag / FIR",
        "",
        "| Estimator | N | lag/FIR | Acc | Rinfo | mean A |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        if str(r["method"]).startswith("Twin") or str(r["method"]).startswith("LDA"):
            lines.append(line(r))
    lines += [
        "",
        "## Signal recoverability (LDA CV)",
        "",
        f"- LDA on residual \(E_{{1:N}}\) N=6: Acc=**{_fmt(next(r for r in rows if r['method']=='LDA residual E N=6')['Acc'])}**",
        f"- LDA on \(U^\\top r\) (3-D): Acc=**{_fmt(next(r for r in rows if r['method']=='LDA U^T r N=6')['Acc'])}**",
        f"- LDA on raw probe E (no residual): Acc=**{_fmt(acc_raw)}**",
        "",
        "## Jacobian stationarity",
        "",
    ]
    if jstat is None:
        lines.append("Not yet. Run `run_p4b3j.sh` (clone 1-step \(g\) at \(t_0,t_0+2,t_0+4\) along Stage-2 twin).")
    else:
        lines.append(
            f"cos(\(g_{{t0}},g_{{+40ms}}\)) median={_fmt(jstat['cos_0_2_median'])} "
            f"(angle {_fmt(jstat['angle_0_2_med_deg'],1)}°); "
            f"cos(\(g_{{t0}},g_{{+80ms}}\)) median={_fmt(jstat['cos_0_4_median'])} "
            f"(angle {_fmt(jstat['angle_0_4_med_deg'],1)}°)."
        )
    lines += [
        "",
        "## Decision",
        "",
        f"**{decision['case']}**. Bottleneck: **{decision['bottleneck']}**.",
        "",
        f"- Residual lag0 Acc={_fmt(decision['acc_residual_lag0'])} (H1 needs ≥0.60).",
        f"- Best lag: {decision['best_lag_method']} Acc={_fmt(decision['acc_best_lag'])}.",
        f"- Best FIR: {decision['best_fir_method']} Acc={_fmt(decision['acc_best_fir'])}.",
        f"- LDA residual Acc={_fmt(decision['acc_lda'])}.",
        "",
        f"Next: {decision['next']}",
        "",
        "Do **not** start P4-C. Do **not** change Stage-2 latent. Probe safety remains frozen.",
    ]
    (root / "MASTER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "p4c_executability").mkdir(parents=True, exist_ok=True)
    (root / "p4c_executability" / "NOT_RUN.md").write_text("NOT RUN — P4-C disabled for P4-B3.\n", encoding="utf-8")
    if decision["case"] != "R3-A":
        (root / "p4b4_virtual_twin").mkdir(parents=True, exist_ok=True)
        (root / "p4b4_virtual_twin" / "NOT_RUN.md").write_text(
            "NOT RUN — Virtual Stage2 Twin only if Case R3-A (residual restores Acc to ~clone).\n",
            encoding="utf-8",
        )
    print(json.dumps(sanitize(decision), indent=2), flush=True)
    for r in headline:
        print(f"{r['method']}: Acc={_fmt(r.get('Acc'))} R={_fmt(r.get('Rinfo'))} A={_fmt(r.get('mean_A'),4)}", flush=True)


if __name__ == "__main__":
    main()
