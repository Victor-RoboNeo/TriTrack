#!/usr/bin/env python3
"""P4-B4A: grouped-CV decode of probe response vs t0 / t_N / net labels. No nets."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from p3_common import sanitize

TERRAINS = ("steps", "slip", "slope_down")
P4B2_DEFAULT = "/data/home/chenxiangyu/robotics/Anybody/results/p4b2_twin_referenced_id"
ROOT_DEFAULT = "/data/home/chenxiangyu/robotics/Anybody/results/p4b4_fast_active_response"
NS = (6, 8, 10)
ACC_CLONE_T0 = 0.658


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{float(v):.{nd}f}"


def _rinfo(acc, acc_raw):
    if acc is None or not np.isfinite(acc) or abs(float(acc_raw) - 0.5) < 1e-6:
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


def _unit_rows(x):
    nrm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(nrm, 1e-12, None)


def _alpha(a5):
    return a5[:, :, 0] - a5[:, :, 1]


def _group_ids(seed, clip, terrain):
    return np.array([f"{tr}|{int(s)}|{c}" for tr, s, c in zip(terrain, seed, clip)])


def _group_folds(groups, n_splits=5, seed=2026):
    uniq = np.unique(groups)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(uniq))
    assign = {uniq[i]: (k % n_splits) for k, i in enumerate(order)}
    return np.array([assign[g] for g in groups], dtype=np.int32)


def _zscore(Xtr, Xte):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _lda_shrink(Xtr, ytr, Xte, shrink=0.1):
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    ytr = np.asarray(ytr, dtype=np.float64)
    classes = np.unique(ytr)
    if classes.size < 2:
        return np.zeros(len(Xte))
    p = Xtr.shape[1]
    Sw = np.zeros((p, p))
    mu = []
    for c in classes:
        Xc = Xtr[ytr == c]
        m = Xc.mean(axis=0)
        mu.append(m)
        d = Xc - m
        Sw = Sw + d.T @ d
    tr = float(np.trace(Sw)) / max(p, 1)
    Sw = (1.0 - float(shrink)) * Sw + float(shrink) * tr * np.eye(p) + 1e-8 * np.eye(p)
    w = np.linalg.solve(Sw, mu[1] - mu[0])
    thr = 0.5 * (mu[0] + mu[1]) @ w
    return np.where(Xte @ w >= thr, classes[1], classes[0])


def _logreg_l2(Xtr, ytr, Xte, l2=1.0, n_iter=120):
    """y in {-1,+1}. L2 logistic with intercept. GD."""
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    y01 = (np.asarray(ytr, dtype=np.float64) > 0).astype(np.float64)
    n, d = Xtr.shape
    w = np.zeros(d)
    b = 0.0
    lr = 1.0 / max(n, 1)
    for _ in range(int(n_iter)):
        z = np.clip(Xtr @ w + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y01
        w = w - lr * (Xtr.T @ err + float(l2) * w)
        b = b - lr * float(err.sum())
    zte = Xte @ w + b
    pred = np.where(zte >= 0.0, 1.0, -1.0)
    return pred


def _linsvm(Xtr, ytr, Xte, C=1.0, n_epoch=50, seed=0):
    """Primal L2-SVM, y in {-1,+1}."""
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    y = np.asarray(ytr, dtype=np.float64)
    n, d = Xtr.shape
    w = np.zeros(d)
    b = 0.0
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    for ep in range(int(n_epoch)):
        rng.shuffle(idx)
        lr = 0.1 / (1.0 + 0.05 * ep)
        for i in idx:
            xi = Xtr[i]
            yi = y[i]
            marg = yi * (float(xi @ w) + b)
            w = w - lr * (w / max(n, 1))
            if marg < 1.0:
                w = w + lr * float(C) * yi * xi
                b = b + lr * float(C) * yi
    return np.where(Xte @ w + b >= 0.0, 1.0, -1.0)


FITTERS = {
    "shrinkage LDA": lambda Xtr, y, Xte: _lda_shrink(Xtr, y, Xte, 0.1),
    "L2 logistic": lambda Xtr, y, Xte: _logreg_l2(Xtr, y, Xte, 1.0),
    "linear SVM": lambda Xtr, y, Xte: _linsvm(Xtr, y, Xte, 1.0),
}


def _cv_pair(X, a5, groups, fit_fn, n_splits=5):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    s_star = np.sign(a5[:, :, 0] - a5[:, :, 1])
    pred = np.zeros_like(s_star)
    filled = np.zeros_like(s_star, dtype=bool)
    fold = _group_folds(groups, n_splits=n_splits)
    for k in range(s_star.shape[1]):
        y = s_star[:, k]
        nz = y != 0
        for f in range(n_splits):
            te = np.where(fold == f)[0]
            tr = np.where(fold != f)[0]
            tr = tr[nz[tr]]
            te_use = te[nz[te]]
            if tr.size < 8 or te_use.size < 1 or np.unique(y[tr]).size < 2:
                continue
            Xtr, Xte = _zscore(X[tr], X[te_use])
            hat = fit_fn(Xtr, y[tr], Xte)
            pred[te_use, k] = hat
            filled[te_use, k] = True
    valid = (s_star != 0) & filled
    if not np.any(valid):
        return float("nan"), 0.0
    return float((pred[valid] == s_star[valid]).mean()), float(filled.mean())


def _stale(a5_t0, a5_tn):
    v0 = _unit_rows(_alpha(a5_t0))
    vn = _unit_rows(_alpha(a5_tn))
    c = np.sum(v0 * vn, axis=-1)
    s0 = np.sign(_alpha(a5_t0))
    sn = np.sign(_alpha(a5_tn))
    nz = (s0 != 0) & (sn != 0)
    agree = float((s0[nz] == sn[nz]).mean()) if np.any(nz) else float("nan")
    ax0 = np.argmax(np.abs(_alpha(a5_t0)), axis=1)
    axn = np.argmax(np.abs(_alpha(a5_tn)), axis=1)
    return {
        "cos_median": float(np.median(c)),
        "cos_mean": float(np.mean(c)),
        "angle_med_deg": float(np.median(np.degrees(np.arccos(np.clip(c, -1, 1))))),
        "sign_agree": agree,
        "argmax_axis_agree": float((ax0 == axn).mean()),
        "frac_sign_flip": float(1.0 - agree) if np.isfinite(agree) else float("nan"),
    }


def _features(Ep, Et, U):
    R = Ep - Et
    raw_pre = np.concatenate([Ep, Et[:, : min(3, Et.shape[1])]], axis=1)
    return {
        "Twin residual sequence": R,
        "Raw response sequence": Ep,
        "Raw + twin-prefix 3": raw_pre,
        "U^T r": R @ np.asarray(U, dtype=np.float64),
    }


def _md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4b2", default=P4B2_DEFAULT)
    ap.add_argument("--root", default=ROOT_DEFAULT)
    args = ap.parse_args()
    root = Path(args.root)
    p4b2 = Path(args.p4b2)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    b2, b4 = {}, {}
    for t in TERRAINS:
        p2 = p4b2 / "p4b2" / "loco" / t / "p4b2.npz"
        p4 = root / "p4b4a" / "loco" / t / "p4b4a.npz"
        if not p2.exists():
            raise FileNotFoundError(p2)
        if not p4.exists():
            raise FileNotFoundError(p4)
        b2[t] = np.load(p2, allow_pickle=True)
        b4[t] = np.load(p4, allow_pickle=True)

    def cat2(key):
        return np.concatenate([b2[t][key] for t in TERRAINS], axis=0)

    def cat4(key):
        return np.concatenate([b4[t][key] for t in TERRAINS], axis=0)

    seed = cat2("seed")
    clip = np.asarray([str(c) for c in cat2("clip")])
    terrain = np.asarray([str(c) for c in cat2("terrain")])
    # IDs must match p4b4a order
    seed4 = cat4("seed")
    t2 = cat2("t")
    t4 = cat4("t")
    if seed.shape != seed4.shape or not np.array_equal(seed, seed4) or not np.array_equal(t2, t4):
        raise RuntimeError("P4-B2 and P4-B4A snapshot IDs do not align")
    groups = _group_ids(seed, clip, terrain)
    a5_t0 = cat4("a5_t0").astype(np.float64)
    n = int(a5_t0.shape[0])
    U = {int(N): np.asarray(b2["steps"][f"U_{N}"], dtype=np.float64) for N in NS}

    rows = []
    stale = {}
    clone_end = {}
    net_zero = {}
    for N in NS:
        Ep = cat2(f"E_probe_{N}").astype(np.float64)
        Et = cat2(f"E_twin_{N}").astype(np.float64)
        a5_tn = cat4(f"a5_end_{N}").astype(np.float64)
        a_net = cat4(f"a_net_{N}").astype(np.float64)
        a_net0 = cat4(f"a_net0_{N}").astype(np.float64)
        g_end = cat4(f"g_end_{N}").astype(np.float64)
        stale[N] = _stale(a5_t0, a5_tn)
        clone_end[N] = {
            "Acc": _pair_acc(g_end, a5_tn),
            "mean_A": float(np.mean(np.max(_alpha(a5_tn), axis=1))),
        }
        a_signed_max = np.max(np.maximum(a_net[:, :, 0], a_net[:, :, 1]), axis=1)
        net_zero[N] = float((a_net0 >= a_signed_max).mean())
        feats = _features(Ep, Et, U[N])
        labels = {
            "t0": a5_t0,
            "tN": a5_tn,
            "net": a_net,
        }
        for fname, X in feats.items():
            for dname, fit in FITTERS.items():
                accs = {}
                for lname, a5 in labels.items():
                    acc, cov = _cv_pair(X, a5, groups, fit)
                    accs[lname] = acc
                    rows.append({
                        "input": fname,
                        "decoder": dname,
                        "N": int(N),
                        "label": lname,
                        "Acc": acc,
                        "coverage": cov,
                        "Rinfo_t0clone": _rinfo(acc, ACC_CLONE_T0),
                        "Rinfo_endclone": _rinfo(acc, clone_end[N]["Acc"]),
                    })
                print(
                    f"[p4b4a] N={N} {fname} / {dname}: "
                    f"t0={_fmt(accs['t0'])} tN={_fmt(accs['tN'])} net={_fmt(accs['net'])}",
                    flush=True,
                )

    def pick(inp, N, lab, dec="shrinkage LDA"):
        hits = [r for r in rows if r["input"] == inp and r["N"] == N and r["label"] == lab and r["decoder"] == dec]
        return hits[0]["Acc"] if hits else float("nan")

    acc_res_t0 = pick("Twin residual sequence", 6, "t0")
    acc_res_tn = pick("Twin residual sequence", 6, "tN")
    acc_raw_tn = pick("Raw response sequence", 6, "tN")
    acc_utr_t0 = pick("U^T r", 6, "t0")

    def ge(a, thr):
        return np.isfinite(a) and a >= thr

    if ge(acc_res_tn, 0.62) and ge(acc_raw_tn, 0.58):
        case, nxt = "A", "train response-to-endpoint-correction decoder. Do not return to Jacobian. Do not start P4-C."
    elif (not np.isfinite(acc_res_tn)) or acc_res_tn <= 0.55:
        case, nxt = "B", "P4-B4B 200 Hz intra-policy microprobe. Do not train decoder on 50 Hz residual yet. Do not start P4-C."
    else:
        case, nxt = "PARTIAL", "report; do not train a large net; P4-B4B still allowed if raw end Acc < 0.58. No P4-C."

    decision = {
        "case": f"B4A-{case}",
        "next": nxt,
        "run_p4c": False,
        "run_virtual_twin": False,
        "run_p4b4b": case == "B",
        "acc_residual_t0": acc_res_t0,
        "acc_residual_tN": acc_res_tn,
        "acc_raw_tN": acc_raw_tn,
        "acc_UTr_t0": acc_utr_t0,
        "acc_clone_end_N6": clone_end[6]["Acc"],
        "stale_N6": stale[6],
        "P_net_zero_N6": net_zero[6],
        "n": n,
        "n_groups": int(len(np.unique(groups))),
        "horizon_A": 10,
        "keep_stage2_latent": True,
        "constant_J_closed": True,
    }

    headline_inputs = (
        "Twin residual sequence",
        "Raw response sequence",
        "U^T r",
        "Raw + twin-prefix 3",
    )
    head_rows = []
    for inp in headline_inputs:
        head_rows.append([
            inp,
            6,
            _fmt(pick(inp, 6, "t0")),
            _fmt(pick(inp, 6, "tN")),
            _fmt(pick(inp, 6, "net")),
        ])

    md = []
    md.append("# P4-B4A — Post-probe endpoint re-labeling")
    md.append("")
    md.append("Grouped 5-fold CV by (terrain, seed, clip). Shrinkage LDA primary. No neural nets. No P4-C.")
    md.append(f"n=**{n}**. Clone burst H=10 (0.2 s), same as frozen t0 `a5_gt`. Case **B4A-{case}**.")
    md.append("")
    md.append("## Headline (shrinkage LDA, N=6)")
    md.append("")
    md.append(_md_table(["Input", "N", "label t0", "label t_N", "net label"], head_rows))
    md.append("")
    md.append("## Label staleness")
    md.append("")
    md.append(_md_table(
        ["N", "median cos(d_t0*, d_tN*)", "median angle", "sign agree", "argmax-axis agree", "P(d_net*=0)"],
        [
            [
                N,
                _fmt(stale[N]["cos_median"]),
                f"{stale[N]['angle_med_deg']:.1f}°",
                _fmt(stale[N]["sign_agree"]),
                _fmt(stale[N]["argmax_axis_agree"]),
                _fmt(net_zero[N]),
            ]
            for N in NS
        ],
    ))
    md.append("")
    md.append(
        f"Clone 1-step Acc at probe-end vs `a5_end` (N=6): **{_fmt(clone_end[6]['Acc'])}**. "
        f"t0 clone Acc remains {ACC_CLONE_T0:.3f}."
    )
    md.append("")
    md.append("## All decoders")
    md.append("")
    md.append(_md_table(
        ["Input", "Decoder", "N", "label", "Acc", "Rinfo vs t0-clone"],
        [
            [
                r["input"],
                r["decoder"],
                r["N"],
                r["label"],
                _fmt(r["Acc"]),
                _fmt(r["Rinfo_t0clone"], 2),
            ]
            for r in rows
            if r["N"] == 6
        ],
    ))
    md.append("")
    md.append("## Decision")
    md.append("")
    md.append(f"**B4A-{case}**. {nxt}")
    md.append("")
    md.append("- Constant-J 50 Hz ID remains closed (P4-B3).")
    md.append("- Do not train Virtual Twin.")
    md.append("- Do not change Stage-2 latent.")
    md.append("- P4-C stays closed.")
    md.append("")
    (root / "MASTER_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (root / "decoder_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (root / "metrics.json").write_text(json.dumps(sanitize({
        "decision": decision,
        "stale": stale,
        "clone_end": clone_end,
        "net_zero": net_zero,
        "rows": rows,
    }), indent=2), encoding="utf-8")
    print(json.dumps(sanitize(decision), indent=2), flush=True)
    print(f"[p4b4a] wrote {root / 'MASTER_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
