#!/usr/bin/env python3
"""P4-B2 twin-referenced online ID analysis. No P4-C."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from p3_common import sanitize, stats

TERRAINS = ("steps", "slip", "slope_down")
ROOT_DEFAULT = "/data/home/chenxiangyu/robotics/Anybody/results/p4b2_twin_referenced_id"
ACC_CLONE = 0.658
MEAN_A_CLONE = 0.020
P_AGT0_CLONE = 0.73
ACC_SEQ = 0.517
TUBE = 0.005
DT = 0.02
NS = (6, 8, 10)
ESTS = ("E1", "E2", "scalar")
BOOT_N = 2000
BOOT_SEED = 2026


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{float(v):.{nd}f}"


def _rinfo(acc, acc_raw=ACC_CLONE):
    if acc is None or acc_raw is None or abs(acc_raw - 0.5) < 1e-6:
        return None
    return (float(acc) - 0.5) / (float(acc_raw) - 0.5)


def _pair_acc(g, a5, mask=None):
    s_star = np.sign(a5[:, :, 0] - a5[:, :, 1])
    s_hat = np.sign(-g)
    valid = s_star != 0
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool).reshape(-1, 1)
    if not np.any(valid):
        return float("nan"), 0, 0
    n_ok = int(np.sum(s_hat[valid] == s_star[valid]))
    n_tot = int(np.sum(valid))
    return float(n_ok / n_tot), n_ok, n_tot


def _boot_ci(values, fn, n=BOOT_N, seed=BOOT_SEED):
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    if values.size == 0:
        return None, None, None
    point = fn(values)
    stats_ = []
    m = len(values)
    for _ in range(int(n)):
        idx = rng.randint(0, m, size=m)
        stats_.append(fn(values[idx]))
    lo, hi = np.quantile(np.asarray(stats_, dtype=np.float64), [0.025, 0.975])
    return float(point), float(lo), float(hi)


def _acc_bundle(g, a5, valid):
    valid = np.asarray(valid, dtype=bool)
    n = int(g.shape[0])
    coverage = float(valid.mean()) if n else 0.0
    acc_valid, n_corr, n_pairs = _pair_acc(g, a5, mask=valid)
    acc_all, _, _ = _pair_acc(g, a5, mask=None)
    acc_eff = coverage * (acc_valid if np.isfinite(acc_valid) else 0.5) + (1.0 - coverage) * 0.5
    return {
        "n": n,
        "n_valid": int(valid.sum()),
        "coverage": coverage,
        "Acc_valid": acc_valid,
        "Acc_all": acc_all,
        "Acc_effective": acc_eff,
        "n_correct_pairs": n_corr,
        "n_pairs_valid": n_pairs,
    }


def _q(x, p):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return float(np.quantile(x, p))


def _row_est(g, a5, valid, A, Anet, peak_ex, n_step, est, terrain="pooled"):
    b = _acc_bundle(g, a5, valid)
    A = np.asarray(A, dtype=np.float64)
    Anet = np.asarray(Anet, dtype=np.float64)
    peak_ex = np.asarray(peak_ex, dtype=np.float64)
    v = np.asarray(valid, dtype=bool)
    twin_v = peak_ex > TUBE
    A_v = A[v] if v.any() else A
    Anet_v = Anet[v] if v.any() else Anet
    return {
        "terrain": terrain,
        "method": f"{est} N={n_step}",
        "est": est,
        "N": int(n_step),
        **b,
        "Rinfo": _rinfo(b["Acc_effective"]),
        "Rinfo_valid": _rinfo(b["Acc_valid"]),
        "mean_A": float(np.mean(A)),
        "median_A": float(np.median(A)),
        "mean_A_valid": float(np.mean(A_v)) if A_v.size else None,
        "P_Agt0": float((A > 0).mean()),
        "P_Agt0_05": float((A > 0.05).mean()),
        "mean_A_net": float(np.mean(Anet)),
        "median_A_net": float(np.median(Anet)),
        "P_Anet_gt0": float((Anet > 0).mean()),
        "p25_A_net": _q(Anet, 0.25),
        "p75_A_net": _q(Anet, 0.75),
        "twin_violation": float(twin_v.mean()),
        "n_twin_safe": int((~twin_v).sum()),
        "Acc_twin_safe": _pair_acc(g[~twin_v] if (~twin_v).any() else g[:0], a5[~twin_v] if (~twin_v).any() else a5[:0])[0]
        if (~twin_v).any()
        else float("nan"),
        "mean_A_twin_safe": float(np.mean(A[~twin_v])) if (~twin_v).any() else None,
        "median_excess_peak": float(np.median(peak_ex)),
        "A": A,
        "A_net": Anet,
        "valid": v,
        "peak_ex": peak_ex,
    }


def _decide(matrix, attr, noise):
    e1 = [r for r in matrix if r["est"] == "E1" and r["terrain"] == "pooled"]
    twin_v = {r["N"]: r["twin_violation"] for r in attr}
    old_v = {r["N"]: r["old_violation"] for r in attr}
    noise_p95 = max((r.get("p95_noise_m") or 0) for r in noise) if noise else 0.0
    any_acc = [r for r in e1 + [x for x in matrix if x["est"] in ("E2", "scalar") and x["terrain"] == "pooled"]]
    best = max(any_acc, key=lambda r: r["Acc_effective"] if r["Acc_effective"] is not None else -1)
    twin_low = all(v <= 0.15 for v in twin_v.values())
    twin_high = any(v > 0.30 for v in twin_v.values())
    old_high = all(v >= 0.70 for v in old_v.values())
    some_good = any(
        (r["Acc_effective"] or 0) >= 0.60 and (r["mean_A"] or 0) > 0 and (r["median_A_net"] or 0) > 0
        for r in any_acc
    )
    all_bad_acc = all((r["Acc_effective"] or 0) <= 0.55 for r in any_acc)
    acc_ok_a_bad = (best["Acc_effective"] or 0) >= 0.60 and (
        (best["mean_A"] or 0) <= 0 or (best["median_A_net"] or 0) <= 0
    )
    if noise_p95 >= 0.004:
        case, nxt = "B2-E", "fix simulator determinism / paired-noise control"
        bottleneck = "simulator counterfactual noise"
    elif twin_high:
        case, nxt = "B2-C", "lower-amplitude probe sweep (0.25/0.5/0.75/1.0°) at N=6"
        bottleneck = "probe duration/amplitude"
    elif old_high and twin_low and some_good:
        case, nxt = "B2-A", "online nominal-drift estimation (deployment still lacks a real twin)"
        bottleneck = "safety attribution (now resolved); next is online twin"
    elif twin_low and all_bad_acc:
        case, nxt = "B2-B", "better causal estimator (recursive / longer quieter / drift model / residual prior)"
        bottleneck = "online estimation"
    elif acc_ok_a_bad:
        case, nxt = "B2-D", "revisit identification target vs 5°/0.2s advantage"
        bottleneck = "identification target"
    else:
        case, nxt = "PARTIAL", "stop and report; do not start P4-C"
        bottleneck = "mixed"
    return {
        "case": case,
        "run_p4c": False,
        "keep_stage2_latent": True,
        "next": nxt,
        "bottleneck": bottleneck,
        "best_method": best["method"],
        "best_Acc_effective": best["Acc_effective"],
        "twin_violation_by_N": twin_v,
        "old_violation_by_N": old_v,
        "noise_p95_m": noise_p95,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    args = ap.parse_args()
    root = Path(args.root)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (root / "rollouts" / "twin").mkdir(parents=True, exist_ok=True)
    (root / "rollouts" / "probe").mkdir(parents=True, exist_ok=True)
    (root / "rollouts" / "probe_correction").mkdir(parents=True, exist_ok=True)

    by = {}
    for t in TERRAINS:
        p = root / "p4b2" / "loco" / t / "p4b2.npz"
        if not p.exists():
            print(f"[p4b2] missing {p}", flush=True)
            continue
        by[t] = np.load(p, allow_pickle=True)
    if not by:
        raise FileNotFoundError("no p4b2.npz")

    def cat(key):
        return np.concatenate([by[t][key] for t in TERRAINS if t in by and key in by[t].files], axis=0)

    sample = next(iter(by.values()))
    ns = [int(x) for x in sample["ns"].tolist()] if "ns" in sample.files else list(NS)
    a5p = cat("a5_gt").astype(np.float64)
    n_pooled = int(a5p.shape[0])

    attr_rows = []
    noise_rows = []
    matrix = []
    per_t_rows = []
    per_state = []

    for n_step in ns:
        peak_old = cat(f"peak_old_{n_step}")
        peak_ex = cat(f"peak_ex_{n_step}")
        peak_nom = cat(f"peak_nom_{n_step}")
        peak_noise = cat(f"peak_noise_{n_step}")
        r_pr = cat(f"r_probe_{n_step}")
        old_step = cat(f"old_step_{n_step}")
        twin_step = cat(f"twin_step_{n_step}")
        attr_rows.append({
            "N": int(n_step),
            "old_violation": float((old_step >= 0).mean()),
            "twin_violation": float((twin_step >= 0).mean()),
            "median_old_peak": float(np.median(peak_old)),
            "median_excess_peak": float(np.median(peak_ex)),
            "median_nominal_drift": float(np.median(peak_nom)),
            "median_R_probe": float(np.median(r_pr)),
            "p90_R_probe": float(np.quantile(r_pr, 0.90)),
            "p95_R_probe": float(np.quantile(r_pr, 0.95)),
            "p95_excess_peak": float(np.quantile(peak_ex, 0.95)),
            "p95_old_peak": float(np.quantile(peak_old, 0.95)),
        })
        noise_rows.append({
            "N": int(n_step),
            "median_noise_m": float(np.median(peak_noise)),
            "p95_noise_m": float(np.quantile(peak_noise, 0.95)),
            "max_noise_m": float(np.max(peak_noise)),
            "median_auc_noise": float(np.median(cat(f"auc_noise_{n_step}"))),
        })
        for est in ESTS:
            g = cat(f"g_{est}_{n_step}").astype(np.float64)
            valid = cat(f"valid_{est}_{n_step}").astype(bool)
            A = cat(f"A_{est}_{n_step}")
            Anet = cat(f"Anet_{est}_{n_step}")
            matrix.append(_row_est(g, a5p, valid, A, Anet, peak_ex, n_step, est, "pooled"))
        seeds, ts, clips, terrs, e0 = cat("seed"), cat("t"), cat("clip"), cat("terrain"), cat("e0")
        for i in range(n_pooled):
            for est in ESTS:
                per_state.append({
                    "terrain": str(terrs[i]),
                    "seed": int(seeds[i]),
                    "t": int(ts[i]),
                    "clip": str(clips[i]),
                    "e0": float(e0[i]),
                    "N": int(n_step),
                    "est": est,
                    "valid": bool(cat(f"valid_{est}_{n_step}")[i]),
                    "A": float(cat(f"A_{est}_{n_step}")[i]),
                    "A_net": float(cat(f"Anet_{est}_{n_step}")[i]),
                    "peak_old": float(peak_old[i]),
                    "peak_ex": float(peak_ex[i]),
                    "peak_nom": float(peak_nom[i]),
                    "peak_noise": float(peak_noise[i]),
                    "r_probe": float(r_pr[i]),
                    "old_step": int(old_step[i]),
                    "twin_step": int(twin_step[i]),
                    "would_old_abort": bool(old_step[i] >= 0),
                    "would_twin_abort": bool(twin_step[i] >= 0),
                })

    for t, z in by.items():
        a5 = z["a5_gt"].astype(np.float64)
        for n_step in ns:
            peak_ex = z[f"peak_ex_{n_step}"]
            for est in ESTS:
                row = _row_est(
                    z[f"g_{est}_{n_step}"].astype(np.float64),
                    a5,
                    z[f"valid_{est}_{n_step}"].astype(bool),
                    z[f"A_{est}_{n_step}"],
                    z[f"Anet_{est}_{n_step}"],
                    peak_ex,
                    n_step,
                    est,
                    t,
                )
                slim = {k: v for k, v in row.items() if k not in ("A", "A_net", "valid", "peak_ex")}
                per_t_rows.append(slim)

    decision = _decide(matrix, attr_rows, noise_rows)
    baselines = [
        {
            "method": "Clone 1-step",
            "est": "clone",
            "N": 1,
            "coverage": 1.0,
            "Acc_valid": ACC_CLONE,
            "Acc_effective": ACC_CLONE,
            "Rinfo": 1.0,
            "mean_A": MEAN_A_CLONE,
            "median_A_net": None,
            "P_Anet_gt0": None,
            "twin_violation": None,
        },
        {
            "method": "Old ± seq",
            "est": "seq",
            "N": 15,
            "coverage": 1.0,
            "Acc_valid": ACC_SEQ,
            "Acc_effective": ACC_SEQ,
            "Rinfo": _rinfo(ACC_SEQ),
            "mean_A": 0.0,
            "median_A_net": None,
            "P_Anet_gt0": None,
            "twin_violation": None,
        },
    ]

    keys_est = [
        "method", "est", "N", "coverage", "Acc_valid", "Acc_effective", "Rinfo",
        "mean_A", "median_A", "P_Agt0", "mean_A_net", "median_A_net", "P_Anet_gt0",
        "twin_violation", "n", "n_valid",
    ]
    with (root / "estimator_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys_est)
        w.writeheader()
        for r in baselines:
            w.writerow({k: r.get(k) for k in keys_est})
        for r in matrix:
            w.writerow({k: r.get(k) for k in keys_est})
    with (root / "attribution_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(attr_rows[0].keys()))
        w.writeheader()
        w.writerows(attr_rows)
    with (root / "twin_noise_floor.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(noise_rows[0].keys()))
        w.writeheader()
        w.writerows(noise_rows)
    with (root / "per_state_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_state[0].keys()))
        w.writeheader()
        w.writerows(per_state)
    pt_keys = [k for k in per_t_rows[0].keys()] if per_t_rows else []
    with (root / "per_terrain_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pt_keys)
        w.writeheader()
        w.writerows(per_t_rows)

    # plots
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = {6: "#1f4e79", 8: "#c44e52", 10: "#2a9d8f"}
    for n_step in ns:
        x = 1000.0 * np.maximum(cat(f"peak_old_{n_step}"), 0)
        y = 1000.0 * cat(f"peak_ex_{n_step}")
        ax.scatter(x, y, s=14, alpha=0.55, color=colors[int(n_step)], label=f"N={n_step}")
    ax.axvline(5.0, color="k", ls="--", lw=1)
    ax.axhline(5.0, color="k", ls="--", lw=1)
    ax.set_xlabel(r"old peak $[E_{probe}-E(t_0)]_+$ (mm)")
    ax.set_ylabel(r"twin excess peak $[E_{probe}-E_{twin}]_+$ (mm)")
    ax.set_title("B2-1 Old vs twin-relative guard")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "B2_1_old_vs_twin_guard.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharey=True)
    for ax, t in zip(axes, TERRAINS):
        if t not in by:
            ax.set_title(t + " missing")
            continue
        z = by[t]
        n_step = 6 if 6 in ns else ns[0]
        ex = z[f"peak_ex_{n_step}"]
        i = int(np.argsort(ex)[len(ex) // 2])
        e0 = float(z["e0"][i])
        tw = z[f"E_twin_{n_step}"][i]
        pr = z[f"E_probe_{n_step}"][i]
        tt = (np.arange(len(tw)) + 1) * DT
        ax.axhline(e0, color="0.5", ls=":", label=r"$E(t_0)$")
        ax.plot(tt, tw, color="#1f4e79", label="Stage2 twin")
        ax.plot(tt, pr, color="#c44e52", label="probe")
        ax.plot(tt, np.maximum(pr - tw, 0), color="#2a9d8f", ls="--", label="excess+")
        ax.set_title(t)
        ax.set_xlabel("t (s)")
        # save representative rollout
        np.savetxt(
            root / "rollouts" / "twin" / f"{t}_N{n_step}_i{i}.csv",
            np.column_stack([tt, tw]),
            delimiter=",",
            header="t,E_twin",
            comments="",
        )
        np.savetxt(
            root / "rollouts" / "probe" / f"{t}_N{n_step}_i{i}.csv",
            np.column_stack([tt, pr, np.maximum(pr - tw, 0)]),
            delimiter=",",
            header="t,E_probe,excess_pos",
            comments="",
        )
    axes[0].set_ylabel(r"$E_I$ (m)")
    axes[0].legend(fontsize=7)
    fig.suptitle("B2-2 Intent trajectories (median-excess state, N=6)")
    fig.tight_layout()
    fig.savefig(plots / "B2_2_intent_trajectories.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for n_step, ls in zip(ns, ("-", "--", ":")):
        old_s = cat(f"old_step_{n_step}")
        twin_s = cat(f"twin_step_{n_step}")
        times = np.arange(int(n_step))
        old_surv = [(old_s < 0).mean() if False else float(np.mean((old_s < 0) | (old_s > k))) for k in times]
        twin_surv = [float(np.mean((twin_s < 0) | (twin_s > k))) for k in times]
        ax.plot((times + 1) * DT, old_surv, color="#c44e52", ls=ls, label=f"old N={n_step}")
        ax.plot((times + 1) * DT, twin_surv, color="#1f4e79", ls=ls, label=f"twin N={n_step}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("fraction still inside 5 mm guard")
    ax.set_title("B2-3 Guard survival (post-hoc, no abort)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "B2_3_guard_survival.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for est, col, mk in (("E1", "#1f4e79", "o"), ("E2", "#c44e52", "s"), ("scalar", "#2a9d8f", "^")):
        rs = [r for r in matrix if r["est"] == est]
        ax.plot([r["N"] for r in rs], [r["Acc_effective"] for r in rs], mk + "-", color=col, label=est)
    ax.axhline(ACC_CLONE, color="0.3", ls=":", label="clone 0.658")
    ax.axhline(ACC_SEQ, color="0.5", ls="--", label="old seq 0.517")
    ax.axhline(0.60, color="#2a9d8f", ls=":", label="gate 0.60")
    ax.axhline(0.50, color="0.7", ls=":")
    ax.set_xlabel("N")
    ax.set_ylabel("Acc_effective")
    ax.set_title("B2-4 Accuracy vs horizon (n=158)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "B2_4_accuracy_vs_horizon.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    data, labs, cols = [], [], []
    for n_step in ns:
        r = next(x for x in matrix if x["est"] == "E1" and x["N"] == n_step)
        data.append(r["A_net"])
        labs.append(f"N={n_step}")
        cols.append(colors[int(n_step)])
    bp = ax.boxplot(data, labels=labs, patch_artist=True, medianprops={"color": "k"})
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
    ax.axhline(0.0, color="0.5", ls=":")
    ax.set_ylabel(r"$A_{net}$")
    ax.set_title("B2-5 Net advantage (E1, includes probe delay)")
    fig.tight_layout()
    fig.savefig(plots / "B2_5_net_advantage.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for n_step in ns:
        rp = cat(f"r_probe_{n_step}")
        ax.hist(np.clip(rp, 0, 5), bins=30, alpha=0.4, color=colors[int(n_step)], label=f"N={n_step}")
    ax.axvline(1.0, color="k", ls="--")
    ax.set_xlabel(r"$R_{probe}$")
    ax.set_ylabel("count")
    ax.set_title("B2-6 Probe contribution ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "B2_6_probe_contribution.png", dpi=140)
    plt.close(fig)

    # bootstrap primary E1 N=6
    prim = next(r for r in matrix if r["est"] == "E1" and r["N"] == 6)
    g6 = cat("g_E1_6")
    v6 = cat("valid_E1_6").astype(bool)
    A6 = cat("A_E1_6")
    An6 = cat("Anet_E1_6")
    ex6 = cat("peak_ex_6")
    rng = np.random.RandomState(BOOT_SEED)

    def boot_states(fn):
        m = n_pooled
        pts = []
        for _ in range(BOOT_N):
            idx = rng.randint(0, m, size=m)
            pts.append(fn(idx))
        return float(fn(np.arange(m))), float(np.quantile(pts, 0.025)), float(np.quantile(pts, 0.975))

    def f_acc_eff(idx):
        return _acc_bundle(g6[idx], a5p[idx], v6[idx])["Acc_effective"]

    def f_acc_val(idx):
        return _acc_bundle(g6[idx], a5p[idx], v6[idx])["Acc_valid"]

    boot = {
        "E1_N6_Acc_effective": boot_states(f_acc_eff),
        "E1_N6_Acc_valid": boot_states(f_acc_val),
        "E1_N6_coverage": boot_states(lambda idx: float(v6[idx].mean())),
        "E1_N6_mean_A": boot_states(lambda idx: float(np.mean(A6[idx]))),
        "E1_N6_median_A_net": boot_states(lambda idx: float(np.median(An6[idx]))),
        "E1_N6_P_Anet_gt0": boot_states(lambda idx: float((An6[idx] > 0).mean())),
        "N6_twin_violation": boot_states(lambda idx: float((ex6[idx] > TUBE).mean())),
        "N6_median_excess_peak": boot_states(lambda idx: float(np.median(ex6[idx]))),
    }

    e1n6 = prim
    n6_attr = next(r for r in attr_rows if r["N"] == 6)
    n8_attr = next(r for r in attr_rows if r["N"] == 8)
    n10_attr = next(r for r in attr_rows if r["N"] == 10)
    noise6 = next(r for r in noise_rows if r["N"] == 6)

    def _line(r):
        return (
            f"| {r.get('method')} | {_fmt(r.get('coverage'))} | {_fmt(r.get('Acc_valid'))} | "
            f"{_fmt(r.get('Acc_effective'))} | {_fmt(r.get('Rinfo'))} | {_fmt(r.get('mean_A'), 4)} | "
            f"{_fmt(r.get('median_A_net'), 4)} | {_fmt(r.get('P_Anet_gt0'))} | {_fmt(r.get('twin_violation'))} |"
        )

    q = {
        "1_old_viol_was_nominal_drift": bool(
            n6_attr["old_violation"] > 0.7 and n6_attr["twin_violation"] < n6_attr["old_violation"] - 0.3
        ),
        "2_twin_violation": {r["N"]: r["twin_violation"] for r in attr_rows},
        "3_noise_floor_mm": {r["N"]: 1000.0 * r["median_noise_m"] for r in noise_rows},
        "4_median_R_probe": {r["N"]: r["median_R_probe"] for r in attr_rows},
        "5_probe_inside_twin_tube": bool(n6_attr["twin_violation"] <= 0.15),
        "6_coverage_acc": {
            "coverage": e1n6["coverage"],
            "Acc_valid": e1n6["Acc_valid"],
            "Acc_effective": e1n6["Acc_effective"],
        },
        "7_best": decision["best_method"],
        "8_N6_useful": bool((e1n6["Acc_effective"] or 0) >= 0.60 and (e1n6["mean_A"] or 0) > 0),
        "9_any_ge_0.60": bool(any((r["Acc_effective"] or 0) >= 0.60 for r in matrix)),
        "10_mean_A_positive": bool((e1n6["mean_A"] or 0) > 0),
        "11_median_Anet_positive": bool((e1n6["median_A_net"] or 0) > 0),
        "12_consistent_terrains": None,
        "13_bottleneck": decision["bottleneck"],
        "14_next": decision["next"],
    }
    accs_t = {}
    for t in TERRAINS:
        rs = [r for r in per_t_rows if r["terrain"] == t and r["est"] == "E1" and r["N"] == 6]
        if rs:
            accs_t[t] = rs[0]["Acc_effective"]
    if accs_t:
        q["12_consistent_terrains"] = bool(max(accs_t.values()) - min(accs_t.values()) < 0.10)

    cfg = {
        "step": "P4-B2",
        "n": n_pooled,
        "ns": ns,
        "primary_horizon": 6,
        "probe_deg": 1.0,
        "k": 3,
        "tube_m": TUBE,
        "dt": DT,
        "no_abort": True,
        "no_p4c": True,
        "reused_p4b_U": True,
        "estimator": "unchanged E1/E2/scalar lstsq",
        "correction_deg": 5.0,
        "correction_horizon_s": K_HORIZON * DT if False else 0.2,
        "bootstrap": {"resamples": BOOT_N, "seed": BOOT_SEED},
    }
    # K_HORIZON not imported
    cfg["correction_horizon_s"] = 0.2
    (root / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    metrics = {
        "n_pooled": n_pooled,
        "ns": ns,
        "acc_clone": ACC_CLONE,
        "attribution": attr_rows,
        "noise_floor": noise_rows,
        "estimator": [{k: r.get(k) for k in keys_est} for r in baselines + matrix],
        "per_terrain_E1_N6": {t: accs_t.get(t) for t in TERRAINS},
        "bootstrap_E1_N6": {k: {"point": v[0], "lo": v[1], "hi": v[2]} for k, v in boot.items()},
        "decision": decision,
        "questions": q,
        "no_p4c": True,
    }
    (root / "metrics.json").write_text(json.dumps(sanitize(metrics), indent=2), encoding="utf-8")

    lines = [
        "# P4-B2 — Twin-Referenced Tube-Safe Online Interaction Identification",
        "",
        "Guard is **post-hoc** and **twin-relative**. No rollout abort. Same 158 P4-B IDs, same \(B_{UCR}^{k=3}\) codes, same E1/E2/scalar.",
        "Human \(I_t\) unchanged. Safety question: did the probe make tracking worse than frozen Stage-2 would have?",
        "",
        f"Pooled n=**{n_pooled}**. Case **{decision['case']}**. P4-C: **False**.",
        "",
        "## Estimator table (full sample)",
        "",
        "| Method | Coverage | Acc valid | Acc effective | Rinfo | mean A | median A_net | P(A_net>0) | twin violation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _line(baselines[0]),
        _line(baselines[1]),
    ]
    for r in matrix:
        if r["est"] in ("E1", "E2") or (r["est"] == "scalar" and r["N"] == 8):
            lines.append(_line(r))
    lines += [
        "",
        "## Attribution table",
        "",
        "| Horizon | old violation | twin violation | median old peak | median excess peak | median nominal drift | median R_probe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in attr_rows:
        lines.append(
            f"| N={r['N']} | {_fmt(r['old_violation'])} | {_fmt(r['twin_violation'])} | "
            f"{_fmt(1000*r['median_old_peak'], 2)} mm | {_fmt(1000*r['median_excess_peak'], 2)} mm | "
            f"{_fmt(1000*r['median_nominal_drift'], 2)} mm | {_fmt(r['median_R_probe'], 3)} |"
        )
    lines += [
        "",
        "## Twin-vs-twin noise floor",
        "",
        "| N | median | p95 | max |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for r in noise_rows:
        lines.append(
            f"| {r['N']} | {_fmt(1000*r['median_noise_m'], 3)} mm | "
            f"{_fmt(1000*r['p95_noise_m'], 3)} mm | {_fmt(1000*r['max_noise_m'], 3)} mm |"
        )
    lines += [
        "",
        "## MASTER questions",
        "",
        f"1. Previous 87% violation primarily nominal Stage2 drift? **{q['1_old_viol_was_nominal_drift']}** "
        f"(old N=6 viol={_fmt(n6_attr['old_violation'])}, twin viol={_fmt(n6_attr['twin_violation'])}).",
        f"2. Twin-relative violation N=6/8/10: **{_fmt(n6_attr['twin_violation'])} / {_fmt(n8_attr['twin_violation'])} / {_fmt(n10_attr['twin_violation'])}**.",
        f"3. Twin-vs-twin noise floor N=6: median **{_fmt(1000*noise6['median_noise_m'], 3)} mm**, p95 **{_fmt(1000*noise6['p95_noise_m'], 3)} mm**.",
        f"4. Median \(R_{{probe}}\) N=6/8/10: **{_fmt(n6_attr['median_R_probe'])} / {_fmt(n8_attr['median_R_probe'])} / {_fmt(n10_attr['median_R_probe'])}**.",
        f"5. Probe inside twin-relative 5 mm tube (N=6 viol≤15%)? **{q['5_probe_inside_twin_tube']}**.",
        f"6. E1 N=6 coverage **{_fmt(e1n6['coverage'])}**, Acc_valid **{_fmt(e1n6['Acc_valid'])}**, Acc_effective **{_fmt(e1n6['Acc_effective'])}**.",
        f"7. Best method: **{decision['best_method']}** (Acc_eff={_fmt(decision['best_Acc_effective'])}).",
        f"8. N=6 recovers useful information (Acc_eff≥0.60 and mean A>0)? **{q['8_N6_useful']}**.",
        f"9. Any method Acc_effective ≥ 0.60? **{q['9_any_ge_0.60']}**.",
        f"10. E1 N=6 mean A > 0? **{q['10_mean_A_positive']}** (mean A={_fmt(e1n6['mean_A'], 4)}).",
        f"11. E1 N=6 median \(A_{{net}}\) > 0? **{q['11_median_Anet_positive']}** (median={_fmt(e1n6['median_A_net'], 4)}).",
        f"12. Consistent across terrains (E1 N=6 Acc_eff range <0.10)? **{q['12_consistent_terrains']}** {accs_t}.",
        f"13. Bottleneck: **{decision['bottleneck']}**.",
        f"14. Next: **{decision['next']}**. Do **not** start P4-C. Do **not** change Stage-2 latent.",
        "",
        f"Bootstrap E1 N=6 Acc_effective: {_fmt(boot['E1_N6_Acc_effective'][0])} "
        f"[{_fmt(boot['E1_N6_Acc_effective'][1])}, {_fmt(boot['E1_N6_Acc_effective'][2])}].",
        "",
        f"**Case {decision['case']}**. P4-C disabled.",
    ]
    (root / "MASTER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    p4c = root / "p4c_executability"
    p4c.mkdir(parents=True, exist_ok=True)
    (p4c / "NOT_RUN.md").write_text(
        "NOT RUN — P4-C disabled for P4-B2. Begins only after this interpretation is reviewed.\n",
        encoding="utf-8",
    )
    print(json.dumps(sanitize(decision), indent=2), flush=True)
    for r in matrix:
        if r["est"] == "E1" or (r["est"] == "scalar" and r["N"] == 8):
            print(
                f"{r['method']}: cov={_fmt(r['coverage'])} Acc_eff={_fmt(r['Acc_effective'])} "
                f"A={_fmt(r['mean_A'],4)} Anet_med={_fmt(r['median_A_net'],4)} "
                f"twin={_fmt(r['twin_violation'])}",
                flush=True,
            )


if __name__ == "__main__":
    main()
