#!/usr/bin/env python3
"""P4-B coded multiplex online ID. No P4-C unless success gates pass."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from p3_common import adv_block, sanitize, stats

TERRAINS = ("steps", "slip", "slope_down")
ROOT_DEFAULT = "/data/home/chenxiangyu/robotics/Anybody/results/p4b_tube_online_id"
ACC_CLONE = 0.658
MEAN_A_CLONE = 0.020
ACC_SEQ = 0.517
TUBE_ABS = 0.005  # 0.1 * 5 cm
NS = (6, 8, 10)
ESTS = ("E1", "E2", "scalar")


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and (not np.isfinite(v))):
        return "n/a"
    return f"{float(v):.{nd}f}"


def _rinfo(acc, acc_raw=ACC_CLONE):
    if acc is None or acc_raw is None or abs(acc_raw - 0.5) < 1e-6:
        return None
    return (float(acc) - 0.5) / (float(acc_raw) - 0.5)


def _sign_acc(g: np.ndarray, a5: np.ndarray, mask=None) -> float:
    s_star = np.sign(a5[:, :, 0] - a5[:, :, 1])
    s_hat = np.sign(-g)
    valid = s_star != 0
    if mask is not None:
        valid = valid & np.asarray(mask).reshape(-1, 1)
    if not valid.any():
        return float("nan")
    return float((s_hat[valid] == s_star[valid]).mean())


def _row(acc, A, Anet, di, abort, n_step, est, terrain="pooled", dE=None):
    ok = ~np.asarray(abort, dtype=bool)
    A = np.asarray(A, dtype=np.float64)
    Anet = np.asarray(Anet, dtype=np.float64)
    peak = np.asarray(di, dtype=np.float64)
    if dE is not None:
        dE = np.asarray(dE, dtype=np.float64)
        with np.errstate(all="ignore"):
            peak = np.nanmax(dE, axis=1)
            mag = np.nanmax(np.abs(dE), axis=1)
        peak = np.where(np.isfinite(peak), peak, 0.0)
        mag = np.where(np.isfinite(mag), mag, 0.0)
    else:
        mag = np.abs(peak)
    viol = peak > TUBE_ABS
    A_ok = A[ok] if ok.any() else A
    Anet_ok = Anet[ok] if ok.any() else Anet
    return {
        "terrain": terrain,
        "method": f"{est} N={n_step}",
        "est": est,
        "N": int(n_step),
        "n": int(A.shape[0]),
        "n_ok": int(ok.sum()),
        "sign_acc": acc,
        "R_info": _rinfo(acc),
        "mean_A": float(np.mean(A_ok)) if A_ok.size else None,
        "median_A": float(np.median(A_ok)) if A_ok.size else None,
        "P_Agt0": float((A_ok > 0).mean()) if A_ok.size else None,
        "mean_A_net": float(np.mean(Anet_ok)) if Anet_ok.size else None,
        "median_A_net": float(np.median(Anet_ok)) if Anet_ok.size else None,
        "P_Anet_gt0": float((Anet_ok > 0).mean()) if Anet_ok.size else None,
        "A": adv_block(A_ok),
        "A_net": adv_block(Anet_ok),
        "tube_violation": float(viol.mean()) if peak.size else None,
        "probe_DI_m": stats(mag),
        "probe_DI_mm": stats(1000.0 * mag),
        "probe_DI_band": stats(mag / 0.05),
        "abort_frac": float((~ok).mean()) if A.size else None,
        "max_probe_DI_m": float(np.nanmax(mag)) if mag.size else None,
        "median_probe_DI_m": float(np.nanmedian(mag)) if mag.size else None,
        "p95_probe_DI_m": float(np.nanquantile(mag, 0.95)) if mag.size else None,
    }


def _decide(matrix):
    coded = [r for r in matrix if r["est"] == "E1" and r["terrain"] == "pooled"]
    primary = next((r for r in coded if r["N"] == 8), None)
    accs = [r["sign_acc"] for r in coded if r["sign_acc"] is not None and np.isfinite(r["sign_acc"])]
    fail_window = bool(accs) and all(a <= 0.55 for a in accs)
    ok = False
    if primary is not None:
        ok = (
            (primary.get("sign_acc") or 0) >= 0.60
            and (primary.get("mean_A") or 0) > 0
            and (primary.get("median_A_net") or 0) > 0
            and (primary.get("tube_violation") or 1) < 0.05
        )
    return {
        "success": ok,
        "run_p4c": ok,
        "fail_short_window": fail_window and not ok,
        "primary": "E1 N=8",
        "keep_stage2_latent": True,
        "no_hard_nullspace": True,
        "no_trajbooster_interface": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    args = ap.parse_args()
    root = Path(args.root)
    plots = root / "p4b_online_id" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    by = {}
    for t in TERRAINS:
        p = root / "p4b" / "loco" / t / "p4b.npz"
        if not p.exists():
            print(f"[p4b] missing {p}", flush=True)
            continue
        by[t] = np.load(p, allow_pickle=True)
    if not by:
        raise FileNotFoundError("no p4b.npz")

    def cat(key):
        return np.concatenate([by[t][key] for t in TERRAINS if t in by and key in by[t].files], axis=0)

    sample = next(iter(by.values()))
    ns = [int(x) for x in sample["ns"].tolist()] if "ns" in sample.files else list(NS)
    matrix = []
    per_t = {}
    for t, z in by.items():
        per_t[t] = {}
        a5 = z["a5_gt"].astype(np.float64)
        for n_step in ns:
            di = z[f"di_max_{n_step}"].astype(np.float64)
            for est in ESTS:
                g = z[f"g_{est}_{n_step}"].astype(np.float64)
                abort = z[f"abort_{est}_{n_step}"].astype(bool)
                acc = _sign_acc(g, a5, mask=~abort)
                row = _row(
                    acc, z[f"A_{est}_{n_step}"], z[f"Anet_{est}_{n_step}"], di, abort, n_step, est, t,
                    dE=z[f"dE_{n_step}"],
                )
                per_t[t][f"{est}:{n_step}"] = {k: v for k, v in row.items() if k not in ("A", "A_net", "probe_DI_m", "probe_DI_mm", "probe_DI_band")}
        # pooled filled later
    a5p = cat("a5_gt").astype(np.float64)
    for n_step in ns:
        di = cat(f"di_max_{n_step}").astype(np.float64)
        for est in ESTS:
            g = cat(f"g_{est}_{n_step}").astype(np.float64)
            abort = cat(f"abort_{est}_{n_step}").astype(bool)
            acc = _sign_acc(g, a5p, mask=~abort)
            matrix.append(_row(
                acc, cat(f"A_{est}_{n_step}"), cat(f"Anet_{est}_{n_step}"), di, abort, n_step, est, "pooled",
                dE=cat(f"dE_{n_step}"),
            ))

    baselines = [
        {
            "terrain": "pooled",
            "method": "Clone 1-step",
            "est": "clone",
            "N": 1,
            "n": int(a5p.shape[0]),
            "sign_acc": ACC_CLONE,
            "R_info": 1.0,
            "mean_A": MEAN_A_CLONE,
            "median_A": None,
            "mean_A_net": None,
            "median_A_net": None,
            "tube_violation": None,
        },
        {
            "terrain": "pooled",
            "method": "Old ± seq",
            "est": "seq",
            "N": 15,
            "n": int(a5p.shape[0]),
            "sign_acc": ACC_SEQ,
            "R_info": _rinfo(ACC_SEQ),
            "mean_A": 0.0,
            "median_A": None,
            "mean_A_net": None,
            "median_A_net": None,
            "tube_violation": None,
        },
    ]
    decision = _decide(matrix)
    show = baselines + [r for r in matrix if r["est"] in ("E1", "scalar") or (r["est"] == "E2")]
    csv_rows = []
    keys = [
        "method", "est", "N", "n", "n_ok", "sign_acc", "R_info", "mean_A", "median_A",
        "P_Agt0", "mean_A_net", "median_A_net", "P_Anet_gt0", "tube_violation",
        "max_probe_DI_m", "median_probe_DI_m", "p95_probe_DI_m", "abort_frac",
    ]
    for r in baselines + matrix:
        csv_rows.append({k: r.get(k) for k in keys})

    out_dir = root / "p4b_online_id"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(csv_rows)

    e1 = [r for r in matrix if r["est"] == "E1"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot([r["N"] for r in e1], [r["sign_acc"] for r in e1], "o-", color="#1f4e79", label="E1 coded")
    e2 = [r for r in matrix if r["est"] == "E2"]
    ax.plot([r["N"] for r in e2], [r["sign_acc"] for r in e2], "s--", color="#c44e52", label="E2 increment")
    ax.axhline(ACC_CLONE, color="0.3", ls=":", label="clone 0.658")
    ax.axhline(0.60, color="#2a9d8f", ls="--", label="gate 0.60")
    ax.axhline(0.50, color="0.6", ls=":")
    ax.set_xlabel("horizon N (steps)")
    ax.set_ylabel("sign accuracy")
    ax.set_title("B-1 Online sign accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "B1_acc_vs_N.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot([r["N"] for r in e1], [r["R_info"] for r in e1], "o-", color="#1f4e79", label="E1")
    ax.plot([r["N"] for r in e2], [r["R_info"] for r in e2], "s--", color="#c44e52", label="E2")
    ax.axhline(0.6, color="0.5", ls=":")
    ax.set_xlabel("horizon N")
    ax.set_ylabel(r"$R_{online}$")
    ax.set_title("B-2 Information retention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "B2_Rinfo_vs_N.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot([r["N"] for r in e1], [r["mean_A"] for r in e1], "o-", label="mean A clone")
    ax.plot([r["N"] for r in e1], [r["mean_A_net"] for r in e1], "s--", label="mean A_net")
    ax.axhline(0.0, color="0.5", ls=":")
    ax.set_xlabel("horizon N")
    ax.set_ylabel("advantage")
    ax.set_title("B-3 Clone A vs net A (includes probe delay)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "B3_A_vs_Anet.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for n_step, col in zip(ns, ("#1f4e79", "#c44e52", "#2a9d8f")):
        mag = 1000.0 * np.nanmax(np.abs(cat(f"dE_{n_step}")), axis=1)
        ax.hist(mag[np.isfinite(mag)], bins=30, alpha=0.45, color=col, label=f"N={n_step}")
    ax.axvline(5.0, color="k", ls="--", label="tube guard 5 mm")
    ax.set_xlabel("max probe ΔE_I (mm)")
    ax.set_ylabel("count")
    ax.set_title("B-4 Probe tube disturbance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "B4_probe_DI.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for t, col in zip(TERRAINS, ("#1f4e79", "#c44e52", "#2a9d8f")):
        if t not in by:
            continue
        xs, ys = [], []
        for n_step in ns:
            xs.append(n_step)
            ys.append(per_t[t][f"E1:{n_step}"]["sign_acc"])
        ax.plot(xs, ys, "o-", color=col, label=t)
    ax.axhline(0.60, color="0.5", ls="--")
    ax.set_xlabel("N")
    ax.set_ylabel("E1 sign acc")
    ax.set_title("B-5 Per-terrain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "B5_per_terrain.png", dpi=140)
    plt.close(fig)

    slim = []
    for r in show:
        slim.append({k: r.get(k) for k in keys if k in r or True})
        if "A" in r:
            slim[-1]["A"] = r["A"]
            slim[-1]["A_net"] = r.get("A_net")
            slim[-1]["probe_DI_mm"] = r.get("probe_DI_mm")

    metrics = {
        "n_pooled": int(a5p.shape[0]),
        "ns": ns,
        "acc_clone": ACC_CLONE,
        "tube_abs_m": TUBE_ABS,
        "matrix": [{k: r.get(k) for k in keys} for r in baselines + matrix],
        "per_terrain": per_t,
        "decision": decision,
        "no_projector": True,
        "no_sequential_fd": True,
        "no_p4c_until_success": True,
    }
    (root / "metrics.json").write_text(json.dumps(sanitize(metrics), indent=2), encoding="utf-8")

    def _line(r):
        return (
            f"| {r.get('method')} | {_fmt(r.get('sign_acc'))} | {_fmt(r.get('R_info'))} | "
            f"{_fmt(r.get('mean_A'), 4)} | {_fmt(r.get('median_A_net'), 4)} | "
            f"{_fmt(r.get('tube_violation'), 3)} |"
        )

    lines = [
        "# P4-B — Tube-Constrained Online Interaction Identification",
        "",
        "Coded multiplex $B_{UCR}^{k=3}$, combined 1° geodesic, runtime tube guard $\\Delta E_I\\le 5$ mm.",
        "No hard nullspace, no learned projector, no sequential $\\pm$ pair, no network.",
        "Human target $I_t$ unchanged. Tracking slack $\\neq$ changing human intention.",
        "",
        f"Pooled n={metrics['n_pooled']}. Clone upper bound Acc=**{ACC_CLONE}**. Old sequential Acc≈**{ACC_SEQ}**.",
        "",
        f"Primary **E1 N=8**. Success: **{decision['success']}**. Run P4-C: **{decision['run_p4c']}**.",
        "",
        "| Method | Acc | Rinfo | mean A | median net A | tube violation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        _line(baselines[0]),
        _line(baselines[1]),
    ]
    for r in matrix:
        if r["est"] == "E1":
            lines.append(_line(r))
    scal = next((r for r in matrix if r["est"] == "scalar" and r["N"] == 8), None)
    if scal:
        scal = dict(scal)
        scal["method"] = "Scalar-grad N=8"
        lines.append(_line(scal))
    lines += [
        "",
        "## E2 increment (same N sweep)",
        "",
        "| Method | Acc | Rinfo | mean A | median net A | tube violation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in matrix:
        if r["est"] == "E2":
            lines.append(_line(r))
    prim = next((r for r in matrix if r["est"] == "E1" and r["N"] == 8), None)
    lines += [
        "",
        "## Gates",
        "",
        f"- Acc ≥ 0.60: {None if prim is None else prim['sign_acc']}",
        f"- mean A > 0: {None if prim is None else prim['mean_A']}",
        f"- median A_net > 0: {None if prim is None else prim['median_A_net']}",
        f"- tube violation < 5%: {None if prim is None else prim['tube_violation']}",
        f"- p95 probe ΔE_I: {None if prim is None else prim['p95_probe_DI_m']} m",
        "",
    ]
    if decision["success"]:
        lines.append("PASS. Next is **P4-C — Self-Maintained Intent Executability**. Do not change the autonomy interface.")
    elif decision["fail_short_window"]:
        lines.append(
            "FAIL — current interaction response is observable (clone Acc=0.658) but **not recoverable from this short causal excitation window**. "
            "Do **not** blame Stage-2 latent (P4-A1 excluded that). Do **not** switch to TrajBooster $[v_x,v_y,v_{yaw},h]$. "
            "Next would be recursive ID / longer quieter excitation / drift predictor / learned residual prior — not P4-C, not a new interface."
        )
    else:
        lines.append("PARTIAL. Not yet closed-loop. Keep Stage-2 latent. Do not start P4-C.")
    lines += [
        "",
        "P4-C is **not run** from this script.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "MASTER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    p4c = root / "p4c_executability"
    p4c.mkdir(parents=True, exist_ok=True)
    if not decision["run_p4c"]:
        (p4c / "NOT_RUN.md").write_text(
            "NOT RUN — P4-C blocked until P4-B Acc≥0.60, mean A>0, median A_net>0, tube violation<5%.\n",
            encoding="utf-8",
        )
    print(json.dumps(sanitize(decision), indent=2), flush=True)
    for r in matrix:
        if r["est"] == "E1" or (r["est"] == "scalar" and r["N"] == 8):
            print(
                f"{r['method']}: Acc={_fmt(r['sign_acc'])} R={_fmt(r['R_info'])} "
                f"A={_fmt(r['mean_A'],4)} Anet_med={_fmt(r['median_A_net'],4)} "
                f"tube={_fmt(r['tube_violation'])}",
                flush=True,
            )


if __name__ == "__main__":
    main()
