#!/usr/bin/env python3
"""P4-A1 intent-tube Pareto. No P4-A2 unless T2/T3. No sequential."""
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
ROOT_DEFAULT = "/data/home/chenxiangyu/robotics/Anybody/results/p4a_intent_tube_interface"


def _pair_and_best(r_e: np.ndarray, a5: np.ndarray) -> dict:
    s_star = np.sign(a5[:, :, 0] - a5[:, :, 1])
    s_hat = np.sign(-r_e)
    valid = s_star != 0
    pair = float((s_hat[valid] == s_star[valid]).mean()) if valid.any() else float("nan")
    ax = np.argmax(np.abs(r_e), axis=1)
    n = r_e.shape[0]
    a_sel = np.empty(n, dtype=np.float64)
    for i in range(n):
        j = int(ax[i])
        si = 0 if (-r_e[i, j]) >= 0 else 1
        a_sel[i] = a5[i, j, si]
    a_star = np.maximum(a5.max(axis=(1, 2)), 0.0)
    ratio = np.where(a_star > 1e-6, a_sel / np.maximum(a_star, 1e-8), np.nan)
    rank_ok = []
    for i in range(n):
        j = int(ax[i])
        best_ax = int(np.unravel_index(np.argmax(a5[i]), a5[i].shape)[0])
        rank_ok.append(j == best_ax)
    return {
        "n": int(n),
        "sign_acc": pair,
        "ranking_acc": float(np.mean(rank_ok)) if rank_ok else float("nan"),
        "mean_A": float(np.mean(a_sel)),
        "median_A": float(np.median(a_sel)),
        "P_Agt0": float((a_sel > 0).mean()),
        "P_Agt0_05": float((a_sel > 0.05).mean()),
        "oracle_ratio": float(np.nanmean(ratio)),
        "A_sel": a_sel,
    }


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{float(v):.{nd}f}"


def _rinfo(acc, acc_raw):
    if acc is None or acc_raw is None or abs(acc_raw - 0.5) < 1e-6:
        return None
    return (acc - 0.5) / (acc_raw - 0.5)


def _level_frac(name: str):
    if name == "raw":
        return None
    if name.startswith("f"):
        return float(name[1:])
    return None


def _blk(z, r_key, a_key, dip_key, dic_key, rd_key=None):
    re = z[r_key].astype(np.float64)
    a5 = z[a_key].astype(np.float64)
    pb = _pair_and_best(re, a5)
    dip = z[dip_key].astype(np.float64)
    dic = z[dic_key].astype(np.float64)
    out = {
        **{k: v for k, v in pb.items() if k != "A_sel"},
        "A": adv_block(pb["A_sel"]),
        "DI_probe": stats(dip.reshape(-1)),
        "DI_corr": stats(dic.reshape(-1)),
    }
    if rd_key and rd_key in z.files:
        rd = z[rd_key].astype(np.float64)
        out["Rd"] = stats(rd.reshape(-1))
        if f"dist_{rd_key.split('Rd_')[-1]}" in z.files or True:
            dk = "dist_" + rd_key.replace("Rd_", "")
            if dk in z.files:
                out["dist"] = stats(z[dk].astype(np.float64).reshape(-1))
    return out


def _decide(pareto_pc):
    """T1 small tube restores info; T2 only near raw; T3 non-monotonic."""
    rows = [r for r in pareto_pc if r["frac"] is not None]
    rows = sorted(rows, key=lambda r: r["frac"])
    accs = [r["sign_acc"] for r in rows if r["sign_acc"] is not None]
    raw = next((r for r in pareto_pc if r["level"] == "raw"), None)
    acc_raw = raw["sign_acc"] if raw else None
    rinfos = [r.get("R_info") for r in rows]
    # T3: not monotonically recovering
    mono = True
    if len(accs) >= 3:
        diffs = np.diff(accs)
        if float(np.sum(diffs < -0.03)) >= 2:
            mono = False
    hit = None
    for r in rows:
        ri = r.get("R_info")
        if ri is not None and ri >= 0.6 and r["frac"] <= 0.4:
            hit = r
            break
    strong = None
    for r in rows:
        ri = r.get("R_info")
        if ri is not None and ri >= 0.7 and r["frac"] <= 0.4:
            strong = r
            break
    if not mono:
        case = "T3"
    elif hit is not None:
        case = "T1"
    elif accs and accs[-1] >= (acc_raw - 0.03 if acc_raw else 0.62) and all((a or 0) <= 0.55 for a in accs[:-1] if a is not None):
        case = "T2"
    elif accs and max(accs) < 0.56:
        case = "T2"
    else:
        # partial recovery only at large frac
        early = [r for r in rows if r["frac"] <= 0.4]
        late_ok = rows[-1].get("R_info") is not None and rows[-1]["R_info"] >= 0.6
        early_bad = all((r.get("R_info") or 0) < 0.6 for r in early)
        case = "T2" if late_ok and early_bad else ("T1" if strong else "T2")
    return {
        "case": case,
        "run_p4a2": case in ("T2", "T3"),
        "R_info_ge_0.6_at": None if hit is None else hit["frac"],
        "strong_2cm": strong is not None,
        "monotonic": mono,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    args = ap.parse_args()
    root = Path(args.root)
    a1 = root / "p4a1_intent_tube"
    plots = a1 / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    by = {}
    for t in TERRAINS:
        p = a1 / "p4a1" / "loco" / t / "p4a1.npz"
        # launcher writes to root/p4a1/loco/...
        p2 = root / "p4a1" / "loco" / t / "p4a1.npz"
        path = p if p.exists() else p2
        if not path.exists():
            print(f"[p4a1] missing {path}", flush=True)
            continue
        by[t] = np.load(path, allow_pickle=True)
    if not by:
        raise FileNotFoundError("no p4a1.npz")

    sample = next(iter(by.values()))
    levels = [str(x) for x in sample["levels"].tolist()] if "levels" in sample.files else ["raw"]
    fracs = [float(x) for x in sample["fracs"].tolist()] if "fracs" in sample.files else []

    def cat(key):
        return np.concatenate([by[t][key] for t in TERRAINS if t in by and key in by[t].files], axis=0)

    pareto_rows = []
    per_t = {}
    for variant, a_from in (("P+C", "self"), ("P-only", "raw")):
        for t, z in by.items():
            per_t.setdefault(t, {})
            for name in levels:
                a_key = f"a5_{name}" if a_from == "self" else "a5_raw"
                if f"r_e_{name}" not in z.files:
                    continue
                b = _blk(z, f"r_e_{name}", a_key, f"di_probe_{name}", f"di_corr_{name}", f"Rd_{name}")
                per_t[t][f"{variant}:{name}"] = {k: v for k, v in b.items() if k != "A"}
        acc_raw = None
        for name in levels:
            if f"r_e_{name}" not in sample.files:
                continue
            a_key = f"a5_{name}" if a_from == "self" else "a5_raw"
            re = cat(f"r_e_{name}")
            a5 = cat(a_key)
            dip = cat(f"di_probe_{name}")
            dic = cat(f"di_corr_{name}")
            pb = _pair_and_best(re, a5)
            if name == "raw":
                acc_raw = pb["sign_acc"]
            frac = _level_frac(name)
            rinfo = _rinfo(pb["sign_acc"], acc_raw)
            rd_med = float(np.median(cat(f"Rd_{name}"))) if f"Rd_{name}" in sample.files else None
            dist_med = float(np.median(cat(f"dist_{name}"))) if f"dist_{name}" in sample.files else None
            row = {
                "variant": variant,
                "level": name,
                "frac": frac if frac is not None else (1.0 if name == "raw" else None),
                "n": pb["n"],
                "sign_acc": pb["sign_acc"],
                "ranking_acc": pb["ranking_acc"],
                "R_info": rinfo,
                "mean_A": pb["mean_A"],
                "median_A": pb["median_A"],
                "P_Agt0": pb["P_Agt0"],
                "P_Agt0_05": pb["P_Agt0_05"],
                "oracle_ratio": pb["oracle_ratio"],
                "DI_probe_median": float(np.median(dip)),
                "DI_probe_mean": float(np.mean(dip)),
                "DI_probe_p90": float(np.quantile(dip, 0.90)),
                "DI_probe_p95": float(np.quantile(dip, 0.95)),
                "DI_corr_median": float(np.median(dic)),
                "DI_corr_mean": float(np.mean(dic)),
                "DI_corr_p90": float(np.quantile(dic, 0.90)),
                "DI_corr_p95": float(np.quantile(dic, 0.95)),
                "Rd_median": rd_med,
                "dist_median": dist_med,
            }
            pareto_rows.append(row)

    # per-terrain sign acc for P+C
    terr_acc = {t: {} for t in by}
    for t, z in by.items():
        acc_r = _pair_and_best(z["r_e_raw"].astype(np.float64), z["a5_raw"].astype(np.float64))["sign_acc"]
        terr_acc[t]["raw"] = acc_r
        for name in levels:
            if f"r_e_{name}" not in z.files:
                continue
            terr_acc[t][name] = _pair_and_best(
                z[f"r_e_{name}"].astype(np.float64), z[f"a5_{name}"].astype(np.float64)
            )["sign_acc"]

    pc = [r for r in pareto_rows if r["variant"] == "P+C"]
    decision = _decide(pc)

    a1.mkdir(parents=True, exist_ok=True)
    with (a1 / "pareto.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pareto_rows[0].keys()))
        w.writeheader()
        w.writerows(pareto_rows)

    # plots
    def _xy(variant, xk, yk, skip_raw=False):
        xs, ys = [], []
        for r in pareto_rows:
            if r["variant"] != variant:
                continue
            if skip_raw and r["level"] == "raw":
                continue
            if r.get(xk) is None or r.get(yk) is None:
                continue
            if isinstance(r[yk], float) and not np.isfinite(r[yk]):
                continue
            xs.append(r[xk])
            ys.append(r[yk])
        return xs, ys

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    xs, ys = _xy("P+C", "frac", "sign_acc")
    ax.plot(xs, ys, "o-", color="#1f4e79", label="P+C")
    xs2, ys2 = _xy("P-only", "frac", "sign_acc")
    ax.plot(xs2, ys2, "s--", color="#c44e52", label="P-only")
    ax.axhline(0.5, color="0.5", ls=":", label="random")
    ax.set_xlabel("intent-tube fraction of 5 cm band")
    ax.set_ylabel("sign accuracy")
    ax.set_title("A1-1 Information vs intent tube")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "A1_1_info_vs_tube.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    xs, ys = _xy("P+C", "DI_probe_median", "R_info", skip_raw=True)
    ax.plot(xs, ys, "o-", color="#1f4e79")
    ax.axhline(0.6, color="0.5", ls=":")
    ax.set_xlabel("median DI_probe")
    ax.set_ylabel(r"$R_{info}$")
    ax.set_title("A1-2 Information retention vs probe DI")
    fig.tight_layout()
    fig.savefig(plots / "A1_2_Rinfo_vs_DIprobe.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    xs, y1 = _xy("P+C", "DI_corr_median", "mean_A")
    _, y2 = _xy("P+C", "DI_corr_median", "median_A")
    ax.plot(xs, y1, "o-", label="mean A")
    ax.plot(xs, y2, "s--", label="median A")
    ax.axhline(0.0, color="0.5", ls=":")
    ax.set_xlabel("median DI_corr")
    ax.set_ylabel("advantage A")
    ax.set_title("A1-3 Recovery authority vs correction DI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "A1_3_A_vs_DIcorr.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    xs, ys = _xy("P+C", "frac", "Rd_median", skip_raw=True)
    ax.plot(xs, ys, "o-", color="#2a9d8f")
    ax.set_xlabel("intent-tube fraction")
    ax.set_ylabel("median Rd")
    ax.set_title("A1-4 Direction retention")
    fig.tight_layout()
    fig.savefig(plots / "A1_4_Rd.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for t, col in zip(TERRAINS, ("#1f4e79", "#c44e52", "#2a9d8f")):
        if t not in terr_acc:
            continue
        xs, ys = [], []
        for name in levels:
            fr = 1.0 if name == "raw" else _level_frac(name)
            if fr is None or name not in terr_acc[t]:
                continue
            xs.append(fr)
            ys.append(terr_acc[t][name])
        ax.plot(xs, ys, "o-", color=col, label=t)
    ax.axhline(0.5, color="0.5", ls=":")
    ax.set_xlabel("intent-tube fraction")
    ax.set_ylabel("sign accuracy")
    ax.set_title("A1-5 Per-terrain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "A1_5_per_terrain.png", dpi=140)
    plt.close(fig)

    metrics = {
        "n_pooled": int(cat("e0").shape[0]),
        "levels": levels,
        "fracs": fracs,
        "pareto": pareto_rows,
        "per_terrain_sign_acc_Pc": terr_acc,
        "decision": decision,
        "no_sequential": True,
        "no_learned_shield": True,
    }
    (root / "metrics.json").write_text(json.dumps(sanitize(metrics), indent=2), encoding="utf-8")

    lines = [
        "# P4-A1 — Intent-Tube Active Probe Pareto",
        "",
        "Human target unchanged. Clone 1° / 5° labels. Oracle $C_I=J_I^\\top J_I$ (5 cm-normalized).",
        "Constraint: linearized 5° DI $\\le$ fraction of the 5 cm band. No sequential. No learned shield.",
        "",
        f"Pooled n={metrics['n_pooled']}. Case **{decision['case']}**. Run P4-A2: **{decision['run_p4a2']}**.",
        "",
        "## P+C (probe and correction both in tube)",
        "",
        "| frac | sign acc | R_info | mean A | P(A>0) | DI_probe med | DI_corr med | Rd |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pc:
        lines.append(
            f"| {_fmt(r['frac'] if r['level']!='raw' else 1.0, 2)}{' raw' if r['level']=='raw' else ''} | "
            f"{_fmt(r['sign_acc'])} | {_fmt(r['R_info'])} | {_fmt(r['mean_A'], 4)} | {_fmt(r['P_Agt0'])} | "
            f"{_fmt(r['DI_probe_median'], 4)} | {_fmt(r['DI_corr_median'], 4)} | {_fmt(r['Rd_median'])} |"
        )
    po = [r for r in pareto_rows if r["variant"] == "P-only"]
    lines += ["", "## P-only (tube probe, raw 5° correction)", "",
              "| frac | sign acc | R_info | mean A | P(A>0) |",
              "|---:|---:|---:|---:|---:|"]
    for r in po:
        lines.append(
            f"| {_fmt(r['frac'] if r['level']!='raw' else 1.0, 2)} | {_fmt(r['sign_acc'])} | "
            f"{_fmt(r['R_info'])} | {_fmt(r['mean_A'], 4)} | {_fmt(r['P_Agt0'])} |"
        )
    lines += [
        "",
        "## Decision questions",
        "",
        f"1. Smooth recovery: monotonic={decision['monotonic']}.",
        f"2. R_info≥0.6 first at frac={decision['R_info_ge_0.6_at']}.",
        f"3. ≤2 cm (frac 0.4) useful: {decision['strong_2cm']}.",
        "4. See P(A>0) vs frac in the table.",
        "5. Per-terrain plot A1-5.",
        f"6. Case **{decision['case']}** — T1=strict nullspace too strong; T2/T3=latent entanglement.",
        "",
    ]
    if decision["case"] == "T1":
        lines.append("STOP. Keep Stage-2 latent. Replace strict nullspace with a bounded intent tube. Do **not** run P4-A2.")
    else:
        lines.append("Proceed to P4-A2 autonomy-interface diagnostic. Do not return to projector training.")
    (a1 / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "MASTER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    a2 = root / "p4a2_interface"
    a2.mkdir(parents=True, exist_ok=True)
    if not decision["run_p4a2"]:
        (a2 / "NOT_RUN.md").write_text(
            "NOT RUN — latent interface retained after intent-tube diagnostic.\n", encoding="utf-8"
        )
    print(json.dumps(sanitize(decision), indent=2), flush=True)


if __name__ == "__main__":
    main()
