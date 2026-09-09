#!/usr/bin/env python3
"""P4-A0: oracle-safe vs raw clone probe. No learned shield. No sequential."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from p3_common import adv_block, sanitize, stats

TERRAINS = ("steps", "slope_down", "slip")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/home/chenxiangyu/robotics/Anybody/results/p4a0_oracle_safe_probe")
    args = ap.parse_args()
    root = Path(args.root)
    by = {}
    for t in TERRAINS:
        p = root / "p4a0" / "loco" / t / "p4a0.npz"
        if not p.exists():
            print(f"[p4a0] missing {p}", flush=True)
            continue
        z = np.load(p, allow_pickle=True)
        by[t] = z
    if not by:
        raise FileNotFoundError("no p4a0.npz")

    def cat(key):
        return np.concatenate([by[t][key] for t in TERRAINS if t in by], axis=0)

    metrics = {"per_terrain": {}, "pooled": {}}
    for t, z in by.items():
        blk = {}
        for sp in ("raw", "oracle_safe"):
            a5 = z[f"a5_{sp}"].astype(np.float64)
            re = z[f"r_e_{sp}"].astype(np.float64)
            di = z[f"di_{sp}"].astype(np.float64)
            blk[sp] = {
                **{k: v for k, v in _pair_and_best(re, a5).items() if k != "A_sel"},
                "DI_mean": float(di.mean()),
                "DI_median": float(np.median(di)),
                "DI_p90": float(np.quantile(di, 0.90)),
                "DI_peak": float(di.max()),
            }
        acc_r, acc_s = blk["raw"]["sign_acc"], blk["oracle_safe"]["sign_acc"]
        blk["R_info"] = (acc_s - 0.5) / (acc_r - 0.5) if acc_r is not None and acc_r != 0.5 else None
        blk["DI_ratio"] = (
            blk["oracle_safe"]["DI_median"] / blk["raw"]["DI_median"]
            if blk["raw"]["DI_median"]
            else None
        )
        metrics["per_terrain"][t] = blk

    pooled = {}
    for sp in ("raw", "oracle_safe"):
        a5 = cat(f"a5_{sp}")
        re = cat(f"r_e_{sp}")
        di = cat(f"di_{sp}")
        pb = _pair_and_best(re, a5)
        pooled[sp] = {
            **{k: v for k, v in pb.items() if k != "A_sel"},
            "DI_mean": float(di.mean()),
            "DI_median": float(np.median(di)),
            "DI_p90": float(np.quantile(di, 0.90)),
            "A": adv_block(pb["A_sel"]),
            "DI": stats(di.reshape(-1)),
        }
    acc_r, acc_s = pooled["raw"]["sign_acc"], pooled["oracle_safe"]["sign_acc"]
    rinfo = (acc_s - 0.5) / (acc_r - 0.5) if acc_r != 0.5 else None
    di_ratio = pooled["oracle_safe"]["DI_median"] / pooled["raw"]["DI_median"] if pooled["raw"]["DI_median"] else None
    if acc_s is not None and acc_s <= 0.55:
        case = "B"
    elif rinfo is not None and rinfo >= 0.7 and di_ratio is not None and di_ratio <= 1.0 / 3.0:
        case = "A"
    elif rinfo is not None and rinfo >= 0.7:
        case = "A_info_ok_DI_miss"
    else:
        case = "mixed"
    metrics["pooled"] = {
        **pooled,
        "R_info": rinfo,
        "DI_ratio_safe_over_raw": di_ratio,
        "case": case,
        "n": int(cat("e0").shape[0]),
    }
    (root / "metrics.json").write_text(json.dumps(sanitize(metrics), indent=2), encoding="utf-8")

    p = metrics["pooled"]
    lines = [
        "# P4-A0 — Oracle Intent-Safe Clone Probe",
        "",
        "RAW $B_{UCR}^{k=3}$ vs $QR(P_T P_I^{oracle} B_{UCR})$. Clone 1°, L=1, 5° labels.",
        "No learned B1/B21. No sequential. Trigger eval-only.",
        "",
        f"Pooled n={p['n']}. Case **{case}**.",
        "",
        "| space | sign acc | rank acc | mean A | median A | P(A>0) | DI med |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| raw | {_fmt(p['raw']['sign_acc'], 3)} | {_fmt(p['raw']['ranking_acc'], 3)} | "
        f"{_fmt(p['raw']['mean_A'], 4)} | {_fmt(p['raw']['median_A'], 4)} | {_fmt(p['raw']['P_Agt0'], 3)} | "
        f"{_fmt(p['raw']['DI_median'], 4)} |",
        f"| oracle-safe | {_fmt(p['oracle_safe']['sign_acc'], 3)} | {_fmt(p['oracle_safe']['ranking_acc'], 3)} | "
        f"{_fmt(p['oracle_safe']['mean_A'], 4)} | {_fmt(p['oracle_safe']['median_A'], 4)} | {_fmt(p['oracle_safe']['P_Agt0'], 3)} | "
        f"{_fmt(p['oracle_safe']['DI_median'], 4)} |",
        "",
        f"$R_{{info}}$ = **{_fmt(rinfo, 2)}**   DI_safe/DI_raw = **{_fmt(di_ratio, 2)}** (target ≤ 1/3)",
        "",
        "## Per terrain",
        "",
    ]
    for t, blk in metrics["per_terrain"].items():
        lines.append(
            f"- **{t}**: raw sign={_fmt(blk['raw']['sign_acc'], 3)} safe={_fmt(blk['oracle_safe']['sign_acc'], 3)} "
            f"R_info={_fmt(blk.get('R_info'), 2)} DI_ratio={_fmt(blk.get('DI_ratio'), 2)}"
        )
    lines += ["", "## Decision", ""]
    if case == "A" or case == "A_info_ok_DI_miss":
        lines.append("Oracle-safe probe retains directional information. Intent preservation and active identification are **not** in hard conflict.")
        if case == "A":
            lines.append("DI also dropped ≥3×. Proceed to P4-B drift-robust sequential estimator.")
        else:
            lines.append("Information retained but probe DI did not drop 3×. Still do not train P4-B until DI target is understood; sequential estimator is allowed only if net A including probe cost is positive.")
    elif case == "B":
        lines.append("**Case B:** oracle shield collapses sign accuracy to near-random.")
        lines.append("Informative physical directions conflict with intent-safe directions.")
        lines.append("Do **not** start P4-B. Next method: Intent-Constrained Information Acquisition.")
    else:
        lines.append("Mixed: report R_info and DI ratio; do not auto-start P4-B.")
    lines.append("")
    lines.append("No learned shield. No P4-C.")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"case": case, "R_info": rinfo, "DI_ratio": di_ratio, "acc_raw": acc_r, "acc_safe": acc_s}, indent=2), flush=True)


if __name__ == "__main__":
    main()
