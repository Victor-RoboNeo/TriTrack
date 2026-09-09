#!/usr/bin/env python3
"""IRR R3-short tables. Terrain is an eval slice only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DT_MS = 20.0
PROBE_LS = (1, 2, 3, 5)


def _pm(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100.0 * float(x):.1f}%"


def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def _pair_and_best(r_e: np.ndarray, a5: np.ndarray) -> dict:
    """r_e [n, n_dir], a5 [n, n_dir, 2] with [...,0]=+5°."""
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
    ratio = np.divide(a_sel, np.maximum(a_star, 1e-8), where=a_star > 1e-6)
    ratio = np.where(a_star > 1e-6, ratio, np.nan)
    return {
        "pair_sign": pair,
        "mean_A": float(np.mean(a_sel)),
        "p_pos": float((a_sel > 0).mean()),
        "oracle_ratio": float(np.nanmean(ratio)),
        "mean_Astar": float(np.mean(a_star)),
    }


def _from_a1(a1: np.ndarray, a5: np.ndarray) -> dict:
    """L=10 clone: use 1° long-horizon A as the probe."""
    r_proxy = -(a1[:, :, 0] - a1[:, :, 1])  # like r_e: negative ⇒ + is better
    return _pair_and_best(r_proxy, a5)


def _one_npz(p: Path) -> dict:
    z = np.load(p, allow_pickle=True)
    a5 = z["a5"].astype(np.float64)
    a1 = z["a1"].astype(np.float64)
    Ls = [int(x) for x in z["probe_L"].tolist()] if "probe_L" in z.files else list(PROBE_LS)
    out = {"n": int(a5.shape[0]), "terrain": str(z["terrain"][0]), "L": {}}
    out["L"][10] = {
        "clone": _from_a1(a1, a5),
        "seq": None,
        "cost_clone": int(a5.shape[1] * 2 * 10 * DT_MS),
        "n_dir": int(a5.shape[1]),
    }
    for li, L in enumerate(Ls):
        r_c = z["r_clone"][:, li, :, 0].astype(np.float64)
        r_s = z["r_seq"][:, li, :, 0].astype(np.float64)
        nd = int(r_c.shape[1])
        out["L"][int(L)] = {
            "clone": _pair_and_best(r_c, a5),
            "seq": _pair_and_best(r_s, a5),
            "cost_clone": int(nd * 2 * int(L) * DT_MS),
            "n_dir": nd,
        }
        if "aB" in z.files and "rB_clone" in z.files:
            aB = z["aB"].astype(np.float64)
            k_max = int(aB.shape[1])
            out["L"][int(L)]["B"] = {}
            for k in (2, 3, 4):
                if k > k_max:
                    continue
                rb_c = z["rB_clone"][:, li, :k, 0].astype(np.float64)
                rb_s = z["rB_seq"][:, li, :k, 0].astype(np.float64)
                blk = {
                    "clone": _pair_and_best(rb_c, aB[:, :k]),
                    "seq": _pair_and_best(rb_s, aB[:, :k]),
                    "cost": int(k * 2 * int(L) * DT_MS),
                }
                kk = (2, 3, 4).index(k)
                if "a_jac_clone" in z.files:
                    ac = z["a_jac_clone"][:, li, kk].astype(np.float64)
                    astar = np.maximum(a5.max(axis=(1, 2)), 0.0)
                    ratio = np.where(astar > 1e-6, ac / np.maximum(astar, 1e-8), np.nan)
                    blk["jac_clone"] = {
                        "mean_A": float(np.mean(ac)),
                        "p_pos": float((ac > 0).mean()),
                        "oracle_ratio": float(np.nanmean(ratio)),
                    }
                    asq = z["a_jac_seq"][:, li, kk].astype(np.float64)
                    ratio_s = np.where(astar > 1e-6, asq / np.maximum(astar, 1e-8), np.nan)
                    blk["jac_seq"] = {
                        "mean_A": float(np.mean(asq)),
                        "p_pos": float((asq > 0).mean()),
                        "oracle_ratio": float(np.nanmean(ratio_s)),
                    }
                out["L"][int(L)]["B"][k] = blk
        if "a_spsa_clone" in z.files:
            astar = np.maximum(a5.max(axis=(1, 2)), 0.0)
            ac = z["a_spsa_clone"][:, li].astype(np.float64)
            asq = z["a_spsa_seq"][:, li].astype(np.float64)
            out["L"][int(L)]["spsa"] = {
                "clone": {
                    "mean_A": float(np.mean(ac)),
                    "p_pos": float((ac > 0).mean()),
                    "oracle_ratio": float(np.nanmean(np.where(astar > 1e-6, ac / np.maximum(astar, 1e-8), np.nan))),
                },
                "seq": {
                    "mean_A": float(np.mean(asq)),
                    "p_pos": float((asq > 0).mean()),
                    "oracle_ratio": float(np.nanmean(np.where(astar > 1e-6, asq / np.maximum(astar, 1e-8), np.nan))),
                },
                "cost": int(2 * int(L) * DT_MS),
            }
    return out


def _row(name, L, nd, pair, mean_a, ratio, cost, mode=""):
    tag = f"{name}" + (f" {mode}" if mode else "")
    return (
        f"| {tag} | {L} | {nd} | {_pm(pair)} | {_f(mean_a)} | {_f(ratio)} | {cost} |\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/home/chenxiangyu/robotics/Anybody/results/irr_response_recovery")
    args = ap.parse_args()
    root = Path(args.root)
    md = [
        "# IRR R3-short — Active Interaction-Conditioned Recovery\n\n",
        "Trigger `e≥0.13` persist-3 is **evaluation-only**, not a deployment gate. "
        "No terrain labels enter the probe or the Jacobian.\n\n",
        "Question: can a 1–2 step ±ε pulse identify a useful 5° recovery direction?\n\n",
    ]
    cells = sorted((root / "r3_short").glob("*/*/r3.npz"))
    if not cells:
        md.append("No `r3_short/**/r3.npz` yet.\n")
        (root / "R3_REPORT.md").write_text("".join(md))
        print("".join(md), flush=True)
        return
    md.append("| Probe | Horizon | Directions | Sign acc | mean A | Oracle ratio | cost (ms) |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    pooled = []
    by_t = {}
    for p in cells:
        d = _one_npz(p)
        pooled.append(d)
        by_t[d["terrain"]] = d
        t = d["terrain"]
        for L, blk in sorted(d["L"].items()):
            nd = blk.get("n_dir", "—")
            if blk.get("clone"):
                c = blk["clone"]
                md.append(_row(f"{t} 15-ax clone", L, nd, c["pair_sign"], c["mean_A"], c["oracle_ratio"], blk.get("cost_clone", "—")))
            if blk.get("seq"):
                s = blk["seq"]
                md.append(_row(f"{t} 15-ax seq", L, nd, s["pair_sign"], s["mean_A"], s["oracle_ratio"], blk.get("cost_clone", "—")))
            for k, bb in (blk.get("B") or {}).items():
                md.append(
                    _row(
                        f"{t} B k={k} clone", L, k,
                        bb["clone"]["pair_sign"], bb["clone"]["mean_A"], bb["clone"]["oracle_ratio"],
                        bb["cost"],
                    )
                )
                md.append(
                    _row(
                        f"{t} B k={k} seq", L, k,
                        bb["seq"]["pair_sign"], bb["seq"]["mean_A"], bb["seq"]["oracle_ratio"],
                        bb["cost"],
                    )
                )
                if bb.get("jac_clone"):
                    jc = bb["jac_clone"]
                    md.append(_row(f"{t} Jac k={k} clone", L, k, None, jc["mean_A"], jc["oracle_ratio"], bb["cost"]))
                if bb.get("jac_seq"):
                    js = bb["jac_seq"]
                    md.append(_row(f"{t} Jac k={k} seq", L, k, None, js["mean_A"], js["oracle_ratio"], bb["cost"]))
            if blk.get("spsa"):
                sp = blk["spsa"]
                md.append(_row(f"{t} SPSA clone", L, 1, None, sp["clone"]["mean_A"], sp["clone"]["oracle_ratio"], sp["cost"]))
                md.append(_row(f"{t} SPSA seq", L, 1, None, sp["seq"]["mean_A"], sp["seq"]["oracle_ratio"], sp["cost"]))
    md.append(
        "\n`cost` is probe time only at dt=20 ms, not the subsequent 5° evaluation rollout.\n"
        "Sequential = +ε then −ε on one trajectory (state drift). Clone = restore between signs.\n"
    )
    (root / "R3_REPORT.md").write_text("".join(md))
    print("".join(md), flush=True)


if __name__ == "__main__":
    main()
