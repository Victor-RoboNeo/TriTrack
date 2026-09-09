#!/usr/bin/env python3
"""R-M3 plots + pooled summary. No Isaac. Frozen eval artifacts only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TERRAINS = ("plane", "light_rough", "slope", "steps")
TERRAIN_LABEL = {
    "plane": "Flat",
    "light_rough": "Light",
    "slope": "Slope",
    "steps": "Steps",
}
VARIANTS = ("parent", "loco6s", "shared5", "adaptive")
VARIANT_LABEL = {
    "parent": "Parent",
    "loco6s": "Loco6S",
    "shared5": "Shared5",
    "adaptive": "Adaptive",
}
TASKS = ("loco", "stoop")
THETA_BINS = (2.5, 5.0, 7.5, 10.0)
METHODS_I = ("loco6s", "shared5", "adaptive")
METHOD_I_KEY = {
    "loco6s": "I_loco6s_cm",
    "shared5": "I_shared5_cm",
    "adaptive": "I_adaptive_cm",
}


def _load(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _pct(x, qs=(25, 50, 75)):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{q}": None for q in qs} | {"n": 0, "mean": None}
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} | {"n": int(x.size), "mean": float(x.mean())}


def _rate(mask):
    m = np.asarray(mask, dtype=bool)
    return float(m.mean()) if m.size else None


def _i_stats(vals):
    x = np.asarray(vals, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "P_I_lt_0": None, "P_I_lt_0.5cm": None, "P_I_lt_1cm": None,
                "median_cm": None, "p25_cm": None, "p75_cm": None}
    return {
        "n": int(x.size),
        "P_I_lt_0": _rate(x < 0),
        "P_I_lt_0.5cm": _rate(x < -0.5),
        "P_I_lt_1cm": _rate(x < -1.0),
        "median_cm": float(np.median(x)),
        "p25_cm": float(np.percentile(x, 25)),
        "p75_cm": float(np.percentile(x, 75)),
        "mean_cm": float(x.mean()),
        "vals": [float(v) for v in x],
    }


def collect(root: Path) -> dict:
    live = {}
    for task in TASKS:
        live[task] = {}
        for var in VARIANTS:
            cells = {}
            pooled = _load(root / task / var / "pooled.json")
            for ter in TERRAINS:
                cells[ter] = _load(root / task / var / ter / "summary.json")
            live[task][var] = {"pooled": pooled, "cells": cells}
    paired = {}
    for task in TASKS:
        paired[task] = {}
        rows = []
        for ter in TERRAINS:
            blob = _load(root / "paired_clone" / task / ter / "i_live.json")
            paired[task][ter] = blob
            if blob and blob.get("rows"):
                rows.extend(blob["rows"])
        paired[task]["rows"] = rows
    return {"live": live, "paired": paired}


def sr_table(data: dict) -> dict:
    out = {}
    for task in TASKS:
        out[task] = {}
        for ter in TERRAINS:
            row = {}
            for var in VARIANTS:
                cell = data["live"][task][var]["cells"].get(ter)
                row[var] = None if cell is None else _finite(cell.get("sr_5cm"))
            out[task][ter] = row
        all_row = {}
        for var in VARIANTS:
            vals = [out[task][ter][var] for ter in TERRAINS if out[task][ter][var] is not None]
            all_row[var] = float(np.mean(vals)) if vals else None
        out[task]["all"] = all_row
    return out


def stoop_sfirst(data: dict) -> dict:
    out = {}
    for var in VARIANTS:
        events = []
        for ter in TERRAINS:
            cell = data["live"]["stoop"][var]["cells"].get(ter)
            if not cell:
                continue
            events.extend([e for e in (cell.get("events") or []) if e.get("trigger_channel") == "S"])
        if not events:
            out[var] = {"n": 0}
            continue
        auc = [e.get("auc") for e in events if e.get("auc") is not None]
        vis = [e.get("median_vis_e") for e in events if e.get("median_vis_e") is not None]
        eps_keys = {(e.get("seed"), e.get("env"), e.get("terrain")) for e in events}
        complete = [not bool(e.get("ep_fail")) for e in events]
        out[var] = {
            "n": len(events),
            "n_unique_eps_approx": len(eps_keys),
            "fall_05": _rate([bool(e.get("fall_05")) for e in events]),
            "fall_10": _rate([bool(e.get("fall_10")) for e in events]),
            "fall_20": _rate([bool(e.get("fall_20")) for e in events]),
            "task_completion": _rate(complete),
            "visible_error_AUC": _pct(auc),
            "median_visible_kp_error": _pct(vis),
            "fail_after_trigger": _rate([bool(e.get("fail_after")) for e in events]),
        }
    return out


def paired_table(data: dict) -> dict:
    out = {}
    for task, groups in (
        ("loco", {"all": lambda r: True}),
        ("stoop", {"S": lambda r: r.get("trigger_channel") == "S",
                   "E": lambda r: r.get("trigger_channel") == "E",
                   "both": lambda r: r.get("trigger_channel") == "both",
                   "all": lambda r: True}),
    ):
        rows = data["paired"][task]["rows"]
        out[task] = {}
        for gname, pred in groups.items():
            rs = [r for r in rows if pred(r)]
            block = {"n": len(rs)}
            for m in METHODS_I:
                block[m] = _i_stats([r[METHOD_I_KEY[m]] for r in rs])
                block[m].pop("vals", None)
            thetas = [r.get("theta_adaptive") for r in rs]
            block["theta_adaptive"] = {
                "hist": {str(th): int(sum(abs(float(v) - th) < 1e-6 for v in thetas if v is not None)) for th in THETA_BINS},
                "mean": float(np.mean([float(v) for v in thetas if v is not None])) if any(v is not None for v in thetas) else None,
            }
            out[task][gname] = block
    return out


def theta_live(data: dict) -> dict:
    out = {}
    for task in TASKS:
        events = []
        for ter in TERRAINS:
            cell = data["live"][task]["adaptive"]["cells"].get(ter)
            if cell:
                events.extend(cell.get("events") or [])
        by = {"all": events, "S": [], "E": [], "both": []}
        for e in events:
            ch = e.get("trigger_channel")
            if ch in by:
                by[ch].append(e)
        out[task] = {}
        for g, evs in by.items():
            vals = []
            for e in evs:
                for b in e.get("bursts") or []:
                    if b.get("theta_deg") is not None:
                        vals.append(float(b["theta_deg"]))
            out[task][g] = {
                "n": len(vals),
                "hist": {str(th): int(sum(abs(v - th) < 1e-6 for v in vals)) for th in THETA_BINS},
                "mean": float(np.mean(vals)) if vals else None,
            }
    return out


def e_windows(data: dict) -> dict:
    """Mean E around trigger, survivor-at-t coverage."""
    out = {}
    t_rel = np.arange(-25, 51)
    for task in TASKS:
        out[task] = {}
        for var in VARIANTS:
            buckets = {int(t): [] for t in t_rel}
            n_cov = {int(t): 0 for t in t_rel}
            n_ev = 0
            for ter in TERRAINS:
                cell = data["live"][task][var]["cells"].get(ter)
                if not cell:
                    continue
                for e in cell.get("events") or []:
                    win = e.get("e_win")
                    lo = e.get("e_win_lo")
                    t0 = e.get("e_win_t0")
                    if win is None or lo is None or t0 is None:
                        continue
                    n_ev += 1
                    for k, val in enumerate(win):
                        abs_t = int(lo) + k
                        rel = abs_t - int(t0)
                        if rel in buckets:
                            buckets[rel].append(float(val) * 100.0)
                            n_cov[rel] += 1
            mean = []
            cov = []
            for t in t_rel:
                xs = buckets[int(t)]
                mean.append(float(np.mean(xs)) if xs else float("nan"))
                cov.append((n_cov[int(t)] / n_ev) if n_ev else 0.0)
            out[task][var] = {
                "t_rel": [int(t) for t in t_rel],
                "mean_E_cm": mean,
                "coverage": cov,
                "n_events": n_ev,
            }
    return out


def save_plots(root: Path, data: dict, sr: dict, paired: dict, th_live: dict, wins: dict) -> None:
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    colors = {
        "parent": "#4a4a4a",
        "loco6s": "#3b6ea5",
        "shared5": "#2a9d8f",
        "adaptive": "#c45c26",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    x = np.arange(len(TERRAINS))
    w = 0.18
    for ax, task, title in zip(axes, TASKS, ("Loco frozen SR@5", "Stoop frozen SR@5")):
        for i, var in enumerate(VARIANTS):
            ys = [sr[task][ter][var] if sr[task][ter][var] is not None else np.nan for ter in TERRAINS]
            ax.bar(x + (i - 1.5) * w, ys, w, label=VARIANT_LABEL[var], color=colors[var])
        ax.set_xticks(x)
        ax.set_xticklabels([TERRAIN_LABEL[t] for t in TERRAINS])
        ax.set_title(title)
        ax.set_ylabel("SR@5")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8)
        ax.axhline(0, color="#ddd", lw=0.5)
    fig.tight_layout()
    fig.savefig(plots / "sr5_by_terrain.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    sfirst = stoop_sfirst(data)
    x = np.arange(3)
    w = 0.18
    keys = ("fall_05", "fall_10", "fall_20")
    for i, var in enumerate(VARIANTS):
        ys = [sfirst.get(var, {}).get(k) or np.nan for k in keys]
        ax.bar(x + (i - 1.5) * w, ys, w, label=VARIANT_LABEL[var], color=colors[var])
    ax.set_xticks(x)
    ax.set_xticklabels(["fall@0.5s", "fall@1.0s", "fall@2.0s"])
    ax.set_ylabel("Post-trigger fall probability")
    ax.set_title("Stoop S-first post-trigger fall")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "stoop_sfirst_fall.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True)
    panels = [
        ("Loco I_live @0.5s", data["paired"]["loco"]["rows"], lambda r: True),
        ("Stoop-S I_live @0.5s", data["paired"]["stoop"]["rows"], lambda r: r.get("trigger_channel") == "S"),
        ("Stoop-E I_live @0.5s", data["paired"]["stoop"]["rows"], lambda r: r.get("trigger_channel") == "E"),
    ]
    for ax, (title, rows, pred) in zip(axes, panels):
        rs = [r for r in rows if pred(r)]
        series = []
        labels = []
        for m, lab in (("loco6s", "Loco6S"), ("shared5", "Shared5"), ("adaptive", "Adaptive")):
            vals = [r[METHOD_I_KEY[m]] for r in rs]
            series.append(vals)
            labels.append(lab)
        if any(len(s) for s in series):
            bp = ax.boxplot(series, labels=labels, showfliers=False, patch_artist=True)
            for patch, var in zip(bp["boxes"], ("loco6s", "shared5", "adaptive")):
                patch.set_facecolor(colors[var])
                patch.set_alpha(0.7)
        ax.axhline(0, color="#666", lw=0.8)
        ax.set_title(f"{title}\nn={len(rs)}")
        ax.set_ylabel("I_live (cm)")
    fig.tight_layout()
    fig.savefig(plots / "paired_i_live.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(3)
    w = 0.22
    groups = [("loco", "all", "Loco"), ("stoop", "S", "Stoop-S"), ("stoop", "E", "Stoop-E")]
    for i, m in enumerate(METHODS_I):
        ys = []
        for task, g, _lab in groups:
            ys.append(paired.get(task, {}).get(g, {}).get(m, {}).get("P_I_lt_0") or np.nan)
        ax.bar(x + (i - 1) * w, ys, w, label={"loco6s": "Loco6S", "shared5": "Shared5", "adaptive": "Adaptive"}[m],
               color=colors[m])
    ax.set_xticks(x)
    ax.set_xticklabels([g[2] for g in groups])
    ax.set_ylabel("P(I_live < 0)")
    ax.set_title("Paired live P(I<0) by method × task")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "paired_p_i_lt0.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(THETA_BINS))
    w = 0.22
    series = [
        ("Loco", th_live.get("loco", {}).get("all", {})),
        ("Stoop-S", th_live.get("stoop", {}).get("S", {})),
        ("Stoop-E", th_live.get("stoop", {}).get("E", {})),
    ]
    cols = ["#3b6ea5", "#c45c26", "#2a9d8f"]
    for i, (lab, blob) in enumerate(series):
        hist = blob.get("hist") or {}
        n = max(int(blob.get("n") or 0), 1)
        ys = [hist.get(str(th), 0) / n for th in THETA_BINS]
        ax.bar(x + (i - 1) * w, ys, w, label=f"{lab} n={blob.get('n', 0)}", color=cols[i])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{th:g}°" for th in THETA_BINS])
    ax.set_ylabel("Fraction of live bursts")
    ax.set_title("Adaptive magnitude histogram (live)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "adaptive_theta_hist.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    t = np.arange(-25, 51) * 0.02
    for ax, task, title in zip(axes, TASKS, ("Loco E around trigger", "Stoop E around trigger")):
        for var in VARIANTS:
            wblob = wins.get(task, {}).get(var) or {}
            y = np.asarray(wblob.get("mean_E_cm") or [], dtype=np.float64)
            cov = np.asarray(wblob.get("coverage") or [], dtype=np.float64)
            if y.size != t.size:
                continue
            ax.plot(t, y, label=f"{VARIANT_LABEL[var]} n={wblob.get('n_events', 0)}", color=colors[var], lw=1.6)
            ax.plot(t, np.where(cov >= 0.5, y, np.nan), color=colors[var], lw=2.4)
        ax.axvline(0, color="#888", lw=0.8, ls="--")
        ax.axvline(0.1, color="#aaa", lw=0.6, ls=":")
        ax.set_title(title)
        ax.set_xlabel("t − t0 (s)")
        ax.set_ylabel("Mean visible E (cm)")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Solid thick = survivor coverage ≥ 50%", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots / "e_trajectory_around_trigger.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/data/home/chenxiangyu/robotics/Anybody/results/rm3_shared_single_burst")
    args = ap.parse_args()
    root = Path(args.root)
    data = collect(root)
    sr = sr_table(data)
    sfirst = stoop_sfirst(data)
    paired = paired_table(data)
    th_live = theta_live(data)
    wins = e_windows(data)
    save_plots(root, data, sr, paired, th_live, wins)
    summary = {
        "sr5": sr,
        "stoop_sfirst": sfirst,
        "paired_i_live": paired,
        "theta_live_adaptive": th_live,
        "e_windows_n": {task: {var: wins[task][var]["n_events"] for var in VARIANTS} for task in TASKS},
    }
    (root / "summary_matrix.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"[rm3-plot] wrote {root / 'plots'} and {root / 'summary_matrix.json'}")


if __name__ == "__main__":
    main()
