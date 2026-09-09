#!/usr/bin/env python3
"""ICR plots A–E. Plot A is produced from M0; B–E appear once M1–M3 exist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TERRAINS = ("plane", "slope", "slope_down", "light_rough", "steps", "slip")
COLORS = {
    "plane": "#4C78A8",
    "slope": "#F58518",
    "slope_down": "#E45756",
    "light_rough": "#72B7B2",
    "steps": "#54A24B",
    "slip": "#B279A2",
}
LABEL = {
    "plane": "flat",
    "slope": "slope+",
    "slope_down": "slope-",
    "light_rough": "rough",
    "steps": "local height",
    "slip": "friction",
}


def _mean_curve(root: Path, method: str, task: str, terrain: str):
    p = root / method / task / terrain / "series.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=True)
    e = z["e"].astype(np.float64)
    return e.mean(axis=0), np.percentile(e, 25, axis=0), np.percentile(e, 75, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/home/chenxiangyu/robotics/Anybody/results/icr_interaction_recovery")
    ap.add_argument("--task", default="loco")
    args = ap.parse_args()
    root = Path(args.root)
    fig_dir = root / "plots"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[icr-plot] matplotlib unavailable: {exc}", flush=True)
        return

    dt = 0.02
    # Plot A: same reference, different terrains, tracking error vs time (M0)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    n_drawn = 0
    for terr in TERRAINS:
        cur = _mean_curve(root, "m0", args.task, terr)
        if cur is None:
            continue
        mean, lo, hi = cur
        t = np.arange(mean.shape[0]) * dt
        ax.plot(t, mean, color=COLORS[terr], lw=1.8, label=LABEL[terr])
        ax.fill_between(t, lo, hi, color=COLORS[terr], alpha=0.12)
        n_drawn += 1
    ax.set_xlabel("time (s)")
    ax.set_ylabel("visible keypoint error (m)")
    ax.set_title(f"Plot A — frozen Stage-2, same {args.task} clips × terrains")
    ax.legend(frameon=False, ncol=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_a = fig_dir / f"plot_a_m0_{args.task}_error.png"
    fig.savefig(out_a, dpi=160)
    plt.close(fig)
    print(f"[icr-plot] A n={n_drawn} -> {out_a}", flush=True)

    # Plot B: ||Δz|| (M1–M3)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    n_b = 0
    for method in ("m1", "m2", "m3"):
        for terr in TERRAINS:
            p = root / method / args.task / terr / "series.npz"
            if not p.is_file():
                continue
            z = np.load(p)
            dz = z["dz"].astype(np.float64).mean(axis=0)
            t = np.arange(dz.shape[0]) * dt
            ax.plot(t, dz, color=COLORS[terr], lw=1.4, alpha=0.85, label=f"{method} {LABEL[terr]}")
            n_b += 1
    if n_b:
        ax.set_xlabel("time (s)")
        ax.set_ylabel(r"$\|\Delta z\|$")
        ax.set_title(f"Plot B — residual magnitude, same {args.task} clips")
        ax.legend(frameon=False, fontsize=8, ncol=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        out_b = fig_dir / f"plot_b_{args.task}_dz.png"
        fig.savefig(out_b, dpi=160)
        print(f"[icr-plot] B n={n_b} -> {out_b}", flush=True)
    plt.close(fig)

    # Plot D: recovery curves M0 vs M1 vs M2 vs M3 on steps (hardest typical)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    n_d = 0
    styles = {"m0": "-", "m1": "--", "m2": "-.", "m3": ":"}
    names = {"m0": "M0 Stage2", "m1": "M1 instant", "m2": "M2 history", "m3": "M3 tangent"}
    for method in ("m0", "m1", "m2", "m3"):
        cur = _mean_curve(root, method, args.task, "steps")
        if cur is None:
            continue
        mean, lo, hi = cur
        t = np.arange(mean.shape[0]) * dt
        ax.plot(t, mean, styles[method], color="#333333", lw=1.8, label=names[method])
        n_d += 1
    if n_d:
        ax.set_xlabel("time (s)")
        ax.set_ylabel("visible keypoint error (m)")
        ax.set_title("Plot D — steps, Stage2 vs residuals")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        out_d = fig_dir / f"plot_d_{args.task}_steps.png"
        fig.savefig(out_d, dpi=160)
        print(f"[icr-plot] D n={n_d} -> {out_d}", flush=True)
    plt.close(fig)
    (fig_dir / "index.json").write_text(json.dumps({"plot_a": str(out_a), "n_a": n_drawn}, indent=2))


if __name__ == "__main__":
    main()
