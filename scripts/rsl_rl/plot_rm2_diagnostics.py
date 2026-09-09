#!/usr/bin/env python3
"""Phase R-M2 diagnostic plots. No PPO. Reads merged data + train metrics + cloned utility."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib required: {exc}") from exc

THETA_BINS = (2.5, 5.0, 7.5, 10.0)
TERRAIN_LABEL = {"plane": "Flat", "light_rough": "Light", "slope": "Slope", "steps": "Steps"}


def _load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _pca2(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/data/home/chenxiangyu/robotics/Anybody/results/rm_intent_conditioned_recovery")
    args = ap.parse_args()
    root = Path(args.root)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    merged = root / "data" / "merged"
    all_rows = _load_json(merged / "all.json") or []
    test_rows = _load_json(merged / "test.json") or []
    train_m = _load_json(root / "checkpoints" / "metrics.json") or {}
    util = _load_json(root / "eval" / "cloned_utility" / "summary.json") or {}

    loco = [r for r in all_rows if r["task_source"] == "loco"]
    stoop = [r for r in all_rows if r["task_source"] == "stoop"]
    sfirst = [r for r in stoop if r.get("trigger_channel") == "S"]
    efirst = [r for r in stoop if r.get("trigger_channel") == "E"]

    # 1. Loco vs Stoop oracle direction cosine (subsample pairwise)
    if loco and stoop:
        rng = np.random.default_rng(0)
        dl = np.stack([r["d_oracle"] for r in loco], 0)
        ds = np.stack([r["d_oracle"] for r in stoop], 0)
        dl = dl / np.clip(np.linalg.norm(dl, axis=1, keepdims=True), 1e-8, None)
        ds = ds / np.clip(np.linalg.norm(ds, axis=1, keepdims=True), 1e-8, None)
        n = min(80, len(dl), len(ds))
        il = rng.choice(len(dl), n, replace=False)
        js = rng.choice(len(ds), n, replace=False)
        mat = dl[il] @ ds[js].T
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_title("Oracle d* cosine: Loco rows vs Stoop cols")
        ax.set_xlabel("Stoop states (subsample)")
        ax.set_ylabel("Loco states (subsample)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(plots / "01_oracle_dir_cosine_matrix.png", dpi=140)
        plt.close(fig)
        (plots / "01_oracle_dir_cosine_stats.json").write_text(
            json.dumps({"mean_pairwise": float(mat.mean()), "median_pairwise": float(np.median(mat))}, indent=2)
        )

    # 2. best-angle histogram
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xs = np.arange(len(THETA_BINS))
    w = 0.25
    for i, (name, rs) in enumerate((("Loco", loco), ("Stoop S-first", sfirst), ("Stoop E-first", efirst))):
        c = Counter(r.get("best_angle") for r in rs)
        h = [c.get(th, 0) / max(len(rs), 1) * 100.0 for th in THETA_BINS]
        ax.bar(xs + (i - 1) * w, h, width=w, label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{th:g}°" for th in THETA_BINS])
    ax.set_ylabel("Share of states (%)")
    ax.set_xlabel("Oracle best angle")
    ax.set_title("Oracle best-angle histogram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "02_oracle_best_angle_hist.png", dpi=140)
    plt.close(fig)

    # 3–5 cloned utility if present
    metrics = (util.get("metrics") or {}) if util else {}
    if metrics.get("loco") or metrics.get("stoop_s_first"):
        methods = ["parent", "loco_6s", "shared_5deg", "shared_adapt", "oracle"]
        labels = ["Parent", "Loco 6S", "Shared 5°", "Shared adaptive", "Oracle"]
        splits = [("loco", "Loco"), ("stoop_s_first", "Stoop-S"), ("stoop_e_first", "Stoop-E")]
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        xs = np.arange(len(splits))
        w = 0.15
        for j, (mk, lab) in enumerate(zip(methods, labels)):
            vals = []
            for sk, _ in splits:
                blk = (metrics.get(sk) or {}).get(mk) or {}
                vals.append(blk.get("median"))
            ax.bar(xs + (j - 2) * w, [0 if v is None else v for v in vals], width=w, label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels([n for _, n in splits])
        ax.set_ylabel("median I (cm)")
        ax.set_title("Cloned-state median I by method")
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.legend(ncols=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(plots / "03_cloned_median_I.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        for j, (mk, lab) in enumerate(zip(methods, labels)):
            vals = []
            for sk, _ in splits:
                blk = (metrics.get(sk) or {}).get(mk) or {}
                v = blk.get("P_I_lt_0")
                vals.append(None if v is None else 100.0 * v)
            ax.bar(xs + (j - 2) * w, [0 if v is None else v for v in vals], width=w, label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels([n for _, n in splits])
        ax.set_ylabel("P(I < 0) (%)")
        ax.set_title("Cloned-state P(I<0) by method")
        ax.legend(ncols=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(plots / "04_cloned_P_I_lt_0.png", dpi=140)
        plt.close(fig)

        # 6. cosine vs utility scatter
        rows = util.get("rows") or []
        if rows:
            fig, ax = plt.subplots(figsize=(5.4, 4.2))
            for task, c, lab in (("loco", "C0", "Loco"), ("stoop", "C1", "Stoop")):
                sub = [r for r in rows if r.get("task_source") == task]
                if not sub:
                    continue
                ax.scatter(
                    [r.get("cos_shared_oracle") for r in sub],
                    [r.get("I_shared_adapt_cm") for r in sub],
                    s=18, alpha=0.7, label=lab,
                )
            ax.axhline(0, color="0.5", lw=0.8)
            ax.axvline(0, color="0.5", lw=0.8)
            ax.set_xlabel("cos(d_shared, d_oracle)")
            ax.set_ylabel("I_shared_adapt (cm)")
            ax.set_title("Direction cosine vs cloned utility")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots / "06_cosine_vs_utility.png", dpi=140)
            plt.close(fig)

    # 7. z_nom PCA
    if all_rows:
        z = np.stack([r["z_nom"] for r in all_rows], 0)
        xy = _pca2(z)
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        for task, lab in (("loco", "Loco"), ("stoop", "Stoop")):
            m = np.array([r["task_source"] == task for r in all_rows])
            ax.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.7, label=lab)
        ax.set_xlabel("PC1 of z_nom")
        ax.set_ylabel("PC2 of z_nom")
        ax.set_title("Nominal latent z_nom PCA (analysis only)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "07_znom_pca_by_task.png", dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        th = np.array([r.get("best_angle") or 5.0 for r in all_rows], dtype=np.float64)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=th, s=14, cmap="viridis", alpha=0.8)
        fig.colorbar(sc, ax=ax, label="oracle best angle (°)")
        ax.set_xlabel("PC1 of z_nom")
        ax.set_ylabel("PC2 of z_nom")
        ax.set_title("z_nom PCA colored by oracle magnitude")
        fig.tight_layout()
        fig.savefig(plots / "07_znom_pca_by_angle.png", dpi=140)
        plt.close(fig)

    print(f"[rm2-plot] wrote {plots} n_all={len(all_rows)} n_test={len(test_rows)} train_keys={list(train_m.keys())[:6]}", flush=True)


if __name__ == "__main__":
    main()
