"""Plots. Failures are shown, never hidden. Slot 0 is chest/torso, not robot head."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .aggregate import read_rows
from .constants import RESULTS


def _save(fig, name: str) -> Path:
    out = RESULTS / "plots" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    return out


def family_bar(rows, title: str, fname: str, key="sr_active_5cm_strict_full_planned"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = defaultdict(list)
    for r in rows:
        by[str(r.get("family"))].append(r.get(key))
    names = [n for n in sorted(by) if n.startswith("F") or n.startswith("H")]
    vals = [float(np.nanmean(by[n])) if by[n] else 0.0 for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(names)), vals, color="#3b6ea5")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("SR_ACTIVE_5CM")
    ax.set_title(title)
    ax.axhline(0.70, color="green", ls="--", lw=0.8)
    ax.axhline(0.60, color="orange", ls="--", lw=0.8)
    fig.tight_layout()
    p = _save(fig, fname)
    plt.close(fig)
    return p


def drift_scatter(csv_path: Path, ykey: str, fname: str, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not csv_path.exists():
        return None
    import csv

    xs, ys = [], []
    with open(csv_path) as f:
        for rec in csv.DictReader(f):
            try:
                xs.append(float(rec.get("D_action_mean") or rec.get("D_gphi_mean") or 0))
                ys.append(float(rec.get(ykey) or 0))
            except (TypeError, ValueError):
                continue
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys, s=12, alpha=0.6)
    ax.set_xlabel("D_action mean")
    ax.set_ylabel(ykey)
    ax.set_title(title)
    fig.tight_layout()
    p = _save(fig, fname)
    plt.close(fig)
    return p


def make_available_plots() -> list[str]:
    made = []
    for name, rel in (
        ("r3_parent_family.png", RESULTS / "03_paired_reeval" / "PARENT.jsonl"),
        ("r5_selected_family.png", RESULTS / "06_clean_p1_eval" / "dev_eval.jsonl"),
    ):
        rows = read_rows(rel) if rel.exists() else read_rows(rel.with_suffix(".parquet"))
        if rows:
            family_bar(rows, name.replace(".png", ""), name)
            made.append(name)
    drift_csv = RESULTS / "04_policy_drift" / "drift_vs_outcome.csv"
    if drift_csv.exists():
        if drift_scatter(drift_csv, "fell", "drift_vs_fall.png", "drift vs fall"):
            made.append("drift_vs_fall.png")
        if drift_scatter(drift_csv, "sr_active_5cm_strict_full_planned", "drift_vs_tracking.png", "drift vs tracking"):
            made.append("drift_vs_tracking.png")
    return made
