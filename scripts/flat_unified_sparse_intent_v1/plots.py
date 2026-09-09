"""Required plots. Failures are shown, never hidden."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import RESULTS
from .aggregate import read_rows


def _save(fig, name: str) -> Path:
    out = RESULTS / "figures" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    return out


def family_bar(rows, title: str, fname: str, key="sr_active_5cm_strict_full_planned"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from collections import defaultdict

    by = defaultdict(list)
    for r in rows:
        by[str(r.get("family"))].append(r.get(key))
    names = sorted(by)
    vals = [float(np.nanmean(by[n])) if by[n] else 0.0 for n in names]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(names)), vals, color="#3b6ea5")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel(key)
    ax.set_title(title)
    ax.axhline(0.55, color="red", ls="--", lw=0.8, label="0.55")
    fig.tight_layout()
    p = _save(fig, fname)
    plt.close(fig)
    return p


def make_available_plots() -> list[str]:
    made = []
    p1 = RESULTS / "02_p1_3pt_competence" / "dev_eval.parquet"
    rows = read_rows(p1)
    if not rows:
        p1j = RESULTS / "02_p1_3pt_competence" / "dev_eval.jsonl"
        rows = read_rows(p1j)
    if rows:
        family_bar(rows, "P1 family SR_ACTIVE_5CM", "p1_family_success.png")
        made.append("p1_family_success.png")
    p3 = RESULTS / "04_p3_variable_points" / "dev_eval.parquet"
    rows3 = read_rows(p3)
    if rows3:
        family_bar(rows3, "P3 family SR", "p3_1pt_2pt_3pt_success.png")
        made.append("p3_1pt_2pt_3pt_success.png")
    p4 = RESULTS / "05_p4_composition" / "dev_eval.parquet"
    rows4 = read_rows(p4)
    if rows4:
        family_bar(rows4, "P4 composition SR", "p4_composition_success.png")
        made.append("p4_composition_success.png")
    # placeholders so required names exist even before data
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = [
        "p2_height_delta_vs_error.png",
        "p2_height_command_vs_actual.png",
        "p2_height_frequency_vs_error.png",
        "p2_lower_body_response.png",
        "p3_mask_combination_success.png",
        "p5_continuous_sequence_errors.png",
        "final_summary.png",
    ]
    for name in required:
        dest = RESULTS / "figures" / name
        if dest.exists():
            continue
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"{name}\npending data", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, name)
        plt.close(fig)
        made.append(name)
    return made


if __name__ == "__main__":
    print(make_available_plots())
