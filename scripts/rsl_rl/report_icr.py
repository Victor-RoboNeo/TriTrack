#!/usr/bin/env python3
"""ICR report: M0 baseline first, then M1–M3 if present."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TASKS = ("loco", "stoop", "reach", "carry")
TERRAINS = ("plane", "slope", "slope_down", "light_rough", "steps", "slip")
METHODS = ("m0", "m1", "m2", "m3")
LABEL = {
    "m0": "M0 Stage2 only",
    "m1": "M1 instantaneous residual",
    "m2": "M2 history residual",
    "m3": "M3 history tangent residual",
}


def _load(p: Path) -> dict:
    s = p / "summary.json"
    return json.loads(s.read_text()) if s.is_file() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/home/chenxiangyu/robotics/Anybody/results/icr_interaction_recovery")
    ap.add_argument("--out", default="/data/home/chenxiangyu/robotics/Anybody/results/icr_interaction_recovery")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    md = ["# ICR — Interaction-Conditioned Recovery\n\n"]
    md.append("Same P1 reference across terrains. No terrain ID / scan / experts in the policy.\n\n")
    md.append("| Method | Task | Terrain | SR | fall | anchor_z | e mean | e p95 | mean ||Δz|| |\n")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
    for m in METHODS:
        for t in TASKS:
            for terr in TERRAINS:
                d = _load(root / m / t / terr)
                if not d:
                    continue
                row = {
                    "method": m,
                    "task": t,
                    "terrain": terr,
                    "sr": d.get("sr_task"),
                    "fall": d.get("fall_frac"),
                    "anchor_z": d.get("anchor_z_frac"),
                    "e_mean": d.get("e_mean"),
                    "e_p95": d.get("e_p95"),
                    "mean_dz": d.get("mean_dz"),
                }
                rows.append(row)

                def f(x, pct=False):
                    if x is None:
                        return "—"
                    return f"{100.0 * float(x):.1f}%" if pct else f"{float(x):.3f}"

                md.append(
                    f"| {LABEL[m]} | {t} | {terr} | {f(row['sr'], True)} | {f(row['fall'], True)} | "
                    f"{f(row['anchor_z'], True)} | {f(row['e_mean'])} | {f(row['e_p95'])} | {f(row['mean_dz'])} |\n"
                )
    md.append("\nM0 is mandatory. M1–M3 train only after M0 is logged. Failure gate is not implemented until M2/M3 beat M0 and M1.\n")
    (out / "TABLE.md").write_text("".join(md))
    (out / "table.json").write_text(json.dumps(rows, indent=2))
    print("".join(md), flush=True)


if __name__ == "__main__":
    main()
