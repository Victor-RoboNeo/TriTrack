#!/usr/bin/env python3
"""Collect P2-C fixed-eval summaries into one JSON table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = ("loco", "stoop", "reach", "carry")
TERRAINS = ("plane", "light_rough", "slope", "steps")
MASK = {"loco": "torso", "stoop": "vr", "reach": "head_right", "carry": "vr"}
P1_REF = {
    ("loco", "plane"): 50.4,
    ("loco", "light_rough"): 52.6,
    ("loco", "slope"): 34.9,
    ("loco", "steps"): 29.5,
    ("stoop", "plane"): 38.6,
    ("stoop", "light_rough"): 41.0,
    ("stoop", "slope"): 29.4,
    ("stoop", "steps"): 27.2,
}


def _cell(root: Path, task: str, terrain: str) -> dict | None:
    p = root / task / terrain / "summary.json"
    if not p.exists():
        return None
    payload = json.loads(p.read_text())
    mask = MASK[task]
    mode = payload.get("modes", {}).get("mapper", {})
    if mask not in mode:
        # fall back to first mask
        mask = next(iter(mode))
    block = mode[mask]
    row = block.get("s42") or block.get("pooled") or next(iter(block.values()))
    return {
        "sr5": round(float(row["sr_5cm"]) * 100.0, 2),
        "fail": round(float(row.get("fail", float("nan"))) * 100.0, 2) if row.get("fail") is not None else None,
        "e_kp_cm": round(float(row.get("e_kp_mean", float("nan"))) * 100.0, 2),
        "ang": None if row.get("terrain_ang_deg") is None else round(float(row["terrain_ang_deg"]), 2),
        "u_perp": None if row.get("terrain_u_perp") is None else round(float(row["terrain_u_perp"]), 4),
        "scan_mode": payload.get("scan_mode", "normal"),
        "path": str(p),
    }


def _matrix(root: Path) -> dict:
    out = {}
    missing = []
    for task in TASKS:
        out[task] = {}
        for terrain in TERRAINS:
            cell = _cell(root, task, terrain)
            if cell is None:
                missing.append(f"{task}/{terrain}")
            else:
                ref = P1_REF.get((task, terrain))
                if ref is not None:
                    cell["p1_ref"] = ref
                    cell["dpp"] = round(cell["sr5"] - ref, 2)
                out[task][terrain] = cell
    return {"cells": out, "missing": missing, "root": str(root)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    runs = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "freeze"):
        if not any((d / t).is_dir() for t in TASKS):
            continue
        runs[d.name] = _matrix(d)
        cf = d / "scan_counterfactual.json"
        if cf.exists():
            runs[d.name]["counterfactual"] = json.loads(cf.read_text())
    payload = {"runs": runs}
    out = root / "summary.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[p2c-eval] wrote {out}")
    for name, mat in runs.items():
        print(f"\n=== {name} missing={mat['missing']} ===")
        print(f"{'task':<8} {'plane':>8} {'light':>8} {'slope':>8} {'steps':>8}  ang(F/L/S/T)")
        for task in TASKS:
            srs = []
            angs = []
            for terrain in TERRAINS:
                cell = mat["cells"].get(task, {}).get(terrain)
                srs.append(f"{cell['sr5']:.1f}" if cell else "  -")
                angs.append(f"{cell['ang']:.1f}" if cell and cell.get("ang") is not None else "-")
            print(f"{task:<8} " + " ".join(f"{x:>8}" for x in srs) + "  " + "/".join(angs))
        cf = mat.get("counterfactual")
        if cf:
            print(
                "  u-angle Flat-Slope={angle_u_flat_slope_deg:.1f} "
                "Flat-Steps={angle_u_flat_steps_deg:.1f} "
                "Slope-Steps={angle_u_slope_steps_deg:.1f} "
                "cos FS={cos_u_flat_slope:.3f} FT={cos_u_flat_steps:.3f} ST={cos_u_slope_steps:.3f}".format(**cf)
            )


if __name__ == "__main__":
    main()
