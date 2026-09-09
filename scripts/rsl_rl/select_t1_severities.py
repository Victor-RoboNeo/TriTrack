#!/usr/bin/env python3
"""Pick T1 low/mid/high severities from calibration summaries. No Isaac."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_cell(p: Path) -> dict | None:
    s = p / "plane" / "summary.json"
    if not s.exists():
        return None
    return json.loads(s.read_text())


def _sr(cell: dict) -> float:
    return float(cell.get("sr_task", float("nan")))


def _fail(cell: dict) -> float:
    return float(cell.get("fail_frac", float("nan")))


def _fall(cell: dict) -> float:
    return float(cell.get("fall_frac", float("nan")))


def _closest(items: list[tuple[str, float, dict]], target: float) -> tuple[str, float, dict] | None:
    if not items:
        return None
    return min(items, key=lambda x: abs(x[1] - target))


def _band(items: list[tuple[str, float, dict]], lo: float, hi: float) -> list[tuple[str, float, dict]]:
    return [x for x in items if lo <= x[1] <= hi]


def _pick_three(items: list[tuple[str, float, dict]]) -> dict:
    """Prefer easy~0.90, mid~0.65, hard~0.40. Report if family is too weak/strong."""
    items = [(n, s, c) for n, s, c in items if np.isfinite(s)]
    items = sorted(items, key=lambda x: x[1], reverse=True)
    note = ""
    srs = [s for _, s, _ in items]
    if not items:
        return {"selected": {}, "note": "no_cells"}
    if min(srs) > 0.90:
        note = "too_weak_all_sr_gt_90"
    elif max(srs) < 0.10:
        note = "too_strong_all_sr_lt_10"
    easy = _closest(_band(items, 0.80, 0.98) or items, 0.90)
    mid = _closest(_band(items, 0.45, 0.80) or items, 0.65)
    hard = _closest(_band(items, 0.15, 0.55) or items, 0.40)
    names = []
    selected = {}
    for key, hit in (("low", easy), ("medium", mid), ("high", hard)):
        if hit is None:
            continue
        name, sr, cell = hit
        if name in names and key != "low":
            continue
        names.append(name)
        selected[key] = {
            "severity": name,
            "sr_task": sr,
            "fail": _fail(cell),
            "fall": _fall(cell),
            "n": cell.get("n_episodes"),
            "dominant_failure": _dom_fail(cell),
        }
    if len(selected) < 2 and items:
        selected["high"] = {
            "severity": items[-1][0],
            "sr_task": items[-1][1],
            "fail": _fail(items[-1][2]),
            "fall": _fall(items[-1][2]),
            "n": items[-1][2].get("n_episodes"),
            "dominant_failure": _dom_fail(items[-1][2]),
        }
    mixed = [k for k, v in selected.items() if 0.20 <= v["sr_task"] <= 0.80]
    return {"selected": selected, "note": note, "has_mixed_regime": bool(mixed), "curve": [
        {"severity": n, "sr_task": s, "fail": _fail(c), "fall": _fall(c)} for n, s, c in items
    ]}


def _dom_fail(cell: dict) -> str:
    tax = (cell.get("episode_monitor") or {}).get("taxonomy") or {}
    keys = [k for k in ("instability", "tracking", "manipulation", "timeout", "other") if tax.get(k, 0) > 0]
    if not keys:
        return "none"
    return max(keys, key=lambda k: tax.get(k, 0))


def _payload_jobs(sel: dict, terrains: list[str], seeds: str) -> list[str]:
    rows = []
    for level, blk in sel.get("selected", {}).items():
        sev = blk["severity"]
        kg = sev.replace("kg_", "")
        for ter in terrains:
            out = f"frozen_matrix/carry_payload/{sev}"
            rows.append(
                f"carry\tshadow\t{ter}\t{out}\t{seeds}\t{sev}\t"
                f"--stress_family payload --payload_kg {kg}"
            )
    return rows


def _com_jobs(sel: dict, terrains: list[str], seeds: str) -> list[str]:
    rows = []
    for level, blk in sel.get("selected", {}).items():
        sev = blk["severity"]
        dy = sev.replace("dy_", "")
        for ter in terrains:
            out = f"frozen_matrix/carry_com/{sev}"
            rows.append(
                f"carry\tshadow\t{ter}\t{out}\t{seeds}\t{sev}\t"
                f"--stress_family com --payload_kg 4.0 --com_y {dy}"
            )
    return rows


def _ws_jobs(sel: dict, terrains: list[str], seeds: str) -> list[str]:
    spec = {
        "nom": (0.0, 0.0, 0.0),
        "lat10": (0.0, 0.10, 0.0),
        "lat20": (0.0, 0.20, 0.0),
        "fwd10": (0.10, 0.0, 0.0),
        "fwd20": (0.20, 0.0, 0.0),
        "low10": (0.0, 0.0, -0.10),
        "cross10": (0.0, -0.10, 0.0),
    }
    rows = []
    for level, blk in sel.get("selected", {}).items():
        sev = blk["severity"]
        fwd, lat, zz = spec.get(sev, (0.0, 0.0, 0.0))
        for ter in terrains:
            out = f"frozen_matrix/reach_workspace/{sev}"
            rows.append(
                f"reach\tshadow\t{ter}\t{out}\t{seeds}\t{sev}\t"
                f"--stress_family workspace --ws_fwd {fwd} --ws_lat {lat} --ws_z {zz}"
            )
    return rows


def _push_jobs(by_task: dict, terrains: list[str], seeds: str) -> list[str]:
    rows = []
    for task, sel in by_task.items():
        kind = "shadow" if task in ("reach", "carry") else "none"
        for level, blk in sel.get("selected", {}).items():
            sev = blk["severity"]
            dv = sev.replace("dv_", "")
            for ter in terrains:
                out = f"frozen_matrix/push/{task}/{sev}"
                rows.append(
                    f"{task}\t{kind}\t{ter}\t{out}\t{seeds}\t{sev}\t"
                    f"--stress_family push --push_dv {dv} --push_axis lat"
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/t1_failure_stress")
    args = ap.parse_args()
    root = Path(args.root)
    cal = root / "calibration"

    payload = []
    for p in sorted((cal / "carry_payload").glob("kg_*")):
        cell = _load_cell(p)
        if cell:
            payload.append((p.name, _sr(cell), cell))
    com = []
    for p in sorted((cal / "carry_com").glob("dy_*")):
        cell = _load_cell(p)
        if cell:
            com.append((p.name, _sr(cell), cell))
    ws = []
    for p in sorted((cal / "reach_workspace").iterdir() if (cal / "reach_workspace").exists() else []):
        if not p.is_dir():
            continue
        cell = _load_cell(p)
        if cell:
            ws.append((p.name, _sr(cell), cell))
    push = {}
    for task in ("loco", "reach", "carry"):
        items = []
        d = cal / "push" / task
        if d.exists():
            for p in sorted(d.glob("dv_*")):
                cell = _load_cell(p)
                if cell:
                    items.append((p.name, _sr(cell), cell))
        push[task] = _pick_three(items)

    picked = {
        "carry_payload": _pick_three(payload),
        "carry_com": _pick_three(com),
        "reach_workspace": _pick_three(ws),
        "push": push,
        "perception": {
            "status": "not_applicable_to_current_parent",
            "note": "Parent and R-M3 do not consume depth/RGB.",
        },
        "formal": {
            "terrains_primary": ["plane", "slope"],
            "seeds": "42,43,44,45,46",
            "n_episodes": 50,
        },
    }
    terrains = picked["formal"]["terrains_primary"]
    seeds = picked["formal"]["seeds"]
    jobs = []
    jobs += _payload_jobs(picked["carry_payload"], terrains, seeds)
    jobs += _com_jobs(picked["carry_com"], terrains, seeds)
    jobs += _ws_jobs(picked["reach_workspace"], terrains, seeds)
    jobs += _push_jobs(picked["push"], terrains, seeds)

    man = root / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    (man / "selected_severities.json").write_text(json.dumps(picked, indent=2), encoding="utf-8")
    (man / "matrix_jobs.tsv").write_text("\n".join(jobs) + ("\n" if jobs else ""), encoding="utf-8")
    print(f"[t1-select] wrote {man / 'selected_severities.json'} jobs={len(jobs)}")
    for fam, blk in picked.items():
        if fam in ("perception", "formal"):
            print(f"  {fam}: {blk}")
        elif fam == "push":
            for t, b in blk.items():
                print(f"  push/{t}: {b.get('note')} selected={list(b.get('selected', {}))} mixed={b.get('has_mixed_regime')}")
        else:
            print(f"  {fam}: {blk.get('note')} selected={list(blk.get('selected', {}))} mixed={blk.get('has_mixed_regime')}")


if __name__ == "__main__":
    main()
