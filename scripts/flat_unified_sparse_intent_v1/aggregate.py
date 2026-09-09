"""Aggregate eval parquet/jsonl into family macros and gates."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io_util import atomic_write_json
from .metrics_active import mean_finite


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        alt = path.with_suffix(".jsonl")
        if alt.exists():
            return [json.loads(l) for l in alt.read_text().splitlines() if l.strip()]
        return []
    if path.suffix == ".parquet" and path.stat().st_size > 100:
        try:
            import pyarrow.parquet as pq

            return pq.read_table(path).to_pylist()
        except Exception:
            pass
    alt = path.with_suffix(".jsonl")
    if alt.exists():
        return [json.loads(l) for l in alt.read_text().splitlines() if l.strip()]
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return []


def _fam(row: dict) -> str:
    return str(row.get("campaign_family") or row.get("family") or "unk")


def family_table(rows: list[dict], key: str = "sr_active_5cm_strict_full_planned") -> dict:
    by = defaultdict(list)
    for r in rows:
        by[_fam(r)].append(r.get(key))
    return {f: mean_finite(vs) for f, vs in sorted(by.items())}


def map_p1_families(rows: list[dict]) -> dict:
    """Map suite families onto F1–F4 for P1 gates."""
    buckets = {"F1_STATIC": [], "F2_REACH": [], "F3_BASIC_HEIGHT": [], "F4_HORIZONTAL": [], "OTHER": []}
    for r in rows:
        fam = _fam(r)
        sr = r.get("sr_active_5cm_strict_full_planned")
        if fam.startswith("F1") or fam.startswith("A_standing"):
            buckets["F1_STATIC"].append(sr)
            buckets["F2_REACH"].append(sr)
        elif fam.startswith("F2") or "hand" in fam.lower() or fam.startswith("A_"):
            buckets["F2_REACH"].append(sr)
        elif fam.startswith("F3") or "squat" in fam.lower() or fam.startswith("B_") or fam.startswith("C_"):
            buckets["F3_BASIC_HEIGHT"].append(sr)
        elif fam.startswith("F4") or fam.startswith("D_") or "loco" in fam.lower() or "walk" in fam.lower():
            buckets["F4_HORIZONTAL"].append(sr)
        else:
            buckets["OTHER"].append(sr)
    return {k: mean_finite(v) for k, v in buckets.items()}


def summarize(rows: list[dict]) -> dict:
    sr_a = [r.get("sr_active_5cm_strict_full_planned") for r in rows]
    sr3 = [r.get("sr_3pt_5cm_strict_full_planned") for r in rows]
    fall = [1.0 - float(bool(r.get("fell"))) for r in rows]
    return {
        "n": len(rows),
        "overall_SR_ACTIVE_5CM": mean_finite(sr_a),
        "overall_SR_3PT_5CM": mean_finite(sr3),
        "fall_free_completion": mean_finite(fall),
        "by_family_SR_ACTIVE_5CM": family_table(rows, "sr_active_5cm_strict_full_planned"),
        "by_family_SR_3PT_5CM": family_table(rows, "sr_3pt_5cm_strict_full_planned"),
        "p1_mapped": map_p1_families(rows),
        "p95_position_error": mean_finite([r.get("p95_position_error") for r in rows]),
    }


def p0_repeat_gate(s1: dict, s2: dict, parent: float, tol: float) -> dict:
    a1 = float(s1.get("overall_SR_3PT_5CM", float("nan")))
    a2 = float(s2.get("overall_SR_3PT_5CM", float("nan")))
    delta = abs(a1 - a2)
    return {
        "repeat_1": a1,
        "repeat_2": a2,
        "delta": delta,
        "parent_reference": parent,
        "delta_vs_parent_1": abs(a1 - parent),
        "delta_vs_parent_2": abs(a2 - parent),
        "PASS_repeat": bool(delta <= tol and np.isfinite(a1) and np.isfinite(a2)),
        "note": "Parent 0.4314 is live-anchor v3 DEV known_preview. "
        "This campaign eval has no 18-D task-error head; small offset is allowed if explained.",
    }


def p1_gate(summary: dict, s_parent: float) -> dict:
    mapped = summary.get("p1_mapped") or {}
    overall = float(summary.get("overall_SR_ACTIVE_5CM", float("nan")))
    f1 = float(mapped.get("F1_STATIC", float("nan")))
    f2 = float(mapped.get("F2_REACH", float("nan")))
    fall = float(summary.get("fall_free_completion", float("nan")))
    return {
        "overall": overall,
        "F1_STATIC": f1,
        "F2_REACH": f2,
        "fall_free": fall,
        "need": {"overall": 0.55, "F1": 0.70, "F2": 0.60, "fall_free": 0.95},
        "PASS": bool(overall >= 0.55 and f1 >= 0.70 and f2 >= 0.60 and fall >= 0.95),
        "s_parent": s_parent,
    }


def write_summary(path: Path, summary: dict) -> None:
    atomic_write_json(path, summary)
