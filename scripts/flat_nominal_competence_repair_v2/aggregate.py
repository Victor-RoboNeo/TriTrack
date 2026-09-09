"""Family macros for clean F1–F4 only. No legacy bleed."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from flat_unified_sparse_intent_v1.metrics_active import mean_finite
from .io_util import atomic_write_json


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        alt = path.with_suffix(".jsonl")
        return [json.loads(l) for l in alt.read_text().splitlines() if l.strip()] if alt.exists() else []
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


def _fam(r: dict) -> str:
    return str(r.get("family") or "unk")


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[_fam(r)].append(r)
    fam = {}
    for f, rs in sorted(by.items()):
        fam[f] = {
            "n": len(rs),
            "SR_ACTIVE_5CM": mean_finite([x.get("sr_active_5cm_strict_full_planned") for x in rs]),
            "fall_free": mean_finite([1.0 - float(bool(x.get("fell"))) for x in rs]),
            "p95": mean_finite([x.get("p95_position_error") for x in rs]),
        }
    f1 = float(fam.get("F1_STATIC", {}).get("SR_ACTIVE_5CM") or float("nan"))
    f2 = float(fam.get("F2_REACH", {}).get("SR_ACTIVE_5CM") or float("nan"))
    fall = mean_finite([1.0 - float(bool(r.get("fell"))) for r in rows])
    h = 0.0
    if f1 > 0 and f2 > 0 and np.isfinite(f1) and np.isfinite(f2):
        h = 2.0 / (1.0 / f1 + 1.0 / f2)
    return {
        "n": len(rows),
        "F1_STATIC": f1,
        "F2_REACH": f2,
        "F3_BASIC_HEIGHT_DIAG": float(fam.get("F3_BASIC_HEIGHT_DIAG", {}).get("SR_ACTIVE_5CM") or float("nan")),
        "F4_HORIZONTAL_DIAG": float(fam.get("F4_HORIZONTAL_DIAG", {}).get("SR_ACTIVE_5CM") or float("nan")),
        "fall_free": fall,
        "F1_F2_HARMONIC": h,
        "basic_macro": float(np.nanmean([f1, f2])),
        "p95_position_error": mean_finite([r.get("p95_position_error") for r in rows]),
        "C_rmse_xyz": mean_finite([r.get("C_rmse_xyz", r.get("H_rmse_xyz")) for r in rows]),
        "LH_rmse_xyz": mean_finite([r.get("LH_rmse_xyz") for r in rows]),
        "RH_rmse_xyz": mean_finite([r.get("RH_rmse_xyz") for r in rows]),
        "by_family": fam,
        "legacy_bleed": [f for f in fam if not str(f).startswith("F") and not str(f).startswith("H")],
        "H_LEGACY_ALIAS": "slot0 metrics H_* == canonical chest/torso C_*",
    }


def p1_gate(s: dict) -> dict:
    f1, f2, fall, macro = s.get("F1_STATIC", 0), s.get("F2_REACH", 0), s.get("fall_free", 0), s.get("basic_macro", 0)
    return {
        "F1": f1,
        "F2": f2,
        "fall_free": fall,
        "basic_macro": macro,
        "need": {"F1": 0.70, "F2": 0.60, "fall_free": 0.95, "basic_macro": 0.65},
        "PASS": bool(f1 >= 0.70 and f2 >= 0.60 and fall >= 0.95 and macro >= 0.65),
    }


def write_summary(path: Path, s: dict) -> None:
    atomic_write_json(path, s)


def write_rows_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def merge_row_files(paths: list[Path], out: Path) -> list[dict]:
    rows = []
    for p in paths:
        rows.extend(read_rows(p))
    write_rows_jsonl(out.with_suffix(".jsonl"), rows)
    try:
        from flat_locomani.next5.tables import write_rows

        write_rows(out, rows)
    except Exception:
        pass
    return rows


def success_flag(r: dict) -> bool:
    sr = r.get("sr_active_5cm_strict_full_planned")
    fell = bool(r.get("fell"))
    try:
        return (not fell) and float(sr) >= 0.5
    except (TypeError, ValueError):
        return not fell


def paired_matrix(parent_rows: list[dict], cand_rows: list[dict]) -> dict:
    p = {str(r.get("traj_id")): r for r in parent_rows}
    c = {str(r.get("traj_id")): r for r in cand_rows}
    ids = sorted(set(p) & set(c))
    gg = gb = bg = bb = 0
    gained, lost = [], []
    for i in ids:
        ps, cs = success_flag(p[i]), success_flag(c[i])
        if ps and cs:
            gg += 1
        elif ps and not cs:
            gb += 1
            lost.append(i)
        elif (not ps) and cs:
            bg += 1
            gained.append(i)
        else:
            bb += 1
    return {
        "n_paired": len(ids),
        "parent_good_cand_good": gg,
        "parent_good_cand_bad": gb,
        "parent_bad_cand_good": bg,
        "parent_bad_cand_bad": bb,
        "GAINED": bg,
        "LOST": gb,
        "gained_ids": gained[:50],
        "lost_ids": lost[:50],
    }


def determinism_gate(s1: dict, s2: dict) -> dict:
    def d(a, b):
        if a is None or b is None:
            return float("nan")
        if not (np.isfinite(a) and np.isfinite(b)):
            return float("nan")
        return abs(float(a) - float(b))

    overall1 = float(s1.get("basic_macro") or 0)
    overall2 = float(s2.get("basic_macro") or 0)
    rec = {
        "overall_delta": d(overall1, overall2),
        "F1_delta": d(s1.get("F1_STATIC"), s2.get("F1_STATIC")),
        "F2_delta": d(s1.get("F2_REACH"), s2.get("F2_REACH")),
        "fall_free_delta": d(s1.get("fall_free"), s2.get("fall_free")),
        "need": {"overall": 0.02, "F1": 0.03, "F2": 0.03, "fall_free": 0.03},
    }
    rec["PASS"] = bool(
        rec["overall_delta"] <= 0.02
        and rec["F1_delta"] <= 0.03
        and rec["F2_delta"] <= 0.03
        and rec["fall_free_delta"] <= 0.03
    )
    return rec


def existing_gate(s: dict) -> dict:
    f1 = float(s.get("F1_STATIC") or 0)
    f2 = float(s.get("F2_REACH") or 0)
    fall = float(s.get("fall_free") or 0)
    return {
        "F1": f1,
        "F2": f2,
        "fall_free": fall,
        "PASS": bool(f1 >= 0.70 and f2 >= 0.60 and fall >= 0.95),
    }
