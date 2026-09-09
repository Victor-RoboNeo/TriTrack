"""Merge P1 1/2/3-point × task × terrain CSVs into Table A / Table B + D_hidden."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

TASKS = ("loco", "reach", "stoop", "carry")
TERRAINS = ("plane", "light_rough", "slope", "steps")
TABLE_A_TERRAINS = ("plane", "light_rough")
MASKS = ("torso", "head_left", "head_right", "vr")
# Table B: the sparse interface that actually matches the task.
CANONICAL_MASK = {
    "loco": "torso",
    "reach": "wrist1",  # mean of head_left + head_right
    "stoop": "vr",
    "carry": "vr",
}
INTENT_LABEL = {
    "torso": "Torso",
    "wrist1": "Torso + 1 wrist",
    "vr": "Torso + 2 wrists",
}


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/home/chenxiangyu/robotics/Anybody/results/p1_matrix")
    ap.add_argument("--mode", default="mapper")
    return ap.parse_args()


def _f(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _mean_ci(xs: list[float]) -> dict:
    arr = np.asarray([x for x in xs if not math.isnan(x)], dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    m = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": m, "std": sd, "ci95": ci}


def _fmt_pct(cell: dict) -> str:
    if not cell["n"]:
        return "—"
    return f"{100.0 * cell['mean']:.1f}±{100.0 * cell['ci95']:.1f}"


def _fmt_cm(cell: dict) -> str:
    if not cell["n"]:
        return "—"
    return f"{100.0 * cell['mean']:.1f}±{100.0 * cell['ci95']:.1f}"


def _fmt_rad(cell: dict) -> str:
    if not cell["n"]:
        return "—"
    return f"{cell['mean']:.3f}±{cell['ci95']:.3f}"


def _tvr_fall_from_curve(npz_path: Path) -> tuple[int, int, str]:
    """TVR = tracking violation (e_vis>25cm or ori>0.8). Fall = ori>1.2. Not the same."""
    if not npz_path.exists():
        return -1, -1, "missing"
    d = np.load(npz_path)
    e_vis = np.asarray(d["e_vis"], dtype=np.float64)
    ori = np.asarray(d["ori"], dtype=np.float64)
    tvr, fall, reason = 0, 0, "none"
    for t in range(10, len(e_vis)):
        if float(ori[t]) > 1.2:
            fall = 1
            tvr = 1
            reason = "fall_ori"
            break
        if tvr == 0 and (float(e_vis[t]) > 0.25 or float(ori[t]) > 0.8):
            tvr = 1
            reason = "tvr"
    return tvr, fall, reason


def _load_rows(root: Path, mode: str) -> list[dict]:
    rows: list[dict] = []
    for csv_path in sorted(root.glob("*/*/*_s*.csv")):
        task = csv_path.parent.parent.name
        terrain = csv_path.parent.name
        with csv_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("mode", mode) != mode:
                    continue
                rec = dict(r)
                rec["task"] = rec.get("task") or task
                rec["terrain"] = rec.get("terrain") or terrain
                rec["_csv"] = str(csv_path)
                stem = Path(rec["clip"]).stem
                mask = rec.get("mask", "")
                seed = rec.get("seed", "")
                npz = csv_path.parent / "curves" / f"{mode}_{mask}_s{seed}_{stem}.npz"
                tvr, fall, reason = _tvr_fall_from_curve(npz)
                if tvr >= 0:
                    rec["tvr"] = str(tvr)
                    rec["fail"] = str(tvr)  # back-compat
                    rec["fall"] = str(fall)
                    rec["fail_reason"] = reason
                rec["_npz"] = str(npz)
                rows.append(rec)
    return rows


def _select(rows: list[dict], task: str, terrain: str | None, mask: str) -> list[dict]:
    out = []
    for r in rows:
        if r["task"] != task:
            continue
        if terrain is not None and r["terrain"] != terrain:
            continue
        if mask == "wrist1":
            if r.get("mask") not in ("head_left", "head_right"):
                continue
        elif r.get("mask") != mask:
            continue
        out.append(r)
    return out


def _metric(rows: list[dict], key: str) -> dict:
    return _mean_ci([_f(r, key) for r in rows])


def _d_pair(npz_a: Path, npz_b: Path) -> tuple[float, float] | None:
    """Command/torso-attached local geometry, not world-path drift.

    Visible: wrists relative to torso (local intent shape).
    Hidden: ankles relative to torso (leg realization).
    Origin is commanded torso if dumped, else robot torso.
    """
    if not npz_a.exists() or not npz_b.exists():
        return None
    a = np.load(npz_a)
    b = np.load(npz_b)
    need = ("torso_xyz", "lw_xyz", "rw_xyz", "ankle_l", "ankle_r")
    if any(k not in a.files or k not in b.files for k in need):
        return None
    n = min(len(a["torso_xyz"]), len(b["torso_xyz"]))
    if n < 20:
        return None

    def local(d):
        origin_key = "goal_torso" if "goal_torso" in d.files else "torso_xyz"
        origin = np.asarray(d[origin_key][:n], dtype=np.float64)
        lw = np.asarray(d["lw_xyz"][:n], dtype=np.float64) - origin
        rw = np.asarray(d["rw_xyz"][:n], dtype=np.float64) - origin
        al = np.asarray(d["ankle_l"][:n], dtype=np.float64) - origin
        ar = np.asarray(d["ankle_r"][:n], dtype=np.float64) - origin
        vis = np.concatenate([lw, rw], axis=1)
        hid = np.concatenate([al, ar], axis=1)
        return vis, hid

    vis_a, hid_a = local(a)
    vis_b, hid_b = local(b)
    d_vis = float(np.linalg.norm(vis_a - vis_b, axis=1).mean())
    d_hid = float(np.linalg.norm(hid_a - hid_b, axis=1).mean())
    return d_vis, d_hid


def _adaptation(rows: list[dict], task: str, mask: str) -> dict:
    by_key: dict[tuple, dict[str, Path]] = defaultdict(dict)
    for r in _select(rows, task, None, mask):
        key = (r["clip"], r.get("seed", ""), r.get("mask", ""))
        by_key[key][r["terrain"]] = Path(r["_npz"])
    out = {}
    for terrain in TERRAINS:
        if terrain == "plane":
            continue
        dvs, dhs = [], []
        for paths in by_key.values():
            if "plane" not in paths or terrain not in paths:
                continue
            pair = _d_pair(paths["plane"], paths[terrain])
            if pair is None:
                continue
            dvs.append(pair[0])
            dhs.append(pair[1])
        out[terrain] = {"d_visible": _mean_ci(dvs), "d_hidden": _mean_ci(dhs)}
    return out


def main() -> None:
    args = _parse()
    root = Path(args.root)
    rows = _load_rows(root, args.mode)
    if not rows:
        raise SystemExit(f"no CSVs under {root}")

    table_a = {}
    for intent in ("torso", "wrist1", "vr"):
        table_a[intent] = {}
        for task in TASKS:
            picked = []
            for terrain in TABLE_A_TERRAINS:
                picked.extend(_select(rows, task, terrain, intent))
            table_a[intent][task] = {
                "sr_5cm": _metric(picked, "sr_5cm"),
                "fail": _metric(picked, "fail"),
                "e_kp_mean": _metric(picked, "e_kp_mean"),
                "ori_pitch_rad": _metric(picked, "ori_pitch_rad"),
                "head_left": _metric(_select(picked, task, None, "head_left"), "sr_5cm")
                if intent == "wrist1"
                else None,
                "head_right": _metric(_select(picked, task, None, "head_right"), "sr_5cm")
                if intent == "wrist1"
                else None,
            }

    table_b = {}
    for task in TASKS:
        mask = CANONICAL_MASK[task]
        table_b[task] = {"mask": mask, "terrains": {}}
        for terrain in TERRAINS:
            picked = _select(rows, task, terrain, mask)
            table_b[task]["terrains"][terrain] = {
                "sr_5cm": _metric(picked, "sr_5cm"),
                "wrist_err_mean": _metric(picked, "wrist_err_mean"),
                "wrist_err_p90": _metric(picked, "wrist_err_p90"),
                "ori_err_rad": _metric(picked, "ori_err_rad"),
                "ori_roll_rad": _metric(picked, "ori_roll_rad"),
                "ori_pitch_rad": _metric(picked, "ori_pitch_rad"),
                "tvr": _metric(picked, "tvr"),
                "fail": _metric(picked, "tvr"),
                "fall": _metric(picked, "fall"),
            }
        table_b[task]["adaptation"] = _adaptation(rows, task, mask if mask != "wrist1" else "vr")

    md = ["# P1 matrix — Mapper-B (frozen)", "", "## Table A — Task performance under different available intent constraints", ""]
    md.append(
        "Flat + Light Rough. SR@5cm is **visible-point** tracking (mean±95% CI, %). "
        "More intent points are more constraints, not an easier task. "
        "Torso-only SR is not comparable to 2/3-point SR."
    )
    md.append("")
    md.append("| Intent | Locomotion | Reach | Stoop/Pick | Carry |")
    md.append("| --- | ---: | ---: | ---: | ---: |")
    for intent in ("torso", "wrist1", "vr"):
        cells = [_fmt_pct(table_a[intent][t]["sr_5cm"]) for t in TASKS]
        md.append(f"| {INTENT_LABEL[intent]} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("Left / right 2-point SR@5cm (raw):")
    md.append("")
    md.append("| Task | Torso+Lw | Torso+Rw |")
    md.append("| --- | ---: | ---: |")
    for task in TASKS:
        left = _fmt_pct(table_a["wrist1"][task]["head_left"])
        right = _fmt_pct(table_a["wrist1"][task]["head_right"])
        md.append(f"| {task} | {left} | {right} |")

    md += ["", "## Table B — Environment adaptation", ""]
    md.append(
        "Canonical sparse interface per task. Wrist errors in cm. "
        "**TVR** = tracking violation (`e_vis>25cm` or `ori>0.8 rad`), not a fall. "
        "**Fall** = `ori>1.2 rad`. "
        "`D_visible`/`D_hidden` are torso-attached local geometry (cm), not world-path drift."
    )
    md.append("")
    for task in TASKS:
        mask = CANONICAL_MASK[task]
        md.append(f"### {task}  (mask=`{mask}`)")
        md.append("")
        md.append("| Terrain | SR@5cm ↑ | Wrist mean ↓ | Wrist P90 ↓ | Ori ↓ | Pitch ↓ | TVR ↓ | Fall ↓ |")
        md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for terrain in TERRAINS:
            c = table_b[task]["terrains"][terrain]
            md.append(
                f"| {terrain} | {_fmt_pct(c['sr_5cm'])} | {_fmt_cm(c['wrist_err_mean'])} | "
                f"{_fmt_cm(c['wrist_err_p90'])} | {_fmt_rad(c['ori_err_rad'])} | "
                f"{_fmt_rad(c['ori_pitch_rad'])} | {_fmt_pct(c['tvr'])} | {_fmt_pct(c['fall'])} |"
            )
        md.append("")
        md.append("Torso-attached local geometry vs Flat (`D_visible` wrists-in-torso ↓, `D_hidden` ankles-in-torso ↑), cm:")
        md.append("")
        md.append("| Terrain | D_visible | D_hidden |")
        md.append("| --- | ---: | ---: |")
        for terrain, cell in table_b[task]["adaptation"].items():
            md.append(
                f"| {terrain} | {_fmt_cm(cell['d_visible'])} | {_fmt_cm(cell['d_hidden'])} |"
            )
        md.append("")

    payload = {
        "n_rows": len(rows),
        "table_a": table_a,
        "table_b": table_b,
        "canonical_mask": CANONICAL_MASK,
    }

    def _jsonable(x):
        if isinstance(x, dict):
            return {k: _jsonable(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_jsonable(v) for v in x]
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x

    (root / "table_ab.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    (root / "TABLES.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print((root / "TABLES.md").read_text(), flush=True)
    print(f"[p1] wrote {root / 'TABLES.md'} and {root / 'table_ab.json'} n_rows={len(rows)}")


if __name__ == "__main__":
    main()
