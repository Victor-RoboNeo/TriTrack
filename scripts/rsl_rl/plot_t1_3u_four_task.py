#!/usr/bin/env python3
"""T1.3-U: merge Loco/Stoop new runs with T1.3 Reach/Carry λ0/1/2. No Isaac."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LAMBDA = {
    0: (1.00, 1.00),
    1: (1.20, 1.10),
    2: (1.40, 1.20),
}
T13_CELL = {
    0: "st_1.00_sx_1.00",
    1: "st_1.20_sx_1.10",
    2: "st_1.40_sx_1.20",
}
TASKS = ("loco", "stoop", "reach", "carry")


def _mean_stat(blk, key="mean") -> float:
    if not isinstance(blk, dict):
        return float("nan")
    v = blk.get(key, blk.get("p50"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    cell = json.loads(path.read_text())
    mon = cell.get("episode_monitor") or {}
    tw = cell.get("time_warp") or {}
    kin = tw.get("kinematics") or {}
    eps = cell.get("episodes") or []
    sr5 = cell.get("sr_5cm", mon.get("sr_5cm"))
    sr2 = cell.get("sr_2cm", mon.get("sr_2cm"))
    if sr2 is None and eps:
        sr2 = float(np.nanmean([e.get("sr_2cm", np.nan) for e in eps]))
    poi = cell.get("poi_err", mon.get("poi_err"))
    if poi is None and eps:
        poi = float(np.nanmean([e.get("e_kp_mean", np.nan) for e in eps]))
    mapper = _mean_stat(mon.get("mapper_agg") or {})
    exec_e = _mean_stat(mon.get("exec_e") or {})
    if not np.isfinite(exec_e) and poi is not None:
        exec_e = float(poi)
    oor = bool(cell.get("HUMAN_INTENT_OUT_OF_PLAUSIBLE_RANGE"))
    imp = bool(cell.get("KINEMATIC_IMPOSSIBLE"))
    return {
        "sr": float(cell.get("sr_task") if cell.get("sr_task") is not None else float("nan")),
        "sr5": float(sr5) if sr5 is not None else float("nan"),
        "sr2": float(sr2) if sr2 is not None else float("nan"),
        "poi": float(poi) if poi is not None else float("nan"),
        "mapper": mapper,
        "exec": exec_e,
        "n": int(cell.get("n_episodes") or 0),
        "n_fail": int(mon.get("n_failed") or 0),
        "oor": oor,
        "imp": imp,
        "plausible": (not oor) and (not imp),
        "hold": tw.get("median_hold_frac") or mon.get("mean_hold_fraction"),
        "wt_max": (tw.get("wrist_torso") or {}).get("max"),
        "wrist_v99": ((kin.get("right_wrist") or kin.get("left_wrist") or {}).get("vel") or {}).get("p99"),
        "wrist_a99": ((kin.get("right_wrist") or kin.get("left_wrist") or {}).get("acc") or {}).get("p99"),
        "torso_v99": ((kin.get("torso") or {}).get("vel") or {}).get("p99"),
        "torso_a99": ((kin.get("torso") or {}).get("acc") or {}).get("p99"),
        "fail_reasons": mon.get("fail_reasons") or {},
        "s_t": float(cell.get("intent_speed") or float("nan")),
        "s_x": float(cell.get("intent_spatial") or float("nan")),
    }


def _prefer_n50(root: Path, task: str, lam: int) -> dict | None:
    exp = _load_summary(root / "new_runs" / task / f"lam_{lam}_n50" / "plane" / "summary.json")
    base = _load_summary(root / "new_runs" / task / f"lam_{lam}" / "plane" / "summary.json")
    return exp or base


def _t13_task(t13: Path, task: str, lam: int) -> dict | None:
    name = T13_CELL[lam]
    return _load_summary(t13 / "calibration" / task / name / "plane" / "summary.json")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _label(d0: dict | None, d2: dict | None) -> str:
    if not d0 or not d2 or not d2.get("plausible"):
        if d0 and d2 and not d2.get("plausible"):
            return "invalid_λ2"
        return "unclear"
    dsr = d2["sr"] - d0["sr"]
    r_m = d2["mapper"] / d0["mapper"] if d0["mapper"] and d0["mapper"] > 1e-6 else float("nan")
    r_e = d2["exec"] / d0["exec"] if d0["exec"] and d0["exec"] > 1e-6 else float("nan")
    if abs(dsr) <= 0.05 and (not np.isfinite(r_m) or r_m < 1.20) and (not np.isfinite(r_e) or r_e < 1.20):
        return "B"
    if np.isfinite(r_m) and np.isfinite(r_e):
        if r_m >= 1.30 and r_e < 1.15:
            return "P"
        if r_e >= 1.30 and r_m < 1.15:
            return "R"
        if r_m >= 1.20 and r_e >= 1.20:
            return "C"
        if r_e >= 1.20 and abs(dsr) > 0.05:
            return "R"
        if r_m >= 1.20 and abs(dsr) > 0.05:
            return "P"
    if abs(dsr) <= 0.10:
        return "B"
    return "C"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t13u", type=str, default="results/t1_3u_four_task_unified")
    ap.add_argument("--t13", type=str, default="results/t1_3_unified_intent_stress")
    ap.add_argument("--t12", type=str, default="results/t1_2_intent_speed")
    args = ap.parse_args()
    root = Path(args.t13u)
    t13 = Path(args.t13)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    grid: dict[str, dict[int, dict]] = {t: {} for t in TASKS}
    for lam in (0, 1, 2):
        for task in ("loco", "stoop"):
            rec = _prefer_n50(root, task, lam)
            if rec:
                rec["source"] = "t1_3u"
                grid[task][lam] = rec
        for task in ("reach", "carry"):
            rec = _t13_task(t13, task, lam)
            if rec:
                rec["source"] = "t1_3_reuse"
                grid[task][lam] = rec

    matrix, table_b, table_c, table_d = [], [], [], []
    for task in TASKS:
        row_a = {"Task": task}
        d0 = grid[task].get(0)
        for lam in (0, 1, 2):
            r = grid[task].get(lam)
            if r:
                row_a[f"λ{lam} SR"] = r["sr"]
                row_a[f"λ{lam} SR@5"] = r["sr5"]
                row_a[f"λ{lam} Mapper"] = r["mapper"]
                row_a[f"λ{lam} Exec"] = r["exec"]
                row_a[f"λ{lam} plausible"] = r["plausible"]
            else:
                row_a[f"λ{lam} SR"] = None
        row_a["λ2 plausible?"] = bool((grid[task].get(2) or {}).get("plausible"))
        matrix.append(row_a)
        if d0:
            d1 = grid[task].get(1)
            d2 = grid[task].get(2)
            table_b.append({
                "Task": task,
                "ΔSR λ1": None if not d1 else d1["sr"] - d0["sr"],
                "ΔSR λ2": None if not d2 else d2["sr"] - d0["sr"],
                "ΔMapper λ2": None if not d2 else d2["mapper"] - d0["mapper"],
                "ΔExec λ2": None if not d2 else d2["exec"] - d0["exec"],
                "ΔSR@5 λ1": None if not d1 else d1["sr5"] - d0["sr5"],
                "ΔSR@5 λ2": None if not d2 else d2["sr5"] - d0["sr5"],
                "ΔMapper λ1": None if not d1 else d1["mapper"] - d0["mapper"],
                "ΔExec λ1": None if not d1 else d1["exec"] - d0["exec"],
            })
            table_c.append({
                "Task": task,
                "λ0 SR@5": d0["sr5"],
                "λ1 SR@5": None if not d1 else d1["sr5"],
                "λ2 SR@5": None if not d2 else d2["sr5"],
            })
            last = d2 if d2 and d2.get("plausible") else (d1 if d1 and d1.get("plausible") else d0)
            table_d.append({
                "Task": task,
                "Mapper trend": None if not last else last["mapper"] - d0["mapper"],
                "Exec trend": None if not last else last["exec"] - d0["exec"],
                "Task trend": None if not last else last["sr"] - d0["sr"],
                "Label": _label(d0, last if last is not d0 else d2),
            })

    # figures
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for task in TASKS:
        xs, ys, mk = [], [], []
        for lam in (0, 1, 2):
            r = grid[task].get(lam)
            if not r:
                continue
            xs.append(lam)
            ys.append(r["sr"])
            mk.append(r["plausible"])
        if not xs:
            continue
        ax.plot(xs, ys, "o-", label=task)
        for x, y, ok in zip(xs, ys, mk):
            if not ok:
                ax.scatter([x], [y], marker="x", s=90, c="k", zorder=5)
    ax.set_xticks([0, 1, 2])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("unified λ")
    ax.set_ylabel("Task Completion SR")
    ax.set_title("Unified Intent-Stress Phase Diagram")
    ax.legend()
    _save(fig, plots / "01_phase_diagram_sr.png")

    for key, ylab, fname in (
        ("mapper", "Mapper E (m)", "02_lambda_mapper.png"),
        ("exec", "Execution E (m)", "03_lambda_exec.png"),
        ("sr5", "SR@5", "05b_lambda_sr5.png"),
    ):
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for task in TASKS:
            xs, ys = [], []
            for lam in (0, 1, 2):
                r = grid[task].get(lam)
                if r and r["plausible"]:
                    xs.append(lam)
                    ys.append(r[key])
                elif r and not r["plausible"]:
                    ax.scatter([lam], [r[key]], marker="x", s=70)
            if xs:
                ax.plot(xs, ys, "o-", label=task)
        ax.set_xticks([0, 1, 2])
        ax.set_xlabel("λ")
        ax.set_ylabel(ylab)
        ax.legend()
        _save(fig, plots / fname)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for task in TASKS:
        d0 = grid[task].get(0)
        if not d0:
            continue
        for lam in (1, 2):
            r = grid[task].get(lam)
            if not r:
                continue
            ax.scatter(
                [r["mapper"] - d0["mapper"]],
                [r["exec"] - d0["exec"]],
                marker="o" if r["plausible"] else "x",
                s=70,
                label=f"{task} λ{lam}",
            )
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("Δ Mapper E from λ0")
    ax.set_ylabel("Δ Exec E from λ0")
    ax.legend(fontsize=7)
    _save(fig, plots / "04_delta_map_vs_exec.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for task in TASKS:
        xs, ys = [], []
        for lam in (0, 1, 2):
            r = grid[task].get(lam)
            if r and r["plausible"]:
                xs.append(r["sr5"])
                ys.append(r["sr"])
        if xs:
            ax.plot(xs, ys, "o-", label=task)
    ax.set_xlabel("SR@5 (intent fidelity)")
    ax.set_ylabel("Task SR")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    _save(fig, plots / "05_tasksr_vs_sr5.png")

    reach_fid = False
    if 0 in grid["reach"] and 2 in grid["reach"]:
        d0, d2 = grid["reach"][0], grid["reach"][2]
        if d2["plausible"] and d2["sr"] >= 0.85 and (d0["sr5"] - d2["sr5"]) >= 0.04:
            reach_fid = True

    impl_ok = all(0 in grid[t] and 1 in grid[t] and 2 in grid[t] for t in TASKS)
    verdict = "UNIFIED_STRESS_AXIS_VALID" if impl_ok else "IMPLEMENTATION_ISSUE"

    out = {
        "step": "T1.3-U",
        "operator": "S_space(S_time(I,s_t),s_x); reused T1.3 eval; no task id",
        "lambda": {str(k): list(v) for k, v in LAMBDA.items()},
        "excluded": "λ=3 (Reach kinematic impossible); T1.2 temporal-only anchors are legacy only",
        "table_a": matrix,
        "table_b": table_b,
        "table_c": table_c,
        "table_d": table_d,
        "grid": {
            t: {
                str(lam): {k: v for k, v in rec.items() if k != "fail_reasons"}
                for lam, rec in grid[t].items()
            }
            for t in TASKS
        },
        "reach_task_high_fidelity_down": reach_fid,
        "legacy_t12": {
            "loco_temporal_1.15": "SR≈0.75 LEGACY TEMPORAL-ONLY MIXED REFERENCE",
            "stoop_temporal_1.50": "SR≈0.70 LEGACY TEMPORAL-ONLY MIXED REFERENCE",
        },
        "verdict": verdict,
    }
    (root / "merged" / "summary_t13u.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    (root / "phase_diagram" / "table_a.json").write_text(json.dumps(matrix, indent=2, default=str), encoding="utf-8")
    (root / "bottleneck" / "table_d.json").write_text(json.dumps(table_d, indent=2, default=str), encoding="utf-8")
    print("[t13u-plot] A", matrix)
    print("[t13u-plot] B", table_b)
    print("[t13u-plot] C", table_c)
    print("[t13u-plot] D", table_d)
    print("[t13u-plot] reach_fid", reach_fid)
    print("[t13u-plot] VERDICT", verdict)


if __name__ == "__main__":
    main()
