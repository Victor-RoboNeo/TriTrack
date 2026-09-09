#!/usr/bin/env python3
"""T1.3 unified coupled-intent plots and selected mixed cells. No Isaac."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LAMBDA_MAP = {
    (1.0, 1.0): 0.0,
    (1.2, 1.1): 1.0,
    (1.4, 1.2): 2.0,
    (1.6, 1.3): 3.0,
}


def _auroc(y, s) -> float:
    y = np.asarray(y, dtype=np.int32)
    s = np.asarray(s, dtype=np.float64)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            ranks[order[i : j + 1]] = avg
        i = j + 1
    n_pos, n_neg = float(pos.size), float(neg.size)
    return (float(ranks[y == 1].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _mean_stat(blk, key="mean") -> float:
    if not isinstance(blk, dict):
        return float("nan")
    v = blk.get(key, blk.get("p50"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _parse_st_sx(name: str) -> tuple[float, float]:
    m = re.search(r"st_([0-9.]+)_sx_([0-9.]+)", name)
    if not m:
        return float("nan"), float("nan")
    return float(m.group(1)), float(m.group(2))


def _lambda(st, sx, cell) -> float:
    lam = cell.get("lambda_id")
    try:
        lam = float(lam)
    except (TypeError, ValueError):
        lam = -1.0
    if lam >= 0:
        return lam
    key = (round(float(st), 2), round(float(sx), 2))
    return LAMBDA_MAP.get(key, float("nan"))


def _load_cal(root: Path, task: str) -> list[dict]:
    d = root / "calibration" / task
    if not d.exists():
        return []
    rows = []
    for p in sorted(d.iterdir()):
        s = p / "plane" / "summary.json"
        if not s.exists():
            continue
        cell = json.loads(s.read_text())
        mon = cell.get("episode_monitor") or {}
        tw = cell.get("time_warp") or {}
        st = float(cell.get("intent_speed") or _parse_st_sx(p.name)[0])
        sx = float(cell.get("intent_spatial") or _parse_st_sx(p.name)[1])
        kin = tw.get("kinematics") or {}
        eps = cell.get("episodes") or []
        sr5 = cell.get("sr_5cm", mon.get("sr_5cm"))
        sr2 = cell.get("sr_2cm", mon.get("sr_2cm"))
        if sr2 is None and eps:
            sr2 = float(np.nanmean([e.get("sr_2cm", np.nan) for e in eps]))
        poi_err = cell.get("poi_err", mon.get("poi_err"))
        if poi_err is None and eps:
            poi_err = float(np.nanmean([e.get("e_kp_mean", np.nan) for e in eps]))
        poi = mon.get("poi") or cell.get("poi") or {}
        if not poi and eps:
            def _body_mean(key):
                vals = []
                for e in eps:
                    v = ((e.get(key) or {}).get("mean"))
                    if v is not None and np.isfinite(float(v)):
                        vals.append(float(v))
                return float(np.mean(vals)) if vals else float("nan")
            poi = {
                "torso": {"mean": _body_mean("e_torso")},
                "lw": {"mean": _body_mean("e_lw")},
                "rw": {"mean": _body_mean("e_rw")},
            }
        rows.append({
            "task": task,
            "name": p.name,
            "s_t": st,
            "s_x": sx,
            "lam": _lambda(st, sx, cell),
            "sr": cell.get("sr_task"),
            "sr5": sr5,
            "sr2": sr2,
            "poi_err": poi_err,
            "poi_torso": _mean_stat((poi.get("torso") or {}), key="mean"),
            "poi_lw": _mean_stat((poi.get("lw") or {}), key="mean"),
            "poi_rw": _mean_stat((poi.get("rw") or {}), key="mean"),
            "poi_torso_sr5": (poi.get("torso") or {}).get("sr_5cm"),
            "poi_lw_sr5": (poi.get("lw") or {}).get("sr_5cm"),
            "poi_rw_sr5": (poi.get("rw") or {}).get("sr_5cm"),
            "mapper": _mean_stat(mon.get("mapper_agg") or {}),
            "exec": _mean_stat(mon.get("exec_e") or {}),
            "n": cell.get("n_episodes"),
            "n_fail": int(mon.get("n_failed") or 0),
            "oor": bool(cell.get("HUMAN_INTENT_OUT_OF_PLAUSIBLE_RANGE")),
            "imp": bool(cell.get("KINEMATIC_IMPOSSIBLE")),
            "hold": tw.get("median_hold_frac") or mon.get("mean_hold_fraction"),
            "active": tw.get("median_active_frac") or mon.get("mean_active_motion_fraction"),
            "n_fail_active": mon.get("n_fail_active"),
            "wt_max": (tw.get("wrist_torso") or {}).get("max"),
            "wrist_v99": ((kin.get("right_wrist") or kin.get("left_wrist") or {}).get("vel") or {}).get("p99"),
            "wrist_a99": ((kin.get("right_wrist") or kin.get("left_wrist") or {}).get("acc") or {}).get("p99"),
            "recall": mon.get("trigger_before_failure_recall"),
            "fp": mon.get("false_positive_trigger_rate"),
            "lead": (mon.get("lead_time_s") or {}).get("p50"),
            "eps": cell.get("episodes") or [],
            "cell": cell,
        })
    rows.sort(key=lambda r: (r["lam"] if np.isfinite(r["lam"]) else 99, r["s_t"], r["s_x"]))
    return rows


def _select(rows: list[dict]) -> dict | None:
    cand = [
        r for r in rows
        if r["sr"] is not None and 0.20 <= r["sr"] <= 0.80 and not r["oor"] and not r["imp"]
    ]
    if not cand:
        return None
    pref = [r for r in cand if 0.40 <= r["sr"] <= 0.70] or cand
    best = min(pref, key=lambda r: abs(r["sr"] - 0.55))
    return {
        "valid": True,
        "s_t": best["s_t"],
        "s_x": best["s_x"],
        "lambda_id": best["lam"] if np.isfinite(best["lam"]) else -1,
        "sr": best["sr"],
        "n_fail": best["n_fail"],
        "mapper": best["mapper"],
        "exec": best["exec"],
        "name": best["name"],
    }


def _t12_ref(t12: Path, task: str) -> list[dict]:
    d = t12 / task
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("speed_*/plane/summary.json")):
        cell = json.loads(p.read_text())
        if cell.get("HUMAN_INTENT_OUT_OF_PLAUSIBLE_RANGE"):
            continue
        st = float(cell.get("intent_speed") or 1.0)
        mon = cell.get("episode_monitor") or {}
        out.append({
            "task": task, "s_t": st, "s_x": 1.0, "lam": float("nan"),
            "sr": cell.get("sr_task"),
            "mapper": _mean_stat(mon.get("mapper_agg") or {}),
            "exec": _mean_stat(mon.get("exec_e") or {}),
            "oor": False, "ref": True,
        })
    return out


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _decomp(root: Path, task: str) -> dict[str, dict]:
    out = {}
    for name in ("nominal", "temporal", "spatial", "coupled"):
        s = root / "decomposition" / task / name / "plane" / "summary.json"
        if not s.exists():
            continue
        cell = json.loads(s.read_text())
        mon = cell.get("episode_monitor") or {}
        eps = cell.get("episodes") or []
        sr2 = cell.get("sr_2cm", mon.get("sr_2cm"))
        if sr2 is None and eps:
            sr2 = float(np.nanmean([e.get("sr_2cm", np.nan) for e in eps]))
        out[name] = {
            "sr": cell.get("sr_task"),
            "sr5": cell.get("sr_5cm", mon.get("sr_5cm")),
            "sr2": sr2,
            "poi": cell.get("poi_err", mon.get("poi_err")),
            "mapper": _mean_stat(mon.get("mapper_agg") or {}),
            "exec": _mean_stat(mon.get("exec_e") or {}),
            "n": cell.get("n_episodes"),
            "n_fail": int(mon.get("n_failed") or 0),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/t1_3_unified_intent_stress")
    ap.add_argument("--t12", type=str, default="results/t1_2_intent_speed")
    ap.add_argument("--write-selected", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    t12 = Path(args.t12)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    data = {t: _load_cal(root, t) for t in ("reach", "carry")}
    selected = {t: _select(rows) for t, rows in data.items()}
    (root / "selected").mkdir(parents=True, exist_ok=True)
    if args.write_selected:
        (root / "selected" / "selected.json").write_text(
            json.dumps(selected, indent=2, default=str), encoding="utf-8"
        )
        print("[t13-plot] selected", selected)

    table_a, table_b = [], []
    for task, rows in data.items():
        for r in rows:
            table_a.append({
                "Task": task, "λ": r["lam"], "s_t": r["s_t"], "s_x": r["s_x"],
                "Task SR": r["sr"], "SR@5": r["sr5"], "SR@2": r["sr2"],
                "POI": r["poi_err"],
                "Mapper E": r["mapper"], "Exec E": r["exec"],
                "Plausible": (not r["oor"]) and (not r["imp"]),
            })
        sel = selected[task]
        if sel:
            table_b.append({
                "Task": task, "λ": sel["lambda_id"], "s_t": sel["s_t"], "s_x": sel["s_x"],
                "Task SR": sel["sr"], "Failure n": sel["n_fail"], "Valid?": True,
            })
        else:
            last = [r for r in rows if not r["oor"] and not r["imp"]]
            hard = last[-1] if last else None
            table_b.append({
                "Task": task, "λ": None if hard is None else hard["lam"],
                "s_t": None if hard is None else hard["s_t"],
                "s_x": None if hard is None else hard["s_x"],
                "Task SR": None if hard is None else hard["sr"],
                "Failure n": 0 if hard is None else hard["n_fail"],
                "Valid?": False,
                "note": "speed_space_too_weak" if hard and hard["sr"] is not None and hard["sr"] > 0.80 else "no_mixed",
            })

    table_c = []
    for task in ("reach", "carry"):
        dec = _decomp(root, task)
        for var in ("nominal", "temporal", "spatial", "coupled"):
            if var not in dec:
                continue
            table_c.append({"Task": task, "Variant": var, **dec[var]})

    table_d = []
    for task, rows in data.items():
        y, re, rs = [], [], []
        fails = 0
        recs, fps, leads = [], [], []
        for r in rows:
            if r["oor"] or r["imp"]:
                continue
            for e in r["eps"]:
                y.append(int(e.get("fail") or 0))
                re.append(float(e.get("max_RE") or np.nan))
                rs.append(float(e.get("max_RS") or np.nan))
                fails += int(e.get("fail") or 0)
            if r["n_fail"]:
                recs.append(r["recall"])
                leads.append(r["lead"])
            fps.append(r["fp"])
        # prefer expanded selected if present
        exp = root / "selected" / task / "plane" / "summary.json"
        if exp.exists():
            cell = json.loads(exp.read_text())
            mon = cell.get("episode_monitor") or {}
            y, re, rs, fails = [], [], [], 0
            for e in cell.get("episodes") or []:
                y.append(int(e.get("fail") or 0))
                re.append(float(e.get("max_RE") or np.nan))
                rs.append(float(e.get("max_RS") or np.nan))
                fails += int(e.get("fail") or 0)
            recs = [mon.get("trigger_before_failure_recall")]
            fps = [mon.get("false_positive_trigger_rate")]
            leads = [(mon.get("lead_time_s") or {}).get("p50")]
        table_d.append({
            "Task": task,
            "failures": fails,
            "RE AUROC": _auroc(y, re),
            "RS AUROC": _auroc(y, rs),
            "trigger recall": float(np.nanmean(recs)) if recs else float("nan"),
            "FP": float(np.nanmean([x for x in fps if x is not None])) if fps else float("nan"),
            "lead": float(np.nanmean([x for x in leads if x is not None])) if leads else float("nan"),
        })

    # figures
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for task, rows in data.items():
        lam_rows = [r for r in rows if np.isfinite(r["lam"])]
        if lam_rows:
            ax.plot([r["lam"] for r in lam_rows], [r["sr"] for r in lam_rows], "o-", label=task)
    for task, st_tgt, sr_tgt, mark in (
        ("loco", 1.15, None, "s"),
        ("stoop", 1.50, None, "D"),
    ):
        refs = _t12_ref(t12, task)
        if not refs:
            continue
        xs, ys = [], []
        for r in refs:
            xs.append(r["s_t"])
            ys.append(r["sr"])
        ax.plot([0], [refs[0]["sr"]], mark, label=f"{task} T1.2 1.0x")
        mixed = [r for r in refs if abs(r["s_t"] - st_tgt) < 0.02]
        if mixed:
            ax.scatter([0.5], [mixed[0]["sr"]], marker=mark, s=60, label=f"{task} T1.2 mixed")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("unified λ")
    ax.set_ylabel("Task SR")
    ax.set_title("Unified λ → Task SR")
    ax.legend(fontsize=8)
    _save(fig, plots / "01_lambda_sr.png")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for task, rows in data.items():
        lam_rows = [r for r in rows if np.isfinite(r["lam"])]
        if not lam_rows:
            continue
        ax.plot([r["lam"] for r in lam_rows], [r["sr"] for r in lam_rows], "o-", label=f"{task} Task SR")
        ax.plot([r["lam"] for r in lam_rows], [r["sr5"] for r in lam_rows], "s--", label=f"{task} SR@5")
        ax.plot([r["lam"] for r in lam_rows], [r["sr2"] for r in lam_rows], "^:", label=f"{task} SR@2")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("unified λ")
    ax.set_ylabel("rate")
    ax.set_title("Unified λ → Task SR / SR@5cm / SR@2cm")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, plots / "14_lambda_sr5_sr2.png")

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for task, rows in data.items():
        lam_rows = [r for r in rows if np.isfinite(r["lam"])]
        if lam_rows:
            ax.plot([r["lam"] for r in lam_rows], [r["sr5"] for r in lam_rows], "o-", label=task)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("λ")
    ax.set_ylabel("SR@5cm")
    ax.set_title("λ → visible-POI SR@5cm")
    ax.legend()
    _save(fig, plots / "14b_lambda_sr5.png")

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for task, rows in data.items():
        lam_rows = [r for r in rows if np.isfinite(r["lam"])]
        if lam_rows:
            ax.plot([r["lam"] for r in lam_rows], [r["sr2"] for r in lam_rows], "o-", label=task)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("λ")
    ax.set_ylabel("SR@2cm")
    ax.set_title("λ → visible-POI SR@2cm")
    ax.legend()
    _save(fig, plots / "15_lambda_sr2.png")

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for task, rows in data.items():
        lam_rows = [r for r in rows if np.isfinite(r["lam"])]
        if lam_rows:
            ax.plot([r["lam"] for r in lam_rows], [r["poi_err"] for r in lam_rows], "o-", label=task)
    ax.set_xlabel("λ")
    ax.set_ylabel("visible POI error (m)")
    ax.set_title("λ → visible-POI mean error")
    ax.legend()
    _save(fig, plots / "16_lambda_poi_err.png")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for task, rows in data.items():
        lam_rows = [r for r in rows if np.isfinite(r["lam"])]
        if not lam_rows:
            continue
        ax.plot([r["lam"] for r in lam_rows], [r["poi_torso"] for r in lam_rows], "o-", label=f"{task} torso")
        ax.plot([r["lam"] for r in lam_rows], [r["poi_lw"] for r in lam_rows], "s--", label=f"{task} L wrist")
        ax.plot([r["lam"] for r in lam_rows], [r["poi_rw"] for r in lam_rows], "^:", label=f"{task} R wrist")
    ax.set_xlabel("λ")
    ax.set_ylabel("POI error (m)")
    ax.set_title("λ → per-POI mean error")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, plots / "16b_lambda_poi_by_body.png")

    for key, ylab, fname in (
        ("mapper", "Mapper E (m)", "02_lambda_mapper.png"),
        ("exec", "Exec E (m)", "03_lambda_exec.png"),
        ("wrist_v99", "wrist v p99 (m/s)", "08_lambda_wrist_vel.png"),
        ("wrist_a99", "wrist a p99 (m/s²)", "08b_lambda_wrist_acc.png"),
        ("wt_max", "max wrist–torso (m)", "09_lambda_wt.png"),
        ("hold", "hold fraction", "13_hold_frac.png"),
        ("active", "active-motion fraction", "13b_active_frac.png"),
    ):
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        for task, rows in data.items():
            lam_rows = [r for r in rows if np.isfinite(r["lam"])]
            if lam_rows:
                ax.plot([r["lam"] for r in lam_rows], [r[key] for r in lam_rows], "o-", label=task)
        ax.legend()
        ax.set_xlabel("λ")
        ax.set_ylabel(ylab)
        _save(fig, plots / fname)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for task, rows in data.items():
        ax.scatter([r["mapper"] for r in rows], [r["exec"] for r in rows], label=task)
    ax.legend()
    ax.set_xlabel("mapper E")
    ax.set_ylabel("exec E")
    _save(fig, plots / "04_map_vs_exec.png")

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for task, rows in data.items():
        ax.scatter([r["exec"] for r in rows], [r["sr"] for r in rows], label=task)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("exec E")
    ax.set_ylabel("Task SR")
    _save(fig, plots / "05_exec_vs_sr.png")

    for task in ("reach", "carry"):
        dec = _decomp(root, task)
        if len(dec) >= 2:
            fig, ax = plt.subplots(figsize=(5.8, 3.6))
            names = [k for k in ("nominal", "temporal", "spatial", "coupled") if k in dec]
            ax.bar(names, [dec[k]["sr"] for k in names])
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{task} coupling ablation Task SR")
            _save(fig, plots / f"06_{task}_ablate_sr.png")

    # constraint delta at same λ
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for task, npts, rows in (("reach", 2, data["reach"]), ("carry", 3, data["carry"])):
        by = {r["lam"]: r for r in rows if np.isfinite(r["lam"])}
        if 0.0 in by:
            xs, ys = [], []
            for lam, r in sorted(by.items()):
                xs.append(lam)
                ys.append(r["sr"] - by[0.0]["sr"])
            ax.plot(xs, ys, "o-", label=f"{task} {npts}-pt")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend()
    ax.set_xlabel("λ")
    ax.set_ylabel("Δ Task SR from λ=0")
    _save(fig, plots / "12_constraint_delta_sr.png")

    verdicts = {}
    for task, rows in data.items():
        sel = selected[task]
        if sel:
            u = "U-A"
        elif any(r["imp"] for r in rows):
            u = "U-C"
        else:
            u = "U-B"
        nom = next((r for r in rows if abs(r["s_t"] - 1) < 1e-9 and abs(r["s_x"] - 1) < 1e-9), None)
        hard = [r for r in rows if not r["oor"] and not r["imp"]]
        hard = hard[-1] if hard else None
        mapper_up = exec_up = float("nan")
        if nom and hard:
            if nom["mapper"] and hard["mapper"] and nom["mapper"] > 1e-6:
                mapper_up = hard["mapper"] / nom["mapper"]
            if nom["exec"] and hard["exec"] and nom["exec"] > 1e-6:
                exec_up = hard["exec"] / nom["exec"]
        if np.isfinite(mapper_up) and np.isfinite(exec_up):
            if mapper_up >= 1.3 and exec_up < 1.15:
                bn = "Mapper-limited"
            elif exec_up >= 1.3 and mapper_up < 1.15:
                bn = "Realization-limited"
            else:
                bn = "Coupled"
        else:
            bn = "unclear"
        drow = next((x for x in table_d if x["Task"] == task), {})
        rec = drow.get("trigger recall")
        fp = drow.get("FP")
        nf = drow.get("failures") or 0
        if nf < 8:
            mon = "weak"
        elif np.isfinite(fp) and fp >= 0.7:
            mon = "false-positive dominated"
        elif np.isfinite(rec) and rec >= 0.7:
            mon = "informative"
        else:
            mon = "weak"
        verdicts[task] = {"U": u, "bottleneck": bn, "monitor": mon, "selected": sel}

    if all((selected[t] or {}).get("valid") for t in ("reach", "carry")):
        final = "unified coupled benchmark can be frozen"
    elif (selected.get("reach") or {}).get("valid") and not (selected.get("carry") or {}).get("valid"):
        final = "Reach only valid"
        if table_b[1].get("note") == "speed_space_too_weak":
            final = "Reach only valid; Carry remains too robust"
    elif not (selected.get("reach") or {}).get("valid") and not (selected.get("carry") or {}).get("valid"):
        if any(r["imp"] for rows in data.values() for r in rows):
            final = "need abrupt-intent unified stress"
        else:
            last_c = [r for r in data["carry"] if not r["oor"] and not r["imp"]]
            last_r = [r for r in data["reach"] if not r["oor"] and not r["imp"]]
            carry_hi = last_c and last_c[-1]["sr"] is not None and last_c[-1]["sr"] > 0.80
            reach_hi = last_r and last_r[-1]["sr"] is not None and last_r[-1]["sr"] > 0.80
            if carry_hi and reach_hi:
                final = "Carry remains too robust"
            else:
                final = "need abrupt-intent unified stress"
    else:
        final = "Carry remains too robust"

    out = {
        "table_a": table_a,
        "table_b": table_b,
        "table_c": table_c,
        "table_d": table_d,
        "selected": selected,
        "verdicts": verdicts,
        "final": final,
        "operator": "S_space(S_time(I,s_t),s_x); no task id",
        "clip_end": "hold_final_valid_human_intent",
        "lambda_map": {"0": [1.0, 1.0], "1": [1.2, 1.1], "2": [1.4, 1.2], "3": [1.6, 1.3]},
    }
    (root / "summary_t13.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("[t13-plot] wrote", plots)
    print("[t13-plot] B", table_b)
    print("[t13-plot] C", table_c)
    print("[t13-plot] verdicts", verdicts)
    print("[t13-plot] FINAL", final)


if __name__ == "__main__":
    main()
