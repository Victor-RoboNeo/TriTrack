#!/usr/bin/env python3
"""T1.2 intent-speed tables A–D + cascade / monitor / kinematics plots. No Isaac."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TASKS = ("reach", "carry", "stoop", "loco")
HORIZONS = ("h0.1", "h0.2", "h0.4")


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


def _num(name: str) -> float:
    for tok in reversed(str(name).replace("-", "_").split("_")):
        try:
            return float(tok)
        except ValueError:
            continue
    return float("nan")


def _mean_stat(blk, key="mean") -> float:
    if not isinstance(blk, dict):
        return float("nan")
    v = blk.get(key, blk.get("p50"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _load_cell(p: Path) -> dict | None:
    s = p / "plane" / "summary.json"
    if not s.exists():
        return None
    return json.loads(s.read_text())


def _curve(root: Path, task: str) -> list[dict]:
    d = root / task
    if not d.exists():
        return []
    rows = []
    for p in sorted(d.iterdir()):
        if not p.is_dir() or not p.name.startswith("speed_"):
            continue
        cell = _load_cell(p)
        if not cell:
            continue
        mon = cell.get("episode_monitor") or {}
        tw = cell.get("time_warp") or {}
        kin = tw.get("kinematics") or {}
        eps = cell.get("episodes") or []
        evs = cell.get("events") or []
        map_mean = _mean_stat(mon.get("mapper_agg") or {})
        if not np.isfinite(map_mean) and eps:
            map_mean = float(np.nanmean([
                _mean_stat((e.get("mapper") or {}).get("agg") or {}) for e in eps
            ]))
        exec_mean = _mean_stat(mon.get("exec_e") or {})
        if not np.isfinite(exec_mean) and eps:
            exec_mean = float(np.nanmean([e.get("e_kp_mean", np.nan) for e in eps]))
        leads = [e.get("lead_time_s") for e in eps if e.get("fail") and e.get("lead_time_s") is not None]
        chs = [str(ev.get("trigger_channel") or "?") for ev in evs]
        n_ch = max(len(chs), 1)
        rows.append({
            "task": task,
            "name": p.name,
            "speed": float(cell.get("intent_speed") or _num(p.name)),
            "sr": cell.get("sr_task"),
            "sr5": cell.get("sr_5cm"),
            "fail": cell.get("fail_frac"),
            "n": cell.get("n_episodes"),
            "n_fail": int(mon.get("n_failed") or 0),
            "n_ok": int(mon.get("n_success") or 0),
            "mapper": map_mean,
            "mapper_h02": _mean_stat(mon.get("mapper_h02") or {}),
            "exec": exec_mean,
            "recall": mon.get("trigger_before_failure_recall"),
            "fp": mon.get("false_positive_trigger_rate"),
            "lead_p50": (mon.get("lead_time_s") or {}).get("p50"),
            "lead_p10": (mon.get("lead_time_s") or {}).get("p10"),
            "lead_p90": (mon.get("lead_time_s") or {}).get("p90"),
            "duty": mon.get("trigger_duty"),
            "oor": bool(cell.get("HUMAN_INTENT_OUT_OF_PLAUSIBLE_RANGE")),
            "informative": bool(cell.get("temporal_stress_informative", True)),
            "hold_frac": tw.get("median_hold_frac"),
            "wrist_v99": ((kin.get("right_wrist") or kin.get("left_wrist") or {}).get("vel") or {}).get("p99"),
            "wrist_a99": ((kin.get("right_wrist") or kin.get("left_wrist") or {}).get("acc") or {}).get("p99"),
            "torso_v50": ((kin.get("torso") or {}).get("vel") or {}).get("p50"),
            "torso_a50": ((kin.get("torso") or {}).get("acc") or {}).get("p50"),
            "reasons": mon.get("fail_reasons") or {},
            "max_RE_ok": _mean_stat(mon.get("success_max_RE") or {}),
            "max_RE_fail": _mean_stat(mon.get("fail_max_RE") or {}),
            "max_RS_ok": _mean_stat(mon.get("success_max_RS") or {}),
            "max_RS_fail": _mean_stat(mon.get("fail_max_RS") or {}),
            "ch_E": chs.count("E") / n_ch,
            "ch_S": chs.count("S") / n_ch,
            "ch_both": chs.count("both") / n_ch,
            "eps": eps,
            "events": evs,
            "example": tw.get("example") or {},
            "cell": cell,
        })
    rows.sort(key=lambda r: r["speed"] if np.isfinite(r["speed"]) else 99)
    return rows


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _corr(xs, ys) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3 or float(np.std(x[m])) < 1e-12 or float(np.std(y[m])) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _base(rows: list[dict]) -> dict | None:
    for r in rows:
        if abs(float(r["speed"]) - 1.0) < 1e-9:
            return r
    return rows[0] if rows else None


def _dominant(base: dict, r: dict) -> str:
    if r.get("oor"):
        return "out_of_plausible"
    bm, be, bs = base.get("mapper") or np.nan, base.get("exec") or np.nan, base.get("sr") or np.nan
    m, e, s = r.get("mapper") or np.nan, r.get("exec") or np.nan, r.get("sr") or np.nan
    md = (m - bm) if np.isfinite(m) and np.isfinite(bm) else float("nan")
    ed = (e - be) if np.isfinite(e) and np.isfinite(be) else float("nan")
    sd = (s - bs) if np.isfinite(s) and np.isfinite(bs) else float("nan")
    map_up = (m / bm) if np.isfinite(m) and np.isfinite(bm) and bm > 1e-6 else float("nan")
    exec_up = (e / be) if np.isfinite(e) and np.isfinite(be) and be > 1e-6 else float("nan")
    if np.isfinite(s) and s >= 0.80 and (not np.isfinite(sd) or sd > -0.10):
        return "none_sr_still_high"
    if np.isfinite(map_up) and np.isfinite(exec_up):
        if map_up >= 1.30 and exec_up < 1.15:
            return "mapper"
        if exec_up >= 1.30 and map_up < 1.15:
            return "execution"
        if map_up >= 1.20 and exec_up >= 1.20:
            return "both"
    if np.isfinite(md) and np.isfinite(ed):
        if abs(md) > abs(ed) * 1.5 and md > 0:
            return "mapper"
        if abs(ed) > abs(md) * 1.5 and ed > 0:
            return "execution"
    return "unclear"


def _verdict_task(rows: list[dict]) -> dict:
    plausible = [r for r in rows if not r.get("oor")]
    if not plausible:
        return {"T": "T-B", "mixed": False, "selected": None, "note": "no plausible cells"}
    srs = [r["sr"] for r in plausible if r["sr"] is not None]
    mixed = [r for r in plausible if r["sr"] is not None and 0.20 <= r["sr"] <= 0.80]
    pref = [r for r in mixed if 0.40 <= r["sr"] <= 0.70] or mixed
    selected = min(pref, key=lambda r: abs(r["sr"] - 0.55)) if pref else None
    base = _base(plausible)
    hard = max(plausible, key=lambda r: r["speed"])
    if selected is not None:
        t = "T-A"
    elif srs and min(srs) > 0.80:
        t = "T-B"
    else:
        t = "T-B"
    if base and hard and t != "T-B":
        d = _dominant(base, selected or hard)
        if d == "mapper":
            t = "T-C"
        elif d == "execution":
            t = "T-D"
        elif d == "both":
            t = "T-E"
        elif t == "T-A" and d in ("none_sr_still_high", "unclear"):
            t = "T-A"
    if not rows[0].get("informative") and rows[0]["task"] == "loco":
        return {
            "T": "T-B",
            "mixed": False,
            "selected": None,
            "note": "temporal_stress_not_informative_for_loco",
        }
    return {"T": t, "mixed": bool(mixed), "selected": selected, "note": ""}


def _monitor_case(rows: list[dict]) -> str:
    fails = sum(int(r["n_fail"] or 0) for r in rows)
    if fails < 8:
        return "insufficient_failures"
    recalls = [r["recall"] for r in rows if r["n_fail"] and np.isfinite(r.get("recall") or np.nan)]
    fps = [r["fp"] for r in rows if r["n_ok"] and np.isfinite(r.get("fp") or np.nan)]
    rec = float(np.nanmean(recalls)) if recalls else float("nan")
    fp = float(np.nanmean(fps)) if fps else float("nan")
    y, re, rs = [], [], []
    for r in rows:
        for e in r["eps"]:
            y.append(int(e.get("fail") or 0))
            re.append(float(e.get("max_RE") or np.nan))
            rs.append(float(e.get("max_RS") or np.nan))
    a_e = _auroc(y, re)
    a_s = _auroc(y, rs)
    if np.isfinite(rec) and rec >= 0.70 and np.isfinite(fp) and fp <= 0.40 and max(a_e, a_s) >= 0.70:
        return "F-A"
    if fails >= 30 and (not np.isfinite(rec) or rec < 0.50) and max(a_e, a_s) < 0.60:
        return "F-B"
    if fails >= 30 and np.isfinite(rec) and rec < 0.55:
        return "F-C"
    if np.isfinite(rec) and rec >= 0.55:
        return "F-A"
    return "F-B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/t1_2_intent_speed")
    args = ap.parse_args()
    root = Path(args.root)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    data = {t: _curve(root, t) for t in TASKS}

    table_a, table_b, table_c, table_d = [], [], [], []
    verdicts = {}
    for task, rows in data.items():
        base = _base(rows)
        v = _verdict_task(rows)
        verdicts[task] = v
        mon = _monitor_case(rows)
        for r in rows:
            table_a.append({
                "Task": task,
                "Speed": r["speed"],
                "Task SR": r["sr"],
                "SR@5": r["sr5"],
                "Mapper Error": r["mapper"],
                "Exec Error": r["exec"],
                "Fail": r["fail"],
                "out_of_range": r["oor"],
            })
            if base:
                table_b.append({
                    "Task": task,
                    "Speed": r["speed"],
                    "Mapper Δ": (r["mapper"] - base["mapper"]) if None not in (r["mapper"], base["mapper"]) else None,
                    "Execution Δ": (r["exec"] - base["exec"]) if None not in (r["exec"], base["exec"]) else None,
                    "Task SR Δ": (r["sr"] - base["sr"]) if None not in (r["sr"], base["sr"]) else None,
                    "Dominant Failure": _dominant(base, r),
                })
            n_e = sum(1 for ev in r["events"] if ev.get("trigger_channel") == "E")
            n_s = sum(1 for ev in r["events"] if ev.get("trigger_channel") == "S")
            n_b = sum(1 for ev in r["events"] if ev.get("trigger_channel") == "both")
            table_c.append({
                "Task": task,
                "Speed": r["speed"],
                "Failures": r["n_fail"],
                "Recall": r["recall"],
                "False Positive": r["fp"],
                "Median Lead": r["lead_p50"],
                "E/S/Both": f"{n_e}/{n_s}/{n_b}",
            })
        if v["selected"] is not None:
            s = v["selected"]
            table_d.append({
                "Task": task, "Selected Speed": s["speed"], "Task SR": s["sr"],
                "Mapper Error": s["mapper"], "Exec Error": s["exec"], "Mixed?": True,
            })
        else:
            table_d.append({
                "Task": task, "Selected Speed": None, "Task SR": None,
                "Mapper Error": None, "Exec Error": None,
                "Mixed?": False, "note": v["note"] or "speed_only_too_weak",
            })

        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.plot([r["speed"] for r in rows], [r["sr"] for r in rows], "o-")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{task} speed → Task SR")
        ax.set_xlabel("speed ×")
        ax.set_ylabel("Task SR")
        _save(fig, plots / f"01_{task}_sr.png")

        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.plot([r["speed"] for r in rows], [r["mapper"] for r in rows], "o-", label="mapper")
        ax.set_title(f"{task} speed → Mapper-B error")
        ax.set_xlabel("speed ×")
        ax.set_ylabel("mean mapper error (m)")
        _save(fig, plots / f"04_{task}_mapper.png")

        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.plot([r["speed"] for r in rows], [r["exec"] for r in rows], "s-", color="C1")
        ax.set_title(f"{task} speed → execution E")
        ax.set_xlabel("speed ×")
        ax.set_ylabel("mean visible E (m)")
        _save(fig, plots / f"05_{task}_exec.png")

        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        xs, ys = [r["mapper"] for r in rows], [r["exec"] for r in rows]
        ax.scatter(xs, ys, c=[r["speed"] for r in rows])
        ax.set_title(f"{task} mapper vs execution")
        ax.set_xlabel("mapper error")
        ax.set_ylabel("exec E")
        _save(fig, plots / f"06_{task}_map_vs_exec.png")

        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.scatter([r["exec"] for r in rows], [r["sr"] for r in rows], c=[r["speed"] for r in rows])
        ax.set_title(f"{task} execution vs Task SR")
        ax.set_xlabel("exec E")
        ax.set_ylabel("Task SR")
        ax.set_ylim(-0.05, 1.05)
        _save(fig, plots / f"07_{task}_exec_vs_sr.png")

        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.plot([r["speed"] for r in rows], [r["recall"] for r in rows], "o-", label="recall")
        ax.plot([r["speed"] for r in rows], [r["fp"] for r in rows], "s-", label="FP")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.set_title(f"{task} trigger recall / FP vs speed")
        _save(fig, plots / f"11_{task}_trigger.png")

        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.plot([r["speed"] for r in rows], [r["lead_p50"] for r in rows], "o-")
        ax.set_title(f"{task} median lead vs speed")
        ax.set_xlabel("speed ×")
        ax.set_ylabel("lead (s)")
        _save(fig, plots / f"13_{task}_lead.png")

        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.plot([r["speed"] for r in rows], [r["wrist_v99"] for r in rows], "o-", label="wrist v p99")
        ax.set_title(f"{task} human wrist velocity p99")
        ax.set_xlabel("speed ×")
        ax.set_ylabel("m/s")
        _save(fig, plots / f"14_{task}_wrist_vel.png")

        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.plot([r["speed"] for r in rows], [r["wrist_a99"] for r in rows], "s-", color="C3")
        ax.set_title(f"{task} human wrist acceleration p99")
        ax.set_xlabel("speed ×")
        ax.set_ylabel("m/s²")
        _save(fig, plots / f"15_{task}_wrist_acc.png")

        # RE/RS success vs fail
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.plot([r["speed"] for r in rows], [r["max_RE_ok"] for r in rows], "o-", label="RE success")
        ax.plot([r["speed"] for r in rows], [r["max_RE_fail"] for r in rows], "o--", label="RE fail")
        ax.plot([r["speed"] for r in rows], [r["max_RS_ok"] for r in rows], "s-", label="RS success")
        ax.plot([r["speed"] for r in rows], [r["max_RS_fail"] for r in rows], "s--", label="RS fail")
        ax.legend(fontsize=8)
        ax.set_title(f"{task} RE/RS success vs failure")
        _save(fig, plots / f"08_{task}_rsrs.png")

        # example trajectories
        picks = [r for r in rows if abs(r["speed"] - 1.0) < 1e-9]
        if len(rows) >= 3:
            picks.append(rows[len(rows) // 2])
        if rows:
            picks.append(rows[-1])
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for r in picks:
            ex = r.get("example") or {}
            rw = np.asarray(ex.get("rw_xyz") or [], dtype=np.float64)
            if rw.size == 0:
                continue
            ax.plot(rw[:, 0], rw[:, 1], label=f"{r['speed']:.2f}x")
        ax.legend()
        ax.set_title(f"{task} example right-wrist XY")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        _save(fig, plots / f"16_{task}_traj.png")

        verdicts[task]["monitor"] = mon
        verdicts[task]["corr_speed_mapper"] = _corr([r["speed"] for r in rows], [r["mapper"] for r in rows])
        verdicts[task]["corr_mapper_exec"] = _corr([r["mapper"] for r in rows], [r["exec"] for r in rows])
        yfail, yexec = [], []
        for r in rows:
            for e in r["eps"]:
                yfail.append(int(e.get("fail") or 0))
                yexec.append(float(e.get("e_kp_mean") or np.nan))
        verdicts[task]["corr_exec_fail"] = _corr(yexec, yfail)

    # combined SR
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for task, rows in data.items():
        if rows:
            ax.plot([r["speed"] for r in rows], [r["sr"] for r in rows], "o-", label=task)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.set_title("Intent speed → Task SR")
    ax.set_xlabel("speed ×")
    _save(fig, plots / "01_all_sr.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for task, rows in data.items():
        if rows:
            ax.plot([r["speed"] for r in rows], [r["mapper"] for r in rows], "o-", label=task)
    ax.legend()
    ax.set_title("speed → Mapper-B future error")
    _save(fig, plots / "04_all_mapper.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for task, rows in data.items():
        if rows:
            ax.plot([r["speed"] for r in rows], [r["exec"] for r in rows], "s-", label=task)
    ax.legend()
    ax.set_title("speed → execution E")
    _save(fig, plots / "05_all_exec.png")

    summary = {
        "table_a": table_a,
        "table_b": table_b,
        "table_c": table_c,
        "table_d": table_d,
        "verdicts": {
            t: {
                "T": v["T"],
                "mixed": v["mixed"],
                "monitor": v.get("monitor"),
                "note": v.get("note"),
                "selected_speed": None if v.get("selected") is None else v["selected"]["speed"],
                "corr_speed_mapper": v.get("corr_speed_mapper"),
                "corr_mapper_exec": v.get("corr_mapper_exec"),
                "corr_exec_fail": v.get("corr_exec_fail"),
            }
            for t, v in verdicts.items()
        },
        "clip_end_rule": "hold_final_valid_human_intent",
        "time_warp": "I_s(t)=I_original(s*t) linear pos + quat SLERP; before Mapper-B",
    }
    any_mixed = any(v["mixed"] for v in verdicts.values())
    loco_rows = data.get("loco") or []
    loco_info = all(r.get("informative", True) for r in loco_rows) if loco_rows else True
    if not loco_info:
        summary["loco_note"] = "temporal_stress_not_informative_for_loco"
    if any_mixed:
        summary["final"] = "freeze intent-speed benchmark (mixed regime found; no oracle/recovery)"
    else:
        impl_ok = any(r.get("wrist_v99") and r["wrist_v99"] > 0.05 for rows in data.values() for r in rows)
        if not impl_ok:
            summary["final"] = "current temporal stress implementation invalid"
        else:
            summary["final"] = "move to speed+amplitude"
    (root / "summary_t12.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("[t12-plot] wrote", plots)
    print("[t12-plot] A", table_a)
    print("[t12-plot] D", table_d)
    print("[t12-plot] verdicts", summary["verdicts"])
    print("[t12-plot] FINAL", summary["final"])


if __name__ == "__main__":
    main()
