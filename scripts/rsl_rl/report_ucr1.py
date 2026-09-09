#!/usr/bin/env python3
"""UCR-1 report: held-out clone utility vs old R-M3 / oracle. Case A/B/C. No closed-loop."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TASKS = ("loco", "stoop", "reach", "carry")
GO = {
    "loco": {"p": 0.70, "median_lt0": True, "mean_le0": False},
    "stoop": {"p": 0.65, "median_lt0": True, "mean_le0": False},
    "reach": {"p": 0.65, "median_lt0": True, "mean_le0": True},
    "carry": {"p": 0.65, "median_lt0": True, "mean_le0": False},
}


def _j(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    raise TypeError(type(o))


def _blk(i_m: np.ndarray) -> dict:
    x = np.asarray(i_m, dtype=np.float64) * 100.0
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "P_I_lt_0": float("nan"), "P_I_lt_0.25cm": float("nan"),
                "median_cm": float("nan"), "mean_cm": float("nan"),
                "P_I_gt_0.5cm": float("nan"), "P_I_gt_1.0cm": float("nan")}
    return {
        "n": int(x.size),
        "P_I_lt_0": float((x < 0).mean()),
        "P_I_lt_0.25cm": float((x < -0.25).mean()),
        "median_cm": float(np.median(x)),
        "mean_cm": float(x.mean()),
        "P_I_gt_0.5cm": float((x > 0.5).mean()),
        "P_I_gt_1.0cm": float((x > 1.0).mean()),
    }


def _task_go(task: str, new: dict, old: dict | None) -> tuple[bool, list[str]]:
    spec = GO[task]
    reasons = []
    ok = True
    p = new.get("P_I_lt_0")
    med = new.get("median_cm")
    mean = new.get("mean_cm")
    if not (isinstance(p, float) and p >= spec["p"]):
        ok = False
        reasons.append(f"P(I<0)={p} < {spec['p']}")
    if spec["median_lt0"] and not (isinstance(med, float) and med < 0):
        ok = False
        reasons.append(f"median I={med} not < 0")
    if spec["mean_le0"] and not (isinstance(mean, float) and mean <= 0):
        ok = False
        reasons.append(f"mean I={mean} not <= 0")
    if task == "reach" and old:
        new_t = new.get("P_I_gt_0.5cm")
        old_t = old.get("P_I_gt_0.5cm")
        new_t1 = new.get("P_I_gt_1.0cm")
        old_t1 = old.get("P_I_gt_1.0cm")
        better = False
        if isinstance(new_t, float) and isinstance(old_t, float) and new_t < old_t - 1e-6:
            better = True
        if isinstance(new_t1, float) and isinstance(old_t1, float) and new_t1 < old_t1 - 1e-6:
            better = True
        if not better:
            ok = False
            reasons.append(f"Reach harmful tail not below old R-M3 (new P>+0.5={new_t} vs old {old_t})")
    return ok, reasons


def load_heldout(root: Path) -> dict[str, dict[str, np.ndarray]]:
    """task -> method -> I (meters)."""
    out: dict[str, dict[str, list]] = {t: {} for t in TASKS}
    for p in sorted(root.glob("*/lam_*/plane/heldout.npz")):
        b = np.load(p, allow_pickle=True)
        task = str(p.parts[-4]) if p.parts[-4] in TASKS else str(np.asarray(b["task"]).astype(str)[0] if "task" in b.files else "")
        if task not in TASKS:
            # clone_eval/scratch_s0/loco/lam_0/plane/heldout.npz
            for part in p.parts:
                if part in TASKS:
                    task = part
                    break
        if task not in TASKS:
            continue
        for k in b.files:
            if not k.startswith("i_"):
                continue
            name = k[2:]
            out[task].setdefault(name, []).append(np.asarray(b[k], dtype=np.float32))
    packed = {}
    for t, methods in out.items():
        packed[t] = {n: np.concatenate(vs, 0) if vs else np.zeros(0, dtype=np.float32) for n, vs in methods.items()}
    return packed


def pick_best(per_task: dict, candidates: list[str]) -> str | None:
    """Max mean P(I<0) across tasks that have data; tie-break lower mean median I."""
    best = None
    best_key = None
    for name in candidates:
        ps = []
        meds = []
        for t in TASKS:
            i = per_task.get(t, {}).get(name)
            if i is None or i.size == 0:
                continue
            blk = _blk(i)
            ps.append(blk["P_I_lt_0"])
            meds.append(blk["median_cm"])
        if not ps:
            continue
        key = (float(np.mean(ps)), -float(np.mean(meds)))
        if best_key is None or key > best_key:
            best_key = key
            best = name
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone_eval", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--supervised", type=str, default="")
    ap.add_argument("--split", type=str, default="")
    args = ap.parse_args()
    root = Path(args.clone_eval)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_task = load_heldout(root)
    methods = sorted({n for t in TASKS for n in per_task.get(t, {})})
    scratch = [n for n in methods if n.startswith("scratch_") or n.startswith("warm_")]
    best_new = pick_best(per_task, [n for n in methods if n not in ("parent", "old_rm3", "oracle")])

    table1 = []
    table2 = []
    go_tasks = {}
    oracle_strong = True
    oracle_weak = True
    for t in TASKS:
        old = _blk(per_task.get(t, {}).get("old_rm3", np.zeros(0)))
        new = _blk(per_task.get(t, {}).get(best_new, np.zeros(0)) if best_new else np.zeros(0))
        ora = _blk(per_task.get(t, {}).get("oracle", np.zeros(0)))
        ok, reasons = _task_go(t, new, old)
        go_tasks[t] = {"go": ok, "reasons": reasons, "old": old, "new": new, "oracle": ora}
        table1.append({
            "task": t, "n": new.get("n", 0),
            "old_P_I_lt_0": old.get("P_I_lt_0"),
            "new_P_I_lt_0": new.get("P_I_lt_0"),
            "new_median_cm": new.get("median_cm"),
            "oracle_P_I_lt_0": ora.get("P_I_lt_0"),
        })
        table2.append({
            "task": t,
            "P_I_lt_0.25cm": new.get("P_I_lt_0.25cm"),
            "mean_cm": new.get("mean_cm"),
            "P_I_gt_0.5cm": new.get("P_I_gt_0.5cm"),
            "P_I_gt_1.0cm": new.get("P_I_gt_1.0cm"),
            "old_P_I_gt_0.5cm": old.get("P_I_gt_0.5cm"),
            "old_P_I_gt_1.0cm": old.get("P_I_gt_1.0cm"),
        })
        if not (isinstance(ora.get("P_I_lt_0"), float) and ora["P_I_lt_0"] >= 0.85 and ora.get("median_cm", 1) < 0):
            oracle_strong = False
        if isinstance(ora.get("P_I_lt_0"), float) and ora["P_I_lt_0"] >= 0.70 and ora.get("median_cm", 1) < 0:
            oracle_weak = False

    pooled_old = _blk(np.concatenate([per_task.get(t, {}).get("old_rm3", np.zeros(0)) for t in TASKS], 0))
    pooled_new = _blk(np.concatenate([
        per_task.get(t, {}).get(best_new, np.zeros(0)) if best_new else np.zeros(0) for t in TASKS
    ], 0))
    pooled_ora = _blk(np.concatenate([per_task.get(t, {}).get("oracle", np.zeros(0)) for t in TASKS], 0))
    pooled_better = (
        isinstance(pooled_new.get("P_I_lt_0"), float)
        and isinstance(pooled_old.get("P_I_lt_0"), float)
        and (
            pooled_new["P_I_lt_0"] > pooled_old["P_I_lt_0"] + 0.02
            or (
                pooled_new["P_I_lt_0"] >= pooled_old["P_I_lt_0"]
                and pooled_new.get("median_cm", 0) < pooled_old.get("median_cm", 0)
            )
        )
    )
    all_go = all(go_tasks[t]["go"] for t in TASKS) and pooled_better
    if all_go:
        case = "A"
        verdict = "FINAL_UNIFIED_RECOVERY_FIELD_VALIDATED"
        note = "All four tasks clone-utility GO vs old R-M3. Do not open closed-loop tonight."
    elif oracle_weak:
        case = "C"
        verdict = "HOLD"
        note = "Oracle itself is weak on the new RE-trigger data. Local recovery assumption needs re-check."
    else:
        case = "B"
        verdict = "HOLD"
        note = "Partial task failure with a still-strong oracle. Coverage/training, not a task-specific patch."

    payload = {
        "step": "UCR-1",
        "no_closed_loop": True,
        "no_ppo": True,
        "selected_ckpt": best_new,
        "candidates": methods,
        "table1": table1,
        "table2": table2,
        "pooled": {"old": pooled_old, "new": pooled_new, "oracle": pooled_ora, "new_better": pooled_better},
        "go": go_tasks,
        "case": case,
        "verdict": verdict,
        "note": note,
        "oracle_strong": oracle_strong,
    }
    if args.split:
        sp = Path(args.split)
        if sp.is_file():
            payload["split"] = json.loads(sp.read_text())
    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=_j))

    lines = [
        "# UCR-1 Final Unified RE-Triggered Recovery Field",
        "",
        f"Selected checkpoint: `{best_new}` (held-out clone utility, not cosine)",
        f"Case {case}: **{verdict}**",
        note,
        "",
        "| Task | n | Old R-M3 P(I<0) | New P(I<0) | New median I | Oracle P(I<0) |",
        "|------|---|------------------|------------|--------------|----------------|",
    ]
    for r in table1:
        lines.append(
            f"| {r['task']} | {r['n']} | {_fmt(r['old_P_I_lt_0'])} | {_fmt(r['new_P_I_lt_0'])} "
            f"| {_fmt(r['new_median_cm'], cm=True)} | {_fmt(r['oracle_P_I_lt_0'])} |"
        )
    lines += [
        "",
        "| Task | P(I<-0.25) | mean I | P(I>+0.5) | P(I>+1cm) |",
        "|------|------------|--------|-----------|-----------|",
    ]
    for r in table2:
        lines.append(
            f"| {r['task']} | {_fmt(r['P_I_lt_0.25cm'])} | {_fmt(r['mean_cm'], cm=True)} "
            f"| {_fmt(r['P_I_gt_0.5cm'])} | {_fmt(r['P_I_gt_1.0cm'])} |"
        )
    lines += [
        "",
        f"Pooled old P(I<0)={_fmt(pooled_old.get('P_I_lt_0'))} median={_fmt(pooled_old.get('median_cm'), cm=True)}",
        f"Pooled new P(I<0)={_fmt(pooled_new.get('P_I_lt_0'))} median={_fmt(pooled_new.get('median_cm'), cm=True)}",
        f"Pooled oracle P(I<0)={_fmt(pooled_ora.get('P_I_lt_0'))} median={_fmt(pooled_ora.get('median_cm'), cm=True)}",
        "",
        "GO per task:",
    ]
    for t in TASKS:
        g = go_tasks[t]
        lines.append(f"- {t}: {'GO' if g['go'] else 'HOLD'} {'; '.join(g['reasons'])}")
    lines += ["", "No closed-loop. No PPO. No new trigger. Stop here."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def _fmt(v, cm=False):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if cm:
        return f"{v:.2f} cm"
    return f"{v:.3f}"


if __name__ == "__main__":
    main()
