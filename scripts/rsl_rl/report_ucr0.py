#!/usr/bin/env python3
"""UCR-0 report: P(I_rec<0 | R_E triggered). Case A/B/C. No closed-loop."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TASKS = ("loco", "stoop", "reach", "carry")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    raise TypeError(type(o))


def _blk(i: np.ndarray) -> dict:
    i = np.asarray(i, dtype=np.float64)
    i = i[np.isfinite(i)]
    if i.size == 0:
        return {
            "n": 0,
            "P_I_lt_0": float("nan"),
            "P_I_lt_0.25cm": float("nan"),
            "median": float("nan"),
            "mean": float("nan"),
            "median_cm": float("nan"),
        }
    return {
        "n": int(i.size),
        "P_I_lt_0": float((i < 0).mean()),
        "P_I_lt_0.25cm": float((i < -0.0025).mean()),
        "median": float(np.median(i)),
        "mean": float(i.mean()),
        "median_cm": float(np.median(i) * 100.0),
        "mean_cm": float(i.mean() * 100.0),
    }


def load_raw(raw: Path) -> dict:
    blobs = []
    for p in sorted(raw.glob("*/lam_*/plane/clones.npz")):
        b = np.load(p, allow_pickle=True)
        if int(b["i_rm3"].shape[0]) == 0:
            print(f"[ucr0] empty {p}", flush=True)
            continue
        blobs.append(b)
        print(f"[ucr0] load {p} n={len(b['i_rm3'])}", flush=True)
    if not blobs:
        raise SystemExit(f"no UCR-0 clones in {raw}")
    out = {}
    for k in (
        "i_rm3", "i_ora", "j_p", "j_r", "j_ora", "re", "rs", "e0", "lam",
        "fail_p", "fail_r", "fail_ora", "theta", "theta_ora", "t", "seed", "sr_ep",
    ):
        out[k] = np.concatenate([b[k] for b in blobs], 0)
    out["task"] = np.concatenate([np.asarray(b["task"]).astype(str) for b in blobs], 0)
    out["clip"] = np.concatenate([np.asarray(b["clip"]).astype(str) for b in blobs], 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/ucr0_re_trigger_transfer/raw")
    ap.add_argument("--out", default="results/ucr0_re_trigger_transfer")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load_raw(Path(args.raw))

    dist = {}
    for t in TASKS:
        for lam in (0.0, 1.0, 2.0):
            m = (d["task"] == t) & (np.round(d["lam"]) == lam)
            dist[f"{t}/lam{int(lam)}"] = int(m.sum())

    rm3, ora = {}, {}
    rm3["pooled"] = _blk(d["i_rm3"])
    ora["pooled"] = _blk(d["i_ora"])
    for t in TASKS:
        m = d["task"] == t
        rm3[t] = _blk(d["i_rm3"][m])
        ora[t] = _blk(d["i_ora"][m])
        for lam in (0, 1, 2):
            ml = m & (np.round(d["lam"]) == float(lam))
            rm3[f"{t}/lam{lam}"] = _blk(d["i_rm3"][ml])
            ora[f"{t}/lam{lam}"] = _blk(d["i_ora"][ml])

    def ok_field(blk, pmin=0.65):
        return blk["n"] >= 8 and blk["P_I_lt_0"] >= pmin and blk["median"] < 0

    def weak_field(blk, pmax=0.55):
        return blk["n"] < 8 or blk["P_I_lt_0"] < pmax or not (blk["median"] < 0)

    a_ok = all(ok_field(rm3[t]) for t in TASKS)
    weak_tasks = [t for t in TASKS if not ok_field(rm3[t])]
    oracle_saves = [t for t in weak_tasks if ok_field(ora[t], pmin=0.80)]
    oracle_dead = [t for t in ("reach", "carry") if weak_field(ora[t], pmax=0.65)]

    if a_ok:
        case = "A"
        next_step = "Unified closed-loop with R_E -> frozen shared R-M3 one-burst -> Parent. Same threshold/persist all tasks."
    elif oracle_saves and not oracle_dead:
        case = "B"
        next_step = (
            "Local oracle shows recovery freedom on RE-trigger states the current R-M3 misses. "
            "Retrain ONE shared adapter on D_loco ∪ D_stoop ∪ D_reach ∪ D_carry. No task ID / experts."
        )
    else:
        case = "C"
        next_step = (
            "Even local oracle has little freedom on some RE-trigger manifolds. "
            "Do not add task-specific controllers. Keep Parent + unified R_E + shared R-M3 in recoverable region."
        )

    table = []
    for t in ("pooled",) + TASKS:
        table.append(
            {
                "task": t,
                "n": rm3[t]["n"],
                "P_rm3_lt_0": rm3[t]["P_I_lt_0"],
                "median_I_rm3_cm": rm3[t]["median_cm"],
                "P_ora_lt_0": ora[t]["P_I_lt_0"],
                "median_I_ora_cm": ora[t]["median_cm"],
            }
        )

    compact = {
        "phase": "UCR-0",
        "trigger": "R_E >= 1.0 persist 3 frames; RS diagnostic only; no task ID",
        "A_dump": {"n": int(len(d["i_rm3"])), "task_lam_n": dist, "n_clips": len(set(zip(d["task"], d["clip"])))},
        "B_rm3": rm3,
        "C_oracle": ora,
        "D_table": table,
        "E_case": {
            "case": case,
            "weak_rm3_tasks": weak_tasks,
            "oracle_saves": oracle_saves,
            "oracle_dead_reach_carry": oracle_dead,
            "next": next_step,
        },
        "F_stop": "No closed-loop. No R-M3 retrain in this phase.",
    }
    (out / "report_compact.json").write_text(json.dumps(compact, indent=2, default=_json_default), encoding="utf-8")
    (out / "summary_ucr0.json").write_text(json.dumps(compact, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"n": compact["A_dump"]["n"], "table": table, "case": case, "next": next_step}, indent=2, default=_json_default))
    print("[ucr0] wrote", out / "report_compact.json")


if __name__ == "__main__":
    main()
