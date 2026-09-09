#!/usr/bin/env python3
"""Parent-triggered recovery baseline on model_50000 Step-3 rollouts.

Same gate as R1a: R = R_E, detect R_E>=1 persist-3, release R_E<0.6 persist-3.
No GPU. Compares natural recovery of parent to R1a training-log snapshot if given.

ΔE horizons are reported two ways:
  complete  — event must still be alive at t0+τ (current R1a logger; survivor-biased)
  last_obs  — if the episode ends earlier, use E[t_end] − E[t0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_p2r_step3 import (
    ALL_TERRAINS,
    DT,
    MATRIX,
    REVERIFY,
    WARMUP,
    _load_root,
    _pct,
)
from rsl_rl.modules.intent_recovery import Q50_E, Q90_E, RecoveryRiskGate

H = {"dE25": 13, "dE50": 25, "dE100": 50}  # frames at 50 Hz
R_OFF = 0.6


def _events_from_e(e: np.ndarray, fail: bool, fail_t: int | None) -> list[dict]:
    """Replay hysteresis gate on a scalar E series. Returns trigger events."""
    e = np.asarray(e, dtype=np.float64)
    t_end = int(e.size - 1)
    if fail and fail_t is not None:
        t_end = min(t_end, int(fail_t))
    gate = RecoveryRiskGate(s_enabled=False)
    active_prev = False
    events: list[dict] = []
    cur = None
    for t in range(e.size):
        if t > t_end:
            break
        out = gate.step(
            torch.tensor([float(e[t])], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
        )
        active = bool(out["active"].item())
        if active and not active_prev and t >= WARMUP:
            cur = {
                "t0": t,
                "e0": float(e[t]),
                "released": False,
                "t_rel": None,
                "auc": 0.0,
                "n_act": 0,
                "failed": False,
            }
            events.append(cur)
        if cur is not None and not cur["released"] and not cur["failed"]:
            if active:
                cur["n_act"] += 1
                cur["auc"] += float(e[t]) * DT
            if active_prev and not active:
                cur["released"] = True
                cur["t_rel"] = t
        active_prev = active
    if fail and events:
        for ev in events:
            if not ev["released"] and ev["t0"] <= t_end:
                ev["failed"] = True
                ev["t_end"] = t_end
    for ev in events:
        t0 = ev["t0"]
        t_stop = t_end
        if ev["released"] and ev["t_rel"] is not None:
            t_stop = min(t_stop, int(ev["t_rel"]))
        ev["t_end"] = int(ev.get("t_end", t_stop))
        ev["Trec"] = (ev["n_act"] * DT) if ev["released"] else None
        for name, h in H.items():
            t_h = t0 + h
            if t_h <= t_end:
                ev[name + "_complete"] = float(e[t_h] - ev["e0"])
                ev[name + "_last"] = float(e[t_h] - ev["e0"])
            else:
                ev[name + "_complete"] = None
                t_use = min(t_end, max(t0, t_end))
                ev[name + "_last"] = float(e[t_use] - ev["e0"])
        ev["lived_1s"] = (t0 + H["dE100"]) <= t_end
        ev["fail_after"] = bool(fail) and (not ev["released"])
    return events


def _agg(events: list[dict], prefix: str = "") -> dict:
    n = len(events)
    n_rel = sum(1 for e in events if e["released"])
    n_fail = sum(1 for e in events if e["fail_after"])
    out = {
        "n_events": n,
        "n_release": n_rel,
        "n_fail_after": n_fail,
        "SR_recover": (n_rel / n) if n else float("nan"),
        "fail_after_trigger": (n_fail / n) if n else float("nan"),
    }
    trec = [e["Trec"] for e in events if e["Trec"] is not None]
    auc = [e["auc"] for e in events]
    out["T_rec"] = _pct(np.array(trec, dtype=np.float64), qs=(25, 50, 75))
    out["AUC_E"] = _pct(np.array(auc, dtype=np.float64), qs=(25, 50, 75))
    for name in H:
        c = [e[name + "_complete"] for e in events if e[name + "_complete"] is not None]
        last = [e[name + "_last"] for e in events]
        out[name + "_complete"] = _pct(np.array(c, dtype=np.float64), qs=(25, 50, 75)) | {"n": len(c)}
        out[name + "_last_obs"] = _pct(np.array(last, dtype=np.float64), qs=(25, 50, 75)) | {"n": len(last)}
        out[name + "_complete_frac"] = (len(c) / n) if n else float("nan")
    out["lived_1s_frac"] = (sum(1 for e in events if e["lived_1s"]) / n) if n else float("nan")
    return out


def _load_loco(root: Path) -> list[dict]:
    rows = []
    for ter in ALL_TERRAINS:
        rows.extend(_load_root(root, "loco", ter, "torso"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="/data/home/chenxiangyu/robotics/Anybody/results/p2r_r1a_diag/parent_recovery.json")
    ap.add_argument("--r1a-curves", type=str, default="", help="optional R1a probe root with loco/*/curves")
    args = ap.parse_args()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_loco(MATRIX)
    by_ter: dict[str, list[dict]] = {t: [] for t in ALL_TERRAINS}
    all_ev: list[dict] = []
    n_ep = {t: 0 for t in ALL_TERRAINS}
    n_trig_ep = {t: 0 for t in ALL_TERRAINS}
    for r in rows:
        ter = r["terrain"]
        n_ep[ter] += 1
        ev = _events_from_e(r["feats"]["E"], bool(r["fail"]), r["fail_t"])
        for e in ev:
            e["terrain"] = ter
            e["fail_ep"] = bool(r["fail"])
        if ev:
            n_trig_ep[ter] += 1
        by_ter[ter].extend(ev)
        all_ev.extend(ev)

    report = {
        "gate": {
            "Q50_E": Q50_E,
            "Q90_E": Q90_E,
            "R_off": R_OFF,
            "persist": 3,
            "s_enabled": False,
            "note": "R=R_E; same as R1a",
        },
        "source": str(MATRIX),
        "n_episodes": n_ep,
        "n_episodes_triggered": n_trig_ep,
        "logger_note": (
            "ΔE@τ complete counts only events still alive at t0+τ. "
            "Current R1a rec_dE100 uses this rule → 1s curve is survivor-biased. "
            "ΔE@τ last_obs imputes E[t_end] when the episode dies earlier."
        ),
        "all": _agg(all_ev),
        "by_terrain": {t: _agg(by_ter[t]) for t in ALL_TERRAINS},
        "healthy_flat_light": _agg(by_ter["plane"] + by_ter["light_rough"]),
        "challenge_slope_steps": _agg(by_ter["slope"] + by_ter["steps"]),
    }

    if args.r1a_curves:
        root = Path(args.r1a_curves)
        r1a_rows = []
        for ter in ALL_TERRAINS:
            cell = root / "loco" / ter
            if not cell.exists():
                # probe layout: OUT/ITER/loco/terrain
                alts = list(root.glob(f"**/loco/{ter}"))
                cell = alts[0] if alts else cell
            r1a_rows.extend(_load_root(cell.parent.parent if False else cell, "", "", "torso") if False else [])
        # load like _load_root but curves live under cell/curves
        r1a_ev = []
        r1a_by = {t: [] for t in ALL_TERRAINS}
        for ter in ALL_TERRAINS:
            csvs = list((root / "loco" / ter).glob("mapper_torso_s*.csv")) if (root / "loco" / ter).exists() else []
            if not csvs:
                csvs = list((root / ter).glob("mapper_torso_s*.csv"))
            for csv_path in csvs:
                import csv as _csv

                with csv_path.open(newline="", encoding="utf-8") as f:
                    for rec in _csv.DictReader(f):
                        clip = Path(rec["clip"]).name
                        stem = Path(clip).stem
                        seed = str(rec.get("seed", "42"))
                        curves = csv_path.parent / "curves"
                        npz = curves / f"mapper_torso_s{seed}_{stem}.npz"
                        if not npz.exists():
                            continue
                        z = np.load(npz)
                        e = np.asarray(z["e_torso"] if "e_torso" in z.files else z["E"], dtype=np.float64)
                        fail = int(float(rec.get("fail", 0)))
                        ep_len = int(float(rec.get("episode_length", e.size)))
                        fail_t = (ep_len - 1) if fail else None
                        ev = _events_from_e(e, bool(fail), fail_t)
                        for x in ev:
                            x["terrain"] = ter
                        r1a_by[ter].extend(ev)
                        r1a_ev.extend(ev)
        if r1a_ev:
            report["r1a_frozen"] = {
                "all": _agg(r1a_ev),
                "by_terrain": {t: _agg(r1a_by[t]) for t in ALL_TERRAINS},
            }

    outp.write_text(json.dumps(report, indent=2, default=str))
    a = report["all"]
    print("PARENT triggered recovery  n_events", a["n_events"])
    print("  SR_recover", round(a["SR_recover"], 4), "fail_after", round(a["fail_after_trigger"], 4))
    print("  T_rec median", a["T_rec"]["p50"], "AUC_E median", a["AUC_E"]["p50"])
    for k in ("dE25", "dE50", "dE100"):
        c, last = a[k + "_complete"], a[k + "_last_obs"]
        print(
            f"  {k} complete n={c['n']} median={c['p50']:.4f}  "
            f"last_obs n={last['n']} median={last['p50']:.4f}  "
            f"complete_frac={a[k + '_complete_frac']:.3f}"
        )
    print("  lived_1s_frac", round(a["lived_1s_frac"], 3))
    for t in ALL_TERRAINS:
        b = report["by_terrain"][t]
        print(
            f"  {t}: n={b['n_events']} SRrec={b['SR_recover']:.3f} "
            f"dE50_c={b['dE50_complete']['p50']:.4f} dE50_last={b['dE50_last_obs']['p50']:.4f}"
        )
    print("wrote", outp)


if __name__ == "__main__":
    main()
