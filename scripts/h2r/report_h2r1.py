#!/usr/bin/env python3
"""H2R-1 report: live-stream analytic vs frozen SOMA-analytic / G1-GT. Recovery OFF. THB OFF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = ("loco", "stoop", "reach", "carry")
H2R0 = Path("/data/home/chenxiangyu/robotics/Anybody/results/h2r_three_point_adapter/end_to_end")


def _sum(path: Path) -> dict:
    p = path / "plane" / "summary.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="/data/home/chenxiangyu/robotics/Anybody/results/next_phase/h2r1_live_human")
    ap.add_argument("--out", default="/data/home/chenxiangyu/robotics/Anybody/results/next_phase/h2r1_live_human")
    args = ap.parse_args()
    live_root = Path(args.live)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man_p = live_root / "manifests" / "live_clips.json"
    man = json.loads(man_p.read_text()) if man_p.is_file() else {}
    lat = (man.get("latency") or {})
    users = man.get("users") or {}
    rows = []
    n_go = 0
    for t in TASKS:
        g1 = _sum(H2R0 / "g1_gt" / t)
        an = _sum(H2R0 / "analytic" / t)
        lv = _sum(live_root / "live" / t)
        row = {
            "task": t,
            "g1_gt_sr": g1.get("sr_task"),
            "analytic_sr": an.get("sr_task"),
            "live_sr": lv.get("sr_task"),
            "live_sr5": lv.get("sr_5cm"),
            "live_poi": lv.get("poi_err"),
            "live_fail": lv.get("fail_frac"),
            "live_n": lv.get("n_episodes"),
            "fail_reasons": (lv.get("episode_monitor") or {}).get("fail_reasons"),
        }
        rows.append(row)
        if isinstance(row["live_sr"], (int, float)) and row["live_sr"] >= 0.80:
            n_go += 1
    # GO: 3/4 tasks stably driven
    latency_ok = True
    total_ms = lat.get("total_front_ms")
    if isinstance(total_ms, (int, float)) and total_ms > 80.0:
        latency_ok = False
    verdict = "LIVE_HUMAN_GO" if n_go >= 3 and latency_ok else "HOLD"
    scales = []
    for variant, tasks in (man.get("variants") or {}).items():
        for t, recs in (tasks or {}).items():
            for r in recs or []:
                if r.get("ok") and "s_h" in r:
                    scales.append(float(r["s_h"]))
    payload = {
        "verdict": verdict,
        "n_tasks_sr_ge_0.80": n_go,
        "no_physical_tracker": True,
        "rows": rows,
        "latency": lat,
        "users": users,
        "s_h": {
            "n": len(scales),
            "min": min(scales) if scales else None,
            "max": max(scales) if scales else None,
            "mean": (sum(scales) / len(scales)) if scales else None,
        },
        "recovery": False,
        "thb": False,
        "learned_h2r": False,
    }
    md = ["# H2R-1 Live Human Three-Point Simulation\n\n"]
    md.append("No physical tracker on this host. Live stream = timestamped SOMA Head/L/R with latency/jitter/drop.\n\n")
    md.append("| Task | SOMA-Analytic SR | Live-Human SR | SR@5 | Latency |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    lat_s = f"{total_ms:.1f} ms front" if isinstance(total_ms, (int, float)) else "n/a"
    for r in rows:
        def f(x):
            return "—" if x is None else f"{float(x):.3f}"
        md.append(f"| {r['task']} | {f(r['analytic_sr'])} | {f(r['live_sr'])} | {f(r['live_sr5'])} | {lat_s} |\n")
    md.append(f"\nusers (SOMA actor_uid stand-in): {users.get('n_actor_uids')}\n")
    md.append(f"scale s_h range: {payload['s_h']}\n")
    md.append(f"calibration: 1.5 s neutral of **this** live stream; robot scale fixed s_G1.\n")
    md.append(f"\n**Verdict: `{verdict}`**\n")
    (out / "REPORT.md").write_text("".join(md))
    (out / "verdict.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"verdict": verdict, "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
