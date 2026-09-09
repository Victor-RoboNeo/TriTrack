"""Merge per-terrain causal closed-loop CSVs into the P0 deliverable layout."""
from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np

ROOT = Path("/data/home/chenxiangyu/robotics/Anybody/results/causal_closed_loop")
MODES = ("hold", "mapper", "oracle")


def _parse_args():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--modes", default=",".join(MODES))
    return ap.parse_args()


def _fail_from_curve(npz_path: Path) -> tuple[int, int, str]:
    """Recompute fail from saved series (ignore the CSV ee_z false positives)."""
    d = np.load(npz_path)
    e_vis = d["e_vis"]
    ori = d["ori"]
    for t in range(10, len(e_vis)):
        if float(e_vis[t]) > 0.25:
            return 1, t + 1, "vis_err"
        if float(ori[t]) > 0.8:
            return 1, t + 1, "ori"
    return 0, int(len(e_vis)), "none"


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _mean(rows: list[dict], key: str) -> float:
    xs = [_f(r, key) for r in rows]
    xs = [x for x in xs if not math.isnan(x)]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _agg(rows: list[dict]) -> dict:
    return {
        "n_clips": len(rows),
        "sr_5cm": _mean(rows, "sr_5cm"),
        "sr_2cm": _mean(rows, "sr_2cm"),
        "e_kp_mean": _mean(rows, "e_kp_mean"),
        "e_kp_p50": _mean(rows, "e_kp_p50"),
        "e_kp_p90": _mean(rows, "e_kp_p90"),
        "e_torso": _mean(rows, "e_torso"),
        "e_lw": _mean(rows, "e_lw"),
        "e_rw": _mean(rows, "e_rw"),
        "e_xy": _mean(rows, "e_xy"),
        "e_z": _mean(rows, "e_z"),
        "wrist_err_mean": _mean(rows, "wrist_err_mean"),
        "wrist_err_p90": _mean(rows, "wrist_err_p90"),
        "ori_err_rad": _mean(rows, "ori_err_rad"),
        "ori_roll_rad": _mean(rows, "ori_roll_rad"),
        "ori_pitch_rad": _mean(rows, "ori_pitch_rad"),
        "fail": _mean(rows, "fail"),
        "episode_length": _mean(rows, "episode_length"),
        "action_smoothness": _mean(rows, "action_smoothness"),
        "abs_dz": _mean(rows, "abs_dz"),
        "mapper_err_0p1": _mean(rows, "mapper_err_0p1"),
        "mapper_err_0p2": _mean(rows, "mapper_err_0p2"),
        "mapper_err_0p3": _mean(rows, "mapper_err_0p3"),
        "mapper_err_0p5": _mean(rows, "mapper_err_0p5"),
        "fail_reasons": _count(rows, "fail_reason"),
    }


def _count(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = r.get(key, "") or "none"
        out[k] = out.get(k, 0) + 1
    return out


def _verdict(pooled: dict[str, dict]) -> dict:
    if "hold" not in pooled or "oracle" not in pooled:
        return {"p0_healthy": False, "reason": "need hold and oracle"}
    mapper_key = next((k for k in ("mapper_b", "mapper_a", "mapper") if k in pooled), None)
    if mapper_key is None:
        return {"p0_healthy": False, "reason": "no mapper mode"}
    h, m, o = pooled["hold"], pooled[mapper_key], pooled["oracle"]
    sr_gain = m["sr_5cm"] - h["sr_5cm"]
    sr_loss_vs_oracle = o["sr_5cm"] - m["sr_5cm"]
    fail_gap = m["fail"] - o["fail"]
    order_ok = h["sr_5cm"] < m["sr_5cm"] <= o["sr_5cm"] + 1e-9
    healthy = sr_gain > 0 and sr_loss_vs_oracle <= 0.15 and fail_gap < 0.03
    out = {
        "mapper_key": mapper_key,
        "hold_lt_mapper_lesssim_oracle": bool(order_ok),
        "sr5_mapper_gt_hold": bool(sr_gain > 0),
        "sr5_mapper_minus_hold": sr_gain,
        "sr5_oracle_minus_mapper": sr_loss_vs_oracle,
        "sr_loss_within_15pp": bool(sr_loss_vs_oracle <= 0.15),
        "fail_mapper_minus_oracle": fail_gap,
        "fail_gap_lt_3pp": bool(fail_gap < 0.03),
        "p0_healthy": bool(healthy and order_ok),
    }
    if "mapper_a" in pooled and "mapper_b" in pooled:
        out["sr5_b_minus_a"] = pooled["mapper_b"]["sr_5cm"] - pooled["mapper_a"]["sr_5cm"]
        out["fail_b_minus_a"] = pooled["mapper_b"]["fail"] - pooled["mapper_a"]["fail"]
        out["b_recovers_vs_a"] = bool(pooled["mapper_b"]["sr_5cm"] > pooled["mapper_a"]["sr_5cm"] + 0.05)
    for k in ("mapper_a_norest", "mapper_b_norest"):
        if k in pooled and k.replace("_norest", "") in pooled:
            base = k.replace("_norest", "")
            out[f"{k}_sr5"] = pooled[k]["sr_5cm"]
            out[f"{base}_with_gphi_sr5"] = pooled[base]["sr_5cm"]
            out[f"{k}_stands_gphi_falls"] = bool(pooled[k]["fail"] < 0.5 and pooled[base]["fail"] > 0.8)
    return out


def main() -> None:
    args = _parse_args()
    root = Path(args.root)
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    root.mkdir(parents=True, exist_ok=True)
    (root / "curves").mkdir(exist_ok=True)
    (root / "videos").mkdir(exist_ok=True)

    by_mode: dict[str, list[dict]] = {m: [] for m in modes}
    fieldnames = None
    for terrain in ("plane", "light_rough"):
        part = root / terrain
        for mode in modes:
            src = part / f"{mode}.csv"
            if not src.exists():
                print(f"[merge] missing {src}")
                continue
            rows = _read(src)
            for r in rows:
                clip_stem = Path(r.get("clip", "")).stem
                npz = part / "curves" / f"{mode}_{clip_stem}.npz"
                if npz.exists():
                    fail, ep, reason = _fail_from_curve(npz)
                    r["fail"] = str(fail)
                    r["episode_length"] = str(ep)
                    r["fail_reason"] = reason
            if rows and fieldnames is None:
                fieldnames = list(rows[0].keys())
            by_mode[mode].extend(rows)
        curves = part / "curves"
        if curves.is_dir():
            for p in curves.glob("*.npz"):
                shutil.copy2(p, root / "curves" / f"{terrain}_{p.name}")

    fieldnames = fieldnames or ["mode", "terrain", "clip"]
    for mode, rows in by_mode.items():
        out = root / f"{mode}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[merge] {out}  n={len(rows)}")

    pooled = {m: _agg(rows) for m, rows in by_mode.items() if rows}
    per_terrain = {}
    for terrain in ("plane", "light_rough"):
        per_terrain[terrain] = {
            m: _agg([r for r in rows if r.get("terrain") == terrain])
            for m, rows in by_mode.items()
            if any(r.get("terrain") == terrain for r in rows)
        }

    summary = {
        "pooled": pooled,
        "per_terrain": per_terrain,
        "verdict": _verdict(pooled),
        "main_table": {
            m: {
                "SR@5cm": pooled[m]["sr_5cm"],
                "Wrist Err": pooled[m]["wrist_err_mean"],
                "P90": pooled[m]["wrist_err_p90"],
                "Ori Err": pooled[m]["ori_err_rad"],
                "Fail": pooled[m]["fail"],
            }
            for m in modes
            if m in pooled
        },
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    v = summary["verdict"]
    print("[merge] main table:")
    for m in modes:
        if m not in summary["main_table"]:
            continue
        t = summary["main_table"][m]
        print(
            f"  {m:18s}  SR@5cm={t['SR@5cm']:.3f}  wrist={t['Wrist Err']:.4f}  "
            f"P90={t['P90']:.4f}  ori={t['Ori Err']:.3f}  fail={t['Fail']:.3f}"
        )
    print(f"[merge] P0 healthy={v.get('p0_healthy')}  {json.dumps(v)}")


if __name__ == "__main__":
    main()
