#!/usr/bin/env python3
"""P2-R Step 2: deviation statistics from existing model_50000 rollouts.

Does not train. Does not launch Isaac. Reads P1 closed-loop curves already dumped
by eval_causal_closed_loop.py.

E_t is visible-intent tracking error under the canonical HeadHands mask:
  loco=torso, reach=head_right, stoop=vr, carry=vr.
Human-unconstrained keypoints never enter E.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

DT = 0.02  # sim.dt 0.005 * decimation 4
WARMUP = 10
PREFAIL_S = 1.0
PREFAIL_EXCLUDE_S = 0.10  # already collapsing
FAIL_TAIL_S = 0.10

HEADHANDS = {
    "loco": "torso",
    "reach": "head_right",
    "stoop": "vr",
    "carry": "vr",
}

SOURCES = {
    "p1_canonical_loco": Path(
        "/data/home/chenxiangyu/robotics/Anybody/results/p2c_fixed_eval/p1_50000"
    ),
    "p1_matrix": Path("/data/home/chenxiangyu/robotics/Anybody/results/p1_matrix"),
}

OUT = Path("/data/home/chenxiangyu/robotics/Anybody/results/p2r_step2")


def _pct(x: np.ndarray, qs=(5, 10, 25, 50, 75, 90, 95, 99)) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{q}": float("nan") for q in qs} | {"n": 0, "mean": float("nan")}
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} | {
        "n": int(x.size),
        "mean": float(x.mean()),
    }


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC of score: higher => positive class. Wilcoxon-Mann-Whitney."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    if pos.size > 20000:
        pos = rng.choice(pos, 20000, replace=False)
    if neg.size > 20000:
        neg = rng.choice(neg, 20000, replace=False)
    try:
        from scipy.stats import mannwhitneyu

        u = mannwhitneyu(pos, neg, alternative="greater", method="asymptotic").statistic
        return float(u / (pos.size * neg.size))
    except Exception:
        # P(pos > neg) + 0.5 P(eq) via searchsorted
        neg_s = np.sort(neg)
        gt = pos.size - np.searchsorted(neg_s, pos, side="right")
        ge = pos.size - np.searchsorted(neg_s, pos, side="left")
        ties = ge - gt
        return float((gt.sum() + 0.5 * ties.sum()) / (pos.size * neg.size))


def _youden_threshold(pos: np.ndarray, neg: np.ndarray) -> dict:
    """Threshold on score (higher = pos). Returns t, TPR, FPR, Youden."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return {"t": float("nan"), "tpr": float("nan"), "fpr": float("nan"), "youden": float("nan")}
    rng = np.random.default_rng(0)
    if pos.size > 15000:
        pos = rng.choice(pos, 15000, replace=False)
    if neg.size > 15000:
        neg = rng.choice(neg, 15000, replace=False)
    cands = np.unique(np.concatenate([np.percentile(neg, [50, 75, 90, 95, 99]), np.percentile(pos, [10, 25, 50, 75])]))
    best = None
    for t in cands:
        tpr = float((pos >= t).mean())
        fpr = float((neg >= t).mean())
        y = tpr - fpr
        row = {"t": float(t), "tpr": tpr, "fpr": fpr, "youden": y}
        if best is None or y > best["youden"]:
            best = row
    return best


def _e_from_series(z: np.lib.npyio.NpzFile, mask: str) -> np.ndarray:
    et = np.asarray(z["e_torso"], dtype=np.float64)
    el = np.asarray(z["e_lw"], dtype=np.float64)
    er = np.asarray(z["e_rw"], dtype=np.float64)
    n = (mask or "vr").lower()
    if n in ("torso", "kp5_torso"):
        return et
    if n in ("head_left",):
        return np.sqrt((et**2 + el**2) / 2.0)
    if n in ("head_right",):
        return np.sqrt((et**2 + er**2) / 2.0)
    # vr / carry: torso + both wrists, equal weight
    return np.sqrt((et**2 + el**2 + er**2) / 3.0)


def _load_cell(root: Path, task: str, terrain: str, mask: str) -> list[dict]:
    csvs = list((root / task / terrain).glob(f"mapper_{mask}_s*.csv"))
    if not csvs:
        csvs = list((root / task / terrain).glob("*.csv"))
        csvs = [p for p in csvs if f"_{mask}_" in p.name]
    rows = []
    for csv_path in csvs:
        with csv_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r.get("mask") or mask) != mask:
                    continue
                clip = Path(r["clip"]).name
                seed = r.get("seed", "42")
                stem = Path(clip).stem
                npz = root / task / terrain / "curves" / f"mapper_{mask}_s{seed}_{stem}.npz"
                if not npz.exists():
                    # some dumps omit mapper_ prefix variations
                    alts = list((root / task / terrain / "curves").glob(f"*_{mask}_s{seed}_{stem}.npz"))
                    if not alts:
                        continue
                    npz = alts[0]
                z = np.load(npz)
                e = _e_from_series(z, mask)
                fail = int(float(r.get("fail", 0)))
                ep_len = int(float(r.get("episode_length", e.size)))
                fail_t = (ep_len - 1) if fail else None
                rows.append(
                    {
                        "task": task,
                        "terrain": terrain,
                        "mask": mask,
                        "clip": clip,
                        "seed": int(seed),
                        "fail": fail,
                        "fail_reason": r.get("fail_reason", ""),
                        "fail_t": fail_t,
                        "sr_5cm": float(r.get("sr_5cm", "nan")),
                        "e": e,
                    }
                )
    return rows


def _split_frames(rows: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    buckets = defaultdict(list)
    d_buckets = defaultdict(list)
    ep_stats = []
    pre_s = int(round(PREFAIL_S / DT))
    ex_s = int(round(PREFAIL_EXCLUDE_S / DT))
    tail_s = int(round(FAIL_TAIL_S / DT))
    for row in rows:
        e = row["e"]
        t0 = min(WARMUP, e.size - 1)
        de = np.zeros_like(e)
        de[1:] = (e[1:] - e[:-1]) / DT
        peak = float(np.nanmax(e[t0:])) if e.size > t0 else float("nan")
        ep_stats.append(
            {
                "task": row["task"],
                "terrain": row["terrain"],
                "fail": row["fail"],
                "sr_5cm": row["sr_5cm"],
                "e_peak": peak,
                "e_mean": float(np.nanmean(e[t0:])) if e.size > t0 else float("nan"),
            }
        )
        if row["fail"] and row["fail_t"] is not None:
            ft = int(row["fail_t"])
            a = max(t0, ft - pre_s)
            b = max(a, ft - ex_s)
            if b > a:
                buckets["prefail"].append(e[a:b])
                d_buckets["prefail"].append(de[a:b])
            c = max(t0, ft - tail_s)
            buckets["fail"].append(e[c : ft + 1])
            d_buckets["fail"].append(de[c : ft + 1])
            # early part of failed episode (still "trying") — exclude from success
        else:
            buckets["success"].append(e[t0:])
            d_buckets["success"].append(de[t0:])
    out = {}
    for k in ("success", "prefail", "fail"):
        e_cat = np.concatenate(buckets[k]) if buckets[k] else np.array([], dtype=np.float64)
        d_cat = np.concatenate(d_buckets[k]) if d_buckets[k] else np.array([], dtype=np.float64)
        out[k] = {"E": e_cat, "dE": d_cat}
    return out, ep_stats


def _gate_duty(e: np.ndarray, t: float) -> float:
    e = np.asarray(e, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0 or not math.isfinite(t):
        return float("nan")
    return float((e >= t).mean())


def analyze_group(name: str, rows: list[dict]) -> dict:
    frames, ep_stats = _split_frames(rows)
    suc, pre, fail = frames["success"], frames["prefail"], frames["fail"]
    e_s, e_p, e_f = suc["E"], pre["E"], fail["E"]
    d_s, d_p, d_f = suc["dE"], pre["dE"], fail["dE"]
    n_ep = len(rows)
    n_fail = sum(r["fail"] for r in rows)
    auroc_e = _auroc(e_p, e_s)
    auroc_de = _auroc(d_p, d_s)
    you_e = _youden_threshold(e_p, e_s)
    you_de = _youden_threshold(d_p, d_s)
    suc_pct = _pct(e_s)
    e_off = suc_pct["p90"]
    e_dead = suc_pct["p90"]
    e_full = suc_pct["p99"] if math.isfinite(suc_pct["p99"]) else suc_pct["p95"]
    # E_on: Youden, but at least E_off
    e_on = you_e["t"]
    if math.isfinite(e_on) and math.isfinite(e_off):
        e_on = max(e_on, e_off)
    return {
        "name": name,
        "n_episodes": n_ep,
        "n_fail_episodes": n_fail,
        "fail_rate": n_fail / n_ep if n_ep else float("nan"),
        "E_success": suc_pct,
        "E_prefail": _pct(e_p),
        "E_fail": _pct(e_f),
        "dE_success": _pct(d_s),
        "dE_prefail": _pct(d_p),
        "dE_fail": _pct(d_f),
        "auroc_E_prefail_vs_success": auroc_e,
        "auroc_dE_prefail_vs_success": auroc_de,
        "youden_E": you_e,
        "youden_dE": you_de,
        "gate_proposal": {
            "E_dead_m": e_dead,
            "E_off_m": e_off,
            "E_on_m": e_on,
            "E_full_m": e_full,
            "note": "E_dead=E_off=P90(success). E_on=max(Youden, E_off). E_full=P99(success).",
        },
        "duty_if_E_dead": {
            "success": _gate_duty(e_s, e_dead),
            "prefail": _gate_duty(e_p, e_dead),
            "fail": _gate_duty(e_f, e_dead),
        },
        "duty_if_E_on": {
            "success": _gate_duty(e_s, e_on),
            "prefail": _gate_duty(e_p, e_on),
            "fail": _gate_duty(e_f, e_on),
        },
        "ep_peak_success": _pct(np.array([x["e_peak"] for x in ep_stats if not x["fail"]])),
        "ep_peak_fail": _pct(np.array([x["e_peak"] for x in ep_stats if x["fail"]])),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "dt": DT,
        "warmup_steps": WARMUP,
        "prefail_s": PREFAIL_S,
        "parent": "model_50000 + Mapper-B",
        "E_def": "RMS of visible intent keypoints only (HeadHands mask)",
        "groups": {},
    }

    # Canonical loco from later P1 protocol (the 53.5/54.9/34.8/31.6 table).
    canon_rows = []
    for terrain in ("plane", "light_rough", "slope", "steps"):
        rows = _load_cell(SOURCES["p1_canonical_loco"], "loco", terrain, "torso")
        g = analyze_group(f"canonical_loco/{terrain}", rows)
        report["groups"][g["name"]] = g
        canon_rows.extend(rows)
    report["groups"]["canonical_loco/all"] = analyze_group("canonical_loco/all", canon_rows)

    # Full HeadHands matrix (4 tasks x 4 terrains, seed 42-46).
    hh_by_task = defaultdict(list)
    hh_by_terrain = defaultdict(list)
    hh_all = []
    for task, mask in HEADHANDS.items():
        for terrain in ("plane", "light_rough", "slope", "steps"):
            rows = _load_cell(SOURCES["p1_matrix"], task, terrain, mask)
            name = f"matrix/{task}/{terrain}"
            g = analyze_group(name, rows)
            report["groups"][name] = g
            hh_by_task[task].extend(rows)
            hh_by_terrain[terrain].extend(rows)
            hh_all.extend(rows)
    for task, rows in hh_by_task.items():
        report["groups"][f"matrix/{task}/all"] = analyze_group(f"matrix/{task}/all", rows)
    for terrain, rows in hh_by_terrain.items():
        report["groups"][f"matrix/all/{terrain}"] = analyze_group(f"matrix/all/{terrain}", rows)
    report["groups"]["matrix/all"] = analyze_group("matrix/all", hh_all)

    # Zero-recovery SR table from canonical loco summaries if present.
    sr = {}
    for terrain in ("plane", "light_rough", "slope", "steps"):
        p = SOURCES["p1_canonical_loco"] / "loco" / terrain / "summary.json"
        if p.exists():
            payload = json.loads(p.read_text())
            cell = payload["modes"]["mapper"]["torso"]["s42"]
            sr[terrain] = {
                "sr_5cm": cell["sr_5cm"],
                "sr_2cm": cell["sr_2cm"],
                "fail": cell.get("fail"),
                "e_kp_mean": cell.get("e_kp_mean"),
            }
    report["zero_recovery_loco_sr"] = sr

    out_json = OUT / "deviation_stats.json"
    out_json.write_text(json.dumps(report, indent=2))

    # Compact CSV for the groups that matter.
    csv_path = OUT / "gate_evidence.csv"
    fields = [
        "group",
        "n_ep",
        "fail_rate",
        "E_suc_p50",
        "E_suc_p90",
        "E_suc_p95",
        "E_pre_p50",
        "E_pre_p90",
        "E_fail_p50",
        "dE_suc_p50",
        "dE_pre_p50",
        "auroc_E",
        "auroc_dE",
        "youden_E_t",
        "youden_E_tpr",
        "youden_E_fpr",
        "duty_suc_at_p90",
        "duty_pre_at_p90",
        "E_dead",
        "E_on",
        "E_full",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, g in report["groups"].items():
            gp = g["gate_proposal"]
            w.writerow(
                {
                    "group": name,
                    "n_ep": g["n_episodes"],
                    "fail_rate": g["fail_rate"],
                    "E_suc_p50": g["E_success"]["p50"],
                    "E_suc_p90": g["E_success"]["p90"],
                    "E_suc_p95": g["E_success"]["p95"],
                    "E_pre_p50": g["E_prefail"]["p50"],
                    "E_pre_p90": g["E_prefail"]["p90"],
                    "E_fail_p50": g["E_fail"]["p50"],
                    "dE_suc_p50": g["dE_success"]["p50"],
                    "dE_pre_p50": g["dE_prefail"]["p50"],
                    "auroc_E": g["auroc_E_prefail_vs_success"],
                    "auroc_dE": g["auroc_dE_prefail_vs_success"],
                    "youden_E_t": g["youden_E"]["t"],
                    "youden_E_tpr": g["youden_E"]["tpr"],
                    "youden_E_fpr": g["youden_E"]["fpr"],
                    "duty_suc_at_p90": g["duty_if_E_dead"]["success"],
                    "duty_pre_at_p90": g["duty_if_E_dead"]["prefail"],
                    "E_dead": gp["E_dead_m"],
                    "E_on": gp["E_on_m"],
                    "E_full": gp["E_full_m"],
                }
            )
    print(f"wrote {out_json}")
    print(f"wrote {csv_path}")
    g = report["groups"]["canonical_loco/all"]
    print(
        "canonical_loco/all",
        f"n={g['n_episodes']} fail={g['fail_rate']:.3f}",
        f"E_suc_p90={g['E_success']['p90']:.4f}",
        f"E_pre_p50={g['E_prefail']['p50']:.4f}",
        f"AUROC_E={g['auroc_E_prefail_vs_success']:.3f}",
        f"AUROC_dE={g['auroc_dE_prefail_vs_success']:.3f}",
    )
    g2 = report["groups"]["matrix/all"]
    print(
        "matrix/all",
        f"n={g2['n_episodes']} fail={g2['fail_rate']:.3f}",
        f"E_suc_p90={g2['E_success']['p90']:.4f}",
        f"AUROC_E={g2['auroc_E_prefail_vs_success']:.3f}",
        f"AUROC_dE={g2['auroc_dE_prefail_vs_success']:.3f}",
    )


if __name__ == "__main__":
    main()
