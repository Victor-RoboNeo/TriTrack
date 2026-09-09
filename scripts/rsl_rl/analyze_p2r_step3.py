#!/usr/bin/env python3
"""P2-R Step 3: causal recovery-risk features from existing model_50000 rollouts.

No GPU, no PPO, no r_eta. Future labels never enter features.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DT = 0.02
WARMUP = 10
E5 = 0.05
E13 = 0.13
HORIZON = int(round(0.5 / DT))  # 25
WIN = {"100": 5, "200": 10, "400": 20}

HEADHANDS = {
    "loco": "torso",
    "reach": "head_right",
    "stoop": "vr",
    "carry": "vr",
}
HEALTHY_TERRAINS = ("plane", "light_rough")
CHALLENGE_TERRAINS = ("slope", "steps")
ALL_TERRAINS = HEALTHY_TERRAINS + CHALLENGE_TERRAINS

REVERIFY = Path("/data/home/chenxiangyu/robotics/Anybody/results/p1_reverify")
MATRIX = Path("/data/home/chenxiangyu/robotics/Anybody/results/p1_matrix")
OUT = Path("/data/home/chenxiangyu/robotics/Anybody/results/p2r_step3")


def _pct(x, qs=(10, 25, 50, 75, 90, 95)) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{q}": float("nan") for q in qs} | {"n": 0, "mean": float("nan")}
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} | {
        "n": int(x.size),
        "mean": float(x.mean()),
    }


def _auroc(pos, neg) -> float:
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    if pos.size > 25000:
        pos = rng.choice(pos, 25000, replace=False)
    if neg.size > 25000:
        neg = rng.choice(neg, 25000, replace=False)
    from scipy.stats import mannwhitneyu

    u = mannwhitneyu(pos, neg, alternative="greater", method="asymptotic").statistic
    return float(u / (pos.size * neg.size))


def _lin_slope(y: np.ndarray, n: int) -> np.ndarray:
    """Causal slope at each t using y[t-n+1:t+1]. Units: /second."""
    y = np.asarray(y, dtype=np.float64)
    out = np.full_like(y, np.nan)
    if n < 2:
        return out
    tgrid = np.arange(n, dtype=np.float64) * DT
    tgrid = tgrid - tgrid.mean()
    denom = float((tgrid**2).sum())
    for t in range(n - 1, y.size):
        w = y[t - n + 1 : t + 1]
        if not np.isfinite(w).all():
            continue
        out[t] = float(((w - w.mean()) * tgrid).sum() / denom)
    return out


def _causal_mean(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    out = np.full_like(y, np.nan)
    c = np.cumsum(np.insert(y, 0, 0.0))
    for t in range(n - 1, y.size):
        out[t] = (c[t + 1] - c[t + 1 - n]) / n
    return out


def _causal_max(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    out = np.full_like(y, np.nan)
    for t in range(n - 1, y.size):
        out[t] = float(np.nanmax(y[t - n + 1 : t + 1]))
    return out


def _e_from_scalars(z, mask: str) -> np.ndarray:
    et = np.asarray(z["e_torso"], dtype=np.float64)
    el = np.asarray(z["e_lw"], dtype=np.float64)
    er = np.asarray(z["e_rw"], dtype=np.float64)
    n = (mask or "vr").lower()
    if n in ("torso", "kp5_torso"):
        return et
    if n == "head_left":
        return np.sqrt((et**2 + el**2) / 2.0)
    if n == "head_right":
        return np.sqrt((et**2 + er**2) / 2.0)
    return np.sqrt((et**2 + el**2 + er**2) / 3.0)


def _e_components(z, mask: str) -> dict[str, np.ndarray]:
    torso = np.asarray(z["torso_xyz"], dtype=np.float64)
    lw = np.asarray(z["lw_xyz"], dtype=np.float64)
    rw = np.asarray(z["rw_xyz"], dtype=np.float64)
    has_goal = "goal_torso" in z.files
    if has_goal:
        gt = np.asarray(z["goal_torso"], dtype=np.float64)
        gl = np.asarray(z["goal_lw"], dtype=np.float64)
        gr = np.asarray(z["goal_rw"], dtype=np.float64)
        et = np.linalg.norm(torso - gt, axis=-1)
        el = np.linalg.norm(lw - gl, axis=-1)
        er = np.linalg.norm(rw - gr, axis=-1)
        et_xy = np.linalg.norm((torso - gt)[:, :2], axis=-1)
        et_z = np.abs(torso[:, 2] - gt[:, 2])
        n = (mask or "vr").lower()
        if n in ("torso", "kp5_torso"):
            e, exy, ez = et, et_xy, et_z
        elif n == "head_left":
            e = np.sqrt((et**2 + el**2) / 2.0)
            exy = np.sqrt((et_xy**2 + np.linalg.norm((lw - gl)[:, :2], axis=-1) ** 2) / 2.0)
            ez = np.sqrt((et_z**2 + np.abs(lw[:, 2] - gl[:, 2]) ** 2) / 2.0)
        elif n == "head_right":
            e = np.sqrt((et**2 + er**2) / 2.0)
            exy = np.sqrt((et_xy**2 + np.linalg.norm((rw - gr)[:, :2], axis=-1) ** 2) / 2.0)
            ez = np.sqrt((et_z**2 + np.abs(rw[:, 2] - gr[:, 2]) ** 2) / 2.0)
        else:
            e = np.sqrt((et**2 + el**2 + er**2) / 3.0)
            exy = np.sqrt(
                (
                    et_xy**2
                    + np.linalg.norm((lw - gl)[:, :2], axis=-1) ** 2
                    + np.linalg.norm((rw - gr)[:, :2], axis=-1) ** 2
                )
                / 3.0
            )
            ez = np.sqrt(
                (et_z**2 + np.abs(lw[:, 2] - gl[:, 2]) ** 2 + np.abs(rw[:, 2] - gr[:, 2]) ** 2) / 3.0
            )
        dz_torso_cmd = torso[:, 2] - gt[:, 2]
    else:
        e = _e_from_scalars(z, mask)
        exy = np.full_like(e, np.nan)
        ez = np.full_like(e, np.nan)
        dz_torso_cmd = np.full_like(e, np.nan)
    root_z = np.asarray(z["root_z"], dtype=np.float64)
    roll = np.asarray(z["roll"], dtype=np.float64)
    pitch = np.asarray(z["pitch"], dtype=np.float64)
    v_root = np.zeros_like(root_z)
    v_root[1:] = (root_z[1:] - root_z[:-1]) / DT
    dz_root_torso = root_z - torso[:, 2]
    v_e_inst = np.zeros_like(e)
    v_e_inst[1:] = (e[1:] - e[:-1]) / DT
    feats = {
        "E": e,
        "E_xy": exy,
        "E_z": ez,
        "vE_inst": v_e_inst,
        "v_root_z": v_root,
        "dz_root_torso": dz_root_torso,
        "dz_torso_cmd": dz_torso_cmd,
        "roll": roll,
        "pitch": pitch,
        "abs_v_root_z": np.abs(v_root),
        "abs_roll": np.abs(roll),
        "abs_pitch": np.abs(pitch),
    }
    for name, nwin in WIN.items():
        feats[f"vE_{name}"] = _lin_slope(e, nwin)
        feats[f"Emean_{name}"] = _causal_mean(e, nwin)
        feats[f"Emax_{name}"] = _causal_max(e, nwin)
    feats["vE_xy_200"] = _lin_slope(exy, WIN["200"])
    feats["v_roll_200"] = _lin_slope(roll, WIN["200"])
    feats["v_pitch_200"] = _lin_slope(pitch, WIN["200"])
    return feats


def _load_root(root: Path, task: str, terrain: str, mask: str) -> list[dict]:
    rows = []
    csvs = list((root / task / terrain).glob(f"mapper_{mask}_s*.csv"))
    for csv_path in csvs:
        with csv_path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r.get("mask") or mask) != mask:
                    continue
                clip = Path(r["clip"]).name
                seed = str(r.get("seed", "42"))
                stem = Path(clip).stem
                npz = root / task / terrain / "curves" / f"mapper_{mask}_s{seed}_{stem}.npz"
                if not npz.exists():
                    alts = list((root / task / terrain / "curves").glob(f"*_{mask}_s{seed}_{stem}.npz"))
                    if not alts:
                        continue
                    npz = alts[0]
                z = np.load(npz)
                feats = _e_components(z, mask)
                fail = int(float(r.get("fail", 0)))
                ep_len = int(float(r.get("episode_length", feats["E"].size)))
                fail_t = (ep_len - 1) if fail else None
                rows.append(
                    {
                        "src": root.name,
                        "task": task,
                        "terrain": terrain,
                        "mask": mask,
                        "clip": clip,
                        "clip_id": stem.split("_", 1)[0] if stem[:2].isdigit() else stem,
                        "seed": int(seed),
                        "fail": fail,
                        "fail_reason": r.get("fail_reason", ""),
                        "fail_t": fail_t,
                        "feats": feats,
                    }
                )
    return rows


def _healthy_frames(rows: list[dict], key: str, terrains=HEALTHY_TERRAINS) -> np.ndarray:
    xs = []
    for row in rows:
        if row["fail"] or row["terrain"] not in terrains:
            continue
        x = row["feats"][key][WARMUP:]
        xs.append(x[np.isfinite(x)])
    return np.concatenate(xs) if xs else np.array([], dtype=np.float64)


def _q50_q90(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(x, 50)), float(np.percentile(x, 90))


def _norm(x, q50, q90):
    scale = max(q90 - q50, 1e-6)
    return (x - q50) / scale


def _first_trigger(e: np.ndarray, th: float, persist: int, t_end: int) -> int | None:
    run = 0
    for t in range(WARMUP, t_end + 1):
        if e[t] >= th:
            run += 1
            if run >= persist:
                return t - persist + 1
        else:
            run = 0
    return None


def _lead_sweep(rows: list[dict], key: str, persist: int = 1) -> list[dict]:
    out = []
    healthy = [r for r in rows if (not r["fail"]) and r["terrain"] in HEALTHY_TERRAINS]
    fails = [r for r in rows if r["fail"]]
    for cm in range(5, 21):
        th = cm / 100.0
        # healthy duty: frame fraction
        n_h = d_h = 0
        for r in healthy:
            e = r["feats"][key]
            sl = e[WARMUP:]
            n_h += sl.size
            if persist <= 1:
                d_h += int((sl >= th).sum())
            else:
                run = 0
                for v in sl:
                    run = run + 1 if v >= th else 0
                    if run >= persist:
                        d_h += 1
        duty = d_h / n_h if n_h else float("nan")
        leads = []
        hits = 0
        for r in fails:
            ft = int(r["fail_t"])
            t0 = _first_trigger(r["feats"][key], th, persist, ft)
            if t0 is None:
                continue
            hits += 1
            leads.append((ft - t0) * DT)
        rec = hits / len(fails) if fails else float("nan")
        lp = _pct(np.array(leads), qs=(10, 50, 90))
        out.append(
            {
                "th_cm": cm,
                "key": key,
                "persist": persist,
                "healthy_duty": duty,
                "prefail_recall": rec,
                "n_fail": len(fails),
                "n_triggered": hits,
                "lead_median_s": lp["p50"],
                "lead_p10_s": lp["p10"],
                "lead_p90_s": lp["p90"],
                "lead_mean_s": lp["mean"],
            }
        )
    return out


def _hysteresis_duty(e: np.ndarray, on: float, off: float, persist: int, t_end: int | None) -> tuple[float, int | None]:
    active = False
    run = 0
    on_frames = 0
    first = None
    last = e.size - 1 if t_end is None else t_end
    n = max(0, last - WARMUP + 1)
    for t in range(WARMUP, last + 1):
        if not active:
            if e[t] >= on:
                run += 1
                if run >= persist:
                    active = True
                    if first is None:
                        first = t - persist + 1
            else:
                run = 0
        else:
            if e[t] < off:
                active = False
                run = 0
        if active:
            on_frames += 1
    duty = on_frames / n if n else float("nan")
    return duty, first


def _self_recover_labels(rows: list[dict], band=(E5, E13), horizon=HORIZON):
    """Frame-level and event-entry samples in (5cm, 13cm). Label from future only."""
    frames = []
    events = []
    for row in rows:
        e = row["feats"]["E"]
        t_end = int(row["fail_t"]) if row["fail"] else e.size - 1
        in_band = False
        event_start = None
        for t in range(WARMUP, t_end - horizon):
            inside = band[0] < e[t] < band[1]
            fut = e[t : t + horizon + 1]
            self_rec = bool(np.nanmin(fut) < E5)
            grew = bool(fut[-1] > e[t] + 0.005)
            persistent = (not self_rec) or grew
            # user: self-recover = min future < 5cm; persistent = stay >5 or increase
            persistent_strict = (not self_rec)
            sample = {
                "row": row,
                "t": t,
                "self_recover": self_rec,
                "persistent": persistent_strict,
                "grew": grew,
            }
            if inside:
                frames.append(sample)
                if not in_band:
                    in_band = True
                    event_start = t
                    events.append(sample)
            else:
                in_band = False
                event_start = None
    return frames, events


def _feat_at(sample: dict, key: str) -> float:
    return float(sample["row"]["feats"][key][sample["t"]])


def _auroc_feat(samples: list[dict], key: str, pos="persistent") -> dict:
    pos_v, neg_v = [], []
    for s in samples:
        v = _feat_at(s, key)
        if not np.isfinite(v):
            continue
        if s[pos]:
            pos_v.append(v)
        else:
            neg_v.append(v)
    return {
        "key": key,
        "auroc": _auroc(np.array(pos_v), np.array(neg_v)),
        "n_pos": len(pos_v),
        "n_neg": len(neg_v),
        "pos_p50": float(np.median(pos_v)) if pos_v else float("nan"),
        "neg_p50": float(np.median(neg_v)) if neg_v else float("nan"),
    }


def _fail_score_frames(rows: list[dict], score_key: str, pre_s=1.0, exclude_s=0.10):
    suc, pre = [], []
    pre_n = int(round(pre_s / DT))
    ex_n = int(round(exclude_s / DT))
    for row in rows:
        x = row["feats"][score_key]
        if row["fail"] and row["fail_t"] is not None:
            ft = int(row["fail_t"])
            a = max(WARMUP, ft - pre_n)
            b = max(a, ft - ex_n)
            sl = x[a:b]
            pre.append(sl[np.isfinite(sl)])
        elif not row["fail"]:
            sl = x[WARMUP:]
            suc.append(sl[np.isfinite(sl)])
    p = np.concatenate(pre) if pre else np.array([])
    n = np.concatenate(suc) if suc else np.array([])
    return p, n


def _combo_score(row, q):
    """R = tilde_E + beta*relu(tilde_vE200) + gamma*tilde_S, using healthy quantiles q."""
    e = row["feats"]["E"]
    ve = row["feats"]["vE_200"]
    s = np.maximum.reduce(
        [
            _norm(row["feats"]["abs_v_root_z"], q["abs_v_root_z"][0], q["abs_v_root_z"][1]),
            _norm(row["feats"]["abs_roll"], q["abs_roll"][0], q["abs_roll"][1]),
            _norm(row["feats"]["abs_pitch"], q["abs_pitch"][0], q["abs_pitch"][1]),
        ]
    )
    te = _norm(e, q["E"][0], q["E"][1])
    tv = _norm(ve, q["vE_200"][0], q["vE_200"][1])
    tv = np.where(np.isfinite(tv), np.maximum(tv, 0.0), 0.0)
    return te, tv, s


def _clip_groups(rows: list[dict]) -> dict[str, list[dict]]:
    g = defaultdict(list)
    for r in rows:
        g[r["clip"]].append(r)
    return g


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"dt": DT, "warmup": WARMUP, "notes": []}

    loco_rev = []
    for ter in ALL_TERRAINS:
        loco_rev.extend(_load_root(REVERIFY, "loco", ter, "torso"))
    loco_mat = []
    for ter in ALL_TERRAINS:
        loco_mat.extend(_load_root(MATRIX, "loco", ter, "torso"))
    stoop_rev = []
    for ter in ALL_TERRAINS:
        stoop_rev.extend(_load_root(REVERIFY, "stoop", ter, "vr"))
    stoop_mat = []
    for ter in ALL_TERRAINS:
        stoop_mat.extend(_load_root(MATRIX, "stoop", ter, "vr"))

    report["n"] = {
        "loco_reverify": len(loco_rev),
        "loco_matrix": len(loco_mat),
        "stoop_reverify": len(stoop_rev),
        "stoop_matrix": len(stoop_mat),
        "loco_rev_fail": sum(r["fail"] for r in loco_rev),
        "loco_mat_fail": sum(r["fail"] for r in loco_mat),
        "stoop_rev_fail": sum(r["fail"] for r in stoop_rev),
        "stoop_mat_fail": sum(r["fail"] for r in stoop_mat),
    }

    # Healthy envelope from canonical loco Flat+Light.
    qE50, qE90 = _q50_q90(_healthy_frames(loco_rev, "E"))
    q = {}
    for k in ("E", "E_xy", "E_z", "vE_200", "abs_v_root_z", "abs_roll", "abs_pitch"):
        q[k] = _q50_q90(_healthy_frames(loco_rev, k))
    report["healthy_flat_light_loco"] = {
        k: {"q50": a, "q90": b} for k, (a, b) in q.items()
    }

    # ---- 1. lead-time sweep ----
    loco_fl_failpool = [r for r in loco_rev if r["terrain"] in ALL_TERRAINS]
    lead = {}
    for key in ("E", "E_xy", "E_z"):
        for persist in (1, 3, 5):
            lead[f"{key}_p{persist}"] = _lead_sweep(loco_fl_failpool, key, persist)
    report["lead_sweep_loco_reverify"] = lead

    # GO snapshot at 13cm persist 3
    def _pick(rows_s, th_cm, persist):
        for row in rows_s:
            if row["th_cm"] == th_cm:
                return row
        return {}

    report["go_loco_E13_p3"] = _pick(lead["E_p3"], 13, 3)
    report["go_loco_E13_p1"] = _pick(lead["E_p1"], 13, 1)
    report["go_loco_E5_p5"] = _pick(lead["E_p5"], 5, 5)

    # Hysteresis at E_on=13, E_off=8 (P75-ish) persist 3
    hy_on, hy_off, hy_p = 0.13, 0.08, 3
    hy_duty_h, hy_leads = [], []
    for r in loco_rev:
        e = r["feats"]["E"]
        if r["fail"]:
            d, first = _hysteresis_duty(e, hy_on, hy_off, hy_p, int(r["fail_t"]))
            if first is not None:
                hy_leads.append((int(r["fail_t"]) - first) * DT)
        elif r["terrain"] in HEALTHY_TERRAINS:
            d, _ = _hysteresis_duty(e, hy_on, hy_off, hy_p, None)
            hy_duty_h.append(d)
    report["hysteresis_Eon13_Eoff8_p3"] = {
        "healthy_duty_mean": float(np.mean(hy_duty_h)) if hy_duty_h else float("nan"),
        "lead": _pct(np.array(hy_leads)),
        "fail_recall": (len(hy_leads) / sum(r["fail"] for r in loco_rev)) if loco_rev else float("nan"),
    }

    # ---- 2. self-recovery 5-13cm on loco Flat+Light (and all loco) ----
    inner_rows = [r for r in loco_rev if r["terrain"] in HEALTHY_TERRAINS]
    frames_fl, events_fl = _self_recover_labels(inner_rows)
    frames_all, events_all = _self_recover_labels(loco_rev)
    inner_keys = [
        "E",
        "vE_inst",
        "vE_100",
        "vE_200",
        "vE_400",
        "Emean_200",
        "Emax_200",
        "E_xy",
        "vE_xy_200",
        "abs_v_root_z",
        "abs_roll",
        "abs_pitch",
    ]
    report["self_recover"] = {
        "flat_light_frames": {
            "n": len(frames_fl),
            "n_self": sum(s["self_recover"] for s in frames_fl),
            "n_persistent": sum(s["persistent"] for s in frames_fl),
            "frac_self": (sum(s["self_recover"] for s in frames_fl) / len(frames_fl)) if frames_fl else float("nan"),
            "auroc": [_auroc_feat(frames_fl, k) for k in inner_keys],
        },
        "flat_light_events": {
            "n": len(events_fl),
            "n_self": sum(s["self_recover"] for s in events_fl),
            "n_persistent": sum(s["persistent"] for s in events_fl),
            "frac_self": (sum(s["self_recover"] for s in events_fl) / len(events_fl)) if events_fl else float("nan"),
            "auroc": [_auroc_feat(events_fl, k) for k in inner_keys],
        },
        "all_terrain_frames": {
            "n": len(frames_all),
            "n_self": sum(s["self_recover"] for s in frames_all),
            "n_persistent": sum(s["persistent"] for s in frames_all),
            "frac_self": (sum(s["self_recover"] for s in frames_all) / len(frames_all)) if frames_all else float("nan"),
            "auroc": [_auroc_feat(frames_all, k) for k in inner_keys],
        },
        "all_terrain_events": {
            "n": len(events_all),
            "n_self": sum(s["self_recover"] for s in events_all),
            "n_persistent": sum(s["persistent"] for s in events_all),
            "frac_self": (sum(s["self_recover"] for s in events_all) / len(events_all)) if events_all else float("nan"),
            "auroc": [_auroc_feat(events_all, k) for k in inner_keys],
        },
    }

    # Combined past-only score for inner region: vE_200 (diverging) is the causal one.
    # R_inner = tilde_E + beta * relu(vE_200)
    best_inner = None
    for beta in (0.0, 0.5, 1.0, 2.0, 4.0):
        pos, neg = [], []
        for s in events_all:
            te = _norm(_feat_at(s, "E"), q["E"][0], q["E"][1])
            tv = _feat_at(s, "vE_200")
            if not np.isfinite(tv):
                tv = 0.0
            tvn = max(_norm(tv, q["vE_200"][0], q["vE_200"][1]), 0.0)
            R = te + beta * tvn
            (pos if s["persistent"] else neg).append(R)
        a = _auroc(np.array(pos), np.array(neg))
        row = {"beta": beta, "auroc_event_all": a, "n_pos": len(pos), "n_neg": len(neg)}
        if best_inner is None or (np.isfinite(a) and a > best_inner["auroc_event_all"]):
            best_inner = row
    report["self_recover"]["best_R_event_all"] = best_inner

    # ---- 3. stoop precursor (matrix has more fails) ----
    stoop_sets = {"reverify": stoop_rev, "matrix": stoop_mat}
    stoop_out = {}
    stoop_keys = [
        "E",
        "E_xy",
        "E_z",
        "vE_200",
        "Emax_200",
        "abs_v_root_z",
        "dz_torso_cmd",
        "dz_root_torso",
        "abs_roll",
        "abs_pitch",
        "v_roll_200",
        "v_pitch_200",
    ]
    for name, rows in stoop_sets.items():
        block = {"n": len(rows), "n_fail": sum(r["fail"] for r in rows), "auroc": {}}
        for k in stoop_keys:
            p, n = _fail_score_frames(rows, k)
            block["auroc"][k] = {"auroc": _auroc(p, n), "n_pre": int(p.size), "n_suc": int(n.size)}
        # additive R
        q_s = {k: _q50_q90(_healthy_frames(rows, k)) for k in ("E", "vE_200", "abs_v_root_z", "abs_roll", "abs_pitch")}
        combos = {
            "E": lambda te, tv, s: te,
            "E+vE200": lambda te, tv, s: te + np.maximum(tv, 0),
            "E+vE+vroot": lambda te, tv, s, vr=None: te + np.maximum(tv, 0) + vr,
            "E+vE+S": lambda te, tv, s: te + np.maximum(tv, 0) + s,
        }
        # build combo scores per frame
        combo_auroc = {}
        for cname in ("E", "E+relu(vE200)", "E+relu(vE200)+|vroot|", "E+relu(vE200)+S"):
            suc, pre = [], []
            for row in rows:
                te = _norm(row["feats"]["E"], q_s["E"][0], q_s["E"][1])
                tv = _norm(row["feats"]["vE_200"], q_s["vE_200"][0], q_s["vE_200"][1])
                tv = np.where(np.isfinite(tv), tv, 0.0)
                vr = _norm(row["feats"]["abs_v_root_z"], q_s["abs_v_root_z"][0], q_s["abs_v_root_z"][1])
                sm = np.maximum.reduce(
                    [
                        vr,
                        _norm(row["feats"]["abs_roll"], q_s["abs_roll"][0], q_s["abs_roll"][1]),
                        _norm(row["feats"]["abs_pitch"], q_s["abs_pitch"][0], q_s["abs_pitch"][1]),
                    ]
                )
                if cname == "E":
                    sc = te
                elif cname == "E+relu(vE200)":
                    sc = te + np.maximum(tv, 0)
                elif cname == "E+relu(vE200)+|vroot|":
                    sc = te + np.maximum(tv, 0) + vr
                else:
                    sc = te + np.maximum(tv, 0) + sm
                row["feats"]["_R_" + cname] = sc
                if row["fail"] and row["fail_t"] is not None:
                    ft = int(row["fail_t"])
                    a = max(WARMUP, ft - 50)
                    b = max(a, ft - 5)
                    sl = sc[a:b]
                    pre.append(sl[np.isfinite(sl)])
                elif not row["fail"]:
                    sl = sc[WARMUP:]
                    suc.append(sl[np.isfinite(sl)])
            p = np.concatenate(pre) if pre else np.array([])
            n = np.concatenate(suc) if suc else np.array([])
            combo_auroc[cname] = {"auroc": _auroc(p, n), "n_pre": int(p.size), "n_suc": int(n.size)}
        block["combo"] = combo_auroc
        # lead for best combo vs E on stoop matrix
        if name == "matrix":
            block["lead_E_p3"] = _lead_sweep(rows, "E", 3)
            # store R for lead: copy into feats as E-like
            for row in rows:
                row["feats"]["R_stoop"] = row["feats"]["_R_E+relu(vE200)+S"]
            block["lead_R_p3"] = _lead_sweep(rows, "R_stoop", 3)
        stoop_out[name] = block
    report["stoop"] = stoop_out

    # stoop slope-only (the 0.37 cell)
    stoop_slope = [r for r in stoop_mat if r["terrain"] == "slope"]
    slope_block = {}
    for k in stoop_keys:
        p, n = _fail_score_frames(stoop_slope, k)
        slope_block[k] = {"auroc": _auroc(p, n), "n_pre": int(p.size), "n_suc": int(n.size)}
    report["stoop_matrix_slope"] = {
        "n": len(stoop_slope),
        "n_fail": sum(r["fail"] for r in stoop_slope),
        "auroc": slope_block,
    }

    # ---- 4. cross-terrain: calibrate on Flat+Light, challenge Slope/Steps ----
    # Use E>=13 persist 3 and also R = tilde_E + relu(tilde_vE200)
    def _duty_on(rows, key, th, persist=3):
        n = d = 0
        ep_on = []
        for r in rows:
            e = r["feats"][key]
            t_end = int(r["fail_t"]) if r["fail"] else e.size - 1
            sl = e[WARMUP : t_end + 1]
            n += sl.size
            run = 0
            on_t = 0
            for v in sl:
                run = run + 1 if v >= th else 0
                if run >= persist:
                    d += 1
                    on_t += 1
            ep_on.append(on_t * DT)
        return {
            "frame_duty": d / n if n else float("nan"),
            "mean_on_s": float(np.mean(ep_on)) if ep_on else float("nan"),
            "n_ep": len(rows),
        }

    xt = {}
    for ter in ALL_TERRAINS:
        sub = [r for r in loco_rev if r["terrain"] == ter]
        xt[ter] = {
            "E13_p3": _duty_on(sub, "E", 0.13, 3),
            "E8_p3": _duty_on(sub, "E", 0.08, 3),
            "n_fail": sum(r["fail"] for r in sub),
        }
    report["cross_terrain_loco_reverify"] = xt

    # R threshold: R_off = 1 (P90 of healthy tilde_E, since tilde_E P90=1 by construction)
    # Build R for loco_rev
    for r in loco_rev:
        te, tv, s = _combo_score(r, q)
        r["feats"]["R"] = te + tv  # beta=1, gamma=0 first
        r["feats"]["R_S"] = te + tv + s
    r_healthy = _healthy_frames(loco_rev, "R")
    r50, r90 = _q50_q90(r_healthy)
    report["R_healthy_q"] = {"q50": r50, "q90": r90}
    xtR = {}
    for ter in ALL_TERRAINS:
        sub = [r for r in loco_rev if r["terrain"] == ter]
        xtR[ter] = _duty_on(sub, "R", r90, 3)
    report["cross_terrain_R_p3_at_healthy_p90"] = xtR
    report["lead_R_p3"] = _lead_sweep(loco_rev, "R", 3)

    # ---- 5. leave-one-clip-out on loco reverify Flat+Light + fails everywhere ----
    clips = sorted({r["clip"] for r in loco_rev})
    loco_rows = []
    for held in clips:
        train = [r for r in loco_rev if r["clip"] != held]
        test = [r for r in loco_rev if r["clip"] == held]
        e_h = _healthy_frames(train, "E")
        q50, q90 = _q50_q90(e_h)
        # pick th: smallest th in 5..20cm with train healthy duty<=0.12 and max recall
        train_lead = _lead_sweep(train, "E", 3)
        cand = [x for x in train_lead if x["healthy_duty"] <= 0.12]
        if not cand:
            cand = train_lead
        # prefer recall then lead
        cand = sorted(cand, key=lambda x: (-(x["prefail_recall"] or 0), -(x["lead_median_s"] or 0), x["healthy_duty"]))
        th_cm = cand[0]["th_cm"]
        test_lead = _lead_sweep(test, "E", 3)
        trow = _pick(test_lead, th_cm, 3)
        # AUROC fail on held-out clip
        p, n = _fail_score_frames(test, "E")
        loco_rows.append(
            {
                "held_clip": held,
                "train_q50_cm": q50 * 100,
                "train_q90_cm": q90 * 100,
                "chosen_th_cm": th_cm,
                "test_duty": trow.get("healthy_duty"),
                "test_recall": trow.get("prefail_recall"),
                "test_lead_med": trow.get("lead_median_s"),
                "test_n_fail": trow.get("n_fail"),
                "test_auroc_E": _auroc(p, n),
            }
        )
    report["leave_clip_out"] = {
        "rows": loco_rows,
        "mean_test_duty": float(np.nanmean([x["test_duty"] for x in loco_rows])),
        "mean_test_recall": float(np.nanmean([x["test_recall"] for x in loco_rows if x["test_n_fail"]])),
        "mean_test_lead": float(np.nanmean([x["test_lead_med"] for x in loco_rows if x["test_n_fail"]])),
        "mean_test_auroc": float(np.nanmean([x["test_auroc_E"] for x in loco_rows])),
        "n_clips_with_fail": int(sum(1 for x in loco_rows if x["test_n_fail"])),
    }

    # GO summary
    e13 = report["go_loco_E13_p3"]
    inner_ev = report["self_recover"]["all_terrain_events"]
    v200 = next((x for x in inner_ev["auroc"] if x["key"] == "vE_200"), {})
    report["go"] = {
        "loco_recall_E13_p3": e13.get("prefail_recall"),
        "loco_healthy_duty_E13_p3": e13.get("healthy_duty"),
        "loco_lead_med_E13_p3": e13.get("lead_median_s"),
        "loco_lead_p10_E13_p3": e13.get("lead_p10_s"),
        "inner_event_frac_self": inner_ev.get("frac_self"),
        "inner_event_auroc_vE200": v200.get("auroc"),
        "inner_event_n": inner_ev.get("n"),
        "stoop_matrix_E_auroc": stoop_out["matrix"]["auroc"]["E"]["auroc"],
        "stoop_matrix_R_auroc": stoop_out["matrix"]["combo"]["E+relu(vE200)+S"]["auroc"],
        "stoop_slope_E_auroc": slope_block["E"]["auroc"],
        "stoop_slope_vroot_auroc": slope_block["abs_v_root_z"]["auroc"],
        "stoop_slope_roll_auroc": slope_block["abs_roll"]["auroc"],
        "lco_mean_recall": report["leave_clip_out"]["mean_test_recall"],
        "lco_mean_lead": report["leave_clip_out"]["mean_test_lead"],
        "lco_mean_duty": report["leave_clip_out"]["mean_test_duty"],
    }

    (OUT / "step3_report.json").write_text(json.dumps(report, indent=2, default=float))

    # compact CSVs
    with (OUT / "lead_time_loco.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "persist",
                "th_cm",
                "healthy_duty",
                "prefail_recall",
                "lead_median_s",
                "lead_p10_s",
                "lead_p90_s",
            ],
        )
        w.writeheader()
        for k, rows in lead.items():
            for row in rows:
                w.writerow(
                    {
                        "key": row["key"],
                        "persist": row["persist"],
                        "th_cm": row["th_cm"],
                        "healthy_duty": row["healthy_duty"],
                        "prefail_recall": row["prefail_recall"],
                        "lead_median_s": row["lead_median_s"],
                        "lead_p10_s": row["lead_p10_s"],
                        "lead_p90_s": row["lead_p90_s"],
                    }
                )

    print("wrote", OUT / "step3_report.json")
    g = report["go"]
    print("GO loco E13 p3", {k: g[k] for k in g if k.startswith("loco")})
    print("inner vE200", g["inner_event_auroc_vE200"], "frac_self", g["inner_event_frac_self"], "n", g["inner_event_n"])
    print("stoop matrix E", g["stoop_matrix_E_auroc"], "R", g["stoop_matrix_R_auroc"])
    print("stoop slope E", g["stoop_slope_E_auroc"], "vroot", g["stoop_slope_vroot_auroc"], "roll", g["stoop_slope_roll_auroc"])
    print("LCO recall/lead/duty", g["lco_mean_recall"], g["lco_mean_lead"], g["lco_mean_duty"])


if __name__ == "__main__":
    main()
