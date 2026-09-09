#!/usr/bin/env python3
"""P3-B4b: analytic decoder+FK Jacobian vs oracle J_I and B1. Offline + UCR json. No training."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from p3_common import adv_block, sanitize, stats, tilde_C_I
from p3b_model import IntentMetricMLP, chol_to_C, random_tangent, soft_P

SEEN = ("plane", "slope", "slope_down", "light_rough", "steps")
HELD = ("slip",)
ALL_T = SEEN + HELD
P3B = Path("/data/home/chenxiangyu/robotics/Anybody/results/p3_intent_projected_adaptation")
CKPT_B1 = P3B / "p3b_learned_projector" / "checkpoints" / "B1_s2028.pt"
N_RAND = 64
DIR_SEED = 2026
EPS = 1e-8
Z_DIM = 16
POS_SCALE = 0.05


def _spear(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return float("nan")
    r = spearmanr(a[m], b[m])
    v = float(r.correlation)
    return v if math.isfinite(v) else float("nan")


def auroc(y, s) -> float:
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    pos, neg = s[y], s[~y]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    neg = np.sort(neg)
    lt = np.searchsorted(neg, pos, side="left")
    le = np.searchsorted(neg, pos, side="right")
    return float(((lt + 0.5 * (le - lt)) / max(neg.size, 1)).mean())


def _qmask(s, frac=0.25, top=True):
    n = s.shape[-1]
    k = max(1, int(round(frac * n)))
    if top:
        idx = np.argpartition(-s, kth=min(k, n - 1), axis=-1)[..., :k]
    else:
        idx = np.argpartition(s, kth=min(k, n - 1), axis=-1)[..., :k]
    m = np.zeros_like(s, dtype=bool)
    row = np.arange(s.shape[0])[:, None]
    m[row, idx] = True
    return m


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def _load_b1(device):
    ckpt = torch.load(CKPT_B1, map_location="cpu", weights_only=False)
    net = IntentMetricMLP(int(ckpt["in_dim"]), out_dim=int(ckpt["out_dim"]))
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    return {
        "net": net,
        "x_mean": torch.as_tensor(ckpt["x_mean"], device=device, dtype=torch.float32),
        "x_std": torch.as_tensor(ckpt["x_std"], device=device, dtype=torch.float32).clamp(min=1e-6),
        "in_dim": int(ckpt["in_dim"]),
    }


@torch.no_grad()
def _b1_C(spec, x, z):
    xn = ((x - spec["x_mean"]) / spec["x_std"]).clamp(-10.0, 10.0)
    raw = spec["net"](xn if spec["in_dim"] != 16 else xn[:, :16])
    return chol_to_C(raw, z)


def _range_angle(J_a, J_b) -> float:
    """Largest principal angle (deg) between row-spaces of J (ny x 16)."""
    def _basis(J):
        if J.ndim != 2 or min(J.shape) == 0:
            return None
        _u, s, vt = np.linalg.svd(J, full_matrices=False)
        if s.size == 0:
            return None
        r = int((s > 1e-6 * max(float(s[0]), 1e-12)).sum())
        r = max(r, 1)
        return vt[:r].T

    A, B = _basis(J_a), _basis(J_b)
    if A is None or B is None:
        return float("nan")
    r = min(A.shape[1], B.shape[1])
    s = np.linalg.svd(A[:, :r].T @ B[:, :r], compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.degrees(np.arccos(float(s.min())))) if s.size else float("nan")


def _load_packed_x():
    z = np.load(P3B / "p3b_learned_projector" / "dataset_b" / "packed.npz", allow_pickle=True)
    splits = {}
    for k in z.files:
        sp, name = k.split("/", 1)
        splits.setdefault(sp, {})[name] = z[k]
    return splits


def _index_packed(packed):
    """(episode_id, t) -> (split, i, x)."""
    out = {}
    for sp, d in packed.items():
        if "episode_id" not in d:
            continue
        for i, (ep, tt) in enumerate(zip(d["episode_id"], d["t"])):
            out[(str(ep), int(tt))] = (sp, i, d["x"][i].astype(np.float32))
    return out


def _geom_from_jac(root: Path, spec, device):
    rows_s, rows_ff, rows_ang = [], [], []
    packed = _load_packed_x()
    pmap = _index_packed(packed)
    for t in ALL_T:
        p = root / "jac_ana" / f"{t}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        if not bool(z["have_oracle"]):
            print(f"[b4b] {t} missing oracle J in dump", flush=True)
            continue
        Jana = z["J_I_ana"].astype(np.float64)
        Jor = z["J_I_oracle"].astype(np.float64)
        z0 = z["z0"].astype(np.float64)
        kind = np.asarray(z["state_kind"]).astype(str)
        pos_scale = float(z["pos_scale"]) if "pos_scale" in z.files else POS_SCALE
        n = Jana.shape[0]
        x_list, keep = [], []
        for i in range(n):
            rec = pmap.get((str(z["episode_id"][i]), int(z["t"][i])))
            if rec is None:
                continue
            keep.append(i)
            x_list.append(rec[2])
        if not keep:
            continue
        keep = np.asarray(keep, dtype=int)
        x = torch.as_tensor(np.stack(x_list), device=device)
        zt = torch.as_tensor(z0[keep], device=device, dtype=torch.float32)
        C_b1 = _b1_C(spec, x, zt).cpu().numpy()
        C_or, C_an = [], []
        for i in keep:
            C_or.append(tilde_C_I(Jor[i], z0[i], pos_scale))
            C_an.append(tilde_C_I(Jana[i], z0[i], pos_scale))
        C_or = np.stack(C_or)
        C_an = np.stack(C_an)
        g = torch.Generator(device=device)
        g.manual_seed(DIR_SEED)
        v = random_tangent(zt, N_RAND, g)
        vnp = v.cpu().numpy()
        star = np.einsum("bdi,bij,bdj->bd", vnp, C_or, vnp)
        sana = np.einsum("bdi,bij,bdj->bd", vnp, C_an, vnp)
        sb1 = np.einsum("bdi,bij,bdj->bd", vnp, C_b1, vnp)
        sens = _qmask(star, 0.25, True)
        free = _qmask(sb1, 0.25, False)
        ff = sens & free
        ana_top = _qmask(sana, 0.25, True)
        dang_recall = float((ana_top & sens).sum() / max(sens.sum(), 1))
        ff_recall = float((ana_top & ff).sum() / max(ff.sum(), 1)) if ff.any() else float("nan")
        rows_s.append(
            {
                "terrain": t,
                "held_out": t in HELD,
                "n": int(keep.size),
                "spearman_ana_oracle": _spear(sana, star),
                "spearman_b1_oracle": _spear(sb1, star),
                "top25_dangerous_recall": dang_recall,
                "ff_recall_top25": ff_recall,
                "ff_auroc_ana": auroc(ff.reshape(-1), sana.reshape(-1)) if ff.any() else float("nan"),
                "ff_n": int(ff.sum()),
                "n_dirs": int(sana.size),
            }
        )
        for i, gi in enumerate(keep):
            ang = _range_angle(Jana[gi], Jor[gi])
            rows_ang.append({"terrain": t, "state_kind": str(kind[gi]), "angle_deg": ang})
        rows_ff.append({"terrain": t, "ff_n": int(ff.sum()), "n_dirs": int(sana.size)})
        print(
            f"[b4b] geom {t} n={keep.size} spear_ana={rows_s[-1]['spearman_ana_oracle']:.3f} "
            f"recall25={dang_recall:.3f} ff_auroc={rows_s[-1]['ff_auroc_ana']}",
            flush=True,
        )
    return rows_s, rows_ang


def _ucr_from_json(root: Path):
    by = {m: {"A": [], "DI": [], "leak": [], "cos": [], "Rd": []} for m in ("oracle", "B1", "analytic")}
    per_t = {}
    for t in ALL_T:
        p = root / "rollout" / f"{t}.json"
        if not p.exists():
            continue
        payload = json.loads(p.read_text())
        blk = {m: {"A": [], "DI": [], "leak": [], "cos": []} for m in ("oracle", "B1", "analytic")}
        for row in payload.get("rows") or []:
            for m in blk:
                rec = (row.get("methods") or {}).get(m) or {}
                if "A" in rec:
                    blk[m]["A"].append(rec["A"])
                    by[m]["A"].append(rec["A"])
                if "DI_5deg" in rec:
                    blk[m]["DI"].append(rec["DI_5deg"])
                    by[m]["DI"].append(rec["DI_5deg"])
                if "oracle_leak" in rec:
                    blk[m]["leak"].append(rec["oracle_leak"])
                    by[m]["leak"].append(rec["oracle_leak"])
                if "cos_ucr" in rec:
                    blk[m]["cos"].append(rec["cos_ucr"])
                    by[m]["cos"].append(rec["cos_ucr"])
                if "Rd" in rec:
                    by[m]["Rd"].append(rec["Rd"])
        per_t[t] = {
            m: {
                "A": adv_block(blk[m]["A"]),
                "DI": stats(blk[m]["DI"]),
                "leak": stats(blk[m]["leak"]),
                "cos": stats(blk[m]["cos"]),
            }
            for m in blk
        }
        per_t[t]["n"] = payload.get("n")
        per_t[t]["held_out"] = payload.get("held_out")
    pooled = {
        m: {
            "A": adv_block(by[m]["A"]),
            "DI": stats(by[m]["DI"]),
            "leak": stats(by[m]["leak"]),
            "cos": stats(by[m]["cos"]),
            "Rd": stats(by[m]["Rd"]),
        }
        for m in by
    }
    return per_t, pooled


def _ratio(a, b):
    if a is None or b is None or not b:
        return None
    return float(a) / float(b)


def _decide(geom, pooled):
    spears = [r["spearman_ana_oracle"] for r in geom if math.isfinite(r["spearman_ana_oracle"])]
    recs = [r["top25_dangerous_recall"] for r in geom if math.isfinite(r["top25_dangerous_recall"])]
    spear = float(np.median(spears)) if spears else float("nan")
    rec = float(np.median(recs)) if recs else float("nan")
    leak_or = (pooled.get("oracle") or {}).get("leak", {}).get("median")
    leak_b1 = (pooled.get("B1") or {}).get("leak", {}).get("median")
    leak_an = (pooled.get("analytic") or {}).get("leak", {}).get("median")
    di_or = (pooled.get("oracle") or {}).get("DI", {}).get("median")
    di_b1 = (pooled.get("B1") or {}).get("DI", {}).get("median")
    di_an = (pooled.get("analytic") or {}).get("DI", {}).get("median")
    leak_b1_r = _ratio(leak_b1, leak_or)
    leak_an_r = _ratio(leak_an, leak_or)
    di_b1_r = _ratio(di_b1, di_or)
    di_an_r = _ratio(di_an, di_or)
    leak_better = (
        leak_an_r is not None and leak_b1_r is not None and leak_an_r < leak_b1_r * 0.75 and leak_an_r < 1.5
    )
    di_better = di_an_r is not None and di_b1_r is not None and di_an_r < di_b1_r * 0.75
    recall_ok = math.isfinite(rec) and rec >= 0.7
    spear_ok = math.isfinite(spear) and spear >= 0.4
    use_analytic = bool(spear_ok and recall_ok and (leak_better or di_better))
    if use_analytic:
        rec_s = "analytic_intent_shield"
    else:
        rec_s = "direction_conditioned_S_theta"
    return {
        "spearman_ana_oracle_med": spear,
        "top25_recall_med": rec,
        "leak_ratio_B1": leak_b1_r,
        "leak_ratio_analytic": leak_an_r,
        "DI_ratio_B1": di_b1_r,
        "DI_ratio_analytic": di_an_r,
        "use_analytic_shield": use_analytic,
        "recommended": rec_s,
        "no_spectral_network": True,
        "no_full_psd_retrain": True,
        "p4b_blocked_by_p4a0_case_B": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/home/chenxiangyu/robotics/Anybody/results/p3b4b_analytic_jacobian")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    root = Path(args.root)
    device = torch.device("cpu")
    spec = _load_b1(device)
    print("[b4b] geometry", flush=True)
    geom, ang = _geom_from_jac(root, spec, device)
    print("[b4b] ucr json", flush=True)
    per_t, pooled = _ucr_from_json(root)
    decision = _decide(geom, pooled)
    ang_stats = stats([r["angle_deg"] for r in ang if math.isfinite(r.get("angle_deg", float("nan")))])
    metrics = {
        "action": "joint_position_target",
        "formula": "J_I,z^ana = J_I,q * diag(scale) * J_a,z",
        "geometry": geom,
        "range_angle_deg": ang_stats,
        "ucr_per_terrain": per_t,
        "ucr_pooled": pooled,
        "decision": decision,
        "no_terrain_in_model": True,
    }
    (root / "metrics.json").write_text(json.dumps(sanitize(metrics), indent=2), encoding="utf-8")
    lines = [
        "# P3-B4b — Analytic Decoder / FK Jacobian",
        "",
        "Stage-2 action is **joint position target** (not torque).",
        r"$$J_{I,z}^{\mathrm{ana}} \approx J_{I,q}\,J_{q,a}\,J_{a,z},\quad J_{q,a}=\mathrm{diag}(\mathrm{scale}).$$",
        r"No terrain / vision / history / network. Compared to closed-loop FD oracle \(J_I\) (1°, L=1) and B1.",
        "",
        "## Geometry (64 random tangent dirs)",
        "",
        "| terrain | n | Spearman ana↔oracle | Spearman B1↔oracle | top-25% danger recall | FF AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in geom:
        lines.append(
            f"| {r['terrain']} | {r['n']} | {_fmt(r['spearman_ana_oracle'])} | {_fmt(r['spearman_b1_oracle'])} | "
            f"{_fmt(r['top25_dangerous_recall'])} | {_fmt(r['ff_auroc_ana'])} |"
        )
    lines += [
        "",
        f"Range angle (principal, deg) median={_fmt(ang_stats.get('median'))} p90={_fmt(ang_stats.get('p90'))}.",
        "",
        "## UCR projected (oracle vs B1 vs analytic), λ=1",
        "",
        "| method | mean A | P(A>0) | DI med | leak med | cos |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in ("oracle", "B1", "analytic"):
        blk = pooled.get(m) or {}
        lines.append(
            f"| {m} | {_fmt((blk.get('A') or {}).get('mean'), 4)} | {_fmt((blk.get('A') or {}).get('P_Agt0'))} | "
            f"{_fmt((blk.get('DI') or {}).get('median'), 4)} | {_fmt((blk.get('leak') or {}).get('median'), 4)} | "
            f"{_fmt((blk.get('cos') or {}).get('median'))} |"
        )
    d = decision
    lines += [
        "",
        "## Ratios vs oracle",
        "",
        f"- leak B1/oracle = **{_fmt(d.get('leak_ratio_B1'))}**, analytic/oracle = **{_fmt(d.get('leak_ratio_analytic'))}**",
        f"- DI B1/oracle = **{_fmt(d.get('DI_ratio_B1'))}**, analytic/oracle = **{_fmt(d.get('DI_ratio_analytic'))}**",
        f"- Spearman ana (median terrains) = **{_fmt(d.get('spearman_ana_oracle_med'))}**",
        f"- top-25% dangerous recall = **{_fmt(d.get('top25_recall_med'))}**",
        "",
        "## Decision",
        "",
        f"**{d.get('recommended')}**  (use_analytic={d.get('use_analytic_shield')})",
        "",
        "If analytic: stop learned projector research; known motor geometry protects WHAT.",
        r"If not: do **not** train spectral frames (B4a); fallback is direction-conditioned \(S_\theta(x,d)\to s_I\).",
        "P4-B remains blocked by P4-A0 Case B.",
        "",
    ]
    (root / "B4B_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(sanitize(d), indent=2), flush=True)
    print(f"[b4b] wrote {root / 'B4B_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
