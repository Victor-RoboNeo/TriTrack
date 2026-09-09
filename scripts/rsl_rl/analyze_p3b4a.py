#!/usr/bin/env python3
"""P3-B4a: oracle intent-risk spectrum diagnostic. Offline. No training. No P4-B."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from p3_common import sanitize, stats, tilde_C_I
from p3b_model import IntentMetricMLP, chol_to_C, random_tangent

SEEN = ("plane", "slope", "slope_down", "light_rough", "steps")
HELD = ("slip",)
ALL_T = SEEN + HELD
P3B = Path("/data/home/chenxiangyu/robotics/Anybody/results/p3_intent_projected_adaptation")
IFS = Path("/data/home/chenxiangyu/robotics/Anybody/results/intent_free_space")
CKPT_B1 = P3B / "p3b_learned_projector" / "checkpoints"
RANKS = (1, 2, 3, 4, 6, 8)
N_RAND = 64
DIR_SEED = 2026
EPS = 1e-8
Z_DIM = 16


def _spear(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return float("nan")
    r = spearmanr(a[m], b[m])
    v = float(r.correlation)
    return v if math.isfinite(v) else float("nan")


def _eigh_desc(C: np.ndarray):
    w, U = np.linalg.eigh(0.5 * (C + C.T))
    idx = np.argsort(w)[::-1]
    return w[idx], U[:, idx]


def _lowrank(U, lam, r: int):
    r = min(int(r), U.shape[1])
    return (U[:, :r] * lam[:r]) @ U[:, :r].T


def _r_trace(lam, r: int) -> float:
    tot = float(np.maximum(lam.sum(), EPS))
    return float(lam[:r].sum() / tot)


def _r_risk(d, C, Cr) -> float:
    s = float(d @ C @ d)
    sr = float(d @ Cr @ d)
    return sr / (s + EPS)


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


def _principal_angles(U1, U2):
    q1, _ = np.linalg.qr(U1)
    q2, _ = np.linalg.qr(U2)
    s = np.linalg.svd(q1.T @ q2, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.arccos(s)


def _load_packed():
    z = np.load(P3B / "p3b_learned_projector" / "dataset_b" / "packed.npz", allow_pickle=True)
    splits = {}
    for k in z.files:
        sp, name = k.split("/", 1)
        splits.setdefault(sp, {})[name] = z[k]
    return splits


def _load_b1_ensemble(device):
    members = []
    for seed in (2026, 2027, 2028):
        p = CKPT_B1 / f"B1_s{seed}.pt"
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        net = IntentMetricMLP(int(ckpt["in_dim"]), out_dim=int(ckpt["out_dim"]))
        net.load_state_dict(ckpt["state_dict"])
        net.to(device).eval()
        members.append(
            {
                "net": net,
                "x_mean": torch.as_tensor(ckpt["x_mean"], device=device, dtype=torch.float32),
                "x_std": torch.as_tensor(ckpt["x_std"], device=device, dtype=torch.float32).clamp(min=1e-6),
                "in_dim": int(ckpt["in_dim"]),
            }
        )
    return members


@torch.no_grad()
def _mu_s(members, x, z, d):
    Cs = []
    for m in members:
        xn = ((x - m["x_mean"]) / m["x_std"]).clamp(-10.0, 10.0)
        raw = m["net"](xn if m["in_dim"] != 16 else xn[:, :16])
        Cs.append(chol_to_C(raw, z))
    C_stack = torch.stack(Cs, 0)
    sm = torch.einsum("mbij,bkj,bki->mbk", C_stack, d, d)
    return sm.mean(0).cpu().numpy()


def _load_ucr_states():
    ddir = P3B / "p3b_learned_projector" / "dataset_b"
    p3a = P3B / "p3a_oracle_projector"
    rows = []
    for t in ALL_T:
        zp, dp = ddir / f"{t}.npz", p3a / f"{t}_dstars.npz"
        if not zp.exists() or not dp.exists():
            continue
        z = np.load(zp, allow_pickle=True)
        dc = np.load(dp, allow_pickle=True)
        dmap = {(str(ep), int(dc["t"][i])): np.asarray(dc["d"][i], np.float64) for i, ep in enumerate(dc["episode_id"])}
        kind = np.asarray(z["state_kind"]).astype(str)
        for i in np.where(kind == "recovery")[0]:
            ep, tt = str(z["episode_id"][i]), int(z["t"][i])
            d = dmap.get((ep, tt))
            if d is None:
                continue
            d = d / (np.linalg.norm(d) + EPS)
            rows.append(
                {
                    "C": z["C"][i].astype(np.float64),
                    "z0": z["z0"][i].astype(np.float64),
                    "dstar": d,
                    "terrain": str(z["terrain"][i]),
                    "held_out": t in HELD,
                    "state_kind": "recovery",
                    "episode_id": ep,
                    "t": tt,
                }
            )
    return rows


def _jac_spectrum_pairs():
    """Consecutive jac frames for principal-angle stability. Reports actual Δt."""
    out = []
    for t in ALL_T:
        jp = IFS / "dataset_a" / t / "jac.npz"
        if not jp.exists():
            continue
        z = np.load(jp, allow_pickle=True)
        ep = np.asarray(z["episode_id"]).astype(str)
        tt = z["t"].astype(int)
        win = np.asarray(z["window"]).astype(str)
        J = z["J_I"].astype(np.float64)
        z0 = z["z0"].astype(np.float64)
        groups = defaultdict(list)
        for i in range(tt.size):
            groups[(ep[i], win[i])].append(i)
        for (e, w), idxs in groups.items():
            idxs = sorted(idxs, key=lambda i: int(tt[i]))
            for a, b in zip(idxs[:-1], idxs[1:]):
                dt = int(tt[b]) - int(tt[a])
                if dt <= 0:
                    continue
                Ca = tilde_C_I(J[a], z0[a])
                Cb = tilde_C_I(J[b], z0[b])
                out.append(
                    {
                        "terrain": t,
                        "held_out": t in HELD,
                        "window": w,
                        "dt_frames": dt,
                        "C_a": Ca,
                        "C_b": Cb,
                    }
                )
    return out


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/home/chenxiangyu/robotics/Anybody/results/p3b4_intent_risk_structure")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = Path(args.out)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    print("[b4a] load data", flush=True)
    packed = _load_packed()
    members = _load_b1_ensemble(device)
    ucr = _load_ucr_states()
    print(f"[b4a] UCR {len(ucr)} packed val {packed['val']['C'].shape[0]}", flush=True)

    # ---- 1. Spectrum on packed splits ----
    spec_rows = []
    rtrace = {sp: {r: [] for r in RANKS} for sp in ("val", "test_seen", "test_slip")}
    for sp in ("val", "test_seen", "test_slip"):
        C = packed[sp]["C"].astype(np.float64)
        kind = np.asarray(packed[sp]["state_kind"]).astype(str)
        terr = np.asarray(packed[sp]["terrain"]).astype(str)
        for i in range(C.shape[0]):
            lam, U = _eigh_desc(C[i])
            rec = {"split": sp, "state_kind": str(kind[i]), "terrain": str(terr[i])}
            for r in RANKS:
                rt = _r_trace(lam, r)
                rec[f"R_trace_{r}"] = rt
                rtrace[sp][r].append(rt)
            spec_rows.append(rec)
    spec_sum = {}
    for sp, blk in rtrace.items():
        spec_sum[sp] = {f"r{r}": stats(blk[r]) for r in RANKS}

    # ---- 2. UCR d* R_risk ----
    ucr_risk = {r: [] for r in RANKS}
    ucr_risk_slip = {r: [] for r in RANKS}
    for row in ucr:
        lam, U = _eigh_desc(row["C"])
        dest = ucr_risk_slip if row["held_out"] else ucr_risk
        for r in RANKS:
            dest[r].append(_r_risk(row["dstar"], row["C"], _lowrank(U, lam, r)))
    ucr_sum = {
        "seen": {f"r{r}": stats(ucr_risk[r]) for r in RANKS},
        "slip": {f"r{r}": stats(ucr_risk_slip[r]) for r in RANKS},
    }

    # ---- 3. False-free R_risk (B1 ensemble, same as P3-B3) ----
    print("[b4a] false-free dirs", flush=True)
    ff_risk = {sp: {r: [] for r in RANKS} for sp in ("test_seen", "test_slip")}
    all_risk = {sp: {r: [] for r in RANKS} for sp in ("test_seen", "test_slip")}
    for sp in ("test_seen", "test_slip"):
        d = packed[sp]
        x = torch.as_tensor(d["x"], device=device)
        z = torch.as_tensor(d["z0"], device=device)
        C = d["C"].astype(np.float64)
        g = torch.Generator(device=device)
        g.manual_seed(DIR_SEED)
        v = random_tangent(z, N_RAND, g)
        C_t = torch.as_tensor(C, device=device, dtype=v.dtype)
        star = torch.einsum("bdi,bij,bdj->bd", v, C_t, v).cpu().numpy()
        mu = _mu_s(members, x, z, v)
        sens = _qmask(star, 0.25, True)
        free = _qmask(mu, 0.25, False)
        ff = sens & free
        vnp = v.cpu().numpy()
        n = C.shape[0]
        for i in range(n):
            lam, U = _eigh_desc(C[i])
            for k in range(N_RAND):
                dvec = vnp[i, k]
                for r in RANKS:
                    rr = _r_risk(dvec, C[i], _lowrank(U, lam, r))
                    all_risk[sp][r].append(rr)
                    if ff[i, k]:
                        ff_risk[sp][r].append(rr)
    ff_sum = {
        sp: {
            "n_ff": int(len(ff_risk[sp][4])),
            "all": {f"r{r}": stats(all_risk[sp][r]) for r in RANKS},
            "false_free": {f"r{r}": stats(ff_risk[sp][r]) for r in RANKS},
        }
        for sp in ("test_seen", "test_slip")
    }

    # ---- 4. Subspace temporal stability ----
    print("[b4a] principal angles", flush=True)
    pairs = _jac_spectrum_pairs()
    ang_rows = []
    buckets = defaultdict(list)
    dt_list = []
    for p in pairs:
        dt_list.append(p["dt_frames"])
        lam_a, Ua = _eigh_desc(p["C_a"])
        lam_b, Ub = _eigh_desc(p["C_b"])
        rec = {
            "terrain": p["terrain"],
            "held_out": int(p["held_out"]),
            "window": p["window"],
            "dt_frames": p["dt_frames"],
            "dt_ms": int(p["dt_frames"]) * 20,
        }
        kind = "recovery" if p["window"] == "recovery" else "nominal"
        if p["held_out"]:
            tag = "slip"
        elif p["terrain"] == "steps":
            tag = "steps"
        else:
            tag = kind
        for r in (3, 4, 6):
            ang = _principal_angles(Ua[:, :r], Ub[:, :r])
            mx = float(np.degrees(ang.max())) if ang.size else float("nan")
            rec[f"max_deg_r{r}"] = mx
            buckets[(tag, r)].append(mx)
            buckets[(f"all_{p['window']}", r)].append(mx)
        ang_rows.append(rec)
    ang_sum = {}
    for (tag, r), xs in buckets.items():
        ang_sum[f"{tag}_r{r}"] = {
            **stats(xs),
            "p90_deg": float(np.quantile(xs, 0.90)) if xs else None,
            "frac_gt_30": float(np.mean(np.asarray(xs) > 30.0)) if xs else None,
        }
    dt_med = float(np.median(dt_list)) if dt_list else None

    # decision
    tr4 = spec_sum["test_seen"]["r4"]["median"]
    rr4 = ucr_sum["seen"]["r4"]["median"]
    ff4 = ff_sum["test_seen"]["false_free"]["r4"]["median"]
    ang4 = (ang_sum.get("nominal_r4") or {}).get("p90_deg")
    ang4_rec = (ang_sum.get("recovery_r4") or ang_sum.get("all_recovery_r4") or {}).get("p90_deg")
    spectrum_ok = bool(tr4 is not None and tr4 >= 0.85)
    ucr_ok = bool(rr4 is not None and rr4 >= 0.9)
    ff_ok = bool(ff4 is not None and ff4 >= 0.85)
    # recovery bucket key
    rec_key = "all_recovery_r4" if "all_recovery_r4" in ang_sum else "recovery_r4"
    nom_key = "all_nominal_r4" if "all_nominal_r4" in ang_sum else "nominal_r4"
    ang4_rec = (ang_sum.get(rec_key) or {}).get("p90_deg")
    ang4_nom = (ang_sum.get(nom_key) or {}).get("p90_deg")
    stable = bool(ang4_rec is not None and ang4_rec <= 30.0)
    if spectrum_ok and ucr_ok and ff_ok and stable:
        next_method = "spectral_subspace"
    elif spectrum_ok and ucr_ok and (not ff_ok or not stable):
        next_method = "direction_conditioned" if not ff_ok else "direction_conditioned_unstable_frame"
    else:
        next_method = "direction_conditioned"
    # analytic still pending B4b
    decision = {
        "R_trace4_seen_med": tr4,
        "R_risk4_ucr_seen_med": rr4,
        "R_risk4_ff_seen_med": ff4,
        "principal_p90_nominal_r4_deg": ang4_nom,
        "principal_p90_recovery_r4_deg": ang4_rec,
        "dt_frames_median": dt_med,
        "spectrum_concentrated": spectrum_ok,
        "ucr_risk_captured": ucr_ok,
        "false_free_captured": ff_ok,
        "subspace_stable_20ms_proxy": stable,
        "note_dt": "jac dumps are stride-sampled; dt_frames_median is the actual spacing, not necessarily 1 frame",
        "recommended_learned_fallback": next_method,
        "analytic_pending": True,
        "no_training": True,
    }

    metrics = {
        "spectrum": spec_sum,
        "ucr_risk": ucr_sum,
        "false_free": ff_sum,
        "principal_angles": ang_sum,
        "n_ucr_seen": int(len(ucr_risk[4])),
        "n_ucr_slip": int(len(ucr_risk_slip[4])),
        "n_angle_pairs": int(len(ang_rows)),
        "decision": decision,
    }
    (out / "b4a_metrics.json").write_text(json.dumps(sanitize(metrics), indent=2), encoding="utf-8")
    _write_csv(out / "spectrum_concentration.csv", spec_rows)
    _write_csv(out / "ucr_risk_capture.csv", [{"r": r, **ucr_sum["seen"][f"r{r}"], "split": "seen"} for r in RANKS] + [{"r": r, **ucr_sum["slip"][f"r{r}"], "split": "slip"} for r in RANKS])
    _write_csv(out / "principal_angles.csv", [{k: v for k, v in r.items()} for r in ang_rows])

    # plots
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    xs = list(RANKS)
    for name, blk in (("seen packed", spec_sum["test_seen"]), ("slip packed", spec_sum["test_slip"])):
        ax.plot(xs, [blk[f"r{r}"]["median"] for r in RANKS], "o-", label=name)
    ax.axhline(0.85, ls="--", c="0.5")
    ax.set_xlabel("r")
    ax.set_ylabel(r"$R_{\mathrm{trace}}(r)$ median")
    ax.set_title("B4a-1 spectrum concentration")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plots / "B4a-1_Rtrace.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(xs, [ucr_sum["seen"][f"r{r}"]["median"] for r in RANKS], "o-", label="UCR d* seen")
    ax.plot(xs, [ucr_sum["slip"][f"r{r}"]["median"] for r in RANKS], "o-", label="UCR d* slip")
    ax.plot(xs, [ff_sum["test_seen"]["false_free"][f"r{r}"]["median"] for r in RANKS], "s-", label="false-free seen")
    ax.axhline(0.9, ls="--", c="0.5")
    ax.set_xlabel("r")
    ax.set_ylabel(r"median $R_{\mathrm{risk}}$")
    ax.set_title("B4a-2 risk capture")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plots / "B4a-2_Rrisk.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    labels, meds, p90s = [], [], []
    for tag in ("all_nominal", "all_recovery", "steps", "slip"):
        k = f"{tag}_r4"
        if k not in ang_sum:
            continue
        labels.append(tag)
        meds.append(ang_sum[k]["median"])
        p90s.append(ang_sum[k]["p90_deg"])
    xx = np.arange(len(labels))
    ax.bar(xx - 0.18, meds, 0.36, label="median max angle")
    ax.bar(xx + 0.18, p90s, 0.36, label="p90 max angle")
    ax.axhline(30, ls="--", c="0.5")
    ax.set_xticks(xx)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("degrees")
    ax.set_title(f"B4a-4 principal angle r=4 (median Δt={dt_med} frames)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plots / "B4a-4_angles.png", dpi=140)
    plt.close(fig)

    lines = [
        "# P3-B4a — Oracle Intent-Risk Spectrum",
        "",
        "Offline eigendecomposition of existing $C_I^{gt}$. No training. No B1 fine-tune.",
        f"Packed splits + UCR d* (n_seen={len(ucr_risk[4])}, n_slip={len(ucr_risk_slip[4])}).",
        f"Principal angles use jac.npz consecutive stored frames (median Δt = {dt_med} frames = {(dt_med or 0)*20:.0f} ms).",
        "",
        "## 1. Spectrum concentration $R_{trace}(r)$ (median)",
        "",
        "| r | val | seen-test | slip |",
        "|---:|---:|---:|---:|",
    ]
    for r in RANKS:
        lines.append(
            f"| {r} | {_fmt(spec_sum['val'][f'r{r}']['median'], 3)} | "
            f"{_fmt(spec_sum['test_seen'][f'r{r}']['median'], 3)} | "
            f"{_fmt(spec_sum['test_slip'][f'r{r}']['median'], 3)} |"
        )
    lines += [
        "",
        f"Top-4 seen $R_{{trace}}$ median = **{_fmt(tr4, 3)}**  (target ≳ 0.85): **{spectrum_ok}**",
        "",
        "## 2. UCR $d^*$ risk capture $R_{risk}(r,d^*)$",
        "",
        "| r | seen median | seen p90 | slip median |",
        "|---:|---:|---:|---:|",
    ]
    for r in RANKS:
        lines.append(
            f"| {r} | {_fmt(ucr_sum['seen'][f'r{r}']['median'], 3)} | "
            f"{_fmt(ucr_sum['seen'][f'r{r}']['p90'], 3)} | "
            f"{_fmt(ucr_sum['slip'][f'r{r}']['median'], 3)} |"
        )
    lines += [
        "",
        f"Top-4 UCR seen median $R_{{risk}}$ = **{_fmt(rr4, 3)}** (target > 0.9): **{ucr_ok}**",
        "",
        "## 3. False-free directions (P3-B3 definition, B1 ensemble μ)",
        "",
        f"n_FF seen-test dirs = {ff_sum['test_seen']['n_ff']}, slip = {ff_sum['test_slip']['n_ff']}",
        "",
        "| r | FF seen median | all-dir seen median | FF slip median |",
        "|---:|---:|---:|---:|",
    ]
    for r in RANKS:
        lines.append(
            f"| {r} | {_fmt(ff_sum['test_seen']['false_free'][f'r{r}']['median'], 3)} | "
            f"{_fmt(ff_sum['test_seen']['all'][f'r{r}']['median'], 3)} | "
            f"{_fmt(ff_sum['test_slip']['false_free'][f'r{r}']['median'], 3)} |"
        )
    lines += [
        "",
        f"Top-4 FF seen median = **{_fmt(ff4, 3)}**. Captured (≥0.85): **{ff_ok}**",
        "",
        "## 4. Sensitive subspace temporal stability (r=4, largest principal angle, deg)",
        "",
        "| slice | median | p90 | P(>30°) |",
        "|---|---:|---:|---:|",
    ]
    for tag in ("all_nominal_r4", "all_recovery_r4", "steps_r4", "slip_r4"):
        blk = ang_sum.get(tag)
        if not blk:
            continue
        lines.append(
            f"| {tag} | {_fmt(blk.get('median'), 1)} | {_fmt(blk.get('p90_deg'), 1)} | {_fmt(blk.get('frac_gt_30'), 2)} |"
        )
    lines += [
        "",
        f"Recovery p90 angle = **{_fmt(ang4_rec, 1)}°** vs 30° threshold. Stable: **{stable}**",
        "",
        "## Decision (learned shield fallback; analytic still evaluated in B4b)",
        "",
        f"- spectrum concentrated: {spectrum_ok}",
        f"- UCR risk in top-4: {ucr_ok}",
        f"- false-free captured by top-4: {ff_ok}",
        f"- subspace stable: {stable}",
        "",
        f"**Recommended learned fallback (if analytic fails): `{next_method}`**",
        "",
        "Do not train the spectral or direction-conditioned network in this round.",
        "P4-A0 oracle-safe probe is independent and should run regardless.",
    ]
    (out / "B4A_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(sanitize(decision), indent=2), flush=True)
    print(f"[b4a] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
