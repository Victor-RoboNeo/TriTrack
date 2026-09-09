#!/usr/bin/env python3
"""P3-B3 ensemble tail-risk diagnostic. Post-hoc only. No training. No P4."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from p3_common import adv_block, sanitize, stats
from p3b_model import IntentMetricMLP, Z_DIM, chol_to_C, random_tangent, sample_ucr_dirs, soft_P

SEEN = ("plane", "slope", "slope_down", "light_rough", "steps")
HELD = ("slip",)
P3B = Path("/data/home/chenxiangyu/robotics/Anybody/results/p3_intent_projected_adaptation")
IAE = Path("/data/home/chenxiangyu/robotics/Anybody/results/intent_autonomous_execution")
CKPT_B1 = P3B / "p3b_learned_projector" / "checkpoints"
CKPT_B21 = IAE / "p3b2_intent_shield" / "checkpoints"
EPS = 1e-8
N_RAND = 64
N_UCR_LIKE = 16
DIR_SEED = 2026
BOOT_N = 2000
BOOT_SEED = 2026
BETAS = (0.0, 0.5, 1.0, 2.0, 3.0)


def _load_ckpt(path: Path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = IntentMetricMLP(int(ckpt["in_dim"]), out_dim=int(ckpt["out_dim"]))
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    return {
        "net": net,
        "x_mean": torch.as_tensor(ckpt["x_mean"], device=device, dtype=torch.float32),
        "x_std": torch.as_tensor(ckpt["x_std"], device=device, dtype=torch.float32).clamp(min=1e-6),
        "in_dim": int(ckpt["in_dim"]),
        "path": str(path),
        "seed": int(ckpt.get("seed") or 0),
    }


def _load_ensemble(kind: str, device):
    root = CKPT_B1 if kind == "B1" else CKPT_B21
    prefix = "B1" if kind == "B1" else "B21"
    members = []
    for seed in (2026, 2027, 2028):
        p = root / f"{prefix}_s{seed}.pt"
        if not p.exists():
            print(f"[p3b3] missing {p}", flush=True)
            continue
        members.append(_load_ckpt(p, device))
    if len(members) < 3:
        raise FileNotFoundError(f"{kind} ensemble size {len(members)} < 3")
    return members


@torch.no_grad()
def _pred_C_stack(members, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    Cs = []
    for m in members:
        xn = ((x - m["x_mean"]) / m["x_std"]).clamp(-10.0, 10.0)
        if m["in_dim"] == Z_DIM:
            raw = m["net"](xn[:, :Z_DIM])
        else:
            raw = m["net"](xn)
        Cs.append(chol_to_C(raw, z))
    return torch.stack(Cs, dim=0)


def _load_packed():
    z = np.load(P3B / "p3b_learned_projector" / "dataset_b" / "packed.npz", allow_pickle=True)
    splits = {}
    for k in z.files:
        sp, name = k.split("/", 1)
        splits.setdefault(sp, {})[name] = z[k]
    return splits


def _parent_split_map(packed) -> dict[str, str]:
    out = {}
    for sp in ("train", "val", "test_seen"):
        if sp not in packed:
            continue
        for p, e in zip(packed[sp]["parent_episode_id"], packed[sp]["episode_id"]):
            out[str(p)] = sp
            out[str(e)] = sp
    return out


def _load_ucr_states(packed):
    smap = _parent_split_map(packed)
    ddir = P3B / "p3b_learned_projector" / "dataset_b"
    p3a = P3B / "p3a_oracle_projector"
    rows = []
    for t in SEEN + HELD:
        zp = ddir / f"{t}.npz"
        dp = p3a / f"{t}_dstars.npz"
        if not zp.exists() or not dp.exists():
            continue
        z = np.load(zp, allow_pickle=True)
        dc = np.load(dp, allow_pickle=True)
        dmap = {(str(ep), int(dc["t"][i])): np.asarray(dc["d"][i], dtype=np.float32) for i, ep in enumerate(dc["episode_id"])}
        kind = np.asarray(z["state_kind"]).astype(str)
        for i in np.where(kind == "recovery")[0]:
            ep = str(z["episode_id"][i])
            tt = int(z["t"][i])
            d = dmap.get((ep, tt))
            if d is None:
                continue
            parent = str(z["parent_episode_id"][i])
            if t in HELD:
                split = "test_slip"
            else:
                split = smap.get(parent, smap.get(ep, "test_seen"))
            rows.append(
                {
                    "x": z["x"][i].astype(np.float32),
                    "z0": z["z0"][i].astype(np.float32),
                    "C": z["C"][i].astype(np.float32),
                    "dstar": d / (np.linalg.norm(d) + 1e-8),
                    "episode_id": ep,
                    "parent_episode_id": parent,
                    "terrain": str(z["terrain"][i]),
                    "t": tt,
                    "split": split,
                    "state_kind": "recovery",
                    "held_out": t in HELD,
                    "e0": float(z["e0"][i]),
                }
            )
    return rows


def _load_rollout_A():
    out = {}
    for t in SEEN + HELD:
        p = IAE / "p3b2_intent_shield" / "rollout" / f"{t}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text()).get("rows") or []:
            key = (str(r["episode_id"]), int(r["t"]))
            out[key] = r.get("methods") or {}
    return out


def _load_b3_rollout():
    out = {}
    root = Path("/data/home/chenxiangyu/robotics/Anybody/results/p3b3_tail_risk/rollout")
    for t in SEEN + HELD:
        p = root / f"{t}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text()).get("rows") or []:
            key = (str(r["episode_id"]), int(r["t"]))
            out[key] = r.get("methods") or {}
    return out


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


def auprc(y, s) -> float:
    y = np.asarray(y, dtype=np.int32)
    s = np.asarray(s, dtype=np.float64)
    npos = int(y.sum())
    if npos == 0 or npos == y.size:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    yt = y[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    prec = tp / np.clip(tp + fp, 1, None)
    rec = tp / float(npos)
    rec = np.concatenate([[0.0], rec])
    prec = np.concatenate([[1.0], prec])
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(prec, rec))


def roc_curve(y, s, n=64):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    thr = np.quantile(s[np.isfinite(s)], np.linspace(0, 1, n))
    fpr, tpr = [], []
    for t in thr[::-1]:
        pred = s >= t
        tp = np.logical_and(pred, y).sum()
        fp = np.logical_and(pred, ~y).sum()
        fn = np.logical_and(~pred, y).sum()
        tn = np.logical_and(~pred, ~y).sum()
        tpr.append(tp / max(tp + fn, 1))
        fpr.append(fp / max(fp + tn, 1))
    fpr = [0.0] + fpr + [1.0]
    tpr = [0.0] + tpr + [1.0]
    return np.asarray(fpr), np.asarray(tpr)


def _spear(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return float("nan")
    r = spearmanr(a[m], b[m])
    v = float(r.correlation)
    return v if math.isfinite(v) else float("nan")


def _boot_ci(fn, n, rng, n_boot=BOOT_N):
    if n <= 1:
        return {"mean": None, "lo": None, "hi": None}
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            v = fn(idx)
        except Exception:
            continue
        if v is not None and math.isfinite(float(v)):
            vals.append(float(v))
    if not vals:
        return {"mean": None, "lo": None, "hi": None}
    a = np.asarray(vals)
    return {"mean": float(a.mean()), "lo": float(np.quantile(a, 0.025)), "hi": float(np.quantile(a, 0.975))}


def _qmask(s, frac, top=True):
    n = s.shape[-1]
    k = max(1, int(round(frac * n)))
    if top:
        idx = np.argpartition(-s, kth=min(k, n - 1), axis=-1)[..., :k]
    else:
        idx = np.argpartition(s, kth=min(k, n - 1), axis=-1)[..., :k]
    m = np.zeros_like(s, dtype=bool)
    if s.ndim == 1:
        m[idx] = True
        return m
    row = np.arange(s.shape[0])[:, None]
    m[row, idx] = True
    return m


@torch.no_grad()
def _dir_stats(C_mem, C_gt, z, d):
    """C_mem (M,B,16,16), d (B,K,16) -> numpy dict of (B,K)."""
    sm = torch.einsum("mbij,bkj,bki->mbk", C_mem, d, d)
    mu = sm.mean(0)
    if sm.shape[0] > 1:
        sig = sm.std(0, unbiased=True)
    else:
        sig = torch.zeros_like(mu)
    smax = sm.max(0).values
    smin = sm.min(0).values
    star = torch.einsum("bij,bkj,bki->bk", C_gt, d, d)
    return {
        "mu": mu.cpu().numpy(),
        "sigma": sig.cpu().numpy(),
        "smax": smax.cpu().numpy(),
        "smin": smin.cpu().numpy(),
        "star": star.cpu().numpy(),
        "sm": sm.cpu().numpy(),
    }


@torch.no_grad()
def _project_leak(C_mem, C_gt, z, d):
    """d (B,16). Returns unit-dir leak and Rd for oracle / B1-best / mean / worst-case / members."""
    B = d.shape[0]
    C_bar = C_mem.mean(0)
    P_or = soft_P(C_gt, z, 1.0)
    P_mu = soft_P(C_bar, z, 1.0)
    z_n = F.normalize(z, dim=-1, eps=1e-8)
    dT = d - (d * z_n).sum(-1, keepdim=True) * z_n
    n0 = dT.norm(dim=-1).clamp(min=1e-12)

    def _one(P):
        dp = torch.bmm(P, dT.unsqueeze(-1)).squeeze(-1)
        dp = dp - (dp * z_n).sum(-1, keepdim=True) * z_n
        n1 = dp.norm(dim=-1)
        rd = (n1 / n0).cpu().numpy()
        dn = F.normalize(dp, dim=-1, eps=1e-8)
        leak = torch.einsum("bi,bij,bj->b", dn, C_gt, dn).cpu().numpy()
        leak_raw = torch.einsum("bi,bij,bj->b", dp, C_gt, dp).cpu().numpy()
        po = torch.bmm(P_or, dT.unsqueeze(-1)).squeeze(-1)
        cos = (dp * po).sum(-1) / (dp.norm(dim=-1) * po.norm(dim=-1) + 1e-8)
        return leak, leak_raw, rd, cos.cpu().numpy(), dn.cpu().numpy()

    out = {}
    out["oracle"] = _one(P_or)
    out["mean"] = _one(P_mu)
    s_d = torch.einsum("mbij,bj,bi->mb", C_mem, dT, dT)
    mstar = s_d.argmax(0)
    P_b1 = soft_P(C_mem[-1], z, 1.0)
    out["B1"] = _one(P_b1)
    idx = torch.arange(B, device=C_mem.device)
    P_wc = soft_P(C_mem[mstar, idx], z, 1.0)
    out["worst"] = _one(P_wc)
    out["mstar"] = mstar.cpu().numpy()
    out["s_d"] = s_d.cpu().numpy()
    return out


def _summarize_arr(x):
    return stats(x)


def _score_pack(st):
    mu, sig, smax = st["mu"], st["sigma"], st["smax"]
    return {
        "sigma_s": sig,
        "cv": sig / (mu + EPS),
        "max_minus_mean": smax - mu,
        "ensemble_max": smax,
    }


def _ff_detect(y, scores: dict, rng, n_state, dirs_per):
    out = {}
    yf = y.reshape(-1)
    for name, sc in scores.items():
        sf = sc.reshape(-1)
        au = auroc(yf, sf)
        ap = auprc(yf, sf)

        def _fn(idx, _sc=sc, _y=y):
            yy = _y[idx].reshape(-1)
            ss = _sc[idx].reshape(-1)
            return auroc(yy, ss)

        ci_au = _boot_ci(_fn, n_state, rng)
        out[name] = {"AUROC": au, "AUPRC": ap, "AUROC_CI": ci_au}
    return out


def _bin_cal(sig, star, mu, ff):
    q = np.quantile(sig, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    q[-1] = q[-1] + 1e-12
    rows = []
    for i in range(5):
        m = (sig >= q[i]) & (sig < q[i + 1])
        if not m.any():
            continue
        u_rel = star[m] / (mu[m] + EPS)
        rows.append(
            {
                "bin": f"{i * 20}-{(i + 1) * 20}%",
                "n": int(m.sum()),
                "median_s_star": float(np.median(star[m])),
                "median_mu_s": float(np.median(mu[m])),
                "median_u_rel": float(np.median(u_rel)),
                "FFR": float(ff[m].mean()) if ff is not None else None,
            }
        )
    return rows


def _risk_coverage(sig, ff, leak, A=None, fracs=(0.0, 0.10, 0.20, 0.30, 0.40)):
    n = sig.size
    order = np.argsort(-sig)
    rows = []
    for f in fracs:
        k = int(round(f * n))
        keep = order[k:]
        rec = {
            "reject": float(f),
            "coverage": float(keep.size / max(n, 1)),
            "n_keep": int(keep.size),
            "FFR": float(ff[keep].mean()) if keep.size else None,
            "median_oracle_leak": float(np.median(leak[keep])) if keep.size else None,
        }
        if A is not None and keep.size:
            rec["mean_A"] = float(np.mean(A[keep]))
            rec["median_A"] = float(np.median(A[keep]))
        rows.append(rec)
    return rows


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_scatter(path, sig, ur, ff, title):
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(sig[~ff], ur[~ff], s=6, alpha=0.25, c="#4c78a8", label="other")
    if ff.any():
        ax.scatter(sig[ff], ur[ff], s=10, alpha=0.7, c="#e45756", label="false-free")
    ax.set_xlabel("ensemble σ_s")
    ax.set_ylabel(r"$s^*/(\mu_s+\epsilon)$")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_roc(path, y, scores, title):
    y = np.asarray(y).reshape(-1).astype(bool)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for name, sc in scores.items():
        sf = np.asarray(sc, dtype=np.float64).reshape(-1)
        fpr, tpr = roc_curve(y, sf)
        ax.plot(fpr, tpr, label=f"{name} AUC={auroc(y, sf):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", c="0.5")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_bins(path, bins, title):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    xs = np.arange(len(bins))
    ax.bar(xs - 0.18, [b["median_u_rel"] for b in bins], 0.36, label="median u_rel")
    ax2 = ax.twinx()
    ax2.plot(xs + 0.18, [b["FFR"] for b in bins], "o-", c="#e45756", label="FFR")
    ax.set_xticks(xs)
    ax.set_xticklabels([b["bin"] for b in bins], rotation=20)
    ax.set_ylabel("median underestimation ratio")
    ax2.set_ylabel("false-free rate")
    ax.set_title(title)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_rc(path, rc, ykey, ylabel, title):
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    cov = [r["coverage"] for r in rc]
    ys = [r.get(ykey) for r in rc]
    ax.plot(cov, ys, "o-")
    ax.set_xlabel("retained coverage")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_pareto(path, pts, title):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for name, di, a in pts:
        ax.scatter([di], [a], s=40, label=name)
        ax.annotate(name, (di, a), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("median DI @5°")
    ax.set_ylabel("median A")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _rho_block(sig, u_rel, u_log, rng, n_state, shape):
    def _fn_rel(idx):
        return _spear(sig[idx].reshape(-1), u_rel[idx].reshape(-1))

    def _fn_log(idx):
        return _spear(sig[idx].reshape(-1), u_log[idx].reshape(-1))

    return {
        "spearman_sigma_urel": _spear(sig.reshape(-1), u_rel.reshape(-1)),
        "spearman_sigma_ulog": _spear(sig.reshape(-1), u_log.reshape(-1)),
        "spearman_sigma_urel_CI": _boot_ci(_fn_rel, n_state, rng),
        "spearman_sigma_ulog_CI": _boot_ci(_fn_log, n_state, rng),
    }


@torch.no_grad()
def _eval_split_dirs(members, x_np, z_np, C_np, pool, device, n_rand=N_RAND, n_ucr=N_UCR_LIKE, seed=DIR_SEED):
    x = torch.as_tensor(x_np, device=device)
    z = torch.as_tensor(z_np, device=device)
    C = torch.as_tensor(C_np, device=device)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    C_mem = _pred_C_stack(members, x, z)
    v_rand = random_tangent(z, n_rand, g)
    v_ucr = sample_ucr_dirs(z, pool, n_ucr, g, noise_std=0.0) if pool is not None else None
    st_rand = _dir_stats(C_mem, C, z, v_rand)
    st_ucr = _dir_stats(C_mem, C, z, v_ucr) if v_ucr is not None else None
    return C_mem, st_rand, st_ucr


def _ff_from_state(star, mu):
    sens = _qmask(star, 0.25, top=True)
    free = _qmask(mu, 0.25, top=False)
    return sens, free, sens & free


def _case(metrics) -> str:
    rho = metrics.get("spearman_seen_sens")
    au = metrics.get("ff_auroc_seen")
    rho = -1 if rho is None or not math.isfinite(rho) else rho
    au = 0.5 if au is None or not math.isfinite(au) else au
    rc20 = metrics.get("ffr_drop_20")
    if rho >= 0.3 and au >= 0.70:
        return "U1"
    if rho < 0.1 and au < 0.55:
        return "U3"
    return "U2"


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="/data/home/chenxiangyu/robotics/Anybody/results/p3b3_tail_risk")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    out = Path(args.out)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[p3b3] device={device}", flush=True)

    members = _load_ensemble("B1", device)
    try:
        members_b21 = _load_ensemble("B21", device)
    except Exception as e:
        members_b21 = None
        print(f"[p3b3] B21 ensemble skipped: {e}", flush=True)

    packed = _load_packed()
    pool = torch.as_tensor(np.load(P3B / "p3b_learned_projector" / "dataset_b" / "ucr_pool.npy"), device=device)
    ucr_states = _load_ucr_states(packed)
    roll_A = _load_rollout_A()
    roll_b3 = _load_b3_rollout()
    print(f"[p3b3] UCR states {len(ucr_states)} B1 members {len(members)}", flush=True)

    config = {
        "ensemble": [m["path"] for m in members],
        "optional_b21": [m["path"] for m in members_b21] if members_b21 else [],
        "oracle_lambda": 1.0,
        "learned_lambda_primary": 1.0,
        "random_tangent_dirs_per_state": N_RAND,
        "ucr_like_dirs_per_state": N_UCR_LIKE,
        "random_direction_seed": DIR_SEED,
        "underestimation_eps": EPS,
        "oracle_sensitive_quantile": 0.75,
        "oracle_free_quantile": 0.25,
        "beta_sweep": list(BETAS),
        "bootstrap": {"resamples": BOOT_N, "confidence_level": 0.95, "seed": BOOT_SEED},
        "no_training": True,
        "no_p4": True,
        "no_terrain_in_model": True,
    }
    (out / "config.yaml").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # ---- packed splits: D2/D3 ----
    split_dir = {}
    for sp in ("val", "test_seen", "test_slip"):
        d = packed[sp]
        print(f"[p3b3] dirs {sp} n={d['x'].shape[0]}", flush=True)
        C_mem, st_r, st_u = _eval_split_dirs(members, d["x"], d["z0"], d["C"], pool, device)
        split_dir[sp] = {
            "kind": np.asarray(d["state_kind"]).astype(str),
            "C_mem": C_mem,
            "z": torch.as_tensor(d["z0"], device=device),
            "C": torch.as_tensor(d["C"], device=device),
            "rand": st_r,
            "ucr_like": st_u,
            "x": d["x"],
        }

    val_r = split_dir["val"]["rand"]
    q_sens = float(np.quantile(val_r["star"], 0.75))
    q_free_gt = float(np.quantile(val_r["star"], 0.25))
    q_pred_free = float(np.quantile(val_r["mu"], 0.25))
    print(f"[p3b3] val global q_sens={q_sens:.4f} q_pred_free={q_pred_free:.4f}", flush=True)

    rng = np.random.RandomState(BOOT_SEED)
    dir_rows_seen = []
    dir_rows_slip = []
    pack_metrics = {}

    for sp in ("val", "test_seen", "test_slip"):
        st = split_dir[sp]["rand"]
        star, mu, sig = st["star"], st["mu"], st["sigma"]
        u_rel = star / (mu + EPS)
        u_log = np.log(star + EPS) - np.log(mu + EPS)
        u_abs = np.maximum(star - mu, 0.0)
        sens, free, ff = _ff_from_state(star, mu)
        ff_g = (star >= q_sens) & (mu <= q_pred_free)
        kind = split_dir[sp]["kind"]
        scores = _score_pack(st)
        n_state = star.shape[0]
        rho_all = _rho_block(sig, u_rel, u_log, rng, n_state, star.shape)
        det = _ff_detect(ff, scores, rng, n_state, N_RAND)
        det_g = _ff_detect(ff_g, scores, rng, n_state, N_RAND)
        bins = _bin_cal(sig.reshape(-1), star.reshape(-1), mu.reshape(-1), ff.reshape(-1))
        C_gt = split_dir[sp]["C"]
        z = split_dir[sp]["z"]
        C_mem = split_dir[sp]["C_mem"]
        # leak of random dirs through mean projector: sample 8 dirs / state for CSV size
        leak_mu = []
        with torch.no_grad():
            g = torch.Generator(device=device)
            g.manual_seed(DIR_SEED)
            v = random_tangent(z, N_RAND, g)
            P_mu = soft_P(C_mem.mean(0), z, 1.0)
            dp = torch.einsum("bij,bkj->bki", P_mu, v)
            dp = F.normalize(dp, dim=-1, eps=1e-8)
            leak = torch.einsum("bki,bij,bkj->bk", dp, C_gt, dp).cpu().numpy()
        rc = _risk_coverage(sig.reshape(-1), ff.reshape(-1), leak.reshape(-1))
        pack_metrics[sp] = {
            "n": int(n_state),
            "rho_all": rho_all,
            "rho_oracle_sensitive": {
                "spearman_sigma_urel": _spear(sig[sens], u_rel[sens]),
                "spearman_sigma_ulog": _spear(sig[sens], u_log[sens]),
            },
            "FFR_state": float(ff[sens].mean()) if sens.any() else None,
            "FFR_global": float(((star >= q_sens) & (mu <= q_pred_free)).mean()),
            "FCR_state": float((_qmask(star, 0.25, False) & _qmask(mu, 0.25, True)).sum() / max(_qmask(star, 0.25, False).sum(), 1)),
            "ff_detect_state": det,
            "ff_detect_global": det_g,
            "calibration_bins": bins,
            "risk_coverage": rc,
            "E_sigma": {
                "all": float(sig.mean()),
                "nominal": float(sig[kind == "nominal"].mean()) if (kind == "nominal").any() else None,
                "recovery": float(sig[kind == "recovery"].mean()) if (kind == "recovery").any() else None,
                "perturbed": float(sig[kind == "perturbed"].mean()) if (kind == "perturbed").any() else None,
                "false_free": float(sig[ff].mean()) if ff.any() else None,
                "not_false_free": float(sig[~ff].mean()) if (~ff).any() else None,
            },
            "median_u_rel_ff": float(np.median(u_rel[ff])) if ff.any() else None,
        }
        dest = dir_rows_slip if sp == "test_slip" else dir_rows_seen
        # subsample CSV: all FF + 4 random dirs / state
        pick = np.zeros_like(star, dtype=bool)
        pick[:, :4] = True
        pick |= ff
        for i in range(n_state):
            for k in np.where(pick[i])[0]:
                dest.append(
                    {
                        "split": sp,
                        "family": "rand",
                        "state_i": i,
                        "dir_i": int(k),
                        "state_kind": str(kind[i]),
                        "s_star": float(star[i, k]),
                        "mu_s": float(mu[i, k]),
                        "sigma_s": float(sig[i, k]),
                        "s_max": float(st["smax"][i, k]),
                        "u_rel": float(u_rel[i, k]),
                        "u_log": float(u_log[i, k]),
                        "ff_state": int(ff[i, k]),
                        "ff_global": int(ff_g[i, k]),
                        "oracle_sens_state": int(sens[i, k]),
                    }
                )

    # beta sweep on val FFR using s_UCB as predicted sensitivity
    val_star, val_mu, val_sig = val_r["star"], val_r["mu"], val_r["sigma"]
    val_sens = _qmask(val_star, 0.25, True)
    beta_rows = []
    best_beta, best_ffr = 1.0, 1e9
    for b in BETAS:
        sucb = val_mu + float(b) * val_sig
        free = _qmask(sucb, 0.25, False)
        ffr = float((val_sens & free)[val_sens].mean()) if val_sens.any() else None
        rho = _spear(sucb.reshape(-1), (val_star / (val_mu + EPS)).reshape(-1))
        au = auroc((val_sens & _qmask(val_mu, 0.25, False)).reshape(-1), sucb.reshape(-1))
        beta_rows.append({"beta": float(b), "val_FFR_ucb_free": ffr, "spearman_sucb_urel": rho, "auroc_sucb_vs_ffmu": au})
        if ffr is not None and ffr < best_ffr - 1e-12:
            best_ffr, best_beta = ffr, float(b)
        elif ffr is not None and abs(ffr - best_ffr) <= 1e-12 and float(b) < best_beta:
            best_beta = float(b)
    # if UCB-as-sensitivity does not beat mu (beta=0), keep beta=1 as specified default diagnostic
    ffr0 = next(r["val_FFR_ucb_free"] for r in beta_rows if r["beta"] == 0.0)
    if best_ffr >= (ffr0 or 1) - 1e-6:
        best_beta = 1.0
    print(f"[p3b3] selected beta={best_beta} val_FFR={best_ffr}", flush=True)
    (out / "selected_beta.json").write_text(
        json.dumps({"beta": best_beta, "val_rows": beta_rows, "reason": "min val FFR of UCB-as-sensitivity; default 1 if no gain vs beta=0"}, indent=2),
        encoding="utf-8",
    )

    # ---- D1 actual UCR d* ----
    def _stack_ucr(subset):
        x = np.stack([r["x"] for r in subset])
        z = np.stack([r["z0"] for r in subset])
        C = np.stack([r["C"] for r in subset])
        d = np.stack([r["dstar"] for r in subset])
        return x, z, C, d

    ucr_seen = [r for r in ucr_states if not r["held_out"]]
    ucr_slip = [r for r in ucr_states if r["held_out"]]
    ucr_val = [r for r in ucr_seen if r["split"] == "val"]
    print(f"[p3b3] D1 seen={len(ucr_seen)} slip={len(ucr_slip)} val={len(ucr_val)}", flush=True)

    ucr_metrics = {}
    ucr_csv = {"seen": [], "slip": []}
    for tag, subset in (("seen", ucr_seen), ("slip", ucr_slip), ("val", ucr_val)):
        if not subset:
            continue
        x, z_np, C_np, d_np = _stack_ucr(subset)
        z = torch.as_tensor(z_np, device=device)
        C = torch.as_tensor(C_np, device=device)
        d = torch.as_tensor(d_np, device=device)
        x_t = torch.as_tensor(x, device=device)
        C_mem = _pred_C_stack(members, x_t, z)
        st = _dir_stats(C_mem, C, z, d.unsqueeze(1))
        mu = st["mu"][:, 0]
        sig = st["sigma"][:, 0]
        star = st["star"][:, 0]
        u_rel = star / (mu + EPS)
        u_log = np.log(star + EPS) - np.log(mu + EPS)
        proj = _project_leak(C_mem, C, z, d)
        sucb = {b: mu + float(b) * sig for b in BETAS}
        alpha = 1.0 / (1.0 + sucb[best_beta])
        leak_or, _, rd_or, _, _ = proj["oracle"]
        leak_b1, _, rd_b1, cos_b1, _ = proj["B1"]
        leak_mu, _, rd_mu, cos_mu, _ = proj["mean"]
        leak_wc, _, rd_wc, cos_wc, _ = proj["worst"]
        A_or = A_b1 = A_mu = A_ucb = A_wc = None
        DI_or = DI_b1 = DI_mu = DI_ucb = DI_wc = None
        a_or, a_b1, di_or, di_b1 = [], [], [], []
        a_mu, a_wc, a_ucb, di_mu, di_wc, di_ucb = [], [], [], [], [], []
        for i, r in enumerate(subset):
            m = roll_A.get((r["episode_id"], r["t"])) or {}
            if "oracle" in m:
                a_or.append(m["oracle"].get("A"))
                di_or.append(m["oracle"].get("DI_5deg"))
            if "B1" in m:
                a_b1.append(m["B1"].get("A"))
                di_b1.append(m["B1"].get("DI_5deg"))
            mb = roll_b3.get((r["episode_id"], r["t"])) or {}
            if "R2_mean" in mb:
                a_mu.append(mb["R2_mean"].get("A"))
                di_mu.append(mb["R2_mean"].get("DI_5deg"))
            if "R4_worst" in mb:
                a_wc.append(mb["R4_worst"].get("A"))
                di_wc.append(mb["R4_worst"].get("DI_5deg"))
            if "R3_ucb" in mb:
                a_ucb.append(mb["R3_ucb"].get("A"))
                di_ucb.append(mb["R3_ucb"].get("DI_5deg"))
        # false-free on UCR: global val thresholds
        ff_g = (star >= q_sens) & (mu <= q_pred_free)
        rho = _rho_block(sig[:, None], u_rel[:, None], u_log[:, None], rng, star.size, (star.size, 1))
        scores = {
            "sigma_s": sig,
            "cv": sig / (mu + EPS),
            "max_minus_mean": st["smax"][:, 0] - mu,
            "ensemble_max": st["smax"][:, 0],
        }
        yff = ff_g
        det = {}
        for nm, sc in scores.items():
            det[nm] = {"AUROC": auroc(yff, sc), "AUPRC": auprc(yff, sc)}
        A_b1_arr = np.asarray([m["B1"]["A"] if (roll_A.get((r["episode_id"], r["t"])) or {}).get("B1") else np.nan for r in subset], dtype=np.float64)
        leak_for_rc = leak_mu
        rc = _risk_coverage(sig, ff_g, leak_for_rc, A=np.where(np.isfinite(A_b1_arr), A_b1_arr, np.nan))
        ucr_metrics[tag] = {
            "n": int(len(subset)),
            "rho": rho,
            "ff_global": float(ff_g.mean()),
            "ff_detect": det,
            "E_sigma": float(sig.mean()),
            "E_sigma_ff": float(sig[ff_g].mean()) if ff_g.any() else None,
            "E_sigma_not_ff": float(sig[~ff_g].mean()) if (~ff_g).any() else None,
            "leak": {
                "oracle": _summarize_arr(leak_or),
                "B1": _summarize_arr(leak_b1),
                "mean": _summarize_arr(leak_mu),
                "worst": _summarize_arr(leak_wc),
            },
            "Rd": {
                "oracle": _summarize_arr(rd_or),
                "B1": _summarize_arr(rd_b1),
                "mean": _summarize_arr(rd_mu),
                "worst": _summarize_arr(rd_wc),
            },
            "cos": {
                "B1": _summarize_arr(cos_b1),
                "mean": _summarize_arr(cos_mu),
                "worst": _summarize_arr(cos_wc),
            },
            "alpha": _summarize_arr(alpha),
            "rollout_A": {
                "oracle": adv_block(a_or) if a_or else None,
                "B1": adv_block(a_b1) if a_b1 else None,
                "R2_mean": adv_block(a_mu) if a_mu else None,
                "R3_ucb": adv_block(a_ucb) if a_ucb else None,
                "R4_worst": adv_block(a_wc) if a_wc else None,
            },
            "rollout_DI": {
                "oracle": stats(di_or) if di_or else None,
                "B1": stats(di_b1) if di_b1 else None,
                "R2_mean": stats(di_mu) if di_mu else None,
                "R3_ucb": stats(di_ucb) if di_ucb else None,
                "R4_worst": stats(di_wc) if di_wc else None,
            },
            "risk_coverage": rc,
        }
        dest = ucr_csv["slip"] if tag == "slip" else ucr_csv["seen"]
        if tag == "val":
            dest = None
        if dest is not None:
            for i, r in enumerate(subset):
                m = roll_A.get((r["episode_id"], r["t"])) or {}
                mb = roll_b3.get((r["episode_id"], r["t"])) or {}
                dest.append(
                    {
                        "episode_id": r["episode_id"],
                        "terrain": r["terrain"],
                        "t": r["t"],
                        "split": r["split"],
                        "s_star": float(star[i]),
                        "mu_s": float(mu[i]),
                        "sigma_s": float(sig[i]),
                        "u_rel": float(u_rel[i]),
                        "ff_global": int(ff_g[i]),
                        "leak_oracle": float(leak_or[i]),
                        "leak_B1": float(leak_b1[i]),
                        "leak_mean": float(leak_mu[i]),
                        "leak_worst": float(leak_wc[i]),
                        "Rd_mean": float(rd_mu[i]),
                        "Rd_worst": float(rd_wc[i]),
                        "cos_mean": float(cos_mu[i]),
                        "alpha_ucb": float(alpha[i]),
                        "A_oracle": (m.get("oracle") or {}).get("A"),
                        "A_B1": (m.get("B1") or {}).get("A"),
                        "DI_oracle": (m.get("oracle") or {}).get("DI_5deg"),
                        "DI_B1": (m.get("B1") or {}).get("DI_5deg"),
                        "A_R2": (mb.get("R2_mean") or {}).get("A"),
                        "A_R3": (mb.get("R3_ucb") or {}).get("A"),
                        "A_R4": (mb.get("R4_worst") or {}).get("A"),
                        "DI_R2": (mb.get("R2_mean") or {}).get("DI_5deg"),
                        "DI_R3": (mb.get("R3_ucb") or {}).get("DI_5deg"),
                        "DI_R4": (mb.get("R4_worst") or {}).get("DI_5deg"),
                    }
                )

    # B21 sanity Spearman on test_seen random dirs
    b21_sanity = None
    if members_b21 is not None:
        d = packed["test_seen"]
        _Cm, st_r, _ = _eval_split_dirs(members_b21, d["x"], d["z0"], d["C"], pool, device)
        star, mu, sig = st_r["star"], st_r["mu"], st_r["sigma"]
        u_rel = star / (mu + EPS)
        sens, free, ff = _ff_from_state(star, mu)
        b21_sanity = {
            "spearman_sigma_urel": _spear(sig, u_rel),
            "FF_AUROC_sigma": auroc(ff.reshape(-1), sig.reshape(-1)),
            "E_sigma": float(sig.mean()),
            "note": "B21 seeds are fine-tunes of the same B1_s2028, not independent",
        }

    # plots
    try:
        ts = split_dir["test_seen"]["rand"]
        sl = split_dir["test_slip"]["rand"]
        ff_ts = _ff_from_state(ts["star"], ts["mu"])[2]
        ff_sl = _ff_from_state(sl["star"], sl["mu"])[2]
        _plot_scatter(plots / "B3-1_seen.png", ts["sigma"].reshape(-1), (ts["star"] / (ts["mu"] + EPS)).reshape(-1), ff_ts.reshape(-1), "B3-1 seen test random dirs")
        _plot_scatter(plots / "B3-1_slip.png", sl["sigma"].reshape(-1), (sl["star"] / (sl["mu"] + EPS)).reshape(-1), ff_sl.reshape(-1), "B3-1 held-out slip random dirs")
        _plot_roc(plots / "B3-2_seen_roc.png", ff_ts.reshape(-1), _score_pack(ts), "B3-2 false-free ROC (seen)")
        _plot_roc(plots / "B3-2_slip_roc.png", ff_sl.reshape(-1), _score_pack(sl), "B3-2 false-free ROC (slip)")
        _plot_bins(plots / "B3-3_seen_bins.png", pack_metrics["test_seen"]["calibration_bins"], "B3-3 seen uncertainty bins")
        _plot_bins(plots / "B3-3_slip_bins.png", pack_metrics["test_slip"]["calibration_bins"], "B3-3 slip uncertainty bins")
        _plot_rc(plots / "B3-4_seen_ffr.png", pack_metrics["test_seen"]["risk_coverage"], "FFR", "false-free rate", "B3-4 seen risk-coverage FFR")
        if ucr_metrics.get("seen"):
            _plot_rc(plots / "B3-4_ucr_A.png", ucr_metrics["seen"]["risk_coverage"], "mean_A", "mean B1 recovery A", "B3-4 UCR retained mean A")
            _plot_rc(plots / "B3-4_ucr_ffr.png", ucr_metrics["seen"]["risk_coverage"], "FFR", "false-free rate", "B3-4 UCR d* risk-coverage FFR")
    except Exception as e:
        print(f"[p3b3] plot warn: {e}", flush=True)

    pts = []
    um = ucr_metrics.get("seen") or {}
    for name, akey, dikey in (
        ("Oracle", "oracle", "oracle"),
        ("B1", "B1", "B1"),
        ("ensemble mean", "R2_mean", "R2_mean"),
        ("UCB", "R3_ucb", "R3_ucb"),
        ("worst-case", "R4_worst", "R4_worst"),
    ):
        A = (um.get("rollout_A") or {}).get(akey)
        DI = (um.get("rollout_DI") or {}).get(dikey)
        if A and DI and A.get("median") is not None and DI.get("median") is not None:
            pts.append((name, DI["median"], A["median"]))
    if len(pts) >= 2:
        _plot_pareto(plots / "B3-5_pareto.png", pts, "B3-5 DI vs A (UCR seen)")
    # B3-6 seen vs slip sigma
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.hist(ts["sigma"].reshape(-1), bins=40, density=True, alpha=0.5, label="seen")
    ax.hist(sl["sigma"].reshape(-1), bins=40, density=True, alpha=0.5, label="slip")
    ax.set_xlabel("σ_s")
    ax.set_ylabel("density")
    ax.set_title("B3-6 ensemble uncertainty seen vs slip")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plots / "B3-6_seen_vs_slip.png", dpi=140)
    plt.close(fig)

    _write_csv(out / "direction_level_seen.csv", dir_rows_seen)
    _write_csv(out / "direction_level_slip.csv", dir_rows_slip)
    _write_csv(out / "ucr_rollout_seen.csv", ucr_csv["seen"])
    _write_csv(out / "ucr_rollout_slip.csv", ucr_csv["slip"])
    cal_rows = []
    for sp, blk in pack_metrics.items():
        for b in blk["calibration_bins"]:
            cal_rows.append({"split": sp, **b})
    _write_csv(out / "uncertainty_calibration.csv", cal_rows)
    rc_rows = []
    for sp, blk in pack_metrics.items():
        for r in blk["risk_coverage"]:
            rc_rows.append({"split": sp, "family": "rand", **r})
    for tag in ("seen", "slip"):
        if tag in ucr_metrics:
            for r in ucr_metrics[tag]["risk_coverage"]:
                rc_rows.append({"split": tag, "family": "ucr_dstar", **r})
    _write_csv(out / "risk_coverage.csv", rc_rows)

    seen_ff_au = ((pack_metrics["test_seen"]["ff_detect_state"].get("sigma_s") or {}).get("AUROC"))
    slip_ff_au = ((pack_metrics["test_slip"]["ff_detect_state"].get("sigma_s") or {}).get("AUROC"))
    best_score_seen = max(
        pack_metrics["test_seen"]["ff_detect_state"].items(),
        key=lambda kv: -1 if kv[1]["AUROC"] != kv[1]["AUROC"] else kv[1]["AUROC"],
    )
    rho_seen = pack_metrics["test_seen"]["rho_all"]["spearman_sigma_urel"]
    rho_sens = pack_metrics["test_seen"]["rho_oracle_sensitive"]["spearman_sigma_urel"]
    rho_ucr = (ucr_metrics.get("seen") or {}).get("rho", {}).get("spearman_sigma_urel")
    ffr0 = pack_metrics["test_seen"]["risk_coverage"][0]["FFR"]
    ffr20 = next((r["FFR"] for r in pack_metrics["test_seen"]["risk_coverage"] if abs(r["reject"] - 0.2) < 1e-9), None)
    ffr_drop_20 = (ffr0 - ffr20) / ffr0 if ffr0 and ffr20 is not None and ffr0 else None
    e_sig_ff = pack_metrics["test_seen"]["E_sigma"]["false_free"]
    e_sig_nff = pack_metrics["test_seen"]["E_sigma"]["not_false_free"]
    confidently_wrong = bool(e_sig_ff is not None and e_sig_nff is not None and e_sig_ff <= e_sig_nff)

    case_in = {
        "spearman_seen_sens": rho_sens,
        "ff_auroc_seen": seen_ff_au,
        "ffr_drop_20": ffr_drop_20,
    }
    case = _case(case_in)

    # leak improvement mean vs B1
    leak_gain = None
    if ucr_metrics.get("seen"):
        lb = ucr_metrics["seen"]["leak"]["B1"]["median"]
        lm = ucr_metrics["seen"]["leak"]["mean"]["median"]
        lo = ucr_metrics["seen"]["leak"]["oracle"]["median"]
        lw = ucr_metrics["seen"]["leak"]["worst"]["median"]
        leak_gain = {
            "B1_over_oracle": (lb / lo) if lo else None,
            "mean_over_oracle": (lm / lo) if lo else None,
            "worst_over_oracle": (lw / lo) if lo else None,
            "mean_vs_B1": (lm / lb) if lb else None,
            "worst_vs_B1": (lw / lb) if lb else None,
        }

    (out / "selected_beta.json").write_text(
        json.dumps({"beta": best_beta, "val_rows": beta_rows, "reason": "min val FFR of UCB-as-sensitivity; default 1 if no gain vs beta=0"}, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "ensemble": [m["path"] for m in members],
        "n_members": len(members),
        "global_val_thresholds": {"q_sens": q_sens, "q_pred_free": q_pred_free, "q_free_gt": q_free_gt},
        "selected_beta": best_beta,
        "beta_sweep_val": beta_rows,
        "packed": pack_metrics,
        "ucr": ucr_metrics,
        "b21_sanity": b21_sanity,
        "case": case,
        "best_ff_score_seen": {"name": best_score_seen[0], **best_score_seen[1]},
        "leak_gain_ucr_seen": leak_gain,
        "confidently_wrong_seen": confidently_wrong,
        "no_training": True,
        "no_p4": True,
    }
    (out / "metrics.json").write_text(json.dumps(sanitize(metrics), indent=2), encoding="utf-8")

    def _au(sp, score="sigma_s"):
        return (pack_metrics[sp]["ff_detect_state"].get(score) or {}).get("AUROC")

    def _ap(sp, score="sigma_s"):
        return (pack_metrics[sp]["ff_detect_state"].get(score) or {}).get("AUPRC")

    lines = [
        "# P3-B3 Ensemble Tail-Risk Diagnostic",
        "",
        "Post-hoc only. No new projector. No P4. Frozen Stage-2 / Mapper-B / decoder.",
        "Primary ensemble: independent B1 seeds 2026/2027/2028. λ=1. Instantaneous state only.",
        "",
        f"Selected UCB β (val only): **{best_beta}**",
        f"Decision case: **{case}**",
        "",
        "## 1. Does ensemble disagreement increase when intent sensitivity is underestimated?",
        "",
        f"- Seen-test Spearman(σ, u_rel) all random dirs: **{_fmt(rho_seen)}** "
        f"(CI {pack_metrics['test_seen']['rho_all']['spearman_sigma_urel_CI']})",
        f"- Oracle-sensitive top-25%: **{_fmt(rho_sens)}**",
        f"- Actual UCR d* seen: **{_fmt(rho_ucr)}**",
        f"- Slip random dirs: **{_fmt(pack_metrics['test_slip']['rho_all']['spearman_sigma_urel'])}**",
        f"- Slip UCR d*: **{_fmt((ucr_metrics.get('slip') or {}).get('rho', {}).get('spearman_sigma_urel'))}**",
        "",
        "## 2. Spearman correlation between uncertainty and underestimation",
        "",
        f"| split | all dirs | oracle-sensitive | UCR d* |",
        f"|---|---:|---:|---:|",
        f"| seen-test | {_fmt(rho_seen)} | {_fmt(rho_sens)} | {_fmt(rho_ucr)} |",
        f"| slip | {_fmt(pack_metrics['test_slip']['rho_all']['spearman_sigma_urel'])} | "
        f"{_fmt(pack_metrics['test_slip']['rho_oracle_sensitive']['spearman_sigma_urel'])} | "
        f"{_fmt((ucr_metrics.get('slip') or {}).get('rho', {}).get('spearman_sigma_urel'))} |",
        "",
        "## 3–5. False-free detection",
        "",
        f"| score | seen AUROC | seen AUPRC | slip AUROC |",
        f"|---|---:|---:|---:|",
    ]
    for sc in ("sigma_s", "cv", "max_minus_mean", "ensemble_max"):
        lines.append(
            f"| {sc} | {_fmt(_au('test_seen', sc))} | {_fmt(_ap('test_seen', sc))} | {_fmt(_au('test_slip', sc))} |"
        )
    lines += [
        "",
        f"Best seen FF score: **{best_score_seen[0]}** AUROC={_fmt(best_score_seen[1]['AUROC'])} "
        f"AUPRC={_fmt(best_score_seen[1]['AUPRC'])}",
        f"Per-state FFR seen: {_fmt(pack_metrics['test_seen']['FFR_state'], 4)}  "
        f"slip: {_fmt(pack_metrics['test_slip']['FFR_state'], 4)}",
        "",
        "## 6. Are false-free directions confidently wrong or uncertain?",
        "",
        f"- E[σ | FF] seen random = {_fmt(e_sig_ff, 4)}",
        f"- E[σ | not FF] = {_fmt(e_sig_nff, 4)}",
        f"- Confidently wrong (FF less uncertain than others): **{confidently_wrong}**",
        "",
        "## 7. Held-out slip",
        "",
        f"Same ensemble / β / val thresholds. Slip Spearman={_fmt(pack_metrics['test_slip']['rho_all']['spearman_sigma_urel'])}, "
        f"FF AUROC={_fmt(slip_ff_au)}.",
        "",
        "## 8. Ensemble mean metric vs best single B1 (UCR d* oracle leakage)",
        "",
    ]
    if leak_gain:
        lines += [
            f"| projector | median leak | /oracle | Rd med | cos med |",
            f"|---|---:|---:|---:|---:|",
            f"| oracle | {_fmt(ucr_metrics['seen']['leak']['oracle']['median'], 4)} | 1.00 | {_fmt(ucr_metrics['seen']['Rd']['oracle']['median'], 3)} | 1.00 |",
            f"| B1 s2028 | {_fmt(ucr_metrics['seen']['leak']['B1']['median'], 4)} | {_fmt(leak_gain['B1_over_oracle'], 2)} | {_fmt(ucr_metrics['seen']['Rd']['B1']['median'], 3)} | {_fmt(ucr_metrics['seen']['cos']['B1']['median'], 3)} |",
            f"| ensemble mean | {_fmt(ucr_metrics['seen']['leak']['mean']['median'], 4)} | {_fmt(leak_gain['mean_over_oracle'], 2)} | {_fmt(ucr_metrics['seen']['Rd']['mean']['median'], 3)} | {_fmt(ucr_metrics['seen']['cos']['mean']['median'], 3)} |",
            f"| worst-case member | {_fmt(ucr_metrics['seen']['leak']['worst']['median'], 4)} | {_fmt(leak_gain['worst_over_oracle'], 2)} | {_fmt(ucr_metrics['seen']['Rd']['worst']['median'], 3)} | {_fmt(ucr_metrics['seen']['cos']['worst']['median'], 3)} |",
            "",
        ]
    lines += [
        "## 9–10. Direction-wise UCB / recovery authority",
        "",
        f"UCB α=1/(1+μ+βσ) with β={best_beta} on UCR d*. Median α seen = "
        f"{_fmt((ucr_metrics.get('seen') or {}).get('alpha', {}).get('median'), 3)}.",
        "Version A scales geodesic angle; A/DI require Isaac R3 (merged if present).",
        f"Seen UCR median A oracle={_fmt(((um.get('rollout_A') or {}).get('oracle') or {}).get('median'), 4)} "
        f"B1={_fmt(((um.get('rollout_A') or {}).get('B1') or {}).get('median'), 4)} "
        f"R2={_fmt(((um.get('rollout_A') or {}).get('R2_mean') or {}).get('median'), 4)} "
        f"R3={_fmt(((um.get('rollout_A') or {}).get('R3_ucb') or {}).get('median'), 4)} "
        f"R4={_fmt(((um.get('rollout_A') or {}).get('R4_worst') or {}).get('median'), 4)}",
        "",
        "## 11. Risk-coverage: reject most uncertain 10–20%",
        "",
        f"Seen random-dir FFR @0%={_fmt(ffr0, 4)} @20%={_fmt(ffr20, 4)} relative drop={_fmt(ffr_drop_20, 2)}.",
        "",
    ]
    if ucr_metrics.get("seen"):
        rcu = ucr_metrics["seen"]["risk_coverage"]
        lines.append("UCR d* coverage table:")
        lines.append("| reject | coverage | FFR | median leak | mean B1 A |")
        lines.append("|---:|---:|---:|---:|---:|")
        for r in rcu:
            lines.append(
                f"| {r['reject']:.2f} | {r['coverage']:.2f} | {_fmt(r.get('FFR'), 4)} | "
                f"{_fmt(r.get('median_oracle_leak'), 4)} | {_fmt(r.get('mean_A'), 4)} |"
            )
        lines.append("")
    lines += [
        "## 12. Is an uncertainty-aware Intent Shield justified?",
        "",
    ]
    if case == "U1":
        lines.append("**Yes (U1).** Ensemble uncertainty tracks underestimation and detects false-free directions well enough to try an uncertainty-aware shield next. Do not implement it in this round.")
    elif case == "U2":
        lines.append("**No (U2).** Ensemble uncertainty is only weakly aligned with high-risk intent-sensitivity error. Do not build a full uncertainty shield from this ensemble.")
        lines.append("Recommended next (not implemented): top-eigenmode spectral shield, or direction-conditioned intent-sensitivity.")
    else:
        lines.append("**No (U3).** The deterministic B1 seeds share the same blind spots: false-free errors are not marked by disagreement.")
        lines.append("Do not build an uncertainty ensemble. Next should explicitly target dangerous spectral modes or direction-conditioned sensitivity.")
    lines += [
        "",
        "## 13. If not U1, next method?",
        "",
    ]
    if case == "U1":
        lines.append("Uncertainty-aware Intent Shield (stop here; do not implement automatically).")
    else:
        lines.append("Prefer **top-eigenmode spectral intent shield** (high-risk C_I modes) over another seed ensemble. Direction-conditioned sensitivity is the alternative if spectral reconstruction still misses the tail.")
    if b21_sanity:
        lines += [
            "",
            "## B21 ensemble sanity",
            "",
            f"Not independent (all init from B1_s2028). Spearman={_fmt(b21_sanity['spearman_sigma_urel'])} FF-AUROC={_fmt(b21_sanity['FF_AUROC_sigma'])}.",
        ]
    lines += [
        "",
        "## Nominal vs recovery vs slip E[σ]",
        "",
        f"- val nominal { _fmt(pack_metrics['val']['E_sigma']['nominal'], 4) } recovery { _fmt(pack_metrics['val']['E_sigma']['recovery'], 4) }",
        f"- seen-test nominal { _fmt(pack_metrics['test_seen']['E_sigma']['nominal'], 4) } recovery { _fmt(pack_metrics['test_seen']['E_sigma']['recovery'], 4) }",
        f"- slip { _fmt(pack_metrics['test_slip']['E_sigma']['all'], 4) }",
        "",
        "Larger global OOD σ is not enough; we care about σ on false-free high-risk directions.",
        "",
        "No P4. No new training. Deterministic full-PSD reconstruction remains stopped.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"case": case, "beta": best_beta, "rho_seen": rho_seen, "FF_AUROC": seen_ff_au, "best_score": best_score_seen[0]}, indent=2), flush=True)
    print(f"[p3b3] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
