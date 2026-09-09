#!/usr/bin/env python3
"""UFR-1: intent-conditioned physical innovation + tiny CFM.

Frozen Parent / Mapper-B / R-M3. Offline. No PPO, no recovery.
No task ID / terrain / λ in the model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DT = 0.02
WARMUP = 10
H_PHYS = 8  # 160 ms
H_FAIL = 25  # 0.5 s physical-risk label
PREFAIL_MARGIN = 50  # do not train on last 1.0 s before fail
TARGET_FPR = 0.10
TASKS = ("loco", "stoop", "reach", "carry")
PHYS_SLICE = slice(26, 37)  # root v/a/w, roll, pitch, wrist_cf
Z_SLICE = slice(37, 53)
N_PHYS = 11
ROLL_I, PITCH_I = 8, 9


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


def n_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


def _auroc(y, s) -> float:
    y = np.asarray(y).astype(np.int32)
    s = np.asarray(s, dtype=np.float64)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    y_s = y[order]
    ranks = np.empty(len(y_s), dtype=np.float64)
    i = 0
    while i < len(y_s):
        j = i + 1
        while j < len(y_s) and s[order[j]] == s[order[i]]:
            j += 1
        ranks[i:j] = 0.5 * (i + j + 1)
        i = j
    return float((ranks[y_s == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _auprc(y, s) -> float:
    y = np.asarray(y).astype(np.int32)
    s = np.asarray(s, dtype=np.float64)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    n_pos = int(y.sum())
    if n_pos == 0 or int((1 - y).sum()) == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / n_pos
    rec = np.concatenate([[0.0], rec])
    prec = np.concatenate([[1.0], prec])
    return float(np.trapz(prec, rec))


def _threshold_at_fpr(neg_scores, target=TARGET_FPR) -> float:
    s = np.sort(np.asarray(neg_scores, dtype=np.float64))
    if len(s) == 0:
        return float("inf")
    k = int(math.floor((1.0 - target) * len(s)))
    k = min(max(k, 0), len(s) - 1)
    return float(s[k])


def wrap_angle(d: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(d), np.cos(d)).astype(np.float32)


def phys_delta(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    d = (x1 - x0).astype(np.float32)
    d[..., ROLL_I] = wrap_angle(d[..., ROLL_I])
    d[..., PITCH_I] = wrap_angle(d[..., PITCH_I])
    return d


def clip_split(eps: list[dict]) -> dict[str, str]:
    """Same fail-stratified clip split as UFR-0."""
    assign = {}
    fail_clips = {f"{e['task']}/{e['clip']}" for e in eps if int(e.get("fail") or 0) == 1}

    def _chunk(items: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(items)
        if n == 0:
            return [], [], []
        if n == 1:
            return [], [], items
        if n == 2:
            return items[:1], [], items[1:]
        if n == 3:
            return items[:1], items[1:2], items[2:]
        n_te = max(1, int(round(0.20 * n)))
        n_va = max(1, int(round(0.20 * n)))
        n_tr = n - n_te - n_va
        if n_tr < 1:
            n_tr, n_va, n_te = 1, max(1, n - 2), n - 1 - max(1, n - 2)
        return items[:n_tr], items[n_tr : n_tr + n_va], items[n_tr + n_va :]

    for task in TASKS:
        clips = sorted({e["clip"] for e in eps if e["task"] == task})
        pos = sorted(c for c in clips if f"{task}/{c}" in fail_clips)
        neg = sorted(c for c in clips if f"{task}/{c}" not in fail_clips)
        tr, va, te = [], [], []
        for group in (pos, neg):
            a, b, c = _chunk(group)
            tr.extend(a)
            va.extend(b)
            te.extend(c)
        if not te and tr:
            te.append(tr.pop())
        if not va and tr:
            va.append(tr.pop())
        for c in tr:
            assign[f"{task}/{c}"] = "train"
        for c in va:
            assign[f"{task}/{c}"] = "val"
        for c in te:
            assign[f"{task}/{c}"] = "test"
    return assign


def load_cells(raw_root: Path) -> list[dict]:
    eps = []
    for npz_path in sorted(raw_root.glob("*/lam_*/plane/ufr_steps.npz")):
        task = npz_path.parts[-4]
        blob = np.load(npz_path, allow_pickle=True)
        meta = json.loads(str(blob["meta"].item()))
        feat = blob["feat"].astype(np.float32)
        re = blob["re"].astype(np.float32)
        rs = blob["rs"].astype(np.float32)
        t_all = blob["t"].astype(np.int32)
        ep_idx = blob["ep_idx"].astype(np.int32)
        clips = [str(x) for x in blob["clip"].tolist()]
        seeds = blob["seed"].astype(np.int32)
        fails = blob["fail"].astype(np.int32)
        fsteps = blob["fail_step"].astype(np.int32)
        srs = blob["sr_task"].astype(np.int32)
        lam = float(meta.get("lambda_id", -1))
        oor = bool(meta.get("HUMAN_INTENT_OUT_OF_PLAUSIBLE_RANGE"))
        for ei, clip in enumerate(clips):
            m = ep_idx == ei
            fe = feat[m]
            tt = t_all[m]
            order = np.argsort(tt)
            fe, tt = fe[order], tt[order]
            T = int(fe.shape[0])
            fs = int(fsteps[ei])
            use_t = T if fs < 0 else min(T, fs + 1)
            fe = fe[:use_t]
            y_phys = np.zeros(use_t, dtype=np.int32)
            if fs >= 0:
                for k in range(use_t):
                    y_phys[k] = int(k < fs <= k + H_FAIL)
            valid = np.zeros(use_t, dtype=bool)
            train_ok = np.zeros(use_t, dtype=bool)
            for k in range(use_t):
                if k < WARMUP:
                    continue
                if k + H_PHYS >= use_t:
                    continue
                valid[k] = True
                if fs < 0:
                    train_ok[k] = True
                elif k + H_PHYS < fs - PREFAIL_MARGIN:
                    train_ok[k] = True
            phys = fe[:, PHYS_SLICE]
            dx = np.zeros((use_t, N_PHYS), dtype=np.float32)
            for k in range(use_t):
                if k + H_PHYS < use_t:
                    dx[k] = phys_delta(phys[k], phys[k + H_PHYS])
            eps.append(
                {
                    "task": task,
                    "lam": lam,
                    "oor": oor,
                    "clip": clip,
                    "seed": int(seeds[ei]),
                    "fail": int(fails[ei]),
                    "fail_step": fs,
                    "sr_task": int(srs[ei]),
                    "feat": fe,
                    "re": re[m][order][:use_t],
                    "rs": rs[m][order][:use_t],
                    "dx": dx,
                    "y_phys": y_phys,
                    "valid": valid,
                    "train_ok": train_ok,
                }
            )
    return eps


def stack_pairs(eps, which: str):
    ctx, dx, meta = [], [], []
    for e in eps:
        mask = e["train_ok"] if which == "train_nom" else e["valid"]
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        ctx.append(e["feat"][idx])
        dx.append(e["dx"][idx])
        for t in idx:
            meta.append(
                {
                    "task": e["task"],
                    "lam": e["lam"],
                    "clip": e["clip"],
                    "seed": e["seed"],
                    "sr_task": e["sr_task"],
                    "fail": e["fail"],
                    "fail_step": e["fail_step"],
                    "t": int(t),
                    "re": float(e["re"][t]),
                    "rs": float(e["rs"][t]),
                    "y_phys": int(e["y_phys"][t]),
                }
            )
    if not ctx:
        raise SystemExit(f"no frames for {which}")
    return np.concatenate(ctx, 0), np.concatenate(dx, 0), meta


class DynMLP(nn.Module):
    def __init__(self, c_dim: int, x_dim: int = N_PHYS, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, c):
        return self.net(c)


class TinyCondFM(nn.Module):
    """Conditional MLP vector field. Target <10M, typically ~1–3M."""

    def __init__(self, x_dim: int, c_dim: int, hidden: int = 512, n_layers: int = 5, t_dim: int = 64):
        super().__init__()
        self.t_emb = nn.Sequential(nn.Linear(1, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.c_emb = nn.Sequential(nn.Linear(c_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.in_proj = nn.Linear(x_dim + t_dim + hidden, hidden)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
                for _ in range(max(n_layers - 1, 1))
            ]
        )
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, x_dim))

    def forward(self, x, t, c):
        te = self.t_emb(t.view(-1, 1))
        ce = self.c_emb(c)
        h = self.in_proj(torch.cat([x, te, ce], dim=-1))
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


def drop_z(c: np.ndarray, use_z: bool) -> np.ndarray:
    if use_z:
        return c
    return np.concatenate([c[:, : PHYS_SLICE.stop],], axis=1)[:, : PHYS_SLICE.stop]


def c_dim(use_z: bool) -> int:
    return 53 if use_z else 37


def train_mlp(model, c, y, epochs, device, lr=1e-3, batch=1024):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.from_numpy(c), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def train_fm(model, c, y, epochs, device, lr=1e-3, batch=1024):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.from_numpy(c), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    model.train()
    for _ in range(epochs):
        for cb, x1 in dl:
            cb = cb.to(device)
            x1 = x1.to(device)
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=device)
            xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1
            v = model(xt, t, cb)
            loss = ((v - (x1 - x0)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def mlp_resid(model, c, y, device, batch=4096):
    outs = []
    for i in range(0, len(c), batch):
        pred = model(torch.from_numpy(c[i : i + batch]).to(device)).cpu().numpy()
        outs.append(pred)
    hat = np.concatenate(outs, 0)
    r = y - hat
    return np.sqrt((r * r).mean(axis=1) + 1e-12), hat


@torch.no_grad()
def fm_sample(model, c, k: int, nfe: int, device, batch=512):
    """Return samples [N, K, D] in normalized Δx space."""
    n, d = c.shape[0], N_PHYS
    all_s = np.zeros((n, k, d), dtype=np.float32)
    dt = 1.0 / nfe
    for i in range(0, n, batch):
        cb = torch.from_numpy(c[i : i + batch]).to(device)
        b = cb.shape[0]
        x = torch.randn(b * k, d, device=device)
        cc = cb.repeat_interleave(k, dim=0)
        for s in range(nfe):
            t = torch.full((b * k,), s * dt, device=device)
            x = x + dt * model(x, t, cc)
        all_s[i : i + b] = x.view(b, k, d).cpu().numpy()
    return all_s


def fm_scores(samples: np.ndarray, y: np.ndarray) -> dict:
    # samples [N,K,D], y [N,D]
    diff = samples - y[:, None, :]
    dist = np.sqrt((diff * diff).mean(axis=-1) + 1e-12)  # [N,K]
    return {
        "min": dist.min(axis=1),
        "mean": dist.mean(axis=1),
        "var": dist.var(axis=1),
        "softmin": -np.logaddexp.reduce(-dist * 8.0, axis=1) / 8.0,
    }


def metrics_block(y, s) -> dict:
    y = np.asarray(y, dtype=np.int32)
    s = np.asarray(s, dtype=np.float64)
    return {
        "n": int(len(y)),
        "pos": int(y.sum()),
        "pos_rate": float(y.mean()) if len(y) else float("nan"),
        "auroc": _auroc(y, s),
        "auprc": _auprc(y, s),
    }


def per_task_metrics(meta, y, s) -> dict:
    out = {"overall": metrics_block(y, s)}
    for task in TASKS:
        m = np.array([r["task"] == task for r in meta])
        out[task] = metrics_block(y[m], s[m])
    return out


def ep_trigger(meta, scores, thresh) -> dict:
    by = {}
    for i, r in enumerate(meta):
        key = (r["task"], r["clip"], r["seed"], r["lam"])
        by.setdefault(
            key,
            {
                "task": r["task"],
                "sr": r["sr_task"],
                "fail": r["fail"],
                "fail_step": r["fail_step"],
                "hit": False,
                "first": None,
            },
        )
        if scores[i] >= thresh:
            by[key]["hit"] = True
            if by[key]["first"] is None:
                by[key]["first"] = r["t"]

    def pack(task=None):
        items = [v for k, v in by.items() if task is None or k[0] == task]
        succ = [v for v in items if v["sr"] == 1]
        fail = [v for v in items if v["fail"] == 1]
        leads = []
        for v in fail:
            if v["hit"] and v["first"] is not None and v["fail_step"] >= 0:
                leads.append((v["fail_step"] - v["first"]) * DT)
        rec = float(np.mean([1.0 if v["hit"] else 0.0 for v in fail])) if fail else float("nan")
        return {
            "n_success_ep": len(succ),
            "n_fail_ep": len(fail),
            "success_ep_fpr": float(np.mean([1.0 if v["hit"] else 0.0 for v in succ])) if succ else float("nan"),
            "fail_recall": rec,
            "lead_time_s_mean": float(np.mean(leads)) if leads else float("nan"),
            "lead_n": len(leads),
        }

    out = {"overall": pack(None)}
    for t in TASKS:
        out[t] = pack(t)
    return out


def choose_thresh(meta, scores) -> float:
    neg = [scores[i] for i, r in enumerate(meta) if r["sr_task"] == 1 and r["y_phys"] == 0]
    if len(neg) < 20:
        neg = [scores[i] for i, r in enumerate(meta) if r["y_phys"] == 0]
    return _threshold_at_fpr(neg, TARGET_FPR)


def method_pack(va_meta, va_s, te_meta, te_s, te_y):
    th = choose_thresh(va_meta, va_s)
    ep = ep_trigger(te_meta, te_s, th)
    return {
        "thresh": th,
        "frame": per_task_metrics(te_meta, te_y, te_s),
        "episode": ep,
        "stoop_recall": ep["stoop"]["fail_recall"],
        "carry_success_fpr": ep["carry"]["success_ep_fpr"],
        "lead": ep["stoop"]["lead_time_s_mean"],
        "loco_recall": ep["loco"]["fail_recall"],
        "reach_recall": ep["reach"]["fail_recall"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/ufr0_unified_failure/raw")
    ap.add_argument("--out", default="results/ufr1_intent_innovation")
    ap.add_argument("--mlp_epochs", type=int, default=25)
    ap.add_argument("--fm_epochs", type=int, default=30)
    ap.add_argument("--nfe", type=int, default=10)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    eps = load_cells(Path(args.raw))
    assign = clip_split(eps)
    for e in eps:
        e["split"] = assign[f"{e['task']}/{e['clip']}"]
    tr_eps = [e for e in eps if e["split"] == "train"]
    va_eps = [e for e in eps if e["split"] == "val"]
    te_eps = [e for e in eps if e["split"] == "test"]

    c_tr_nom, dx_tr_nom, _ = stack_pairs(tr_eps, "train_nom")
    c_va, dx_va, va_meta = stack_pairs(va_eps, "eval")
    c_te, dx_te, te_meta = stack_pairs(te_eps, "eval")
    y_va = np.array([r["y_phys"] for r in va_meta], dtype=np.int32)
    y_te = np.array([r["y_phys"] for r in te_meta], dtype=np.int32)

    mu_c = c_tr_nom.mean(0)
    sd_c = np.where(c_tr_nom.std(0) < 1e-6, 1.0, c_tr_nom.std(0))
    mu_x = dx_tr_nom.mean(0)
    sd_x = np.where(dx_tr_nom.std(0) < 1e-6, 1.0, dx_tr_nom.std(0))

    def norm_c(c, use_z):
        cn = (c - mu_c) / sd_c
        return cn if use_z else cn[:, :37]

    def norm_x(x):
        return (x - mu_x) / sd_x

    c_tr_z = norm_c(c_tr_nom, True)
    c_tr_nz = norm_c(c_tr_nom, False)
    y_tr = norm_x(dx_tr_nom)
    c_va_z, c_va_nz = norm_c(c_va, True), norm_c(c_va, False)
    c_te_z, c_te_nz = norm_c(c_te, True), norm_c(c_te, False)
    y_va_n, y_te_n = norm_x(dx_va), norm_x(dx_te)

    print(f"[ufr1] train_nom={len(c_tr_z)} val={len(c_va_z)} test={len(c_te_z)} device={device}", flush=True)

    mlp_z = train_mlp(DynMLP(53), c_tr_z, y_tr, args.mlp_epochs, device)
    mlp_nz = train_mlp(DynMLP(37), c_tr_nz, y_tr, args.mlp_epochs, device)
    fm_z = train_fm(TinyCondFM(N_PHYS, 53), c_tr_z, y_tr, args.fm_epochs, device)
    fm_nz = train_fm(TinyCondFM(N_PHYS, 37), c_tr_nz, y_tr, args.fm_epochs, device)

    counts = {
        "mlp_with_z": n_params(mlp_z),
        "mlp_without_z": n_params(mlp_nz),
        "fm_with_z": n_params(fm_z),
        "fm_without_z": n_params(fm_nz),
    }
    print("[ufr1] params", counts, flush=True)

    torch.save(
        {
            "mlp_z": mlp_z.state_dict(),
            "mlp_nz": mlp_nz.state_dict(),
            "fm_z": fm_z.state_dict(),
            "fm_nz": fm_nz.state_dict(),
            "mu_c": mu_c,
            "sd_c": sd_c,
            "mu_x": mu_x,
            "sd_x": sd_x,
            "counts": counts,
        },
        out / "checkpoints" / "ufr1.pt",
    )

    mlp_z_va, _ = mlp_resid(mlp_z, c_va_z, y_va_n, device)
    mlp_z_te, _ = mlp_resid(mlp_z, c_te_z, y_te_n, device)
    mlp_nz_va, _ = mlp_resid(mlp_nz, c_va_nz, y_va_n, device)
    mlp_nz_te, _ = mlp_resid(mlp_nz, c_te_nz, y_te_n, device)

    print("[ufr1] FM sampling K=16", flush=True)
    samp_va = fm_sample(fm_z, c_va_z, 16, args.nfe, device)
    samp_te = fm_sample(fm_z, c_te_z, 16, args.nfe, device)
    samp_va_nz = fm_sample(fm_nz, c_va_nz, 16, args.nfe, device)
    samp_te_nz = fm_sample(fm_nz, c_te_nz, 16, args.nfe, device)

    sc_va = fm_scores(samp_va, y_va_n)
    sc_te = fm_scores(samp_te, y_te_n)
    sc_va_nz = fm_scores(samp_va_nz, y_va_n)
    sc_te_nz = fm_scores(samp_te_nz, y_te_n)

    def k_slice(samp, y, k):
        return fm_scores(samp[:, :k], y)

    methods = {}
    # baselines on same frames
    re_te = np.array([r["re"] for r in te_meta])
    rs_te = np.array([r["rs"] for r in te_meta])
    re_va = np.array([r["re"] for r in va_meta])
    rs_va = np.array([r["rs"] for r in va_meta])
    methods["RE"] = method_pack(va_meta, re_va, te_meta, re_te, y_te)
    methods["RS"] = method_pack(va_meta, rs_va, te_meta, rs_te, y_te)
    methods["max_RE_RS"] = method_pack(
        va_meta, np.maximum(re_va, rs_va), te_meta, np.maximum(re_te, rs_te), y_te
    )
    methods["MLP_innov_with_z"] = method_pack(va_meta, mlp_z_va, te_meta, mlp_z_te, y_te)
    methods["MLP_innov_without_z"] = method_pack(va_meta, mlp_nz_va, te_meta, mlp_nz_te, y_te)

    for k in (1, 8, 16):
        va_k = k_slice(samp_va, y_va_n, k)
        te_k = k_slice(samp_te, y_te_n, k)
        methods[f"FM_min_K{k}_with_z"] = method_pack(va_meta, va_k["min"], te_meta, te_k["min"], y_te)
        methods[f"FM_mean_K{k}_with_z"] = method_pack(va_meta, va_k["mean"], te_meta, te_k["mean"], y_te)
        if k == 16:
            methods["FM_var_K16_with_z"] = method_pack(va_meta, va_k["var"], te_meta, te_k["var"], y_te)
            methods["FM_softmin_K16_with_z"] = method_pack(
                va_meta, va_k["softmin"], te_meta, te_k["softmin"], y_te
            )

    methods["FM_min_K16_without_z"] = method_pack(
        va_meta, sc_va_nz["min"], te_meta, sc_te_nz["min"], y_te
    )

    # handwritten gate RE/RS >= 1 for reference
    methods["RS_gate1"] = {
        "thresh": 1.0,
        "frame": per_task_metrics(te_meta, y_te, rs_te),
        "episode": ep_trigger(te_meta, rs_te, 1.0),
    }
    methods["RS_gate1"]["stoop_recall"] = methods["RS_gate1"]["episode"]["stoop"]["fail_recall"]
    methods["RS_gate1"]["carry_success_fpr"] = methods["RS_gate1"]["episode"]["carry"]["success_ep_fpr"]
    methods["RS_gate1"]["lead"] = methods["RS_gate1"]["episode"]["stoop"]["lead_time_s_mean"]
    methods["max_gate1"] = {
        "thresh": 1.0,
        "frame": per_task_metrics(te_meta, y_te, np.maximum(re_te, rs_te)),
        "episode": ep_trigger(te_meta, np.maximum(re_te, rs_te), 1.0),
    }
    methods["max_gate1"]["stoop_recall"] = methods["max_gate1"]["episode"]["stoop"]["fail_recall"]
    methods["max_gate1"]["carry_success_fpr"] = methods["max_gate1"]["episode"]["carry"]["success_ep_fpr"]
    methods["max_gate1"]["lead"] = methods["max_gate1"]["episode"]["stoop"]["lead_time_s_mean"]

    # pick best innovation by Carry FP drop while Stoop recall >= 0.8
    cand_names = [
        "MLP_innov_with_z",
        "MLP_innov_without_z",
        "FM_min_K1_with_z",
        "FM_min_K8_with_z",
        "FM_min_K16_with_z",
        "FM_mean_K16_with_z",
        "FM_softmin_K16_with_z",
        "FM_min_K16_without_z",
    ]
    rs_fp = methods["RS_gate1"]["carry_success_fpr"]
    rs_rec = methods["RS_gate1"]["stoop_recall"]

    def ok_go(name):
        m = methods[name]
        rec = m["stoop_recall"]
        fp = m["carry_success_fpr"]
        if rec != rec or fp != fp:
            return False
        if rec < 0.80:
            return False
        if not (rs_fp == rs_fp) or fp > rs_fp - 0.15:
            return False
        # no collapse: at least 3/4 tasks phys AUROC >= 0.55
        n_ok = 0
        for t in TASKS:
            a = m["frame"][t]["auroc"]
            if a == a and a >= 0.55:
                n_ok += 1
        return n_ok >= 3

    go_hits = [n for n in cand_names if ok_go(n)]
    mlp_au = methods["MLP_innov_with_z"]["frame"]["overall"]["auroc"]
    fm_au = methods["FM_min_K16_with_z"]["frame"]["overall"]["auroc"]
    fm_better = (
        methods["FM_min_K16_with_z"]["carry_success_fpr"]
        < methods["MLP_innov_with_z"]["carry_success_fpr"] - 0.05
        or (fm_au == fm_au and mlp_au == mlp_au and fm_au >= mlp_au + 0.03)
    )
    keep = "tiny_flow_matching" if fm_better else "deterministic_mlp"
    z_helps = (
        methods["MLP_innov_with_z"]["carry_success_fpr"]
        < methods["MLP_innov_without_z"]["carry_success_fpr"] - 0.05
        or methods["MLP_innov_with_z"]["frame"]["overall"]["auroc"]
        >= methods["MLP_innov_without_z"]["frame"]["overall"]["auroc"] + 0.02
    )

    verdict = {
        "decision": "GO" if go_hits else "HOLD",
        "go_methods": go_hits,
        "keep": keep if go_hits else ("HOLD_neither" if not go_hits else keep),
        "deterministic_vs_fm": keep,
        "fm_clearly_better": bool(fm_better),
        "z_nom_helps": bool(z_helps),
        "rs_stoop_recall": rs_rec,
        "rs_carry_fp": rs_fp,
        "mlp_stoop_recall": methods["MLP_innov_with_z"]["stoop_recall"],
        "mlp_carry_fp": methods["MLP_innov_with_z"]["carry_success_fpr"],
        "fm16_stoop_recall": methods["FM_min_K16_with_z"]["stoop_recall"],
        "fm16_carry_fp": methods["FM_min_K16_with_z"]["carry_success_fpr"],
        "note": (
            "GO requires Stoop recall>=0.80 and Carry-success FP at least 0.15 below RS>=1, "
            "unified val FPR=0.10 threshold, no 4-task collapse"
        ),
    }
    if not go_hits:
        verdict["reasons"] = [
            "no intent-conditioned innovation meets Stoop-recall + Carry-FP bar at a shared threshold"
        ]

    split_info = {
        "reuse": "UFR-0 dumps / T1.3-U λ=0/1/2",
        "n_episodes": len(eps),
        "n_train_ep": len(tr_eps),
        "n_val_ep": len(va_eps),
        "n_test_ep": len(te_eps),
        "n_train_nom_frames": int(len(c_tr_z)),
        "n_val_frames": int(len(c_va_z)),
        "n_test_frames": int(len(c_te_z)),
        "split": "fail-stratified clip split, identical rule to UFR-0",
        "horizon_s": H_PHYS * DT,
        "horizon_frames": H_PHYS,
        "target": "Δx_phys = phys(t+H)-phys(t); phys=[root_v(3), a_z, acc, w(3), roll, pitch, wrist_cf]",
        "no_q_dq": "full joint q/dq not in UFR-0 dump; not re-rolled",
        "train_on": "successful episodes + failed episodes until 1.0s before fail",
        "by_task_test_fail": {
            t: {
                "n_test_ep": sum(1 for e in te_eps if e["task"] == t),
                "n_test_fail": sum(1 for e in te_eps if e["task"] == t and e["fail"]),
            }
            for t in TASKS
        },
    }

    payload = {
        "phase": "UFR-1",
        "frozen": ["Parent_50000", "Mapper-B", "R-M3"],
        "no_task_id": True,
        "params": counts,
        "split": split_info,
        "methods": methods,
        "verdict": verdict,
    }
    (out / "summary_ufr1.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )

    def row(name):
        m = methods[name]
        return {
            "method": name,
            "phys_auroc": m["frame"]["overall"]["auroc"],
            "auprc": m["frame"]["overall"]["auprc"],
            "stoop_recall": m.get("stoop_recall", m["episode"]["stoop"]["fail_recall"]),
            "carry_success_fp": m.get("carry_success_fpr", m["episode"]["carry"]["success_ep_fpr"]),
            "stoop_lead_s": m.get("lead", m["episode"]["stoop"]["lead_time_s_mean"]),
        }

    compact = {
        "A_data": split_info,
        "B_params": counts,
        "C_main_table": [
            row(n)
            for n in (
                "RE",
                "RS",
                "max_RE_RS",
                "RS_gate1",
                "max_gate1",
                "MLP_innov_with_z",
                "MLP_innov_without_z",
                "FM_min_K1_with_z",
                "FM_min_K8_with_z",
                "FM_min_K16_with_z",
                "FM_mean_K16_with_z",
                "FM_min_K16_without_z",
            )
        ],
        "D_z_ablation": {
            "mlp_with_z": row("MLP_innov_with_z"),
            "mlp_without_z": row("MLP_innov_without_z"),
            "fm_with_z": row("FM_min_K16_with_z"),
            "fm_without_z": row("FM_min_K16_without_z"),
        },
        "E_fm_k": {f"K{k}": row(f"FM_min_K{k}_with_z") for k in (1, 8, 16)},
        "F_per_task": {
            n: {t: methods[n]["frame"][t] for t in TASKS}
            | {"episode": methods[n]["episode"]}
            for n in (
                "RE",
                "RS",
                "max_RE_RS",
                "MLP_innov_with_z",
                "FM_min_K16_with_z",
            )
        },
        "G_verdict": verdict,
    }
    (out / "report_compact.json").write_text(
        json.dumps(compact, indent=2, default=_json_default), encoding="utf-8"
    )
    print(json.dumps({"params": counts, "verdict": verdict, "table": compact["C_main_table"]}, indent=2, default=_json_default))
    print("[ufr1] wrote", out / "summary_ufr1.json")


if __name__ == "__main__":
    main()
