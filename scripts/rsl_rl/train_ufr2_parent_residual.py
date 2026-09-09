#!/usr/bin/env python3
"""UFR-2: Parent-relative execution residual + tiny MLP + MLP-MoE.

Frozen Parent / Mapper-B / R-M3. Offline. No task ID. No PPO.
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
H_FAIL = 25
H_PHYS = 8
TARGET_FPR = 0.10
TASKS = ("loco", "stoop", "reach", "carry")
PHYS_SLICE = slice(26, 37)
N_PHYS = 11
N_J = 29


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
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
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
    rec = np.concatenate([[0.0], tp / n_pos])
    prec = np.concatenate([[1.0], prec])
    return float(np.trapz(prec, rec))


def _threshold_at_fpr(neg, target=TARGET_FPR) -> float:
    s = np.sort(np.asarray(neg, dtype=np.float64))
    if len(s) == 0:
        return float("inf")
    k = min(max(int(math.floor((1.0 - target) * len(s))), 0), len(s) - 1)
    return float(s[k])


def clip_split(eps: list[dict]) -> dict[str, str]:
    assign = {}
    fail_clips = {f"{e['task']}/{e['clip']}" for e in eps if int(e.get("fail") or 0) == 1}

    def _chunk(items):
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


def rms(x: np.ndarray, axis=-1) -> np.ndarray:
    return np.sqrt((x * x).mean(axis=axis) + 1e-12)


def load_cells(raw: Path) -> list[dict]:
    eps = []
    for p in sorted(raw.glob("*/lam_*/plane/ufr2_steps.npz")):
        task = p.parts[-4]
        blob = np.load(p, allow_pickle=True)
        meta = json.loads(str(blob["meta"].item()))
        feat = blob["feat"].astype(np.float32)
        q = blob["q"].astype(np.float32)
        dq = blob["dq"].astype(np.float32)
        q_cmd = blob["q_cmd"].astype(np.float32)
        q_pre = blob["q_pre"].astype(np.float32)
        root_v = feat[:, 26:29]
        root_w = feat[:, 31:34]
        contact = feat[:, 36:37]
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
        for ei, clip in enumerate(clips):
            m = ep_idx == ei
            order = np.argsort(t_all[m])
            fe, qq, dd, qc, qp = feat[m][order], q[m][order], dq[m][order], q_cmd[m][order], q_pre[m][order]
            rv, rw, cf = root_v[m][order], root_w[m][order], contact[m][order]
            T = int(fe.shape[0])
            fs = int(fsteps[ei])
            use_t = T if fs < 0 else min(T, fs + 1)
            fe, qq, dd, qc, qp = fe[:use_t], qq[:use_t], dd[:use_t], qc[:use_t], qp[:use_t]
            rv, rw, cf = rv[:use_t], rw[:use_t], cf[:use_t]
            r_q = qq - qc
            r_dq = dd - (qc - qp) / DT
            r_root_v = rv
            r_root_w = rw
            r_contact = cf
            resid = np.concatenate([r_q, r_dq, r_root_v, r_root_w, r_contact], axis=1)
            track = np.concatenate([r_q, r_dq], axis=1)
            y_phys = np.zeros(use_t, dtype=np.int32)
            if fs >= 0:
                for k in range(use_t):
                    y_phys[k] = int(k < fs <= k + H_FAIL)
            valid = np.zeros(use_t, dtype=bool)
            valid[WARMUP:use_t] = True
            if fs >= 0:
                valid[fs:] = False
            success_nom = valid.copy()
            if fs >= 0:
                success_nom[:] = False
            elif int(srs[ei]) != 1:
                success_nom[:] = False
            eps.append(
                {
                    "task": task,
                    "lam": lam,
                    "clip": clip,
                    "seed": int(seeds[ei]),
                    "fail": int(fails[ei]),
                    "fail_step": fs,
                    "sr_task": int(srs[ei]),
                    "feat": fe,
                    "re": re[m][order][:use_t],
                    "rs": rs[m][order][:use_t],
                    "resid": resid.astype(np.float32),
                    "track": track.astype(np.float32),
                    "r_q": r_q.astype(np.float32),
                    "r_dq": r_dq.astype(np.float32),
                    "y_phys": y_phys,
                    "valid": valid,
                    "success_nom": success_nom,
                }
            )
    return eps


def stack(eps, key_mask="valid"):
    xs, ys, meta = [], [], []
    for e in eps:
        idx = np.where(e[key_mask])[0]
        if len(idx) == 0:
            continue
        xs.append(idx)
        for t in idx:
            meta.append(
                {
                    "task": e["task"],
                    "clip": e["clip"],
                    "seed": e["seed"],
                    "lam": e["lam"],
                    "sr_task": e["sr_task"],
                    "fail": e["fail"],
                    "fail_step": e["fail_step"],
                    "t": int(t),
                    "re": float(e["re"][t]),
                    "rs": float(e["rs"][t]),
                    "y_phys": int(e["y_phys"][t]),
                }
            )
        ys.append(e)
    # flatten via episode pointers
    feat, resid, track, rq = [], [], [], []
    re, rs, y = [], [], []
    ep_ref = []
    k = 0
    for e, idx in zip([e for e in eps if e[key_mask].any()], [np.where(e[key_mask])[0] for e in eps if e[key_mask].any()]):
        feat.append(e["feat"][idx])
        resid.append(e["resid"][idx])
        track.append(e["track"][idx])
        rq.append(e["r_q"][idx])
        re.append(e["re"][idx])
        rs.append(e["rs"][idx])
        y.append(e["y_phys"][idx])
        ep_ref.append(e)
        k += 1
    if not feat:
        raise SystemExit("empty stack")
    return {
        "feat": np.concatenate(feat, 0),
        "resid": np.concatenate(resid, 0),
        "track": np.concatenate(track, 0),
        "r_q": np.concatenate(rq, 0),
        "re": np.concatenate(re, 0),
        "rs": np.concatenate(rs, 0),
        "y": np.concatenate(y, 0).astype(np.int32),
        "meta": meta,
    }


def metrics_block(y, s):
    y = np.asarray(y, dtype=np.int32)
    s = np.asarray(s, dtype=np.float64)
    return {"n": int(len(y)), "pos": int(y.sum()), "auroc": _auroc(y, s), "auprc": _auprc(y, s)}


def per_task(meta, y, s):
    out = {"overall": metrics_block(y, s)}
    for t in TASKS:
        m = np.array([r["task"] == t for r in meta])
        out[t] = metrics_block(y[m], s[m])
    return out


def ep_trigger(meta, scores, thresh):
    by = {}
    for i, r in enumerate(meta):
        key = (r["task"], r["clip"], r["seed"], r["lam"])
        by.setdefault(
            key,
            {"task": r["task"], "sr": r["sr_task"], "fail": r["fail"], "fail_step": r["fail_step"], "hit": False, "first": None},
        )
        if scores[i] >= thresh:
            by[key]["hit"] = True
            if by[key]["first"] is None:
                by[key]["first"] = r["t"]

    def pack(task=None):
        items = [v for k, v in by.items() if task is None or k[0] == task]
        succ = [v for v in items if v["sr"] == 1]
        fail = [v for v in items if v["fail"] == 1]
        leads = [
            (v["fail_step"] - v["first"]) * DT
            for v in fail
            if v["hit"] and v["first"] is not None and v["fail_step"] >= 0
        ]
        rec = float(np.mean([1.0 if v["hit"] else 0.0 for v in fail])) if fail else float("nan")
        return {
            "n_success_ep": len(succ),
            "n_fail_ep": len(fail),
            "success_ep_fpr": float(np.mean([1.0 if v["hit"] else 0.0 for v in succ])) if succ else float("nan"),
            "fail_recall": rec,
            "lead_time_s_mean": float(np.mean(leads)) if leads else float("nan"),
        }

    out = {"overall": pack(None)}
    for t in TASKS:
        out[t] = pack(t)
    return out


def choose_thresh(meta, scores):
    neg = [scores[i] for i, r in enumerate(meta) if r["sr_task"] == 1 and r["y_phys"] == 0]
    if len(neg) < 20:
        neg = [scores[i] for i, r in enumerate(meta) if r["y_phys"] == 0]
    return _threshold_at_fpr(neg)


def pack_method(va_meta, va_s, te_meta, te_s, te_y):
    th = choose_thresh(va_meta, va_s)
    ep = ep_trigger(te_meta, te_s, th)
    return {
        "thresh": th,
        "frame": per_task(te_meta, te_y, te_s),
        "episode": ep,
        "stoop_recall": ep["stoop"]["fail_recall"],
        "carry_success_fpr": ep["carry"]["success_ep_fpr"],
        "lead": ep["stoop"]["lead_time_s_mean"],
    }


class DynMLP(nn.Module):
    def __init__(self, c_dim=53, x_dim=11, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, c):
        return self.net(c)


class RiskMLP(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TinyMoE(nn.Module):
    def __init__(self, d, n_exp=4, hidden=128):
        super().__init__()
        self.n_exp = n_exp
        self.router = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, n_exp))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d, hidden), nn.SiLU(),
                    nn.Linear(hidden, hidden), nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(n_exp)
            ]
        )

    def forward(self, x):
        logits = self.router(x)
        gate = torch.softmax(logits, dim=-1)
        outs = torch.cat([e(x) for e in self.experts], dim=-1)
        y = (gate * outs).sum(dim=-1)
        return y, gate, logits


def train_risk(model, x, y, epochs, device, extra_loss=None, lr=1e-3, batch=1024):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    pw = torch.tensor(neg / pos, device=device)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            if extra_loss is None:
                logit = model(xb)
                loss = nn.functional.binary_cross_entropy_with_logits(logit, yb, pos_weight=pw)
            else:
                logit, gate, _ = model(xb)
                loss = nn.functional.binary_cross_entropy_with_logits(logit, yb, pos_weight=pw)
                loss = loss + extra_loss(gate)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_logit(model, x, device, moe=False, batch=4096):
    ys, gs = [], []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i : i + batch]).to(device)
        if moe:
            y, g, _ = model(xb)
            ys.append(torch.sigmoid(y).cpu().numpy())
            gs.append(g.cpu().numpy())
        else:
            ys.append(torch.sigmoid(model(xb)).cpu().numpy())
    y = np.concatenate(ys, 0)
    g = np.concatenate(gs, 0) if gs else None
    return y, g


def bal_loss_fn(lam=0.01):
    def _f(gate):
        p = gate.mean(0)
        f = p
        return lam * gate.shape[1] * (f * p).sum()
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/ufr2_parent_residual/raw")
    ap.add_argument("--out", default="results/ufr2_parent_residual")
    ap.add_argument("--ufr1_ckpt", default="results/ufr1_intent_innovation/checkpoints/ufr1.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    eps = load_cells(Path(args.raw))
    if not eps:
        raise SystemExit(f"no ufr2 dumps in {args.raw}")
    assign = clip_split(eps)
    for e in eps:
        e["split"] = assign[f"{e['task']}/{e['clip']}"]
    tr = [e for e in eps if e["split"] == "train"]
    va = [e for e in eps if e["split"] == "val"]
    te = [e for e in eps if e["split"] == "test"]

    # success-only stats for residual normalization
    suc = np.concatenate([e["resid"][e["success_nom"]] for e in tr if e["success_nom"].any()], 0)
    mu_r = suc.mean(0)
    sd_r = np.where(suc.std(0) < 1e-6, 1.0, suc.std(0))
    suc_t = np.concatenate([e["track"][e["success_nom"]] for e in tr if e["success_nom"].any()], 0)
    mu_t = suc_t.mean(0)
    sd_t = np.where(suc_t.std(0) < 1e-6, 1.0, suc_t.std(0))
    suc_q = np.concatenate([e["r_q"][e["success_nom"]] for e in tr if e["success_nom"].any()], 0)
    sd_q = np.where(suc_q.std(0) < 1e-6, 1.0, suc_q.std(0))

    def z(a, mu, sd):
        return (a - mu) / sd

    Sva, Ste = stack(va), stack(te)
    Str = stack(tr)
    y_te, y_va = Ste["y"], Sva["y"]

    r_parent_va = rms(z(Sva["resid"], mu_r, sd_r))
    r_parent_te = rms(z(Ste["resid"], mu_r, sd_r))
    r_track_va = rms(z(Sva["track"], mu_t, sd_t))
    r_track_te = rms(z(Ste["track"], mu_t, sd_t))
    r_q_va = rms(z(Sva["r_q"], suc_q.mean(0), sd_q))
    r_q_te = rms(z(Ste["r_q"], suc_q.mean(0), sd_q))

    methods = {
        "RE": pack_method(Sva["meta"], Sva["re"], Ste["meta"], Ste["re"], y_te),
        "RS": pack_method(Sva["meta"], Sva["rs"], Ste["meta"], Ste["rs"], y_te),
        "max_RE_RS": pack_method(
            Sva["meta"], np.maximum(Sva["re"], Sva["rs"]),
            Ste["meta"], np.maximum(Ste["re"], Ste["rs"]), y_te,
        ),
        "R_parent": pack_method(Sva["meta"], r_parent_va, Ste["meta"], r_parent_te, y_te),
        "R_track": pack_method(Sva["meta"], r_track_va, Ste["meta"], r_track_te, y_te),
        "R_q": pack_method(Sva["meta"], r_q_va, Ste["meta"], r_q_te, y_te),
    }
    methods["RS_gate1"] = {
        "thresh": 1.0,
        "frame": per_task(Ste["meta"], y_te, Ste["rs"]),
        "episode": ep_trigger(Ste["meta"], Ste["rs"], 1.0),
    }
    methods["RS_gate1"]["stoop_recall"] = methods["RS_gate1"]["episode"]["stoop"]["fail_recall"]
    methods["RS_gate1"]["carry_success_fpr"] = methods["RS_gate1"]["episode"]["carry"]["success_ep_fpr"]
    methods["RS_gate1"]["lead"] = methods["RS_gate1"]["episode"]["stoop"]["lead_time_s_mean"]
    methods["max_gate1"] = {
        "thresh": 1.0,
        "frame": per_task(Ste["meta"], y_te, np.maximum(Ste["re"], Ste["rs"])),
        "episode": ep_trigger(Ste["meta"], np.maximum(Ste["re"], Ste["rs"]), 1.0),
    }
    methods["max_gate1"]["stoop_recall"] = methods["max_gate1"]["episode"]["stoop"]["fail_recall"]
    methods["max_gate1"]["carry_success_fpr"] = methods["max_gate1"]["episode"]["carry"]["success_ep_fpr"]
    methods["max_gate1"]["lead"] = methods["max_gate1"]["episode"]["stoop"]["lead_time_s_mean"]

    # UFR-1 MLP innovation
    ckpt_p = Path(args.ufr1_ckpt)
    if ckpt_p.is_file():
        blob = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        mlp = DynMLP()
        mlp.load_state_dict(blob["mlp_z"])
        mlp.to(device).eval()
        mu_c = blob["mu_c"]
        sd_c = blob["sd_c"]
        mu_x = blob["mu_x"]
        sd_x = blob["sd_x"]

        def ufr1_score(feat):
            n = len(feat)
            sc = np.full(n, np.nan, dtype=np.float32)
            # need t+H phys; approximate with current-frame residual vs predicted 0-horizon
            # Use same 160ms delta as UFR-1 when possible via consecutive rows in episode.
            return sc

        # rebuild from episodes for correct Δx
        def score_eps(elist):
            scores = []
            meta = []
            y = []
            for e in elist:
                fe = e["feat"]
                T = len(fe)
                idx = np.where(e["valid"])[0]
                idx = idx[idx + H_PHYS < T]
                if len(idx) == 0:
                    continue
                c = (fe[idx] - mu_c) / sd_c
                dx = (fe[idx + H_PHYS][:, PHYS_SLICE] - fe[idx][:, PHYS_SLICE] - mu_x) / sd_x
                with torch.no_grad():
                    hat = mlp(torch.from_numpy(c.astype(np.float32)).to(device)).cpu().numpy()
                r = rms(dx - hat)
                scores.append(r)
                y.append(e["y_phys"][idx])
                for t in idx:
                    meta.append(
                        {
                            "task": e["task"], "clip": e["clip"], "seed": e["seed"], "lam": e["lam"],
                            "sr_task": e["sr_task"], "fail": e["fail"], "fail_step": e["fail_step"],
                            "t": int(t), "re": float(e["re"][t]), "rs": float(e["rs"][t]),
                            "y_phys": int(e["y_phys"][t]),
                        }
                    )
            return np.concatenate(scores), np.concatenate(y).astype(np.int32), meta

        s_va, yv, m_va = score_eps(va)
        s_te, yt, m_te = score_eps(te)
        methods["UFR1_MLP_innov"] = pack_method(m_va, s_va, m_te, s_te, yt)
        print("[ufr2] UFR-1 MLP scores ready", flush=True)
    else:
        print("[ufr2] UFR-1 ckpt missing, skip", flush=True)

    # learned residual -> risk
    # input: z-scored parent residual + e/M + root + z_nom
    def make_x(S):
        r = z(S["resid"], mu_r, sd_r).astype(np.float32)
        fe = S["feat"]
        ctx = np.concatenate([r, fe[:, :8], fe[:, 26:37], fe[:, 37:53]], axis=1).astype(np.float32)
        return ctx

    x_tr, x_va, x_te = make_x(Str), make_x(Sva), make_x(Ste)
    mu_x = x_tr.mean(0)
    sd_x = np.where(x_tr.std(0) < 1e-6, 1.0, x_tr.std(0))
    x_tr = ((x_tr - mu_x) / sd_x).astype(np.float32)
    x_va = ((x_va - mu_x) / sd_x).astype(np.float32)
    x_te = ((x_te - mu_x) / sd_x).astype(np.float32)
    d_in = int(x_tr.shape[1])
    print(f"[ufr2] in_dim={d_in} train={len(x_tr)} device={device}", flush=True)

    mlp_r = train_risk(RiskMLP(d_in), x_tr, Str["y"], args.epochs, device)
    moe = train_risk(TinyMoE(d_in), x_tr, Str["y"], args.epochs, device, extra_loss=bal_loss_fn(0.01))
    counts = {"residual_mlp": n_params(mlp_r), "moe": n_params(moe), "in_dim": d_in}
    print("[ufr2] params", counts, flush=True)

    pv_mlp, _ = predict_logit(mlp_r, x_va, device)
    pt_mlp, _ = predict_logit(mlp_r, x_te, device)
    pv_moe, gv = predict_logit(moe, x_va, device, moe=True)
    pt_moe, gt = predict_logit(moe, x_te, device, moe=True)
    methods["MLP_residual"] = pack_method(Sva["meta"], pv_mlp, Ste["meta"], pt_mlp, y_te)
    methods["MoE"] = pack_method(Sva["meta"], pv_moe, Ste["meta"], pt_moe, y_te)

    # expert analysis (test)
    arg = gt.argmax(1)
    ent = -(gt * np.log(np.clip(gt, 1e-8, 1))).sum(1)
    util = gt.mean(0)
    task_mat = {}
    for t in TASKS:
        m = np.array([r["task"] == t for r in Ste["meta"]])
        task_mat[t] = {
            "mean_gate": gt[m].mean(0).tolist() if m.any() else None,
            "argmax_frac": np.bincount(arg[m], minlength=4).astype(float).tolist() if m.any() else None,
        }
        if m.any():
            task_mat[t]["argmax_frac"] = (np.bincount(arg[m], minlength=4) / m.sum()).tolist()
    fail_m = np.array([r["fail"] == 1 for r in Ste["meta"]])
    suc_m = np.array([r["sr_task"] == 1 for r in Ste["meta"]])
    t_arr = np.array([r["t"] for r in Ste["meta"]], dtype=np.float32)
    phase = np.where(t_arr < 133, "early", np.where(t_arr < 266, "mid", "late"))
    phase_mat = {}
    for ph in ("early", "mid", "late"):
        m = phase == ph
        phase_mat[ph] = {
            "mean_gate": gt[m].mean(0).tolist() if m.any() else None,
            "n": int(m.sum()),
        }
    expert = {
        "utilization": util.tolist(),
        "mean_entropy": float(ent.mean()),
        "max_util": float(util.max()),
        "task_x_expert": task_mat,
        "fail_mean_gate": gt[fail_m].mean(0).tolist() if fail_m.any() else None,
        "success_mean_gate": gt[suc_m].mean(0).tolist() if suc_m.any() else None,
        "phase_x_expert": phase_mat,
        "looks_like_task_router": False,
    }
    # task router if each task has one dominant unique expert >0.8
    dom = {}
    for t in TASKS:
        g = np.array(task_mat[t]["mean_gate"] or [0, 0, 0, 0])
        dom[t] = int(g.argmax()) if g.sum() else -1
        expert[f"{t}_dom_expert"] = int(dom[t])
        expert[f"{t}_dom_mass"] = float(g.max()) if g.size else float("nan")
    unique_dom = set(dom.values())
    high = all(expert[f"{t}_dom_mass"] >= 0.80 for t in TASKS)
    expert["looks_like_task_router"] = bool(high and len(unique_dom) >= 3)

    torch.save(
        {"mlp": mlp_r.state_dict(), "moe": moe.state_dict(), "mu_r": mu_r, "sd_r": sd_r, "mu_x": mu_x, "sd_x": sd_x, "counts": counts},
        out / "checkpoints" / "ufr2.pt",
    )

    rs_fp = methods["RS_gate1"]["carry_success_fpr"]
    def go_ok(name):
        m = methods[name]
        rec, fp = m["stoop_recall"], m["carry_success_fpr"]
        if rec != rec or fp != fp or rec < 0.80:
            return False
        if not (rs_fp == rs_fp) or fp > rs_fp - 0.15:
            return False
        n_ok = sum(1 for t in TASKS if (a := m["frame"][t]["auroc"]) == a and a >= 0.55)
        return n_ok >= 3

    cands = ["R_parent", "R_track", "R_q", "MLP_residual", "MoE"]
    hits = [n for n in cands if go_ok(n)]
    moe_keep = (
        go_ok("MoE")
        and methods["MoE"]["frame"]["overall"]["auroc"] >= methods["MLP_residual"]["frame"]["overall"]["auroc"] + 0.02
        and methods["MoE"]["carry_success_fpr"] < methods["MLP_residual"]["carry_success_fpr"] - 0.05
        and not expert["looks_like_task_router"]
    )
    mlp_vs_simple = methods["MLP_residual"]["carry_success_fpr"] < methods["R_parent"]["carry_success_fpr"] - 0.05
    keep_learned = "MoE" if moe_keep else ("MLP_residual" if go_ok("MLP_residual") and mlp_vs_simple else ("simple_residual" if hits else "none"))
    verdict = {
        "decision": "GO" if hits else "HOLD",
        "go_methods": hits,
        "keep": keep_learned if hits else "HOLD_stop_stacking_failure_models",
        "moe_keep": bool(moe_keep),
        "moe_is_task_router": expert["looks_like_task_router"],
        "mlp_vs_simple_residual": bool(mlp_vs_simple),
        "rs_stoop_recall": methods["RS_gate1"]["stoop_recall"],
        "rs_carry_fp": rs_fp,
        "r_parent_stoop": methods["R_parent"]["stoop_recall"],
        "r_parent_carry_fp": methods["R_parent"]["carry_success_fpr"],
        "mlp_stoop": methods["MLP_residual"]["stoop_recall"],
        "mlp_carry_fp": methods["MLP_residual"]["carry_success_fpr"],
        "moe_stoop": methods["MoE"]["stoop_recall"],
        "moe_carry_fp": methods["MoE"]["carry_success_fpr"],
    }
    if not hits:
        verdict["reasons"] = ["Parent-relative residual cannot separate Stoop dangerous vs Carry normal at a shared threshold"]

    def row(name, params=0):
        m = methods[name]
        return {
            "method": name,
            "params": params,
            "phys_auroc": m["frame"]["overall"]["auroc"],
            "auprc": m["frame"]["overall"]["auprc"],
            "stoop_recall": m.get("stoop_recall", m["episode"]["stoop"]["fail_recall"]),
            "carry_success_fp": m.get("carry_success_fpr", m["episode"]["carry"]["success_ep_fpr"]),
            "stoop_lead_s": m.get("lead", m["episode"]["stoop"]["lead_time_s_mean"]),
        }

    split_info = {
        "n_ep": len(eps),
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
        "n_test_frames": len(Ste["y"]),
        "horizon_label_s": H_FAIL * DT,
        "residual": "r_q=q-q_cmd, r_dq=dq-(q_cmd-q_pre)/dt, r_root=actual v/w (no parent root cmd), r_contact=wrist_cf; R_parent=RMS z-score on success-train",
        "q_cmd": "Isaac joint_pos_target after Parent decode (causal, current step only)",
        "no_future_action_seq": "cannot rollout Parent 100-300ms without future proprio; not dumped",
        "norm": "success-only train trajectories, shared across 4 tasks",
        "by_task_test_fail": {
            t: {
                "n_test": sum(1 for e in te if e["task"] == t),
                "n_fail": sum(1 for e in te if e["task"] == t and e["fail"]),
            }
            for t in TASKS
        },
    }
    payload = {
        "phase": "UFR-2",
        "params": counts,
        "split": split_info,
        "methods": methods,
        "expert": expert,
        "verdict": verdict,
    }
    (out / "summary_ufr2.json").write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    table_names = ["RE", "RS", "max_RE_RS", "RS_gate1", "R_parent", "R_track", "R_q", "MLP_residual", "MoE"]
    if "UFR1_MLP_innov" in methods:
        table_names.insert(4, "UFR1_MLP_innov")
    compact = {
        "A_dump": split_info,
        "B_residual": split_info["residual"],
        "C_table": [row(n, counts.get("residual_mlp" if n == "MLP_residual" else ("moe" if n == "MoE" else 0), 0)) for n in table_names],
        "D_per_task": {n: {"frame": methods[n]["frame"], "episode": methods[n]["episode"]} for n in ["RE", "RS", "R_parent", "MLP_residual", "MoE"] + (["UFR1_MLP_innov"] if "UFR1_MLP_innov" in methods else [])},
        "E_simple_vs_learned": {"R_parent": row("R_parent"), "MLP": row("MLP_residual", counts["residual_mlp"]), "MoE": row("MoE", counts["moe"])},
        "expert": expert,
        "F_verdict": verdict,
        "params": counts,
    }
    (out / "report_compact.json").write_text(json.dumps(compact, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"params": counts, "verdict": verdict, "table": compact["C_table"], "expert_util": expert["utilization"], "entropy": expert["mean_entropy"], "task_router": expert["looks_like_task_router"]}, indent=2, default=_json_default))
    print("[ufr2] wrote", out / "summary_ufr2.json")


if __name__ == "__main__":
    main()
