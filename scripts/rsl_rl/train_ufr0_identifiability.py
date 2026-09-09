#!/usr/bin/env python3
"""UFR-0 offline identifiability. Frozen Parent / Mapper-B / R-M3. No PPO.

Shared F_psi(h_t) vs handwritten RE/RS. Episode/clip split only.
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
H05 = 25
H10 = 50
HIST = 16  # 320 ms
DEV_THR = 0.05
TARGET_FPR = 0.10
TASKS = ("loco", "stoop", "reach", "carry")


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
    sum_pos = float(ranks[y_s == 1].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


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


def _ece(y, p, n_bins: int = 10) -> float:
    y = np.asarray(y).astype(np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        msk = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if not np.any(msk):
            continue
        ece += (msk.mean()) * abs(float(y[msk].mean()) - float(p[msk].mean()))
    return float(ece)


def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 8:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra.astype(np.float64)
    rb = rb.astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if den <= 1e-12:
        return float("nan")
    return float((ra * rb).sum() / den)


def _metrics(y, s, p=None) -> dict:
    y = np.asarray(y).astype(np.int32)
    s = np.asarray(s, dtype=np.float64)
    out = {
        "n": int(len(y)),
        "pos": int(y.sum()),
        "pos_rate": float(y.mean()) if len(y) else float("nan"),
        "auroc": _auroc(y, s),
        "auprc": _auprc(y, s),
    }
    if p is not None:
        out["ece"] = _ece(y, p)
        out["brier"] = float(np.mean((np.asarray(p) - y) ** 2)) if len(y) else float("nan")
    else:
        out["ece"] = float("nan")
        out["brier"] = float("nan")
    return out


def _threshold_at_fpr(y_neg_scores, target=TARGET_FPR) -> float:
    s = np.sort(np.asarray(y_neg_scores, dtype=np.float64))
    if len(s) == 0:
        return float("inf")
    k = int(math.floor((1.0 - target) * len(s)))
    k = min(max(k, 0), len(s) - 1)
    return float(s[k])


def load_cells(raw_root: Path) -> list[dict]:
    eps = []
    for npz_path in sorted(raw_root.glob("*/lam_*/plane/ufr_steps.npz")):
        parts = npz_path.parts
        task = parts[-4]
        lam_name = parts[-3]
        blob = np.load(npz_path, allow_pickle=True)
        meta = json.loads(str(blob["meta"].item()))
        feat = blob["feat"].astype(np.float32)
        re = blob["re"].astype(np.float32)
        rs = blob["rs"].astype(np.float32)
        e_vis = blob["e_vis"].astype(np.float32)
        t = blob["t"].astype(np.int32)
        ep_idx = blob["ep_idx"].astype(np.int32)
        clips = [str(x) for x in blob["clip"].tolist()]
        seeds = blob["seed"].astype(np.int32)
        fails = blob["fail"].astype(np.int32)
        fsteps = blob["fail_step"].astype(np.int32)
        srs = blob["sr_task"].astype(np.int32)
        sr5 = blob["sr_5cm"].astype(np.float32)
        lam = float(meta.get("lambda_id", -1))
        oor = bool(meta.get("HUMAN_INTENT_OUT_OF_PLAUSIBLE_RANGE"))
        for ei, clip in enumerate(clips):
            m = ep_idx == ei
            fe = feat[m]
            tt = t[m]
            order = np.argsort(tt)
            fe, tt = fe[order], tt[order]
            ev = e_vis[m][order]
            T = int(fe.shape[0])
            fs = int(fsteps[ei])
            use_t = T if fs < 0 else min(T, fs + 1)
            fe, ev = fe[:use_t], ev[:use_t]
            future_max_05 = np.zeros(use_t, dtype=np.float32)
            future_max_10 = np.zeros(use_t, dtype=np.float32)
            future_mean_05 = np.zeros(use_t, dtype=np.float32)
            y_dev_05 = np.zeros(use_t, dtype=np.int32)
            y_dev_10 = np.zeros(use_t, dtype=np.int32)
            y_phys_05 = np.zeros(use_t, dtype=np.int32)
            y_phys_10 = np.zeros(use_t, dtype=np.int32)
            for k in range(use_t):
                a05, b05 = k + 1, min(use_t, k + 1 + H05)
                a10, b10 = k + 1, min(use_t, k + 1 + H10)
                if a05 < b05:
                    future_max_05[k] = float(ev[a05:b05].max())
                    future_mean_05[k] = float(ev[a05:b05].mean())
                if a10 < b10:
                    future_max_10[k] = float(ev[a10:b10].max())
                y_dev_05[k] = int(future_max_05[k] >= DEV_THR)
                y_dev_10[k] = int(future_max_10[k] >= DEV_THR)
                if fs >= 0:
                    y_phys_05[k] = int(k < fs <= k + H05)
                    y_phys_10[k] = int(k < fs <= k + H10)
            valid = np.zeros(use_t, dtype=bool)
            valid[WARMUP:] = True
            if fs >= 0:
                valid[fs:] = False
            if use_t > H05:
                valid[use_t - 1] = False
            eps.append(
                {
                    "task": task,
                    "lam": lam,
                    "lam_name": lam_name,
                    "oor": oor,
                    "clip": clip,
                    "seed": int(seeds[ei]),
                    "fail": int(fails[ei]),
                    "fail_step": fs,
                    "sr_task": int(srs[ei]),
                    "sr_5cm": float(sr5[ei]),
                    "feat": fe,
                    "re": re[m][order][:use_t],
                    "rs": rs[m][order][:use_t],
                    "e_vis": ev,
                    "future_max_05": future_max_05,
                    "future_mean_05": future_mean_05,
                    "y_dev_05": y_dev_05,
                    "y_dev_10": y_dev_10,
                    "y_phys_05": y_phys_05,
                    "y_phys_10": y_phys_10,
                    "valid": valid,
                }
            )
    return eps


def clip_split(eps: list[dict]) -> dict[str, str]:
    """Clip split, stratified so fail-containing clips appear in val/test.

    Lexical 60/20/20 put all failing clips in train (clips 01–05) and left
    test as success-only, which makes physical-risk metrics undefined.
    """
    assign = {}
    fail_clips = {
        f"{e['task']}/{e['clip']}" for e in eps if int(e.get("fail") or 0) == 1
    }

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


def stack_frames(eps, keys=("feat", "re", "rs")):
    idx = []
    for i, e in enumerate(eps):
        v = np.where(e["valid"])[0]
        for t in v:
            idx.append((i, int(t)))
    return idx


class MLPHead(nn.Module):
    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.dev = nn.Linear(hidden, 2)
        self.phys = nn.Linear(hidden, 2)

    def forward(self, x):
        h = self.net(x)
        return self.dev(h), self.phys(h)


class GRUHead(nn.Module):
    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.dev = nn.Linear(hidden, 2)
        self.phys = nn.Linear(hidden, 2)

    def forward(self, x):
        _, h = self.gru(x)
        h = h[-1]
        return self.dev(h), self.phys(h)


def make_xy(eps, idx, mu, sd, hist: int | None):
    x, yd, yp, fut = [], [], [], []
    for i, t in idx:
        e = eps[i]
        if hist is None:
            xt = e["feat"][t]
        else:
            a = max(0, t - hist + 1)
            sl = e["feat"][a : t + 1]
            if sl.shape[0] < hist:
                pad = np.repeat(sl[:1], hist - sl.shape[0], axis=0)
                sl = np.concatenate([pad, sl], axis=0)
            xt = sl
        x.append(xt)
        yd.append([e["y_dev_05"][t], e["y_dev_10"][t]])
        yp.append([e["y_phys_05"][t], e["y_phys_10"][t]])
        fut.append(e["future_max_05"][t])
    x = np.asarray(x, dtype=np.float32)
    if hist is None:
        x = (x - mu) / sd
    else:
        x = (x - mu.reshape(1, 1, -1)) / sd.reshape(1, 1, -1)
    return (
        torch.from_numpy(x),
        torch.from_numpy(np.asarray(yd, dtype=np.float32)),
        torch.from_numpy(np.asarray(yp, dtype=np.float32)),
        np.asarray(fut, dtype=np.float32),
    )


def pos_weight(y: torch.Tensor) -> torch.Tensor:
    pos = y.sum(0).clamp(min=1)
    neg = (1 - y).sum(0).clamp(min=1)
    return (neg / pos).cpu()


def train_model(model, x, yd, yp, epochs, device, lr=1e-3, batch=256):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    w_d = pos_weight(yd).to(device)
    w_p = pos_weight(yp).to(device)
    ds = TensorDataset(x, yd, yp)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    model.train()
    for _ in range(epochs):
        for xb, db, pb in dl:
            xb = xb.to(device)
            db = db.to(device)
            pb = pb.to(device)
            ld, lp = model(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(
                ld, db, pos_weight=w_d
            ) + nn.functional.binary_cross_entropy_with_logits(lp, pb, pos_weight=w_p)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, x, device):
    outs_d, outs_p = [], []
    for i in range(0, len(x), 1024):
        xb = x[i : i + 1024].to(device)
        d, p = model(xb)
        outs_d.append(torch.sigmoid(d).cpu().numpy())
        outs_p.append(torch.sigmoid(p).cpu().numpy())
    return np.concatenate(outs_d, 0), np.concatenate(outs_p, 0)


def frame_table(eps, idx):
    rows = []
    for i, t in idx:
        e = eps[i]
        rows.append(
            {
                "task": e["task"],
                "lam": e["lam"],
                "clip": e["clip"],
                "seed": e["seed"],
                "sr_task": e["sr_task"],
                "fail": e["fail"],
                "fail_step": e["fail_step"],
                "oor": e["oor"],
                "t": t,
                "re": float(e["re"][t]),
                "rs": float(e["rs"][t]),
                "max_rers": float(max(e["re"][t], e["rs"][t])),
                "sum_rers": float(0.5 * (e["re"][t] + e["rs"][t])),
                "y_dev_05": int(e["y_dev_05"][t]),
                "y_dev_10": int(e["y_dev_10"][t]),
                "y_phys_05": int(e["y_phys_05"][t]),
                "y_phys_10": int(e["y_phys_10"][t]),
                "future_max_05": float(e["future_max_05"][t]),
                "e_vis": float(e["e_vis"][t]),
                "sr_5cm": e["sr_5cm"],
            }
        )
    return rows


def eval_scores(rows, score_key, y_key, p_key=None, tasks=None) -> dict:
    if tasks is None:
        tasks = ("overall",) + TASKS
    out = {}
    for task in tasks:
        sub = rows if task == "overall" else [r for r in rows if r["task"] == task]
        y = np.array([r[y_key] for r in sub], dtype=np.int32)
        s = np.array([r[score_key] for r in sub], dtype=np.float64)
        p = np.array([r[p_key] for r in sub], dtype=np.float64) if p_key else None
        rec = _metrics(y, s, p)
        rec["spearman_future_e"] = _spearman(
            s, np.array([r["future_max_05"] for r in sub], dtype=np.float64)
        )
        out[task] = rec
    return out


def success_ep_fpr(eps, idx, scores, thresh) -> dict:
    """Episode-level FP: success episode ever exceeds thresh."""
    by_ep = {}
    for k, (i, t) in enumerate(idx):
        e = eps[i]
        key = (e["task"], e["clip"], e["seed"], e["lam"])
        by_ep.setdefault(key, {"task": e["task"], "sr": e["sr_task"], "hit": False, "fail": e["fail"],
                               "fail_step": e["fail_step"], "first": None})
        if scores[k] >= thresh:
            by_ep[key]["hit"] = True
            if by_ep[key]["first"] is None:
                by_ep[key]["first"] = t

    def pack(task=None):
        items = [v for k, v in by_ep.items() if task is None or k[0] == task]
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


def recall_at_fpr(rows, score_key, y_key, thresh) -> dict:
    out = {}
    for task in ("overall",) + TASKS:
        sub = rows if task == "overall" else [r for r in rows if r["task"] == task]
        y = np.array([r[y_key] for r in sub])
        s = np.array([r[score_key] for r in sub])
        pred = s >= thresh
        pos = y == 1
        neg = y == 0
        out[task] = {
            "recall": float(pred[pos].mean()) if pos.any() else float("nan"),
            "fpr": float(pred[neg].mean()) if neg.any() else float("nan"),
            "thresh": float(thresh),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/ufr0_unified_failure/raw")
    ap.add_argument("--out", default="results/ufr0_unified_failure")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    eps = load_cells(raw)
    if not eps:
        raise SystemExit(f"no ufr dumps under {raw}")
    assign = clip_split(eps)
    for e in eps:
        e["split"] = assign[f"{e['task']}/{e['clip']}"]

    train_eps = [e for e in eps if e["split"] == "train"]
    val_eps = [e for e in eps if e["split"] == "val"]
    test_eps = [e for e in eps if e["split"] == "test"]
    tr_idx = stack_frames(train_eps)
    va_idx = stack_frames(val_eps)
    te_idx = stack_frames(test_eps)

    feats = np.concatenate([e["feat"][e["valid"]] for e in train_eps], axis=0)
    mu = feats.mean(0)
    sd = feats.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)

    x_tr, yd_tr, yp_tr, _ = make_xy(train_eps, tr_idx, mu, sd, None)
    x_va, yd_va, yp_va, _ = make_xy(val_eps, va_idx, mu, sd, None)
    x_te, yd_te, yp_te, fut_te = make_xy(test_eps, te_idx, mu, sd, None)
    xh_tr, _, _, _ = make_xy(train_eps, tr_idx, mu, sd, HIST)
    xh_va, _, _, _ = make_xy(val_eps, va_idx, mu, sd, HIST)
    xh_te, _, _, _ = make_xy(test_eps, te_idx, mu, sd, HIST)

    d = int(x_tr.shape[-1])
    mlp = train_model(MLPHead(d), x_tr, yd_tr, yp_tr, args.epochs, device)
    gru = train_model(GRUHead(d), xh_tr, yd_tr, yp_tr, args.epochs, device)

    mlp_d_va, mlp_p_va = predict(mlp, x_va, device)
    gru_d_va, gru_p_va = predict(gru, xh_va, device)
    mlp_d_te, mlp_p_te = predict(mlp, x_te, device)
    gru_d_te, gru_p_te = predict(gru, xh_te, device)

    te_rows = frame_table(test_eps, te_idx)
    va_rows = frame_table(val_eps, va_idx)
    for rows, d_hat, p_hat in (
        (te_rows, mlp_d_te, mlp_p_te),
        (va_rows, mlp_d_va, mlp_p_va),
    ):
        pass
    for i, r in enumerate(te_rows):
        r["mlp_dev"] = float(mlp_d_te[i, 0])
        r["mlp_phys"] = float(mlp_p_te[i, 0])
        r["gru_dev"] = float(gru_d_te[i, 0])
        r["gru_phys"] = float(gru_p_te[i, 0])
        r["mlp_dev10"] = float(mlp_d_te[i, 1])
        r["mlp_phys10"] = float(mlp_p_te[i, 1])
        r["gru_dev10"] = float(gru_d_te[i, 1])
        r["gru_phys10"] = float(gru_p_te[i, 1])
    for i, r in enumerate(va_rows):
        r["mlp_dev"] = float(mlp_d_va[i, 0])
        r["mlp_phys"] = float(mlp_p_va[i, 0])
        r["gru_dev"] = float(gru_d_va[i, 0])
        r["gru_phys"] = float(gru_p_va[i, 0])

    methods = {
        "RE_only": "re",
        "RS_only": "rs",
        "max_RE_RS": "max_rers",
        "simple_RE_RS": "sum_rers",
        "current_frame_mlp_dev": "mlp_dev",
        "current_frame_mlp_phys": "mlp_phys",
        "history_gru_dev": "gru_dev",
        "history_gru_phys": "gru_phys",
    }

    # global threshold from VAL successful-episode frames (y=0 for the head)
    def val_thresh(score_key, y_key):
        neg = [r[score_key] for r in va_rows if r["sr_task"] == 1 and r[y_key] == 0]
        if len(neg) < 20:
            neg = [r[score_key] for r in va_rows if r[y_key] == 0]
        return _threshold_at_fpr(neg, TARGET_FPR)

    tables = {}
    op = {}
    for name, key in methods.items():
        y_dev = "y_dev_05"
        y_phys = "y_phys_05"
        # baselines scored against both labels; learned heads use matching score
        if name.endswith("_phys"):
            y_use = y_phys
            p_key = key
        elif name.endswith("_dev"):
            y_use = y_dev
            p_key = key
        else:
            y_use = None
            p_key = None
        block = {}
        if y_use is None:
            block["intent_deviation"] = eval_scores(te_rows, key, y_dev)
            block["physical_risk"] = eval_scores(te_rows, key, y_phys)
            th_dev = val_thresh(key, y_dev)
            th_phys = val_thresh(key, y_phys)
            block["intent_deviation"]["recall_at_fpr10"] = recall_at_fpr(te_rows, key, y_dev, th_dev)
            block["physical_risk"]["recall_at_fpr10"] = recall_at_fpr(te_rows, key, y_phys, th_phys)
            scores = np.array([r[key] for r in te_rows])
            block["intent_deviation"]["success_ep"] = success_ep_fpr(test_eps, te_idx, scores, th_dev)
            block["physical_risk"]["success_ep"] = success_ep_fpr(test_eps, te_idx, scores, th_phys)
            block["handwritten_gate"] = success_ep_fpr(test_eps, te_idx, scores, 1.0)
            op[name] = {"th_dev": th_dev, "th_phys": th_phys}
        else:
            lab = "physical_risk" if name.endswith("_phys") else "intent_deviation"
            yk = y_phys if lab == "physical_risk" else y_dev
            block[lab] = eval_scores(te_rows, key, yk, p_key=p_key)
            th = val_thresh(key, yk)
            block[lab]["recall_at_fpr10"] = recall_at_fpr(te_rows, key, yk, th)
            scores = np.array([r[key] for r in te_rows])
            block[lab]["success_ep"] = success_ep_fpr(test_eps, te_idx, scores, th)
            op[name] = {"th": th}
        tables[name] = block

    # Reach λ separation on success episodes (intent-fidelity while Task SR high)
    reach_succ = [r for r in te_rows if r["task"] == "reach" and r["sr_task"] == 1]
    reach_lam = {}
    if reach_succ:
        y_lam = np.array([int(r["lam"] > 0) for r in reach_succ])
        for name, key in (
            ("RE_only", "re"),
            ("RS_only", "rs"),
            ("max_RE_RS", "max_rers"),
            ("current_frame_mlp_dev", "mlp_dev"),
            ("history_gru_dev", "gru_dev"),
        ):
            s = np.array([r[key] for r in reach_succ])
            by_lam = {}
            for lam in (0.0, 1.0, 2.0):
                ss = [r[key] for r in reach_succ if abs(r["lam"] - lam) < 1e-6]
                fe = [r["future_max_05"] for r in reach_succ if abs(r["lam"] - lam) < 1e-6]
                by_lam[str(int(lam))] = {
                    "n": len(ss),
                    "score_mean": float(np.mean(ss)) if ss else float("nan"),
                    "future_e_mean": float(np.mean(fe)) if fe else float("nan"),
                }
            reach_lam[name] = {
                "auroc_lam_gt0": _auroc(y_lam, s),
                "spearman_future_e": _spearman(s, [r["future_max_05"] for r in reach_succ]),
                "by_lambda": by_lam,
            }

    # Stoop fail recall vs Carry success FP at handwritten RS>=1 and at unified thresh
    focus = {}
    for name in ("RS_only", "max_RE_RS", "simple_RE_RS", "current_frame_mlp_phys", "history_gru_phys"):
        key = methods[name]
        if name in ("current_frame_mlp_phys", "history_gru_phys"):
            th = op[name]["th"]
        else:
            th = 1.0
        scores = np.array([r[key] for r in te_rows])
        ep = success_ep_fpr(test_eps, te_idx, scores, th)
        focus[name] = {
            "thresh": th,
            "stoop_fail_recall": ep["stoop"]["fail_recall"],
            "carry_success_fpr": ep["carry"]["success_ep_fpr"],
            "loco_fail_recall": ep["loco"]["fail_recall"],
            "per_task": ep,
        }

    split_info = {
        "n_episodes": len(eps),
        "n_train_ep": len(train_eps),
        "n_val_ep": len(val_eps),
        "n_test_ep": len(test_eps),
        "n_train_frames": len(tr_idx),
        "n_val_frames": len(va_idx),
        "n_test_frames": len(te_idx),
        "split_rule": "by (task, clip); all λ/seeds of a clip stay together; no random frame split",
        "split_frac": "fail-stratified clip split: fail-clips and success-clips split separately; no random frames",
        "by_task": {},
        "by_split_task_lam": {},
    }
    for task in TASKS:
        clips = sorted({e["clip"] for e in eps if e["task"] == task})
        split_info["by_task"][task] = {
            "n_clips": len(clips),
            "train_clips": sorted({e["clip"] for e in train_eps if e["task"] == task}),
            "val_clips": sorted({e["clip"] for e in val_eps if e["task"] == task}),
            "test_clips": sorted({e["clip"] for e in test_eps if e["task"] == task}),
            "n_ep": sum(1 for e in eps if e["task"] == task),
            "n_fail": sum(1 for e in eps if e["task"] == task and e["fail"]),
            "n_test_ep": sum(1 for e in test_eps if e["task"] == task),
            "n_test_fail": sum(1 for e in test_eps if e["task"] == task and e["fail"]),
        }
    for e in eps:
        k = f"{e['split']}/{e['task']}/lam{int(e['lam']) if e['lam']>=0 else 'x'}"
        split_info["by_split_task_lam"].setdefault(k, 0)
        split_info["by_split_task_lam"][k] += 1

    # GO / HOLD
    gru_dev = tables["history_gru_dev"]["intent_deviation"]
    gru_phys = tables["history_gru_phys"]["physical_risk"]
    maxc = tables["max_RE_RS"]
    mlp_dev = tables["current_frame_mlp_dev"]["intent_deviation"]
    reasons = []
    go = True
    # pooled AUROC better than max(RE,RS)
    if gru_dev["overall"]["auroc"] <= (maxc["intent_deviation"]["overall"]["auroc"] + 0.02):
        if mlp_dev["overall"]["auroc"] <= (maxc["intent_deviation"]["overall"]["auroc"] + 0.02):
            go = False
            reasons.append("pooled intent AUROC not clearly above max(RE,RS)")
    # not single-task driven: at least 3/4 tasks AUROC>=0.55 for one unified head
    n_ok = 0
    for t in TASKS:
        a = gru_dev[t]["auroc"]
        if a == a and a >= 0.58:
            n_ok += 1
    if n_ok < 3:
        go = False
        reasons.append(f"intent AUROC>=0.58 on only {n_ok}/4 tasks")
    carry_fp_base = focus["RS_only"]["carry_success_fpr"]
    carry_fp_u = min(
        focus["history_gru_phys"]["carry_success_fpr"],
        focus["current_frame_mlp_phys"]["carry_success_fpr"],
    )
    if not (carry_fp_u == carry_fp_u) or not (
        carry_fp_base == carry_fp_base and carry_fp_u < carry_fp_base - 0.05
    ):
        go = False
        reasons.append("Carry success FP not clearly below RS-only")
    stoop_r = max(
        focus["history_gru_phys"]["stoop_fail_recall"],
        focus["current_frame_mlp_phys"]["stoop_fail_recall"],
    )
    if not (stoop_r == stoop_r) or stoop_r < 0.5:
        go = False
        reasons.append("Stoop fail recall < 0.5")
    reach_ok = False
    if reach_lam:
        reach_ok = (
            reach_lam["history_gru_dev"]["auroc_lam_gt0"] >= 0.58
            or reach_lam["current_frame_mlp_dev"]["auroc_lam_gt0"] >= 0.58
            or reach_lam["history_gru_dev"]["spearman_future_e"] >= 0.25
        )
    if not reach_ok:
        go = False
        reasons.append("Reach intent-fidelity not identifiable (λ or future-e)")
    hist_better = gru_dev["overall"]["auroc"] >= mlp_dev["overall"]["auroc"] + 0.01

    verdict = {
        "decision": "GO" if go else "HOLD",
        "reasons": reasons if not go else ["unified model meets cross-task identifiability bar"],
        "history_helps": bool(hist_better),
        "n_tasks_intent_auroc_ge_058": n_ok,
        "carry_fp_rs": carry_fp_base,
        "carry_fp_unified": carry_fp_u,
        "stoop_recall_unified": stoop_r,
        "reach_lam_auroc_gru": (reach_lam or {}).get("history_gru_dev", {}).get("auroc_lam_gt0"),
    }

    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"mlp": mlp.state_dict(), "gru": gru.state_dict(), "mu": mu, "sd": sd, "feat_dim": d},
        ckpt_dir / "ufr0_baseline.pt",
    )

    payload = {
        "phase": "UFR-0",
        "frozen": ["Parent model_50000", "Mapper-B", "R-M3", "T1.3-U operator"],
        "no_task_id": True,
        "no_terrain": True,
        "no_lambda_in_input": True,
        "horizon_s": {"dev_phys_primary": 0.5, "secondary": 1.0},
        "history_ms": HIST * DT * 1000,
        "split": split_info,
        "operating_points": op,
        "tables": tables,
        "focus": focus,
        "reach_intent_fidelity": reach_lam,
        "verdict": verdict,
        "notes": {
            "baselines_shared_rule": "same threshold / score for all four tasks",
            "handwritten_gate": "RE>=1 / RS>=1 / max>=1 as in RecoveryRiskGate",
            "labels": {
                "intent_deviation": f"max future {H05*DT:.2f}s masked e_vis >= {DEV_THR}",
                "physical_risk": f"official fail in next {H05*DT:.2f}s",
            },
            "loco_lam2": "included if dumped; tagged oor in meta",
        },
    }
    (out / "summary_ufr0.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    compact = {
        "A_split": split_info,
        "B_overall": {
            name: {
                k: tables[name].get(k, {}).get("overall")
                for k in ("intent_deviation", "physical_risk")
                if k in tables[name]
            }
            for name in tables
        },
        "C_per_task": {
            name: {
                k: {t: tables[name][k][t] for t in TASKS if t in tables[name][k]}
                for k in ("intent_deviation", "physical_risk")
                if k in tables[name]
            }
            for name in tables
        },
        "D_focus": focus,
        "D_reach": reach_lam,
        "E_hist_vs_frame": {
            "mlp_dev_auroc": mlp_dev["overall"]["auroc"],
            "gru_dev_auroc": gru_dev["overall"]["auroc"],
            "mlp_phys_auroc": tables["current_frame_mlp_phys"]["physical_risk"]["overall"]["auroc"],
            "gru_phys_auroc": gru_phys["overall"]["auroc"],
            "history_helps": verdict["history_helps"],
        },
        "F_verdict": verdict,
    }
    (out / "report_compact.json").write_text(
        json.dumps(compact, indent=2, default=_json_default), encoding="utf-8"
    )
    print(json.dumps({"verdict": verdict, "split": {
        "n_ep": split_info["n_episodes"],
        "n_test_ep": split_info["n_test_ep"],
        "n_test_frames": split_info["n_test_frames"],
    }}, indent=2, default=_json_default))
    print("[ufr0] wrote", out / "summary_ufr0.json")


if __name__ == "__main__":
    main()
