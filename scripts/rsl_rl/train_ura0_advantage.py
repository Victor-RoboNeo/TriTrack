#!/usr/bin/env python3
"""URA-0: unified recovery-advantage identifiability. Offline. No PPO. No closed-loop."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TASKS = ("loco", "stoop", "reach", "carry")
EPS = 0.0025
# feat: e9(9)+edot(9)+M(3)+root(11)+z(16)+d(16)+th(1) = 65
SLICE_M = slice(18, 21)
SLICE_Z = slice(32, 48)


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


def clip_split(tasks, clips, useful) -> dict[str, str]:
    """Clip-level split. useful = clip has any I_rec < -eps in the pool."""
    assign = {}
    keys = [f"{t}/{c}" for t, c in zip(tasks, clips)]
    by_task: dict[str, list[str]] = {t: [] for t in TASKS}
    seen = set()
    useful_set = set()
    for t, c, u in zip(tasks, clips, useful):
        k = f"{t}/{c}"
        if u:
            useful_set.add(k)
        if k in seen:
            continue
        seen.add(k)
        by_task[t].append(c)

    def _chunk(items):
        items = sorted(items)
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
        clips_t = sorted(set(by_task[task]))
        pos = sorted(c for c in clips_t if f"{task}/{c}" in useful_set)
        neg = sorted(c for c in clips_t if f"{task}/{c}" not in useful_set)
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


def drop_intent(x: np.ndarray) -> np.ndarray:
    keep = [i for i in range(x.shape[1]) if not (SLICE_M.start <= i < SLICE_M.stop or SLICE_Z.start <= i < SLICE_Z.stop)]
    return x[:, keep]


def regret(j_p, j_r, intervene) -> np.ndarray:
    j_ch = np.where(intervene, j_r, j_p)
    return j_ch - np.minimum(j_p, j_r)


def pack_dec(intervene, j_p, j_r, y, tasks, name, thresh, params=0):
    r = regret(j_p, j_r, intervene)
    y = np.asarray(y).astype(np.int32)
    pred = np.asarray(intervene).astype(np.int32)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n_int = int(pred.sum())
    out = {
        "method": name,
        "params": params,
        "thresh": thresh,
        "mean_regret": float(r.mean()),
        "median_regret": float(np.median(r)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "p_intervene": float(pred.mean()),
        "n": int(len(y)),
        "n_useful": int(y.sum()),
        "mean_improvement": float(np.where(pred == 1, j_p - j_r, 0.0).mean()),
        "per_task": {},
    }
    for t in TASKS:
        m = tasks == t
        if not m.any():
            continue
        rt = r[m]
        yt, pt = y[m], pred[m]
        tp_t = int(((pt == 1) & (yt == 1)).sum())
        fp_t = int(((pt == 1) & (yt == 0)).sum())
        fn_t = int(((pt == 0) & (yt == 1)).sum())
        out["per_task"][t] = {
            "n": int(m.sum()),
            "mean_regret": float(rt.mean()),
            "median_regret": float(np.median(rt)),
            "precision": float(tp_t / max(tp_t + fp_t, 1)),
            "recall": float(tp_t / max(tp_t + fn_t, 1)),
            "p_intervene": float(pt.mean()),
            "n_useful": int(yt.sum()),
            "carry_nobeh_inter": None,
            "stoop_useful_recall": None,
        }
    return out


def pick_score_thresh(score, j_p, j_r, higher_means_intervene: bool):
    """Minimize mean regret on this split. intervene if score ? thresh."""
    s = np.asarray(score, dtype=np.float64)
    grid = np.unique(np.quantile(s[np.isfinite(s)], np.linspace(0.02, 0.98, 49)))
    best_t, best_r = 0.0, 1e9
    for t in grid:
        if higher_means_intervene:
            inter = s >= t
        else:
            inter = s < t
        mr = float(regret(j_p, j_r, inter).mean())
        if mr < best_r:
            best_r, best_t = mr, float(t)
    return best_t


class LinearProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Linear(d, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AdvMLP(nn.Module):
    def __init__(self, d, hidden=(256, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden[0]), nn.SiLU(),
            nn.Linear(hidden[0], hidden[1]), nn.SiLU(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_reg(model, x, y, epochs, device, lr=1e-3, batch=256):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=min(batch, max(len(x), 1)), shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = nn.functional.huber_loss(pred, yb, delta=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, x, device, batch=4096):
    ys = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i : i + batch]).to(device)
        ys.append(model(xb).cpu().numpy())
    return np.concatenate(ys, 0) if ys else np.zeros(0, dtype=np.float32)


def load_raw(raw: Path) -> dict:
    feats, hists, rest = [], [], []
    for p in sorted(raw.glob("*/lam_*/plane/clones.npz")):
        blob = np.load(p, allow_pickle=True)
        n = int(blob["feat"].shape[0])
        task = np.asarray(blob["task"]).astype(str)
        clip = np.asarray(blob["clip"]).astype(str)
        feats.append(blob["feat"].astype(np.float32))
        hists.append(blob["hist"].astype(np.float32))
        rest.append(
            dict(
                j_p=blob["j_p"].astype(np.float32),
                j_r=blob["j_r"].astype(np.float32),
                i_rec=blob["i_rec"].astype(np.float32),
                re=blob["re"].astype(np.float32),
                rs=blob["rs"].astype(np.float32),
                fail_p=blob["fail_p"].astype(np.int32),
                fail_r=blob["fail_r"].astype(np.int32),
                e_p=blob["e_p"].astype(np.float32),
                e_r=blob["e_r"].astype(np.float32),
                sr_ep=blob["sr_ep"].astype(np.int32),
                s_enabled=blob["s_enabled"].astype(np.int32),
                lam=blob["lam"].astype(np.float32),
                seed=blob["seed"].astype(np.int32),
                t=blob["t"].astype(np.int32),
                task=task,
                clip=clip,
            )
        )
        print(f"[ura0] load {p} n={n}", flush=True)
    if not feats:
        raise SystemExit(f"no clones in {raw}")
    out = {"feat": np.concatenate(feats, 0), "hist": np.concatenate(hists, 0)}
    for k in rest[0]:
        if k in ("task", "clip"):
            out[k] = np.concatenate([r[k] for r in rest], 0)
        else:
            out[k] = np.concatenate([r[k] for r in rest], 0)
    return out


def oracle_block(d, mask=None):
    if mask is None:
        mask = np.ones(len(d["i_rec"]), dtype=bool)
    i = d["i_rec"][mask]
    return {
        "n": int(mask.sum()),
        "P_I_lt_0": float((i < 0).mean()) if mask.any() else float("nan"),
        "P_I_lt_eps": float((i < -EPS).mean()) if mask.any() else float("nan"),
        "median_I_rec": float(np.median(i)) if mask.any() else float("nan"),
        "mean_I_rec": float(i.mean()) if mask.any() else float("nan"),
        "median_I_cm": float(np.median(i) * 100.0) if mask.any() else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/ura0_unified_recovery_advantage/raw")
    ap.add_argument("--out", default="results/ura0_unified_recovery_advantage")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    d = load_raw(Path(args.raw))
    y = (d["i_rec"] < -EPS).astype(np.int32)
    useful_clip = []
    for t, c in zip(d["task"], d["clip"]):
        useful_clip.append(0)  # filled below
    useful_by = {}
    for i, (t, c) in enumerate(zip(d["task"], d["clip"])):
        k = f"{t}/{c}"
        useful_by[k] = useful_by.get(k, 0) | int(y[i])
    clip_useful = np.array([useful_by[f"{t}/{c}"] for t, c in zip(d["task"], d["clip"])], dtype=np.int32)
    assign = clip_split(d["task"], d["clip"], clip_useful)
    split = np.array([assign[f"{t}/{c}"] for t, c in zip(d["task"], d["clip"])])
    tr, va, te = split == "train", split == "val", split == "test"

    x = d["feat"].astype(np.float32)
    mu = x[tr].mean(0)
    sd = np.where(x[tr].std(0) < 1e-6, 1.0, x[tr].std(0))
    xz = ((x - mu) / sd).astype(np.float32)
    x_no = drop_intent(x)
    mu_n = x_no[tr].mean(0)
    sd_n = np.where(x_no[tr].std(0) < 1e-6, 1.0, x_no[tr].std(0))
    xz_no = ((x_no - mu_n) / sd_n).astype(np.float32)
    hist = d["hist"].reshape(len(x), -1).astype(np.float32)
    xh = np.concatenate([xz, (hist - hist[tr].mean(0)) / np.where(hist[tr].std(0) < 1e-6, 1.0, hist[tr].std(0))], 1).astype(np.float32)
    tgt = d["i_rec"].astype(np.float32)

    print(f"[ura0] n={len(x)} tr/va/te={int(tr.sum())}/{int(va.sum())}/{int(te.sum())} dim={x.shape[1]} device={device}", flush=True)

    lin = train_reg(LinearProbe(x.shape[1]), xz[tr], tgt[tr], args.epochs, device)
    mlp = train_reg(AdvMLP(x.shape[1]), xz[tr], tgt[tr], args.epochs, device)
    mlp_no = train_reg(AdvMLP(xz_no.shape[1]), xz_no[tr], tgt[tr], args.epochs, device)
    mlp_h = train_reg(AdvMLP(xh.shape[1]), xh[tr], tgt[tr], args.epochs, device)
    counts = {
        "linear": n_params(lin),
        "mlp": n_params(mlp),
        "mlp_no_intent": n_params(mlp_no),
        "mlp_hist": n_params(mlp_h),
        "in_dim": int(x.shape[1]),
        "in_dim_no_intent": int(xz_no.shape[1]),
        "in_dim_hist": int(xh.shape[1]),
    }
    print("[ura0] params", counts, flush=True)

    hat = {
        "linear": predict(lin, xz, device),
        "mlp": predict(mlp, xz, device),
        "mlp_no_intent": predict(mlp_no, xz_no, device),
        "mlp_hist": predict(mlp_h, xh, device),
    }

    def decide_learned(name):
        th = pick_score_thresh(hat[name][va], d["j_p"][va], d["j_r"][va], higher_means_intervene=False)
        inter = hat[name][te] < th
        blk = pack_dec(inter, d["j_p"][te], d["j_r"][te], y[te], d["task"][te], name, th, counts.get(name, 0))
        blk["auroc"] = _auroc(y[te], -hat[name][te])
        blk["auprc"] = _auprc(y[te], -hat[name][te])
        return blk, th

    methods = {}
    methods["always_parent"] = pack_dec(
        np.zeros(int(te.sum()), dtype=bool), d["j_p"][te], d["j_r"][te], y[te], d["task"][te], "always_parent", None
    )
    methods["always_parent"]["auroc"] = float("nan")
    methods["always_recovery"] = pack_dec(
        np.ones(int(te.sum()), dtype=bool), d["j_p"][te], d["j_r"][te], y[te], d["task"][te], "always_recovery", None
    )
    methods["always_recovery"]["auroc"] = float("nan")

    th_re = pick_score_thresh(d["re"][va], d["j_p"][va], d["j_r"][va], True)
    methods["RE"] = pack_dec(d["re"][te] >= th_re, d["j_p"][te], d["j_r"][te], y[te], d["task"][te], "RE", th_re)
    methods["RE"]["auroc"] = _auroc(y[te], d["re"][te])
    methods["RE"]["auprc"] = _auprc(y[te], d["re"][te])

    th_rs = pick_score_thresh(d["rs"][va], d["j_p"][va], d["j_r"][va], True)
    methods["RS"] = pack_dec(d["rs"][te] >= th_rs, d["j_p"][te], d["j_r"][te], y[te], d["task"][te], "RS", th_rs)
    methods["RS"]["auroc"] = _auroc(y[te], d["rs"][te])
    methods["RS"]["auprc"] = _auprc(y[te], d["rs"][te])

    mx_va = np.maximum(d["re"][va], d["rs"][va])
    mx_te = np.maximum(d["re"][te], d["rs"][te])
    th_mx = pick_score_thresh(mx_va, d["j_p"][va], d["j_r"][va], True)
    methods["max_RE_RS"] = pack_dec(mx_te >= th_mx, d["j_p"][te], d["j_r"][te], y[te], d["task"][te], "max_RE_RS", th_mx)
    methods["max_RE_RS"]["auroc"] = _auroc(y[te], mx_te)
    methods["max_RE_RS"]["auprc"] = _auprc(y[te], mx_te)

    hist_gate = (d["re"][te] >= 1.0) | ((d["s_enabled"][te] == 1) & (d["rs"][te] >= 1.0))
    methods["RM3_hist_gate"] = pack_dec(
        hist_gate, d["j_p"][te], d["j_r"][te], y[te], d["task"][te], "RM3_hist_gate", 1.0
    )
    methods["RM3_hist_gate"]["auroc"] = _auroc(y[te], np.maximum(d["re"][te], d["rs"][te] * d["s_enabled"][te]))

    oracle_inter = y[te] == 1
    methods["oracle"] = pack_dec(oracle_inter, d["j_p"][te], d["j_r"][te], y[te], d["task"][te], "oracle", -EPS)
    methods["oracle"]["auroc"] = 1.0

    for name in ("linear", "mlp", "mlp_no_intent", "mlp_hist"):
        blk, _th = decide_learned(name)
        methods[name] = blk

    # Carry no-benefit intervention + Stoop useful recall on test
    te_task = d["task"][te]
    te_y = y[te]
    te_sr = d["sr_ep"][te]
    extra = {}
    for name, m in methods.items():
        if name in ("always_parent", "always_recovery", "oracle"):
            if name == "always_parent":
                inter = np.zeros(int(te.sum()), dtype=bool)
            elif name == "always_recovery":
                inter = np.ones(int(te.sum()), dtype=bool)
            else:
                inter = te_y == 1
        elif name == "RE":
            inter = d["re"][te] >= methods["RE"]["thresh"]
        elif name == "RS":
            inter = d["rs"][te] >= methods["RS"]["thresh"]
        elif name == "max_RE_RS":
            inter = mx_te >= methods["max_RE_RS"]["thresh"]
        elif name == "RM3_hist_gate":
            inter = hist_gate
        else:
            inter = hat[name][te] < methods[name]["thresh"]
        carry_nb = (te_task == "carry") & (te_y == 0)
        stoop_u = (te_task == "stoop") & (te_y == 1)
        carry_ok = (te_task == "carry") & (te_sr == 1) & (te_y == 0)
        extra[name] = {
            "carry_nobeh_inter": float(inter[carry_nb].mean()) if carry_nb.any() else float("nan"),
            "carry_success_nobeh_inter": float(inter[carry_ok].mean()) if carry_ok.any() else float("nan"),
            "stoop_useful_recall": float(inter[stoop_u].mean()) if stoop_u.any() else float("nan"),
            "n_carry_nobeh": int(carry_nb.sum()),
            "n_stoop_useful": int(stoop_u.sum()),
        }

    dist = {}
    for t in TASKS:
        for lam in (0.0, 1.0, 2.0):
            m = (d["task"] == t) & (np.round(d["lam"]) == lam)
            dist[f"{t}/lam{int(lam)}"] = int(m.sum())

    oracle = {"pooled": oracle_block(d), "test": oracle_block(d, te), "by_task": {}}
    for t in TASKS:
        oracle["by_task"][t] = oracle_block(d, d["task"] == t)
        oracle["by_task"][f"{t}_test"] = oracle_block(d, te & (d["task"] == t))

    learned_name = "mlp"
    base_reg = min(methods["RE"]["mean_regret"], methods["RS"]["mean_regret"], methods["max_RE_RS"]["mean_regret"])
    mlp_reg = methods["mlp"]["mean_regret"]
    lin_reg = methods["linear"]["mean_regret"]
    keep_simple = lin_reg <= mlp_reg + 1e-4
    best_learned = "linear" if keep_simple else "mlp"
    if methods["mlp_hist"]["mean_regret"] < methods[best_learned]["mean_regret"] - 0.001:
        best_learned = "mlp_hist"

    stoop_rec = extra[best_learned]["stoop_useful_recall"]
    carry_nb = extra[best_learned]["carry_nobeh_inter"]
    rs_carry = extra["RS"]["carry_nobeh_inter"]
    hist_carry = extra["RM3_hist_gate"]["carry_nobeh_inter"]
    carry_ref = rs_carry if rs_carry == rs_carry else hist_carry

    loco_ok = methods[best_learned]["per_task"].get("loco", {}).get("mean_regret", 1e9) <= methods["always_parent"]["per_task"].get("loco", {}).get("mean_regret", 0) + 0.02
    reach_ok = methods[best_learned]["per_task"].get("reach", {}).get("mean_regret", 1e9) <= methods["always_parent"]["per_task"].get("reach", {}).get("mean_regret", 0) + 0.02
    regret_win = mlp_reg < base_reg - 0.002 or methods[best_learned]["mean_regret"] < base_reg - 0.002
    stoop_ok = stoop_rec == stoop_rec and stoop_rec >= 0.70
    carry_ok_v = carry_nb == carry_nb and carry_ref == carry_ref and carry_nb <= carry_ref - 0.15

    reach_oracle = oracle["by_task"]["reach"]["P_I_lt_0"]
    carry_oracle = oracle["by_task"]["carry"]["P_I_lt_0"]
    field_gap = (reach_oracle == reach_oracle and reach_oracle < 0.10) or (
        carry_oracle == carry_oracle and carry_oracle < 0.10 and oracle["by_task"]["carry"]["median_I_rec"] >= 0
    )

    z_drop = methods["mlp"]["mean_regret"] - methods["mlp_no_intent"]["mean_regret"]
    intent_needed = methods["mlp"]["mean_regret"] < methods["mlp_no_intent"]["mean_regret"] - 0.001

    go = bool(regret_win and stoop_ok and carry_ok_v and loco_ok and reach_ok)
    verdict = {
        "decision": "GO_UNIFIED_RECOVERY_ADVANTAGE" if go else "HOLD",
        "best_learned": best_learned,
        "keep_simple_linear": bool(keep_simple and best_learned == "linear"),
        "regret_win_vs_RERS": bool(regret_win),
        "stoop_useful_recall_ok": bool(stoop_ok),
        "carry_nobeh_ok": bool(carry_ok_v),
        "loco_ok": bool(loco_ok),
        "reach_ok": bool(reach_ok),
        "intent_conditioning_helps": bool(intent_needed),
        "zM_ablation_regret_delta": float(z_drop),
        "rm3_field_gap_reach_carry": bool(field_gap),
        "note": (
            "Oracle single-burst advantage is rare on Reach/Carry; problem may be R-M3 field coverage, not gating."
            if field_gap
            else ""
        ),
    }
    if not go:
        reasons = []
        if not regret_win:
            reasons.append("learned regret not clearly below RE/RS/max")
        if not stoop_ok:
            reasons.append("Stoop useful-recovery recall insufficient")
        if not carry_ok_v:
            reasons.append("Carry no-benefit intervention not clearly below RS/hist gate")
        if not loco_ok or not reach_ok:
            reasons.append("Loco/Reach regret collapse vs always-Parent")
        verdict["reasons"] = reasons

    torch.save(
        {"mlp": mlp.state_dict(), "linear": lin.state_dict(), "mu": mu, "sd": sd, "counts": counts},
        out / "checkpoints" / "ura0.pt",
    )

    split_info = {
        "n_states": int(len(x)),
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "n_test": int(te.sum()),
        "n_clips": len(assign),
        "split": "clip-level, useful-stratified, not random-state",
        "task_lam_n": dist,
        "eps_m": EPS,
        "lam_fail": 1.0,
        "J": "masked_e_H + 1.0 * official_fail",
        "sample_ts": [40, 100, 160, 220, 280, 340],
        "recovery": "frozen R-M3 adaptive 100ms burst then Parent, H=0.5s",
    }
    table_c = []
    for n in (
        "always_parent",
        "always_recovery",
        "RE",
        "RS",
        "max_RE_RS",
        "RM3_hist_gate",
        "linear",
        "mlp",
        "mlp_no_intent",
        "mlp_hist",
        "oracle",
    ):
        m = methods[n]
        table_c.append(
            {
                "method": n,
                "params": m.get("params", 0),
                "mean_regret": m["mean_regret"],
                "median_regret": m["median_regret"],
                "auroc": m.get("auroc"),
                "precision": m["precision"],
                "recall": m["recall"],
                "p_intervene": m["p_intervene"],
            }
        )
    per_task = {}
    for t in TASKS:
        per_task[t] = {
            "oracle": oracle["by_task"][f"{t}_test"],
            "learned_regret": methods[best_learned]["per_task"].get(t, {}),
            "RE_regret": methods["RE"]["per_task"].get(t, {}),
            "RS_regret": methods["RS"]["per_task"].get(t, {}),
            "always_parent_regret": methods["always_parent"]["per_task"].get(t, {}),
        }
    compact = {
        "A_dump": split_info,
        "B_oracle": oracle,
        "C_table": table_c,
        "D_per_task": per_task,
        "E_stoop_carry": extra,
        "F_intent_ablation": {
            "mlp": {"regret": methods["mlp"]["mean_regret"], "auroc": methods["mlp"]["auroc"]},
            "mlp_no_z_M": {
                "regret": methods["mlp_no_intent"]["mean_regret"],
                "auroc": methods["mlp_no_intent"]["auroc"],
            },
            "intent_helps": bool(intent_needed),
        },
        "G_verdict": verdict,
        "params": counts,
        "methods": methods,
    }
    (out / "report_compact.json").write_text(json.dumps(compact, indent=2, default=_json_default), encoding="utf-8")
    (out / "summary_ura0.json").write_text(json.dumps(compact, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"params": counts, "verdict": verdict, "C": table_c, "oracle_pooled": oracle["pooled"]}, indent=2, default=_json_default))
    print("[ura0] wrote", out / "report_compact.json")


if __name__ == "__main__":
    main()
