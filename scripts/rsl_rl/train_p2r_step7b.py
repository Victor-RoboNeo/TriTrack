#!/usr/bin/env python3
"""Step 7B-Small — Process-state recovery-advantage identifiability. No PPO. No Isaac.

Ablations (burst index is never an input):
  A: o_R
  B: A + E / ΔE 100–200 ms (burst-boundary lags)
  C: B + previous recovery response (ΔE_prev, ΔR_E prev)
  D: C + d_{b-1}, cosine(d_b, d_{b-1}), ||Δd||, T_rec

Train on |I_500| > eps only. Deploy: score>0 continue else Parent (uncertain→Parent).
Primary metric: decision regret vs Always-Parent / Max-2 / Oracle.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_p2r_step6s import SupervisedRecoveryMLP, _project_unit  # noqa: E402
from train_p2r_step7a import (  # noqa: E402
    BURST_GROUPS,
    IN_DIM,
    TERRAINS,
    _attach_obs,
    _auroc,
    _ep_key,
    _load_rows,
    _prf,
    _regret_block,
    _sanitize,
    _split_episodes,
)

STEP6S_CKPT = "/data/home/chenxiangyu/robotics/Anybody/results/p2r_step6s_full/model_best.pt"
DT = 0.02
Z_SLICE = slice(471, 487)
MODELS = ("A", "B", "C", "D")


def _cluster_prev(rows: list[dict]) -> None:
    """Attach prev / prev2 / t0_event using (clip, t0, burst_index). dt==5 in this dump."""
    by: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        by.setdefault((r["terrain"], int(r["seed"]), r["clip"]), []).append(i)
    for idxs in by.values():
        idxs.sort(key=lambda i: (int(rows[i]["t0"]), int(rows[i]["burst_index"])))
        b1_t0 = [int(rows[i]["t0"]) for i in idxs if int(rows[i]["burst_index"]) == 1]
        for j, i in enumerate(idxs):
            r = rows[i]
            r["prev_i"] = None
            r["prev2_i"] = None
            t0 = int(r["t0"])
            b = int(r["burst_index"])
            for p in reversed(idxs[:j]):
                dt = t0 - int(rows[p]["t0"])
                db = b - int(rows[p]["burst_index"])
                if 1 <= dt <= 12 and db == 1:
                    r["prev_i"] = p
                    break
            if r["prev_i"] is not None:
                rp = rows[r["prev_i"]]
                j2 = idxs.index(r["prev_i"])
                for p2 in reversed(idxs[:j2]):
                    dt = int(rp["t0"]) - int(rows[p2]["t0"])
                    db = int(rp["burst_index"]) - int(rows[p2]["burst_index"])
                    if 1 <= dt <= 12 and db == 1:
                        r["prev2_i"] = p2
                        break
            ev = t0
            for t1 in b1_t0:
                if t1 <= t0 and (t0 - t1) < 400:
                    ev = t1
                    break
            r["t0_event"] = int(ev)


def _predict_dirs(rows: list[dict], ckpt: str) -> np.ndarray:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = SupervisedRecoveryMLP()
    model.load_state_dict(blob["model"])
    model.eval()
    mu = torch.as_tensor(blob["obs_mean"], dtype=torch.float32)
    sd = torch.as_tensor(blob["obs_std"], dtype=torch.float32)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    x = np.stack([r["rec_obs"] for r in rows], axis=0).astype(np.float32)
    z = torch.from_numpy(x[:, Z_SLICE])
    xt = (torch.from_numpy(x) - mu) / sd
    with torch.no_grad():
        d = _project_unit(model(xt), z).numpy()
    return d.astype(np.float32)


def _process_block(r: dict, rows: list[dict], d_all: np.ndarray, i: int) -> dict[str, np.ndarray]:
    e_t = float(r["E0_cm"])
    re_t = float(r["R_E"])
    pi, p2 = r.get("prev_i"), r.get("prev2_i")
    e_m5 = float(rows[pi]["E0_cm"]) if pi is not None else e_t
    e_m10 = float(rows[p2]["E0_cm"]) if p2 is not None else e_m5
    d_e100 = e_t - e_m5
    d_e200 = e_t - e_m10
    d_e_prev = d_e100 if pi is not None else 0.0
    d_r_prev = (re_t - float(rows[pi]["R_E"])) if pi is not None else 0.0
    t_rec = (int(r["t0"]) - int(r["t0_event"])) * DT
    d_now = d_all[i]
    if pi is not None:
        d_prev = d_all[pi]
        cos = float(np.dot(d_now, d_prev))
        l2 = float(np.linalg.norm(d_now - d_prev))
    else:
        d_prev = np.zeros(16, dtype=np.float32)
        cos, l2 = 0.0, 0.0
    hist = np.array([e_t, e_m5, e_m10, d_e100, d_e200], dtype=np.float32)
    resp = np.array([d_e_prev, d_r_prev], dtype=np.float32)
    dfeat = np.concatenate([d_prev, np.array([cos, l2, t_rec], dtype=np.float32)])
    return {"hist": hist, "resp": resp, "dfeat": dfeat}


def _phi(rows: list[dict], d_all: np.ndarray, kind: str) -> np.ndarray:
    blocks = [_process_block(r, rows, d_all, i) for i, r in enumerate(rows)]
    o = np.stack([r["rec_obs"] for r in rows], axis=0).astype(np.float32)
    if kind == "A":
        return o
    hist = np.stack([b["hist"] for b in blocks], axis=0)
    if kind == "B":
        return np.concatenate([o, hist], axis=1)
    resp = np.stack([b["resp"] for b in blocks], axis=0)
    if kind == "C":
        return np.concatenate([o, hist, resp], axis=1)
    dfeat = np.stack([b["dfeat"] for b in blocks], axis=0)
    return np.concatenate([o, hist, resp, dfeat], axis=1)


class AdvantageMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        h = 128 if in_dim >= 256 else 64
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _clear_mask(rows: list[dict], eps: float) -> np.ndarray:
    a = np.array([r["I_500_cm"] for r in rows], dtype=np.float64)
    return np.abs(a) > float(eps)


def _y_continue(rows: list[dict], eps: float) -> np.ndarray:
    return np.array([r["I_500_cm"] < -float(eps) for r in rows], dtype=np.int32)


def _eval_policy(rows: list[dict], pred: np.ndarray, score: np.ndarray | None, eps: float, name: str) -> dict:
    pred = np.asarray(pred, dtype=bool)
    y = _y_continue(rows, eps).astype(bool)
    clear = _clear_mask(rows, eps)
    cls = _prf(y, pred)
    # continue precision against clear-help labels (FP continue = over-intervention)
    if pred.any():
        cls["continue_precision"] = float(y[pred].mean())
    else:
        cls["continue_precision"] = float("nan")
    out = {
        "n": len(rows),
        "n_clear": int(clear.sum()),
        "P_continue_label": float(y.mean()) if y.size else float("nan"),
        "cls": cls,
        "regret": _regret_block(rows, pred),
        "policy": name,
    }
    if score is not None:
        out["AUROC"] = _auroc(y.astype(np.int32), score)
        if clear.any() and y[clear].min() != y[clear].max():
            out["AUROC_clear"] = _auroc(y[clear].astype(np.int32), np.asarray(score)[clear])
        else:
            out["AUROC_clear"] = float("nan")
    by_g = {}
    for g in BURST_GROUPS:
        idx = [i for i, r in enumerate(rows) if r.get("burst_group") == g]
        if not idx:
            continue
        rs = [rows[i] for i in idx]
        pg = pred[idx]
        blk = {"n": len(idx), "cls": _prf(y[idx], pg), "regret": _regret_block(rs, pg)}
        if pg.any():
            blk["cls"]["continue_precision"] = float(y[idx][pg].mean())
        if score is not None:
            blk["AUROC"] = _auroc(y[idx].astype(np.int32), np.asarray(score)[idx])
        by_g[g] = blk
    out["by_burst_group"] = by_g
    return out


def _train_mlp(x_tr, y_tr, x_va, y_va, seed: int, epochs: int, batch: int, lr: float, wd: float, patience: int):
    torch.manual_seed(seed)
    model = AdvantageMLP(x_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    xv, yv = torch.from_numpy(x_va), torch.from_numpy(y_va)
    n = xt.shape[0]
    best_state, best_val, stale = None, -1e9, 0
    yv_np = y_va.astype(np.int32)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            s_val = model(xv).numpy()
        auroc_val = _auroc(yv_np, s_val) if yv_np.min() != yv_np.max() else float("nan")
        if np.isfinite(auroc_val) and auroc_val > best_val + 1e-4:
            best_val = float(auroc_val)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience and epoch >= 40:
            break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_val = float("nan")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="results/p2r_step6s_r")
    ap.add_argument("--out", type=str, default="results/p2r_step7b_small")
    ap.add_argument("--step6s_ckpt", type=str, default=STEP6S_CKPT)
    ap.add_argument("--eps_cm", type=float, default=0.25)
    ap.add_argument("--split_seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=40)
    args = ap.parse_args()
    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = float(args.eps_cm)
    rows = _load_rows(root)
    n_obs = _attach_obs(rows, root)
    assert n_obs == len(rows) and all("rec_obs" in r for r in rows)
    _cluster_prev(rows)
    n_prev = sum(r.get("prev_i") is not None for r in rows)
    n_prev2 = sum(r.get("prev2_i") is not None for r in rows)
    print(f"[7b] n={len(rows)} prev={n_prev} prev2={n_prev2} clear={int(_clear_mask(rows, eps).sum())}", flush=True)

    d_all = _predict_dirs(rows, args.step6s_ckpt)
    phis = {k: _phi(rows, d_all, k) for k in MODELS}
    for k, x in phis.items():
        print(f"[7b] phi_{k} dim={x.shape[1]}", flush=True)

    split_seeds = [int(s) for s in args.split_seeds.split(",") if s.strip()]
    y_all = _y_continue(rows, eps).astype(np.float32)
    payload = {
        "step": "7B-Small",
        "no_ppo": True,
        "eps_cm": eps,
        "n_rows": len(rows),
        "n_prev": n_prev,
        "n_prev2": n_prev2,
        "n_clear": int(_clear_mask(rows, eps).sum()),
        "phi_dim": {k: int(phis[k].shape[1]) for k in MODELS},
        "note": "Train |I|>eps; deploy score>0 continue else Parent. No burst_index/terrain/depth.",
        "by_seed": [],
    }

    agg = {k: {"auroc": [], "auroc_clear": [], "regret": [], "cont_prec": []} for k in MODELS}
    agg.update({b: {"regret": [], "cont_prec": []} for b in ("always_parent", "always_recovery", "max2", "response_heuristic", "oracle")})

    for seed in split_seeds:
        splits = _split_episodes(rows, seed=seed)
        idx = {name: [i for i, r in enumerate(rows) if _ep_key(r) in splits[name]] for name in ("train", "val", "test")}
        tr_i, va_i, te_i = idx["train"], idx["val"], idx["test"]
        te_rows = [rows[i] for i in te_i]
        bidx = np.array([int(r["burst_index"]) for r in te_rows])
        a_te = np.array([r["I_500_cm"] for r in te_rows])
        dE_prev = np.array(
            [
                (float(r["E0_cm"]) - float(rows[r["prev_i"]]["E0_cm"])) if r.get("prev_i") is not None else -1e9
                for r in te_rows
            ]
        )
        has_prev = np.array([r.get("prev_i") is not None for r in te_rows])
        # First burst: continue. Later: continue only if last 100 ms E dropped.
        heur = (~has_prev) | (dE_prev < -eps)
        baselines = {
            "always_parent": np.zeros(len(te_rows), dtype=bool),
            "always_recovery": np.ones(len(te_rows), dtype=bool),
            "max2": bidx <= 2,
            "response_heuristic": heur,
            "oracle": a_te < -eps,
        }
        seed_blk = {"split_seed": seed, "n_test": len(te_rows), "baselines": {}, "models": {}}
        for name, mask in baselines.items():
            seed_blk["baselines"][name] = _eval_policy(te_rows, mask, None, eps, name)
            agg[name]["regret"].append(seed_blk["baselines"][name]["regret"]["mean_regret_cm"])
            agg[name]["cont_prec"].append(
                seed_blk["baselines"][name]["cls"].get("continue_precision", float("nan"))
            )

        clear_tr = _clear_mask([rows[i] for i in tr_i], eps)
        clear_va = _clear_mask([rows[i] for i in va_i], eps)
        tr_use = [tr_i[j] for j, c in enumerate(clear_tr) if c]
        va_use = [va_i[j] for j, c in enumerate(clear_va) if c] or tr_use
        if len(va_use) < 4:
            va_use = tr_use
        print(f"[7b] seed={seed} train_clear={len(tr_use)} val_clear={len(va_use)} test={len(te_i)}", flush=True)

        for kind in MODELS:
            x_tr = phis[kind][np.array(tr_use)]
            y_tr = y_all[np.array(tr_use)]
            x_va = phis[kind][np.array(va_use)]
            y_va = y_all[np.array(va_use)]
            mu = x_tr.mean(0)
            sd = np.clip(x_tr.std(0), 1e-6, None)
            model, val_a = _train_mlp(
                ((x_tr - mu) / sd).astype(np.float32),
                y_tr,
                ((x_va - mu) / sd).astype(np.float32),
                y_va,
                seed=seed,
                epochs=int(args.epochs),
                batch=int(args.batch),
                lr=float(args.lr),
                wd=float(args.wd),
                patience=int(args.patience),
            )
            x_te = ((phis[kind][np.array(te_i)] - mu) / sd).astype(np.float32)
            with torch.no_grad():
                score = model(torch.from_numpy(x_te)).numpy()
            pred = score > 0.0
            ev = _eval_policy(te_rows, pred, score, eps, f"mlp_{kind}")
            ev["val_AUROC"] = val_a
            seed_blk["models"][kind] = ev
            agg[kind]["auroc"].append(ev.get("AUROC", float("nan")))
            agg[kind]["auroc_clear"].append(ev.get("AUROC_clear", float("nan")))
            agg[kind]["regret"].append(ev["regret"]["mean_regret_cm"])
            agg[kind]["cont_prec"].append(ev["cls"].get("continue_precision", float("nan")))
            print(
                f"[7b] seed={seed} {kind} dim={phis[kind].shape[1]} "
                f"AUROC={ev.get('AUROC'):.3f} clear={ev.get('AUROC_clear')} "
                f"regret={ev['regret']['mean_regret_cm']:.3f} "
                f"Cprec={ev['cls'].get('continue_precision')} val={val_a:.3f}",
                flush=True,
            )
        payload["by_seed"].append(seed_blk)

    def _meanstd(xs):
        a = np.array(xs, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1) if a.size > 1 else 0.0), "n": int(a.size)}

    summary = {}
    for k in ("always_parent", "always_recovery", "max2", "response_heuristic", "oracle"):
        summary[k] = {
            "mean_regret_cm": _meanstd(agg[k]["regret"]),
            "continue_precision": _meanstd(agg[k]["cont_prec"]),
        }
    for k in MODELS:
        summary[k] = {
            "AUROC": _meanstd(agg[k]["auroc"]),
            "AUROC_clear": _meanstd(agg[k]["auroc_clear"]),
            "mean_regret_cm": _meanstd(agg[k]["regret"]),
            "continue_precision": _meanstd(agg[k]["cont_prec"]),
        }
    payload["pooled_over_splits"] = summary
    max2_r = summary["max2"]["mean_regret_cm"]["mean"]
    best_k, best_r = "A", 1e9
    for k in MODELS:
        r = summary[k]["mean_regret_cm"]["mean"]
        if r is not None and r < best_r:
            best_k, best_r = k, r
    best_a = summary[best_k]["AUROC"]["mean"]
    go_full = bool(
        best_r is not None
        and max2_r is not None
        and best_r < max2_r - 0.05
        and best_a is not None
        and best_a >= 0.62
    )
    payload["gates"] = {
        "best_model": best_k,
        "best_mean_regret_cm": best_r,
        "best_mean_AUROC": best_a,
        "max2_mean_regret_cm": max2_r,
        "GO_7B_Full": go_full,
    }
    payload["next"] = (
        "GO 7B-Full: 800–1200 clones focused on burst 2–4, episode CV. Still no PPO."
        if go_full
        else "HOLD 7B-Full / 7-CL. Process features did not beat Max-2 reliably. Inspect C vs A; do not PPO."
    )
    print(
        f"[7b] BEST {best_k} AUROC={best_a} regret={best_r:.3f} max2={max2_r:.3f} GO_Full={go_full}",
        flush=True,
    )
    (out / "summary.json").write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    print("[7b] wrote", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
