#!/usr/bin/env python3
"""Step 7B-Full — Event-level 5-fold CV on 50 Hz process-state. No PPO. No Isaac.

A: current-state o_R (487)
B: o_R + handcrafted 100–200 ms trajectory features
C: lightweight GRU on 50 Hz traj (15×62) + projected o_R

Train on |I_500| > eps. Deploy: score>0 continue else Parent.
Primary metric: decision regret. GO 7-CL iff stably below Max-2 bar 0.671 cm.
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_p2r_step7a import (  # noqa: E402
    _auroc,
    _prf,
    _regret_block,
    _sanitize,
)
from train_p2r_step7b import AdvantageMLP, _eval_policy  # noqa: E402

TERRAINS = ("plane", "light_rough", "slope", "steps")
BURST_GROUPS = ("2", "3", "4")
MAX2_BAR_CM = 0.671
TRAJ_LEN = 15
TRAJ_DIM = 62
HAND_DIM = 10
OBS_DIM = 487


def _event_key(r: dict) -> str:
    return f"{r['terrain']}|{r['seed']}|{r['clip']}|{int(r['t_event0'])}"


def _clear_mask(rows: list[dict], eps: float) -> np.ndarray:
    a = np.array([r["I_500_cm"] for r in rows], dtype=np.float64)
    return np.abs(a) > float(eps)


def _y_continue(rows: list[dict], eps: float) -> np.ndarray:
    return np.array([r["I_500_cm"] < -float(eps) for r in rows], dtype=np.int32)


def _load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    missing = []
    for ter in TERRAINS:
        p = root / ter / "rows.json"
        if not p.exists():
            missing.append(ter)
            continue
        for r in json.loads(p.read_text()):
            r = dict(r)
            r["burst_group"] = str(r.get("burst_group") or int(r["burst_index"]))
            rows.append(r)
    if missing:
        print(f"[7b-full] missing terrains {missing}", flush=True)
    if not rows:
        raise SystemExit(f"no rows under {root}")
    return rows


def _attach_traj(rows: list[dict], root: Path) -> int:
    n_hit = 0
    for ter in TERRAINS:
        p = root / ter / "traj.npz"
        if not p.exists():
            p = root / ter / "rec_obs.npz"
        if not p.exists():
            continue
        blob = np.load(p, allow_pickle=True)
        rec = blob["rec_obs"]
        traj = blob["traj"]
        traj_len = blob["traj_len"]
        d_prev = blob["d_prev"]
        d_now = blob["d_now"]
        keys = {}
        for i in range(rec.shape[0]):
            k = (
                str(blob["terrain"][i]),
                int(blob["seed"][i]),
                str(blob["clip"][i]),
                int(blob["t0"][i]),
                int(blob["burst_index"][i]),
            )
            keys[k] = i
        for r in rows:
            if r["terrain"] != ter:
                continue
            k = (str(r["terrain"]), int(r["seed"]), str(r["clip"]), int(r["t0"]), int(r["burst_index"]))
            if k not in keys:
                continue
            i = keys[k]
            r["rec_obs"] = rec[i].astype(np.float32)
            r["traj"] = traj[i].astype(np.float32)
            r["traj_len"] = int(traj_len[i])
            r["d_prev"] = d_prev[i].astype(np.float32)
            r["d_now"] = d_now[i].astype(np.float32)
            n_hit += 1
    return n_hit


def _right_pad(traj: np.ndarray, traj_len: np.ndarray) -> np.ndarray:
    out = np.zeros_like(traj)
    for i, L in enumerate(traj_len):
        L = int(L)
        if L > 0:
            out[i, :L] = traj[i, -L:]
    return out


def _handcrafted(rows: list[dict]) -> np.ndarray:
    """[E_t, E_{t-5}, E_{t-10}, ΔE_100, ΔE_200, ΔE_prev, ΔR_prev, cos, ||Δd||, T_rec]."""
    feats = np.zeros((len(rows), HAND_DIM), dtype=np.float32)
    for i, r in enumerate(rows):
        traj = np.asarray(r["traj"], dtype=np.float32)
        L = int(r["traj_len"])
        e_cm = traj[:, 0] * 100.0
        e_t = float(e_cm[-1])
        e_m5 = float(e_cm[-6]) if L >= 6 else e_t
        e_m10 = float(e_cm[-11]) if L >= 11 else e_m5
        dE100 = e_t - e_m5
        dE200 = e_t - e_m10
        dE_prev = float(r.get("dE_prev", 0.0)) * 100.0
        dR_prev = float(r.get("dR_prev", 0.0))
        d_now = np.asarray(r["d_now"], dtype=np.float32)
        d_prev = np.asarray(r["d_prev"], dtype=np.float32)
        n_prev = float(np.linalg.norm(d_prev))
        if n_prev > 1e-6:
            cos = float(np.dot(d_now, d_prev) / (float(np.linalg.norm(d_now)) * n_prev + 1e-8))
            l2 = float(np.linalg.norm(d_now - d_prev))
        else:
            cos, l2 = 0.0, 0.0
        t_rec = float(r.get("T_rec", 0.0))
        feats[i] = np.array(
            [e_t, e_m5, e_m10, dE100, dE200, dE_prev, dR_prev, cos, l2, t_rec],
            dtype=np.float32,
        )
    return feats


def _event_folds(rows: list[dict], n_folds: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    by_ter: dict[str, list[str]] = {t: [] for t in TERRAINS}
    seen: set[str] = set()
    for r in rows:
        k = _event_key(r)
        if k in seen:
            continue
        seen.add(k)
        by_ter.setdefault(r["terrain"], []).append(k)
    event_fold: dict[str, int] = {}
    for ter, evs in by_ter.items():
        rng.shuffle(evs)
        for i, k in enumerate(evs):
            event_fold[k] = i % n_folds
    return event_fold


def _train_val_events(events: list[str], rows: list[dict], rng: np.random.Generator, val_frac: float = 0.2):
    by_ter: dict[str, list[str]] = {}
    ev_set = set(events)
    seen: set[str] = set()
    for r in rows:
        k = _event_key(r)
        if k not in ev_set or k in seen:
            continue
        seen.add(k)
        by_ter.setdefault(r["terrain"], []).append(k)
    tr, va = [], []
    for ter, evs in by_ter.items():
        rng.shuffle(evs)
        n_va = max(1, int(round(val_frac * len(evs)))) if len(evs) >= 5 else max(0, len(evs) // 5)
        va.extend(evs[:n_va])
        tr.extend(evs[n_va:])
    if not tr:
        tr, va = list(events), []
    return tr, va


def _idx_for_events(rows: list[dict], events: set[str] | list[str]) -> list[int]:
    ev = set(events)
    return [i for i, r in enumerate(rows) if _event_key(r) in ev]


def _traj_stats(traj: np.ndarray, traj_len: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chunks = []
    for i, L in enumerate(traj_len):
        L = max(int(L), 1)
        chunks.append(traj[i, :L])
    cat = np.concatenate(chunks, axis=0)
    mu = cat.mean(axis=0).astype(np.float32)
    sd = np.clip(cat.std(axis=0), 1e-6, None).astype(np.float32)
    return mu, sd


def _norm_traj(traj: np.ndarray, traj_len: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    x = (traj - mu) / sd
    for i, L in enumerate(traj_len):
        x[i, int(L) :] = 0.0
    return x.astype(np.float32)


class TemporalEncoder(nn.Module):
    """GRU on 50 Hz traj; projected current-state o_R. No terrain."""

    def __init__(self, traj_dim: int = TRAJ_DIM, obs_dim: int = OBS_DIM, hid: int = 32):
        super().__init__()
        self.obs_proj = nn.Sequential(nn.Linear(obs_dim, hid), nn.ReLU())
        self.gru = nn.GRU(traj_dim, hid, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hid * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor, traj: torch.Tensor, traj_len: torch.Tensor) -> torch.Tensor:
        lengths = traj_len.detach().cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            traj, lengths, batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = h[-1]
        z = torch.cat([self.obs_proj(obs), h], dim=-1)
        return self.head(z).squeeze(-1)


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


def _train_temporal(
    obs_tr, traj_tr, len_tr, y_tr,
    obs_va, traj_va, len_va, y_va,
    seed: int, epochs: int, batch: int, lr: float, wd: float, patience: int,
):
    torch.manual_seed(seed)
    model = TemporalEncoder()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    ot = torch.from_numpy(obs_tr)
    tt = torch.from_numpy(traj_tr)
    lt = torch.from_numpy(len_tr.astype(np.int64))
    yt = torch.from_numpy(y_tr)
    ov = torch.from_numpy(obs_va)
    tv = torch.from_numpy(traj_va)
    lv = torch.from_numpy(len_va.astype(np.int64))
    n = ot.shape[0]
    best_state, best_val, stale = None, -1e9, 0
    yv_np = y_va.astype(np.int32)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(ot[idx], tt[idx], lt[idx]), yt[idx]
            )
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            s_val = model(ov, tv, lv).numpy()
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


def _meanstd(xs) -> dict:
    a = np.array(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": None, "std": None, "n": 0, "vals": []}
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1) if a.size > 1 else 0.0),
        "n": int(a.size),
        "vals": [float(x) for x in a],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="results/p2r_step7b_full")
    ap.add_argument("--out", type=str, default="results/p2r_step7b_full/cv")
    ap.add_argument("--eps_cm", type=float, default=0.25)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=0)
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
    n_folds = int(args.n_folds)

    rows = _load_rows(root)
    n_hit = _attach_traj(rows, root)
    assert n_hit == len(rows) and all("traj" in r and "rec_obs" in r for r in rows)
    obs = np.stack([r["rec_obs"] for r in rows], axis=0).astype(np.float32)
    traj_left = np.stack([r["traj"] for r in rows], axis=0).astype(np.float32)
    traj_len = np.array([max(int(r["traj_len"]), 1) for r in rows], dtype=np.int32)
    traj = _right_pad(traj_left, traj_len)
    hand = _handcrafted(rows)
    y_all = _y_continue(rows, eps).astype(np.float32)
    clear = _clear_mask(rows, eps)
    event_fold = _event_folds(rows, n_folds, seed=int(args.fold_seed))
    n_events = len(event_fold)
    print(
        f"[7b-full] n={len(rows)} events={n_events} matched={n_hit} "
        f"clear={int(clear.sum())} traj={traj.shape} hand={hand.shape}",
        flush=True,
    )

    names = ("always_parent", "always_recovery", "max2", "dE100_heuristic", "oracle", "A", "B", "C")
    agg = {k: {"auroc": [], "regret": [], "cont_prec": []} for k in names}
    payload = {
        "step": "7B-Full-CV",
        "no_ppo": True,
        "eps_cm": eps,
        "n_rows": len(rows),
        "n_events": n_events,
        "n_clear": int(clear.sum()),
        "n_by_terrain": {t: int(sum(r["terrain"] == t for r in rows)) for t in TERRAINS},
        "n_by_group": {g: int(sum(str(r["burst_group"]) == g for r in rows)) for g in BURST_GROUPS},
        "max2_bar_cm": MAX2_BAR_CM,
        "note": "Event-level 5-fold CV. Train |I|>eps; deploy score>0 continue. No burst_index/terrain/depth.",
        "by_fold": [],
    }

    rng_master = np.random.default_rng(int(args.fold_seed) + 17)
    for fold in range(n_folds):
        te_events = [k for k, f in event_fold.items() if f == fold]
        pool = [k for k, f in event_fold.items() if f != fold]
        tr_events, va_events = _train_val_events(pool, rows, rng_master)
        tr_i = _idx_for_events(rows, tr_events)
        va_i = _idx_for_events(rows, va_events) or tr_i
        te_i = _idx_for_events(rows, te_events)
        te_rows = [rows[i] for i in te_i]
        bidx = np.array([int(r["burst_index"]) for r in te_rows], dtype=np.int32)
        a_te = np.array([r["I_500_cm"] for r in te_rows], dtype=np.float64)
        dE100 = hand[np.array(te_i), 3]
        baselines = {
            "always_parent": np.zeros(len(te_rows), dtype=bool),
            "always_recovery": np.ones(len(te_rows), dtype=bool),
            "max2": bidx <= 2,
            "dE100_heuristic": dE100 < -eps,
            "oracle": a_te < -eps,
        }
        fold_blk = {
            "fold": fold,
            "n_train_events": len(tr_events),
            "n_val_events": len(va_events),
            "n_test_events": len(te_events),
            "n_test": len(te_rows),
            "baselines": {},
            "models": {},
        }
        for name, mask in baselines.items():
            fold_blk["baselines"][name] = _eval_policy(te_rows, mask, None, eps, name)
            agg[name]["regret"].append(fold_blk["baselines"][name]["regret"]["mean_regret_cm"])
            agg[name]["cont_prec"].append(
                fold_blk["baselines"][name]["cls"].get("continue_precision", float("nan"))
            )

        clear_tr = clear[np.array(tr_i)]
        clear_va = clear[np.array(va_i)]
        tr_use = [tr_i[j] for j, c in enumerate(clear_tr) if c]
        va_use = [va_i[j] for j, c in enumerate(clear_va) if c] or tr_use
        print(
            f"[7b-full] fold={fold} train_clear={len(tr_use)} val_clear={len(va_use)} "
            f"test={len(te_i)} events_te={len(te_events)}",
            flush=True,
        )

        phi = {
            "A": obs,
            "B": np.concatenate([obs, hand], axis=1).astype(np.float32),
        }
        for kind in ("A", "B"):
            x_tr = phi[kind][np.array(tr_use)]
            y_tr = y_all[np.array(tr_use)]
            x_va = phi[kind][np.array(va_use)]
            y_va = y_all[np.array(va_use)]
            mu = x_tr.mean(0)
            sd = np.clip(x_tr.std(0), 1e-6, None)
            model, val_a = _train_mlp(
                ((x_tr - mu) / sd).astype(np.float32),
                y_tr,
                ((x_va - mu) / sd).astype(np.float32),
                y_va,
                seed=int(args.fold_seed) + fold,
                epochs=int(args.epochs),
                batch=int(args.batch),
                lr=float(args.lr),
                wd=float(args.wd),
                patience=int(args.patience),
            )
            x_te = ((phi[kind][np.array(te_i)] - mu) / sd).astype(np.float32)
            with torch.no_grad():
                score = model(torch.from_numpy(x_te)).numpy()
            pred = score > 0.0
            ev = _eval_policy(te_rows, pred, score, eps, f"mlp_{kind}")
            ev["val_AUROC"] = val_a
            fold_blk["models"][kind] = ev
            agg[kind]["auroc"].append(ev.get("AUROC", float("nan")))
            agg[kind]["regret"].append(ev["regret"]["mean_regret_cm"])
            agg[kind]["cont_prec"].append(ev["cls"].get("continue_precision", float("nan")))
            print(
                f"[7b-full] fold={fold} {kind} AUROC={ev.get('AUROC'):.3f} "
                f"regret={ev['regret']['mean_regret_cm']:.3f} val={val_a:.3f}",
                flush=True,
            )

        obs_mu = obs[np.array(tr_use)].mean(0)
        obs_sd = np.clip(obs[np.array(tr_use)].std(0), 1e-6, None)
        t_mu, t_sd = _traj_stats(traj[np.array(tr_use)], traj_len[np.array(tr_use)])
        def _pack(idx):
            o = ((obs[np.array(idx)] - obs_mu) / obs_sd).astype(np.float32)
            t = _norm_traj(traj[np.array(idx)], traj_len[np.array(idx)], t_mu, t_sd)
            L = traj_len[np.array(idx)]
            return o, t, L
        o_tr, t_tr, l_tr = _pack(tr_use)
        o_va, t_va, l_va = _pack(va_use)
        o_te, t_te, l_te = _pack(te_i)
        model_c, val_c = _train_temporal(
            o_tr, t_tr, l_tr, y_all[np.array(tr_use)],
            o_va, t_va, l_va, y_all[np.array(va_use)],
            seed=int(args.fold_seed) + 100 + fold,
            epochs=int(args.epochs),
            batch=int(args.batch),
            lr=float(args.lr),
            wd=float(args.wd),
            patience=int(args.patience),
        )
        with torch.no_grad():
            score_c = model_c(
                torch.from_numpy(o_te),
                torch.from_numpy(t_te),
                torch.from_numpy(l_te.astype(np.int64)),
            ).numpy()
        pred_c = score_c > 0.0
        ev_c = _eval_policy(te_rows, pred_c, score_c, eps, "temporal_C")
        ev_c["val_AUROC"] = val_c
        fold_blk["models"]["C"] = ev_c
        agg["C"]["auroc"].append(ev_c.get("AUROC", float("nan")))
        agg["C"]["regret"].append(ev_c["regret"]["mean_regret_cm"])
        agg["C"]["cont_prec"].append(ev_c["cls"].get("continue_precision", float("nan")))
        print(
            f"[7b-full] fold={fold} C AUROC={ev_c.get('AUROC'):.3f} "
            f"regret={ev_c['regret']['mean_regret_cm']:.3f} val={val_c:.3f}",
            flush=True,
        )
        payload["by_fold"].append(fold_blk)

    summary = {}
    for k in names:
        blk = {"mean_regret_cm": _meanstd(agg[k]["regret"]), "continue_precision": _meanstd(agg[k]["cont_prec"])}
        if agg[k]["auroc"]:
            blk["AUROC"] = _meanstd(agg[k]["auroc"])
        summary[k] = blk
    payload["pooled_over_folds"] = summary

    learned = ("A", "B", "C")
    best_k, best_r = "A", 1e9
    for k in learned:
        r = summary[k]["mean_regret_cm"]["mean"]
        if r is not None and r < best_r:
            best_k, best_r = k, r
    best_vals = summary[best_k]["mean_regret_cm"]["vals"]
    n_below = int(sum(v < MAX2_BAR_CM for v in best_vals))
    max2_mean = summary["max2"]["mean_regret_cm"]["mean"]
    best_a = (summary[best_k].get("AUROC") or {}).get("mean")
    stably = bool(best_r is not None and best_r < MAX2_BAR_CM and n_below >= 4)
    beats_max2 = bool(best_r is not None and max2_mean is not None and best_r < max2_mean)
    go = bool(stably and beats_max2)
    payload["gates"] = {
        "best_model": best_k,
        "best_mean_regret_cm": best_r,
        "best_fold_regrets_cm": best_vals,
        "n_folds_below_0.671": n_below,
        "best_mean_AUROC": best_a,
        "max2_mean_regret_cm": max2_mean,
        "max2_bar_cm": MAX2_BAR_CM,
        "stably_below_bar": stably,
        "beats_this_max2": beats_max2,
        "GO_7CL": go,
    }
    payload["next"] = (
        "GO 7-CL: learned continuation closed-loop. Still no PPO / no new direction net / no 5° change."
        if go
        else "HOLD 7-CL. 50 Hz process-state did not stably beat Max-2 0.671 cm. Do not PPO."
    )
    print(
        f"[7b-full] BEST {best_k} AUROC={best_a} regret={best_r:.3f} "
        f"max2={max2_mean:.3f} bar=0.671 below={n_below}/5 GO_7CL={go}",
        flush=True,
    )
    (out / "summary.json").write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    (root / "cv_summary.json").write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    print("[7b-full] wrote", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
