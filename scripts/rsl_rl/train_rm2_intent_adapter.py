#!/usr/bin/env python3
"""Phase R-M2 — Intent-Conditioned Recovery Adapter. No PPO. No Isaac.

Shared MLP: o_rec(487) → 256 → 128 → {direction 16, magnitude 4 bins}.
No task ID, no terrain, no depth. Parent / Mapper-B / g_φ stay frozen.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_DIM = 487
LATENT = 16
THETA_BINS = (2.5, 5.0, 7.5, 10.0)
N_THETA = 4
Z_SLICE = slice(471, 487)


def _project_unit(raw: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z, dim=-1, eps=1e-8)
    d = raw - (raw * z).sum(-1, keepdim=True) * z
    return F.normalize(d, dim=-1, eps=1e-8)


def _cos_stats(pred: np.ndarray, dstar: np.ndarray) -> dict:
    pn = pred / np.clip(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-8, None)
    dn = dstar / np.clip(np.linalg.norm(dstar, axis=-1, keepdims=True), 1e-8, None)
    c = (pn * dn).sum(-1)
    if c.size == 0:
        return {"n": 0}
    return {
        "n": int(c.size),
        "median": float(np.median(c)),
        "mean": float(c.mean()),
        "P_gt_0": float((c > 0).mean()),
        "P_gt_0.25": float((c > 0.25).mean()),
        "P_gt_0.5": float((c > 0.5).mean()),
        "P_gt_0.75": float((c > 0.75).mean()),
    }


def _soft_theta(rows: list[dict], tol_cm: float) -> np.ndarray:
    """Adjacent-bin smoothing when utilities are within tol of the min."""
    y = np.zeros((len(rows), N_THETA), dtype=np.float32)
    for i, r in enumerate(rows):
        sw = r.get("utility_by_angle") or {}
        vals = np.array([float(sw.get(str(th), sw.get(f"{th:.1f}", 1e9))) for th in THETA_BINS], dtype=np.float64)
        finite = np.isfinite(vals)
        if not finite.any():
            idx = int(r.get("theta_idx", 1))
            idx = min(max(idx, 0), N_THETA - 1)
            y[i, idx] = 1.0
            continue
        best = float(np.min(vals[finite]))
        mass = (vals <= best + float(tol_cm)) & finite
        if not mass.any():
            mass[int(np.nanargmin(vals))] = True
        y[i, mass] = 1.0
        y[i] /= max(float(y[i].sum()), 1e-8)
    return y


class IntentConditionedRecoveryAdapter(nn.Module):
    """487 → 256 → 128 → dir(16) + theta(4). Not zero-init."""

    def __init__(self, in_dim: int = IN_DIM, latent: int = LATENT, n_theta: int = N_THETA):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        self.dir_head = nn.Linear(128, latent)
        self.mag_head = nn.Linear(128, n_theta)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.dir_head(h), self.mag_head(h)


def _slice_mask(rows: list[dict], pred) -> np.ndarray:
    return np.array([bool(pred(r)) for r in rows], dtype=bool)


def _angle_metrics(logits: np.ndarray, y_soft: np.ndarray, hard: np.ndarray) -> dict:
    pred = logits.argmax(-1)
    top1 = float((pred == hard).mean()) if hard.size else float("nan")
    within = float((np.abs(pred - hard) <= 1).mean()) if hard.size else float("nan")
    cm = np.zeros((N_THETA, N_THETA), dtype=int)
    for t, p in zip(hard.tolist(), pred.tolist()):
        cm[int(t), int(p)] += 1
    pred_hist = {str(THETA_BINS[k]): int((pred == k).sum()) for k in range(N_THETA)}
    true_hist = {str(THETA_BINS[k]): int((hard == k).sum()) for k in range(N_THETA)}
    return {
        "n": int(hard.size),
        "top1": top1,
        "within_one_bin": within,
        "confusion": cm.tolist(),
        "pred_hist": pred_hist,
        "true_hist": true_hist,
        "mean_true_angle": float(np.mean([THETA_BINS[i] for i in hard])) if hard.size else float("nan"),
        "mean_pred_angle": float(np.mean([THETA_BINS[i] for i in pred])) if pred.size else float("nan"),
    }


def _pack_split(rows: list[dict], mu: np.ndarray, sd: np.ndarray, tol: float, device) -> dict:
    x = np.asarray([r["rec_obs"] for r in rows], dtype=np.float32)
    d = np.asarray([r["d_oracle"] for r in rows], dtype=np.float32)
    z = np.asarray([r["z_nom"] for r in rows], dtype=np.float32)
    y = _soft_theta(rows, tol)
    hard = y.argmax(-1).astype(np.int64)
    xn = (x - mu) / sd
    return {
        "x": torch.from_numpy(xn).to(device),
        "d": torch.from_numpy(d).to(device),
        "z": torch.from_numpy(z).to(device),
        "y": torch.from_numpy(y).to(device),
        "hard": torch.from_numpy(hard).to(device),
        "rows": rows,
    }


def _balanced_idx(rows: list[dict], n: int, rng: np.random.Generator) -> np.ndarray:
    loco = np.array([i for i, r in enumerate(rows) if r["task_source"] == "loco"], dtype=np.int64)
    stoop = np.array([i for i, r in enumerate(rows) if r["task_source"] == "stoop"], dtype=np.int64)
    n_l = n // 2
    n_s = n - n_l
    if loco.size == 0:
        return rng.choice(stoop, size=n, replace=True)
    if stoop.size == 0:
        return rng.choice(loco, size=n, replace=True)
    a = rng.choice(loco, size=n_l, replace=True)
    b = rng.choice(stoop, size=n_s, replace=True)
    idx = np.concatenate([a, b])
    rng.shuffle(idx)
    return idx


def _eval_group(pred: np.ndarray, d: np.ndarray, logits: np.ndarray, y: np.ndarray, hard: np.ndarray, rows: list[dict]) -> dict:
    out = {
        "all": {"dir": _cos_stats(pred, d), "angle": _angle_metrics(logits, y, hard)},
    }
    groups = {
        "loco": lambda r: r["task_source"] == "loco",
        "stoop": lambda r: r["task_source"] == "stoop",
        "stoop_s_first": lambda r: r["task_source"] == "stoop" and r["trigger_channel"] == "S",
        "stoop_e_first": lambda r: r["task_source"] == "stoop" and r["trigger_channel"] == "E",
    }
    for name, fn in groups.items():
        m = _slice_mask(rows, fn)
        if not m.any():
            out[name] = {"dir": {"n": 0}, "angle": {"n": 0}}
            continue
        out[name] = {
            "dir": _cos_stats(pred[m], d[m]),
            "angle": _angle_metrics(logits[m], y[m], hard[m]),
        }
    return out


@torch.no_grad()
def _forward_numpy(model, pack) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw, logits = model(pack["x"])
    pred = _project_unit(raw, pack["z"])
    return pred.cpu().numpy(), logits.cpu().numpy(), pack["d"].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=str,
        default="/data/home/chenxiangyu/robotics/Anybody/results/rm_intent_conditioned_recovery/data/merged",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="/data/home/chenxiangyu/robotics/Anybody/results/rm_intent_conditioned_recovery/checkpoints",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--lambda_theta", type=float, default=0.25)
    ap.add_argument("--tol_cm", type=float, default=0.15)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    buckets = {k: json.loads((data / f"{k}.json").read_text()) for k in ("train", "val", "test")}
    for k, rs in buckets.items():
        print(f"[rm2-train] {k} n={len(rs)}", flush=True)

    x_tr = np.asarray([r["rec_obs"] for r in buckets["train"]], dtype=np.float32)
    mu = x_tr.mean(0)
    sd = x_tr.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    print(f"[rm2-train] device={device}", flush=True)
    packs = {k: _pack_split(v, mu, sd, float(args.tol_cm), device) for k, v in buckets.items()}

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    model = IntentConditionedRecoveryAdapter().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))

    n_tr = len(buckets["train"])
    bs = min(int(args.batch), max(n_tr, 2))
    steps_per = max(1, int(math.ceil(n_tr / bs)))
    best_score = -1e9
    best_state = None
    stale = 0
    hist: list[dict] = []

    def _val_score(ev: dict) -> float:
        c_l = float((ev.get("loco") or {}).get("dir", {}).get("median") or 0.0)
        c_s = float((ev.get("stoop") or {}).get("dir", {}).get("median") or 0.0)
        a_l = float((ev.get("loco") or {}).get("angle", {}).get("top1") or 0.0)
        a_s = float((ev.get("stoop") or {}).get("angle", {}).get("top1") or 0.0)
        return 0.5 * (c_l + c_s) + 0.15 * (a_l + a_s)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for _ in range(steps_per):
            idx = _balanced_idx(buckets["train"], bs, rng)
            tidx = torch.from_numpy(idx).to(device=device, dtype=torch.long)
            x = packs["train"]["x"][tidx]
            d = packs["train"]["d"][tidx]
            z = packs["train"]["z"][tidx]
            y = packs["train"]["y"][tidx]
            raw, logits = model(x)
            pred = _project_unit(raw, z)
            l_dir = (1.0 - (pred * d).sum(-1)).mean()
            logp = F.log_softmax(logits, dim=-1)
            l_th = -(y * logp).sum(-1).mean()
            loss = l_dir + float(args.lambda_theta) * l_th
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        epoch_m = {"epoch": epoch, "loss": float(np.mean(losses))}
        with torch.no_grad():
            for name in ("train", "val", "test"):
                pred, logits, dnp = _forward_numpy(model, packs[name])
                ynp = packs[name]["y"].cpu().numpy()
                hnp = packs[name]["hard"].cpu().numpy()
                epoch_m[name] = _eval_group(pred, dnp, logits, ynp, hnp, buckets[name])
        hist.append({"epoch": epoch, "loss": epoch_m["loss"]})
        score = _val_score(epoch_m["val"])
        if score > best_score + 1e-4:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 20 == 0 or epoch == 1:
            vd = epoch_m["val"]
            print(
                f"[rm2-train] ep {epoch:3d} loss={epoch_m['loss']:.4f} "
                f"C_loco={vd['loco']['dir'].get('median', float('nan')):.3f} "
                f"C_stoop={vd['stoop']['dir'].get('median', float('nan')):.3f} "
                f"acc_loco={vd['loco']['angle'].get('top1', float('nan')):.2f} "
                f"acc_stoop={vd['stoop']['angle'].get('top1', float('nan')):.2f} "
                f"score={score:.3f} stale={stale}",
                flush=True,
            )
        if stale >= int(args.patience) and epoch >= 80:
            print(f"[rm2-train] early stop epoch={epoch} best_score={best_score:.3f}", flush=True)
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    final = {}
    with torch.no_grad():
        for name in ("train", "val", "test"):
            pred, logits, dnp = _forward_numpy(model, packs[name])
            ynp = packs[name]["y"].cpu().numpy()
            hnp = packs[name]["hard"].cpu().numpy()
            final[name] = _eval_group(pred, dnp, logits, ynp, hnp, buckets[name])

    ckpt = {
        "model": best_state,
        "obs_mean": mu,
        "obs_std": sd,
        "in_dim": IN_DIM,
        "latent": LATENT,
        "theta_bins": list(THETA_BINS),
        "lambda_theta": float(args.lambda_theta),
        "tol_cm": float(args.tol_cm),
        "split_seed": int(args.seed),
        "best_val_score": best_score,
        "tag": "rm2-intent-conditioned-recovery",
        "note": "No PPO. No task ID. Direction cosine + discrete magnitude. Parent frozen.",
    }
    torch.save(ckpt, out / "model_best.pt")
    (out / "history.json").write_text(json.dumps(hist))
    payload = {
        "step": "rm2-intent-conditioned-recovery",
        "no_ppo": True,
        "n_states": {k: len(v) for k, v in buckets.items()},
        "metrics": final,
        "hparams": {
            "lr": float(args.lr),
            "wd": float(args.wd),
            "batch": int(args.batch),
            "lambda_theta": float(args.lambda_theta),
            "tol_cm": float(args.tol_cm),
            "seed": int(args.seed),
        },
        "early_stop": {"best_val_score": best_score, "patience": int(args.patience)},
        "decision_note": "GO/NO-GO is cloned-state P(I<0) / median I, not cosine.",
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: {g: v["dir"] for g, v in final[k].items() if "dir" in v} for k in final}, indent=2), flush=True)
    print(f"[rm2-train] wrote {out / 'model_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
