#!/usr/bin/env python3
"""P3-B2-1 conservative intent shield. Init from B1. Same P3-B split. No Isaac."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from p3b_model import (
    Z_DIM,
    IntentMetricMLP,
    chol_to_C,
    conservative_losses,
    random_tangent,
    sample_ucr_dirs,
    soft_P,
)

P3B = Path("/data/home/chenxiangyu/robotics/Anybody/results/p3_intent_projected_adaptation")
B1_CKPT = P3B / "p3b_learned_projector" / "checkpoints" / "B1_s2028.pt"
N_TRIL = Z_DIM * (Z_DIM + 1) // 2


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


def _load_packed():
    z = np.load(P3B / "p3b_learned_projector" / "dataset_b" / "packed.npz", allow_pickle=True)
    splits = {}
    for k in z.files:
        sp, name = k.split("/", 1)
        splits.setdefault(sp, {})[name] = z[k]
    return splits


def _norm_x(x, mean, std):
    return np.clip((x - mean) / std, -10.0, 10.0).astype(np.float32)


def _lr(epoch, total, warmup, lr, min_lr):
    if epoch < warmup:
        return lr * float(epoch + 1) / float(max(warmup, 1))
    t = (epoch - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * t))


def _dirs(z, pool, g):
    vr = random_tangent(z, 8, g)
    vu = sample_ucr_dirs(z, pool, 8, g, noise_std=0.05)
    return torch.cat([vr, vu], dim=1)


@torch.no_grad()
def _eval_split(net, x, z, C, pool, device, bs=512):
    net.eval()
    tot = {"lf": 0.0, "lq": 0.0, "lp": 0.0, "lk": 0.0, "n": 0, "cos": [], "leak": []}
    g = torch.Generator(device=device)
    g.manual_seed(0)
    for i in range(0, x.shape[0], bs):
        xb = torch.as_tensor(x[i : i + bs], device=device)
        zb = torch.as_tensor(z[i : i + bs], device=device)
        cb = torch.as_tensor(C[i : i + bs], device=device)
        chat = chol_to_C(net(xb), zb)
        v = _dirs(zb, pool, g)
        lf, lq, lp, lk, _ = conservative_losses(chat, cb, zb, v)
        n = xb.shape[0]
        tot["lf"] += float(lf) * n
        tot["lq"] += float(lq) * n
        tot["lp"] += float(lp) * n
        tot["lk"] += float(lk) * n
        tot["n"] += n
        Ph = soft_P(chat, zb, 1.0)
        Pg = soft_P(cb, zb, 1.0)
        ph = torch.einsum("bij,bdj->bdi", Ph, v)
        pg = torch.einsum("bij,bdj->bdi", Pg, v)
        c = (ph * pg).sum(-1) / (ph.norm(dim=-1) * pg.norm(dim=-1) + 1e-8)
        tot["cos"].append(c.cpu().numpy())
        dn = torch.nn.functional.normalize(ph, dim=-1, eps=1e-8)
        leak = torch.einsum("bdi,bij,bdj->bd", dn, cb, dn)
        tot["leak"].append(leak.cpu().numpy())
    n = max(tot["n"], 1)
    return {
        "L_F": tot["lf"] / n,
        "L_Q": tot["lq"] / n,
        "L_P": tot["lp"] / n,
        "L_leak": tot["lk"] / n,
        "L": 0.25 * tot["lf"] / n + tot["lq"] / n + 2.0 * tot["lp"] / n + 3.0 * tot["lk"] / n,
        "cos": float(np.concatenate(tot["cos"]).mean()) if tot["cos"] else None,
        "oracle_leak": float(np.median(np.concatenate(tot["leak"]))) if tot["leak"] else None,
        "n": tot["n"],
    }


def train_one(data, mean, std, pool, device, seed, out_dir: Path, args, init_ckpt: Path):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_tr = _norm_x(data["train"]["x"], mean, std)
    in_dim = int(x_tr.shape[1])
    net = IntentMetricMLP(in_dim, out_dim=N_TRIL).to(device)
    ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = net.load_state_dict(ck["state_dict"], strict=False)
    print(f"[b21] init from {init_ckpt} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x_tr), torch.as_tensor(data["train"]["z0"]), torch.as_tensor(data["train"]["C"])),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    best_lk = 1e9
    best_ep = -1
    patience = 0
    hist = []
    ckpt_p = out_dir / "checkpoints" / f"B21_s{seed}.pt"
    ckpt_p.parent.mkdir(parents=True, exist_ok=True)
    xva = _norm_x(data["val"]["x"], mean, std)
    for epoch in range(args.epochs):
        lr = _lr(epoch, args.epochs, args.warmup_epochs, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        net.train()
        tr = {"lf": 0.0, "lq": 0.0, "lp": 0.0, "lk": 0.0, "n": 0}
        ggen = torch.Generator(device=device)
        ggen.manual_seed(seed * 1000 + epoch)
        for xb, zb, cb in loader:
            xb, zb, cb = xb.to(device), zb.to(device), cb.to(device)
            opt.zero_grad(set_to_none=True)
            ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16)
            with ctx:
                chat = chol_to_C(net(xb), zb)
                v = _dirs(zb, pool, ggen)
                lf, lq, lp, lk, loss = conservative_losses(chat, cb, zb, v)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()
            n = xb.shape[0]
            tr["lf"] += float(lf.detach()) * n
            tr["lq"] += float(lq.detach()) * n
            tr["lp"] += float(lp.detach()) * n
            tr["lk"] += float(lk.detach()) * n
            tr["n"] += n
        n = max(tr["n"], 1)
        val = _eval_split(net, xva, data["val"]["z0"], data["val"]["C"], pool, device)
        row = {
            "seed": seed,
            "epoch": epoch,
            "lr": lr,
            "train_L_F": tr["lf"] / n,
            "train_L_Q": tr["lq"] / n,
            "train_L_P": tr["lp"] / n,
            "train_L_leak": tr["lk"] / n,
            "val_L_F": val["L_F"],
            "val_L_Q": val["L_Q"],
            "val_L_P": val["L_P"],
            "val_L_leak": val["L_leak"],
            "val_cos": val["cos"],
            "val_oracle_leak": val["oracle_leak"],
        }
        hist.append(row)
        improved = val["L_leak"] < best_lk - 1e-8
        if improved:
            best_lk = val["L_leak"]
            best_ep = epoch
            patience = 0
            torch.save(
                {
                    "model": "B21",
                    "mode": "chol",
                    "in_dim": in_dim,
                    "out_dim": N_TRIL,
                    "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                    "x_mean": mean.astype(np.float32),
                    "x_std": std.astype(np.float32),
                    "seed": seed,
                    "best_epoch": epoch,
                    "val_L_leak": best_lk,
                    "lambda_soft": 1.0,
                    "init_from": str(init_ckpt),
                },
                ckpt_p,
            )
        else:
            patience += 1
        if epoch % 5 == 0 or improved:
            print(
                f"[B21 s{seed}] ep={epoch:03d} trLeak={row['train_L_leak']:.5f} "
                f"valLeak={val['L_leak']:.5f} best={best_lk:.5f}@{best_ep} cos={val['cos']}",
                flush=True,
            )
        if patience >= args.patience:
            print(f"[B21 s{seed}] early stop ep={epoch} best_leak={best_lk:.5f}", flush=True)
            break
    ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    metrics = {"model": "B21", "seed": seed, "best_epoch": best_ep, "best_val_L_leak": best_lk, "ckpt": str(ckpt_p)}
    for sp in ("val", "test_seen", "test_slip"):
        if sp not in data:
            continue
        xn = _norm_x(data[sp]["x"], mean, std)
        metrics[sp] = _eval_split(net, xn, data[sp]["z0"], data[sp]["C"], pool, device)
    return hist, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/intent_autonomous_execution/p3b2_intent_shield")
    ap.add_argument("--b1_ckpt", type=str, default=str(B1_CKPT))
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seeds", type=str, default="2026,2027,2028")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = _load_packed()
    norm = np.load(P3B / "p3b_learned_projector" / "dataset_b" / "x_norm.npz")
    mean, std = norm["mean"], norm["std"]
    pool = torch.as_tensor(np.load(P3B / "p3b_learned_projector" / "dataset_b" / "ucr_pool.npy"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = pool.to(device)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    all_hist, all_met, best = [], [], None
    for seed in seeds:
        hist, met = train_one(data, mean, std, pool, device, seed, out, args, Path(args.b1_ckpt))
        all_hist.extend(hist)
        all_met.append(met)
        if best is None or met["best_val_L_leak"] < best["best_val_L_leak"]:
            best = met
    selected = {"B21": {"ckpt": best["ckpt"], "seed": best["seed"], "val_L_leak": best["best_val_L_leak"],
                        "best_epoch": best["best_epoch"]}}
    with (out / "training_curves.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_hist[0].keys()))
        w.writeheader()
        w.writerows(all_hist)
    (out / "selected.json").write_text(json.dumps(_sanitize(selected), indent=2), encoding="utf-8")
    (out / "train_metrics.json").write_text(json.dumps(_sanitize(all_met), indent=2), encoding="utf-8")
    print("[b21] selected", json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()
