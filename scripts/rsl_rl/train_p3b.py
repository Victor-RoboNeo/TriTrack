#!/usr/bin/env python3
"""P3-B: pack dataset + train B0/B1/B2. No Isaac. λ_soft=1 frozen. Slip held out."""
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
    lowrank_to_C,
    losses,
    random_tangent,
    sample_ucr_dirs,
    soft_P,
)

SEEN = ("plane", "slope", "slope_down", "light_rough", "steps")
HELD = ("slip",)
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


def _load_terrain(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        "x": z["x"].astype(np.float32),
        "z0": z["z0"].astype(np.float32),
        "C": z["C"].astype(np.float32),
        "episode_id": np.asarray(z["episode_id"]).astype(str),
        "parent_episode_id": np.asarray(z["parent_episode_id"]).astype(str),
        "state_kind": np.asarray(z["state_kind"]).astype(str),
        "window": np.asarray(z["window"]).astype(str),
        "terrain": np.asarray(z["terrain"]).astype(str),
        "t": z["t"].astype(np.int32),
        "e0": z["e0"].astype(np.float32),
    }


def _concat(parts: list[dict]) -> dict:
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def _split_episodes(eps: np.ndarray, seed: int = 2026) -> dict[str, str]:
    uniq = sorted(set(eps.tolist()))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(0.80 * n))
    n_va = int(round(0.10 * n))
    out = {}
    for i, k in enumerate(uniq):
        if i < n_tr:
            out[k] = "train"
        elif i < n_tr + n_va:
            out[k] = "val"
        else:
            out[k] = "test_seen"
    return out


def _subsample_5050(kind: np.ndarray, n_target: int, rng: np.random.RandomState) -> np.ndarray:
    nom = np.where(kind == "nominal")[0]
    off = np.where(kind != "nominal")[0]
    n_each = n_target // 2
    take = []
    if nom.size and n_each:
        take.append(rng.choice(nom, size=min(n_each, nom.size), replace=False))
    if off.size and n_each:
        take.append(rng.choice(off, size=min(n_each, off.size), replace=False))
    if not take:
        return np.arange(min(n_target, kind.size))
    idx = np.concatenate(take)
    if idx.size < n_target:
        rest = np.setdiff1d(np.arange(kind.size), idx)
        if rest.size:
            extra = rng.choice(rest, size=min(n_target - idx.size, rest.size), replace=False)
            idx = np.concatenate([idx, extra])
    return np.sort(idx)


def _slice(d: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in d.items()}


def pack(root: Path) -> dict:
    ddir = root / "p3b_learned_projector" / "dataset_b"
    seen_parts = []
    for t in SEEN:
        p = ddir / f"{t}.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        seen_parts.append(_load_terrain(p))
    seen = _concat(seen_parts)
    split_map = _split_episodes(seen["parent_episode_id"], seed=2026)
    split = np.asarray([split_map[e] for e in seen["parent_episode_id"]])
    rng = np.random.RandomState(2026)
    targets = {"train": 4800, "val": 600, "test_seen": 600}
    packed = {}
    counts = {}
    for sp, n_t in targets.items():
        idx = np.where(split == sp)[0]
        sub = _subsample_5050(seen["state_kind"][idx], n_t, rng)
        idx = idx[sub]
        packed[sp] = _slice(seen, idx)
        kinds, kc = np.unique(packed[sp]["state_kind"], return_counts=True)
        counts[sp] = {str(k): int(c) for k, c in zip(kinds, kc)}
        counts[sp]["n"] = int(idx.size)
    slip_p = ddir / "slip.npz"
    if slip_p.exists():
        slip = _load_terrain(slip_p)
        if slip["x"].shape[0] > 600:
            idx = rng.choice(slip["x"].shape[0], 600, replace=False)
            idx.sort()
            slip = _slice(slip, idx)
        packed["test_slip"] = slip
        kinds, kc = np.unique(slip["state_kind"], return_counts=True)
        counts["test_slip"] = {str(k): int(c) for k, c in zip(kinds, kc)}
        counts["test_slip"]["n"] = int(slip["x"].shape[0])
    out = ddir / "packed.npz"
    save = {}
    for sp, d in packed.items():
        for k, v in d.items():
            save[f"{sp}/{k}"] = v
    np.savez_compressed(out, **save)
    train_eps = set(packed["train"]["parent_episode_id"].tolist())
    ucr = _load_ucr_pool(root, train_eps)
    np.save(ddir / "ucr_pool.npy", ucr)
    mean = packed["train"]["x"].mean(axis=0)
    std = packed["train"]["x"].std(axis=0).clip(min=1e-6)
    np.savez(ddir / "x_norm.npz", mean=mean, std=std, clip=np.asarray(10.0))
    meta = {
        "counts": counts,
        "x_dim": int(packed["train"]["x"].shape[1]),
        "n_ucr_pool": int(ucr.shape[0]),
        "lambda_soft": 1.0,
        "mix": "50/50 nominal vs recovery+off-manifold (best effort)",
        "slip_in_train": False,
        "no_terrain_in_model": True,
    }
    (ddir / "pack_meta.json").write_text(json.dumps(_sanitize(meta), indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return packed


def _load_ucr_pool(root: Path, train_eps: set[str]) -> np.ndarray:
    dirs = []
    for t in SEEN:
        jp = root / "p3a_oracle_projector" / f"{t}.json"
        dp = root / "p3a_oracle_projector" / f"{t}_dstars.npz"
        if not jp.exists() or not dp.exists():
            continue
        rows = json.loads(jp.read_text()).get("rows") or []
        a_map = {}
        for r in rows:
            a = ((r.get("methods") or {}).get("A1_full") or {}).get("A")
            a_map[(str(r["episode_id"]), int(r["t"]))] = float(a) if a is not None else 0.0
        z = np.load(dp, allow_pickle=True)
        for i, ep in enumerate(z["episode_id"]):
            ep = str(ep)
            if ep not in train_eps:
                continue
            if a_map.get((ep, int(z["t"][i])), 0.0) > 0.0:
                dirs.append(np.asarray(z["d"][i], dtype=np.float32))
    bpath = Path("/data/home/chenxiangyu/robotics/Anybody/results/intent_free_space/basis/B_ucr.npz")
    if bpath.exists():
        B = np.load(bpath)["B"][:, :4].T.astype(np.float32)
        dirs.extend(list(B))
    if not dirs:
        rng = np.random.RandomState(2026)
        v = rng.randn(32, Z_DIM).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        return v
    return np.stack(dirs, axis=0)


def _load_packed(root: Path) -> dict:
    z = np.load(root / "p3b_learned_projector" / "dataset_b" / "packed.npz", allow_pickle=True)
    splits = {}
    for k in z.files:
        sp, name = k.split("/", 1)
        splits.setdefault(sp, {})[name] = z[k]
    return splits


def _norm_x(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip((x - mean) / std, -10.0, 10.0).astype(np.float32)


def _lr(epoch: int, total: int, warmup: int, lr: float, min_lr: float) -> float:
    if epoch < warmup:
        return lr * float(epoch + 1) / float(max(warmup, 1))
    t = (epoch - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * t))


def _forward(net, xb, zb, mode: str, rank: int):
    raw = net(xb)
    if mode == "lowrank":
        return lowrank_to_C(raw, zb, rank)
    return chol_to_C(raw, zb)


@torch.no_grad()
def _eval_split(net, x, z, C, pool, device, mode, rank, bs=512):
    net.eval()
    tot = {"lf": 0.0, "lq": 0.0, "lp": 0.0, "n": 0, "cos_rand": [], "cos_ucr": [], "sp": []}
    g = torch.Generator(device=device)
    g.manual_seed(0)
    for i in range(0, x.shape[0], bs):
        xb = torch.as_tensor(x[i : i + bs], device=device)
        zb = torch.as_tensor(z[i : i + bs], device=device)
        cb = torch.as_tensor(C[i : i + bs], device=device)
        chat = _forward(net, xb, zb, mode, rank)
        vr = random_tangent(zb, 8, g)
        vu = sample_ucr_dirs(zb, pool, 4, g)
        lf, lq, lp, _ = losses(chat, cb, zb, vr, vu, lam=1.0)
        n = xb.shape[0]
        tot["lf"] += float(lf) * n
        tot["lq"] += float(lq) * n
        tot["lp"] += float(lp) * n
        tot["n"] += n
        Ph = soft_P(chat, zb, 1.0)
        Pg = soft_P(cb, zb, 1.0)
        for name, v in (("cos_rand", vr), ("cos_ucr", vu)):
            ph = torch.einsum("bij,bdj->bdi", Ph, v)
            pg = torch.einsum("bij,bdj->bdi", Pg, v)
            c = (ph * pg).sum(-1) / (ph.norm(dim=-1) * pg.norm(dim=-1) + 1e-8)
            tot[name].append(c.detach().cpu().numpy())
        sh = torch.einsum("bdi,bij,bdj->bd", torch.cat([vr, vu], 1), chat, torch.cat([vr, vu], 1))
        sg = torch.einsum("bdi,bij,bdj->bd", torch.cat([vr, vu], 1), cb, torch.cat([vr, vu], 1))
        tot["sp"].append(np.stack([sg.cpu().numpy(), sh.cpu().numpy()], axis=0))
    n = max(tot["n"], 1)
    lf, lq, lp = tot["lf"] / n, tot["lq"] / n, tot["lp"] / n
    cr = np.concatenate(tot["cos_rand"]).mean() if tot["cos_rand"] else None
    cu = np.concatenate(tot["cos_ucr"]).mean() if tot["cos_ucr"] else None
    sp = None
    if tot["sp"]:
        sg = np.concatenate([a[0] for a in tot["sp"]], axis=0).reshape(-1)
        sh = np.concatenate([a[1] for a in tot["sp"]], axis=0).reshape(-1)
        if sg.size > 5:
            rs = sg.argsort().argsort().astype(np.float64)
            rh = sh.argsort().argsort().astype(np.float64)
            rs = rs - rs.mean()
            rh = rh - rh.mean()
            den = float(np.sqrt((rs * rs).sum() * (rh * rh).sum()) + 1e-12)
            sp = float((rs * rh).sum() / den)
    return {
        "L_F": lf,
        "L_Q": lq,
        "L_P": lp,
        "L": 0.5 * lf + 1.0 * lq + 2.0 * lp,
        "cos_rand": cr,
        "cos_ucr": cu,
        "spearman": sp,
        "n": tot["n"],
    }


def train_one(name, mode, rank, in_use, data, mean, std, pool, device, seed, out_dir: Path, args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_tr = _norm_x(data["train"]["x"], mean, std)
    if in_use == Z_DIM:
        x_tr = x_tr[:, :Z_DIM]
        mean_u, std_u = mean[:Z_DIM], std[:Z_DIM]
        in_dim = Z_DIM
    else:
        mean_u, std_u = mean, std
        in_dim = int(x_tr.shape[1])
    out_dim = 16 * rank if mode == "lowrank" else N_TRIL
    net = IntentMetricMLP(in_dim, out_dim=out_dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = TensorDataset(
        torch.as_tensor(x_tr),
        torch.as_tensor(data["train"]["z0"]),
        torch.as_tensor(data["train"]["C"]),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    best_lp = 1e9
    best_ep = -1
    patience = 0
    hist = []
    ckpt_p = out_dir / "checkpoints" / f"{name}_s{seed}.pt"
    ckpt_p.parent.mkdir(parents=True, exist_ok=True)

    def prep_split(sp):
        xn = _norm_x(data[sp]["x"], mean, std)
        if in_use == Z_DIM:
            xn = xn[:, :Z_DIM]
        return xn, data[sp]["z0"], data[sp]["C"]

    xva, zva, cva = prep_split("val")
    for epoch in range(args.epochs):
        lr = _lr(epoch, args.epochs, args.warmup_epochs, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        net.train()
        tr = {"lf": 0.0, "lq": 0.0, "lp": 0.0, "n": 0}
        ggen = torch.Generator(device=device)
        ggen.manual_seed(seed * 1000 + epoch)
        for xb, zb, cb in loader:
            xb = xb.to(device)
            zb = zb.to(device)
            cb = cb.to(device)
            opt.zero_grad(set_to_none=True)
            ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16)
            with ctx:
                chat = _forward(net, xb, zb, mode, rank)
                vr = random_tangent(zb, 8, ggen)
                vu = sample_ucr_dirs(zb, pool, 4, ggen)
                lf, lq, lp, loss = losses(chat, cb, zb, vr, vu, lam=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()
            n = xb.shape[0]
            tr["lf"] += float(lf.detach()) * n
            tr["lq"] += float(lq.detach()) * n
            tr["lp"] += float(lp.detach()) * n
            tr["n"] += n
        n = max(tr["n"], 1)
        val = _eval_split(net, xva, zva, cva, pool, device, mode, rank)
        row = {
            "model": name,
            "seed": seed,
            "epoch": epoch,
            "lr": lr,
            "train_L_F": tr["lf"] / n,
            "train_L_Q": tr["lq"] / n,
            "train_L_P": tr["lp"] / n,
            "train_L": 0.5 * tr["lf"] / n + tr["lq"] / n + 2.0 * tr["lp"] / n,
            "val_L_F": val["L_F"],
            "val_L_Q": val["L_Q"],
            "val_L_P": val["L_P"],
            "val_L": val["L"],
            "val_cos_ucr": val["cos_ucr"],
            "val_spearman": val["spearman"],
        }
        hist.append(row)
        lpv = val["L_P"]
        improved = lpv < best_lp - 1e-6
        if improved:
            best_lp = lpv
            best_ep = epoch
            patience = 0
            torch.save(
                {
                    "model": name,
                    "mode": mode,
                    "rank": rank,
                    "in_dim": in_dim,
                    "out_dim": out_dim,
                    "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                    "x_mean": mean.astype(np.float32),
                    "x_std": std.astype(np.float32),
                    "seed": seed,
                    "best_epoch": epoch,
                    "val_L_P": best_lp,
                    "lambda_soft": 1.0,
                },
                ckpt_p,
            )
        else:
            patience += 1
        if epoch % 5 == 0 or improved:
            print(
                f"[{name} s{seed}] ep={epoch:03d} trP={row['train_L_P']:.4f} "
                f"valP={lpv:.4f} best={best_lp:.4f}@{best_ep} cosU={val['cos_ucr']}",
                flush=True,
            )
        if patience >= args.patience:
            print(f"[{name} s{seed}] early stop ep={epoch} best_L_P={best_lp:.4f}", flush=True)
            break
    ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    metrics = {"model": name, "seed": seed, "best_epoch": best_ep, "best_val_L_P": best_lp, "ckpt": str(ckpt_p)}
    for sp in ("val", "test_seen", "test_slip"):
        if sp not in data:
            continue
        xn, zs, cs = prep_split(sp)
        metrics[sp] = _eval_split(net, xn, zs, cs, pool, device, mode, rank)
        by = {}
        kind = data[sp]["state_kind"]
        for tag in ("nominal", "recovery", "perturbed"):
            idx = np.where(kind == tag)[0]
            if idx.size < 8:
                continue
            by[tag] = _eval_split(net, xn[idx], zs[idx], cs[idx], pool, device, mode, rank)
        off = np.where(kind != "nominal")[0]
        if off.size >= 8:
            by["off_manifold"] = _eval_split(net, xn[off], zs[off], cs[off], pool, device, mode, rank)
        metrics[f"{sp}_by_kind"] = by
    return hist, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/p3_intent_projected_adaptation")
    ap.add_argument("--pack_only", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seeds", type=str, default="2026,2027,2028")
    args = ap.parse_args()
    root = Path(args.root)
    packed_p = root / "p3b_learned_projector" / "dataset_b" / "packed.npz"
    if not packed_p.exists():
        pack(root)
    if args.pack_only:
        return
    data = _load_packed(root)
    norm = np.load(root / "p3b_learned_projector" / "dataset_b" / "x_norm.npz")
    mean, std = norm["mean"], norm["std"]
    pool_np = np.load(root / "p3b_learned_projector" / "dataset_b" / "ucr_pool.npy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = torch.as_tensor(pool_np, device=device, dtype=torch.float32)
    out_dir = root / "p3b_learned_projector"
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    models = [
        ("B0", "chol", 0, Z_DIM),
        ("B1", "chol", 0, -1),
        ("B2", "lowrank", 8, -1),
    ]
    curves_p = out_dir / "training_curves.csv"
    test_p = out_dir / "test_metrics.csv"
    all_hist = []
    all_metrics = []
    selected = {}
    for name, mode, rank, in_use in models:
        best = None
        for seed in seeds:
            hist, met = train_one(name, mode, rank, in_use, data, mean, std, pool, device, seed, out_dir, args)
            all_hist.extend(hist)
            all_metrics.append(met)
            if best is None or met["best_val_L_P"] < best["best_val_L_P"]:
                best = met
        selected[name] = {
            "ckpt": best["ckpt"],
            "seed": best["seed"],
            "val_L_P": best["best_val_L_P"],
            "best_epoch": best["best_epoch"],
        }
        print(f"[p3b] select {name} seed={best['seed']} val_L_P={best['best_val_L_P']:.4f}", flush=True)
    with curves_p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_hist[0].keys()))
        w.writeheader()
        w.writerows(all_hist)
    with test_p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "split", "kind", "L_F", "L_Q", "L_P", "L", "cos_rand", "cos_ucr", "spearman", "n"])
        for met in all_metrics:
            for sp in ("val", "test_seen", "test_slip"):
                if sp not in met:
                    continue
                s = met[sp]
                w.writerow([met["model"], met["seed"], sp, "all", s["L_F"], s["L_Q"], s["L_P"], s["L"],
                            s["cos_rand"], s["cos_ucr"], s["spearman"], s["n"]])
                for kind, s2 in (met.get(f"{sp}_by_kind") or {}).items():
                    w.writerow([met["model"], met["seed"], sp, kind, s2["L_F"], s2["L_Q"], s2["L_P"], s2["L"],
                                s2["cos_rand"], s2["cos_ucr"], s2["spearman"], s2["n"]])
    (out_dir / "selected.json").write_text(json.dumps(_sanitize(selected), indent=2), encoding="utf-8")
    (out_dir / "train_metrics.json").write_text(json.dumps(_sanitize(all_metrics), indent=2), encoding="utf-8")
    print("[p3b] train done", json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()
