#!/usr/bin/env python3
"""P2-R Step 6S — Supervised recovery direction. No PPO. No Isaac.

Train ``o_R (487) → d* (16)`` from Step-5 Probe-B oracle labels.
Split by (terrain, seed, clip) episode. Inference amplitude is unused here;
cosine is the offline metric. Cloned-state utility is a separate eval.
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

TERRAINS = ("plane", "light_rough", "slope", "steps")
IN_DIM = 487
LATENT = 16
Z_SLICE = slice(471, 487)  # last 16 of o_R = sg(z_nom)


def _ep_key(r: dict) -> str:
    return f"{r['terrain']}|{r['seed']}|{r['clip']}"


def _load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for ter in TERRAINS:
        p = root / ter / "states.json"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        for r in json.loads(p.read_text()):
            rows.append(r)
    return rows


def _split_episodes(rows: list[dict], seed: int = 0) -> dict[str, set[str]]:
    """70/15/15 by episode, stratified by terrain. Never random-state split."""
    rng = np.random.default_rng(seed)
    by_ter: dict[str, list[str]] = {t: [] for t in TERRAINS}
    seen: set[str] = set()
    for r in rows:
        k = _ep_key(r)
        if k in seen:
            continue
        seen.add(k)
        by_ter[r["terrain"]].append(k)
    splits = {"train": set(), "val": set(), "test": set()}
    for ter in TERRAINS:
        eps = list(by_ter[ter])
        rng.shuffle(eps)
        n = len(eps)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        n_test = n - n_train - n_val
        splits["train"].update(eps[:n_train])
        splits["val"].update(eps[n_train : n_train + n_val])
        splits["test"].update(eps[n_train + n_val :])
        print(
            f"[6s] {ter}: episodes {n} → train {n_train} val {n_val} test {n_test}",
            flush=True,
        )
    return splits


def _stack(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([r["rec_obs"] for r in rows], dtype=np.float32)
    d = np.asarray([r["d_star"] for r in rows], dtype=np.float32)
    z = np.asarray([r["z_nom"] for r in rows], dtype=np.float32)
    d = d / np.clip(np.linalg.norm(d, axis=-1, keepdims=True), 1e-8, None)
    return x, d, z


def _cos_stats(pred: np.ndarray, dstar: np.ndarray) -> dict:
    pn = pred / np.clip(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-8, None)
    dn = dstar / np.clip(np.linalg.norm(dstar, axis=-1, keepdims=True), 1e-8, None)
    c = (pn * dn).sum(-1)
    return {
        "n": int(c.size),
        "median": float(np.median(c)),
        "mean": float(c.mean()),
        "P_gt_0": float((c > 0).mean()),
        "P_gt_0.25": float((c > 0.25).mean()),
        "P_gt_0.5": float((c > 0.5).mean()),
        "P_gt_0.75": float((c > 0.75).mean()),
    }


class SupervisedRecoveryMLP(nn.Module):
    """487 → 256 → 128 → 16. Not zero-init (supervised, not parent residual)."""

    def __init__(self, in_dim: int = IN_DIM, latent: int = LATENT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _project_unit(raw: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z, dim=-1, eps=1e-8)
    d = raw - (raw * z).sum(-1, keepdim=True) * z
    return F.normalize(d, dim=-1, eps=1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="/data/home/chenxiangyu/robotics/Anybody/results/p2r_step5")
    ap.add_argument("--out", type=str, default="/data/home/chenxiangyu/robotics/Anybody/results/p2r_step6s_small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--amp_coef", type=float, default=0.0)
    ap.add_argument("--tag", type=str, default="6S-full")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(Path(args.data))
    splits = _split_episodes(rows, seed=int(args.seed))
    buckets = {k: [r for r in rows if _ep_key(r) in splits[k]] for k in ("train", "val", "test")}
    for k, rs in buckets.items():
        print(f"[6s] {k} states={len(rs)}", flush=True)

    x_tr, d_tr, z_tr = _stack(buckets["train"])
    mu = x_tr.mean(0)
    sd = x_tr.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)

    def _norm(x: np.ndarray) -> np.ndarray:
        return (x - mu) / sd

    tensors = {}
    for name in ("train", "val", "test"):
        x, d, z = _stack(buckets[name])
        tensors[name] = {
            "x": torch.from_numpy(_norm(x)),
            "d": torch.from_numpy(d),
            "z": torch.from_numpy(z),
            "raw_x": x,
        }

    split_dump = {
        k: sorted(splits[k]) for k in ("train", "val", "test")
    }
    (out / "split.json").write_text(json.dumps({
        "unit": "terrain|seed|clip",
        "n_episodes": {k: len(splits[k]) for k in splits},
        "n_states": {k: len(buckets[k]) for k in buckets},
        "episodes": split_dump,
        "obs_mean": mu.tolist(),
        "obs_std": sd.tolist(),
    }, indent=2))

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    model = SupervisedRecoveryMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))

    n_tr = int(tensors["train"]["x"].shape[0])
    bs = min(int(args.batch), n_tr)
    best_val = -1e9
    best_state = None
    stale = 0
    hist: list[dict] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        perm = torch.randperm(n_tr)
        losses = []
        for i in range(0, n_tr, bs):
            idx = perm[i : i + bs]
            x = tensors["train"]["x"][idx]
            d = tensors["train"]["d"][idx]
            z = tensors["train"]["z"][idx]
            raw = model(x)
            pred = _project_unit(raw, z)
            l_dir = (1.0 - (pred * d).sum(-1)).mean()
            if float(args.amp_coef) > 0:
                l_amp = (raw.norm(dim=-1) - 1.0).pow(2).mean()
                loss = l_dir + float(args.amp_coef) * l_amp
            else:
                loss = l_dir
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            epoch_m = {"epoch": epoch, "loss": float(np.mean(losses))}
            for name in ("train", "val", "test"):
                raw = model(tensors[name]["x"])
                pred = _project_unit(raw, tensors[name]["z"])
                st = _cos_stats(pred.numpy(), tensors[name]["d"].numpy())
                epoch_m[name] = st
            hist.append(epoch_m)
        c_val = epoch_m["val"]["median"]
        if c_val > best_val + 1e-4:
            best_val = c_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"[6s] ep {epoch:3d} loss={epoch_m['loss']:.4f} "
                f"C_tr={epoch_m['train']['median']:.3f} "
                f"C_val={epoch_m['val']['median']:.3f} "
                f"C_te={epoch_m['test']['median']:.3f} "
                f"gap={epoch_m['train']['median'] - epoch_m['test']['median']:.3f} "
                f"P>0.5 tr/te={epoch_m['train']['P_gt_0.5']:.2f}/{epoch_m['test']['P_gt_0.5']:.2f}",
                flush=True,
            )
        if stale >= int(args.patience) and epoch >= 80:
            print(f"[6s] early stop epoch={epoch} best_val_median={best_val:.3f}", flush=True)
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    final = {}
    with torch.no_grad():
        for name in ("train", "val", "test"):
            raw = model(tensors[name]["x"])
            pred = _project_unit(raw, tensors[name]["z"])
            final[name] = _cos_stats(pred.numpy(), tensors[name]["d"].numpy())
            # per-terrain on this split
            by = {}
            for ter in TERRAINS:
                idx = [i for i, r in enumerate(buckets[name]) if r["terrain"] == ter]
                if not idx:
                    continue
                by[ter] = _cos_stats(pred.numpy()[idx], tensors[name]["d"].numpy()[idx])
            final[name]["by_terrain"] = by

    ckpt = {
        "model": best_state,
        "obs_mean": mu,
        "obs_std": sd,
        "in_dim": IN_DIM,
        "latent": LATENT,
        "split_seed": int(args.seed),
        "best_val_median_cos": best_val,
        "amp_coef": float(args.amp_coef),
        "fixed_theta_deg": 5.0,
        "tag": str(args.tag),
        "note": "Probe-B d* labels; clip/event split; cosine-only; no PPO",
    }
    torch.save(ckpt, out / "model_best.pt")
    (out / "history.json").write_text(json.dumps(hist))
    gap = float(final["train"]["median"] - final["test"]["median"])
    payload = {
        "step": str(args.tag),
        "n_states": {k: len(buckets[k]) for k in buckets},
        "n_episodes": {k: len(splits[k]) for k in splits},
        "cosine_best_val_epoch": final,
        "C_train_minus_C_test": gap,
        "early_stop": {"best_val_median": best_val, "patience": int(args.patience)},
        "hparams": {
            "lr": float(args.lr),
            "wd": float(args.wd),
            "batch": int(args.batch),
            "amp_coef": float(args.amp_coef),
            "split_seed": int(args.seed),
        },
        "decision_note": (
            "GO/NO-GO is cloned-state P(I_pred<0) at 1° and 5°, not cosine. "
            f"C_train - C_test = {gap:.3f}."
        ),
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(final, indent=2), flush=True)
    print(f"[6s] C_train={final['train']['median']:.3f} C_test={final['test']['median']:.3f} gap={gap:.3f}", flush=True)
    print(f"[6s] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
