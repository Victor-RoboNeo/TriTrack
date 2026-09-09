#!/usr/bin/env python3
"""GPU4: tiny residual MLP, 3 seeds × scale-aug on/off. No DDP. No task ID."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h2r.constants import OUT
from h2r.mlp import ResidualMLP


def pairwise_loss(pred, gt):
    def pts(u):
        h, l, r = u[:, 0:3], u[:, 0:3] + u[:, 3:6], u[:, 0:3] + u[:, 6:9]
        return h, l, r

    ph, pl, pr = pts(pred)
    gh, gl, gr = pts(gt)
    d_pred = torch.stack(
        [(pl - ph).norm(dim=-1), (pr - ph).norm(dim=-1), (pl - pr).norm(dim=-1)], dim=-1
    )
    d_gt = torch.stack(
        [(gl - gh).norm(dim=-1), (gr - gh).norm(dim=-1), (gl - gr).norm(dim=-1)], dim=-1
    )
    l_pair = F.mse_loss(d_pred, d_gt)
    v_pred = torch.cat([pl - ph, pr - ph, pr - pl], dim=-1)
    v_gt = torch.cat([gl - gh, gr - gh, gr - gl], dim=-1)
    l_dist = F.mse_loss(v_pred, v_gt)
    return l_pair, l_dist


def load_split(npz_path: Path):
    d = np.load(str(npz_path), allow_pickle=True)
    return torch.from_numpy(np.asarray(d["u_h"], dtype=np.float32)), torch.from_numpy(
        np.asarray(d["u_g"], dtype=np.float32)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(OUT / "paired_dataset/full"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale_aug", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()
    data = Path(args.data)
    uh_tr, ug_tr = load_split(data / "train.npz")
    uh_va, ug_va = load_split(data / "val.npz")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ResidualMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    n = uh_tr.shape[0]
    best = 1e9
    best_state = None
    hist = []
    tag = f"seed{args.seed}_{'aug' if args.scale_aug else 'noaug'}"
    out = OUT / "mlp" / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"[mlp] params={model.n_params()} n_train={n} n_val={uh_va.shape[0]} {tag}", flush=True)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        nb = 0
        for i in range(0, n, args.bs):
            sl = perm[i : i + args.bs]
            uh = uh_tr[sl].to(device)
            ug = ug_tr[sl].to(device)
            if args.scale_aug:
                # u already dimensionless; scale aug is identity in u-space.
                # Inject a dummy multiplicative jitter on input then train to match gt (invariance).
                a = torch.empty(uh.shape[0], 1, device=device).uniform_(0.85, 1.15)
                uh_in = uh * a
                inv = F.mse_loss(model(uh_in), model(uh).detach())
            else:
                uh_in = uh
                inv = uh_in.sum() * 0.0
            pred = model(uh_in)
            lp = F.mse_loss(pred, ug)
            lpair, ldist = pairwise_loss(pred, ug)
            loss = lp + 0.5 * lpair + 0.2 * ldist + 0.1 * inv
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        model.eval()
        with torch.no_grad():
            pv = model(uh_va.to(device))
            val = float(F.mse_loss(pv, ug_va.to(device)).item())
            mpjpe = float((pv - ug_va.to(device)).reshape(-1, 3, 3).norm(dim=-1).mean().item())
        hist.append({"epoch": ep, "train": tot / max(nb, 1), "val_mse": val, "val_u_l2": mpjpe})
        print(f"[mlp] {tag} ep={ep} train={hist[-1]['train']:.5f} val={val:.5f} uL2={mpjpe:.5f}", flush=True)
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "params": model.n_params(), "val": best, "args": vars(args)}, out / "model_best.pt")
    (out / "hist.json").write_text(json.dumps(hist, indent=2))
    # test
    uh_te, ug_te = load_split(data / "test.npz")
    with torch.no_grad():
        pt = model(uh_te.to(device))
        test_mse = float(F.mse_loss(pt, ug_te.to(device)).item())
        test_l2 = float((pt - ug_te.to(device)).reshape(-1, 3, 3).norm(dim=-1).mean().item())
    (out / "metrics.json").write_text(
        json.dumps({"params": model.n_params(), "val_mse": best, "test_mse": test_mse, "test_u_l2": test_l2, "hist": hist}, indent=2)
    )
    print(f"[mlp] done {tag} val={best:.5f} test_mse={test_mse:.5f}", flush=True)


if __name__ == "__main__":
    main()
