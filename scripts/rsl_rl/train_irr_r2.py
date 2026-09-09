#!/usr/bin/env python3
"""IRR R2: Q(h, δz) → Â. Offline. No PPO. No terrain labels.

R1 (h→Δz) is the failed ICR residual, used only as a negative baseline in the report.
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


def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra, rb = ra.astype(np.float64), rb.astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if den < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / den)


class ResponseQ(nn.Module):
    def __init__(self, tok=128, hid=128, dz=16, use_hist=True):
        super().__init__()
        self.use_hist = bool(use_hist)
        self.gru = nn.GRU(tok, hid, batch_first=True)
        inn = hid + dz if self.use_hist else dz
        self.head = nn.Sequential(
            nn.Linear(inn, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, hist: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
        if self.use_hist:
            y, _ = self.gru(hist)
            x = torch.cat([y[:, -1], dz], dim=-1)
        else:
            x = dz
        return self.head(x).squeeze(-1)


def _load_pool(root: Path, pattern: str = "r0/*/*/r0.npz") -> dict:
    hists, dzs, a, windows, clips, terrains, e0 = [], [], [], [], [], [], []
    paths = sorted(root.glob(pattern))
    for p in paths:
        z = np.load(p, allow_pickle=True)
        hist = z["hist_tok"].astype(np.float32)
        dz = z["dz"].astype(np.float32)
        aa = z["a"].astype(np.float32)
        n, c = hist.shape[0], dz.shape[1]
        hists.append(np.repeat(hist[:, None], c, axis=1).reshape(n * c, hist.shape[1], hist.shape[2]))
        dzs.append(dz.reshape(n * c, dz.shape[-1]))
        a.append(aa.reshape(n * c))
        win = np.asarray(z["window"]).astype(str)
        clip = np.asarray(z["clip"]).astype(str)
        terr = np.asarray(z["terrain"]).astype(str)
        windows.append(np.repeat(win, c))
        clips.append(np.repeat(clip, c))
        terrains.append(np.repeat(terr, c))
        e0.append(np.repeat(z["e0"].astype(np.float32), c))
    if not hists:
        raise FileNotFoundError(f"no r0.npz under {root}/{pattern}")
    states = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        n = int(z["a"].shape[0])
        for i in range(n):
            states.append(
                {
                    "hist": z["hist_tok"][i].astype(np.float32),
                    "dz": z["dz"][i].astype(np.float32),
                    "a": z["a"][i].astype(np.float32),
                    "clip": str(z["clip"][i]),
                    "window": str(z["window"][i]),
                    "terrain": str(z["terrain"][i]),
                    "e0": float(z["e0"][i]),
                    "star": int(z["star"][i]),
                }
            )
    return {
        "hist": np.concatenate(hists, 0),
        "dz": np.concatenate(dzs, 0),
        "a": np.concatenate(a, 0),
        "clip": np.concatenate(clips, 0),
        "window": np.concatenate(windows, 0),
        "terrain": np.concatenate(terrains, 0),
        "states": states,
    }


def _clip_split(clips: np.ndarray, seed=7) -> dict[str, str]:
    uniq = sorted(set(str(c) for c in clips))
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(len(uniq))
    n = len(uniq)
    n_test = max(1, n // 5)
    n_val = max(1, n // 5)
    assign = {}
    for k, idx in enumerate(order):
        if k < n - n_test - n_val:
            assign[uniq[idx]] = "train"
        elif k < n - n_test:
            assign[uniq[idx]] = "val"
        else:
            assign[uniq[idx]] = "test"
    return assign


@torch.no_grad()
def _rank_metrics(model, states, assign, device, want="test") -> dict:
    model.eval()
    rhos, hit, regret = [], [], []
    rec_rhos, rec_hit, rec_regret = [], [], []
    sign_n, sign_ok, rec_sign_n, rec_sign_ok = 0, 0, 0, 0
    pred_pos, rec_pred_pos = [], []
    by_terrain = {}
    for s in states:
        if assign.get(s["clip"]) != want:
            continue
        hist = torch.from_numpy(s["hist"]).unsqueeze(0).repeat(s["dz"].shape[0], 1, 1).to(device)
        dz = torch.from_numpy(s["dz"]).to(device)
        hat = model(hist, dz).detach().cpu().numpy()
        a = s["a"]
        rho = _spearman(a, hat)
        rhos.append(rho)
        pred = int(np.argmax(hat))
        hit.append(int(pred == int(s["star"]) or (a[pred] >= a[int(s["star"])] - 1e-6)))
        regret.append(float(a[int(s["star"])] - a[pred]))
        pred_pos.append(int(a[pred] > 0))
        rec = s["window"] == "recovery"
        if rec:
            rec_rhos.append(rho)
            rec_hit.append(hit[-1])
            rec_regret.append(regret[-1])
            rec_pred_pos.append(pred_pos[-1])
        terr = str(s["terrain"])
        by_terrain.setdefault(terr, {"rho": [], "hit": []})
        by_terrain[terr]["rho"].append(rho)
        by_terrain[terr]["hit"].append(hit[-1])
        for j in range(a.size):
            if abs(a[j]) < 1e-6:
                continue
            ok = int((hat[j] > 0) == (a[j] > 0))
            sign_n += 1
            sign_ok += ok
            if rec:
                rec_sign_n += 1
                rec_sign_ok += ok

    def m(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        return None if x.size == 0 else float(x.mean())

    return {
        "n_states": len(rhos),
        "spearman": m(rhos),
        "spearman_recovery": m(rec_rhos),
        "top1": m(hit),
        "top1_recovery": m(rec_hit),
        "oracle_regret": m(regret),
        "oracle_regret_recovery": m(rec_regret),
        "sign_acc": (sign_ok / max(sign_n, 1)) if sign_n else None,
        "sign_acc_recovery": (rec_sign_ok / max(rec_sign_n, 1)) if rec_sign_n else None,
        "P_pred_star_gt0": m(pred_pos),
        "P_pred_star_gt0_recovery": m(rec_pred_pos),
        "by_terrain": {
            k: {"spearman": m(v["rho"]), "top1": m(v["hit"]), "n": len(v["rho"])}
            for k, v in sorted(by_terrain.items())
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/home/chenxiangyu/robotics/Anybody/results/irr_response_recovery")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no_hist", action="store_true")
    ap.add_argument("--glob", default="r0/*/*/r0.npz")
    args = ap.parse_args()
    root = Path(args.root)
    pool = _load_pool(root, args.glob)
    assign = _clip_split(np.asarray([s["clip"] for s in pool["states"]]))
    clips_pair = pool["clip"]
    split = np.asarray([assign.get(str(c), "train") for c in clips_pair])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = int(pool["hist"].shape[-1])
    dz_dim = int(pool["dz"].shape[-1])
    model = ResponseQ(tok=tok, dz=dz_dim, use_hist=not args.no_hist).to(device)
    tr = split == "train"
    ds = TensorDataset(
        torch.from_numpy(pool["hist"][tr]),
        torch.from_numpy(pool["dz"][tr]),
        torch.from_numpy(pool["a"][tr]),
    )
    loader = DataLoader(ds, batch_size=int(args.batch), shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.MSELoss()
    best = math.inf
    best_state = None
    for ep in range(int(args.epochs)):
        model.train()
        tot, n = 0.0, 0
        for h, d, y in loader:
            h, d, y = h.to(device), d.to(device), y.to(device)
            pred = model(h, d)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * int(y.shape[0])
            n += int(y.shape[0])
        val = _rank_metrics(model, pool["states"], assign, device, want="val")
        score = -(val["spearman"] or 0.0)
        if score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"[irr-r2] ep {ep+1}/{args.epochs} mse={tot / max(n, 1):.4f} "
                f"val_spearman={val['spearman']} val_top1={val['top1']} val_regret={val['oracle_regret']}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    test = _rank_metrics(model, pool["states"], assign, device, want="test")
    out = {
        "use_hist": not args.no_hist,
        "n_pairs": int(pool["a"].size),
        "n_states": len(pool["states"]),
        "test": test,
        "split": {k: int(sum(1 for s in pool["states"] if assign.get(s["clip"]) == k)) for k in ("train", "val", "test")},
    }
    tag = "r2_nohist" if args.no_hist else "r2"
    (root / f"{tag}.json").write_text(json.dumps(out, indent=2))
    torch.save({"model": model.state_dict(), "cfg": {"tok": tok, "dz": dz_dim, "use_hist": not args.no_hist}}, root / f"{tag}.pt")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
