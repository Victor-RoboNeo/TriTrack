#!/usr/bin/env python3
"""UCR-1: retrain the frozen R-M3 architecture on four-task RE-trigger oracle.

Same MLP, same cosine+CE loss. Change the training distribution, not the model.
Checkpoint selection for GO is held-out clone utility (see eval), not cosine.
This script still writes last + cosine-best for that later pick.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_rm2_intent_adapter import (  # noqa: E402
    IN_DIM,
    LATENT,
    THETA_BINS,
    IntentConditionedRecoveryAdapter,
    _angle_metrics,
    _cos_stats,
    _eval_group as _eval_group_rm2,
    _pack_split,
    _project_unit,
    _slice_mask,
)

TASKS = ("loco", "stoop", "reach", "carry")


def _balanced_idx(rows: list[dict], n: int, rng: np.random.Generator) -> np.ndarray:
    groups = []
    for t in TASKS:
        ix = np.array([i for i, r in enumerate(rows) if r.get("task_source") == t], dtype=np.int64)
        if ix.size:
            groups.append(ix)
    if not groups:
        return rng.choice(len(rows), size=n, replace=True)
    parts = []
    base = n // len(groups)
    rem = n - base * len(groups)
    for k, g in enumerate(groups):
        take = base + (1 if k < rem else 0)
        parts.append(rng.choice(g, size=max(take, 1), replace=True))
    idx = np.concatenate(parts)[:n]
    rng.shuffle(idx)
    return idx


def _eval_group(pred, d, logits, y, hard, rows):
    out = _eval_group_rm2(pred, d, logits, y, hard, rows)
    for t in TASKS:
        if t in out:
            continue
        m = _slice_mask(rows, lambda r, tt=t: r.get("task_source") == tt)
        if not m.any():
            out[t] = {"dir": {"n": 0}, "angle": {"n": 0}}
            continue
        out[t] = {
            "dir": _cos_stats(pred[m], d[m]),
            "angle": _angle_metrics(logits[m], y[m], hard[m]),
        }
    return out


@torch.no_grad()
def _forward_numpy(model, pack):
    raw, logits = model(pack["x"])
    pred = _project_unit(raw, pack["z"])
    return pred.cpu().numpy(), logits.cpu().numpy(), pack["d"].cpu().numpy()


def _cosine_score(ev: dict) -> float:
    meds = []
    accs = []
    for t in TASKS:
        d = (ev.get(t) or {}).get("dir") or {}
        a = (ev.get(t) or {}).get("angle") or {}
        if int(d.get("n") or 0) <= 0:
            continue
        meds.append(float(d.get("median") or 0.0))
        accs.append(float(a.get("top1") or 0.0))
    if not meds:
        return float("-inf")
    return float(np.mean(meds) + 0.3 * np.mean(accs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lambda_theta", type=float, default=0.25)
    ap.add_argument("--tol_cm", type=float, default=0.15)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--init_ckpt", type=str, default="",
                    help="Optional warm-start from old R-M3. Secondary contrast only.")
    ap.add_argument("--tag", type=str, default="scratch")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    buckets = {k: json.loads((data / f"{k}.json").read_text()) for k in ("train", "val", "test")}
    for k, rs in buckets.items():
        by = {t: sum(1 for r in rs if r.get("task_source") == t) for t in TASKS}
        print(f"[ucr1-train] {k} n={len(rs)} by_task={by}", flush=True)

    x_tr = np.asarray([r["rec_obs"] for r in buckets["train"]], dtype=np.float32)
    mu = x_tr.mean(0)
    sd = x_tr.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    packs = {k: _pack_split(v, mu, sd, float(args.tol_cm), device) for k, v in buckets.items()}

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    model = IntentConditionedRecoveryAdapter().to(device)
    if args.init_ckpt:
        blob = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        print(f"[ucr1-train] warm-start {args.init_ckpt}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))

    n_tr = len(buckets["train"])
    bs = min(int(args.batch), max(n_tr, 2))
    steps_per = max(1, int(math.ceil(n_tr / bs)))
    best_score = -1e9
    best_state = None
    last_state = None
    hist = []

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
        last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        score = _cosine_score(epoch_m["val"])
        if score > best_score + 1e-4:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        hist.append({"epoch": epoch, "loss": epoch_m["loss"], "val_cosine_score": score})
        if epoch % 20 == 0 or epoch == 1:
            vd = epoch_m["val"]
            bits = " ".join(
                f"{t}={vd.get(t, {}).get('dir', {}).get('median', float('nan')):.3f}"
                for t in TASKS
            )
            print(f"[ucr1-train] ep {epoch:3d} loss={epoch_m['loss']:.4f} {bits} cscore={score:.3f}", flush=True)

    assert last_state is not None and best_state is not None

    def _dump(state, tag: str):
        model.load_state_dict(state)
        model.eval()
        final = {}
        gap = {}
        with torch.no_grad():
            for name in ("train", "val", "test"):
                pred, logits, dnp = _forward_numpy(model, packs[name])
                ynp = packs[name]["y"].cpu().numpy()
                hnp = packs[name]["hard"].cpu().numpy()
                final[name] = _eval_group(pred, dnp, logits, ynp, hnp, buckets[name])
        for t in TASKS:
            tr = (final["train"].get(t) or {}).get("dir") or {}
            te = (final["test"].get(t) or {}).get("dir") or {}
            gap[t] = {
                "train_median_cos": tr.get("median"),
                "test_median_cos": te.get("median"),
                "train_n": tr.get("n"),
                "test_n": te.get("n"),
            }
        ckpt = {
            "model": state,
            "obs_mean": mu,
            "obs_std": sd,
            "in_dim": IN_DIM,
            "latent": LATENT,
            "theta_bins": list(THETA_BINS),
            "lambda_theta": float(args.lambda_theta),
            "tol_cm": float(args.tol_cm),
            "split_seed": int(args.seed),
            "best_val_cosine_score": best_score,
            "tag": f"ucr1-{args.tag}-{tag}",
            "note": "No PPO. No task ID. Same R-M3 architecture. Select by clone utility, not cosine.",
        }
        path = out / f"model_{tag}.pt"
        torch.save(ckpt, path)
        (out / f"metrics_{tag}.json").write_text(json.dumps({
            "tag": args.tag,
            "ckpt_tag": tag,
            "n_states": {k: len(v) for k, v in buckets.items()},
            "metrics": final,
            "train_test_gap": gap,
            "hparams": {
                "lr": float(args.lr),
                "wd": float(args.wd),
                "batch": int(args.batch),
                "lambda_theta": float(args.lambda_theta),
                "seed": int(args.seed),
                "init_ckpt": args.init_ckpt or None,
            },
        }, indent=2))
        print(f"[ucr1-train] wrote {path}", flush=True)
        return final, gap

    _dump(last_state, "last")
    _dump(best_state, "cosine")
    (out / "history.json").write_text(json.dumps(hist))
    print("[ucr1-train] done; pick checkpoint by held-out clone utility", flush=True)


if __name__ == "__main__":
    main()
