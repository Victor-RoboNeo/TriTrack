#!/usr/bin/env python3
"""Step 7A — Offline recovery-advantage identifiability. No PPO. No Isaac.

From 6S-R clone states: o_R (487) → y = 1[I_500 < -eps]. Episode split.
Decision regret vs Parent / Always-R / Max-2 / Oracle. Burst index is NOT an input.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

TERRAINS = ("plane", "light_rough", "slope", "steps")
BURST_GROUPS = ("1", "2", "3", "4-7", ">=8")
IN_DIM = 487


def _ep_key(r: dict) -> str:
    return f"{r['terrain']}|{r['seed']}|{r['clip']}"


def _burst_group(b: int) -> str:
    b = int(b)
    if b <= 3:
        return str(b)
    if b <= 7:
        return "4-7"
    return ">=8"


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


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int32)
    s = np.asarray(s, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Mann–Whitney / Wilcoxon: P(score_pos > score_neg) + 0.5 P(eq)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # average ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_pos = float(ranks[y == 1].sum())
    n_pos, n_neg = float(pos.size), float(neg.size)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _prf(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0) else float("nan")
    acc = float((y == pred).mean()) if y.size else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "tp": tp, "fp": fp, "fn": fn}


def _load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for ter in TERRAINS:
        p = root / ter / "rows.json"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        for r in json.loads(p.read_text()):
            r = dict(r)
            r["burst_group"] = str(r.get("burst_group") or _burst_group(int(r["burst_index"])))
            rows.append(r)
    return rows


def _attach_obs(rows: list[dict], root: Path) -> int:
    """Match rec_obs.npz to rows by (terrain, seed, clip, t0, burst_index)."""
    n_hit = 0
    for ter in TERRAINS:
        p = root / ter / "rec_obs.npz"
        if not p.exists():
            continue
        blob = np.load(p, allow_pickle=True)
        rec = blob["rec_obs"]
        keys = {}
        for i in range(rec.shape[0]):
            k = (
                str(blob["terrain"][i]),
                int(blob["seed"][i]),
                str(blob["clip"][i]),
                int(blob["t0"][i]),
                int(blob["burst_index"][i]),
            )
            keys[k] = rec[i]
        for r in rows:
            if r["terrain"] != ter:
                continue
            k = (str(r["terrain"]), int(r["seed"]), str(r["clip"]), int(r["t0"]), int(r["burst_index"]))
            if k in keys:
                r["rec_obs"] = keys[k].astype(np.float32)
                n_hit += 1
    return n_hit


def _split_episodes(rows: list[dict], seed: int = 0) -> dict[str, set[str]]:
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
        n_train = max(1, int(round(0.70 * n)))
        n_val = int(round(0.15 * n))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        n_test = n - n_train - n_val
        splits["train"].update(eps[:n_train])
        splits["val"].update(eps[n_train : n_train + n_val])
        splits["test"].update(eps[n_train + n_val :])
        print(f"[7a] {ter}: episodes {n} → train {n_train} val {n_val} test {n_test}", flush=True)
    return splits


def _regret_block(rows: list[dict], chosen_r: np.ndarray) -> dict:
    jr = np.array([r["E_R_500_cm"] for r in rows], dtype=np.float64)
    jp = np.array([r["E_P_500_cm"] for r in rows], dtype=np.float64)
    oracle = np.minimum(jr, jp)
    chosen = np.where(chosen_r, jr, jp)
    reg = chosen - oracle
    return {
        "n": len(rows),
        "mean_regret_cm": float(reg.mean()) if reg.size else float("nan"),
        "median_regret_cm": float(np.median(reg)) if reg.size else float("nan"),
        "p90_regret_cm": float(np.percentile(reg, 90)) if reg.size else float("nan"),
        "mean_J_chosen_cm": float(chosen.mean()) if chosen.size else float("nan"),
        "mean_J_oracle_cm": float(oracle.mean()) if oracle.size else float("nan"),
        "P_continue": float(np.mean(chosen_r)) if chosen_r.size else float("nan"),
    }


def _policy_continue_masks(rows: list[dict], eps: float) -> dict[str, np.ndarray]:
    a = np.array([r["I_500_cm"] for r in rows], dtype=np.float64)
    b = np.array([int(r["burst_index"]) for r in rows], dtype=np.int32)
    return {
        "always_parent": np.zeros(len(rows), dtype=bool),
        "always_recovery": np.ones(len(rows), dtype=bool),
        "max2": b <= 2,
        "oracle": a < -float(eps),
    }


class AdvantageMLP(nn.Module):
    def __init__(self, in_dim: int = IN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 400, lr: float = 0.05) -> np.ndarray:
    """Tiny L2 logistic; returns weight vector including bias as last dim."""
    n, d = x.shape
    xb = np.concatenate([x, np.ones((n, 1), dtype=np.float64)], axis=1)
    w = np.zeros(d + 1, dtype=np.float64)
    for _ in range(steps):
        z = xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        grad = xb.T @ (p - y) / n + 1e-3 * w
        w -= lr * grad
    return w


def _eval_split(rows: list[dict], score: np.ndarray | None, pred: np.ndarray, eps: float, name: str) -> dict:
    y = np.array([1 if r["I_500_cm"] < -eps else 0 for r in rows], dtype=np.int32)
    out = {
        "n": len(rows),
        "P_continue_label": float(y.mean()) if y.size else float("nan"),
        "cls": _prf(y.astype(bool), pred),
        "regret": _regret_block(rows, pred),
    }
    if score is not None:
        out["AUROC"] = _auroc(y, score)
    by_g = {}
    for g in BURST_GROUPS:
        idx = [i for i, r in enumerate(rows) if r["burst_group"] == g]
        if not idx:
            continue
        rs = [rows[i] for i in idx]
        yg = y[idx]
        pg = pred[idx]
        blk = {"n": len(idx), "cls": _prf(yg.astype(bool), pg), "regret": _regret_block(rs, pg)}
        if score is not None:
            blk["AUROC"] = _auroc(yg, score[idx])
        by_g[g] = blk
    out["by_burst_group"] = by_g
    by_t = {}
    for ter in TERRAINS:
        idx = [i for i, r in enumerate(rows) if r["terrain"] == ter]
        if not idx:
            continue
        rs = [rows[i] for i in idx]
        by_t[ter] = {
            "n": len(idx),
            "cls": _prf(y[idx].astype(bool), pred[idx]),
            "regret": _regret_block(rs, pred[idx]),
        }
        if score is not None:
            by_t[ter]["AUROC"] = _auroc(y[idx], score[idx])
    out["by_terrain"] = by_t
    out["policy"] = name
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="results/p2r_step6s_r")
    ap.add_argument("--out", type=str, default="results/p2r_step7a")
    ap.add_argument("--eps_cm", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
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
    print(f"[7a] rows={len(rows)} rec_obs_matched={n_obs}", flush=True)
    splits = _split_episodes(rows, seed=int(args.seed))
    buckets = {name: [r for r in rows if _ep_key(r) in splits[name]] for name in ("train", "val", "test")}
    for name, rs in buckets.items():
        print(f"[7a] {name} states={len(rs)}", flush=True)

    payload: dict = {
        "step": "7A",
        "no_ppo": True,
        "eps_cm": eps,
        "n_rows": len(rows),
        "n_obs_matched": n_obs,
        "n_episodes": {k: len(v) for k, v in splits.items()},
        "n_states": {k: len(v) for k, v in buckets.items()},
        "note": "y=1[I_500 < -eps]; uncertain→release. Burst index not an MLP input.",
    }

    # Baselines on test (and all, for closed-loop analog)
    for split_name, rs in [("test", buckets["test"]), ("all", rows)]:
        policies = _policy_continue_masks(rs, eps)
        payload.setdefault("baselines", {})[split_name] = {
            name: _eval_split(rs, None, mask, eps, name) for name, mask in policies.items()
        }

    # E-only logistic (E0, R_E) — shows E is not a sufficient statistic
    tr = buckets["train"]
    te = buckets["test"]
    x_tr = np.array([[r["E0_cm"], r["R_E"]] for r in tr], dtype=np.float64)
    y_tr = np.array([1.0 if r["I_500_cm"] < -eps else 0.0 for r in tr], dtype=np.float64)
    mu = x_tr.mean(0)
    sd = np.clip(x_tr.std(0), 1e-6, None)
    w = _fit_logistic((x_tr - mu) / sd, y_tr)
    x_te = (np.array([[r["E0_cm"], r["R_E"]] for r in te], dtype=np.float64) - mu) / sd
    xb = np.concatenate([x_te, np.ones((len(te), 1))], axis=1)
    score_e = xb @ w
    pred_e = score_e > 0.0
    payload["e_only"] = _eval_split(te, score_e, pred_e, eps, "e_only_logistic")
    print(
        f"[7a] E-only test AUROC={payload['e_only'].get('AUROC')} "
        f"regret={payload['e_only']['regret']['mean_regret_cm']:.3f}cm",
        flush=True,
    )

    mlp_ok = n_obs == len(rows) and all("rec_obs" in r for r in rows)
    if mlp_ok:
        x_lin_tr = np.stack([r["rec_obs"] for r in tr], axis=0).astype(np.float64)
        y_lin = np.array([1.0 if r["I_500_cm"] < -eps else 0.0 for r in tr], dtype=np.float64)
        mu_l = x_lin_tr.mean(0)
        sd_l = np.clip(x_lin_tr.std(0), 1e-6, None)
        w_or = _fit_logistic((x_lin_tr - mu_l) / sd_l, y_lin, steps=800, lr=0.02)
        x_lin_te = (np.stack([r["rec_obs"] for r in te], axis=0).astype(np.float64) - mu_l) / sd_l
        xb_or = np.concatenate([x_lin_te, np.ones((len(te), 1))], axis=1)
        score_or = xb_or @ w_or
        payload["oR_linear"] = _eval_split(te, score_or, score_or > 0.0, eps, "oR_linear")
        print(
            f"[7a] oR-linear test AUROC={payload['oR_linear'].get('AUROC')} "
            f"regret={payload['oR_linear']['regret']['mean_regret_cm']:.3f}cm",
            flush=True,
        )

    mlp_ok = n_obs == len(rows) and all("rec_obs" in r for r in rows)
    if not mlp_ok:
        payload["mlp"] = None
        payload["next"] = "Need rec_obs.npz dump (6S-R --dump_obs_only) then re-run 7A."
        print("[7a] rec_obs incomplete; skip MLP. Dump obs then rerun.", flush=True)
        (out / "summary.json").write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
        return

    def _xy(rs):
        x = np.stack([r["rec_obs"] for r in rs], axis=0).astype(np.float32)
        y = np.array([1.0 if r["I_500_cm"] < -eps else 0.0 for r in rs], dtype=np.float32)
        return x, y

    x_tr, y_tr_t = _xy(tr)
    x_va, y_va = _xy(buckets["val"] or tr)
    x_te, _ = _xy(te)
    mu_x = x_tr.mean(0)
    sd_x = np.clip(x_tr.std(0), 1e-6, None)
    x_tr_n = (x_tr - mu_x) / sd_x
    x_va_n = (x_va - mu_x) / sd_x
    x_te_n = (x_te - mu_x) / sd_x

    torch.manual_seed(int(args.seed))
    model = AdvantageMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))
    best_state = None
    best_val = -1e9
    stale = 0
    xt = torch.from_numpy(x_tr_n)
    yt = torch.from_numpy(y_tr_t)
    xv = torch.from_numpy(x_va_n)
    yv = torch.from_numpy(y_va)
    n = xt.shape[0]
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        perm = torch.randperm(n)
        losses = []
        for i in range(0, n, int(args.batch)):
            idx = perm[i : i + int(args.batch)]
            opt.zero_grad()
            logit = model(xt[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(logit, yt[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            s_val = torch.sigmoid(model(xv)).numpy()
            auroc_val = _auroc(y_va.astype(np.int32), s_val)
        if np.isfinite(auroc_val) and auroc_val > best_val + 1e-4:
            best_val = float(auroc_val)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"[7a] ep {epoch:3d} loss={np.mean(losses):.4f} val_AUROC={auroc_val:.3f} best={best_val:.3f}",
                flush=True,
            )
        if stale >= int(args.patience) and epoch >= 40:
            print(f"[7a] early stop epoch={epoch}", flush=True)
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        score_mlp = model(torch.from_numpy(x_te_n)).numpy()
    pred_mlp = score_mlp > 0.0
    payload["mlp"] = _eval_split(te, score_mlp, pred_mlp, eps, "mlp_oR")
    payload["mlp"]["val_AUROC"] = best_val
    torch.save(
        {"model": best_state, "obs_mean": mu_x, "obs_std": sd_x, "eps_cm": eps, "in_dim": IN_DIM},
        out / "model_best.pt",
    )
    te_mlp = payload["mlp"]
    te_max2 = payload["baselines"]["test"]["max2"]["regret"]["mean_regret_cm"]
    te_or = payload["baselines"]["test"]["oracle"]["regret"]["mean_regret_cm"]
    auroc = te_mlp.get("AUROC", float("nan"))
    reg = te_mlp["regret"]["mean_regret_cm"]
    go = bool(np.isfinite(auroc) and auroc >= 0.75 and np.isfinite(reg) and reg < te_max2 - 0.05)
    payload["gates"] = {
        "AUROC_ge_0.75": bool(np.isfinite(auroc) and auroc >= 0.75),
        "regret_lt_max2": bool(np.isfinite(reg) and reg < te_max2 - 0.05),
        "GO_7CL": go,
        "test_AUROC": auroc,
        "test_regret_cm": reg,
        "test_max2_regret_cm": te_max2,
        "test_oracle_regret_cm": te_or,
    }
    payload["next"] = (
        "GO Step 7-CL advantage-gated recovery"
        if go
        else "HOLD 7-CL. If AUROC weak, try 7B recovery-history; do not PPO."
    )
    print(
        f"[7a] MLP test AUROC={auroc} regret={reg:.3f}cm max2={te_max2:.3f} oracle={te_or:.3f} GO={go}",
        flush=True,
    )
    (out / "summary.json").write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    print("[7a] wrote", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
