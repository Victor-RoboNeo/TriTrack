#!/usr/bin/env python3
"""P3-B2-0: select global learned λ on val only. FFR/leak. No Isaac."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from p3_common import adv_block, sanitize, stats
from p3b_model import IntentMetricMLP, chol_to_C, random_tangent, soft_P, Z_DIM

SEEN = ("plane", "slope", "slope_down", "light_rough", "steps")
HELD = ("slip",)
ALL_T = SEEN + HELD
P3B_DATA = Path("/data/home/chenxiangyu/robotics/Anybody/results/p3_intent_projected_adaptation")
B1_CKPT = P3B_DATA / "p3b_learned_projector" / "checkpoints" / "B1_s2028.pt"


def _split_episodes(eps, seed=2026):
    uniq = sorted(set(eps.tolist() if hasattr(eps, "tolist") else list(eps)))
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


def _p3b_split_map() -> dict[str, str]:
    ddir = P3B_DATA / "p3b_learned_projector" / "dataset_b"
    eps = []
    for t in SEEN:
        z = np.load(ddir / f"{t}.npz", allow_pickle=True)
        eps.extend(np.asarray(z["parent_episode_id"]).astype(str).tolist())
    return _split_episodes(np.asarray(eps), seed=2026)


def _load_rows(root: Path) -> list[dict]:
    rows = []
    smap = _p3b_split_map()
    for t in ALL_T:
        p = root / "calibration" / f"{t}.json"
        if not p.exists():
            print(f"[p3b2-analyze] missing {p}", flush=True)
            continue
        data = json.loads(p.read_text())
        for r in data.get("rows") or []:
            r = dict(r)
            if r.get("terrain") in HELD:
                r["p3_split"] = "test_slip"
            else:
                r["p3_split"] = smap.get(str(r["episode_id"]), "val")
            rows.append(r)
    return rows


def _methods(rows):
    if not rows:
        return []
    keys = list(rows[0]["methods"].keys())
    out = ["oracle"] + [k for k in keys if k != "oracle"]
    return [k for k in out if k in keys]


def _col(rows, method, key):
    xs = []
    for r in rows:
        m = (r.get("methods") or {}).get(method)
        if m is None:
            continue
        xs.append(m[key])
    return np.asarray(xs, dtype=np.float64)


def _summarize(rows, methods) -> dict:
    out = {}
    for name in methods:
        A = _col(rows, name, "A")
        out[name] = {
            "A": adv_block(A),
            "DI_5deg": stats(_col(rows, name, "DI_5deg")),
            "DI_peak": stats(_col(rows, name, "DI_peak")),
            "DI_int": stats(_col(rows, name, "DI_int")),
            "Rd": stats(_col(rows, name, "Rd")),
            "cos_ucr": stats(_col(rows, name, "cos_ucr")),
            "oracle_leak": stats(_col(rows, name, "oracle_leak")),
            "P_suppressed": float(np.mean(_col(rows, name, "suppressed"))) if rows else None,
        }
    return out


def _select_lambda(val_sum: dict, methods: list[str]) -> dict:
    oracle = val_sum.get("oracle") or {}
    di_o = (oracle.get("DI_5deg") or {}).get("median")
    if di_o is None:
        return {"lambda_learned": None, "rule_satisfied": False, "reason": "no oracle on val"}
    thresh = 1.5 * float(di_o)
    cands = []
    closest = None
    for name in methods:
        if not name.startswith("B1_"):
            continue
        lam = float(name.split("_", 1)[1])
        di = (val_sum[name]["DI_5deg"] or {}).get("median")
        a = (val_sum[name]["A"] or {}).get("median")
        p = (val_sum[name]["A"] or {}).get("P_Agt0")
        p90 = (val_sum[name]["DI_5deg"] or {}).get("p90")
        rd = (val_sum[name]["Rd"] or {}).get("median")
        rec = (lam, a or -1e9, p or 0.0, -(p90 or 1e9), rd or 0.0, di, name)
        if closest is None or (di is not None and di < closest[5]):
            closest = rec
        if di is None or a is None:
            continue
        if di <= thresh:
            cands.append(rec)
    if not cands:
        return {
            "lambda_learned": None if closest is None else closest[0],
            "selected_method": None if closest is None else closest[6],
            "rule_satisfied": False,
            "DI_oracle_median": di_o,
            "DI_threshold": thresh,
            "fallback": "no lambda met 1.5x DI on val; recorded lowest-DI lambda",
        }
    cands.sort(key=lambda x: (-x[1], -x[2], -x[3], -x[4], x[0]))
    lam, a, p, negp90, rd, di, name = cands[0]
    return {
        "lambda_learned": lam,
        "selected_method": name,
        "rule_satisfied": True,
        "DI_oracle_median": di_o,
        "DI_threshold": thresh,
        "selected_DI_median": di,
        "selected_A_median": a,
        "selected_P_Agt0": p,
        "selected_Rd": rd,
    }


def _gate(learned, oracle, a_ratio=0.8, di_mult=1.5, p_delta=0.10) -> dict:
    a_l = (learned.get("A") or {}).get("median")
    a_o = (oracle.get("A") or {}).get("median")
    di_l = (learned.get("DI_5deg") or {}).get("median")
    di_o = (oracle.get("DI_5deg") or {}).get("median")
    p_l = (learned.get("A") or {}).get("P_Agt0")
    p_o = (oracle.get("A") or {}).get("P_Agt0")
    ratio = (a_l / a_o) if a_l is not None and a_o else None
    c1 = bool(ratio is not None and ratio >= a_ratio)
    c2 = bool(di_l is not None and di_o is not None and di_l <= di_mult * di_o)
    c3 = bool(p_l is not None and p_o is not None and p_l >= p_o - p_delta)
    return {
        "median_A_ratio": ratio,
        "A_ok": c1,
        "DI_ok": c2,
        "P_Agt0_ok": c3,
        "pass": bool(c1 and c2 and c3),
        "median_A_learned": a_l,
        "median_A_oracle": a_o,
        "DI_learned": di_l,
        "DI_oracle": di_o,
        "DI_ratio": (di_l / di_o) if di_l is not None and di_o else None,
        "P_Agt0_learned": p_l,
        "P_Agt0_oracle": p_o,
    }


def _ffr_fcr(s, sh, frac=0.25):
    n = s.size
    k = max(1, int(round(frac * n)))
    sens = np.argpartition(-s, kth=min(k, n - 1))[:k]
    free_hat = np.argpartition(sh, kth=min(k, n - 1))[:k]
    free_gt = np.argpartition(s, kth=min(k, n - 1))[:k]
    sens_hat = np.argpartition(-sh, kth=min(k, n - 1))[:k]
    ffr = float(np.intersect1d(sens, free_hat).size / max(sens.size, 1))
    fcr = float(np.intersect1d(free_gt, sens_hat).size / max(free_gt.size, 1))
    return ffr, fcr


@torch.no_grad()
def _offline_ffr(ckpt_p: Path, lam: float) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    net = IntentMetricMLP(int(ckpt["in_dim"]), out_dim=int(ckpt["out_dim"]))
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    mean = torch.as_tensor(ckpt["x_mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(ckpt["x_std"], device=device, dtype=torch.float32).clamp(min=1e-6)
    packed = np.load(P3B_DATA / "p3b_learned_projector" / "dataset_b" / "packed.npz", allow_pickle=True)
    out = {}
    g = torch.Generator(device=device)
    g.manual_seed(2026)
    train_s, train_sh = [], []
    bs = 256
    for sp in ("train", "val", "test_seen", "test_slip"):
        if f"{sp}/x" not in packed.files:
            continue
        x = torch.as_tensor(packed[f"{sp}/x"], device=device)
        z = torch.as_tensor(packed[f"{sp}/z0"], device=device)
        C = torch.as_tensor(packed[f"{sp}/C"], device=device)
        xn = ((x - mean) / std).clamp(-10, 10)
        Chat = chol_to_C(net(xn), z)
        ffrs, fcrs = [], []
        leaks_hat, leaks_or = [], []
        bs = 256
        for i in range(0, x.shape[0], bs):
            zb = z[i : i + bs]
            cb = C[i : i + bs]
            ch = Chat[i : i + bs]
            v = random_tangent(zb, 64, g)
            s = torch.einsum("bdi,bij,bdj->bd", v, cb, v)
            sh = torch.einsum("bdi,bij,bdj->bd", v, ch, v)
            Ph = soft_P(ch, zb, lam)
            Pg = soft_P(cb, zb, 1.0)
            dh = torch.einsum("bij,bdj->bdi", Ph, v)
            dg = torch.einsum("bij,bdj->bdi", Pg, v)
            dh = torch.nn.functional.normalize(dh, dim=-1, eps=1e-8)
            dg = torch.nn.functional.normalize(dg, dim=-1, eps=1e-8)
            leak_h = torch.einsum("bdi,bij,bdj->bd", dh, cb, dh)
            leak_g = torch.einsum("bdi,bij,bdj->bd", dg, cb, dg)
            sn = s.cpu().numpy()
            shn = sh.cpu().numpy()
            for j in range(sn.shape[0]):
                ffr, fcr = _ffr_fcr(sn[j], shn[j])
                ffrs.append(ffr)
                fcrs.append(fcr)
            leaks_hat.append(leak_h.cpu().numpy().reshape(-1))
            leaks_or.append(leak_g.cpu().numpy().reshape(-1))
            if sp == "train":
                train_s.append(sn.reshape(-1))
                train_sh.append(shn.reshape(-1))
        glob = None
        if train_s:
            ts = np.concatenate(train_s)
            tsh = np.concatenate(train_sh)
            q_s = float(np.quantile(ts, 0.75))
            q_free = float(np.quantile(tsh, 0.25))
            # recompute last-split global with stored? do after loop using this split's dirs
            glob = {"train_s75": q_s, "train_sh25": q_free}
        out[sp] = {
            "FFR_per_state_mean": float(np.mean(ffrs)) if ffrs else None,
            "FCR_per_state_mean": float(np.mean(fcrs)) if fcrs else None,
            "n": int(x.shape[0]),
            "probe_leak_hat": stats(np.concatenate(leaks_hat)) if leaks_hat else {},
            "probe_leak_oracle": stats(np.concatenate(leaks_or)) if leaks_or else {},
        }
        if glob and leaks_hat:
            # global FFR on this split using train quantiles from accumulated train (only valid after train)
            pass
        out[sp]["global_quantiles"] = glob
    if train_s:
        ts = np.concatenate(train_s)
        tsh = np.concatenate(train_sh)
        q_s = float(np.quantile(ts, 0.75))
        q_free = float(np.quantile(tsh, 0.25))
        out["global_thresholds"] = {"oracle_sensitive_s75": q_s, "pred_free_sh25": q_free}
        for sp in ("val", "test_seen", "test_slip"):
            if f"{sp}/x" not in packed.files:
                continue
            x = torch.as_tensor(packed[f"{sp}/x"], device=device)
            z = torch.as_tensor(packed[f"{sp}/z0"], device=device)
            C = torch.as_tensor(packed[f"{sp}/C"], device=device)
            xn = ((x - mean) / std).clamp(-10, 10)
            Chat = chol_to_C(net(xn), z)
            num = den = 0
            num_fcr = den_fcr = 0
            for i in range(0, x.shape[0], 256):
                zb = z[i : i + bs]
                v = random_tangent(zb, 64, g)
                s = torch.einsum("bdi,bij,bdj->bd", v, C[i : i + bs], v).cpu().numpy().reshape(-1)
                sh = torch.einsum("bdi,bij,bdj->bd", v, Chat[i : i + bs], v).cpu().numpy().reshape(-1)
                sens = s >= q_s
                pred_free = sh <= q_free
                pred_sens = sh >= q_s
                oracle_free = s <= np.quantile(ts, 0.25)
                den += int(sens.sum())
                num += int((sens & pred_free).sum())
                den_fcr += int(oracle_free.sum())
                num_fcr += int((oracle_free & pred_sens).sum())
            if sp in out:
                out[sp]["FFR_global"] = (num / den) if den else None
                out[sp]["FCR_global"] = (num_fcr / den_fcr) if den_fcr else None
    return out


def _fmt(st, key="median"):
    if not st or st.get(key) is None:
        return "n/a"
    return f"{st[key]:.4f}"


def _table(methods, blk):
    lines = [
        "| method | mean A | median A | P(A>0) | P(A>0.02) | D_I med | D_I p90 | Rd | leak med | cos |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in methods:
        s = blk.get(m) or {}
        lines.append(
            f"| {m} | {_fmt(s.get('A'), 'mean')} | {_fmt(s.get('A'))} | "
            f"{(s.get('A') or {}).get('P_Agt0')} | {(s.get('A') or {}).get('P_Agt0_02')} | "
            f"{_fmt(s.get('DI_5deg'))} | {_fmt(s.get('DI_5deg'), 'p90')} | "
            f"{_fmt(s.get('Rd'))} | {_fmt(s.get('oracle_leak'))} | {_fmt(s.get('cos_ucr'))} |"
        )
    return lines


def _csv(path: Path, methods, blk):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_A", "median_A", "P_Agt0", "P_Agt0_02", "P_Agt0_05",
                    "median_DI", "p90_DI", "p95_DI", "mean_DI", "median_DI_peak", "median_DI_int",
                    "median_Rd", "median_leak", "median_cos"])
        for m in methods:
            s = blk.get(m) or {}
            w.writerow([
                m,
                (s.get("A") or {}).get("mean"), (s.get("A") or {}).get("median"),
                (s.get("A") or {}).get("P_Agt0"), (s.get("A") or {}).get("P_Agt0_02"),
                (s.get("A") or {}).get("P_Agt0_05"),
                (s.get("DI_5deg") or {}).get("median"), (s.get("DI_5deg") or {}).get("p90"),
                (s.get("DI_5deg") or {}).get("p95"), (s.get("DI_5deg") or {}).get("mean"),
                (s.get("DI_peak") or {}).get("median"), (s.get("DI_int") or {}).get("median"),
                (s.get("Rd") or {}).get("median"), (s.get("oracle_leak") or {}).get("median"),
                (s.get("cos_ucr") or {}).get("median"),
            ])


def _plots(root: Path, methods, val_sum, test_sum, slip_sum, sel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdir = root / "calibration" / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    b1s = [m for m in methods if m.startswith("B1_")]
    lams, di, a, p, rd = [], [], [], [], []
    for m in b1s:
        lams.append(float(m.split("_", 1)[1]))
        di.append(((val_sum.get(m) or {}).get("DI_5deg") or {}).get("median"))
        a.append(((val_sum.get(m) or {}).get("A") or {}).get("median"))
        p.append(((val_sum.get(m) or {}).get("A") or {}).get("P_Agt0"))
        rd.append(((val_sum.get(m) or {}).get("Rd") or {}).get("median"))

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    odi = ((val_sum.get("oracle") or {}).get("DI_5deg") or {}).get("median")
    oa = ((val_sum.get("oracle") or {}).get("A") or {}).get("median")
    ax.scatter(odi, oa, s=120, marker="*", label="Oracle λ=1", zorder=5)
    ax.plot(di, a, marker="o", label="B1 λ sweep (val)")
    ax.set_xlabel("median DI @5°")
    ax.set_ylabel("median A")
    ax.set_title("B2-0 val Pareto: conservative λ vs recovery")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "plot1_lambda_pareto.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
    for ax, ys, ylab in (
        (axes[0, 0], di, "median DI"),
        (axes[0, 1], a, "median A"),
        (axes[1, 0], p, "P(A>0)"),
        (axes[1, 1], rd, "median Rd"),
    ):
        ax.plot(lams, ys, marker="o")
        ax.set_xlabel("λ_learned")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    fig.suptitle("B2-0 val: λ vs recovery / leakage")
    fig.tight_layout()
    fig.savefig(pdir / "plot2_lambda_curves.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    names = ["oracle"] + b1s
    data = []
    labs = []
    for m in names:
        xs = []
        # leak dist needs raw rows — skip here, plot medians
        med = ((test_sum.get(m) or {}).get("oracle_leak") or {}).get("median")
        if med is not None:
            labs.append(m)
            data.append(med)
    ax.bar(np.arange(len(data)), data)
    ax.set_xticks(np.arange(len(labs)))
    ax.set_xticklabels(labs, rotation=30, ha="right")
    ax.set_ylabel("median oracle leak  d^T C_gt d")
    ax.set_title("Test-seen oracle leakage (lower is safer)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "plot3_oracle_leak.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    pick = ["oracle"]
    if sel.get("selected_method"):
        pick.append(sel["selected_method"])
    if "B1_1" in methods or "B1_1.0" in methods:
        b11 = "B1_1" if "B1_1" in methods else "B1_1.0"
        if b11 not in pick:
            pick.append(b11)
    for blk, lab, mk in ((test_sum, "seen", "o"), (slip_sum, "slip", "s")):
        for m in pick:
            x = ((blk.get(m) or {}).get("DI_5deg") or {}).get("median")
            y = ((blk.get(m) or {}).get("A") or {}).get("median")
            if x is None or y is None:
                continue
            ax.scatter(x, y, marker=mk, s=90, label=f"{lab}:{m}")
    ax.set_xlabel("median DI @5°")
    ax.set_ylabel("median A")
    ax.set_title("Seen vs slip after calibration")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "plot5_seen_vs_slip.png", dpi=140)
    plt.close(fig)


def write_report(root, methods, sel, gates, val_sum, test_sum, all_seen, slip_sum, ffr, n_counts):
    name = sel.get("selected_method")
    b11 = "B1_1" if "B1_1" in methods else ("B1_1.0" if "B1_1.0" in methods else None)
    g_seen = gates.get("test_seen") or {}
    g_slip = gates.get("slip") or {}
    g_all = gates.get("all_seen") or {}
    pass_b20 = bool(g_seen.get("pass") and g_slip.get("pass") and sel.get("rule_satisfied"))
    leak_o = (test_sum.get("oracle") or {}).get("oracle_leak") or {}
    leak_b1 = (test_sum.get(b11) or {}).get("oracle_leak") or {} if b11 else {}
    leak_cal = (test_sum.get(name) or {}).get("oracle_leak") or {}
    lines = [
        "# P3-B2-0 — Conservative λ calibration of frozen B1",
        "",
        "Frozen: B1 checkpoint, Stage-2 / Mapper-B / decoder, oracle λ=1.0, no history, no terrain.",
        "Learned projector: P = (I + λ_L Ĉ)^{-1}. Only λ_L is swept.",
        "",
        f"asymmetric_training: {'NOT RUN — calibration already passed.' if pass_b20 else 'NOT RUN YET — B2-0 report first; B2-1 only if calibration failed.'}",
        "",
        f"Split counts: {json.dumps(n_counts)}",
        "",
        "## λ selection (validation only, unseen test/slip)",
        "",
        f"- selected λ_L = **{sel.get('lambda_learned')}** ({name})",
        f"- rule_satisfied (val DI ≤ 1.5× oracle) = **{sel.get('rule_satisfied')}**",
        f"- val DI threshold = {sel.get('DI_threshold')} (oracle median {sel.get('DI_oracle_median')})",
        f"- fallback: {sel.get('fallback')}",
        "",
        "## Validation (selection set)",
        "",
        *_table(methods, val_sum),
        "",
        "## Test seen (not used for λ)",
        "",
        *_table(methods, test_sum),
        "",
        "## All seen recovery (P3-B-comparable, not used for λ)",
        "",
        *_table(methods, all_seen),
        "",
        "## Held-out slip (same λ, no slip tuning)",
        "",
        *_table(methods, slip_sum),
        "",
        "## Pass criteria (selected λ)",
        "",
        f"- test_seen: A_ratio={g_seen.get('median_A_ratio')} DI_ratio={g_seen.get('DI_ratio')} pass={g_seen.get('pass')}",
        f"- all_seen: A_ratio={g_all.get('median_A_ratio')} DI_ratio={g_all.get('DI_ratio')} pass={g_all.get('pass')}",
        f"- slip: A_ratio={g_slip.get('median_A_ratio')} DI_ratio={g_slip.get('DI_ratio')} pass={g_slip.get('pass')}",
        "",
        f"## P3-B successful via calibration: **{pass_b20}**",
        f"## proceed_to_p3c: **False** (do not auto-start P3-C)",
        f"## proceed_to_b21: **{not pass_b20}**",
        "",
        "## Ten questions",
        "",
        f"1. Global scale vs anisotropic? "
        f"{'Likely global scale if a single λ meets DI without killing A.' if pass_b20 else 'If no λ in {{1..5}} meets val DI≤1.5× while keeping A, leakage is not a pure global scale error.'}",
        f"2. Selected λ_L = {sel.get('lambda_learned')} (oracle remains 1.0).",
        f"3. Does calibration satisfy seen DI? test_seen DI_ok={g_seen.get('DI_ok')} all_seen DI_ok={g_all.get('DI_ok')}.",
        f"4. Same λ on slip DI? DI_ok={g_slip.get('DI_ok')} ratio={g_slip.get('DI_ratio')}.",
        f"5. Recovery after calibration: test median A ratio={g_seen.get('median_A_ratio')}, "
        f"P(A>0) learned={g_seen.get('P_Agt0_learned')} oracle={g_seen.get('P_Agt0_oracle')}.",
        f"6. Oracle leak (test median): oracle={leak_o.get('median')} B1_λ1={leak_b1.get('median')} "
        f"B1_cal={leak_cal.get('median')}.",
        f"7. FFR: {json.dumps(sanitize({k: (v.get('FFR_per_state_mean') if isinstance(v, dict) else v) for k,v in (ffr or {}).items()}))}",
        "8. Retraining: not in B2-0.",
        "9. FCR: see FFR block; FFR is the primary safety metric.",
        f"10. Ready for P3-C? **False until B2-0 or B2-1 pass criteria hold. Current B2-0 pass={pass_b20}.**",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return pass_b20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/p3b2_conservative_projector")
    ap.add_argument("--b1_ckpt", type=str, default=str(B1_CKPT))
    args = ap.parse_args()
    root = Path(args.root)
    rows = _load_rows(root)
    methods = _methods(rows)
    val = [r for r in rows if r.get("p3_split") == "val"]
    test = [r for r in rows if r.get("p3_split") == "test_seen"]
    slip = [r for r in rows if r.get("p3_split") == "test_slip"]
    all_seen = [r for r in rows if r.get("terrain") in SEEN]
    n_counts = {"val": len(val), "test_seen": len(test), "all_seen": len(all_seen), "slip": len(slip)}
    print(f"[p3b2-analyze] splits {n_counts}", flush=True)
    val_sum = _summarize(val, methods)
    test_sum = _summarize(test, methods)
    slip_sum = _summarize(slip, methods)
    all_sum = _summarize(all_seen, methods)
    sel = _select_lambda(val_sum, methods)
    name = sel.get("selected_method")
    b11 = "B1_1" if "B1_1" in methods else ("B1_1.0" if "B1_1.0" in methods else None)
    gates = {
        "test_seen": _gate(test_sum.get(name) or {}, test_sum.get("oracle") or {}),
        "all_seen": _gate(all_sum.get(name) or {}, all_sum.get("oracle") or {}),
        "slip": _gate(slip_sum.get(name) or {}, slip_sum.get("oracle") or {}, a_ratio=0.7, di_mult=2.0),
        "b1_lambda1_all_seen": _gate(all_sum.get(b11) or {}, all_sum.get("oracle") or {}) if b11 else {},
    }
    pass_b20 = bool(gates["test_seen"].get("pass") and gates["slip"].get("pass") and sel.get("rule_satisfied"))
    _csv(root / "calibration" / "lambda_sweep.csv", methods, val_sum)
    _csv(root / "seen_eval" / "results.csv", methods, test_sum)
    _csv(root / "slip_eval" / "results.csv", methods, slip_sum)
    ffr = {}
    try:
        lam = float(sel.get("lambda_learned") or 1.0)
        ffr = _offline_ffr(Path(args.b1_ckpt), lam)
    except Exception as e:
        print(f"[p3b2-analyze] FFR failed {e}", flush=True)
        ffr = {"error": str(e)}
    try:
        _plots(root, methods, val_sum, test_sum, slip_sum, sel)
    except Exception as e:
        print(f"[p3b2-analyze] plot failed {e}", flush=True)
    write_report(root, methods, sel, gates, val_sum, test_sum, all_sum, slip_sum, ffr, n_counts)
    payload = {
        "selection": sel,
        "gates": gates,
        "n": n_counts,
        "val": val_sum,
        "test_seen": test_sum,
        "all_seen": all_sum,
        "slip": slip_sum,
        "ffr": ffr,
        "pass_b20": pass_b20,
        "proceed_to_p3c": False,
        "proceed_to_b21": (not pass_b20),
        "asymmetric_training": "NOT RUN — calibration already passed." if pass_b20 else "NOT RUN YET",
        "oracle_lambda": 1.0,
        "b1_ckpt": args.b1_ckpt,
        "no_terrain_in_model": True,
    }
    (root / "metrics.json").write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")
    print(json.dumps(sanitize({"selection": sel, "gates": gates, "pass_b20": pass_b20, "n": n_counts}), indent=2), flush=True)
    if pass_b20:
        print("[p3b2-analyze] STOP P3-B2. Calibration passed. Do not retrain. Do not start P3-C.", flush=True)
    else:
        print("[p3b2-analyze] B2-0 FAILED. B2-1 may start after this report.", flush=True)


if __name__ == "__main__":
    main()
