#!/usr/bin/env python3
"""P3-B2-1 analyze: B21 vs B1 vs Oracle. No Isaac. No P4."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from p3_common import adv_block, sanitize, stats
from p3b_model import IntentMetricMLP, chol_to_C, random_tangent

SEEN = ("plane", "slope", "slope_down", "light_rough", "steps")
HELD = ("slip",)
ALL_T = SEEN + HELD
P3B = Path("/data/home/chenxiangyu/robotics/Anybody/results/p3_intent_projected_adaptation")


def _load_rows(root: Path) -> list[dict]:
    rows = []
    for t in ALL_T:
        p = root / "p3b2_intent_shield" / "rollout" / f"{t}.json"
        if not p.exists():
            print(f"[b21-analyze] missing {p}", flush=True)
            continue
        rows.extend(json.loads(p.read_text()).get("rows") or [])
    return rows


def _col(rows, method, key):
    xs = [((r.get("methods") or {}).get(method) or {}).get(key) for r in rows]
    xs = [x for x in xs if x is not None]
    return np.asarray(xs, dtype=np.float64)


def _summarize(rows, methods):
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
        }
    return out


def _gate(learned, oracle, a_ratio=0.8, di_mult=1.5, p_delta=0.10):
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
        "cos": (learned.get("cos_ucr") or {}).get("median"),
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
def _offline_ffr(ckpt_p: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    net = IntentMetricMLP(int(ckpt["in_dim"]), out_dim=int(ckpt["out_dim"]))
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    mean = torch.as_tensor(ckpt["x_mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(ckpt["x_std"], device=device, dtype=torch.float32).clamp(min=1e-6)
    packed = np.load(P3B / "p3b_learned_projector" / "dataset_b" / "packed.npz", allow_pickle=True)
    g = torch.Generator(device=device)
    g.manual_seed(2026)
    out = {}
    for sp in ("val", "test_seen", "test_slip"):
        if f"{sp}/x" not in packed.files:
            continue
        x = torch.as_tensor(packed[f"{sp}/x"], device=device)
        z = torch.as_tensor(packed[f"{sp}/z0"], device=device)
        C = torch.as_tensor(packed[f"{sp}/C"], device=device)
        xn = ((x - mean) / std).clamp(-10, 10)
        Chat = chol_to_C(net(xn), z)
        ffrs, fcrs = [], []
        for i in range(0, x.shape[0], 256):
            zb = z[i : i + 256]
            v = random_tangent(zb, 64, g)
            s = torch.einsum("bdi,bij,bdj->bd", v, C[i : i + 256], v).cpu().numpy()
            sh = torch.einsum("bdi,bij,bdj->bd", v, Chat[i : i + 256], v).cpu().numpy()
            for j in range(s.shape[0]):
                ffr, fcr = _ffr_fcr(s[j], sh[j])
                ffrs.append(ffr)
                fcrs.append(fcr)
        out[sp] = {"FFR": float(np.mean(ffrs)), "FCR": float(np.mean(fcrs)), "n": int(x.shape[0])}
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


def _plots(root, methods, seen, slip):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdir = root / "p3b2_intent_shield" / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = np.arange(len(methods))
    for off, blk, lab in ((-0.18, seen, "seen"), (0.18, slip, "slip")):
        ax.bar(xs + off, [((blk.get(m) or {}).get("A") or {}).get("median") or 0 for m in methods], width=0.32, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels(methods)
    ax.set_ylabel("median A")
    ax.set_title("B2-1 recovery vs Oracle / B1")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "medianA.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for blk, lab, mk in ((seen, "seen", "o"), (slip, "slip", "s")):
        for m in methods:
            x = ((blk.get(m) or {}).get("DI_5deg") or {}).get("median")
            y = ((blk.get(m) or {}).get("A") or {}).get("median")
            if x is None or y is None:
                continue
            ax.scatter(x, y, marker=mk, s=90, label=f"{lab}:{m}")
    ax.set_xlabel("median DI @5°")
    ax.set_ylabel("median A")
    ax.set_title("B2-1: leakage vs recovery")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "A_vs_DI.png", dpi=140)
    plt.close(fig)


def write_report(iae, methods, seen, slip, gates, ffr, selected, stop_det):
    g = gates.get("B21_seen") or {}
    gs = gates.get("B21_slip") or {}
    pass_b21 = bool(g.get("pass") and gs.get("pass"))
    lines = [
        "# P3-B2-1 — Conservative Learned Intent Shield",
        "",
        "Init from frozen B1. Same P3-B split. Instantaneous state only. λ_pred=1, oracle λ=1.",
        "Loss: 0.25 L_F + 1.0 L_Q^asym(w=3) + 2.0 L_P + 3.0 L_leak(κ=1.25).",
        "",
        f"Selected: {json.dumps(selected)}",
        "",
        "## Seen UCR burst",
        "",
        *_table(methods, seen),
        "",
        "## Held-out slip",
        "",
        *_table(methods, slip),
        "",
        "## Gates",
        "",
        f"- seen: A_ratio={g.get('median_A_ratio')} DI_ratio={g.get('DI_ratio')} P_ok={g.get('P_Agt0_ok')} pass={g.get('pass')}",
        f"- slip: A_ratio={gs.get('median_A_ratio')} DI_ratio={gs.get('DI_ratio')} pass={gs.get('pass')}",
        "",
        f"## P3-B2-1 success: **{pass_b21}**",
        f"## proceed_to_p4a: **{pass_b21}**",
        f"## proceed_to_p3c: **False**",
        f"## stop_deterministic_full_psd: **{stop_det}**",
        "",
        f"FFR/FCR: {json.dumps(sanitize(ffr))}",
        "",
        "If A≈oracle, cosine>0.95, DI still high: deterministic full-PSD reconstruction is insufficient.",
        "Do not enlarge the network. Do not add history.",
        "",
    ]
    (iae / "p3b2_intent_shield" / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    master = iae / "MASTER_REPORT.md"
    prev = master.read_text(encoding="utf-8") if master.exists() else "# Intent-Preserving Autonomous Execution\n\nVision-free. Human specifies WHAT. Robot decides HOW.\n\n"
    marker = "# P3-B2-1"
    if marker in prev:
        prev = prev.split(marker)[0].rstrip() + "\n\n"
    master.write_text(prev + "\n".join(lines), encoding="utf-8")
    return pass_b21


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/intent_autonomous_execution")
    args = ap.parse_args()
    iae = Path(args.root)
    rows = _load_rows(iae)
    methods = [m for m in ("oracle", "B1", "B21") if rows and m in rows[0]["methods"]]
    seen = [r for r in rows if r.get("terrain") in SEEN]
    slip = [r for r in rows if r.get("terrain") in HELD]
    seen_sum = _summarize(seen, methods)
    slip_sum = _summarize(slip, methods)
    gates = {
        "B21_seen": _gate(seen_sum.get("B21") or {}, seen_sum.get("oracle") or {}),
        "B21_slip": _gate(slip_sum.get("B21") or {}, slip_sum.get("oracle") or {}, a_ratio=0.7, di_mult=2.0),
        "B1_seen": _gate(seen_sum.get("B1") or {}, seen_sum.get("oracle") or {}),
    }
    selected = {}
    sel_p = iae / "p3b2_intent_shield" / "selected.json"
    if sel_p.exists():
        selected = json.loads(sel_p.read_text())
    ffr = {}
    ck = (selected.get("B21") or {}).get("ckpt")
    if ck:
        try:
            ffr = _offline_ffr(Path(ck))
        except Exception as e:
            ffr = {"error": str(e)}
    g = gates["B21_seen"]
    stop_det = bool(
        (g.get("median_A_ratio") or 0) >= 0.95
        and (g.get("cos") or 0) >= 0.95
        and not g.get("DI_ok")
    )
    pass_b21 = bool(g.get("pass") and gates["B21_slip"].get("pass"))
    try:
        _plots(iae, methods, seen_sum, slip_sum)
    except Exception as e:
        print(f"[b21-analyze] plot failed {e}", flush=True)
    write_report(iae, methods, seen_sum, slip_sum, gates, ffr, selected, stop_det)
    payload = {
        "n_seen": len(seen),
        "n_slip": len(slip),
        "seen": seen_sum,
        "slip": slip_sum,
        "gates": gates,
        "ffr": ffr,
        "selected": selected,
        "pass_b21": pass_b21,
        "proceed_to_p4a": pass_b21,
        "proceed_to_p3c": False,
        "stop_deterministic_full_psd": stop_det,
        "no_terrain_in_model": True,
    }
    (iae / "p3b2_intent_shield" / "metrics.json").write_text(json.dumps(sanitize(payload), indent=2), encoding="utf-8")
    (iae / "summary_metrics.json").write_text(json.dumps(sanitize({"p3b21": payload}), indent=2), encoding="utf-8")
    with (iae / "p3b2_intent_shield" / "seen_eval.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_A", "median_A", "P_Agt0", "median_DI", "p90_DI", "Rd", "leak", "cos"])
        for m in methods:
            s = seen_sum.get(m) or {}
            w.writerow([m, (s.get("A") or {}).get("mean"), (s.get("A") or {}).get("median"),
                        (s.get("A") or {}).get("P_Agt0"), (s.get("DI_5deg") or {}).get("median"),
                        (s.get("DI_5deg") or {}).get("p90"), (s.get("Rd") or {}).get("median"),
                        (s.get("oracle_leak") or {}).get("median"), (s.get("cos_ucr") or {}).get("median")])
    print(json.dumps(sanitize({"gates": gates, "pass_b21": pass_b21, "stop_det": stop_det}), indent=2), flush=True)
    if pass_b21:
        print("[b21-analyze] B2-1 PASSED. P4-A may start.", flush=True)
    elif stop_det:
        print("[b21-analyze] STOP Intent Shield: A≈oracle, cosine high, DI still fails. Do not enlarge net.", flush=True)
    else:
        print("[b21-analyze] B2-1 did not pass. Do not start P4-A.", flush=True)


if __name__ == "__main__":
    main()
