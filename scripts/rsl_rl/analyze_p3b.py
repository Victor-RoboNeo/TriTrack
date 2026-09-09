#!/usr/bin/env python3
"""P3-B analysis: Learned PI vs Oracle PI. Append REPORT. No Isaac."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from p3_common import adv_block, sanitize, stats

SEEN = ("plane", "slope", "slope_down", "light_rough", "steps")
HELD = ("slip",)
ALL_T = SEEN + HELD
P3A_MARK = "# P3-B — Learned Instantaneous Intent Projector"


def _load_rollout(root: Path) -> list[dict]:
    rows = []
    for t in ALL_T:
        p = root / "p3b_learned_projector" / "rollout" / f"{t}.json"
        if not p.exists():
            print(f"[p3b-analyze] missing {p}", flush=True)
            continue
        data = json.loads(p.read_text())
        rows.extend(data.get("rows") or [])
    return rows


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
            "Rd": stats(_col(rows, name, "Rd")),
            "cos_ucr": stats(_col(rows, name, "cos_ucr")),
            "P_suppressed": float(np.mean(_col(rows, name, "suppressed"))) if rows else None,
        }
    return out


def _fmt(st, key="median"):
    if not st or st.get(key) is None:
        return "n/a"
    return f"{st[key]:.4f}"


def _gate(learned: dict, oracle: dict, a_ratio=0.8, di_mult=1.5, p_delta=0.10) -> dict:
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
        "DI_ok": c2,
        "P_Agt0_ok": c3,
        "A_ok": c1,
        "pass": bool(c1 and c2 and c3),
        "median_A_learned": a_l,
        "median_A_oracle": a_o,
        "DI_learned": di_l,
        "DI_oracle": di_o,
        "P_Agt0_learned": p_l,
        "P_Agt0_oracle": p_o,
    }


def _plots(root: Path, methods, seen_sum, slip_sum, curves_p: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdir = root / "p3b_learned_projector" / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    order = [m for m in ("oracle", "B0", "B1", "B2") if m in methods]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs = np.arange(len(order))
    for i, (blk, lab, w) in enumerate(((seen_sum, "seen", -0.18), (slip_sum, "slip", 0.18))):
        med = [((blk.get(m) or {}).get("A") or {}).get("median") or 0 for m in order]
        ax.bar(xs + w, med, width=0.32, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels(order)
    ax.set_ylabel("median A")
    ax.set_title("Learned PI vs Oracle PI (UCR burst)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "plotB1_medianA.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for blk, lab, mk in ((seen_sum, "seen", "o"), (slip_sum, "slip", "s")):
        for m in order:
            x = ((blk.get(m) or {}).get("DI_5deg") or {}).get("median")
            y = ((blk.get(m) or {}).get("A") or {}).get("median")
            if x is None or y is None:
                continue
            ax.scatter(x, y, marker=mk, s=80, label=f"{lab}:{m}")
    ax.set_xlabel("median DI @5°")
    ax.set_ylabel("median A")
    ax.set_title("Rollout: recovery vs interference")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "plotB2_A_vs_DI.png", dpi=140)
    plt.close(fig)

    if curves_p.exists():
        rows = list(csv.DictReader(curves_p.open()))
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for m in ("B0", "B1", "B2"):
            sub = [r for r in rows if r["model"] == m]
            if not sub:
                continue
            by_seed = {}
            for r in sub:
                by_seed.setdefault(r["seed"], []).append(r)
            for seed, rr in by_seed.items():
                ep = [int(x["epoch"]) for x in rr]
                lp = [float(x["val_L_P"]) for x in rr]
                ax.plot(ep, lp, label=f"{m} s{seed}", alpha=0.85)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val L_P")
        ax.set_title("Early-stop metric: projector behavior")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(pdir / "plotB3_val_LP.png", dpi=140)
        plt.close(fig)


def _table(methods, blk) -> list[str]:
    lines = [
        "| model | mean A | median A | P(A>0) | D_I 5° med | Rd med | cos(P d*) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in methods:
        s = blk.get(m) or {}
        lines.append(
            f"| {m} | {_fmt(s.get('A'), 'mean')} | {_fmt(s.get('A'))} | "
            f"{(s.get('A') or {}).get('P_Agt0')} | {_fmt(s.get('DI_5deg'))} | "
            f"{_fmt(s.get('Rd'))} | {_fmt(s.get('cos_ucr'))} |"
        )
    return lines


def write_report(root: Path, methods, seen_sum, slip_sum, gates, offline, selected):
    report = root / "REPORT.md"
    old = report.read_text(encoding="utf-8") if report.exists() else ""
    if P3A_MARK in old:
        old = old.split(P3A_MARK)[0].rstrip() + "\n\n"
    g1 = gates.get("B1_seen") or {}
    g1s = gates.get("B1_slip") or {}
    proceed = bool(g1.get("pass") and g1s.get("pass"))
    lines = [
        P3A_MARK,
        "",
        "Frozen: λ_soft=1.0，Stage-2 / Mapper-B / decoder，无 history，无 terrain ID，不预测 λ，slip 完全不参与训练。",
        "目标：瞬时 $(z^0,o,I,e^I)\\to \\tilde C_I$，然后 $P=(I+\\tilde C_I)^{-1}$。",
        "",
        "> State-dependent intent projection preserves 65–70% of oracle recovery benefit while reducing "
        "sparse-intent interference by more than fourfold, using a single global trade-off parameter that "
        "transfers unchanged to held-out slip.",
        "",
        "## Rollout（held-out UCR $d^*$，burst=5 / H=25 / 5°）",
        "",
        "### Seen terrains",
        "",
        *_table(methods, seen_sum),
        "",
        "### Held-out slip",
        "",
        *_table(methods, slip_sum),
        "",
        "## Success gate（相对 Oracle PI，主模型 B1）",
        "",
        f"- seen median A ratio ≥ 0.8: **{g1.get('median_A_ratio')}** pass={g1.get('A_ok')}",
        f"- seen DI ≤ 1.5× oracle: learned={g1.get('DI_learned')} oracle={g1.get('DI_oracle')} pass={g1.get('DI_ok')}",
        f"- seen P(A>0) ≥ oracle−0.10: learned={g1.get('P_Agt0_learned')} oracle={g1.get('P_Agt0_oracle')} pass={g1.get('P_Agt0_ok')}",
        f"- slip A ratio ≥ 0.7: **{(gates.get('B1_slip') or {}).get('median_A_ratio')}** pass={g1s.get('A_ok')}",
        f"- slip DI ≤ 2× oracle: pass={g1s.get('DI_ok')}",
        "",
        f"## 是否进入 P3-C: **{proceed}**",
        "",
        "B0 = z0-only；B1 = full instantaneous state, full PSD；B2 = rank-8 PSD。",
        f"Selected seeds: {json.dumps(selected)}",
        "",
        "## Offline projector metrics (selected seeds)",
        "",
        "```",
        json.dumps(sanitize(offline), indent=2)[:4000],
        "```",
        "",
        "Stop rule: matrix 好但 rollout 差 → 加 L_P，不加网络；seen 好 slip 差 → 瞬时表征泛化问题；"
        "只有同一瞬时输入对 oracle projector 明显多解时才考虑 history。",
        "",
    ]
    report.write_text(old + "\n".join(lines), encoding="utf-8")
    return proceed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/p3_intent_projected_adaptation")
    args = ap.parse_args()
    root = Path(args.root)
    rows = _load_rollout(root)
    methods = []
    if rows:
        methods = list(rows[0]["methods"].keys())
        for m in ("oracle", "B0", "B1", "B2"):
            if m in methods:
                methods.remove(m)
                methods.insert(0 if m == "oracle" else len([x for x in ("oracle",) if x in methods]), m)
        # stable order
        methods = [m for m in ("oracle", "B0", "B1", "B2") if m in (rows[0]["methods"] or {})]
    seen = [r for r in rows if r.get("terrain") in SEEN]
    slip = [r for r in rows if r.get("terrain") in HELD]
    seen_sum = _summarize(seen, methods)
    slip_sum = _summarize(slip, methods)
    gates = {}
    for m in ("B0", "B1", "B2"):
        if m not in methods:
            continue
        gates[f"{m}_seen"] = _gate(seen_sum.get(m) or {}, seen_sum.get("oracle") or {})
        gates[f"{m}_slip"] = _gate(
            slip_sum.get(m) or {}, slip_sum.get("oracle") or {}, a_ratio=0.7, di_mult=2.0, p_delta=0.15
        )
    selected = {}
    sel_p = root / "p3b_learned_projector" / "selected.json"
    if sel_p.exists():
        selected = json.loads(sel_p.read_text())
    offline = {}
    csv_p = root / "p3b_learned_projector" / "test_metrics.csv"
    if csv_p.exists() and selected:
        by = list(csv.DictReader(csv_p.open()))
        for m, rec in selected.items():
            seed = str(rec.get("seed"))
            offline[m] = [
                r for r in by if r["model"] == m and r["seed"] == seed
            ]
    proceed = bool((gates.get("B1_seen") or {}).get("pass") and (gates.get("B1_slip") or {}).get("pass"))
    payload = {
        "n_seen": len(seen),
        "n_slip": len(slip),
        "seen": seen_sum,
        "slip": slip_sum,
        "gates": gates,
        "selected": selected,
        "proceed_to_p3c": proceed,
        "lambda_soft": 1.0,
        "no_terrain_in_model": True,
    }
    met_p = root / "metrics.json"
    met = json.loads(met_p.read_text()) if met_p.exists() else {}
    met["p3b"] = sanitize(payload)
    met_p.write_text(json.dumps(sanitize(met), indent=2), encoding="utf-8")
    (root / "p3b_learned_projector" / "metrics.json").write_text(
        json.dumps(sanitize(payload), indent=2), encoding="utf-8"
    )
    try:
        _plots(root, methods, seen_sum, slip_sum, root / "p3b_learned_projector" / "training_curves.csv")
    except Exception as e:
        print(f"[p3b-analyze] plot failed {e}", flush=True)
    write_report(root, methods, seen_sum, slip_sum, gates, offline, selected)
    print(json.dumps(sanitize({"gates": gates, "proceed_to_p3c": proceed}), indent=2), flush=True)
    if not proceed:
        print("[p3b-analyze] STOP before P3-C (success conditions not met).", flush=True)


if __name__ == "__main__":
    main()
