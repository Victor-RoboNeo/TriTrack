#!/usr/bin/env python3
"""P3-A: select global soft lambda, plots, REPORT. No Isaac."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from p3_common import SOFT_LAMBDAS, adv_block, sanitize, stats

SEEN = ("plane", "slope", "slope_down", "light_rough", "steps")
HELD = ("slip",)
ALL_T = SEEN + HELD


def _load_rows(root: Path) -> list[dict]:
    rows = []
    for t in ALL_T:
        p = root / "p3a_oracle_projector" / f"{t}.json"
        if not p.exists():
            print(f"[p3a-analyze] missing {p}", flush=True)
            continue
        data = json.loads(p.read_text())
        rows.extend(data.get("rows") or [])
    return rows


def _split_rows(rows: list[dict], seed: int = 2026) -> list[dict]:
    seen_eps = sorted({r["episode_id"] for r in rows if r.get("terrain") in SEEN})
    rng = np.random.RandomState(seed)
    rng.shuffle(seen_eps)
    n_va = int(round(0.40 * len(seen_eps)))
    val_set = set(seen_eps[:n_va])
    out = []
    for r in rows:
        r = dict(r)
        if r.get("terrain") in HELD:
            r["p3_split"] = "test_slip"
        elif r["episode_id"] in val_set:
            r["p3_split"] = "val"
        else:
            r["p3_split"] = "test"
        out.append(r)
    return out


def _method_names(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    return list(rows[0]["methods"].keys())


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
            "DB_5deg": stats(_col(rows, name, "DB_5deg")),
            "Rd": stats(_col(rows, name, "Rd")),
            "dE": stats(_col(rows, name, "dE")),
            "P_suppressed": float(np.mean(_col(rows, name, "suppressed"))) if rows else None,
        }
    return out


def _select_lambda(val_sum: dict, methods: list[str]) -> dict:
    raw = val_sum.get("A1_full") or {}
    di_raw = (raw.get("DI_5deg") or {}).get("median")
    a_full = (raw.get("A") or {}).get("median")
    if di_raw is None:
        return {"lambda_soft_primary": None, "reason": "no A1_full on val"}
    thresh = 0.25 * float(di_raw)
    cands = []
    for name in methods:
        if not name.startswith("soft_"):
            continue
        lam = float(name.split("_", 1)[1])
        di = (val_sum[name]["DI_5deg"] or {}).get("median")
        a = (val_sum[name]["A"] or {}).get("median")
        p = (val_sum[name]["A"] or {}).get("P_Agt0")
        p90 = (val_sum[name]["DI_5deg"] or {}).get("p90")
        if di is None or a is None:
            continue
        if di <= thresh:
            cands.append((lam, a, p or 0.0, -(p90 or 0.0), di, name))
    if not cands:
        # fallback: smallest DI among soft, report failure of the rule
        best = None
        for name in methods:
            if not name.startswith("soft_"):
                continue
            di = (val_sum[name]["DI_5deg"] or {}).get("median")
            a = (val_sum[name]["A"] or {}).get("median")
            if di is None:
                continue
            rec = (float(name.split("_", 1)[1]), a or -1e9, di, name)
            if best is None or rec[2] < best[2]:
                best = rec
        return {
            "lambda_soft_primary": None if best is None else best[0],
            "selected_method": None if best is None else best[3],
            "rule_satisfied": False,
            "DI_raw_median": di_raw,
            "DI_threshold": thresh,
            "A_full_median": a_full,
            "fallback": "no lambda met 0.25x DI rule; recorded lowest-DI soft lambda",
        }
    cands.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
    lam, a, p, negp90, di, name = cands[0]
    return {
        "lambda_soft_primary": lam,
        "selected_method": name,
        "rule_satisfied": True,
        "DI_raw_median": di_raw,
        "DI_threshold": thresh,
        "A_full_median": a_full,
        "selected_DI_median": di,
        "selected_A_median": a,
        "selected_P_Agt0": p,
    }


def _success(sel: dict, test_sum: dict) -> dict:
    name = sel.get("selected_method")
    full = test_sum.get("A1_full") or {}
    soft = test_sum.get(name or "") or {}
    di_s = (soft.get("DI_5deg") or {}).get("median")
    di_u = (full.get("DI_5deg") or {}).get("median")
    a_s = (soft.get("A") or {})
    a_u = (full.get("A") or {})
    ratio_di = (di_u / di_s) if di_s and di_u else None
    ret_med = (a_s.get("median") / a_u["median"]) if a_s.get("median") is not None and a_u.get("median") else None
    ret_mean = (a_s.get("mean") / a_u["mean"]) if a_s.get("mean") is not None and a_u.get("mean") else None
    cond_a = bool(ratio_di is not None and ratio_di >= 4.0)
    cond_b = bool((ret_med is not None and ret_med >= 0.5) or (ret_mean is not None and ret_mean >= 0.6))
    return {
        "condition_A_4x_DI": cond_a,
        "condition_B_A_retention": cond_b,
        "proceed_to_p3b": bool(cond_a and cond_b and sel.get("rule_satisfied")),
        "DI_reduction": ratio_di,
        "A_median_retention": ret_med,
        "A_mean_retention": ret_mean,
    }


def _write_csv(path: Path, methods, blk: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_A", "median_A", "P_Agt0", "P_Agt0_02", "P_Agt0_05",
                    "median_DI_5deg", "p90_DI_5deg", "median_DI_peak", "median_Rd", "median_DB"])
        for name in methods:
            s = blk.get(name) or {}
            w.writerow([
                name,
                (s.get("A") or {}).get("mean"),
                (s.get("A") or {}).get("median"),
                (s.get("A") or {}).get("P_Agt0"),
                (s.get("A") or {}).get("P_Agt0_02"),
                (s.get("A") or {}).get("P_Agt0_05"),
                (s.get("DI_5deg") or {}).get("median"),
                (s.get("DI_5deg") or {}).get("p90"),
                (s.get("DI_peak") or {}).get("median"),
                (s.get("Rd") or {}).get("median"),
                (s.get("DB_5deg") or {}).get("median"),
            ])


def _plots(root: Path, methods, val_sum, test_sum, slip_sum, sel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdir = root / "p3a_oracle_projector" / "plots"
    pdir.mkdir(parents=True, exist_ok=True)

    def scatter(ax, blk, label_prefix=""):
        order = ["A1_full"] + [n for n in methods if n.startswith("A2_hard_")] + [n for n in methods if n.startswith("soft_")]
        for name in order:
            s = blk.get(name) or {}
            x = (s.get("DI_5deg") or {}).get("median")
            y = (s.get("A") or {}).get("median")
            if x is None or y is None:
                continue
            ax.scatter(x, y, s=70, label=f"{label_prefix}{name}")

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter(ax, val_sum)
    ax.set_xlabel("median DI @5°")
    ax.set_ylabel("median A")
    ax.set_title("Plot A1  recovery vs intent Pareto (val)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "plotA1_pareto.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    lams, ps = [], []
    for name in methods:
        if not name.startswith("soft_"):
            continue
        lams.append(float(name.split("_", 1)[1]))
        ps.append((val_sum.get(name, {}).get("A") or {}).get("P_Agt0"))
    ax.semilogx(lams, ps, marker="o")
    ax.set_xlabel("lambda_soft")
    ax.set_ylabel("P(A>0)")
    ax.set_title("Plot A2  P(A>0) vs lambda (val)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "plotA2_pA_vs_lambda.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in methods:
        if not name.startswith("soft_") and not name.startswith("A2_hard") and name != "A1_full":
            continue
        s = val_sum.get(name) or {}
        x = (s.get("DI_5deg") or {}).get("median")
        y = (s.get("Rd") or {}).get("median")
        if x is None or y is None:
            continue
        ax.scatter(x, y, s=70, label=name)
    ax.set_xlabel("median DI @5°")
    ax.set_ylabel("median Rd")
    ax.set_title("Plot A3  projection retention vs interference")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "plotA3_Rd_vs_DI.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter(ax, test_sum, "seen:")
    scatter(ax, slip_sum, "slip:")
    ax.set_xlabel("median DI @5°")
    ax.set_ylabel("median A")
    ax.set_title("Plot A4  seen test vs held-out slip")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdir / "plotA4_seen_vs_slip.png", dpi=140)
    plt.close(fig)


def _fmt(st, key="median"):
    if not st or st.get(key) is None:
        return "n/a"
    return f"{st[key]:.4f}"


def write_report(root: Path, sel: dict, succ: dict, val_sum, test_sum, slip_sum, methods):
    name = sel.get("selected_method")
    hard = test_sum.get("A2_hard_0.01") or test_sum.get("A2_hard_0.01") or {}
    for k in methods:
        if k.startswith("A2_hard") and "0.01" in k:
            hard = test_sum.get(k) or hard
    full = test_sum.get("A1_full") or {}
    soft = test_sum.get(name or "") or {}
    hard_a = (hard.get("A") or {}).get("mean")
    full_a = (full.get("A") or {}).get("mean")
    soft_a = (soft.get("A") or {}).get("mean")
    q1 = (
        f"是。hard PI 把 5° D_I 中位压到 {_fmt(hard.get('DI_5deg'))}（raw UCR {_fmt(full.get('DI_5deg'))}），"
        f"但 mean A 从 {_fmt(full.get('A'), 'mean')} 降到 {_fmt(hard.get('A'), 'mean')}，Rd={_fmt(hard.get('Rd'))}。"
        if hard else "hard 结果缺失。"
    )
    q2 = (
        f"soft λ={sel.get('lambda_soft_primary')} 在 test 上 median A={_fmt(soft.get('A'))}，"
        f"保留比 median={succ.get('A_median_retention')} mean={succ.get('A_mean_retention')}；"
        f"D_I 降幅 {succ.get('DI_reduction')}×。"
        if soft else "soft 未选出。"
    )
    q3 = f"validation 选出 λ={sel.get('lambda_soft_primary')}（{sel.get('selected_method')}），rule_satisfied={sel.get('rule_satisfied')}。"
    slip_s = slip_sum.get(name or "") or {}
    q4 = f"slip 上该 λ 的 median A={_fmt(slip_s.get('A'))}，D_I={_fmt(slip_s.get('DI_5deg'))}，P(A>0)={(slip_s.get('A') or {}).get('P_Agt0')}。"
    q5 = f"test 上 Rd 中位={_fmt(soft.get('Rd'))}。接近 0 表示方向被删掉；接近 1 表示几乎未投影。"
    q6 = (
        "若 Rd 仍高但 A 掉：意图与恢复在潜空间冲突。"
        "若 Rd 很低：投影把恢复方向删掉，不是数值病态。"
        f" 当前 Rd={_fmt(soft.get('Rd'))}，A 保留={succ.get('A_median_retention')}。"
    )
    lines = [
        "# P3-A — Oracle Hard/Soft Intent Projector",
        "",
        "Frozen Stage-2。地形只作切片。A = E_parent − E_pert（0.5s，burst=5）。",
        "λ 只在 seen-val 上按「D_I ≤ 0.25× raw UCR」且最大化 median A 选出，然后冻结。",
        "",
        "## 六个问题",
        "",
        f"1. **hard PI 是否过猛？** {q1}",
        f"2. **soft PI 是否找回有意义的 UCR 权威？** {q2}",
        f"3. **全局 λ？** {q3}",
        f"4. **同一 λ 在 held-out slip？** {q4}",
        f"5. **原恢复方向被去掉多少？** {q5}",
        f"6. **主要限制是意图/恢复冲突还是投影数值？** {q6}",
        "",
        f"## 是否进入 P3-B: **{succ.get('proceed_to_p3b')}**",
        "",
        f"- Condition A (≥4× D_I 下降): {succ.get('condition_A_4x_DI')} (reduction={succ.get('DI_reduction')})",
        f"- Condition B (A 保留 ≥0.5 med 或 ≥0.6 mean): {succ.get('condition_B_A_retention')} "
        f"(med={succ.get('A_median_retention')}, mean={succ.get('A_mean_retention')})",
        "",
        "## Test（seen，不含 slip）",
        "",
        "| method | mean A | median A | P(A>0) | P(A>0.05) | D_I 5° med | Rd med |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n in methods:
        s = test_sum.get(n) or {}
        lines.append(
            f"| {n} | {_fmt(s.get('A'), 'mean')} | {_fmt(s.get('A'))} | "
            f"{(s.get('A') or {}).get('P_Agt0')} | {(s.get('A') or {}).get('P_Agt0_05')} | "
            f"{_fmt(s.get('DI_5deg'))} | {_fmt(s.get('Rd'))} |"
        )
    lines += ["", "## Held-out slip", "",
              "| method | mean A | median A | P(A>0) | D_I 5° med |",
              "|---|---:|---:|---:|---:|"]
    for n in methods:
        s = slip_sum.get(n) or {}
        lines.append(
            f"| {n} | {_fmt(s.get('A'), 'mean')} | {_fmt(s.get('A'))} | {(s.get('A') or {}).get('P_Agt0')} | {_fmt(s.get('DI_5deg'))} |"
        )
    lines += ["", "未通过则停止，不训练 learned projector，不进入 P3-C。", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/p3_intent_projected_adaptation")
    args = ap.parse_args()
    root = Path(args.root)
    rows = _split_rows(_load_rows(root))
    methods = _method_names(rows)
    val = [r for r in rows if r.get("p3_split") == "val"]
    test = [r for r in rows if r.get("p3_split") == "test"]
    slip = [r for r in rows if r.get("p3_split") == "test_slip"]
    val_sum = _summarize(val, methods)
    test_sum = _summarize(test, methods)
    slip_sum = _summarize(slip, methods)
    all_sum = _summarize(rows, methods)
    sel = _select_lambda(val_sum, methods)
    succ = _success(sel, test_sum)
    metrics = {
        "n_val": len(val),
        "n_test": len(test),
        "n_slip": len(slip),
        "selection": sel,
        "success": succ,
        "val": val_sum,
        "test": test_sum,
        "slip": slip_sum,
        "all": all_sum,
        "no_terrain_in_model": True,
    }
    (root / "metrics.json").write_text(json.dumps(sanitize(metrics), indent=2), encoding="utf-8")
    _write_csv(root / "p3a_oracle_projector" / "lambda_sweep.csv", methods, val_sum)
    _write_csv(root / "p3a_oracle_projector" / "pareto.csv", methods, test_sum)
    (root / "p3a_oracle_projector" / "lambda_selected.json").write_text(
        json.dumps(sanitize({"selection": sel, "success": succ}), indent=2), encoding="utf-8"
    )
    try:
        _plots(root, methods, val_sum, test_sum, slip_sum, sel)
    except Exception as e:
        print(f"[p3a-analyze] plot failed {e}", flush=True)
    write_report(root, sel, succ, val_sum, test_sum, slip_sum, methods)
    print(json.dumps(sanitize({"selection": sel, "success": succ}), indent=2), flush=True)
    if not succ.get("proceed_to_p3b"):
        print("[p3a-analyze] STOP after P3-A (success conditions not met).", flush=True)


if __name__ == "__main__":
    main()
