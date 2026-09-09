#!/usr/bin/env python3
"""UCR-2P report: process kNN clone utility vs static UCR-1D. No GRU unless process wins."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TASKS = ("loco", "stoop", "reach", "carry")
PROC = ("knn_p200", "knn_p500", "knn_p200_same", "knn_p500_same", "knn_p200_sum", "knn_p500_sum")
STATIC_UCR1D = {
    "loco": {"n": 62, "P_I_lt_0": 0.597, "median_cm": None},
    "stoop": {"n": 79, "P_I_lt_0": 0.342, "median_cm": None},
    "reach": {"n": 60, "P_I_lt_0": 0.433, "median_cm": None},
    "carry": {"n": 55, "P_I_lt_0": 0.127, "median_cm": None},
}
STATIC_SAME = {"loco": 0.645, "stoop": 0.392, "reach": 0.367, "carry": 0.164}
ORACLE = {t: 1.0 for t in TASKS}


def _blk(i_m: np.ndarray) -> dict:
    x = np.asarray(i_m, dtype=np.float64) * 100.0
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "P_I_lt_0": float("nan"), "P_I_lt_0.25cm": float("nan"),
                "median_cm": float("nan"), "mean_cm": float("nan"),
                "P_I_gt_0.5cm": float("nan"), "P_I_gt_1.0cm": float("nan")}
    return {
        "n": int(x.size),
        "P_I_lt_0": float((x < 0).mean()),
        "P_I_lt_0.25cm": float((x < -0.25).mean()),
        "median_cm": float(np.median(x)),
        "mean_cm": float(x.mean()),
        "P_I_gt_0.5cm": float((x > 0.5).mean()),
        "P_I_gt_1.0cm": float((x > 1.0).mean()),
    }


def load_probe(root: Path) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, list]] = {t: {} for t in TASKS}
    for p in sorted(root.glob("*/lam_*/plane/probe.npz")):
        b = np.load(p, allow_pickle=True)
        task = None
        for part in p.parts:
            if part in TASKS:
                task = part
                break
        if task is None:
            continue
        n = int(b["i_oracle"].shape[0]) if "i_oracle" in b.files else 0
        if n == 0:
            continue
        for k in b.files:
            if not k.startswith("i_"):
                continue
            name = k[2:]
            out[task].setdefault(name, []).append(np.asarray(b[k], dtype=np.float32))
    packed = {}
    for t, methods in out.items():
        packed[t] = {n: np.concatenate(vs, 0) if vs else np.zeros(0, dtype=np.float32) for n, vs in methods.items()}
    return packed


def _fmt(v, cm=False):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if cm:
        return f"{v:.2f}"
    return f"{v:.3f}"


def pooled_p(per_task: dict, method: str) -> float:
    xs = []
    for t in TASKS:
        a = per_task.get(t, {}).get(method)
        if a is None or a.size == 0:
            continue
        xs.append(np.asarray(a, dtype=np.float64))
    if not xs:
        return float("nan")
    x = np.concatenate(xs)
    x = x[np.isfinite(x)]
    return float((x < 0).mean()) if x.size else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--offline", required=True)
    ap.add_argument("--ucr1d", default="/data/home/chenxiangyu/robotics/Anybody/results/ucr1d_identifiability")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_task = load_probe(Path(args.probe))
    offline = json.loads(Path(args.offline).read_text()) if Path(args.offline).is_file() else {}

    rows = []
    methods_show = ["oracle"] + list(PROC)
    table = {m: {} for m in ["static", "static_same", "p200", "p500", "p200_same", "p500_same", "oracle"]}
    for t in TASKS:
        table["static"][t] = STATIC_UCR1D[t]["P_I_lt_0"]
        table["static_same"][t] = STATIC_SAME[t]
        table["oracle"][t] = 1.0
        table["p200"][t] = _blk(per_task.get(t, {}).get("knn_p200", np.zeros(0)))["P_I_lt_0"]
        table["p500"][t] = _blk(per_task.get(t, {}).get("knn_p500", np.zeros(0)))["P_I_lt_0"]
        table["p200_same"][t] = _blk(per_task.get(t, {}).get("knn_p200_same", np.zeros(0)))["P_I_lt_0"]
        table["p500_same"][t] = _blk(per_task.get(t, {}).get("knn_p500_same", np.zeros(0)))["P_I_lt_0"]

    # GO: pooled process clearly above static, no task collapse
    p_static = float(np.nanmean([STATIC_UCR1D[t]["P_I_lt_0"] for t in TASKS]))
    p_p500 = pooled_p(per_task, "knn_p500")
    p_p200 = pooled_p(per_task, "knn_p200")
    p_same500 = pooled_p(per_task, "knn_p500_same")
    collapse = any(
        (isinstance(table["p500"][t], float) and table["p500"][t] + 0.05 < STATIC_UCR1D[t]["P_I_lt_0"])
        for t in TASKS
    )
    better = (isinstance(p_p500, float) and not math.isnan(p_p500) and p_p500 >= p_static + 0.08)
    same_also_bad = (
        isinstance(p_same500, float) and not math.isnan(p_same500) and p_same500 < p_static + 0.05
        and isinstance(p_p500, float) and p_p500 < p_static + 0.05
    )
    if better and not collapse:
        verdict = "PROCESS_IDENTIFIABLE"
    else:
        verdict = "RECOVERY_NOT_IDENTIFIABLE"
    allow_gru = better and not collapse

    md = []
    md.append("# UCR-2P Process-Conditioned Recovery Identifiability\n")
    md.append("No closed-loop. No task ID in method. Tiny GRU only if process kNN beats static.\n")
    md.append("| Representation | Loco P(I<0) | Stoop | Reach | Carry | pooled |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    def line(name, getter):
        vals = [getter(t) for t in TASKS]
        pool = float(np.nanmean(vals)) if any(isinstance(v, float) and not math.isnan(v) for v in vals) else float("nan")
        md.append(
            f"| {name} | " + " | ".join(_fmt(v) for v in vals) + f" | {_fmt(pool)} |\n"
        )
        return pool

    line("static oR (UCR-1D)", lambda t: table["static"][t])
    line("200ms process kNN", lambda t: table["p200"][t])
    line("500ms process kNN", lambda t: table["p500"][t])
    line("200ms same-task (diag)", lambda t: table["p200_same"][t])
    line("500ms same-task (diag)", lambda t: table["p500_same"][t])
    line("oracle", lambda t: 1.0)
    md.append("\n### median I (cm) and harmful tail P(I>0.5cm)\n")
    md.append("| Method | task | n | P(I<0) | P(I<-0.25cm) | median I | P(I>0.5cm) |\n")
    md.append("|---|---|---:|---:|---:|---:|---:|\n")
    for m in PROC:
        for t in TASKS:
            blk = _blk(per_task.get(t, {}).get(m, np.zeros(0)))
            md.append(
                f"| {m} | {t} | {blk['n']} | {_fmt(blk['P_I_lt_0'])} | {_fmt(blk['P_I_lt_0.25cm'])} | "
                f"{_fmt(blk['median_cm'], True)} | {_fmt(blk['P_I_gt_0.5cm'])} |\n"
            )
    md.append(f"\n**Verdict: `{verdict}`**\n")
    md.append(f"- pooled static P(I<0)={p_static:.3f}\n")
    md.append(f"- pooled 500ms process P(I<0)={_fmt(p_p500)}\n")
    md.append(f"- pooled 500ms same-task P(I<0)={_fmt(p_same500)}\n")
    md.append(f"- tiny GRU allowed: {bool(allow_gru)}\n")
    if same_also_bad:
        md.append(
            "\nSTOP: 500 ms history kNN and same-task history kNN both fail to transfer recovery utility. "
            "Learned recovery research stops. Local freedom exists (oracle P=1); unified direction is not "
            "identifiable from available causal state.\n"
        )
    if not allow_gru:
        md.append("\nNo Transformer / MoE / longer history / PPO. No tiny GRU trained.\n")
    md.append("\n```json\n")
    md.append(json.dumps({"offline": offline, "verdict": verdict, "allow_gru": allow_gru,
                          "p_static": p_static, "p_p500": p_p500, "p_p200": p_p200}, indent=2, default=str))
    md.append("\n```\n")
    (out / "REPORT.md").write_text("".join(md))
    (out / "verdict.json").write_text(json.dumps({
        "verdict": verdict, "allow_gru": allow_gru, "p_static": p_static,
        "p_p200": p_p200, "p_p500": p_p500, "p_same500": p_same500, "table": table,
    }, indent=2, default=str))
    print(f"[ucr2p] {verdict} p500={p_p500} static={p_static} gru={allow_gru}", flush=True)


if __name__ == "__main__":
    main()
