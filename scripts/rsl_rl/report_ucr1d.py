#!/usr/bin/env python3
"""UCR-1D report: identifiability of beneficial recovery directions from o_R.

No UCR-2. No closed-loop. Decision tree on pooled vs same-task kNN vs MLP.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TASKS = ("loco", "stoop", "reach", "carry")
KNN = ("knn_pooled", "knn_same", "knn_from_loco", "knn_from_stoop", "knn_from_reach", "knn_from_carry")
GO_P = {"loco": 0.70, "stoop": 0.65, "reach": 0.65, "carry": 0.65}


def _j(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    raise TypeError(type(o))


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


def _fmt(v, cm=False):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if cm:
        return f"{v:.2f} cm"
    return f"{v:.3f}"


def _high(blk: dict, task: str) -> bool:
    p = blk.get("P_I_lt_0")
    med = blk.get("median_cm")
    return isinstance(p, float) and p >= GO_P[task] and isinstance(med, float) and med < 0


def load_probe(root: Path) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, list]] = {t: {} for t in TASKS}
    fd_n_eq = {t: [] for t in TASKS}
    fd_n_eq1 = {t: [] for t in TASKS}
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
            if name.startswith("fd_") and name not in ("fd_best",):
                continue
            out[task].setdefault(name, []).append(np.asarray(b[k], dtype=np.float32))
        if "i_fd" in b.files and "i_oracle" in b.files:
            fd = np.asarray(b["i_fd"], dtype=np.float64)
            star = np.asarray(b["i_oracle"], dtype=np.float64)[:, None]
            n_eq = (fd <= star + 0.0025).sum(axis=1)
            n_eq1 = (fd <= fd.min(axis=1, keepdims=True) + 0.0025).sum(axis=1)
            fd_n_eq[task].append(n_eq)
            fd_n_eq1[task].append(n_eq1)
    packed = {}
    for t, methods in out.items():
        packed[t] = {n: np.concatenate(vs, 0) if vs else np.zeros(0, dtype=np.float32) for n, vs in methods.items()}
    fd_stats = {}
    for t in TASKS:
        if not fd_n_eq[t]:
            fd_stats[t] = {"n": 0}
            continue
        a = np.concatenate(fd_n_eq[t])
        b = np.concatenate(fd_n_eq1[t])
        fd_stats[t] = {
            "n": int(a.size),
            "mean_n_eq_vs_oracle": float(a.mean()),
            "P_n_eq_ge_2_vs_oracle": float((a >= 2).mean()),
            "median_n_eq_vs_oracle": float(np.median(a)),
            "mean_n_eq_vs_best_1deg": float(b.mean()),
            "P_n_eq_ge_2_vs_best_1deg": float((b >= 2).mean()),
        }
    return packed, fd_stats


def load_mlp(ucr1: Path) -> dict[str, np.ndarray]:
    """scratch_s0_last I from UCR-1 heldout."""
    out: dict[str, list] = {t: [] for t in TASKS}
    for p in sorted((ucr1 / "clone_eval").glob("*/lam_*/plane/heldout.npz")):
        b = np.load(p, allow_pickle=True)
        task = None
        for part in p.parts:
            if part in TASKS:
                task = part
                break
        if task is None or "i_scratch_s0_last" not in b.files:
            continue
        out[task].append(np.asarray(b["i_scratch_s0_last"], dtype=np.float32))
    return {t: np.concatenate(vs, 0) if vs else np.zeros(0, dtype=np.float32) for t, vs in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--offline", required=True)
    ap.add_argument("--ucr1", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_task, fd_stats = load_probe(Path(args.probe))
    mlp = load_mlp(Path(args.ucr1))
    offline = json.loads(Path(args.offline).read_text()) if Path(args.offline).is_file() else {}

    table_knn = []
    matrix = {tgt: {} for tgt in TASKS}
    flags = {t: {} for t in TASKS}
    for t in TASKS:
        pooled = _blk(per_task.get(t, {}).get("knn_pooled", np.zeros(0)))
        same = _blk(per_task.get(t, {}).get("knn_same", np.zeros(0)))
        neu = _blk(mlp.get(t, np.zeros(0)))
        old = _blk(per_task.get(t, {}).get("old_rm3", np.zeros(0)))
        ora = _blk(per_task.get(t, {}).get("oracle", np.zeros(0)))
        fdb = _blk(per_task.get(t, {}).get("fd_best", np.zeros(0)))
        table_knn.append({
            "task": t, "n": pooled.get("n", 0),
            "old": old, "mlp": neu, "knn_pooled": pooled, "knn_same": same,
            "oracle": ora, "fd_best_1deg": fdb,
        })
        flags[t] = {
            "pooled_high": _high(pooled, t),
            "same_high": _high(same, t),
            "mlp_high": _high(neu, t),
            "oracle_strong": isinstance(ora.get("P_I_lt_0"), float) and ora["P_I_lt_0"] >= 0.90,
        }
        for src in TASKS:
            matrix[t][src] = _blk(per_task.get(t, {}).get(f"knn_from_{src}", np.zeros(0)))

    n_pooled = sum(1 for t in TASKS if flags[t]["pooled_high"])
    n_same = sum(1 for t in TASKS if flags[t]["same_high"])
    n_mlp = sum(1 for t in TASKS if flags[t]["mlp_high"])
    pooled_better_mlp = []
    for t in TASKS:
        p = table_knn[TASKS.index(t)]["knn_pooled"].get("P_I_lt_0")
        m = table_knn[TASKS.index(t)]["mlp"].get("P_I_lt_0")
        if isinstance(p, float) and isinstance(m, float):
            pooled_better_mlp.append(p > m + 0.05)

    if n_pooled >= 3 and n_mlp <= 1:
        branch = "objective"
        verdict = "SHARED_FIELD_IDENTIFIABLE_OBJECTIVE_BROKEN"
        note = (
            "Pooled kNN recovers beneficial directions from unified o_R. "
            "The cosine+CE unique-d* objective is averaging a non-unique field. "
            "Next: utility / set-valued supervision, not task experts."
        )
    elif n_same >= 3 and n_pooled <= 1:
        branch = "representation"
        verdict = "MANIFOLDS_NOT_SEPARATED_IN_o_R"
        note = (
            "Same-task kNN works; pooled kNN does not. Current (e,ė,M,proprio,z_nom) "
            "does not separate intent manifolds. Next: richer intent representation, not task ID."
        )
    elif n_same <= 1 and n_pooled <= 1:
        branch = "not_identifiable"
        verdict = "STATIC_o_R_TO_d_NOT_IDENTIFIABLE"
        note = (
            "Even same-task kNN fails on RE-trigger clones. Static o_R→d is not locally identifiable. "
            "Do not enlarge the MLP. Revisit process state / history, not UCR-2 training."
        )
    elif n_pooled >= 3 and n_same >= 3 and n_mlp <= 1:
        branch = "objective"
        verdict = "SHARED_FIELD_IDENTIFIABLE_OBJECTIVE_BROKEN"
        note = (
            "Both kNN probes work; the trained MLP does not. Easiest save: change the supervised objective."
        )
    else:
        branch = "mixed"
        verdict = "MIXED_IDENTIFIABILITY"
        note = "See per-task flags. Do not add task experts. Do not start UCR-2 tonight."

    payload = {
        "step": "UCR-1D",
        "no_new_controller": True,
        "no_ucr2": True,
        "branch": branch,
        "verdict": verdict,
        "note": note,
        "n_tasks_high": {"knn_pooled": n_pooled, "knn_same": n_same, "mlp": n_mlp},
        "per_task": table_knn,
        "matrix_source_to_target": matrix,
        "flags": flags,
        "fd_label_set": fd_stats,
        "offline": {
            "pooled_nn_source": offline.get("pooled_nn_source"),
            "cos_neighbor_d": offline.get("cos_neighbor_d_vs_own_oracle"),
            "k5_pairwise": offline.get("k5_neighbor_d_pairwise_cosine"),
            "magnitude_ambiguity": offline.get("magnitude_ambiguity_0.25cm"),
        },
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, default=_j))

    lines = [
        "# UCR-1D — Recovery Field Identifiability",
        "",
        f"**{verdict}** (`{branch}`)",
        note,
        "",
        "No new controller. No task ID in the method. No UCR-2.",
        "",
        "## 1. kNN clone utility vs MLP / oracle",
        "",
        "| Task | n | Old R-M3 | MLP | pooled kNN | same-task kNN | Oracle |",
        "|------|---|----------|-----|------------|---------------|--------|",
    ]
    for r in table_knn:
        lines.append(
            f"| {r['task']} | {r['n']} | {_fmt(r['old'].get('P_I_lt_0'))} | {_fmt(r['mlp'].get('P_I_lt_0'))} "
            f"| {_fmt(r['knn_pooled'].get('P_I_lt_0'))} | {_fmt(r['knn_same'].get('P_I_lt_0'))} "
            f"| {_fmt(r['oracle'].get('P_I_lt_0'))} |"
        )
    lines += [
        "",
        "| Task | pooled median I | same median I | MLP median I | MLP P(I>+0.5) |",
        "|------|-----------------|---------------|--------------|---------------|",
    ]
    for r in table_knn:
        lines.append(
            f"| {r['task']} | {_fmt(r['knn_pooled'].get('median_cm'), cm=True)} "
            f"| {_fmt(r['knn_same'].get('median_cm'), cm=True)} "
            f"| {_fmt(r['mlp'].get('median_cm'), cm=True)} "
            f"| {_fmt(r['mlp'].get('P_I_gt_0.5cm'))} |"
        )
    lines += [
        "",
        "## 2. Cross-task oracle compatibility  P(I<0)  (source field → target clones)",
        "",
        "| target \\ source | Loco | Stoop | Reach | Carry |",
        "|------------------|------|-------|-------|-------|",
    ]
    for tgt in TASKS:
        cells = " | ".join(_fmt(matrix[tgt][s].get("P_I_lt_0")) for s in TASKS)
        lines.append(f"| {tgt} | {cells} |")
    lines += [
        "",
        "Median I (cm):",
        "",
        "| target \\ source | Loco | Stoop | Reach | Carry |",
        "|------------------|------|-------|-------|-------|",
    ]
    for tgt in TASKS:
        cells = " | ".join(_fmt(matrix[tgt][s].get("median_cm"), cm=True) for s in TASKS)
        lines.append(f"| {tgt} | {cells} |")
    lines += ["", "## 3. Label-set width (±1° FD, I(d) ≤ I* + 0.25 cm)", ""]
    for t in TASKS:
        s = fd_stats.get(t) or {}
        lines.append(
            f"- {t}: n={s.get('n', 0)} mean |{{d: I≤I*+0.25cm}}|={_fmt(s.get('mean_n_eq_vs_oracle'))} "
            f"P(≥2)={_fmt(s.get('P_n_eq_ge_2_vs_oracle'))} "
            f"(vs best-1° mean={_fmt(s.get('mean_n_eq_vs_best_1deg'))})"
        )
    src = (offline.get("pooled_nn_source") or {}).get("stoop") or {}
    lines += [
        "",
        "## 4. Why Stoop broke: pooled NN source (offline, no task ID)",
        "",
        f"Stoop test neighbors come from: {json.dumps(src)}",
        f"frac_other={src.get('frac_other')}",
        "",
        f"Decision: **{branch}**. Do not train UCR-2. Do not add task experts.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        M = np.array([[matrix[tgt][s].get("P_I_lt_0") or 0 for s in TASKS] for tgt in TASKS], dtype=float)
        im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(4), TASKS)
        ax.set_yticks(range(4), TASKS)
        ax.set_xlabel("source train oracle field")
        ax.set_ylabel("target test clones")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, label="P(I<0)")
        fig.tight_layout()
        (out / "plots").mkdir(exist_ok=True)
        fig.savefig(out / "plots" / "compat_matrix.png", dpi=140)
        print("wrote plots/compat_matrix.png", flush=True)
    except Exception as e:
        print("plot skipped", e, flush=True)


if __name__ == "__main__":
    main()
