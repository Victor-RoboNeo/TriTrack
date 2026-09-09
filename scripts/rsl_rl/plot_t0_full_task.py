#!/usr/bin/env python3
"""T0 plots + summary. No Isaac."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TERRAINS = ("plane", "light_rough", "slope", "steps")
TERRAIN_LABEL = {"plane": "Flat", "light_rough": "Light", "slope": "Slope", "steps": "Steps"}
TASKS = ("loco", "stoop", "reach", "carry")
THETA_BINS = (2.5, 5.0, 7.5, 10.0)
CONSTRAINT = {"loco": (1, "T / 1-point"), "reach": (2, "T + one wrist"), "carry": (3, "T + two wrists")}


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _pct(x, qs=(10, 50, 90)):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{q}": None for q in qs} | {"n": 0, "mean": None}
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} | {"n": int(x.size), "mean": float(x.mean())}


def _rate(m):
    a = np.asarray(m, dtype=bool)
    return float(a.mean()) if a.size else None


def collect(root: Path) -> dict:
    parent, rm3, shadow = {}, {}, {}
    for task in TASKS:
        parent[task] = {ter: _load(root / "parent" / task / ter / "summary.json") for ter in TERRAINS}
        parent[task]["pooled"] = _load(root / "parent" / task / "pooled.json")
        if task in ("loco", "stoop"):
            rm3[task] = {ter: _load(root / "rm3_active" / task / ter / "summary.json") for ter in TERRAINS}
            rm3[task]["pooled"] = _load(root / "rm3_active" / task / "pooled.json")
        if task in ("reach", "carry"):
            # Reach/Carry live in shadow/ (Parent + adapter log)
            shadow[task] = {ter: _load(root / "shadow" / task / ter / "summary.json") for ter in TERRAINS}
            shadow[task]["pooled"] = _load(root / "shadow" / task / "pooled.json")
            parent[task] = shadow[task]
    return {"parent": parent, "rm3": rm3, "shadow": shadow}


def sr_task(blob, ter):
    cell = blob.get(ter) if blob else None
    if not cell:
        return None
    return _finite(cell.get("sr_task") if cell.get("sr_task") is not None else cell.get("task_completion"))


def eps_of(blob) -> list[dict]:
    rows = []
    if not blob:
        return rows
    for ter in TERRAINS:
        cell = blob.get(ter)
        if cell:
            rows.extend(cell.get("episodes") or [])
    return rows


def events_of(blob) -> list[dict]:
    rows = []
    if not blob:
        return rows
    for ter in TERRAINS:
        cell = blob.get(ter)
        if cell:
            rows.extend(cell.get("events") or [])
    return rows


def shadow_rows(root: Path, task: str) -> list[dict]:
    rows = []
    for ter in TERRAINS:
        p = root / "shadow" / task / ter / "shadow.json"
        if p.exists():
            rows.extend(json.loads(p.read_text()))
        p2 = root / "rm3_active" / task / ter / "shadow.json"
        if p2.exists():
            rows.extend(json.loads(p2.read_text()))
    return rows


def parent_matrix(data) -> dict:
    out = {}
    for task in TASKS:
        row = {ter: sr_task(data["parent"][task], ter) for ter in TERRAINS}
        vals = [v for v in row.values() if v is not None]
        row["overall"] = float(np.mean(vals)) if vals else None
        out[task] = row
    return out


def taxonomy(eps: list[dict]) -> dict:
    n = len(eps)
    keys = ("tracking", "instability", "manipulation", "timeout", "other", "success")
    counts = {k: 0 for k in keys}
    for r in eps:
        if not r.get("fail"):
            counts["success"] += 1
            continue
        t = r.get("fail_taxonomy") or "other"
        if t == "instability" or r.get("fail_reason") in ("fall", "anchor_z", "anchor_ori"):
            counts["instability"] += 1
        elif t in counts:
            counts[t] += 1
        else:
            counts["other"] += 1
    return {k: (counts[k] / n if n else None) for k in keys} | {"n": n, "counts": counts}


def case_f(eps: list[dict], events: list[dict]) -> str:
    failed = [r for r in eps if r.get("fail")]
    if not failed:
        return "F-C"
    recall = _rate([bool(r.get("trigger_before_fail")) for r in failed])
    leads = [r["lead_time_s"] for r in failed if r.get("lead_time_s") is not None]
    med_lead = float(np.median(leads)) if leads else None
    # manipulation: no official term in locomani
    if recall is not None and recall >= 0.70 and med_lead is not None and med_lead >= 0.2:
        return "F-A"
    if recall is not None and recall < 0.40:
        return "F-B"
    if recall is not None and recall >= 0.70 and (med_lead is None or med_lead < 0.2):
        return "F-A"  # predicts, short lead
    return "F-B"


def save_plots(root: Path, data, mat, out_sum: dict) -> None:
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    colors = {"loco": "#3b6ea5", "stoop": "#c45c26", "reach": "#2a9d8f", "carry": "#7b5ea7",
              "parent": "#4a4a4a", "rm3": "#c45c26"}

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(TERRAINS))
    w = 0.18
    for i, task in enumerate(TASKS):
        ys = [mat[task][ter] if mat[task][ter] is not None else np.nan for ter in TERRAINS]
        ax.bar(x + (i - 1.5) * w, ys, w, label=task, color=colors[task])
    ax.set_xticks(x)
    ax.set_xticklabels([TERRAIN_LABEL[t] for t in TERRAINS])
    ax.set_ylabel("Task Completion SR")
    ax.set_title("Parent Task Completion SR by task × terrain")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "parent_task_sr_terrain.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ys = [mat[t]["overall"] if mat[t]["overall"] is not None else np.nan for t in TASKS]
    ax.bar(TASKS, ys, color=[colors[t] for t in TASKS])
    ax.set_ylabel("Overall Task SR")
    ax.set_title("Parent overall Task Completion SR")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(plots / "parent_task_sr_overall.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    pts, sr_t, sr5, fail = [], [], [], []
    for task in ("loco", "reach", "carry"):
        eps = eps_of(data["parent"][task])
        if not eps:
            continue
        pts.append(CONSTRAINT[task][0])
        sr_t.append(float(np.mean([r.get("sr_task", r.get("task_completion", 0)) for r in eps])))
        sr5.append(float(np.mean([r["sr_5cm"] for r in eps])))
        fail.append(float(np.mean([r["fail"] for r in eps])))
    ax.plot(pts, sr_t, "o-", label="Task SR", color="#c45c26")
    ax.plot(pts, sr5, "s--", label="SR@5", color="#3b6ea5")
    ax.plot(pts, fail, "^:", label="Fail rate", color="#666")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["1-pt Loco", "2-pt Reach", "3-pt Carry"])
    ax.set_ylabel("Rate")
    ax.set_title("Task SR vs human constraint level (Stoop excluded)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "constraint_level.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for task in TASKS:
        eps = eps_of(data["parent"][task])
        if not eps:
            continue
        ax.scatter(
            [r["sr_5cm"] for r in eps],
            [r.get("sr_task", r.get("task_completion", 0)) for r in eps],
            s=12, alpha=0.35, label=task, color=colors[task],
        )
    ax.set_xlabel("SR@5 (intent fidelity)")
    ax.set_ylabel("Task success (0/1)")
    ax.set_title("SR@5 vs Task SR (per episode, Parent)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "sr5_vs_task_sr.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    labels = list(TASKS)
    inst = []
    track = []
    manip = []
    tout = []
    other = []
    for task in TASKS:
        tax = taxonomy(eps_of(data["parent"][task]))
        inst.append(tax["instability"] or 0)
        track.append(tax["tracking"] or 0)
        manip.append(tax["manipulation"] or 0)
        tout.append(tax["timeout"] or 0)
        other.append(tax["other"] or 0)
    x = np.arange(len(labels))
    b0 = np.zeros(len(labels))
    for lab, arr, c in (
        ("instability/fall", inst, "#c45c26"),
        ("tracking", track, "#3b6ea5"),
        ("manipulation", manip, "#2a9d8f"),
        ("timeout", tout, "#888"),
        ("other", other, "#bbb"),
    ):
        ax.bar(x, arr, bottom=b0, label=lab, color=c)
        b0 = b0 + np.asarray(arr)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of episodes")
    ax.set_title("Failure taxonomy (failed mass only stacked; rest is success)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "failure_taxonomy.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    recs = []
    for task in TASKS:
        eps = eps_of(data["parent"][task])
        failed = [r for r in eps if r.get("fail")]
        recs.append(_rate([bool(r.get("trigger_before_fail")) for r in failed]) or 0)
    ax.bar(TASKS, recs, color=[colors[t] for t in TASKS])
    ax.set_ylabel("Trigger-before-failure recall")
    ax.set_title("Failure monitor recall on failed episodes")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(plots / "trigger_recall.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    series, labs = [], []
    for task in TASKS:
        leads = [r["lead_time_s"] for r in eps_of(data["parent"][task]) if r.get("fail") and r.get("lead_time_s") is not None]
        if leads:
            series.append(leads)
            labs.append(task)
    if series:
        ax.boxplot(series, labels=labs, showfliers=False)
    ax.set_ylabel("Lead time t_fail − t_first_trigger (s)")
    ax.set_title("Trigger lead time on failed episodes")
    ax.axhline(0, color="#888", lw=0.6)
    fig.tight_layout()
    fig.savefig(plots / "trigger_lead.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    x = np.arange(2)
    w = 0.35
    for i, (name, key) in enumerate((("Parent", "parent"), ("+R-M3", "rm3"))):
        ys = []
        for task in ("loco", "stoop"):
            blob = data[key].get(task)
            vals = [sr_task(blob, ter) for ter in TERRAINS]
            vals = [v for v in vals if v is not None]
            ys.append(float(np.mean(vals)) if vals else np.nan)
        ax.bar(x + (i - 0.5) * w, ys, w, label=name, color=colors["parent"] if i == 0 else colors["rm3"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Loco", "Stoop"])
    ax.set_ylabel("Task SR")
    ax.set_title("Parent vs +R-M3 Task SR (validated tasks)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plots / "rm3_vs_parent_task_sr.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, task in zip(axes, ("reach", "carry")):
        eps = eps_of(data["parent"][task])
        ok = [r.get("n_events", 0) for r in eps if not r.get("fail")]
        bad = [r.get("n_events", 0) for r in eps if r.get("fail")]
        ax.boxplot([ok, bad], labels=["success", "fail"], showfliers=False)
        ax.set_title(f"{task} trigger count")
        ax.set_ylabel("events / episode")
    fig.suptitle("Reach/Carry shadow: trigger rate success vs failure", fontsize=10)
    fig.tight_layout()
    fig.savefig(plots / "shadow_trigger_success_vs_fail.png", dpi=140)
    plt.close(fig)

    # PCA of z_nom
    zs, labs, thetas = [], [], []
    for task in TASKS:
        for ter in TERRAINS:
            for folder in ("parent", "shadow", "rm3_active"):
                p = root / folder / task / ter / "z_nom.npz"
                if not p.exists():
                    continue
                blob = np.load(p, allow_pickle=True)
                z = blob["z"]
                zs.append(z)
                labs.extend([task] * len(z))
                for m in blob["meta"]:
                    try:
                        d = json.loads(str(m))
                        thetas.append(d.get("theta_pred"))
                    except Exception:
                        thetas.append(None)
                break
    if zs:
        Z = np.concatenate(zs, axis=0)
        y = np.array(labs)
        try:
            from sklearn.decomposition import PCA
            xy = PCA(n_components=2, random_state=0).fit_transform(Z)
            fig, ax = plt.subplots(figsize=(6.2, 4.6))
            for task in TASKS:
                m = y == task
                if not m.any():
                    continue
                ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.45, label=task, color=colors[task])
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("z_nom PCA by task")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(plots / "znom_pca.png", dpi=140)
            plt.close(fig)
        except Exception as exc:
            print(f"[t0-plot] PCA skipped: {exc}")
        try:
            import umap
            xy = umap.UMAP(n_components=2, random_state=0).fit_transform(Z)
            fig, ax = plt.subplots(figsize=(6.2, 4.6))
            for task in TASKS:
                m = y == task
                if m.any():
                    ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.45, label=task, color=colors[task])
            ax.set_title("z_nom UMAP by task")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(plots / "znom_umap.png", dpi=140)
            plt.close(fig)
        except Exception as exc:
            print(f"[t0-plot] UMAP skipped: {exc}")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(THETA_BINS))
    w = 0.2
    series = []
    # loco / stoop-S from rm3 shadow.json; reach/carry from shadow/
    def theta_hist(rows, pred=None):
        vals = []
        for r in rows:
            if pred and not pred(r):
                continue
            if r.get("theta_pred") is not None:
                vals.append(float(r["theta_pred"]))
        n = max(len(vals), 1)
        return [sum(abs(v - th) < 1e-6 for v in vals) / n for th in THETA_BINS], len(vals)
    packs = [
        ("Loco", shadow_rows(root, "loco") or [e for e in events_of(data["rm3"].get("loco")) for b in (e.get("bursts") or []) for e in [e]], None),
    ]
    loco_sh = shadow_rows(root, "loco")
    stoop_sh = shadow_rows(root, "stoop")
    reach_sh = shadow_rows(root, "reach")
    carry_sh = shadow_rows(root, "carry")
    named = [
        ("Loco", loco_sh, None),
        ("Stoop-S", stoop_sh, lambda r: r.get("trigger_channel") == "S"),
        ("Reach-shadow", reach_sh, None),
        ("Carry-shadow", carry_sh, None),
    ]
    cols = ["#3b6ea5", "#c45c26", "#2a9d8f", "#7b5ea7"]
    for i, (lab, rows, pred) in enumerate(named):
        hist, n = theta_hist(rows, pred)
        # fallback to event bursts
        if n == 0:
            evs = events_of(data["rm3"].get("loco" if lab == "Loco" else "stoop") if lab.startswith("Stoop") or lab == "Loco" else None)
            vals = []
            for e in events_of(data["rm3"].get("stoop") if lab.startswith("Stoop") else data["rm3"].get("loco") if lab == "Loco" else {}):
                if lab.startswith("Stoop") and e.get("trigger_channel") != "S":
                    continue
                for b in e.get("bursts") or []:
                    if b.get("theta_deg") is not None:
                        vals.append(float(b["theta_deg"]))
            n = max(len(vals), 1)
            hist = [sum(abs(v - th) < 1e-6 for v in vals) / n for th in THETA_BINS]
            n = len(vals)
        ax.bar(x + (i - 1.5) * w, hist, w, label=f"{lab} n={n}", color=cols[i])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{th:g}°" for th in THETA_BINS])
    ax.set_ylabel("Fraction")
    ax.set_title("R-M3 predicted θ histogram")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "theta_hist_by_task.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/home/chenxiangyu/robotics/Anybody/results/t0_full_task_matrix")
    args = ap.parse_args()
    root = Path(args.root)
    data = collect(root)
    mat = parent_matrix(data)
    constraint = {}
    for task in ("loco", "reach", "carry"):
        eps = eps_of(data["parent"][task])
        constraint[task] = {
            "n_points": CONSTRAINT[task][0],
            "label": CONSTRAINT[task][1],
            "sr_task": float(np.mean([r.get("sr_task", r.get("task_completion", 0)) for r in eps])) if eps else None,
            "sr_5cm": float(np.mean([r["sr_5cm"] for r in eps])) if eps else None,
            "fail": float(np.mean([r["fail"] for r in eps])) if eps else None,
        }
    rm3_eff = {}
    for task in ("loco", "stoop"):
        p_eps = eps_of(data["parent"][task])
        r_eps = eps_of(data["rm3"].get(task))
        p_sr = float(np.mean([e.get("sr_task", e.get("task_completion", 0)) for e in p_eps])) if p_eps else None
        r_sr = float(np.mean([e.get("sr_task", e.get("task_completion", 0)) for e in r_eps])) if r_eps else None
        p_f = float(np.mean([e["fail"] for e in p_eps])) if p_eps else None
        r_f = float(np.mean([e["fail"] for e in r_eps])) if r_eps else None
        rm3_eff[task] = {
            "parent_sr_task": p_sr,
            "rm3_sr_task": r_sr,
            "delta_sr_task": None if p_sr is None or r_sr is None else r_sr - p_sr,
            "parent_fail": p_f,
            "rm3_fail": r_f,
            "delta_fail": None if p_f is None or r_f is None else r_f - p_f,
            "parent_sr5": float(np.mean([e["sr_5cm"] for e in p_eps])) if p_eps else None,
            "rm3_sr5": float(np.mean([e["sr_5cm"] for e in r_eps])) if r_eps else None,
        }
    tax = {task: taxonomy(eps_of(data["parent"][task])) for task in TASKS}
    shadow = {}
    for task in ("reach", "carry"):
        eps = eps_of(data["parent"][task])
        failed = [r for r in eps if r.get("fail")]
        leads = [r["lead_time_s"] for r in failed if r.get("lead_time_s") is not None]
        evs = events_of(data["parent"][task])
        ch = [e.get("trigger_channel") for e in evs]
        from collections import Counter
        c = Counter(ch)
        shadow[task] = {
            "fail_count": len(failed),
            "n": len(eps),
            "trigger_recall": _rate([bool(r.get("trigger_before_fail")) for r in failed]),
            "lead_time": _pct(leads),
            "dominant_channel": (c.most_common(1)[0][0] if c else None),
            "channel_hist": dict(c),
            "case": case_f(eps, evs),
        }
    summary = {
        "parent_task_sr": mat,
        "constraint": constraint,
        "rm3_effect": rm3_eff,
        "taxonomy": tax,
        "shadow_monitor": shadow,
        "stoop_note": "Stoop excluded from 1/2/3-point constraint curve (dynamic regime).",
        "other_gym_tasks_not_in_p1": ["writing", "obstacle-reach"],
    }
    save_plots(root, data, mat, summary)
    (root / "summary_matrix.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (root / "taxonomy" / "taxonomy.json").write_text(json.dumps(tax, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[t0-plot] wrote {root / 'plots'}")


if __name__ == "__main__":
    main()
