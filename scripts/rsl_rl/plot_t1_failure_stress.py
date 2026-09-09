#!/usr/bin/env python3
"""T1 phase diagrams and tables. No Isaac. Reads calibration + frozen_matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TASKS = ("loco", "stoop", "reach", "carry")
T0_NOMINAL = {"loco": 0.87, "stoop": 0.87, "reach": 0.995, "carry": 1.0}


def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _cells(root: Path) -> list[dict]:
    rows = []
    for p in root.rglob("summary.json"):
        if "calibration" not in str(p) and "frozen_matrix" not in str(p):
            continue
        cell = _load(p)
        if not cell:
            continue
        rel = p.relative_to(root)
        parts = rel.parts
        stage = parts[0]
        cell["_stage"] = stage
        cell["_path"] = str(p)
        rows.append(cell)
    return rows


def _mon(cell: dict) -> dict:
    return cell.get("episode_monitor") or {}


def _pct(xs, q):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=np.float64)
    if a.size == 0:
        return None
    return float(np.percentile(a, q))


def _num_from_name(name: str) -> float:
    for tok in reversed(str(name).replace("-", "_").split("_")):
        try:
            return float(tok)
        except ValueError:
            continue
    return float("nan")


def _family_curve(cells: list[dict], family: str, task: str | None = None) -> list[dict]:
    out = []
    for c in cells:
        st = c.get("stress") or {}
        if st.get("family") != family:
            continue
        if task and c.get("p1_task") != task:
            continue
        if c.get("terrain") not in (None, "plane") and c.get("_stage") == "calibration":
            if c.get("terrain") != "plane":
                continue
        if c.get("_stage") == "calibration" and c.get("terrain") != "plane":
            continue
        sev = st.get("severity_name") or ""
        x = {
            "payload": st.get("payload_kg"),
            "com": (st.get("com_xyz_m") or [0, 0, 0])[1],
            "workspace": np.linalg.norm([st.get("ws_fwd_m") or 0, st.get("ws_lat_m") or 0, st.get("ws_z_m") or 0]),
            "push": st.get("push_dv_mps"),
        }.get(family, _num_from_name(sev))
        mon = _mon(c)
        out.append({
            "severity": sev,
            "x": float(x) if x is not None else _num_from_name(sev),
            "sr_task": c.get("sr_task"),
            "fail": c.get("fail_frac"),
            "fall": c.get("fall_frac"),
            "sr5": c.get("sr_5cm"),
            "recall": mon.get("trigger_before_failure_recall"),
            "fp": mon.get("false_positive_trigger_rate"),
            "lead": (mon.get("lead_time_s") or {}).get("p50"),
            "n_fail": mon.get("n_failed"),
            "task": c.get("p1_task"),
            "terrain": c.get("terrain"),
            "stage": c.get("_stage"),
            "cell": c,
        })
    out.sort(key=lambda r: (r["x"] if np.isfinite(r["x"]) else 99, r["severity"]))
    return out


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _phase(ax, rows, title, xlabel):
    if not rows:
        return
    xs = [r["x"] for r in rows]
    ax.plot(xs, [r["sr_task"] for r in rows], "o-", label="Task SR")
    ax.plot(xs, [r["fail"] for r in rows], "s--", label="Fail")
    ax.axhspan(0.20, 0.80, color="0.85", alpha=0.5, label="mixed 20–80%")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("rate")
    ax.legend(fontsize=8)


def _case_label(task: str, n_fail: int, recall, fp, lead) -> str:
    if n_fail is None or n_fail < 8:
        return "insufficient_failures"
    rec = float(recall) if recall is not None and np.isfinite(recall) else float("nan")
    fpr = float(fp) if fp is not None and np.isfinite(fp) else float("nan")
    if rec >= 0.7 and (not np.isfinite(fpr) or fpr < 0.85) and lead is not None:
        return "F-A"
    if rec < 0.4:
        return "F-B"
    if fpr >= 0.8 and rec >= 0.7:
        return "F-C"
    if rec >= 0.55:
        return "F-A"
    return "F-B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/t1_failure_stress")
    args = ap.parse_args()
    root = Path(args.root)
    cells = _cells(root)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    cal = [c for c in cells if c.get("_stage") == "calibration"]
    payload = _family_curve(cal, "payload", "carry")
    com = _family_curve(cal, "com", "carry")
    ws = _family_curve(cal, "workspace", "reach")
    push = {t: _family_curve(cal, "push", t) for t in ("loco", "reach", "carry")}

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    _phase(ax, payload, "Carry payload → Task SR (cal, Flat)", "extra wrist payload (kg)")
    _save(fig, plots / "01_carry_payload_phase.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    _phase(ax, com, "Carry COM lateral offset → Task SR (cal, Flat, 4 kg)", "wrist COM y (m)")
    _save(fig, plots / "02_carry_com_phase.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    _phase(ax, ws, "Reach workspace offset norm → Task SR (cal, Flat)", "||Δtarget|| (m)")
    _save(fig, plots / "03_reach_workspace_phase.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for t, rows in push.items():
        if rows:
            ax.plot([r["x"] for r in rows], [r["sr_task"] for r in rows], "o-", label=t)
    ax.axhspan(0.20, 0.80, color="0.85", alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Matched lateral push → Task SR (cal, Flat)")
    ax.set_xlabel("Δv lateral (m/s)")
    ax.set_ylabel("Task SR")
    ax.legend()
    _save(fig, plots / "04_push_phase.png")

    # Constraint ΔSR at matched push high/mid
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels, dsr, fall = [], [], []
    for t, nlab in (("loco", "1-pt Loco"), ("reach", "2-pt Reach"), ("carry", "3-pt Carry")):
        rows = [r for r in push[t] if r["x"] > 0]
        if not rows:
            continue
        # use hardest calibrated point that still has finite SR
        hard = min(rows, key=lambda r: r["sr_task"] if r["sr_task"] is not None else 1)
        nom = T0_NOMINAL[t]
        labels.append(nlab)
        dsr.append(nom - float(hard["sr_task"]))
        fall.append(hard["fall"])
    if labels:
        x = np.arange(len(labels))
        ax.bar(x - 0.15, dsr, 0.3, label="Δ Task SR vs T0 nominal")
        ax.bar(x + 0.15, fall, 0.3, label="Fall rate at selected stress")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("Constraint-level robustness drop (cal push, hardest)")
        ax.set_ylabel("rate")
        ax.legend(fontsize=8)
    _save(fig, plots / "05_constraint_dsr.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    tax_keys = ["instability", "tracking", "manipulation", "timeout", "other"]
    names, stacks = [], {k: [] for k in tax_keys}
    for fam, rows in (("payload", payload), ("com", com), ("workspace", ws)):
        if not rows:
            continue
        hard = min(rows, key=lambda r: r["sr_task"] if r["sr_task"] is not None else 1)
        tax = (_mon(hard["cell"]).get("taxonomy") or {})
        names.append(f"{fam}/{hard['severity']}")
        for k in tax_keys:
            stacks[k].append(float(tax.get(k, 0.0)))
    if names:
        x = np.arange(len(names))
        bottom = np.zeros(len(names))
        for k in tax_keys:
            ax.bar(x, stacks[k], bottom=bottom, label=k)
            bottom = bottom + np.asarray(stacks[k])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_title("Failure taxonomy at hardest calibrated cell")
        ax.set_ylabel("fraction of episodes")
        ax.legend(fontsize=8)
    _save(fig, plots / "06_taxonomy_stacked.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, rows in (("carry-payload", payload), ("carry-com", com), ("reach-ws", ws)):
        xs = [r["x"] for r in rows if r["recall"] is not None]
        ys = [r["recall"] for r in rows if r["recall"] is not None]
        if xs:
            ax.plot(xs, ys, "o-", label=name)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Trigger-before-failure recall vs severity")
    ax.set_xlabel("severity (family units)")
    ax.set_ylabel("recall")
    ax.legend(fontsize=8)
    _save(fig, plots / "07_trigger_recall.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, rows in (("carry-payload", payload), ("carry-com", com), ("reach-ws", ws)):
        xs = [r["x"] for r in rows if r["fp"] is not None]
        ys = [r["fp"] for r in rows if r["fp"] is not None]
        if xs:
            ax.plot(xs, ys, "o-", label=name)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("False-positive trigger rate on successes")
    ax.set_xlabel("severity (family units)")
    ax.set_ylabel("FP rate")
    ax.legend(fontsize=8)
    _save(fig, plots / "08_trigger_fp.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, rows in (("carry-payload", payload), ("carry-com", com), ("reach-ws", ws)):
        xs = [r["x"] for r in rows if r["lead"] is not None]
        ys = [r["lead"] for r in rows if r["lead"] is not None]
        if xs:
            ax.plot(xs, ys, "o-", label=name)
    ax.set_title("Median trigger lead time vs severity")
    ax.set_xlabel("severity (family units)")
    ax.set_ylabel("lead (s)")
    ax.legend(fontsize=8)
    _save(fig, plots / "09_trigger_lead.png")

    def _rs_hist(task: str, family: str, fname: str, title: str):
        rows = _family_curve(cal, family, task)
        failed_re, ok_re, failed_rs, ok_rs = [], [], [], []
        for r in rows:
            for ep in r["cell"].get("episodes") or []:
                if ep.get("fail"):
                    failed_re.append(ep.get("max_RE"))
                    failed_rs.append(ep.get("max_RS"))
                else:
                    ok_re.append(ep.get("max_RE"))
                    ok_rs.append(ep.get("max_RS"))
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
        for ax, ok, bad, lab in (
            (axes[0], ok_re, failed_re, "max RE"),
            (axes[1], ok_rs, failed_rs, "max RS"),
        ):
            if any(v is not None for v in ok):
                ax.hist([v for v in ok if v is not None], bins=16, alpha=0.55, label="success")
            if any(v is not None for v in bad):
                ax.hist([v for v in bad if v is not None], bins=16, alpha=0.55, label="fail")
            ax.set_title(lab)
            ax.set_xlabel(lab)
            ax.legend(fontsize=8)
        fig.suptitle(title)
        _save(fig, plots / fname)

    _rs_hist("reach", "workspace", "10_reach_re_rs.png", "Reach success vs fail max RE/RS (workspace cal)")
    _rs_hist("carry", "payload", "11_carry_re_rs.png", "Carry success vs fail max RE/RS (payload cal)")

    # PCA of z_nom if present
    zs, labs = [], []
    for c in cal:
        zp = Path(c["_path"]).parent / "z_nom.npz"
        if not zp.exists():
            continue
        blob = np.load(zp, allow_pickle=True)
        z = np.asarray(blob["z"], dtype=np.float32)
        if z.size == 0:
            continue
        take = min(80, z.shape[0])
        idx = np.linspace(0, z.shape[0] - 1, take).astype(int)
        zs.append(z[idx])
        labs.extend([str(c.get("p1_task"))] * take)
    if zs:
        zall = np.concatenate(zs, axis=0)
        zall = zall - zall.mean(0, keepdims=True)
        u, s, vt = np.linalg.svd(zall, full_matrices=False)
        xy = zall @ vt[:2].T
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        for t in TASKS:
            m = np.array(labs) == t
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.55, label=t)
        ax.set_title("z_nom PCA (T1 calibration triggers/subsamples)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        _save(fig, plots / "12_znom_pca.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for t in TASKS:
        th = []
        for c in cal:
            if c.get("p1_task") != t:
                continue
            sh = Path(c["_path"]).parent / "shadow.json"
            if not sh.exists():
                continue
            for row in json.loads(sh.read_text()):
                if row.get("theta_pred") is not None:
                    th.append(float(row["theta_pred"]))
        if th:
            ax.hist(th, bins=[1.25, 3.75, 6.25, 8.75, 11.25], alpha=0.45, label=f"{t} n={len(th)}")
    ax.set_title("R-M3 shadow θ histogram (calibration)")
    ax.set_xlabel("theta_pred (deg)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    _save(fig, plots / "13_theta_hist.png")

    # Tables
    table_a = []
    for fam, rows, task in (
        ("payload", payload, "carry"),
        ("com", com, "carry"),
        ("workspace", ws, "reach"),
    ):
        for r in rows:
            table_a.append({
                "Task": task, "Stress": fam, "Severity": r["severity"],
                "Task SR": r["sr_task"], "Fail": r["fail"], "Fall": r["fall"],
            })
    for t, rows in push.items():
        for r in rows:
            table_a.append({
                "Task": t, "Stress": "push", "Severity": r["severity"],
                "Task SR": r["sr_task"], "Fail": r["fail"], "Fall": r["fall"],
            })

    sel_p = root / "manifests" / "selected_severities.json"
    selected = _load(sel_p) or {}
    table_b = []
    for fam in ("carry_payload", "carry_com", "reach_workspace"):
        blk = selected.get(fam) or {}
        task = "carry" if "carry" in fam else "reach"
        for level, v in (blk.get("selected") or {}).items():
            table_b.append({
                "Task": task, "Stress": fam, "Selected": f"{level}:{v.get('severity')}",
                "Task SR": v.get("sr_task"), "dominant failure": v.get("dominant_failure"),
            })

    table_c = []
    for fam, rows, task in (("payload", payload, "carry"), ("com", com, "carry"), ("workspace", ws, "reach")):
        for r in rows:
            table_c.append({
                "Task": task, "Stress": fam, "failures": r["n_fail"],
                "trigger recall": r["recall"], "false positive": r["fp"],
                "median lead": r["lead"],
                "dominant channel": _dom_ch(r["cell"]),
            })
    for t, rows in push.items():
        if t == "loco":
            continue
        for r in rows:
            table_c.append({
                "Task": t, "Stress": "push", "failures": r["n_fail"],
                "trigger recall": r["recall"], "false positive": r["fp"],
                "median lead": r["lead"],
                "dominant channel": _dom_ch(r["cell"]),
            })

    table_d = []
    for t, nlab, npts in (("loco", "1 point", 1), ("reach", "2 point", 2), ("carry", "3 point", 3)):
        rows = [r for r in push[t] if r["x"] > 0]
        if not rows:
            continue
        hard = min(rows, key=lambda r: r["sr_task"] if r["sr_task"] is not None else 1)
        nom = next((r["sr_task"] for r in push[t] if abs(r["x"]) < 1e-9), T0_NOMINAL[t])
        table_d.append({
            "Intent level": nlab, "Task": t, "Nominal SR": nom,
            "Push SR": hard["sr_task"], "ΔSR": (nom - hard["sr_task"]) if nom is not None else None,
            "Fall": hard["fall"],
            "Tracking Δ": None if hard["sr5"] is None else (None),
            "push_severity": hard["severity"],
        })

    reach_fail_n = sum(int(r["n_fail"] or 0) for r in ws)
    carry_fail_n = sum(int(r["n_fail"] or 0) for r in payload + com)
    reach_case = _case_label(
        "reach", reach_fail_n,
        np.nanmean([r["recall"] for r in ws if r["recall"] is not None]) if any(r["recall"] is not None for r in ws) else None,
        np.nanmean([r["fp"] for r in ws if r["fp"] is not None]) if any(r["fp"] is not None for r in ws) else None,
        True,
    )
    carry_case = _case_label(
        "carry", carry_fail_n,
        np.nanmean([r["recall"] for r in payload if r["recall"] is not None]) if any(r["recall"] is not None for r in payload) else None,
        np.nanmean([r["fp"] for r in payload if r["fp"] is not None]) if any(r["fp"] is not None for r in payload) else None,
        True,
    )

    summary = {
        "n_cells": len(cells),
        "n_cal": len(cal),
        "table_a_calibration": table_a,
        "table_b_selected": table_b,
        "table_c_monitor": table_c,
        "table_d_constraint_push": table_d,
        "selected": selected,
        "perception": "not_applicable_to_current_parent",
        "verdict": {
            "reach": reach_case,
            "carry": carry_case,
            "reach_failures_cal": reach_fail_n,
            "carry_failures_cal": carry_fail_n,
        },
        "t0_nominal": T0_NOMINAL,
    }
    (root / "summary_matrix.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[t1-plot] cells={len(cells)} cal={len(cal)} wrote {plots}")
    print(f"[t1-plot] verdict Reach={reach_case} Carry={carry_case}")


def _dom_ch(cell: dict) -> str:
    ev = cell.get("events") or []
    hist = {"E": 0, "S": 0, "both": 0}
    for e in ev:
        ch = e.get("trigger_channel")
        if ch in hist:
            hist[ch] += 1
    if not any(hist.values()):
        return "neither"
    return max(hist, key=hist.get)


if __name__ == "__main__":
    main()
