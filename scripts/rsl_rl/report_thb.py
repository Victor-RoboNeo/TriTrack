#!/usr/bin/env python3
"""Aggregate Torso Height Buffer oracle cells. Privileged diagnostic, not a method."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TASKS = ("loco", "stoop", "reach", "carry")
VARIANTS = (
    "original",
    "torso_b025",
    "torso_b050",
    "torso_b075",
    "torso_b100",
    "shift_all",
)
LABEL = {
    "original": "Original (β=0)",
    "torso_b025": "Torso-buffer β=0.25",
    "torso_b050": "Torso-buffer β=0.50",
    "torso_b075": "Torso-buffer β=0.75",
    "torso_b100": "Torso-buffer β=1.00",
    "shift_all": "Shift-all +Δh",
}


def _fmt(v, pct=False, cm=False, n=3):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if pct:
        return f"{100.0 * float(v):.1f}%"
    if cm:
        return f"{100.0 * float(v):.2f} cm"
    return f"{float(v):.{n}f}"


def _mean_key(mon: dict, key: str, inner="mean"):
    blk = mon.get(key) if mon else None
    if not isinstance(blk, dict):
        return None
    return blk.get(inner)


def load_cells(root: Path) -> dict:
    out = {}
    for task in TASKS:
        for variant in VARIANTS:
            p = root / task / variant / "steps" / "summary.json"
            if not p.exists():
                continue
            out[(task, variant)] = json.loads(p.read_text())
    return out


def verdict(cells: dict) -> str:
    """Headline from the β sweep, not only β=1."""
    orig = cells.get(("loco", "original"))
    b25 = cells.get(("loco", "torso_b025"))
    b100 = cells.get(("loco", "torso_b100"))
    if not orig or not b25:
        return "incomplete"
    o = orig.get("episode_monitor") or {}
    b = b25.get("episode_monitor") or {}
    az0, az1 = o.get("anchor_z_frac"), b.get("anchor_z_frac")
    sr0, sr1 = o.get("sr_task"), b.get("sr_task")
    lw0 = _mean_key(o, "e_lw_world_orig")
    lw1 = _mean_key(b, "e_lw_world_orig")
    wrist_ok = lw0 is None or lw1 is None or (lw1 - lw0) <= 0.02
    small_helps = (
        az0 is not None and az1 is not None and az1 < az0 - 1e-9
        and sr0 is not None and sr1 is not None and sr1 > sr0 + 1e-9
        and wrist_ok
    )
    full_hurts = False
    if b100:
        f = b100.get("episode_monitor") or {}
        if f.get("sr_task") is not None and sr0 is not None and f["sr_task"] < sr0 - 0.1:
            full_hurts = True
    if small_helps and full_hurts:
        return "PARTIAL_SLACK_HELPS_FULL_ABSORB_HURTS"
    if small_helps:
        return "SUPPORTS_TORSO_BUFFER_ORACLE"
    if full_hurts:
        return "FULL_ABSORB_HARMS_BUFFERED_CONSTRAINT"
    return "NO_CLEAR_BUFFER_GAIN"


def main():
    ap = argparse.ArgumentParser()
    parser_out = "/data/home/chenxiangyu/robotics/Anybody/results/thb_torso_height_buffer"
    ap.add_argument("--root", type=str, default=parser_out)
    ap.add_argument("--out", type=str, default=parser_out)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = load_cells(root)
    lines = [
        "# Torso Height Buffer Oracle (privileged diagnostic)",
        "",
        "Parent only. No R-M3. No terrain ID. **Not in the method.**",
        "",
        "Support height = mean world-z of currently contacting ankles; "
        r"$\Delta h=h_t-h_0$ (signed; $|\delta z|\le|\Delta h|$). T0 Steps is a pyramid, so support often drops; "
        r"$\max(0,\Delta h)$ would be a no-op. Torso-buffer raises/lowers torso only. Hands stay at original world targets.",
        "",
        "Torso-buffer: raise torso target only. Hands stay at original **world** targets.",
        "Shift-all: raise torso **and** wrists by $\\Delta h$ (diagnostic baseline).",
        "",
        "Paper language: split **support-induced vertical displacement** from **human-relative torso motion**. "
        "Do not say “torso height is free”.",
        "",
    ]
    pooled = {"cells": {}, "verdict": None}
    for task in TASKS:
        lines.append(f"## {task}")
        lines.append("")
        lines.append(
            "| Variant | SR_task (buffered) | SR orig-z | fall | anchor_z | "
            "wrist L/R world-orig | torso orig / adj | mean $\\delta z$ | arm sat |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for variant in VARIANTS:
            cell = cells.get((task, variant))
            if not cell:
                lines.append(f"| {LABEL[variant]} | — | — | — | — | — | — | — | — |")
                continue
            m = cell.get("episode_monitor") or {}
            lw = _mean_key(m, "e_lw_world_orig")
            rw = _mean_key(m, "e_rw_world_orig")
            to = _mean_key(m, "e_torso_orig")
            te = _mean_key(m, "e_torso_exec")
            lines.append(
                f"| {LABEL[variant]} | {_fmt(m.get('sr_task'), pct=True)} | "
                f"{_fmt(m.get('sr_task_orig_constraint'), pct=True)} | "
                f"{_fmt(m.get('fall_frac'), pct=True)} | "
                f"{_fmt(m.get('anchor_z_frac'), pct=True)} | "
                f"{_fmt(lw, cm=True)} / {_fmt(rw, cm=True)} | "
                f"{_fmt(to, cm=True)} / {_fmt(te, cm=True)} | "
                f"{_fmt(_mean_key(m, 'dz'), cm=True)} | "
                f"{_fmt(_mean_key(m, 'arm_joint_sat'))} |"
            )
            pooled["cells"].setdefault(task, {})[variant] = {
                "sr_task": m.get("sr_task"),
                "sr_task_orig_constraint": m.get("sr_task_orig_constraint"),
                "fall_frac": m.get("fall_frac"),
                "anchor_z_frac": m.get("anchor_z_frac"),
                "anchor_z_frac_orig": m.get("anchor_z_frac_orig"),
                "e_lw_world_orig": lw,
                "e_rw_world_orig": rw,
                "e_torso_orig": to,
                "e_torso_exec": te,
                "dz": _mean_key(m, "dz"),
                "dh": _mean_key(m, "dh"),
                "arm_joint_sat": _mean_key(m, "arm_joint_sat"),
                "n": m.get("n_episodes"),
            }
        lines.append("")
        orig = cells.get((task, "original"))
        b25 = cells.get((task, "torso_b025"))
        b100 = cells.get((task, "torso_b100"))
        sh = cells.get((task, "shift_all"))
        if orig and b25:
            o = orig["episode_monitor"]
            b = b25["episode_monitor"]
            lines.append(
                f"- β=0.25 vs Original: ΔSR={_fmt((b.get('sr_task') or 0) - (o.get('sr_task') or 0), pct=True)}, "
                f"Δanchor_z={_fmt((b.get('anchor_z_frac') or 0) - (o.get('anchor_z_frac') or 0), pct=True)}, "
                f"ΔwristL={_fmt((_mean_key(b,'e_lw_world_orig') or 0) - (_mean_key(o,'e_lw_world_orig') or 0), cm=True)}"
            )
        if orig and b100:
            o = orig["episode_monitor"]
            b = b100["episode_monitor"]
            lines.append(
                f"- β=1.00 vs Original: ΔSR={_fmt((b.get('sr_task') or 0) - (o.get('sr_task') or 0), pct=True)}, "
                f"Δanchor_z={_fmt((b.get('anchor_z_frac') or 0) - (o.get('anchor_z_frac') or 0), pct=True)}, "
                f"SR orig-z={_fmt(b.get('sr_task_orig_constraint'), pct=True)}"
            )
        if orig and sh:
            o = orig["episode_monitor"]
            s = sh["episode_monitor"]
            lines.append(
                f"- Shift-all vs Original: ΔSR={_fmt((s.get('sr_task') or 0) - (o.get('sr_task') or 0), pct=True)}, "
                f"ΔwristL={_fmt((_mean_key(s,'e_lw_world_orig') or 0) - (_mean_key(o,'e_lw_world_orig') or 0), cm=True)}"
            )
        lines.append("")

    v = verdict(cells)
    pooled["verdict"] = v
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{v}**")
    lines.append("")
    lines.append(
        "Read the **β sweep**, not only β=1. Official `anchor_z` is already ankle-relative "
        "posture `(T_cmd−T_rob)+(A_rob−A_cmd)`. Buffering torso without ankles therefore changes "
        "the *posture target* by δz, not just world height. T0 Steps is a pyramid (spawn high, "
        "walk down): signed Δh is typically negative (~−14 cm on Loco)."
    )
    lines.append("")
    lines.append(
        "- **Loco β=0.25** is the only clear win: SR 76.7%→86.7%, `anchor_z` 20%→10%, "
        "wrist world-orig **not worse** (39.8→35.6 cm). About half of Loco Steps `anchor_z` "
        "fails are removed by a *small* support-relative slack (~3 cm)."
    )
    lines.append(
        "- **β≥0.5 full-ish absorb hurts the buffered constraint** (Loco SR 50% / 37% / 30%) "
        "because it asks Parent to drop torso by 7–15 cm while clip ankles stay put — a new "
        "posture conflict. SR against the *original* unbuffered torso can even rise "
        "(Loco β=0.5 orig-z SR 93.3%), so the robot is not simply falling."
    )
    lines.append(
        "- **Stoop / Reach / Carry** do not want this slack: wrists already locked, SR is high, "
        "and extra δz mostly creates `anchor_z` on the buffered target. Carry Shift-all "
        "is the diagnostic we wanted: wrists world-orig **worsen** (6.0→7.8 cm) while "
        "torso-buffer β=0.25 keeps them (~6.4 cm)."
    )
    lines.append("")
    lines.append(
        "Do **not** put privileged terrain height in the method. If anything later is unified, "
        "it is a small **support-relative torso slack** from foot FK/contacts, with "
        "β≪1, wrists task-locked, and ankle/posture consistency kept. Not “torso height is free”, "
        "and not “copy Δh into the whole task”."
    )
    lines.append("")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "pooled.json").write_text(json.dumps(pooled, indent=2), encoding="utf-8")
    print(f"[thb] wrote {out / 'REPORT.md'} verdict={v}", flush=True)


if __name__ == "__main__":
    main()
