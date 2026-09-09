#!/usr/bin/env python3
"""T1.1 physical-sanity plots and validity verdict. No Isaac."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(p: Path) -> dict | None:
    s = p / "plane" / "summary.json"
    if not s.exists():
        return None
    return json.loads(s.read_text())


def _num(name: str) -> float:
    for tok in reversed(str(name).replace("-", "_").split("_")):
        try:
            return float(tok)
        except ValueError:
            continue
    return float("nan")


def _phys(cell: dict, key: str) -> float:
    ph = (cell.get("episode_monitor") or {}).get("physics") or {}
    blk = ph.get(key) or {}
    if isinstance(blk, dict):
        return float(blk.get("mean", float("nan")))
    return float(blk) if blk is not None else float("nan")


def _curve(root: Path, rel: str) -> list[dict]:
    d = root / rel
    if not d.exists():
        return []
    rows = []
    for p in sorted(d.iterdir()):
        if not p.is_dir():
            continue
        cell = _load(p)
        if not cell:
            continue
        st = cell.get("stress") or {}
        ws = cell.get("workspace") or {}
        rows.append({
            "name": p.name,
            "x": _num(p.name),
            "sr": cell.get("sr_task"),
            "fail": cell.get("fail_frac"),
            "arm_tau": _phys(cell, "arm_torque_mean"),
            "root_acc": _phys(cell, "root_acc_mean"),
            "peak_roll": _phys(cell, "peak_roll"),
            "peak_E": _phys(cell, "peak_E"),
            "sat": _phys(cell, "action_sat"),
            "wrist_cf": _phys(cell, "wrist_contact"),
            "meas_dv": _phys(cell, "measured_push_dv"),
            "exc_before": ws.get("median_torso_wrist_before_m"),
            "exc_after": ws.get("median_torso_wrist_after_m"),
            "scale": st.get("intent_scale"),
            "mode": st.get("intent_mode"),
            "impossible": ws.get("impossible"),
            "n": cell.get("n_episodes"),
            "cell": cell,
        })
    rows.sort(key=lambda r: r["x"] if np.isfinite(r["x"]) else 99)
    return rows


def _corr(xs, ys) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3 or float(np.std(x[m])) < 1e-12 or float(np.std(y[m])) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _valid(corr: float, lo: float = 0.40) -> str:
    if not np.isfinite(corr):
        return "INVALID"
    return "VALID" if corr >= lo else "INVALID"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _line(ax, rows, yk, title, xlabel, ylabel):
    xs = [r["x"] for r in rows]
    ys = [r[yk] for r in rows]
    ax.plot(xs, ys, "o-")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/t1_1_stress_validation")
    args = ap.parse_args()
    root = Path(args.root)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    payload = _curve(root, "payload_sanity")
    com = _curve(root, "com_sanity")
    push = _curve(root, "push_sanity")
    lat = _curve(root, "reach_api/lateral")
    fwd = _curve(root, "reach_api/forward")
    pilot = _curve(root, "intent_stress_pilot")

    if payload:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
        _line(axes[0], payload, "arm_tau", "Payload vs arm torque", "added mass (kg)", "mean |τ_arm|")
        _line(axes[1], payload, "wrist_cf", "Payload vs wrist contact", "added mass (kg)", "mean |F_wrist|")
        _save(fig, plots / "01_payload_arm_load.png")
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        _line(ax, payload, "root_acc", "Payload vs root |a|", "added mass (kg)", "mean |a_root|")
        _save(fig, plots / "02_payload_root.png")
    if com:
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        _line(ax, com, "arm_tau", "COM offset vs arm torque", "lateral COM (m)", "mean |τ_arm|")
        _save(fig, plots / "03_com_arm_moment.png")
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        _line(ax, com, "peak_roll", "COM offset vs peak |roll|", "lateral COM (m)", "peak |roll| (rad)")
        _save(fig, plots / "04_com_roll.png")
    if push:
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        _line(ax, push, "meas_dv", "Push command vs measured Δv", "command Δv (m/s)", "measured |Δv| (m/s)")
        _save(fig, plots / "05_push_dv.png")
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        _line(ax, push, "peak_roll", "Push command vs peak tilt", "command Δv (m/s)", "peak |roll| (rad)")
        _save(fig, plots / "06_push_tilt.png")
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        _line(ax, push, "peak_E", "Push command vs peak E", "command Δv (m/s)", "peak E (m)")
        _save(fig, plots / "07_push_E.png")
    if lat or fwd:
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        if lat:
            ax.plot([r["scale"] for r in lat], [r["exc_after"] for r in lat], "o-", label="lateral")
        if fwd:
            ax.plot([r["scale"] for r in fwd], [r["exc_after"] for r in fwd], "s-", label="forward")
        ax.set_title("Reach scale vs target excursion")
        ax.set_xlabel("intent scale s")
        ax.set_ylabel("median ||wrist-torso|| (m)")
        ax.legend()
        _save(fig, plots / "08_reach_excursion.png")
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        if lat:
            ax.plot([r["scale"] for r in lat], [r["sr"] for r in lat], "o-", label="lateral SR")
        if fwd:
            ax.plot([r["scale"] for r in fwd], [r["sr"] for r in fwd], "s-", label="forward SR")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Reach scale vs Task SR")
        ax.set_xlabel("intent scale s")
        ax.set_ylabel("Task SR")
        ax.legend()
        _save(fig, plots / "09_reach_sr.png")
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        if lat:
            ax.plot([r["scale"] for r in lat], [r["peak_E"] for r in lat], "o-", label="lateral peak E")
        if fwd:
            ax.plot([r["scale"] for r in fwd], [r["peak_E"] for r in fwd], "s-", label="forward peak E")
        ax.set_title("Reach scale vs tracking error")
        ax.set_xlabel("intent scale s")
        ax.set_ylabel("peak E (m)")
        ax.legend()
        _save(fig, plots / "10_reach_E.png")
    if pilot:
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        _line(ax, pilot, "sr", "Intent speed vs Task SR", "speed ×", "Task SR")
        _save(fig, plots / "12_intent_speed.png")

    c_pay = _corr([r["x"] for r in payload], [r["arm_tau"] for r in payload])
    c_com = _corr([r["x"] for r in com], [r["arm_tau"] for r in com])
    c_push = _corr([r["x"] for r in push], [r["meas_dv"] for r in push])
    c_lat = _corr([r["scale"] for r in lat], [r["exc_after"] for r in lat])
    reach_sr_min = min([r["sr"] for r in lat + fwd if r["sr"] is not None], default=None)
    reach_verdict = "STILL_INVALID"
    if lat and np.isfinite(c_lat) and c_lat >= 0.4:
        if reach_sr_min is not None and reach_sr_min <= 0.80:
            reach_verdict = "API_FIXED_AND_VALID"
        else:
            reach_verdict = "API_FIXED_BUT_NO_DIFFICULTY_RANGE"

    table_a = [
        {"Stress": "payload", "Valid?": _valid(c_pay), "Evidence": f"corr(mass, arm_τ)={c_pay:.3f}",
         "Main issue": "no carry object; mass on wrists" if _valid(c_pay) == "VALID" else "physical load did not rise with mass"},
        {"Stress": "com", "Valid?": _valid(c_com), "Evidence": f"corr(offset, arm_τ)={c_com:.3f}",
         "Main issue": "wrist COM + 4kg payload" if _valid(c_com) == "VALID" else "COM offset did not change arm moment"},
        {"Stress": "push", "Valid?": _valid(c_push), "Evidence": f"corr(cmd Δv, measured Δv)={c_push:.3f}",
         "Main issue": "write_root_velocity_to_sim instantaneous Δv" if _valid(c_push) == "VALID" else "commanded Δv not realized"},
    ]
    mixed = []
    for name, rows, stress in (
        ("carry", payload, "payload"),
        ("carry", com, "com"),
        ("loco", push, "push"),
        ("reach", lat, "intent_lateral"),
        ("reach", fwd, "intent_forward"),
    ):
        for r in rows:
            sr = r["sr"]
            mixed.append({
                "Task": name, "Stress": stress, "severity": r["name"], "Task SR": sr,
                "valid mixed regime?": bool(sr is not None and 0.20 <= sr <= 0.80),
            })

    pays = [r["sr"] for r in payload if r["sr"] is not None]
    parent_robust = bool(pays) and min(pays) >= 0.80 and _valid(c_pay) == "VALID"
    if any(t["Valid?"] == "INVALID" for t in table_a):
        final = "physical stress still invalid"
    elif any(m["valid mixed regime?"] for m in mixed):
        final = "T1 benchmark can now be frozen"
    elif parent_robust:
        final = "Parent genuinely too robust → move to intent stress"
    else:
        final = "physical stress VALID but no 20–80% mixed regime yet"

    out = {
        "table_a_validity": table_a,
        "correlations": {"payload_arm_tau": c_pay, "com_arm_tau": c_com, "push_dv": c_push, "reach_lat_exc": c_lat},
        "payload": payload,
        "com": com,
        "push": push,
        "reach_lateral": lat,
        "reach_forward": fwd,
        "reach_verdict": reach_verdict,
        "difficulty": mixed,
        "final_verdict": final,
        "reach_api": {
            "old_bug": "cmd.motion does not exist on PartialMaskedMultiMotionCommand",
            "actual_source": "command.motion_dir_loader.body_pos_w via gather(); Mapper-B uses the same gather",
            "fixed_implementation": "p_rel=p_wrist-p_torso; p_wrist'=p_torso+s*p_rel_component; orientation unchanged; before Mapper-B",
        },
    }
    # drop cell blobs
    for key in ("payload", "com", "push", "reach_lateral", "reach_forward"):
        for r in out[key]:
            r.pop("cell", None)
    (root / "summary_validity.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[t11-plot] wrote {plots}")
    print(f"[t11-plot] validity {table_a}")
    print(f"[t11-plot] reach {reach_verdict}")
    print(f"[t11-plot] FINAL {final}")


if __name__ == "__main__":
    main()
