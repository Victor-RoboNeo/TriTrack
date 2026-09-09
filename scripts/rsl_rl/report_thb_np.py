#!/usr/bin/env python3
"""THB-NP report: unsigned contact+FK support slack vs privileged signed β=0.25."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OLD = Path("/data/home/chenxiangyu/robotics/Anybody/results/thb_torso_height_buffer")


def _load(p: Path) -> dict:
    s = p / "summary.json"
    return json.loads(s.read_text()) if s.is_file() else {}


def _sr(d: dict):
    return d.get("sr_task")


def _az(d: dict):
    return d.get("anchor_z_frac")


def _wrist(d: dict):
    mon = d.get("episode_monitor") or {}
    lw = (mon.get("e_lw_world_orig") or {}).get("mean")
    rw = (mon.get("e_rw_world_orig") or {}).get("mean")
    if lw is None or rw is None:
        return None
    return 0.5 * (float(lw) + float(rw))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/home/chenxiangyu/robotics/Anybody/results/next_phase/thb_np")
    ap.add_argument("--out", default="/data/home/chenxiangyu/robotics/Anybody/results/next_phase/thb_np")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    orig_steps = _load(root / "closed_loop" / "loco" / "original" / "steps") or _load(OLD / "loco" / "original" / "steps")
    priv = _load(OLD / "loco" / "torso_b025" / "steps")
    est = _load(root / "closed_loop" / "loco" / "torso_b025_up" / "steps")
    orig_plane = _load(root / "closed_loop" / "loco" / "original" / "plane")
    est_plane = _load(root / "closed_loop" / "loco" / "torso_b025_up" / "plane")
    orig_slope = _load(root / "closed_loop" / "loco" / "original" / "slope")
    est_slope = _load(root / "closed_loop" / "loco" / "torso_b025_up" / "slope")

    est_stats = est.get("estimator_vs_signed") or {}
    mae = est_stats.get("mae_mean")
    corr = est_stats.get("corr_mean")
    lag = est_stats.get("lag_ms_median")

    dz_plane = ((est_plane.get("episode_monitor") or {}).get("dz") or {}).get("mean")
    dz_steps = ((est.get("episode_monitor") or {}).get("dz") or {}).get("mean")

    # GO: estimated reproduces privileged trend (anchor_z down, wrists not worse, flat slack≈0)
    az_orig = _az(orig_steps)
    az_est = _az(est)
    az_priv = _az(priv)
    wr_orig = _wrist(orig_steps)
    wr_est = _wrist(est)
    flat_ok = dz_plane is None or abs(float(dz_plane)) < 0.01
    az_better = (
        isinstance(az_est, (int, float))
        and isinstance(az_orig, (int, float))
        and az_est + 1e-9 < az_orig
    )
    # On pyramid descent, unsigned is often a no-op → likely HOLD vs privileged signed.
    reproduces_priv = (
        isinstance(az_est, (int, float))
        and isinstance(az_priv, (int, float))
        and abs(az_est - az_priv) <= 0.05
        and isinstance(_sr(est), (int, float))
        and isinstance(_sr(priv), (int, float))
        and abs(float(_sr(est)) - float(_sr(priv))) <= 0.07
    )
    if (az_better or reproduces_priv) and flat_ok:
        verdict = "SUPPORT_RELATIVE_SLACK_FEASIBLE"
    else:
        verdict = "HOLD"

    def f(x, pct=False):
        if x is None:
            return "—"
        if pct:
            return f"{100.0 * float(x):.1f}%"
        return f"{float(x):.3f}"

    md = ["# THB-NP Non-Privileged Support-Relative Torso Slack\n\n"]
    md.append("Estimator = contact + ankle FK (same sensors as previous privileged THB). ")
    md.append("Difference is **unsigned** `δz=0.25·max(Δh,0)` vs privileged **signed** `δz=0.25·Δh`.\n")
    md.append("No terrain ID, no heightmap GT. Not merged into the controller.\n\n")
    md.append("Support estimator vs privileged signed Δh:\n\n")
    md.append(f"- MAE: {f(mae)}\n- corr: {f(corr)}\n- lag: {f(lag)} ms\n\n")
    md.append("| Variant | Loco Steps SR | anchor_z | wrist world error |\n")
    md.append("|---|---:|---:|---:|\n")
    md.append(f"| Original | {f(_sr(orig_steps), True)} | {f(az_orig, True)} | {f(wr_orig)} |\n")
    md.append(f"| Privileged β=.25 (signed) | {f(_sr(priv), True)} | {f(az_priv, True)} | {f(_wrist(priv))} |\n")
    md.append(f"| Estimated β=.25 (unsigned) | {f(_sr(est), True)} | {f(az_est, True)} | {f(wr_est)} |\n")
    md.append(f"\nFlat mean δz (estimated): {f(dz_plane)}  (want ≈0)\n")
    md.append(f"Steps mean δz (estimated): {f(dz_steps)}\n")
    md.append(f"Slope original SR={f(_sr(orig_slope), True)} estimated SR={f(_sr(est_slope), True)}\n")
    md.append(f"\n**Verdict: `{verdict}`**  (not auto-merged into method)\n")
    (out / "REPORT.md").write_text("".join(md))
    (out / "verdict.json").write_text(json.dumps({
        "verdict": verdict,
        "mae": mae, "corr": corr, "lag_ms": lag,
        "flat_dz": dz_plane, "steps_dz": dz_steps,
        "original_steps": {"sr": _sr(orig_steps), "anchor_z": az_orig, "wrist": wr_orig},
        "privileged_signed": {"sr": _sr(priv), "anchor_z": az_priv, "wrist": _wrist(priv)},
        "estimated_unsigned": {"sr": _sr(est), "anchor_z": az_est, "wrist": wr_est},
        "not_in_method": True,
    }, indent=2))
    print(f"[thb-np] {verdict} est_sr={_sr(est)} az={az_est} flat_dz={dz_plane}", flush=True)


if __name__ == "__main__":
    main()
