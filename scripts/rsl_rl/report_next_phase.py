#!/usr/bin/env python3
"""Assemble next_phase REPORT.md from H2R-1 / UCR-2P / THB-NP verdicts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/data/home/chenxiangyu/robotics/Anybody/results/next_phase")


def _read(p: Path) -> str:
    return p.read_text() if p.is_file() else f"(missing {p})\n"


def _j(p: Path) -> dict:
    return json.loads(p.read_text()) if p.is_file() else {}


def main() -> None:
    h = _j(ROOT / "h2r1_live_human" / "verdict.json")
    u = _j(ROOT / "ucr2p_process_recovery" / "verdict.json")
    t = _j(ROOT / "thb_np" / "verdict.json")
    parts = [
        "# Phase NEXT — Live Human Intent + Process-Conditioned Recovery Diagnosis\n\n",
        "Canonical Parent frozen. Recovery OFF in H2R-1. THB OFF in H2R-1. ",
        "UCR-2P clone/offline only. No task ID / terrain ID / experts.\n\n",
        "------------------------------------------------------------\nA. H2R-1\n------------------------------------------------------------\n\n",
        _read(ROOT / "h2r1_live_human" / "REPORT.md"),
        "\n------------------------------------------------------------\nB. UCR-2P\n------------------------------------------------------------\n\n",
        _read(ROOT / "ucr2p_process_recovery" / "REPORT.md"),
        "\n------------------------------------------------------------\nC. THB-NP\n------------------------------------------------------------\n\n",
        _read(ROOT / "thb_np" / "REPORT.md"),
        "\n------------------------------------------------------------\nDecision tree\n------------------------------------------------------------\n\n",
        f"- H2R-1 `{h.get('verdict', 'PENDING')}` → "
        + ("allow REAL ROBOT PARENT-ONLY integration (do not wait for recovery).\n" if h.get("verdict") == "LIVE_HUMAN_GO" else "HOLD live human; debug frame/calib/timestamp/scale/heading/latency. No full-body retarget.\n"),
        f"- UCR-2P `{u.get('verdict', 'PENDING')}` → "
        + ("next: unified temporal recovery field, held-out clone, 4-task closed-loop.\n" if u.get("verdict") == "PROCESS_IDENTIFIABLE" else "learned recovery STOPS. Keep oracle freedom + negative identifiability. Method = analytic H2R + Mapper-B + frozen Parent.\n"),
        f"- THB-NP `{t.get('verdict', 'PENDING')}` → "
        + ("record SUPPORT_RELATIVE_SLACK_FEASIBLE; do not auto-merge; wait for real robot.\n" if t.get("verdict") == "SUPPORT_RELATIVE_SLACK_FEASIBLE" else "HOLD unsigned slack for final method.\n"),
    ]
    (ROOT / "REPORT.md").write_text("".join(parts))
    (ROOT / "verdicts.json").write_text(json.dumps({
        "h2r1": h.get("verdict"), "ucr2p": u.get("verdict"), "thb_np": t.get("verdict"),
    }, indent=2))
    print((ROOT / "REPORT.md").read_text()[:4000], flush=True)


if __name__ == "__main__":
    main()
