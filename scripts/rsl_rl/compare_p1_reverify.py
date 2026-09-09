#!/usr/bin/env python3
"""Compare freshly re-run Mapper-B + model_50000 cells to the frozen P1 table."""
from __future__ import annotations

import json
from pathlib import Path

NEW = Path("/data/home/chenxiangyu/robotics/Anybody/results/p1_reverify")
OLD = Path("/data/home/chenxiangyu/robotics/Anybody/results/p2c_fixed_eval/p1_50000")
MASK = {
    "loco": "torso",
    "reach": "head_right",
    "stoop": "vr",
    "carry": "vr",
}
OLD_LOCO = {
    "plane": 53.5,
    "light_rough": 54.85,
    "slope": 34.8,
    "steps": 31.575,
}


def _sr(root: Path, task: str, terrain: str, mask: str) -> float | None:
    p = root / task / terrain / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    try:
        return float(d["modes"]["mapper"][mask]["s42"]["sr_5cm"]) * 100.0
    except (KeyError, TypeError):
        return None


def main() -> None:
    print("=" * 72)
    print("P1 re-verify vs frozen canonical (Mapper-B + model_50000, seed 42)")
    print("=" * 72)
    print(f"{'cell':<28} {'old':>8} {'new':>8} {'dpp':>8}")
    for task, mask in MASK.items():
        for terrain in ("plane", "light_rough", "slope", "steps"):
            new = _sr(NEW, task, terrain, mask)
            old = _sr(OLD, task, terrain, mask)
            if old is None and task == "loco":
                old = OLD_LOCO.get(terrain)
            label = f"{task}/{terrain}/{mask}"
            if new is None:
                print(f"{label:<28} {old if old is not None else float('nan'):8.1f} {'—':>8} {'':>8}")
                continue
            dpp = (new - old) if old is not None else float("nan")
            flag = ""
            if old is not None and abs(dpp) >= 3.0:
                flag = "  DRIFT>=3pp"
            elif old is not None and abs(dpp) >= 1.0:
                flag = "  ~1pp"
            print(f"{label:<28} {old if old is not None else float('nan'):8.1f} {new:8.1f} {dpp:8.1f}{flag}")
    print("=" * 72)


if __name__ == "__main__":
    main()
