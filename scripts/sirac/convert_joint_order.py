#!/usr/bin/env python3
"""Name-based G1 joint remapping utility (HTD ↔ AnyBody ↔ arbitrary Isaac order).

Does not assume USD joint indices match URDF order. Prints a JSON mapping.

    python scripts/sirac/convert_joint_order.py --from-names a.txt --to-names b.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
import sirac_isaacfree  # noqa: E402

sirac_isaacfree.install()
sys.path.insert(0, str(ROOT / "source" / "whole_body_tracking"))

from whole_body_tracking.sirac.mappings import G1_JOINT_NAMES_29, LOWER_WAIST_NAMES, lower_indices_in


def _load_names(path: str | None) -> list[str]:
    if path is None:
        return list(G1_JOINT_NAMES_29)
    text = Path(path).read_text().strip()
    if path.endswith(".json"):
        data = json.loads(text)
        return list(data)
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--from-names", default=None, help="source joint-name list (default HTD/AnyBody 29)")
    p.add_argument("--to-names", default=None, help="destination joint-name list (default same)")
    p.add_argument("--lower-only", action="store_true")
    args = p.parse_args()
    src = _load_names(args.from_names)
    dst = _load_names(args.to_names)
    names = LOWER_WAIST_NAMES if args.lower_only else G1_JOINT_NAMES_29
    mapping = []
    for n in names:
        mapping.append({"name": n, "src": src.index(n), "dst": dst.index(n)})
    print(json.dumps({"n": len(mapping), "mapping": mapping}, indent=2))
    # sanity
    lower_indices_in(src)
    lower_indices_in(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
