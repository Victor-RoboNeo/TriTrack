#!/usr/bin/env python
"""CLI: resample every NPZ clip in a motion directory by a speed factor.

Example::

    python scripts/motion_tools/resample_motion_dir.py \\
        --input /data/sonic/test_set \\
        --output /data/sonic/test_set__speed1p5 \\
        --speed 1.5

Output NPZs are written with the same filenames; an extra ``resample_speed`` scalar field
is added for traceability. Use ``MOTION_DIR=<output>`` in the eval shell script to consume.
"""
from __future__ import annotations

import argparse
import importlib.util
import os

# Load resample.py directly (the parent package's __init__.py imports isaaclab — pointless
# for an offline CLI that only needs numpy).
_RESAMPLE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        "source",
        "whole_body_tracking",
        "whole_body_tracking",
        "motion_tools",
        "resample.py",
    )
)
_spec = importlib.util.spec_from_file_location("resample", _RESAMPLE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
resample_motion_dir = _mod.resample_motion_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Input motion directory.")
    p.add_argument("--output", required=True, help="Output motion directory (created if absent).")
    p.add_argument(
        "--speed",
        type=float,
        required=True,
        help="Speed factor. >1 = faster (shorter clip); <1 = slower (longer clip). "
        "Velocity tensors are scaled by this factor; fps is preserved.",
    )
    p.add_argument("--file_glob", default="*.npz")
    args = p.parse_args()

    written = resample_motion_dir(
        in_dir=args.input,
        out_dir=args.output,
        speed=float(args.speed),
        file_glob=args.file_glob,
    )
    print(f"[resample_motion_dir] wrote {len(written)} clips to {args.output}", flush=True)


if __name__ == "__main__":
    main()
