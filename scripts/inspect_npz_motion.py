"""Print the structure of a .npz motion reference file.

.. code-block:: bash

    python scripts/inspect_npz_motion.py MOSAIC_Dataset/G1/inertial_mocap/g1_dance_rosbag2_20251210_112436_0_0_299.npz
    python scripts/inspect_npz_motion.py motion1.npz motion2.npz
    python scripts/inspect_npz_motion.py --dir path/to/motions/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def inspect_npz(path: Path) -> None:
    """Print structure and stats for one .npz file."""
    print(f"\n{'='*60}")
    print(f"  {path.name}")
    print(f"  {path.resolve()}")
    print("=" * 60)

    with np.load(path, allow_pickle=True) as data:
        keys = list(data.keys())
        print(f"  Keys ({len(keys)}): {keys}\n")

        for key in keys:
            arr = data[key]
            if hasattr(arr, "shape"):
                print(f"  {key}")
                print(f"    shape: {arr.shape}")
                print(f"    dtype: {arr.dtype}")
                if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                    print(f"    min:  {np.min(arr):.6g}")
                    print(f"    max:  {np.max(arr):.6g}")
                    print(f"    mean: {np.mean(arr):.6g}")
                elif arr.shape == () or (arr.ndim == 1 and arr.size <= 4):
                    print(f"    value: {arr}")
                print()
            else:
                print(f"  {key}: (non-array) {type(arr)}")
                print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print structure of .npz motion reference file(s).",
        epilog="Example: python scripts/inspect_npz_motion.py path/to/motion.npz",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Path(s) to .npz file(s) or directory containing .npz",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Inspect all .npz files under DIR",
    )
    args = parser.parse_args()

    if args.dir is not None:
        paths = sorted(args.dir.rglob("*.npz"))
        if not paths:
            print(f"No .npz files found under {args.dir}")
            return
        print(f"Found {len(paths)} .npz file(s) under {args.dir}")
    else:
        if not args.paths:
            parser.print_help()
            print("\nError: provide at least one path or use --dir.")
            return
        paths = []
        for p in args.paths:
            p = Path(p)
            if p.is_dir():
                paths.extend(sorted(p.rglob("*.npz")))
            elif p.suffix.lower() == ".npz" and p.exists():
                paths.append(p)
            elif p.exists():
                print(f"Skip (not .npz): {p}")
            else:
                print(f"Skip (not found): {p}")
        if not paths:
            print("No .npz files to inspect.")
            return

    for path in paths:
        try:
            inspect_npz(path)
        except Exception as e:
            print(f"\n[ERROR] {path}: {e}\n")


if __name__ == "__main__":
    main()
