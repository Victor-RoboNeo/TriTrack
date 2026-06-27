#!/usr/bin/env python3
"""Randomly sample .npz files per subclass into MOSAIC_Dataset/visualize/<class>/.

Reads .npz filenames in --mocap_dir (flat, no subdirs). Class = text between "g1_" and "_rosbag2"
(e.g. g1_dance_rosbag2_... -> dance, g1_in_place_motions_rosbag2_... -> in_place_motions).
Creates MOSAIC_Dataset/visualize/dance/, visualize/in_place_motions/, ... with 0000.npz, 0001.npz, ...
inside each subfolder (symlinks by default). Use --seed for reproducible sampling.

Usage:
  python scripts/prepare_visualize_motions.py
  python scripts/prepare_visualize_motions.py --num_per_class 3 --mocap_dir MOSAIC_Dataset/G1/inertial_mocap --out_dir MOSAIC_Dataset/visualize
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path


def class_from_filename(name: str) -> str | None:
    """Extract class from npz filename: between g1_ and _rosbag2."""
    if not name.endswith(".npz"):
        return None
    base = name[:-4]
    if not base.startswith("g1_") or "_rosbag2_" not in base:
        return None
    return base[3:].split("_rosbag2_")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="First 3 .npz per class (by filename) -> MOSAIC_Dataset/visualize")
    parser.add_argument(
        "--mocap_dir",
        type=Path,
        default=Path("MOSAIC_Dataset/G1/inertial_mocap"),
        help="Flat directory of .npz files (class parsed from filename: g1_<class>_rosbag2_...).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("MOSAIC_Dataset/visualize"),
        help="Output folder; will contain <class>/0000.npz, 0001.npz, ... per class.",
    )
    parser.add_argument("--num_per_class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling (reproducible).")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking.")
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    mocap_dir = args.mocap_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not mocap_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {mocap_dir}")

    # Group .npz by class (class = between g1_ and _rosbag2 in filename)
    class_to_files: dict[str, list[Path]] = {}
    for p in mocap_dir.iterdir():
        if not p.is_file() or p.suffix != ".npz":
            continue
        cls = class_from_filename(p.name)
        if cls is None:
            continue
        class_to_files.setdefault(cls, []).append(p)

    for cls in class_to_files:
        class_to_files[cls] = sorted(class_to_files[cls], key=lambda p: p.name)

    if not class_to_files:
        raise SystemExit(f"No .npz with g1_<class>_rosbag2_ pattern in {mocap_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    n = args.num_per_class
    total = 0
    for class_name in sorted(class_to_files.keys()):
        npz_list = class_to_files[class_name]
        take = min(n, len(npz_list))
        chosen = random.sample(npz_list, take)
        subfolder = out_dir / class_name
        subfolder.mkdir(parents=True, exist_ok=True)
        for i in range(take):
            src = chosen[i]
            dest = subfolder / f"{i:04d}.npz"
            if dest.exists():
                dest.unlink()
            if args.copy:
                shutil.copy2(src, dest)
            else:
                dest.symlink_to(os.path.relpath(src.resolve(), subfolder))
            total += 1
        print(f"  {class_name}: {take} files -> {out_dir.name}/{class_name}/ ({chosen[0].name} ...)")

    print(f"\nDone: {out_dir} has {total} motions in {len(class_to_files)} subfolders (first {n} from each class).")


if __name__ == "__main__":
    main()
