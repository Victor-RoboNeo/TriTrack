#!/usr/bin/env python3
"""
Compute average motion length (frames/steps) across a directory of .npz motions.

Motion length is computed as `joint_pos.shape[0]` by default, matching how the
simulators/motion loaders derive `time_step_total`.

Examples:
  python scripts/avg_npz_motion_steps.py --dir /home/lsn/MOSAIC/motions --recursive
  python scripts/avg_npz_motion_steps.py --dir /home/lsn/Datasets/bones-seed/g1/npz_splits/train --recursive
  python scripts/avg_npz_motion_steps.py --dir MOSAIC_Dataset/G1 --recursive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute average motion length (frames/steps) across .npz motion files."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Root directory containing .npz files (searched recursively by default).",
    )
    parser.add_argument(
        "--file_glob",
        type=str,
        default="*.npz",
        help='Glob pattern for .npz files (default: "*.npz").',
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search under --dir. (If unset, only searches direct children.)",
    )
    parser.add_argument(
        "--length_key",
        type=str,
        default="joint_pos",
        help='Array key used to compute length (default: "joint_pos").',
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Optional cap on number of files processed (0 = no cap).",
    )
    parser.add_argument(
        "--print_each",
        action="store_true",
        help="Print per-file lengths (can be very verbose).",
    )
    return parser.parse_args()


def iter_motion_files(root: Path, file_glob: str, recursive: bool) -> list[Path]:
    if recursive:
        paths = sorted(root.rglob(file_glob))
    else:
        paths = sorted(root.glob(file_glob))
    return [p for p in paths if p.is_file()]


def compute_length(npz_path: Path, length_key: str) -> tuple[int, float | None]:
    # mmap_mode keeps this lightweight for large motion arrays.
    with np.load(npz_path, mmap_mode="r", allow_pickle=True) as data:
        if length_key in data:
            arr = data[length_key]
        elif "joint_pos" in data:
            arr = data["joint_pos"]
        else:
            # Fall back to the first array-like entry we can interpret.
            arr = None
            for k in data.keys():
                v = data[k]
                if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 1:
                    arr = v
                    break
            if arr is None:
                raise KeyError(f"Could not find an array to compute length in {npz_path}")

        length = int(arr.shape[0])

        fps_val: float | None = None
        if "fps" in data:
            fps_arr = np.asarray(data["fps"]).reshape(-1)
            if fps_arr.size > 0:
                fps_val = float(fps_arr[0])

        return length, fps_val


def main() -> None:
    args = parse_args()
    root = args.dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Error: --dir is not a directory: {root}")

    motion_files = iter_motion_files(root, args.file_glob, args.recursive)
    if args.max_files and args.max_files > 0:
        motion_files = motion_files[: args.max_files]

    if not motion_files:
        print(f"[WARN] No files matched under {root} with pattern '{args.file_glob}'")
        return

    lengths: list[int] = []
    seconds: list[float] = []

    for p in motion_files:
        length, fps_val = compute_length(p, args.length_key)
        lengths.append(length)
        if fps_val is not None and fps_val > 0:
            seconds.append(length / fps_val)
        if args.print_each:
            if fps_val is None:
                print(f"{p}: length_frames={length}")
            else:
                print(f"{p}: length_frames={length} fps={fps_val:.3f} seconds={length / fps_val:.3f}")

    total = sum(lengths)
    n = len(lengths)
    avg = total / n
    min_v = min(lengths)
    max_v = max(lengths)
    # Median without importing statistics (keeps deps stable).
    sorted_lengths = sorted(lengths)
    median = sorted_lengths[n // 2] if (n % 2 == 1) else 0.5 * (sorted_lengths[n // 2 - 1] + sorted_lengths[n // 2])

    print(f"[INFO] Count: {n} .npz file(s)")
    print(f"[INFO] Length key: {args.length_key!r} (fallbacks enabled)")
    print(f"[INFO] Frames: avg={avg:.2f} min={min_v} median={median:.2f} max={max_v} total={total}")

    if seconds:
        sec_avg = sum(seconds) / len(seconds)
        sec_min = min(seconds)
        sec_max = max(seconds)
        print(f"[INFO] Seconds (using per-file fps when present): avg={sec_avg:.3f}s min={sec_min:.3f}s max={sec_max:.3f}s")
    else:
        print("[INFO] Seconds: skipped (no valid per-file `fps` found).")


if __name__ == "__main__":
    main()

