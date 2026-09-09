#!/usr/bin/env python3
"""Build Anybody sonic_102k loco-manip CSV list from Bones-SEED G1 files.

Filters:
1. Official GEAR-SONIC filename keywords (furniture / climb / sit / acrobatics / ...).
2. AnyBody paper extras for loco-manipulation: crawl / jump (and close variants).
3. Optional min duration from CSV row count at 120 fps.

Writes a keep-list and copies/symlinks CSVs into:
  datasets/SONIC_npzs/g1/csv_splits_loco_manip/train
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

# From NVlabs/GR00T-WholeBodyControl gear_sonic/data_process/filter_and_copy_bones_data.py
SONIC_KEYWORDS = [
    "bed",
    "bike",
    "chair",
    "climb",
    "com_up_50cm",
    "sitting",
    "step_on",
    "seat",
    "table",
    "_sit_",
    "sit_",
    "ladder",
    "crutch",
    "_bed_",
    "_ride_",
    "scooter",
    "stepdown",
    "acrobatics_",
    "box_HSPU",
    "cartwheel",
    "50cm_box_",
    "on_box",
    "fall_from",
    "handstand_ff_",
    "on_1m",
    "form_box",
    "off_1m",
    "230m",
    "jump_over_obstacle_",
    "lift_crate_come_up_",
    "jump_to_shoulder_roll",
    "kozak_dance",
    "stair",
    "handstand",
    "box_jump",
    "monkey_jump",
    "safety_roll",
    "box_dips",
    "walking_on_edge",
    "push_obstacle",
]

# AnyBody paper: drop crawling and highly dynamic jumping for loco-manipulation.
ANYBODY_EXTRA = [
    "crawl",
    "crawling",
    "jump",
]


def should_drop(name: str, keywords: list[str]) -> bool:
    n = name.lower()
    return any(k.lower() in n for k in keywords)


def csv_duration_s(path: Path, fps: float = 120.0) -> float:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        n = sum(1 for line in f if line.strip())
    # header row
    frames = max(0, n - 1)
    return frames / fps if fps > 0 else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv_root", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    p.add_argument("--min_duration", type=float, default=3.0)
    p.add_argument("--skip_duration", action="store_true")
    p.add_argument("--anybody_jump_crawl", action="store_true", default=True)
    p.add_argument("--no_anybody_jump_crawl", action="store_true")
    args = p.parse_args()
    extra = [] if args.no_anybody_jump_crawl else ANYBODY_EXTRA
    keywords = SONIC_KEYWORDS + extra

    csvs = sorted(args.csv_root.rglob("*.csv"))
    if not csvs:
        raise SystemExit(f"no csv under {args.csv_root}")

    keep: list[Path] = []
    dropped_kw = 0
    dropped_dur = 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    for src in csvs:
        rel = str(src.relative_to(args.csv_root))
        if should_drop(rel, keywords) or should_drop(src.name, keywords):
            dropped_kw += 1
            continue
        if not args.skip_duration:
            try:
                dur = csv_duration_s(src)
            except Exception:
                dropped_dur += 1
                continue
            if dur < args.min_duration:
                dropped_dur += 1
                continue
        keep.append(src)

    with args.manifest.open("w", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relpath", "abs_src"])
        for src in keep:
            rel = src.relative_to(args.csv_root)
            dst = args.out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.mode == "copy":
                shutil.copy2(src, dst)
            else:
                os.symlink(src, dst)
            w.writerow([str(rel), str(src)])

    print(f"total={len(csvs)} keep={len(keep)} drop_keyword={dropped_kw} drop_duration={dropped_dur}")
    print(f"keywords={keywords}")
    print(f"out={args.out_dir}")
    print(f"manifest={args.manifest}")


if __name__ == "__main__":
    main()
