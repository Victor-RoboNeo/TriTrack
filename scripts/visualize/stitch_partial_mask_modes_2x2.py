#!/usr/bin/env python3
"""2×2 grid per clip folder: end_effector | full / upper | upper_end_effector.

Expects play.py outputs ``rl-video_mask_<mode>-step-0.mp4`` in each immediate subdirectory of
``VIDEOS_PARENT`` (e.g. ``.../videos_test/<clip_id>/``). Writes ``grid_mask_modes_2x2.mp4`` in
each subfolder. Uses ``stitch_videos_grid.py`` (ffmpeg). Requires ffmpeg/ffprobe on PATH.

Example::

 python3 scripts/visualize/stitch_partial_mask_modes_2x2.py \\
 logs/rsl_rl/.../videos_test --sync-frames 200
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Row-major 2×2 layout: top row then bottom row (matches user request order).
MODES_ROW_MAJOR = ["end_effector", "full", "upper", "upper_end_effector"]


def _video_for_mode(subdir: Path, mode: str) -> Path:
    return subdir / f"rl-video_mask_{mode}-step-0.mp4"


def main() -> None:
    ap = argparse.ArgumentParser(description="Stitch four partial-mask play videos into a labeled 2×2 grid.")
    ap.add_argument(
        "videos_parent",
        type=Path,
        help="Directory whose child folders each contain the mask-mode MP4s (e.g. videos_test).",
    )
    ap.add_argument(
        "--output-name",
        default="grid_mask_modes_2x2.mp4",
        help="Output basename written inside each clip subfolder.",
    )
    ap.add_argument(
        "--sync-frames",
        type=int,
        default=None,
        metavar="N",
        help="If set, pad shorter inputs with cloned last frame (same as stitch_videos_grid --sync-frames).",
    )
    args = ap.parse_args()

    parent = args.videos_parent.expanduser().resolve()
    if not parent.is_dir():
        print(f"[stitch_partial_mask_modes_2x2] Not a directory: {parent}", file=sys.stderr)
        sys.exit(1)

    grid_script = Path(__file__).resolve().parent / "stitch_videos_grid.py"
    if not grid_script.is_file():
        print(f"[stitch_partial_mask_modes_2x2] Missing {grid_script}", file=sys.stderr)
        sys.exit(1)

    subdirs = sorted(p for p in parent.iterdir() if p.is_dir())
    if not subdirs:
        print(f"[stitch_partial_mask_modes_2x2] No subdirectories under {parent}", file=sys.stderr)
        sys.exit(1)

    ok = 0
    for d in subdirs:
        inputs: list[str] = []
        skip = False
        for m in MODES_ROW_MAJOR:
            p = _video_for_mode(d, m)
            if not p.is_file():
                print(f"[stitch_partial_mask_modes_2x2] Skip {d.name}: missing {p.name}", file=sys.stderr)
                skip = True
                break
            inputs.append(str(p.resolve()))
        if skip:
            continue

        out = d / args.output_name
        cmd = [
            sys.executable,
            str(grid_script),
            "--inputs",
            *inputs,
            "--grid-rows",
            "2",
            "--grid-cols",
            "2",
            "--cell-labels",
            *MODES_ROW_MAJOR,
            "-o",
            str(out),
        ]
        if args.sync_frames is not None and int(args.sync_frames) > 0:
            cmd.extend(["--sync-frames", str(int(args.sync_frames))])

        print(f"[stitch_partial_mask_modes_2x2] {d.name} -> {out.name}")
        subprocess.check_call(cmd)
        ok += 1

    print(f"[stitch_partial_mask_modes_2x2] Done. Wrote grids for {ok}/{len(subdirs)} subfolder(s).")


if __name__ == "__main__":
    main()
