#!/usr/bin/env python3
"""Side-by-side MP4: whole body / full KPI (left) vs upper only (right), top-right labels.

For each immediate subdirectory name present under both --whole-body-dir and --upper-only-dir,
reads one video per side (see --video-name), writes the stitched file only under the
upper-only subdirectory (default: whole_body_vs_upper.mp4). Requires ffmpeg and ffprobe.

The shorter input is extended by cloning the last frame (tpad) so length matches
the longer clip; originals are not modified.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _ffprobe_duration(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def _resolve_video_in_subdir(
    subdir: Path, video_basename: str | None, exclude_names: set[str]
) -> Path | None:
    if video_basename:
        cand = subdir / video_basename
        if cand.is_file():
            return cand.resolve()
    mp4s = sorted(p for p in subdir.glob("*.mp4") if p.name not in exclude_names)
    if not mp4s:
        return None
    return mp4s[0].resolve()


def _default_fontfile() -> str | None:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _drawtext_top_right_vf(label: str, fontfile: str | None) -> str:
    base = (
        "drawtext=text='"
        + label.replace("'", r"\'")
        + "':fontcolor=white:fontsize=h/26:box=1:boxcolor=black@0.55:boxborderw=6:"
        "x=w-tw-16:y=16"
    )
    if fontfile:
        esc = fontfile.replace("\\", "\\\\").replace(":", r"\:")
        return f"{base}:fontfile={esc}"
    return base


def _chain_pad_scale_label(
    input_idx: int,
    pad_sec: float,
    height: int,
    drawtext_part: str,
    out_label: str,
) -> str:
    h = max(64, int(height))
    scale = f"scale=-2:{h}:flags=lanczos,setsar=1,{drawtext_part}"
    if pad_sec > 1e-4:
        return (
            f"[{input_idx}:v]tpad=stop_mode=clone:stop_duration={pad_sec:.6f},{scale}[{out_label}]"
        )
    return f"[{input_idx}:v]{scale}[{out_label}]"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stitch whole-body (left) and upper-only (right) test videos with top-right labels; "
        "output only under each upper-only subdirectory."
    )
    ap.add_argument(
        "--whole-body-dir",
        type=Path,
        required=True,
        help="videos_test root for full KPI / whole body (left).",
    )
    ap.add_argument(
        "--upper-only-dir",
        type=Path,
        required=True,
        help="videos_test root for upper KPI (right); outputs are written here per subdir.",
    )
    ap.add_argument(
        "--video-name",
        default="rl-video-step-0.mp4",
        metavar="NAME",
        help="Preferred basename in each subdir; if missing, first *.mp4 alphabetically.",
    )
    ap.add_argument(
        "--output-name",
        default="whole_body_vs_upper.mp4",
        help="Written as UPPER_ONLY_SUBDIR/OUTPUT_NAME (default: whole_body_vs_upper.mp4).",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=720,
        help="Scale both sides to this height (width follows aspect).",
    )
    ap.add_argument(
        "--left-label",
        default="whole body",
        help="On-frame label for the left panel (top right of that panel).",
    )
    ap.add_argument(
        "--right-label",
        default="upper only",
        help="On-frame label for the right panel (top right of that panel).",
    )
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("[stitch_whole_body_upper] ffmpeg and ffprobe must be on PATH.", file=sys.stderr)
        sys.exit(1)

    left_root = args.whole_body_dir.expanduser().resolve()
    right_root = args.upper_only_dir.expanduser().resolve()
    if not left_root.is_dir() or not right_root.is_dir():
        print(
            "[stitch_whole_body_upper] --whole-body-dir and --upper-only-dir must be existing directories.",
            file=sys.stderr,
        )
        sys.exit(1)

    vname = (args.video_name or "").strip() or None
    exclude = {args.output_name}
    left_subs = {p.name: p for p in left_root.iterdir() if p.is_dir()}
    right_subs = {p.name: p for p in right_root.iterdir() if p.is_dir()}
    common = sorted(set(left_subs.keys()) & set(right_subs.keys()))
    if not common:
        print(
            "[stitch_whole_body_upper] No common subdirectory names between whole-body and upper-only roots.",
            file=sys.stderr,
        )
        sys.exit(1)

    fontfile = _default_fontfile()
    dt_left = _drawtext_top_right_vf(args.left_label, fontfile)
    dt_right = _drawtext_top_right_vf(args.right_label, fontfile)
    h = max(64, int(args.height))

    for name in common:
        ldir = left_subs[name]
        rdir = right_subs[name]
        lvid = _resolve_video_in_subdir(ldir, vname, exclude)
        rvid = _resolve_video_in_subdir(rdir, vname, exclude)
        if lvid is None:
            print(f"[stitch_whole_body_upper] Skip {name}: no video in whole-body dir {ldir}", file=sys.stderr)
            continue
        if rvid is None:
            print(f"[stitch_whole_body_upper] Skip {name}: no video in upper-only dir {rdir}", file=sys.stderr)
            continue

        out_path = rdir / args.output_name
        ls = str(lvid)
        rs = str(rvid)

        try:
            d0 = _ffprobe_duration(ls)
            d1 = _ffprobe_duration(rs)
        except Exception as exc:
            print(f"[stitch_whole_body_upper] Skip {name}: ffprobe duration failed: {exc}", file=sys.stderr)
            continue

        pad0 = max(0.0, d1 - d0)
        pad1 = max(0.0, d0 - d1)

        c0 = _chain_pad_scale_label(0, pad0, h, dt_left, "v0")
        c1 = _chain_pad_scale_label(1, pad1, h, dt_right, "v1")
        vf = f"{c0};{c1};[v0][v1]hstack=inputs=2[outv]"

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            ls,
            "-i",
            rs,
            "-filter_complex",
            vf,
            "-map",
            "[outv]",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        print(f"[stitch_whole_body_upper] {name} -> {out_path}")
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"[stitch_whole_body_upper] ffmpeg failed for {name}: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"[stitch_whole_body_upper] Done ({len(common)} pair(s)).")


if __name__ == "__main__":
    main()
