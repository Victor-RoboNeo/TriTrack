#!/usr/bin/env python3
"""Side-by-side MP4: GMT (left) vs Pulse (right), with on-frame labels.

For each immediate subdirectory name present under both --gmt-dir and --pulse-dir,
reads one video per side (see --video-name), writes the stitched file only under
the Pulse subdirectory (default: gmt_vs_pulse.mp4). Requires ffmpeg and ffprobe.

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


def _drawtext_vf(label: str, fontfile: str | None) -> str:
    base = (
        "drawtext=text='"
        + label.replace("'", r"\'")
        + "':fontcolor=white:fontsize=28:box=1:boxcolor=black@0.55:boxborderw=6:x=24:y=24"
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
    """Build [i:v] -> tpad (optional) -> scale -> drawtext -> [out_label]."""
    h = max(64, int(height))
    scale = f"scale=-2:{h}:flags=lanczos,setsar=1,{drawtext_part}"
    if pad_sec > 1e-4:
        return (
            f"[{input_idx}:v]tpad=stop_mode=clone:stop_duration={pad_sec:.6f},{scale}[{out_label}]"
        )
    return f"[{input_idx}:v]{scale}[{out_label}]"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stitch GMT (left) and Pulse (right) test videos with labels; "
        "output only under each Pulse subdirectory."
    )
    ap.add_argument(
        "--gmt-dir",
        type=Path,
        required=True,
        help="videos_test root for GMT / mosaic hybrid (left).",
    )
    ap.add_argument(
        "--pulse-dir",
        type=Path,
        required=True,
        help="videos_test root for Pulse (right); outputs are written here per subdir.",
    )
    ap.add_argument(
        "--video-name",
        default="rl-video-step-0.mp4",
        metavar="NAME",
        help="Preferred basename in each subdir; if missing, first *.mp4 alphabetically.",
    )
    ap.add_argument(
        "--output-name",
        default="gmt_vs_pulse.mp4",
        help="Written as PULSE_SUBDIR/OUTPUT_NAME (default: gmt_vs_pulse.mp4).",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=720,
        help="Scale both sides to this height (width follows aspect).",
    )
    ap.add_argument(
        "--left-label",
        default="GMT",
        help="On-frame label for the left (GMT) panel.",
    )
    ap.add_argument(
        "--right-label",
        default="Pulse",
        help="On-frame label for the right (Pulse) panel.",
    )
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("[stitch_gmt_pulse] ffmpeg and ffprobe must be on PATH.", file=sys.stderr)
        sys.exit(1)

    gmt_root = args.gmt_dir.expanduser().resolve()
    pulse_root = args.pulse_dir.expanduser().resolve()
    if not gmt_root.is_dir() or not pulse_root.is_dir():
        print("[stitch_gmt_pulse] --gmt-dir and --pulse-dir must be existing directories.", file=sys.stderr)
        sys.exit(1)

    vname = (args.video_name or "").strip() or None
    exclude = {args.output_name}
    gmt_subs = {p.name: p for p in gmt_root.iterdir() if p.is_dir()}
    pulse_subs = {p.name: p for p in pulse_root.iterdir() if p.is_dir()}
    common = sorted(set(gmt_subs.keys()) & set(pulse_subs.keys()))
    if not common:
        print("[stitch_gmt_pulse] No common subdirectory names between GMT and Pulse roots.", file=sys.stderr)
        sys.exit(1)

    fontfile = _default_fontfile()
    dt_left = _drawtext_vf(args.left_label, fontfile)
    dt_right = _drawtext_vf(args.right_label, fontfile)
    h = max(64, int(args.height))

    for name in common:
        gdir = gmt_subs[name]
        pdir = pulse_subs[name]
        gvid = _resolve_video_in_subdir(gdir, vname, exclude)
        pv = _resolve_video_in_subdir(pdir, vname, exclude)
        if gvid is None:
            print(f"[stitch_gmt_pulse] Skip {name}: no video in GMT dir {gdir}", file=sys.stderr)
            continue
        if pv is None:
            print(f"[stitch_gmt_pulse] Skip {name}: no video in Pulse dir {pdir}", file=sys.stderr)
            continue

        out_path = pdir / args.output_name
        gs = str(gvid)
        ps = str(pv)

        try:
            d0 = _ffprobe_duration(gs)
            d1 = _ffprobe_duration(ps)
        except Exception as exc:
            print(f"[stitch_gmt_pulse] Skip {name}: ffprobe duration failed: {exc}", file=sys.stderr)
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
            gs,
            "-i",
            ps,
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
        print(f"[stitch_gmt_pulse] {name} -> {out_path}")
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"[stitch_gmt_pulse] ffmpeg failed for {name}: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"[stitch_gmt_pulse] Done ({len(common)} pair(s)).")


if __name__ == "__main__":
    main()

"""
python scripts/visualize/stitch_gmt_pulse_side_by_side.py \
  --gmt-dir logs/rsl_rl/g1_flat_mosaic_hybrid/2026-04-11_14-03-04_walk_jog_mdp/videos_test \
  --pulse-dir logs/rsl_rl/g1_flat_pulse_distillation/2026-04-12_10-04-28_mdp_kl_0.01_0.001_4000_8000/videos_test
"""