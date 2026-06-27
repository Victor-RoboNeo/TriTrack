#!/usr/bin/env python3
"""Combine N videos into a single MP4 in a rows×cols grid (ffmpeg xstack).

Each clip is scaled to fit one cell; output resolution matches the first input's frame size
(full canvas), with cell size = (W // cols) × (H // rows). Requires ffmpeg and ffprobe on PATH.

Either pass explicit --inputs (row-major order) or --from-dir to take one video from each
immediate subdirectory (sorted by name), e.g. videos_test/<episode_id>/rl-video-step-0.mp4.
With --from-dir, --output defaults to FROM_DIR/grid_{rows}x{cols}.mp4.

Use --sync-frames (e.g. same as play.py --video_length) to tpad-clone the last frame on any
clip that is shorter, so the grid does not end early when inputs differ (matches prior rollout
defaults when each clip is padded to VIDEO_LENGTH).

Optional --row-labels (one per row) draws a left label on each row; optional --col-labels
(one per column) draws a top-centered label on each column. Use e.g. rows=subfolders and
cols=std mode, or the reverse.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _parse_frame_rate(s: str) -> float:
    s = str(s).strip()
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)


def _ffprobe_fps_and_frame_count(path: str) -> tuple[float, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    raw = subprocess.check_output(cmd, text=True)
    data = json.loads(raw)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe: no video stream in {path}")
    st = streams[0]
    fps_s = st.get("avg_frame_rate") or st.get("r_frame_rate") or "30/1"
    fps = _parse_frame_rate(fps_s)
    nbf = st.get("nb_frames")
    if nbf not in (None, "N/A") and str(nbf).isdigit():
        return fps, int(nbf)
    dur_s = st.get("duration") or (data.get("format") or {}).get("duration")
    dur = float(dur_s) if dur_s else 0.0
    if dur <= 0 or fps <= 0:
        return fps, 0
    return fps, int(round(dur * fps))


def _pad_clone_last_frame(path: str, target_frames: int) -> bool:
    """Extend video in-place by cloning the last frame (same idea as play.py prior_freeze pad)."""
    try:
        fps, n_have = _ffprobe_fps_and_frame_count(path)
    except Exception as exc:
        print(f"[stitch_videos_grid] pad: ffprobe failed for {path}: {exc}", file=sys.stderr)
        return False
    if n_have <= 0 or fps <= 0:
        return False
    need = int(target_frames) - int(n_have)
    if need <= 0:
        return True
    pad_sec = need / fps
    tmp = path + ".stitch_pad_tmp.mp4"
    vf = f"tpad=stop_mode=clone:stop_duration={pad_sec:.6f}"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        tmp,
    ]
    try:
        subprocess.check_call(cmd)
        os.replace(tmp, path)
    except Exception as exc:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        print(f"[stitch_videos_grid] pad: ffmpeg failed for {path}: {exc}", file=sys.stderr)
        return False
    print(f"[stitch_videos_grid] pad: {path} -> ~{target_frames} frames (+{need} clone-frames)")
    return True


def _collect_videos_from_subdirs(base: Path, video_basename: str | None) -> list[Path]:
    """One video per immediate child directory of ``base``, order = sorted subdir names."""
    if not base.is_dir():
        raise FileNotFoundError(f"Not a directory: {base}")
    subdirs = sorted(p for p in base.iterdir() if p.is_dir())
    out: list[Path] = []
    for d in subdirs:
        if video_basename:
            cand = d / video_basename
            if cand.is_file():
                out.append(cand.resolve())
                continue
        mp4s = sorted(d.glob("*.mp4"))
        if not mp4s:
            raise FileNotFoundError(f"No .mp4 in subdirectory: {d}")
        out.append(mp4s[0].resolve())
    return out


def _find_font_file() -> str | None:
    """DejaVu/Liberation if present (drawtext); else None for ffmpeg default font."""
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if os.path.isfile(p):
            return p
    return None


def _escape_filter_path(p: str) -> str:
    """Escape path for use inside ffmpeg filter (textfile=, fontfile=)."""
    return p.replace("\\", "/").replace(":", "\\:")


def _ffprobe_size(path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe: no video stream in {path}")
    w = int(streams[0]["width"])
    h = int(streams[0]["height"])
    return w, h


def main() -> None:
    ap = argparse.ArgumentParser(description="Stitch videos into a grid MP4 (ffmpeg xstack).")
    ap.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Video files in row-major order: row0 left→right, then row1, … (row = first grid dimension). "
        "Omit when using --from-dir.",
    )
    ap.add_argument(
        "--from-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Parent directory: use one video per immediate subdirectory (sorted by subdir name). "
        "See --video-name.",
    )
    ap.add_argument(
        "--video-name",
        default="rl-video-step-0.mp4",
        metavar="NAME",
        help="With --from-dir, prefer this basename inside each subdir; if missing, first *.mp4. "
        "Pass empty string to always pick first *.mp4 alphabetically.",
    )
    ap.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output .mp4 path. With --from-dir, defaults to FROM_DIR/grid_{rows}x{cols}.mp4.",
    )
    ap.add_argument("--grid-rows", type=int, default=4)
    ap.add_argument("--grid-cols", type=int, default=4)
    ap.add_argument(
        "--pad-color",
        default="black",
        help="Pad color when scaling (ffmpeg color name).",
    )
    ap.add_argument(
        "--sync-frames",
        type=int,
        default=None,
        metavar="N",
        help="If set, any input with fewer than N frames is tpad-padded (clone last frame) in place "
        "before stacking. Use the same N as play.py --video_length when combining prior rollouts.",
    )
    ap.add_argument(
        "--no-report-lengths",
        action="store_true",
        help="Do not print per-input frame estimates before stitching.",
    )
    ap.add_argument(
        "--row-labels",
        nargs="*",
        default=None,
        metavar="TEXT",
        help="Exactly --grid-rows labels, drawn on the left of each row (vertically centered).",
    )
    ap.add_argument(
        "--col-labels",
        nargs="*",
        default=None,
        metavar="TEXT",
        help="Exactly --grid-cols labels, drawn near the top of each column (horizontally centered in column).",
    )
    ap.add_argument(
        "--cell-labels",
        nargs="*",
        default=None,
        metavar="TEXT",
        help="Exactly rows×cols labels in row-major order; drawn at the upper-left (x=12,y=12) of each cell "
        "after scaling (independent of --row-labels / --col-labels).",
    )
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("[stitch_videos_grid] ffmpeg and ffprobe must be on PATH.", file=sys.stderr)
        sys.exit(1)

    rows, cols = int(args.grid_rows), int(args.grid_cols)
    slots = rows * cols

    from_dir_resolved: Path | None = None
    if args.from_dir is not None:
        base = args.from_dir.expanduser().resolve()
        from_dir_resolved = base
        vname = (args.video_name or "").strip() or None
        try:
            paths = _collect_videos_from_subdirs(base, vname)
        except FileNotFoundError as exc:
            print(f"[stitch_videos_grid] {exc}", file=sys.stderr)
            sys.exit(1)
        if args.inputs:
            print("[stitch_videos_grid] Ignoring --inputs because --from-dir was set.", file=sys.stderr)
    elif args.inputs:
        paths = [Path(p).expanduser().resolve() for p in args.inputs]
    else:
        ap.error("Provide --inputs VIDEO [VIDEO ...] or --from-dir DIR")

    for p in paths:
        if not p.is_file():
            print(f"[stitch_videos_grid] Missing file: {p}", file=sys.stderr)
            sys.exit(1)

    if len(paths) > slots:
        print(f"[stitch_videos_grid] Truncating inputs from {len(paths)} to {slots}.", file=sys.stderr)
        paths = paths[:slots]
    elif len(paths) < slots:
        print(
            f"[stitch_videos_grid] Need {slots} inputs for {rows}×{cols}; got {len(paths)}. Pad with duplicate last clip.",
            file=sys.stderr,
        )
        while len(paths) < slots:
            paths.append(paths[-1])

    str_paths = [str(p) for p in paths]
    if not args.no_report_lengths:
        counts: list[int] = []
        for p in str_paths:
            try:
                _, nf = _ffprobe_fps_and_frame_count(p)
                counts.append(nf)
            except Exception:
                counts.append(-1)
        print(
            "[stitch_videos_grid] Input frame estimates (min/max): "
            f"{min(counts)}/{max(counts)}  (negative = probe failed)"
        )

    if args.sync_frames is not None and int(args.sync_frames) > 0:
        tgt = int(args.sync_frames)
        print(
            f"[stitch_videos_grid] Syncing inputs to ~{tgt} frames in place (tpad clone; re-encodes if shorter)."
        )
        for p in str_paths:
            if not _pad_clone_last_frame(p, tgt):
                print(f"[stitch_videos_grid] WARNING: could not pad to {tgt} frames: {p}", file=sys.stderr)

    out_w, out_h = _ffprobe_size(paths[0])
    cell_w = max(1, out_w // cols)
    cell_h = max(1, out_h // rows)

    cell_labels_arg = args.cell_labels
    has_cell = cell_labels_arg is not None and len(cell_labels_arg) > 0
    if has_cell and len(cell_labels_arg) != slots:
        ap.error(
            f"--cell-labels requires exactly {slots} entries (row-major, one per cell); "
            f"got {len(cell_labels_arg)}"
        )

    cell_label_paths: list[str] = []
    if has_cell:
        for lab in cell_labels_arg:
            fd, tpath = tempfile.mkstemp(prefix="stitch_cell_lbl_", suffix=".txt", text=True)
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                tf.write(lab)
            cell_label_paths.append(os.path.abspath(tpath))

    font = _find_font_file()
    font_opt = f":fontfile={_escape_filter_path(font)}" if font else ""
    dt_style_row = (
        f"{font_opt}:fontcolor=white:box=1:boxcolor=black@0.62:boxborderw=8:borderw=2:bordercolor=black"
    )
    dt_style_col = (
        f"{font_opt}:fontcolor=white@0.92:box=1:boxcolor=black@0.42:boxborderw=4:borderw=1:bordercolor=black@0.7"
    )
    dt_style_cell = (
        f"{font_opt}:fontcolor=white:box=1:boxcolor=black@0.62:boxborderw=6:borderw=2:bordercolor=black"
    )
    fs_cell = max(14, cell_h // 22)

    # Build filter: scale+crop each stream, optional per-cell label, then xstack
    parts: list[str] = []
    stack_refs: list[str] = []
    for i in range(slots):
        ref = f"v{i}"
        chain = (
            f"[{i}:v]scale={cell_w}:{cell_h}:force_original_aspect_ratio=increase,"
            f"crop={cell_w}:{cell_h}:(iw-ow)/2:(ih-oh)/2,format=yuv420p"
        )
        if has_cell:
            path_esc = _escape_filter_path(cell_label_paths[i])
            chain += f",drawtext=textfile={path_esc}:x=12:y=12:fontsize={fs_cell}{dt_style_cell}"
        chain += f"[{ref}]"
        parts.append(chain)
        stack_refs.append(f"[{ref}]")

    layouts: list[str] = []
    for r in range(rows):
        for c in range(cols):
            x = c * cell_w
            y = r * cell_h
            layouts.append(f"{x}_{y}")
    layout = "|".join(layouts)

    row_labels = args.row_labels
    col_labels = args.col_labels
    has_row = row_labels is not None and len(row_labels) > 0
    has_col = col_labels is not None and len(col_labels) > 0
    if has_row and len(row_labels) != rows:
        ap.error(f"--row-labels requires exactly {rows} entries (one per row); got {len(row_labels)}")
    if has_col and len(col_labels) != cols:
        ap.error(f"--col-labels requires exactly {cols} entries (one per column); got {len(col_labels)}")

    xstack_out = "[gridbase]" if (has_row or has_col) else "[outv]"

    xstack = (
        f"{''.join(stack_refs)}xstack=inputs={slots}:layout={layout}:fill={args.pad_color}:shortest=1{xstack_out}"
    )
    parts.append(xstack)

    label_temp_files: list[str] = []

    if has_col:
        for lab in col_labels:
            fd, tpath = tempfile.mkstemp(prefix="stitch_col_lbl_", suffix=".txt", text=True)
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                tf.write(lab)
            label_temp_files.append(os.path.abspath(tpath))

    if has_row:
        for lab in row_labels:
            fd, tpath = tempfile.mkstemp(prefix="stitch_row_lbl_", suffix=".txt", text=True)
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                tf.write(lab)
            label_temp_files.append(os.path.abspath(tpath))

    # Column labels first (top), then row labels (left), so row labels stay readable at the corner.
    cur = "[gridbase]"
    if has_col:
        n_c = len(col_labels)
        for c, tpath in enumerate(label_temp_files[:n_c]):
            path_esc = _escape_filter_path(tpath)
            xexpr = f"(w*({2 * c}+1))/(2*{cols})-text_w/2"
            last_c = c == n_c - 1
            out_ref = "[postcol]" if (last_c and has_row) else ("[outv]" if last_c and not has_row else f"[cc{c}]")
            parts.append(
                f"{cur}drawtext=textfile={path_esc}:x={xexpr}:y=16:fontsize=h/38{dt_style_col}{out_ref}"
            )
            cur = out_ref

    if has_row:
        row_files = label_temp_files[len(col_labels) :] if has_col else label_temp_files
        row_in = "[postcol]" if has_col else "[gridbase]"
        cur = row_in
        n_r = len(row_files)
        for r, tpath in enumerate(row_files):
            path_esc = _escape_filter_path(tpath)
            yexpr = f"(h*({2 * r}+1))/(2*{rows})-text_h/2"
            last_r = r == n_r - 1
            out_ref = "[outv]" if last_r else f"[lr{r}]"
            parts.append(
                f"{cur}drawtext=textfile={path_esc}:x=24:y={yexpr}:fontsize=h/22{dt_style_row}{out_ref}"
            )
            cur = out_ref

    filter_complex = ";".join(parts)

    if args.output is not None:
        out_path = Path(args.output).expanduser().resolve()
    elif from_dir_resolved is not None:
        out_path = from_dir_resolved / f"grid_{rows}x{cols}.mp4"
    else:
        ap.error("--output (-o) is required when using --inputs without --from-dir")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y"]
    for p in str_paths:
        cmd.extend(["-i", p])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
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
    )
    print("[stitch_videos_grid] Running ffmpeg xstack ->", out_path)
    all_temp_labels = label_temp_files + cell_label_paths
    try:
        subprocess.check_call(cmd)
    finally:
        for p in all_temp_labels:
            try:
                os.unlink(p)
            except OSError:
                pass
    print(f"[stitch_videos_grid] Wrote {out_path}")


if __name__ == "__main__":
    main()



"""
python3 scripts/visualize/stitch_videos_grid.py \
  --from-dir logs/rsl_rl/g1_flat_mosaic_hybrid/2026-04-11_14-03-04_walk_jog_mdp/videos_train \
  --grid-rows 3 --grid-cols 3


python3 scripts/visualize/stitch_videos_grid.py \
  --inputs logs/rsl_rl/g1_flat_pulse_distillation/2026-04-12_10-04-28_mdp_kl_0.01_0.001_4000_8000/prior_videos/230124/rl-video-step-0.mp4 logs/rsl_rl/g1_flat_pulse_distillation/2026-04-12_10-04-28_mdp_kl_0.01_0.001_4000_8000/prior_videos/230301/rl-video-step-0.mp4 logs/rsl_rl/g1_flat_pulse_distillation/2026-04-12_10-04-28_mdp_kl_0.01_0.001_4000_8000/prior_videos/230302/rl-video-step-0.mp4 logs/rsl_rl/g1_flat_pulse_distillation/2026-04-12_10-04-28_mdp_kl_0.01_0.001_4000_8000/prior_videos/230906/rl-video-step-0.mp4 \
  --grid-rows 2 --grid-cols 2 \
  --output logs/rsl_rl/g1_flat_pulse_distillation/2026-04-12_10-04-28_mdp_kl_0.01_0.001_4000_8000/prior_videos/grid_2x2.mp4

# 3x3: rows = subfolders, cols = std — inputs row-major; column labels only (smaller/elegant):
#   --col-labels "predicted std" "fixed std=0.3" "fixed std=0.05"
"""