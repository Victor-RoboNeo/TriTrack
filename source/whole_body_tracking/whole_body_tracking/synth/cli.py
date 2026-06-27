"""Command-line interface for the synthetic-motion generator.

Subcommands::

    # List recipes.
    python -m whole_body_tracking.synth.cli list

    # Generate ONE recipe.
    python -m whole_body_tracking.synth.cli generate \\
        --recipe squat_15cm_1p0hz \\
        --seed-clip /home/lsn/Datasets/SONIC_npzs/g1/npz_by_motion/test/loco/walk_forward/*.npz \\
        --out-dir /home/lsn/Datasets/SONIC_npzs/g1/npz_synth/squat_15cm_1p0hz

    # Generate ALL recipes (each into its own subdirectory under --out-root).
    python -m whole_body_tracking.synth.cli generate-all \\
        --seed-dir /home/lsn/Datasets/SONIC_npzs/g1/npz_by_motion/test/loco/walk_forward \\
        --out-root /home/lsn/Datasets/SONIC_npzs/g1/npz_synth

Per-recipe output is a directory containing exactly one ``.npz`` (so play.py's
motion-directory loader picks it up cleanly).
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path
from typing import Optional

from .motion_writer import SyntheticMotionBuilder
from .recipes import RECIPES, get_recipe, list_recipes


def _pick_seed(seed_clip: Optional[str], seed_dir: Optional[str], rng_seed: Optional[int]) -> Path:
    """Resolve --seed-clip / --seed-dir into a concrete file path."""
    if seed_clip and seed_dir:
        raise ValueError("Provide either --seed-clip OR --seed-dir, not both.")
    if seed_clip:
        p = Path(seed_clip)
        if not p.is_file():
            raise FileNotFoundError(f"--seed-clip not found: {p}")
        return p
    if seed_dir:
        files = sorted(glob.glob(os.path.join(seed_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"No .npz files under --seed-dir: {seed_dir}")
        if rng_seed is None:
            return Path(files[0])
        rng = random.Random(rng_seed)
        return Path(rng.choice(files))
    raise ValueError("Provide either --seed-clip <npz> or --seed-dir <dir>.")


def _cmd_list(_args: argparse.Namespace) -> int:
    names = list_recipes()
    print(f"[synth.cli] {len(names)} recipes:")
    for n in names:
        r = RECIPES[n]
        print(f"  {n}  ({r.duration_s:.1f}s @ {r.fps}fps)  — {r.description}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    seed = _pick_seed(args.seed_clip, args.seed_dir, args.seed_rng_seed)
    recipe = get_recipe(args.recipe)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{recipe.name}.npz"

    print(f"[synth.cli] recipe={recipe.name} duration={recipe.duration_s:.1f}s fps={recipe.fps}")
    print(f"[synth.cli] seed_clip={seed}")
    print(f"[synth.cli] out={out_path}")

    builder = recipe.make_builder(str(seed), args.robot)
    builder.write(out_path)
    print(f"[synth.cli] wrote {out_path}")
    return 0


def _cmd_generate_all(args: argparse.Namespace) -> int:
    seed = _pick_seed(args.seed_clip, args.seed_dir, args.seed_rng_seed)
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()

    targets = [n for n in list_recipes() if (only is None or n in only) and n not in skip]
    if not targets:
        raise ValueError(f"No recipes selected (only={only}, skip={skip}).")

    print(f"[synth.cli] seed_clip={seed}")
    print(f"[synth.cli] out_root={out_root}")
    print(f"[synth.cli] generating {len(targets)} recipe(s): {targets}")

    for name in targets:
        recipe = get_recipe(name)
        out_dir = out_root / recipe.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{recipe.name}.npz"
        builder = recipe.make_builder(str(seed), args.robot)
        builder.write(out_path)
        print(f"  [{name}] wrote {out_path}")
    print(f"[synth.cli] done ({len(targets)} recipes)")
    return 0


# ---------------------------------------------------------------------------------
# Word-pool helpers (for the RL fine-tuning training distribution)
# ---------------------------------------------------------------------------------
_DICT_AMERICAN_ENGLISH = "/usr/share/dict/american-english"
_DICT_CRACKLIB_SMALL = "/usr/share/dict/cracklib-small"


def _load_wordlist_intersection(min_len: int, max_len: int) -> list[str]:
    """Return the intersection of cracklib-small ∩ american-english under the length
    filter, restricted to lowercase a-z words (no apostrophes, numbers, oddballs).

    Intersection is a cheap "more common" heuristic: a word in both lists is more
    likely to be in everyday use than something only in the huge alphabetical
    american-english dictionary.
    """
    import os

    sources = [p for p in (_DICT_AMERICAN_ENGLISH, _DICT_CRACKLIB_SMALL) if os.path.isfile(p)]
    if len(sources) < 2:
        raise FileNotFoundError(
            f"Need both {_DICT_AMERICAN_ENGLISH} and {_DICT_CRACKLIB_SMALL}; "
            f"found only: {sources}. Install with `apt install american-english cracklib-runtime`."
        )
    sets = []
    for path in sources:
        with open(path) as fh:
            keep = set()
            for ln in fh:
                w = ln.strip()
                if min_len <= len(w) <= max_len and w.isascii() and w.isalpha() and w.islower():
                    keep.add(w)
        sets.append(keep)
    return sorted(sets[0] & sets[1])


def _filter_to_letters_supported(words: list[str], allowed: set[str]) -> list[str]:
    """Keep only words whose uppercased characters are all in the LETTERS library."""
    return [w for w in words if set(w.upper()).issubset(allowed)]


def _cmd_pool_words(args: argparse.Namespace) -> int:
    """Produce a candidate wordlist for the RL training pool."""
    import random as _random

    from .letters import LETTERS

    candidates = _load_wordlist_intersection(args.min_length, args.max_length)
    # Drop any word containing characters our LETTERS dict can't draw.
    candidates = _filter_to_letters_supported(candidates, set(LETTERS.keys()))
    if args.n > len(candidates):
        raise ValueError(
            f"Asked for {args.n} words but the filter only produced {len(candidates)} candidates. "
            f"Relax --min-length / --max-length, or lower --n."
        )
    rng = _random.Random(args.seed)
    picked = rng.sample(candidates, args.n)
    picked.sort()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for w in picked:
            fh.write(w + "\n")
    print(f"[synth.cli pool-words] candidates after filter: {len(candidates)}")
    print(f"[synth.cli pool-words] sampled {len(picked)} words (seed={args.seed})")
    print(f"[synth.cli pool-words] wrote {out}")
    print()
    print("Preview (first 30 + random 30):")
    for w in picked[:30]:
        print(f"  {w}")
    print("  …")
    for w in rng.sample(picked, 30):
        print(f"  {w}")
    print(f"\nReview the full list at {out} and edit if desired, then run:")
    print(f"  python scripts/synth_cli.py pool-generate --words-file {out} --out-root <DIR> [opts]")
    return 0


def _cmd_pool_generate(args: argparse.Namespace) -> int:
    """Generate one npz per word in the wordlist with deterministic per-word random
    sampling of letter_size / plane offsets / stroke speed. Re-running with the same
    --seed reproduces the same params per word."""
    import hashlib
    import random as _random

    from .letters import LETTERS, compose_word_strokes_3d
    import numpy as _np

    from .recipes import build_wrist_writing_builder

    words_path = Path(args.words_file)
    if not words_path.is_file():
        raise FileNotFoundError(f"--words-file not found: {words_path}")
    with words_path.open() as fh:
        words = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
    if not words:
        raise ValueError(f"No words in {words_path}.")

    # Validate every word is renderable.
    allowed = set(LETTERS.keys())
    bad = [w for w in words if not set(w.upper()).issubset(allowed)]
    if bad:
        raise ValueError(
            f"{len(bad)} word(s) contain unsupported chars; first few: {bad[:5]}. "
            f"Edit {words_path} or extend LETTERS."
        )

    seed_path = _pick_seed(args.seed_clip, args.seed_dir, args.seed_rng_seed)
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    body = args.body
    body_tag = {"right_wrist_yaw_link": "rwrist", "left_wrist_yaw_link": "lwrist"}.get(
        body, body.replace("_link", "")
    )

    size_lo, size_hi = float(args.size_min), float(args.size_max)
    fwd_lo, fwd_hi = float(args.forward_min), float(args.forward_max)
    vert_lo, vert_hi = float(args.vertical_min), float(args.vertical_max)
    speed_jit = float(args.speed_jitter)
    if not (0 < size_lo <= size_hi):
        raise ValueError(f"bad size range [{size_lo}, {size_hi}]")
    if not (0 <= speed_jit < 1):
        raise ValueError(f"speed_jitter must be in [0, 1); got {speed_jit}")

    print(f"[pool-generate] words={len(words)}  out_root={out_root}")
    print(f"[pool-generate] seed_clip={seed_path}  global_seed={args.seed}")
    print(
        f"[pool-generate] size~U[{size_lo:.2f},{size_hi:.2f}] m  "
        f"forward~U[{fwd_lo:.2f},{fwd_hi:.2f}] m  vertical~U[{vert_lo:+.2f},{vert_hi:+.2f}] m  "
        f"speed_jitter=±{speed_jit*100:.0f}%"
    )

    def _per_word_rng(word: str) -> _random.Random:
        # Deterministic seed: hash(word|global_seed) → 64-bit int → Random.
        digest = hashlib.md5(f"{word}|{args.seed}".encode()).digest()
        seed = int.from_bytes(digest[:8], "big")
        return _random.Random(seed)

    manifest_path = out_root / "_pool_manifest.tsv"
    manifest_lines = ["word\trecipe_name\tletter_size_m\tplane_forward_m\tplane_vertical_m\tstroke_speed_ms\tduration_s\tnpz_path"]
    n_written = 0
    n_failed = 0
    for idx, word in enumerate(words):
        wrng = _per_word_rng(word)
        letter_size_m = wrng.uniform(size_lo, size_hi)
        letter_spacing_m = round(letter_size_m / 3.0, 4)
        # Auto stroke speed scales with size; then jitter ±speed_jit.
        auto_speed = 0.30 + 1.33 * (letter_size_m - 0.12)
        speed_factor = wrng.uniform(1.0 - speed_jit, 1.0 + speed_jit)
        stroke_speed_ms = max(0.05, auto_speed * speed_factor)
        plane_forward_m = wrng.uniform(fwd_lo, fwd_hi)
        plane_vertical_m = wrng.uniform(vert_lo, vert_hi)
        # Duration from compiled polyline arc + 1 s buffer (same recipe as `write`).
        unit_strokes = compose_word_strokes_3d(
            word.upper(), letter_size_m=letter_size_m, letter_spacing_m=letter_spacing_m,
            plane_origin=_np.zeros(3), plane_x_axis=_np.array([1.0, 0.0, 0.0]),
            plane_y_axis=_np.array([0.0, 1.0, 0.0]),
            resample_step_m=max(0.015, letter_size_m * 0.16),
        )
        total_arc = 0.0
        prev_end = None
        for s in unit_strokes:
            if s.shape[0] >= 2:
                total_arc += float(_np.linalg.norm(_np.diff(s, axis=0), axis=1).sum())
            if prev_end is not None and s.shape[0] >= 1:
                total_arc += float(_np.linalg.norm(s[0] - prev_end))
            if s.shape[0] >= 1:
                prev_end = s[-1]
        total_arc += 0.30  # lift-in
        duration_s = round(total_arc / stroke_speed_ms + 1.0, 1)

        size_cm = int(round(letter_size_m * 100))
        safe_text = "".join(c for c in word.upper() if c.isalnum())
        recipe_name = f"pool_{idx:04d}_{safe_text}_{body_tag}_size{size_cm:02d}cm"
        out_dir = out_root / recipe_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{recipe_name}.npz"

        try:
            builder = build_wrist_writing_builder(
                seed_clip_path=str(seed_path),
                text=word,
                letter_size_m=letter_size_m,
                letter_spacing_m=letter_spacing_m,
                stroke_speed_ms=stroke_speed_ms,
                duration_s=duration_s,
                body_name=body,
                resample_step_m=max(0.015, letter_size_m * 0.16),
                robot=args.robot,
                fps=args.fps,
                plane_forward_offset_m=plane_forward_m,
                plane_vertical_offset_m=plane_vertical_m,
            )
            builder.write(out_path)
            n_written += 1
            manifest_lines.append(
                f"{word}\t{recipe_name}\t{letter_size_m:.4f}\t{plane_forward_m:.4f}\t"
                f"{plane_vertical_m:.4f}\t{stroke_speed_ms:.4f}\t{duration_s:.2f}\t{out_path}"
            )
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            print(f"  [{idx:04d}] FAIL {word!r}: {e}")
            continue
        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1:4d}/{len(words):4d}] written ({n_failed} failures)")

    with manifest_path.open("w") as fh:
        fh.write("\n".join(manifest_lines) + "\n")
    print(f"[pool-generate] DONE: wrote {n_written} npzs, {n_failed} failures.")
    print(f"[pool-generate] manifest: {manifest_path}")
    return 0


def _cmd_write(args: argparse.Namespace) -> int:
    """Ad-hoc wrist writing: arbitrary text + arbitrary letter size (no recipe registry)."""
    from .letters import LETTERS
    from .recipes import build_wrist_writing_builder

    text = args.text
    if not text:
        raise ValueError("--text must be non-empty.")
    missing = sorted({ch.upper() for ch in text if ch.upper() not in LETTERS})
    if missing:
        raise ValueError(
            f"text {text!r} contains characters not in the letter library: {missing}. "
            f"Available: {sorted(LETTERS.keys())}"
        )

    seed = _pick_seed(args.seed_clip, args.seed_dir, args.seed_rng_seed)

    letter_size_m = float(args.letter_size)
    if letter_size_m <= 0:
        raise ValueError(f"--letter-size must be > 0; got {letter_size_m}")
    # Default spacing proportional to size (matches the registered sweep).
    letter_spacing_m = (
        float(args.letter_spacing) if args.letter_spacing is not None else round(letter_size_m / 3.0, 3)
    )
    stroke_speed_ms = (
        float(args.stroke_speed) if args.stroke_speed is not None
        # Scale roughly linearly with size so total drawing time stays in 8-14s range.
        else round(0.30 + 1.33 * (letter_size_m - 0.12), 3)
    )
    stroke_speed_ms = max(0.05, stroke_speed_ms)
    # Default duration = (lift-in + total arc) / speed + 1 s tail buffer.
    if args.duration is not None:
        duration_s = float(args.duration)
    else:
        # Compile the word once to get the actual polyline arc length — way more
        # accurate than a per-letter heuristic (different letters have very
        # different stroke lengths; "I" is 1 unit, "S" is ~3 units, "M" is ~4).
        from .letters import compose_word_strokes_3d
        import numpy as _np

        # Use a unit plane at origin just to measure arc; the actual plane is
        # placed later by build_wrist_writing_builder.
        unit_strokes = compose_word_strokes_3d(
            text,
            letter_size_m=letter_size_m,
            letter_spacing_m=letter_spacing_m,
            plane_origin=_np.zeros(3),
            plane_x_axis=_np.array([1.0, 0.0, 0.0]),
            plane_y_axis=_np.array([0.0, 1.0, 0.0]),
            resample_step_m=max(0.015, letter_size_m * 0.16),
        )
        # Stroke arc + inter-stroke jumps + ~30cm lift-in from wrist natural pose.
        total_arc = 0.0
        prev_end = None
        for s in unit_strokes:
            if s.shape[0] >= 2:
                total_arc += float(_np.linalg.norm(_np.diff(s, axis=0), axis=1).sum())
            if prev_end is not None and s.shape[0] >= 1:
                total_arc += float(_np.linalg.norm(s[0] - prev_end))
            if s.shape[0] >= 1:
                prev_end = s[-1]
        total_arc += 0.30  # lift-in
        duration_s = round(total_arc / stroke_speed_ms + 1.0, 1)
    if duration_s <= 0:
        raise ValueError(f"--duration must be > 0; got {duration_s}")
    fps = int(args.fps)

    # Slug for the recipe name (used as the output subdir + npz filename).
    body_tag = {"right_wrist_yaw_link": "rwrist", "left_wrist_yaw_link": "lwrist"}.get(
        args.body, args.body.replace("_link", "")
    )
    size_cm = int(round(letter_size_m * 100))
    safe_text = "".join(ch for ch in text.upper() if ch.isalnum()) or "WORD"
    recipe_name = args.name or f"write_{safe_text}_{body_tag}_size{size_cm:02d}cm"

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{recipe_name}.npz"

    print(f"[synth.cli write] text={text!r} letter_size={letter_size_m:.3f} m  "
          f"spacing={letter_spacing_m:.3f} m  speed={stroke_speed_ms:.3f} m/s  duration={duration_s:.1f}s")
    print(f"[synth.cli write] body={args.body}  fps={fps}  recipe_name={recipe_name}")
    print(f"[synth.cli write] seed_clip={seed}")
    print(f"[synth.cli write] out={out_path}")

    builder = build_wrist_writing_builder(
        seed_clip_path=str(seed),
        text=text,
        letter_size_m=letter_size_m,
        letter_spacing_m=letter_spacing_m,
        stroke_speed_ms=stroke_speed_ms,
        duration_s=duration_s,
        body_name=args.body,
        resample_step_m=max(0.015, letter_size_m * 0.16),
        robot=args.robot,
        fps=fps,
    )
    builder.write(out_path)
    print(f"[synth.cli write] wrote {out_path}")
    print(f"[synth.cli write] play it with:")
    print(
        f"  python scripts/rsl_rl/play.py --num_envs=1 "
        f"--task=<TASK> --motion {out_dir} --load_run=<RUN> --checkpoint=<CKPT> "
        f"--start_frame=0 --mask_modes right_wrist_only --synth_eval --headless "
        f"--video --video_length={int(duration_s * fps + 50)}"
    )
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List all recipes.")
    p_list.set_defaults(func=_cmd_list)

    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--seed-clip", type=str, default=None, help="Path to a single seed npz.")
        p.add_argument("--seed-dir", type=str, default=None, help="Directory of candidate seed npzs.")
        p.add_argument(
            "--seed-rng-seed",
            type=int,
            default=None,
            help="When --seed-dir is given: deterministic pick. If omitted, the first sorted clip is used.",
        )
        p.add_argument("--robot", type=str, default="g1", help="Robot platform key for body-name cache.")

    p_gen = sub.add_parser("generate", help="Generate ONE recipe.")
    _add_common(p_gen)
    p_gen.add_argument("--recipe", type=str, required=True, help=f"Recipe name. See `list`.")
    p_gen.add_argument("--out-dir", type=str, required=True, help="Output directory (npz is written inside).")
    p_gen.set_defaults(func=_cmd_generate)

    p_all = sub.add_parser("generate-all", help="Generate every recipe.")
    _add_common(p_all)
    p_all.add_argument("--out-root", type=str, required=True, help="Root dir; one subdir per recipe.")
    p_all.add_argument("--only", type=str, default=None, help="Comma-separated whitelist of recipe names.")
    p_all.add_argument("--skip", type=str, default=None, help="Comma-separated blacklist of recipe names.")
    p_all.set_defaults(func=_cmd_generate_all)

    p_write = sub.add_parser(
        "write",
        help=(
            "Ad-hoc wrist writing: draw an arbitrary string at an arbitrary letter size "
            "(no recipe registration needed). Output npz is play.py-ready under --out-dir."
        ),
    )
    _add_common(p_write)
    p_write.add_argument("--text", type=str, required=True,
                         help="String to draw. Letters limited to the LETTERS library "
                              "(A-Z + 0-9 + ! ? . , and space).")
    p_write.add_argument("--letter-size", type=float, required=True,
                         help="Letter height in metres (e.g. 0.20 for 20cm letters).")
    p_write.add_argument("--letter-spacing", type=float, default=None,
                         help="Spacing between letters in metres. Default: letter_size/3.")
    p_write.add_argument("--stroke-speed", type=float, default=None,
                         help="Wrist ground speed in m/s. Default: scales with letter size.")
    p_write.add_argument("--duration", type=float, default=None,
                         help="Total motion duration in seconds. Default: computed from arc + speed.")
    p_write.add_argument("--fps", type=int, default=50, help="Output frame rate (default 50).")
    p_write.add_argument("--body", type=str, default="right_wrist_yaw_link",
                         help="Body that draws the letters (default: right_wrist_yaw_link).")
    p_write.add_argument("--name", type=str, default=None,
                         help="Override the recipe name (default: derived from text + body + size).")
    p_write.add_argument("--out-dir", type=str, required=True,
                         help="Output directory; the npz is written inside as <name>.npz.")
    p_write.set_defaults(func=_cmd_write)

    # ---- Pool subcommands (RL training pool of randomized wrist-writing recipes) ----
    p_pool_words = sub.add_parser(
        "pool-words",
        help="Sample a candidate wordlist from system dictionaries (cracklib ∩ american-english), "
             "filtered to letters renderable by the LETTERS dict. Writes one word per line for review.",
    )
    p_pool_words.add_argument("--n", type=int, default=2000, help="Number of words to sample (default 2000).")
    p_pool_words.add_argument("--seed", type=int, default=42, help="RNG seed for the sample (default 42).")
    p_pool_words.add_argument("--min-length", type=int, default=3, help="Min word length (default 3).")
    p_pool_words.add_argument("--max-length", type=int, default=8, help="Max word length (default 8).")
    p_pool_words.add_argument("--out", type=str, required=True, help="Output text file (one word per line).")
    p_pool_words.set_defaults(func=_cmd_pool_words)

    p_pool_gen = sub.add_parser(
        "pool-generate",
        help="Read a wordlist and generate one synth npz per word with deterministic per-word random "
             "sampling of letter_size / plane offsets / stroke speed. Re-running with the same --seed "
             "produces bit-identical per-word params.",
    )
    _add_common(p_pool_gen)
    p_pool_gen.add_argument("--words-file", type=str, required=True, help="Path to wordlist (one word per line).")
    p_pool_gen.add_argument("--out-root", type=str, required=True, help="Root dir; one subdir per recipe.")
    p_pool_gen.add_argument("--seed", type=int, default=42, help="Global seed for per-word param RNG (default 42).")
    p_pool_gen.add_argument("--body", type=str, default="right_wrist_yaw_link")
    p_pool_gen.add_argument("--fps", type=int, default=50)
    p_pool_gen.add_argument("--size-min", type=float, default=0.15, help="Min letter size in m (default 0.15).")
    p_pool_gen.add_argument("--size-max", type=float, default=0.45, help="Max letter size in m (default 0.45).")
    p_pool_gen.add_argument("--forward-min", type=float, default=0.30, help="Min plane forward offset (default 0.30 m).")
    p_pool_gen.add_argument("--forward-max", type=float, default=0.55, help="Max plane forward offset (default 0.55 m).")
    p_pool_gen.add_argument("--vertical-min", type=float, default=-0.05, help="Min plane vertical offset (default -0.05 m).")
    p_pool_gen.add_argument("--vertical-max", type=float, default=0.20, help="Max plane vertical offset (default +0.20 m).")
    p_pool_gen.add_argument("--speed-jitter", type=float, default=0.25, help="Fractional jitter around auto stroke speed (default 0.25 = ±25%%).")
    p_pool_gen.set_defaults(func=_cmd_pool_generate)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
