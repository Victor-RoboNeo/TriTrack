"""Hand-curated letter strokes for the in-air writing demo.

Each letter is a list of **strokes**; each stroke is a list of 2D waypoints in
unit-square letter coordinates (``[0,1] × [0,1]``, origin at lower-left, +x
right, +y up). A letter with multiple strokes (e.g. ``E`` with the middle bar)
is drawn by visiting them in list order.

The helpers in this module compile a string into:
- a single 3D **polyline** the wrist actually traces (continuous; inter-stroke
  segments are direct "pen-up" lines through the air);
- a 3D **trail-point cloud** showing only the stroke waypoints (used by
  ``play.py`` to render the static red dots that visualise the target letter
  shape).

Inter-stroke segments are intentionally NOT in the trail — that way the static
dots show the letter shape cleanly even though the wrist's actual motion
includes connecting moves between strokes.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


# --------------------------------------------------------------------------------------
# 2D stroke library (unit-square coordinates).
#
# Convention: strokes are drawn in the order listed; each stroke is a polyline
# (list of (x, y) waypoints). For letters whose natural drawing order has
# discontinuous strokes (e.g. ``E``'s middle bar), the second+ strokes appear
# as separate list entries.
# --------------------------------------------------------------------------------------
LETTERS: dict[str, list[list[tuple[float, float]]]] = {
    "A": [
        [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)],
        [(0.2, 0.4), (0.8, 0.4)],
    ],
    "B": [
        [(0.0, 0.0), (0.0, 1.0), (0.65, 1.0), (0.9, 0.85), (0.9, 0.65), (0.7, 0.5), (0.0, 0.5)],
        [(0.0, 0.5), (0.7, 0.5), (0.95, 0.35), (0.95, 0.15), (0.7, 0.0), (0.0, 0.0)],
    ],
    "C": [
        [
            (1.0, 0.85), (0.85, 1.0), (0.3, 1.0), (0.05, 0.8), (0.0, 0.55),
            (0.0, 0.45), (0.05, 0.2), (0.3, 0.0), (0.85, 0.0), (1.0, 0.15),
        ],
    ],
    "D": [
        [(0.0, 0.0), (0.0, 1.0), (0.55, 1.0), (0.9, 0.75), (0.95, 0.5), (0.9, 0.25), (0.55, 0.0), (0.0, 0.0)],
    ],
    "E": [
        [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)],
        [(0.0, 0.5), (0.7, 0.5)],
    ],
    "F": [
        [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        [(0.0, 0.5), (0.7, 0.5)],
    ],
    "G": [
        [
            (1.0, 0.85), (0.85, 1.0), (0.3, 1.0), (0.05, 0.8), (0.0, 0.55),
            (0.0, 0.45), (0.05, 0.2), (0.3, 0.0), (0.85, 0.0), (1.0, 0.15),
            (1.0, 0.45), (0.55, 0.45),
        ],
    ],
    "H": [
        [(0.0, 0.0), (0.0, 1.0)],
        [(1.0, 0.0), (1.0, 1.0)],
        [(0.0, 0.5), (1.0, 0.5)],
    ],
    "I": [
        [(0.5, 0.0), (0.5, 1.0)],
    ],
    "J": [
        [(1.0, 1.0), (1.0, 0.2), (0.85, 0.05), (0.5, 0.0), (0.15, 0.05), (0.0, 0.2)],
    ],
    "K": [
        [(0.0, 0.0), (0.0, 1.0)],
        [(0.0, 0.45), (1.0, 1.0)],
        [(0.0, 0.45), (1.0, 0.0)],
    ],
    "L": [
        [(0.0, 1.0), (0.0, 0.0), (1.0, 0.0)],
    ],
    "M": [
        [(0.0, 0.0), (0.0, 1.0), (0.5, 0.4), (1.0, 1.0), (1.0, 0.0)],
    ],
    "N": [
        [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)],
    ],
    "O": [
        [
            (0.5, 1.0), (0.85, 0.92), (1.0, 0.65), (1.0, 0.35), (0.85, 0.08),
            (0.5, 0.0), (0.15, 0.08), (0.0, 0.35), (0.0, 0.65), (0.15, 0.92), (0.5, 1.0),
        ],
    ],
    "P": [
        [(0.0, 0.0), (0.0, 1.0), (0.65, 1.0), (0.9, 0.85), (0.9, 0.65), (0.65, 0.5), (0.0, 0.5)],
    ],
    "Q": [
        [
            (0.5, 1.0), (0.85, 0.92), (1.0, 0.65), (1.0, 0.35), (0.85, 0.08),
            (0.5, 0.0), (0.15, 0.08), (0.0, 0.35), (0.0, 0.65), (0.15, 0.92), (0.5, 1.0),
        ],
        [(0.6, 0.3), (1.0, -0.05)],
    ],
    "R": [
        [(0.0, 0.0), (0.0, 1.0), (0.65, 1.0), (0.9, 0.85), (0.9, 0.65), (0.65, 0.5), (0.0, 0.5)],
        [(0.45, 0.5), (1.0, 0.0)],
    ],
    "S": [
        [
            (1.0, 0.9), (0.85, 1.0), (0.15, 1.0), (0.0, 0.85), (0.0, 0.65), (0.15, 0.5),
            (0.85, 0.5), (1.0, 0.35), (1.0, 0.15), (0.85, 0.0), (0.15, 0.0), (0.0, 0.1),
        ],
    ],
    "T": [
        [(0.0, 1.0), (1.0, 1.0)],
        [(0.5, 1.0), (0.5, 0.0)],
    ],
    "U": [
        [(0.0, 1.0), (0.0, 0.3), (0.15, 0.08), (0.5, 0.0), (0.85, 0.08), (1.0, 0.3), (1.0, 1.0)],
    ],
    "V": [
        [(0.0, 1.0), (0.5, 0.0), (1.0, 1.0)],
    ],
    "W": [
        [(0.0, 1.0), (0.25, 0.0), (0.5, 0.55), (0.75, 0.0), (1.0, 1.0)],
    ],
    "X": [
        [(0.0, 0.0), (1.0, 1.0)],
        [(0.0, 1.0), (1.0, 0.0)],
    ],
    "Y": [
        [(0.0, 1.0), (0.5, 0.5), (1.0, 1.0)],
        [(0.5, 0.5), (0.5, 0.0)],
    ],
    "Z": [
        [(0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0)],
    ],
    "0": [
        [
            (0.5, 1.0), (0.85, 0.92), (1.0, 0.65), (1.0, 0.35), (0.85, 0.08),
            (0.5, 0.0), (0.15, 0.08), (0.0, 0.35), (0.0, 0.65), (0.15, 0.92), (0.5, 1.0),
        ],
    ],
    "1": [
        [(0.2, 0.85), (0.5, 1.0), (0.5, 0.0)],
        [(0.2, 0.0), (0.8, 0.0)],
    ],
    "2": [
        [(0.0, 0.85), (0.15, 1.0), (0.85, 1.0), (1.0, 0.85), (1.0, 0.7), (0.0, 0.0), (1.0, 0.0)],
    ],
    "3": [
        [(0.0, 0.85), (0.15, 1.0), (0.85, 1.0), (1.0, 0.85), (1.0, 0.65), (0.85, 0.5), (0.3, 0.5)],
        [(0.85, 0.5), (1.0, 0.35), (1.0, 0.15), (0.85, 0.0), (0.15, 0.0), (0.0, 0.15)],
    ],
    "4": [
        [(0.0, 1.0), (0.0, 0.4), (1.0, 0.4)],
        [(0.8, 1.0), (0.8, 0.0)],
    ],
    "5": [
        [(1.0, 1.0), (0.0, 1.0), (0.0, 0.55), (0.85, 0.55), (1.0, 0.4), (1.0, 0.15), (0.85, 0.0), (0.15, 0.0), (0.0, 0.1)],
    ],
    "6": [
        [
            (1.0, 0.85), (0.85, 1.0), (0.3, 1.0), (0.05, 0.8), (0.0, 0.5),
            (0.0, 0.2), (0.15, 0.0), (0.85, 0.0), (1.0, 0.15), (1.0, 0.35),
            (0.85, 0.5), (0.15, 0.5), (0.0, 0.4),
        ],
    ],
    "7": [
        [(0.0, 1.0), (1.0, 1.0), (0.3, 0.0)],
    ],
    "8": [
        [
            (0.5, 0.5), (0.15, 0.55), (0.0, 0.75), (0.15, 0.95), (0.5, 1.0), (0.85, 0.95),
            (1.0, 0.75), (0.85, 0.55), (0.5, 0.5), (0.15, 0.45), (0.0, 0.25),
            (0.15, 0.05), (0.5, 0.0), (0.85, 0.05), (1.0, 0.25), (0.85, 0.45), (0.5, 0.5),
        ],
    ],
    "9": [
        [
            (1.0, 0.5), (0.85, 0.6), (0.15, 0.6), (0.0, 0.75), (0.0, 0.9),
            (0.15, 1.0), (0.85, 1.0), (1.0, 0.9), (1.0, 0.15), (0.85, 0.0), (0.15, 0.0),
        ],
    ],
    "!": [
        [(0.5, 1.0), (0.5, 0.25)],
        [(0.5, 0.05), (0.5, 0.0)],
    ],
    "?": [
        [(0.0, 0.85), (0.15, 1.0), (0.85, 1.0), (1.0, 0.85), (1.0, 0.65), (0.5, 0.45), (0.5, 0.25)],
        [(0.5, 0.05), (0.5, 0.0)],
    ],
    ".": [
        [(0.5, 0.05), (0.5, 0.0)],
    ],
    ",": [
        [(0.5, 0.1), (0.4, -0.05)],
    ],
    " ": [],
}


# --------------------------------------------------------------------------------------
# Resampling helpers.
# --------------------------------------------------------------------------------------


def _resample_polyline_2d(stroke: list[tuple[float, float]], step_norm: float) -> np.ndarray:
    """Resample a 2D polyline at uniform arc-length intervals.

    ``step_norm`` is the arc-length step in the *unit-square letter frame*. For
    a 0.04 m sampling step on a letter of size 0.20 m, pass ``step_norm = 0.2``.
    """
    pts = np.array(stroke, dtype=np.float64)
    if pts.shape[0] < 2:
        return pts.copy()
    diffs = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    if total <= 0.0:
        return pts[:1].copy()
    n = max(2, int(math.ceil(total / step_norm)) + 1)
    arc = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=np.float64)
    for i, s in enumerate(arc):
        idx = int(np.searchsorted(cum, s, side="right") - 1)
        idx = max(0, min(idx, len(seg_lens) - 1))
        seg_len = max(float(seg_lens[idx]), 1e-12)
        u = (s - cum[idx]) / seg_len
        out[i] = pts[idx] + u * diffs[idx]
    return out


def _stroke_to_plane(
    stroke_2d: np.ndarray,
    plane_origin: np.ndarray,
    plane_x_axis: np.ndarray,
    plane_y_axis: np.ndarray,
) -> np.ndarray:
    """Map ``(N, 2)`` letter-frame stroke points to ``(N, 3)`` world-frame points."""
    return plane_origin[None, :] + (
        stroke_2d[:, 0:1] * plane_x_axis[None, :] + stroke_2d[:, 1:2] * plane_y_axis[None, :]
    )


# --------------------------------------------------------------------------------------
# Word compiler.
# --------------------------------------------------------------------------------------


def compose_word_strokes_3d(
    text: str,
    letter_size_m: float,
    letter_spacing_m: float,
    plane_origin: np.ndarray,
    plane_x_axis: np.ndarray,
    plane_y_axis: np.ndarray,
    resample_step_m: float = 0.03,
) -> list[np.ndarray]:
    """Return a list of resampled stroke polylines in world frame.

    Each stroke is an ``(N_i, 3)`` array; strokes are NOT joined into a single
    polyline here — the caller can stitch them for the wrist trajectory or use
    them per-stroke for the visualisation trail.
    """
    if letter_size_m <= 0:
        raise ValueError(f"letter_size_m must be > 0; got {letter_size_m}")
    if resample_step_m <= 0:
        raise ValueError(f"resample_step_m must be > 0; got {resample_step_m}")
    step_norm = resample_step_m / letter_size_m

    strokes_3d: list[np.ndarray] = []
    x_cursor = 0.0
    for ch in text.upper():
        if ch not in LETTERS:
            raise KeyError(
                f"Letter {ch!r} not in LETTERS library. Available: {sorted(LETTERS.keys())}"
            )
        letter_strokes = LETTERS[ch]
        for stroke_2d in letter_strokes:
            resampled = _resample_polyline_2d(stroke_2d, step_norm)
            # Scale to metres and offset by cursor.
            resampled_m = resampled * letter_size_m
            resampled_m[:, 0] += x_cursor
            stroke_3d = _stroke_to_plane(resampled_m, plane_origin, plane_x_axis, plane_y_axis)
            strokes_3d.append(stroke_3d)
        x_cursor += letter_size_m + letter_spacing_m
    return strokes_3d


def stitch_strokes(strokes_3d: Iterable[np.ndarray]) -> np.ndarray:
    """Concatenate strokes into a single continuous polyline.

    Inter-stroke connections are implicit straight lines between the END of
    stroke ``i`` and the START of stroke ``i+1``. The wrist actually traverses
    these connecting segments during writing; they're absent from the
    trail-point cloud (see :func:`trail_points_from_strokes`).
    """
    strokes = [s for s in strokes_3d if s.shape[0] > 0]
    if not strokes:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(strokes, axis=0)


def trail_points_from_strokes(strokes_3d: Iterable[np.ndarray]) -> np.ndarray:
    """Return only the per-stroke waypoints (no inter-stroke segments).

    Equivalent to ``stitch_strokes`` for the *set of points*: the inter-stroke
    "pen-up" segments aren't sampled, so they don't appear as red dots. (Within
    a single stroke the resampled waypoints already form a dense outline of the
    letter.)
    """
    return stitch_strokes(strokes_3d)


# --------------------------------------------------------------------------------------
# High-level convenience: build a Polyline3D primitive + a trail-point cloud for
# a string of letters laid out in a writing plane.
# --------------------------------------------------------------------------------------


def write_word(
    text: str,
    *,
    letter_size_m: float,
    letter_spacing_m: float,
    stroke_speed_ms: float,
    plane_origin: tuple[float, float, float],
    plane_x_axis: tuple[float, float, float],
    plane_y_axis: tuple[float, float, float],
    resample_step_m: float = 0.03,
):
    """Compile a string into ``(Polyline3D, trail_points_3d)``.

    The polyline is what the wrist tracks. The trail points are what
    ``play.py`` renders as static red dots — *only* in-stroke waypoints, so the
    letter shape is visible without the inter-stroke connecting lines.

    Returns
    -------
    polyline : Polyline3D
    trail_points : np.ndarray of shape ``(K, 3)``
    """
    from .primitives import Polyline3D

    origin = np.asarray(plane_origin, dtype=np.float64)
    x_axis = np.asarray(plane_x_axis, dtype=np.float64)
    y_axis = np.asarray(plane_y_axis, dtype=np.float64)
    # Normalise plane axes so the letter scale is in metres regardless of input.
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-12)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-12)

    strokes_3d = compose_word_strokes_3d(
        text,
        letter_size_m=letter_size_m,
        letter_spacing_m=letter_spacing_m,
        plane_origin=origin,
        plane_x_axis=x_axis,
        plane_y_axis=y_axis,
        resample_step_m=resample_step_m,
    )
    trail = trail_points_from_strokes(strokes_3d)
    polyline_arr = stitch_strokes(strokes_3d)
    if polyline_arr.shape[0] < 2:
        raise ValueError(f"write_word({text!r}) produced fewer than 2 waypoints; nothing to draw.")
    polyline = Polyline3D(
        waypoints=tuple(tuple(p) for p in polyline_arr),
        speed_ms=stroke_speed_ms,
    )
    return polyline, trail
