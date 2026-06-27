"""Temporal resampling of motion clip NPZs.

Given a directory of clip NPZs and a speed factor ``s``:

- Output frame count = ``round(T_in / s)``; output fps is unchanged (rendering cadence held fixed).
- Positions (``joint_pos``, ``body_pos_w``): linear interpolation in original time.
- Velocities (``joint_vel``, ``body_lin_vel_w``, ``body_ang_vel_w``): linear interpolation,
  then scaled by ``s`` (a single output dt covers ``s × dt_in`` of original motion).
- Orientations (``body_quat_w``): spherical linear interpolation.

This isolates the *velocity statistics* axis from the spatial trajectory: spatial paths
are preserved up to interpolation; only the velocity envelope changes. That's the right
control for testing "dragged at variable speed" generalization.

Quaternion convention: matches the input clips (Isaac Lab uses (w, x, y, z)).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


# Canonical NPZ fields used by ``MotionLoader`` in tasks/tracking/mdp/commands.py.
REQUIRED_FIELDS_POS = ("joint_pos", "body_pos_w")
REQUIRED_FIELDS_VEL = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")
REQUIRED_FIELD_QUAT = "body_quat_w"


def _interp_linear(arr: np.ndarray, t_in_frames: np.ndarray) -> np.ndarray:
    """Linear interpolation of an array shaped ``[T, ...]`` at fractional indices ``t_in_frames``."""
    T_in = arr.shape[0]
    t = np.clip(t_in_frames, 0.0, T_in - 1.0)
    lo = np.floor(t).astype(np.int64)
    hi = np.minimum(lo + 1, T_in - 1)
    frac = (t - lo).astype(arr.dtype)
    # Broadcast frac over trailing dims.
    expand_shape = (-1,) + (1,) * (arr.ndim - 1)
    frac = frac.reshape(expand_shape)
    return (1.0 - frac) * arr[lo] + frac * arr[hi]


def _slerp_quat(quats: np.ndarray, t_in_frames: np.ndarray) -> np.ndarray:
    """Spherical linear interpolation of unit quaternions shaped ``[T, B, 4]`` (w, x, y, z).

    Returns shape ``[T_out, B, 4]``. Per-pair sign-flips ensure the shortest-arc path.
    """
    T_in, B, _ = quats.shape
    t = np.clip(t_in_frames, 0.0, T_in - 1.0)
    lo = np.floor(t).astype(np.int64)
    hi = np.minimum(lo + 1, T_in - 1)
    frac = (t - lo)

    q0 = quats[lo]  # [T_out, B, 4]
    q1 = quats[hi]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    # Flip q1 where dot < 0 (shortest path).
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.where(dot < 0.0, -dot, dot)
    dot = np.clip(dot, -1.0, 1.0)

    # If quats are very close, fall back to lerp (avoid sin(0) division).
    close = dot > 1.0 - 1e-6
    omega = np.arccos(dot)  # [T_out, B, 1]
    sin_omega = np.sin(omega)
    f = frac.reshape((-1,) + (1,) * (quats.ndim - 1))
    s0 = np.where(close, 1.0 - f, np.sin((1.0 - f) * omega) / np.where(close, 1.0, sin_omega))
    s1 = np.where(close, f, np.sin(f * omega) / np.where(close, 1.0, sin_omega))
    out = s0 * q0 + s1 * q1
    # Renormalize (slerp is exact in theory but float drift is real).
    out = out / np.linalg.norm(out, axis=-1, keepdims=True).clip(min=1e-12)
    return out


def resample_clip(in_npz: dict, speed: float) -> dict:
    """Resample a single clip dict (already-loaded NPZ keys → arrays) at the given speed factor.

    ``speed > 1`` shortens the clip (faster), ``speed < 1`` stretches (slower). ``fps`` is preserved.
    Returns a new dict with the same keys as the input plus a ``resample_speed`` scalar.
    """
    if speed <= 0.0:
        raise ValueError(f"speed must be > 0, got {speed!r}")
    fps = float(in_npz["fps"])
    T_in = int(in_npz["joint_pos"].shape[0])
    if T_in < 2:
        raise ValueError(f"Clip has < 2 frames ({T_in}); cannot resample.")
    T_out = max(2, int(round(T_in / float(speed))))
    # Fractional original-frame indices to sample at.
    t_in_frames = np.arange(T_out, dtype=np.float64) * (float(T_in - 1) / float(T_out - 1))

    out = {"fps": np.array(fps, dtype=np.float32), "resample_speed": np.array(float(speed), dtype=np.float32)}
    for k in REQUIRED_FIELDS_POS:
        out[k] = _interp_linear(in_npz[k], t_in_frames).astype(in_npz[k].dtype)
    for k in REQUIRED_FIELDS_VEL:
        # Interpolate, then scale by speed (per-frame dt is fixed, but each step covers ``speed×dt`` of motion).
        out[k] = (_interp_linear(in_npz[k], t_in_frames) * float(speed)).astype(in_npz[k].dtype)
    out[REQUIRED_FIELD_QUAT] = _slerp_quat(in_npz[REQUIRED_FIELD_QUAT], t_in_frames).astype(
        in_npz[REQUIRED_FIELD_QUAT].dtype
    )
    # Preserve any extra fields (e.g. ``body_indexes``) untouched.
    for k, v in in_npz.items():
        if k not in out:
            out[k] = v
    return out


def _load_npz(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def resample_motion_dir(in_dir: str, out_dir: str, speed: float, file_glob: str = "*.npz") -> list[str]:
    """Resample every NPZ in ``in_dir`` and write to ``out_dir`` with the same filenames.

    Returns the list of output paths. Skips non-matching files.
    """
    in_p = Path(in_dir)
    out_p = Path(out_dir)
    if not in_p.is_dir():
        raise FileNotFoundError(f"Input motion dir not found: {in_dir}")
    out_p.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for src in sorted(in_p.rglob(file_glob)):
        if not src.is_file():
            continue
        rel = src.relative_to(in_p)
        dst = out_p / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        clip = _load_npz(str(src))
        new_clip = resample_clip(clip, speed)
        np.savez(str(dst), **new_clip)
        written.append(str(dst))
        print(f"[resample] {src.name}: T_in={clip['joint_pos'].shape[0]} → T_out={new_clip['joint_pos'].shape[0]} (speed={speed})")
    if not written:
        raise RuntimeError(f"No NPZ files matched {file_glob!r} under {in_dir!r}.")
    return written
