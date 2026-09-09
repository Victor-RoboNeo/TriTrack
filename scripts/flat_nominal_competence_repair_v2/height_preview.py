"""HEIGHT_PREVIEW_ONLY suite. H1B is chest-only, never 'head only'."""
from __future__ import annotations

import json

import numpy as np

from flat_locomani.reference_suite import FPS, H0_M, TrajectoryMeta, _alloc, _standing_pose, min_jerk, save_suite

from .constants import RESULTS
from .manifests import geometry_ok


def _shift(delta_h: float, mode: str, k: int):
    pose = _standing_pose(H0_M)
    T = 180
    traj = _alloc(T, pose)
    u = min_jerk(np.linspace(0, 1, T // 2))
    if mode == "H1A":
        traj[: T // 2, :, 2] += u[:, None] * delta_h
        traj[T // 2 :, :, 2] = traj[T // 2 - 1, :, 2]
        fam = "H1A_WHOLE_BODY"
    elif mode == "H1B":
        traj[: T // 2, 0, 2] += u * delta_h
        traj[T // 2 :, 0, 2] = traj[T // 2 - 1, 0, 2]
        fam = "H1B_CHEST_ONLY"
    else:
        raise ValueError(mode)
    meta = TrajectoryMeta(
        traj_id=f"{mode}_dz{delta_h:+.2f}_{k:02d}",
        family=fam,
        hand_mode="world_fixed" if mode == "H1B" else "body_follow",
        source_semantics=mode,
        h0_m=H0_M,
        target_height_drop_m=max(0.0, -delta_h),
        transition_duration_s=T / 2 / FPS,
        low_hold_s=T / 2 / FPS,
        horizontal_speed_mps=0.0,
        is_stress=False,
        n_frames=T,
        fps=FPS,
    )
    return traj, meta


def _hand(dz: float, which: str, k: int):
    pose = _standing_pose(H0_M)
    T = 160
    traj = _alloc(T, pose)
    if which in ("LH", "both", "asym"):
        traj[:, 1, 2] += dz
    if which in ("RH", "both"):
        traj[:, 2, 2] += dz
    if which == "asym":
        traj[:, 2, 2] -= dz
    meta = TrajectoryMeta(
        traj_id=f"H1C_{which}_{dz:+.2f}_{k:02d}",
        family="H1C_HAND_HEIGHT",
        hand_mode="world_fixed",
        source_semantics="hand_height",
        h0_m=H0_M,
        target_height_drop_m=0.0,
        transition_duration_s=0.0,
        low_hold_s=T / FPS,
        horizontal_speed_mps=0.0,
        is_stress=False,
        n_frames=T,
        fps=FPS,
    )
    return traj, meta


def _ramp(amp: float, k: int):
    pose = _standing_pose(H0_M)
    T = 200
    traj = _alloc(T, pose)
    u = min_jerk(np.linspace(0, 1, T // 3))
    traj[: T // 3, :, 2] -= u[:, None] * amp
    traj[T // 3 : 2 * T // 3, :, 2] = traj[T // 3 - 1, :, 2]
    u2 = min_jerk(np.linspace(0, 1, T - 2 * T // 3))
    traj[2 * T // 3 :, :, 2] = traj[2 * T // 3 - 1, :, 2] + u2[:, None] * amp
    meta = TrajectoryMeta(
        traj_id=f"H2_ramp_{amp:.2f}_{k:02d}",
        family="H2_RAMP",
        hand_mode="body_follow",
        source_semantics="ramp",
        h0_m=H0_M,
        target_height_drop_m=amp,
        transition_duration_s=T / 3 / FPS,
        low_hold_s=T / 3 / FPS,
        horizontal_speed_mps=0.0,
        is_stress=False,
        n_frames=T,
        fps=FPS,
    )
    return traj, meta


def _sin(amp: float, freq: float, k: int):
    T = int(5.0 * FPS)
    pose = _standing_pose(H0_M)
    traj = _alloc(T, pose)
    t = np.arange(T) / FPS
    traj[:, :, 2] += amp * np.sin(2 * np.pi * freq * t)[:, None]
    meta = TrajectoryMeta(
        traj_id=f"H2_sin_{amp:.2f}_{freq:.2f}_{k:02d}",
        family="H2_SINUSOID",
        hand_mode="body_follow",
        source_semantics="sinusoid",
        h0_m=H0_M,
        target_height_drop_m=0.0,
        transition_duration_s=5.0,
        low_hold_s=0.0,
        horizontal_speed_mps=0.0,
        is_stress=False,
        n_frames=T,
        fps=FPS,
    )
    return traj, meta


def build_height_preview(n_per: int = 2) -> dict:
    root = RESULTS / "manifests" / "height_preview"
    man = root / "manifest.json"
    if man.exists():
        return {"skipped": True, "root": str(root)}
    items = []
    for dh in (-0.05, -0.10, -0.15, 0.05):
        for k in range(n_per):
            w, m = _shift(dh, "H1A", k)
            if geometry_ok(w):
                items.append((w, m))
            w, m = _shift(dh, "H1B", k)
            if geometry_ok(w):
                items.append((w, m))
    for dz in (-0.10, 0.10):
        for which in ("LH", "RH", "both"):
            w, m = _hand(dz, which, 0)
            if geometry_ok(w):
                items.append((w, m))
    for amp in (0.10, 0.15):
        w, m = _ramp(amp, 0)
        if geometry_ok(w):
            items.append((w, m))
        w, m = _sin(amp, 0.2, 0)
        if geometry_ok(w):
            items.append((w, m))
    info = save_suite(root, items=items)
    (root / "NOTE.txt").write_text("H1B_CHEST_ONLY: slot0 is torso_link, not robot head.\n")
    return {"root": str(root), "n": info["n"]}
