"""P2 height-sweep WORLD-z suites (eval only; no knee/hip targets)."""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from flat_locomani.reference_suite import FPS, H0_M, TrajectoryMeta, _alloc, _standing_pose, min_jerk, save_suite

from .constants import RESULTS
from .manifests import geometry_ok


DELTAS = (0.10, 0.05, 0.00, -0.05, -0.10, -0.15, -0.20, -0.25)
HAND_DZ = (-0.15, -0.10, -0.05, 0.05, 0.10, 0.15)


def _static_shift(delta_h: float, mode: str, seed: int, k: int) -> tuple[np.ndarray, TrajectoryMeta]:
    rng = np.random.default_rng(seed + k)
    pose = _standing_pose(H0_M)
    pose[1, 0] += rng.uniform(0.0, 0.08)
    pose[2, 0] += rng.uniform(0.0, 0.08)
    T = 200
    traj = _alloc(T, pose)
    if mode == "H1A":
        traj[:, :, 2] += delta_h
        fam = "H1A_WHOLE_BODY"
    elif mode == "H1B":
        traj[:, 0, 2] += delta_h
        fam = "H1B_HEAD_ONLY"
    else:
        raise ValueError(mode)
    meta = TrajectoryMeta(
        traj_id=f"{mode}_dh{delta_h:+.2f}_{k:02d}",
        family=fam,
        hand_mode="world_fixed" if mode == "H1B" else "body_follow",
        source_semantics=mode,
        h0_m=H0_M,
        target_height_drop_m=max(0.0, -delta_h),
        transition_duration_s=0.0,
        low_hold_s=4.0,
        horizontal_speed_mps=0.0,
        is_stress=False,
        n_frames=T,
        fps=FPS,
    )
    return traj, meta


def _hand_height(dz: float, which: str, k: int) -> tuple[np.ndarray, TrajectoryMeta]:
    pose = _standing_pose(H0_M)
    T = 180
    traj = _alloc(T, pose)
    if which in ("LH", "both", "asym"):
        traj[:, 1, 2] += dz
    if which in ("RH", "both"):
        traj[:, 2, 2] += dz
    if which == "asym":
        traj[:, 2, 2] -= dz
    meta = TrajectoryMeta(
        traj_id=f"H1C_{which}_dz{dz:+.2f}_{k:02d}",
        family="H1C_HAND_HEIGHT",
        hand_mode="world_fixed",
        source_semantics="hand_height",
        h0_m=H0_M,
        target_height_drop_m=0.0,
        transition_duration_s=0.0,
        low_hold_s=3.6,
        horizontal_speed_mps=0.0,
        is_stress=False,
        n_frames=T,
        fps=FPS,
    )
    return traj, meta


def _ramp(amp: float, speed: float, k: int) -> tuple[np.ndarray, TrajectoryMeta]:
    # stand -> lower -> hold -> rise -> hold
    v = max(speed, 1e-6)
    n_move = int(round(amp / v * FPS))
    n_hold = int(1.0 * FPS)
    n_stand = int(0.6 * FPS)
    pose = _standing_pose(H0_M)
    parts = []
    parts.append(_alloc(n_stand, pose))
    u = min_jerk(np.linspace(0, 1, n_move))
    down = _alloc(n_move, pose)
    down[:, :, 2] -= amp * u[:, None]
    parts.append(down)
    hold = down[-1:].repeat(n_hold, axis=0)
    parts.append(hold)
    up = _alloc(n_move, pose)
    up[:, :, 2] = hold[-1, :, 2] + amp * u[:, None]
    parts.append(up)
    parts.append(up[-1:].repeat(n_hold, axis=0))
    traj = np.concatenate(parts, axis=0)
    meta = TrajectoryMeta(
        traj_id=f"H2_ramp_A{amp:.2f}_v{speed:.2f}_{k:02d}",
        family="H2_RAMP",
        hand_mode="body_follow",
        source_semantics="ramp",
        h0_m=H0_M,
        target_height_drop_m=amp,
        transition_duration_s=float(n_move / FPS),
        low_hold_s=1.0,
        horizontal_speed_mps=0.0,
        is_stress=False,
        n_frames=int(traj.shape[0]),
        fps=FPS,
    )
    return traj, meta


def _sinusoid(amp: float, freq: float, k: int) -> tuple[np.ndarray, TrajectoryMeta]:
    T = int(5.0 * FPS)
    pose = _standing_pose(H0_M)
    traj = _alloc(T, pose)
    t = np.arange(T) / FPS
    traj[:, :, 2] += amp * np.sin(2 * np.pi * freq * t)[:, None]
    meta = TrajectoryMeta(
        traj_id=f"H2_sin_A{amp:.2f}_f{freq:.2f}_{k:02d}",
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


def build_height_suite(split: str, n_per: int) -> dict:
    root = RESULTS / "manifests" / f"height_{split.lower()}"
    man = root / "manifest.json"
    if man.exists():
        return {"skipped": True, "root": str(root), "n": json.loads(man.read_text()).get("n")}
    items = []
    seed = 7301 if split == "DEV" else 9301
    for dh in DELTAS:
        for k in range(n_per):
            items.append(_static_shift(dh, "H1A", seed, k + 10 * int(abs(dh) * 100)))
            items.append(_static_shift(dh, "H1B", seed + 1, k + 10 * int(abs(dh) * 100)))
    for dz in HAND_DZ:
        for which in ("LH", "RH", "both", "asym"):
            for k in range(max(1, n_per // 2)):
                items.append(_hand_height(dz, which, k))
    for amp in (0.10, 0.20):
        for spd in (0.05, 0.10, 0.15):
            for k in range(max(1, n_per // 2)):
                items.append(_ramp(amp, spd, k))
        for f in (0.10, 0.20, 0.30):
            for k in range(max(1, n_per // 2)):
                items.append(_sinusoid(amp, f, k))
    kept = [(w, m) for w, m in items if geometry_ok(w)]
    info = save_suite(root, items=kept)
    return {"root": str(root), "n_in": len(items), "n_out": len(kept), **info}


if __name__ == "__main__":
    print(build_height_suite("DEV", 10))
    print(build_height_suite("TEST", 20))
