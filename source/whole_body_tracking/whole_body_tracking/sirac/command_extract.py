"""Analytic 7-D realization-command extractor.

The command is a *robot-owned* internal variable. It is never a human
interface. Human-owned intent remains head / left-hand / right-hand SE(3).

Extracts HTD student command order:

    [lin_vel_x, lin_vel_y, ang_vel_z, height, body_roll, body_pitch, body_yaw]

from a Stage-2 / Stage2Twin nominal rollout using the *reward* frames
documented in CONTROLLER_AUDIT.md, not the UniformVelocityCommand docstring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import (
    finite_difference,
    torso_in_pelvis_yaw_rpy,
    world_yaw_rate,
    yaw_frame_lin_vel_xy,
    yaw_quat_wxyz,
)
from .mappings import clip_command


@dataclass
class RealizationState:
    """One timestep of quantities needed to form c_t.

    Quaternions are wxyz. Positions/velocities are world-frame unless noted.
    """

    pelvis_pos_w: np.ndarray
    pelvis_quat_w: np.ndarray
    pelvis_lin_vel_w: np.ndarray
    pelvis_ang_vel_w: np.ndarray
    torso_pos_w: np.ndarray
    torso_quat_w: np.ndarray


def extract_realization_command(
    state: RealizationState | None = None,
    *,
    pelvis_pos_w: np.ndarray | None = None,
    pelvis_quat_w: np.ndarray | None = None,
    pelvis_lin_vel_w: np.ndarray | None = None,
    pelvis_ang_vel_w: np.ndarray | None = None,
    torso_pos_w: np.ndarray | None = None,
    torso_quat_w: np.ndarray | None = None,
    clip: bool = True,
    deploy_clip: bool = True,
) -> np.ndarray:
    """Map current (or nominal Stage-2) kinematics to HTD 7-D command.

    If linear/angular velocities are omitted, they must be supplied by the
    caller via finite differences (see ``extract_from_rollout``).
    """
    if state is not None:
        pelvis_quat_w = state.pelvis_quat_w
        pelvis_lin_vel_w = state.pelvis_lin_vel_w
        pelvis_ang_vel_w = state.pelvis_ang_vel_w
        torso_pos_w = state.torso_pos_w
        torso_quat_w = state.torso_quat_w
    assert pelvis_quat_w is not None
    assert pelvis_lin_vel_w is not None
    assert pelvis_ang_vel_w is not None
    assert torso_pos_w is not None
    assert torso_quat_w is not None

    vxy = yaw_frame_lin_vel_xy(pelvis_quat_w, pelvis_lin_vel_w)
    wz = world_yaw_rate(pelvis_ang_vel_w)
    height = np.asarray(torso_pos_w, dtype=np.float64)[..., 2]
    rpy = torso_in_pelvis_yaw_rpy(torso_quat_w, pelvis_quat_w)
    c = np.concatenate(
        [
            np.asarray(vxy, dtype=np.float64),
            np.asarray(wz, dtype=np.float64)[..., None],
            np.asarray(height, dtype=np.float64)[..., None],
            rpy,
        ],
        axis=-1,
    )
    if clip:
        c = clip_command(c, deploy=deploy_clip)
    return c


def extract_from_rollout(
    pelvis_pos_w: np.ndarray,
    pelvis_quat_w: np.ndarray,
    torso_pos_w: np.ndarray,
    torso_quat_w: np.ndarray,
    dt: float,
    *,
    pelvis_lin_vel_w: np.ndarray | None = None,
    pelvis_ang_vel_w: np.ndarray | None = None,
    clip: bool = True,
    deploy_clip: bool = True,
) -> np.ndarray:
    """Extract a (T, 7) command sequence from a Stage-2 kinematic rollout.

    Velocities default to finite differences of pelvis translation and of
    pelvis yaw (world z rate). Prefer simulator velocities when available.
    """
    pelvis_pos_w = np.asarray(pelvis_pos_w, dtype=np.float64)
    pelvis_quat_w = np.asarray(pelvis_quat_w, dtype=np.float64)
    torso_pos_w = np.asarray(torso_pos_w, dtype=np.float64)
    torso_quat_w = np.asarray(torso_quat_w, dtype=np.float64)
    t = pelvis_pos_w.shape[0]
    if pelvis_lin_vel_w is None:
        pelvis_lin_vel_w = finite_difference(pelvis_pos_w, dt)
    if pelvis_ang_vel_w is None:
        yaw = np.arctan2(
            2.0 * (pelvis_quat_w[:, 0] * pelvis_quat_w[:, 3] + pelvis_quat_w[:, 1] * pelvis_quat_w[:, 2]),
            1.0 - 2.0 * (pelvis_quat_w[:, 2] ** 2 + pelvis_quat_w[:, 3] ** 2),
        )
        wz = finite_difference(np.unwrap(yaw), dt)
        pelvis_ang_vel_w = np.zeros((t, 3), dtype=np.float64)
        pelvis_ang_vel_w[:, 2] = wz
    return extract_realization_command(
        pelvis_quat_w=pelvis_quat_w,
        pelvis_lin_vel_w=np.asarray(pelvis_lin_vel_w, dtype=np.float64),
        pelvis_ang_vel_w=np.asarray(pelvis_ang_vel_w, dtype=np.float64),
        torso_pos_w=torso_pos_w,
        torso_quat_w=torso_quat_w,
        clip=clip,
        deploy_clip=deploy_clip,
    )


def world_to_anchor_yaw_pose(
    pos_w: np.ndarray,
    quat_w: np.ndarray,
    pelvis_quat_w: np.ndarray,
    pelvis_pos_w: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a body pose in the pelvis yaw (virtual-anchor heading) frame.

    Used by the Phase-1B sparse-intent encoder. Does not enter the human UI.
    Positions are relative to the pelvis origin when ``pelvis_pos_w`` is given.
    """
    from .frames import quat_apply_inverse, quat_conjugate, quat_mul

    yaw_q = yaw_quat_wxyz(pelvis_quat_w)
    delta = np.asarray(pos_w, dtype=np.float64)
    if pelvis_pos_w is not None:
        delta = delta - np.asarray(pelvis_pos_w, dtype=np.float64)
    pos_yaw = quat_apply_inverse(yaw_q, delta)
    quat_yaw = quat_mul(quat_conjugate(yaw_q), quat_w)
    return pos_yaw, quat_yaw
