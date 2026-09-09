"""Coordinate-frame helpers matching HTD reward semantics.

Verified from ``legged_lab/envs/base/base_env.py`` reward functions, not from
the ``UniformVelocityCommand`` class comment (which claims "base frame").

Convention (wxyz quaternions, Isaac / Isaac Lab style):
- lin_vel_xy command is tracked against yaw-aligned world velocity of the
  *root/pelvis*, not ``root_lin_vel_b``.
- ang_vel_z command is tracked against *world* yaw rate ``root_ang_vel_w[:, 2]``.
- height command is *torso_link* world z.
- roll/pitch/yaw command is torso orientation relative to the pelvis yaw frame.

Relative-yaw frame used here is the same "virtual-anchor yaw" idea: keep
global heading in the yaw of the pelvis, express torso RPY in that frame.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def wrap_to_pi(a: np.ndarray | float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, _EPS, None)


def yaw_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Extract yaw-only quaternion from a wxyz orientation (Isaac Lab ``yaw_quat``)."""
    q = quat_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    out = np.zeros_like(q)
    out[..., 0] = np.cos(half)
    out[..., 3] = np.sin(half)
    return quat_normalize(out)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, wxyz, batched on leading dims."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (wxyz). Matches Isaac Lab ``quat_apply``."""
    q = quat_normalize(q)
    v = np.asarray(v, dtype=np.float64)
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_apply(quat_conjugate(q), v)


def quat_to_rpy_wxyz(q: np.ndarray) -> np.ndarray:
    """XYZ (roll, pitch, yaw) from wxyz. Matches typical robotics / Isaac usage."""
    q = quat_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1.0, np.sign(sinp) * (np.pi / 2.0), np.arcsin(np.clip(sinp, -1.0, 1.0)))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=-1)


def projected_gravity_b(root_quat_w: np.ndarray, gravity_w: np.ndarray | None = None) -> np.ndarray:
    if gravity_w is None:
        gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    g = np.broadcast_to(np.asarray(gravity_w, dtype=np.float64), root_quat_w[..., :3].shape)
    return quat_apply_inverse(root_quat_w, g)


def yaw_frame_lin_vel_xy(root_quat_w: np.ndarray, root_lin_vel_w: np.ndarray) -> np.ndarray:
    """HTD ``track_lin_vel_xy_yaw_frame_exp`` measurement."""
    return quat_apply_inverse(yaw_quat_wxyz(root_quat_w), root_lin_vel_w)[..., :2]


def world_yaw_rate(root_ang_vel_w: np.ndarray) -> np.ndarray:
    """HTD ``track_ang_vel_z_world_exp`` measurement."""
    return np.asarray(root_ang_vel_w, dtype=np.float64)[..., 2]


def torso_in_pelvis_yaw_rpy(torso_quat_w: np.ndarray, pelvis_quat_w: np.ndarray) -> np.ndarray:
    """Torso RPY relative to pelvis yaw frame. Matches HTD ``_get_body_orientation``."""
    rel = quat_mul(quat_conjugate(yaw_quat_wxyz(pelvis_quat_w)), torso_quat_w)
    rpy = quat_to_rpy_wxyz(rel)
    rpy[..., 2] = wrap_to_pi(rpy[..., 2])
    return rpy


def finite_difference(x: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        raise ValueError("need at least 2 frames for finite difference")
    dx = np.empty_like(x)
    dx[0] = 0.0
    dx[1:] = (x[1:] - x[:-1]) / dt
    return dx
