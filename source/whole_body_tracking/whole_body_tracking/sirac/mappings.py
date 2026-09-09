"""Verified G1 joint / command mappings for SIRAC Phase 1.

Sources (do not invent extra joints):
- HTD IsaacLab-Decoupled-WBC ``legged_lab/envs/base/base_env.py`` ``target_joint_order``
- HTD ``deploy/configs/g1_student_htd.yaml``
- AnyBody ``robots/robot_registry.py`` G1 platform
- AnyBody URDF ``assets/unitree_description/urdf/g1/main.urdf`` revolute joints

All four agree on the 29-DoF name order below. Tests in ``tests/sirac`` freeze this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# HTD ``target_joint_order`` and AnyBody ``ROBOT_PLATFORMS["g1"].joint_names``.
G1_JOINT_NAMES_29: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

LOWER_WAIST_NAMES: tuple[str, ...] = G1_JOINT_NAMES_29[:15]
ARM_NAMES: tuple[str, ...] = G1_JOINT_NAMES_29[15:]
WAIST_PITCH_INDEX = 14  # in the 15-D HTD action vector

# Unitree SDK2 motor indices for G1 29-DoF (matches HTD joint2motor_idx / arm_joint2motor_idx).
LOWER_MOTOR_IDX: tuple[int, ...] = tuple(range(15))
ARM_MOTOR_IDX: tuple[int, ...] = tuple(range(15, 29))

# HTD deploy default lower/waist pose (rad). Not AnyBody's Beyond-Mimic init pose.
HTD_DEFAULT_LOWER = np.array(
    [-0.20, 0.0, 0.0, 0.42, -0.23, 0.0, -0.20, 0.0, 0.0, 0.42, -0.23, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
HTD_DEFAULT_ARM = np.zeros(14, dtype=np.float64)

# AnyBody G1_CYLINDER_CFG init (regex-expanded). Used only to document mismatch.
ANYBODY_DEFAULT_LOWER = np.array(
    [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0, -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)

HTD_ACTION_SCALE = 0.25
HTD_NUM_ACTIONS = 15
HTD_NUM_OBS = 58
HTD_HISTORY = 2
HTD_CONTROL_DT = 0.02
HTD_SIM_DT = 0.005
HTD_DECIMATION = 4

# Command vector order from HTD deploy + teacher obs [6:13].
COMMAND_NAMES: tuple[str, ...] = (
    "lin_vel_x",
    "lin_vel_y",
    "ang_vel_z",
    "height",
    "body_roll",
    "body_pitch",
    "body_yaw",
)

# Deploy clip ranges (g1_student_htd.yaml). Training ranges in g1_config.py are slightly wider.
HTD_COMMAND_RANGE = {
    "lin_vel_x": (-0.55, 0.55),
    "lin_vel_y": (-0.55, 0.55),
    "ang_vel_z": (-1.57, 1.57),
    "height": (0.35, 0.8),
    "body_roll": (-0.5, 0.5),
    "body_pitch": (-0.52, 1.22),
    "body_yaw": (-1.27, 1.27),
}
HTD_TRAIN_COMMAND_RANGE = {
    "lin_vel_x": (-0.5, 0.5),
    "lin_vel_y": (-0.5, 0.5),
    "ang_vel_z": (-1.57, 1.57),
    "height": (0.35, 0.8),
    "body_roll": (-0.7, 0.7),
    "body_pitch": (-0.52, 1.57),
    "body_yaw": (-1.57, 1.57),
}
WAIST_PITCH_LIMITS = (-0.60, 0.60)
DEFAULT_HEIGHT = 0.72  # HTD keyboard reset; deploy Y/A buttons around this

# Teacher 60-D / student 58-D observation slices.
OBS_ANG_VEL = slice(0, 3)
OBS_GRAVITY = slice(3, 6)
OBS_COMMAND = slice(6, 13)
OBS_JOINT_POS = slice(13, 28)
OBS_JOINT_VEL = slice(28, 43)
OBS_LAST_ACTION = slice(43, 58)
OBS_FEET_CONTACT = slice(58, 60)  # teacher only


def name_index(name: str) -> int:
    return G1_JOINT_NAMES_29.index(name)


def lower_indices_in(full_names: list[str] | tuple[str, ...]) -> np.ndarray:
    """Map HTD 15-D order into an arbitrary Isaac/URDF joint-name list."""
    lookup = {n: i for i, n in enumerate(full_names)}
    missing = [n for n in LOWER_WAIST_NAMES if n not in lookup]
    if missing:
        raise KeyError(f"missing lower/waist joints: {missing}")
    return np.array([lookup[n] for n in LOWER_WAIST_NAMES], dtype=np.int64)


def arm_indices_in(full_names: list[str] | tuple[str, ...]) -> np.ndarray:
    lookup = {n: i for i, n in enumerate(full_names)}
    missing = [n for n in ARM_NAMES if n not in lookup]
    if missing:
        raise KeyError(f"missing arm joints: {missing}")
    return np.array([lookup[n] for n in ARM_NAMES], dtype=np.int64)


def clip_command(c: np.ndarray, *, deploy: bool = True) -> np.ndarray:
    rng = HTD_COMMAND_RANGE if deploy else HTD_TRAIN_COMMAND_RANGE
    out = np.asarray(c, dtype=np.float64).copy()
    if out.shape[-1] != 7:
        raise ValueError(f"command last dim must be 7, got {out.shape}")
    lo = np.array([rng[n][0] for n in COMMAND_NAMES], dtype=np.float64)
    hi = np.array([rng[n][1] for n in COMMAND_NAMES], dtype=np.float64)
    return np.clip(out, lo, hi)


@dataclass(frozen=True)
class HtdObsLayout:
    num_obs: int = HTD_NUM_OBS
    history: int = HTD_HISTORY
    num_actions: int = HTD_NUM_ACTIONS
    control_dt: float = HTD_CONTROL_DT
    action_scale: float = HTD_ACTION_SCALE
