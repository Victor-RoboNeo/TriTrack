from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, quat_conjugate, quat_error_magnitude, quat_mul

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def motion_global_anchor_rpy_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, axis: str = "pitch"
) -> torch.Tensor:
    """Exp-kernel on a single Euler axis of the torso/anchor orientation error.

    Rough-terrain loophole is forward lean (pitch) / roll, not yaw. Split the
    old quat-magnitude term so roll/pitch can be up-weighted independently.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    q_err = quat_mul(quat_conjugate(command.anchor_quat_w), command.robot_anchor_quat_w)
    roll, pitch, yaw = euler_xyz_from_quat(q_err)
    axes = {"roll": _wrap_pi(roll), "pitch": _wrap_pi(pitch), "yaw": _wrap_pi(yaw)}
    if axis not in axes:
        raise ValueError(f"axis must be roll|pitch|yaw, got {axis!r}")
    error = axes[axis] ** 2
    return torch.exp(-error / std**2)


def diag_torso_pitch_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Logging-only: |pitch| of robot vs command torso (radians). Use weight=0."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    q_err = quat_mul(quat_conjugate(command.anchor_quat_w), command.robot_anchor_quat_w)
    _, pitch, _ = euler_xyz_from_quat(q_err)
    return _wrap_pi(pitch).abs()


def diag_pelvis_height_terrain_rel(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Logging-only: robot root z minus ankle-mean z (terrain-relative pelvis height)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    names = list(command.cfg.body_names)
    idxs = [names.index(n) for n in _ANKLE_BODY_NAMES if n in names]
    root_z = command.robot_anchor_pos_w[:, 2]
    if not idxs:
        return root_z
    ankle_z = command.robot_body_pos_w[:, idxs, 2].mean(dim=-1)
    return root_z - ankle_z


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_visible_kp_position_error_exp_world(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """D5 reward: world-frame position accuracy of the VISIBLE points of interest.

    The deployment-true objective for masked-KP latent-RL finetuning
    (see ``docs/latent_rl_finetune_plan.md`` D5/D5a/D5b/D5c). Generalizes
    :func:`teleop_vr_3point` (world-frame keypoint exp-kernel) by selecting the
    bodies dynamically from the command's per-env per-step KP visibility mask
    instead of a static name list:

    - World frame, NOT anchor frame: uses ``command.body_pos_w`` (reference,
      from the clip + env origin) vs ``command.robot_body_pos_w`` (actual). The
      anchor-frame term (:func:`motion_relative_body_position_error_exp`) is the
      one this replaces — anchor drift can hide large world-frame error and the
      drag demo cares about world placement.
    - Points of interest = the VISIBLE (unmasked) KP set, per env, per step
      (decision D5b). ``PartialMaskedMultiMotionCommand._env_body_mask`` is
      ``(num_envs, num_bodies)`` float, ``1.0 = visible, 0.0 = masked``, indexed
      by the command's ``body_names`` — exactly the body axis of ``body_pos_w``.
    - exp-kernel, mean over the visible set (decision D5c). Envs with no visible
      body (degenerate) get a neutral 0 error (reward 1) via the denom clamp;
      they contribute no learning signal for this term.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    diff = command.body_pos_w - command.robot_body_pos_w  # [N, B, 3] world frame
    per_body_sq_err = torch.sum(torch.square(diff), dim=-1)  # [N, B]

    vis = getattr(command, "_env_body_mask", None)  # [N, B], 1=visible
    if vis is None:
        vis = torch.ones_like(per_body_sq_err)
    vis = vis.to(per_body_sq_err.dtype)
    denom = vis.sum(dim=-1).clamp_min(1.0)  # avoid div-by-zero on all-masked envs
    err = (per_body_sq_err * vis).sum(dim=-1) / denom  # [N], mean over visible
    return torch.exp(-err / std**2)


_ANKLE_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _ankle_mean_z_offset(command: MotionCommand) -> torch.Tensor:
    """Robot ankle-mean z minus reference ankle-mean z.

    Flat-recorded clips sit at z≈0 while the robot on stairs/slopes is higher, so a raw
    world-z POI error punishes terrain following. Subtracting this offset scores *posture*
    height (torso/wrist vs feet) instead of absolute world height. Ankles stay in the KP5
    body set even when the VR 3-point mask hides them.
    """
    names = list(command.cfg.body_names)
    idxs = [names.index(n) for n in _ANKLE_BODY_NAMES if n in names]
    n_env = command.body_pos_w.shape[0]
    if not idxs:
        return command.body_pos_w.new_zeros(n_env)
    robot_z = command.robot_body_pos_w[:, idxs, 2].mean(dim=-1)
    ref_z = command.body_pos_w[:, idxs, 2].mean(dim=-1)
    return robot_z - ref_z


def _visible_mean_sq(per_body: torch.Tensor, command: MotionCommand) -> torch.Tensor:
    vis = getattr(command, "_env_body_mask", None)
    if vis is None:
        vis = torch.ones_like(per_body)
    vis = vis.to(per_body.dtype)
    denom = vis.sum(dim=-1).clamp_min(1.0)
    return (per_body * vis).sum(dim=-1) / denom


def motion_visible_kp_xy_error_exp_world(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """World-frame XY accuracy of visible keypoints (plan-view loco / reach)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    diff_xy = command.body_pos_w[..., :2] - command.robot_body_pos_w[..., :2]
    per_body = torch.sum(torch.square(diff_xy), dim=-1)
    err = _visible_mean_sq(per_body, command)
    return torch.exp(-err / std**2)


def motion_visible_kp_z_error_exp_world_terrain_rel(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Terrain-relative Z accuracy of visible keypoints.

    ``diff_z`` is shifted by the ankle-mean offset so squat/stoop still trains height
    change, while stairs do not look like a constant z failure vs a flat clip.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    offset = _ankle_mean_z_offset(command).unsqueeze(-1)
    diff_z = (command.body_pos_w[..., 2] - command.robot_body_pos_w[..., 2]) + offset
    err = _visible_mean_sq(torch.square(diff_z), command)
    return torch.exp(-err / std**2)


def motion_global_anchor_position_error_exp_terrain_rel(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Global anchor pos with the same ankle-mean z offset as the terrain-rel POI z term."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    diff = command.anchor_pos_w - command.robot_anchor_pos_w
    offset = _ankle_mean_z_offset(command)
    adj = torch.stack((diff[:, 0], diff[:, 1], diff[:, 2] + offset), dim=-1)
    error = torch.sum(torch.square(adj), dim=-1)
    return torch.exp(-error / std**2)


def motion_torso_position_error_l2(
    env: ManagerBasedRLEnv, command_name: str, body_name: str = "torso_link"
) -> torch.Tensor:
    """Squared world-frame torso position error (terrain-relative z).

    Unbounded L2 so large forward lean / drift keeps growing, unlike the saturating
    exp kernels. Pair with a *small* negative RewTerm weight (e.g. -1.5): 10 cm →
    0.015, 30 cm → 0.135, 50 cm → 0.375. Z is shifted by the ankle-mean offset so
    standing on a bump is not a fake height error.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    names = list(command.cfg.body_names)
    if body_name not in names:
        return command.body_pos_w.new_zeros(command.body_pos_w.shape[0])
    idx = names.index(body_name)
    diff = command.body_pos_w[:, idx] - command.robot_body_pos_w[:, idx]
    offset = _ankle_mean_z_offset(command)
    adj = torch.stack((diff[:, 0], diff[:, 1], diff[:, 2] + offset), dim=-1)
    return torch.sum(torch.square(adj), dim=-1)


def motion_visible_kp_lin_vel_error_exp_world(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """D5 companion: world-frame LINEAR-VELOCITY accuracy of the VISIBLE points of interest.

    Mask-aware twin of :func:`motion_global_body_linear_velocity_error_exp`, with
    identical visibility semantics to :func:`motion_visible_kp_position_error_exp_world`
    (``command._env_body_mask``: ``1.0=visible``). Pairs with the POI position term so
    the policy tracks not just where the visible points are but how fast they move —
    important for responsive, non-laggy drag tracking.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    diff = command.body_lin_vel_w - command.robot_body_lin_vel_w  # [N, B, 3] world frame
    per_body_sq_err = torch.sum(torch.square(diff), dim=-1)  # [N, B]

    vis = getattr(command, "_env_body_mask", None)  # [N, B], 1=visible
    if vis is None:
        vis = torch.ones_like(per_body_sq_err)
    vis = vis.to(per_body_sq_err.dtype)
    denom = vis.sum(dim=-1).clamp_min(1.0)
    err = (per_body_sq_err * vis).sum(dim=-1) / denom  # [N], mean over visible
    return torch.exp(-err / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

# ===== MOSAIC Expert Teleop-style Rewards (World Frame, Fine-grained) =====

def teleop_body_position_extend(
    env: ManagerBasedRLEnv,
    command_name: str,
    upper_body_std: float = 0.05,
    lower_body_std: float = 0.05,
    upper_weight: float = 1.0,
    lower_weight: float = 1.0,
) -> torch.Tensor:
    """
    Upper/lower body position tracking with separate weights (MOSAIC style).
    Tracks body positions in world frame with fine-grained upper/lower body separation.

    Args:
        env: The environment.
        command_name: Name of the motion command.
        upper_body_std: Std (in meters) for upper body exponential reward.
        lower_body_std: Std (in meters) for lower body exponential reward.
        upper_weight: Weight for upper body reward.
        lower_weight: Weight for lower body reward.

    Returns:
        Combined upper + lower body position reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    # Upper body names
    upper_body_names = [
        "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
        "left_elbow_link", "right_shoulder_pitch_link", "right_shoulder_roll_link",
        "right_shoulder_yaw_link", "right_elbow_link", "left_hand_link",
        "right_hand_link", "head_link"
    ]

    # Lower body names
    lower_body_names = [
        "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
        "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
        "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
        "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
        "waist_yaw_link", "waist_roll_link", "torso_link"
    ]

    upper_idx = _get_body_indexes(command, upper_body_names)
    lower_idx = _get_body_indexes(command, lower_body_names)

    # Compute errors (world frame)
    if len(upper_idx) > 0:
        upper_diff = command.body_pos_w[:, upper_idx, :] - command.robot_body_pos_w[:, upper_idx, :]
        upper_error = (upper_diff ** 2).mean(dim=-1).mean(dim=-1)  # [N]
        r_upper = torch.exp(-upper_error / (upper_body_std ** 2))
    else:
        r_upper = torch.zeros(env.num_envs, device=env.device)

    if len(lower_idx) > 0:
        lower_diff = command.body_pos_w[:, lower_idx, :] - command.robot_body_pos_w[:, lower_idx, :]
        lower_error = (lower_diff ** 2).mean(dim=-1).mean(dim=-1)
        r_lower = torch.exp(-lower_error / (lower_body_std ** 2))
    else:
        r_lower = torch.zeros(env.num_envs, device=env.device)

    return r_lower * lower_weight + r_upper * upper_weight


def teleop_vr_3point(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.04,
) -> torch.Tensor:
    """
    VR 3-point tracking (head + hands) in world frame.

    Args:
        env: The environment.
        command_name: Name of the motion command.
        std: Std (in meters) for exponential reward.

    Returns:
        VR 3-point position tracking reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    keypoint_names = ["head_link", "left_hand_link", "right_hand_link"]
    keypoint_idx = _get_body_indexes(command, keypoint_names)

    if len(keypoint_idx) > 0:
        diff = command.body_pos_w[:, keypoint_idx, :] - command.robot_body_pos_w[:, keypoint_idx, :]
        error = (diff ** 2).mean(dim=-1).mean(dim=-1)
        return torch.exp(-error / (std ** 2))
    else:
        return torch.zeros(env.num_envs, device=env.device)


def teleop_body_position_feet(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.03,
) -> torch.Tensor:
    """
    Feet position tracking in world frame (high precision).

    Args:
        env: The environment.
        command_name: Name of the motion command.
        std: Std (in meters) for exponential reward.

    Returns:
        Feet position tracking reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    feet_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    feet_idx = _get_body_indexes(command, feet_names)

    if len(feet_idx) > 0:
        diff = command.body_pos_w[:, feet_idx, :] - command.robot_body_pos_w[:, feet_idx, :]
        error = (diff ** 2).mean(dim=-1).mean(dim=-1)
        return torch.exp(-error / (std ** 2))
    else:
        return torch.zeros(env.num_envs, device=env.device)


def teleop_body_rotation_extend(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.2,
) -> torch.Tensor:
    """
    Full body rotation tracking in world frame.

    Args:
        env: The environment.
        command_name: Name of the motion command.
        std: Std (in radians) for exponential reward.

    Returns:
        Body rotation tracking reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Use all bodies
    rotation_error = quat_error_magnitude(command.body_quat_w, command.robot_body_quat_w)  # [N, num_bodies]
    error = (rotation_error ** 2).mean(dim=-1)  # [N]
    return torch.exp(-error / (std ** 2))


def teleop_body_velocity_extend(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.5,
) -> torch.Tensor:
    """
    Full body linear velocity tracking in world frame.

    Args:
        env: The environment.
        command_name: Name of the motion command.
        std: Std (in m/s) for exponential reward.

    Returns:
        Body linear velocity tracking reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    diff = command.body_lin_vel_w - command.robot_body_lin_vel_w
    error = (diff ** 2).mean(dim=-1).mean(dim=-1)
    return torch.exp(-error / (std ** 2))


def teleop_body_ang_velocity_extend(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 2.0,
) -> torch.Tensor:
    """
    Full body angular velocity tracking in world frame.

    Args:
        env: The environment.
        command_name: Name of the motion command.
        std: Std (in rad/s) for exponential reward.

    Returns:
        Body angular velocity tracking reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    diff = command.body_ang_vel_w - command.robot_body_ang_vel_w
    error = (diff ** 2).mean(dim=-1).mean(dim=-1)
    return torch.exp(-error / (std ** 2))


def motion_anchor_linear_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 1.0,
) -> torch.Tensor:
    """
    Anchor (base) linear velocity tracking in world frame.
    Tracks the robot anchor/pelvis linear velocity.

    Args:
        env: The environment.
        command_name: Name of the motion command.
        std: Std for exponential reward.

    Returns:
        Anchor linear velocity tracking reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(
        torch.square(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w),
        dim=-1
    )
    return torch.exp(-error / std**2)

def contact_feet(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.05,
) -> torch.Tensor:
    """
    Reward function that checks if the robot's foot contact state matches the reference trajectory.

    A foot is considered "in contact" if its height (Z-coordinate) is less than the threshold.
    Scoring: +0.5 per foot if the state (contact or swing) matches the reference, 
    resulting in a max reward of 1.0 for both feet matching.

    Args:
        env: The environment instance.
        command_name: Name of the motion command term.
        threshold: Height threshold (in meters) to determine contact. Defaults to 0.05.

    Returns:
        Contact consistency reward [num_envs].
    """
    # 1. Access the motion command term
    command: MotionCommand = env.command_manager.get_term(command_name)
    feet_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    feet_idx = _get_body_indexes(command, feet_names)

    if len(feet_idx) > 0:
        # 2. Extract Z-coordinates for both robot and reference trajectory
        # Shape: [num_envs, 2]
        current_feet_z = command.robot_body_pos_w[:, feet_idx, 2]
        reference_feet_z = command.body_pos_w[:, feet_idx, 2]

        # 3. Determine contact state (True if height < threshold)
        # Shape: [num_envs, 2] (bool)
        current_contact = current_feet_z < threshold
        reference_contact = reference_feet_z < threshold

        # 4. Compare current state with reference state
        # A match occurs if both are in contact OR both are in swing
        # Shape: [num_envs, 2] (float: 1.0 for match, 0.0 for mismatch)
        matching_states = (current_contact == reference_contact).float()

        # 5. Calculate final reward
        # Sum across the two feet and multiply by 0.5 (max reward 1.0)
        return matching_states.sum(dim=-1) * 0.5
    else:
        return torch.zeros(env.num_envs, device=env.device)
    
def teleop_body_position_feet_z(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.03,
) -> torch.Tensor:
    """
    Feet position_z tracking in world frame (high precision).

    Args:
        env: The environment.
        command_name: Name of the motion command.
        std: Std (in meters) for exponential reward.

    Returns:
        Feet position tracking reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    feet_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    feet_idx = _get_body_indexes(command, feet_names)

    if len(feet_idx) > 0:
        diff = command.body_pos_w[:, feet_idx, 2] - command.robot_body_pos_w[:, feet_idx, 2]
        error = (diff ** 2).mean(dim=-1)
        return torch.exp(-error / (std ** 2))
    else:
        return torch.zeros(env.num_envs, device=env.device)