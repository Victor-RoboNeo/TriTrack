from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)

def bad_visible_kp_pos_world(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    grace_steps: int = 60,
) -> torch.Tensor:
    """Terminate when the VISIBLE keypoints' world-frame RMS position error exceeds
    ``threshold`` (metres), after a ``grace_steps`` lift-in window.

    Mask-aware: averages squared error over the per-env per-step VISIBLE body set
    (``command._env_body_mask``, ``1.0=visible``) — the SAME set the world-frame POI reward
    (:func:`motion_visible_kp_position_error_exp_world`) scores. So the termination fires on
    exactly the quantity being optimized.

    For the wrist-writing task this truncates episodes where the wrist has diverged
    hopelessly from the moving target (e.g. the policy can't keep up on a wide word), so
    rollout time isn't wasted finishing a doomed clip. The ``grace_steps`` window skips the
    initial lift-in from the standing wrist pose to the first letter point (~0.4-0.6 m), which
    would otherwise trip the termination immediately at episode start. ``command.time_steps``
    is the per-env frame index since reset (start_from_beginning resets it to 0), so the grace
    is measured from episode start.

    Returns a bool tensor ``[N]``. Use as a FAILURE termination (``time_out=False``, the
    DoneTerm default) so the value bootstrap is cut for the doomed state.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    diff = command.body_pos_w - command.robot_body_pos_w  # [N, B, 3] world frame
    per_body_sq = torch.sum(torch.square(diff), dim=-1)  # [N, B]

    vis = getattr(command, "_env_body_mask", None)  # [N, B], 1=visible
    if vis is None:
        vis = torch.ones_like(per_body_sq)
    vis = vis.to(per_body_sq.dtype)
    denom = vis.sum(dim=-1).clamp_min(1.0)
    mean_sq = (per_body_sq * vis).sum(dim=-1) / denom  # [N] mean sq err over visible
    rms = torch.sqrt(mean_sq.clamp_min(0.0))  # [N] metres

    past_grace = command.time_steps > int(grace_steps)
    return past_grace & (rms > float(threshold))


def motion_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not hasattr(command, "motion_end_buf"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return command.motion_end_buf


def fall_to_ground(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height: float,
) -> torch.Tensor:
    """True when any listed body is below ``min_height`` in world frame (collapsed / fallen).

    Use torso or pelvis links—not feet—or a low threshold will fire while standing.
    Intended for robustness evaluation and optional use as a termination term.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    needs_resolve = body_ids is None or isinstance(body_ids, slice)
    if not needs_resolve and isinstance(body_ids, torch.Tensor) and body_ids.numel() == 0:
        needs_resolve = True
    if needs_resolve:
        names = getattr(asset_cfg, "body_names", None)
        if not names:
            raise ValueError("fall_to_ground requires body_names on asset_cfg (or resolved body_ids).")
        if isinstance(names, str):
            names = [names]
        body_ids, _ = asset.find_bodies(names, preserve_order=True)
    z = asset.data.body_pos_w[:, body_ids, 2]
    if z.dim() == 1:
        return z < min_height
    return torch.any(z < min_height, dim=-1)
