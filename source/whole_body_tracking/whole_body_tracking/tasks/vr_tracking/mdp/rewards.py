from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from whole_body_tracking.tasks.tracking.mdp.rewards import *  # noqa: F401, F403


def _decoded_action_pair(
    env: ManagerBasedRLEnv, action_name: str = "joint_pos"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (current, previous) actions in decoded 29D action space."""
    action_term = env.action_manager.get_term(action_name)

    curr = getattr(action_term, "decoded_actions", None)
    if curr is None:
        curr = getattr(action_term, "_decoded_actions", None)
    if not isinstance(curr, torch.Tensor):
        curr = env.action_manager.action

    prev = getattr(action_term, "prev_decoded_actions", None)
    if prev is None:
        prev = getattr(action_term, "_prev_decoded_actions", None)
    if not isinstance(prev, torch.Tensor):
        prev = torch.zeros_like(curr)

    return (
        torch.nan_to_num(curr, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(prev, nan=0.0, posinf=0.0, neginf=0.0),
    )


def action_l2(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """L2 norm of decoded actions (29D space)."""
    curr, _ = _decoded_action_pair(env, action_name=action_name)
    return torch.sum(torch.square(curr), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """L2 norm of decoded action delta (29D space)."""
    curr, prev = _decoded_action_pair(env, action_name=action_name)
    return torch.sum(torch.square(curr - prev), dim=1)


def motion_active_body_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Per-active-keypoint exp tracking, then mean so single-EE vs both-EE have similar scale."""
    command = env.command_manager.get_term(command_name)
    err = torch.norm(command.body_pos_relative_w - command.robot_body_pos_w, dim=-1)  # [N, B]
    mask = command.env_body_mask  # [N, B], 0/1
    per = torch.exp(-(err**2) / (std**2))
    denom = torch.clamp(mask.sum(dim=-1), min=1.0)
    return (per * mask).sum(dim=-1) / denom


def motion_active_body_linear_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Per-active-keypoint exp tracking on linear velocity, then mean (same scale as single EE)."""
    command = env.command_manager.get_term(command_name)
    err = torch.norm(command.body_lin_vel_w - command.robot_body_lin_vel_w, dim=-1)  # [N, B]
    mask = command.env_body_mask  # [N, B], 0/1
    per = torch.exp(-(err**2) / (std**2))
    denom = torch.clamp(mask.sum(dim=-1), min=1.0)
    return (per * mask).sum(dim=-1) / denom

