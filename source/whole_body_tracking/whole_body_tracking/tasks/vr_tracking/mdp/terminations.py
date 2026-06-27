from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from whole_body_tracking.tasks.tracking.mdp.terminations import *  # noqa: F401, F403


def bad_motion_active_body_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Terminate when active (mode-selected) command bodies violate z-position error threshold."""
    command = env.command_manager.get_term(command_name)
    error = torch.abs(command.body_pos_relative_w[:, :, -1] - command.robot_body_pos_w[:, :, -1])  # [N, B]
    mask = command.env_body_mask > 0.5  # [N, B]
    return torch.any(mask & (error > threshold), dim=-1)

