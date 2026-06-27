"""Reuse tracking observation terms for vr_tracking."""

from whole_body_tracking.tasks.tracking.mdp.observations import *  # noqa: F401, F403
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def last_action(env: "ManagerBasedRLEnv", action_name: str | None = "joint_pos") -> torch.Tensor:
    """Return last action in pre-processed action space (tracking-consistent).

    For latent-residual policies, this should be the decoder output (delta-like action before
    JointPositionAction affine transform), not the final processed joint targets.
    """
    action_term = env.action_manager.get_term(action_name)
    decoded = getattr(action_term, "_decoded_actions", None)
    if isinstance(decoded, torch.Tensor):
        return torch.nan_to_num(decoded, nan=0.0, posinf=0.0, neginf=0.0)
    processed = getattr(action_term, "processed_actions", None)
    if isinstance(processed, torch.Tensor):
        return torch.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
    # Final fallback for unexpected action terms.
    fallback = env.action_manager.action
    return torch.nan_to_num(fallback, nan=0.0, posinf=0.0, neginf=0.0)

