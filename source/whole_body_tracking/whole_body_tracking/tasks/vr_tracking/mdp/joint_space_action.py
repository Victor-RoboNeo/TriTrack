"""Joint-position action for VR tracking with ``_proprio_dim`` for split obs normalizer (matches latent layout)."""

from __future__ import annotations

import torch
from isaaclab.envs.mdp import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass


class VRJointPositionAction(JointPositionAction):
    """Same as :class:`JointPositionAction`; exposes ``_proprio_dim`` for ``try_build_rl_split_normalizers_from_checkpoint``.

    Layout matches policy history stacking: ``(num_joints*3 + 3) * proprio_history_length``.
    """

    cfg: "VRJointPositionActionCfg"

    def __init__(self, cfg: "VRJointPositionActionCfg", env):
        super().__init__(cfg, env)
        j = int(self.action_dim)
        h = int(cfg.proprio_history_length)
        self._proprio_dim = (j * 3 + 3) * h

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)


@configclass
class VRJointPositionActionCfg(JointPositionActionCfg):
    class_type = VRJointPositionAction
    proprio_history_length: int = 5


__all__ = ["VRJointPositionAction", "VRJointPositionActionCfg"]
