# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mask config and mask matrix building for latent-consistent distillation.

Mode spec maps mode name -> list of body names that are visible.
Body order is given by body_names (same as env command's body_names).
Output mask has shape (num_modes, num_bodies * 3), or (num_modes, num_bodies * 6) when
``duplicate_for_ref_body_lin_vel`` is True (same visibility for position and reference linear
velocity blocks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class MaskCfg:
    """Config for keypoint masking modes and their sampling distribution."""

    # mode_spec: mode_name -> list of body names that are visible in this mode
    mode_spec: Dict[str, List[str]]
    # mode_probs: probability per mode (same order as mode_names). If None, uniform.
    mode_probs: List[float] | None = None

    def get_mode_probs(self, device: torch.device) -> torch.Tensor:
        """Return tensor of mode probabilities (normalized)."""
        names = list(self.mode_spec.keys())
        n = len(names)
        if self.mode_probs is not None:
            if len(self.mode_probs) != n:
                raise ValueError(
                    f"mode_probs length {len(self.mode_probs)} != number of modes {n}"
                )
            p = torch.tensor(self.mode_probs, dtype=torch.float32, device=device)
        else:
            p = torch.ones(n, dtype=torch.float32, device=device)
        p = p / p.sum()
        return p


def build_mask_matrix(
    body_names: List[str],
    mode_spec: Dict[str, List[str]],
    device: torch.device | str = "cpu",
    *,
    duplicate_for_ref_body_lin_vel: bool = True,
) -> tuple[torch.Tensor, List[str]]:
    """Build per-body mask tensor from body order and mode spec.

    Args:
        body_names: Order of bodies in the observation (each body has 3 position coords).
        mode_spec: Map mode_name -> list of body names visible in that mode.
        device: Device for the output tensor.
        duplicate_for_ref_body_lin_vel: If True, concatenate a second copy of the mask so
            reference body linear velocity dims are masked identically to position (layout
            ``[pos | vel | goal_tail]`` per step).

    Returns:
        masks: (num_modes, num_bodies * 3) or (num_modes, num_bodies * 6) float tensor, 0.0 or 1.0.
        mode_names: List of mode names in the same order as the first dimension.
    """
    device = torch.device(device) if isinstance(device, str) else device
    mode_names = list(mode_spec.keys())
    num_bodies = len(body_names)
    body_to_idx = {b: i for i, b in enumerate(body_names)}
    num_kp = num_bodies * 3

    masks = torch.zeros(len(mode_names), num_kp, dtype=torch.float32, device=device)
    for i, mode_name in enumerate(mode_names):
        visible_bodies = set(mode_spec[mode_name])
        for body_name, idx in body_to_idx.items():
            if body_name in visible_bodies:
                masks[i, idx * 3 : (idx + 1) * 3] = 1.0
    if duplicate_for_ref_body_lin_vel:
        masks = torch.cat([masks, masks.clone()], dim=-1)
    return masks, mode_names


def sync_policy_keypoint_mask_from_motion_command(
    *,
    policy: torch.nn.Module,
    motion_term: object,
    num_envs: int,
    fixed_mode_idx: int | None = None,
) -> bool:
    """Copy per-env mask rows from a partial-mask motion term onto the policy.

    Returns:
        True if the policy mask tensors were updated (command-driven masking).
    """
    if not getattr(motion_term, "uses_command_manager_mask_sampling", False):
        return False
    mask_matrix = getattr(policy, "_mask_matrix", None)
    if mask_matrix is None:
        return False
    device = mask_matrix.device

    if fixed_mode_idx is not None:
        m = int(fixed_mode_idx)
        idx = torch.full((num_envs,), m, dtype=torch.long, device=device)
    else:
        # motion_term is an environment command term with an `env_keypoint_mask_mode_indices` tensor.
        # (Typed as `object` here for generic helper use.)
        idx = motion_term.env_keypoint_mask_mode_indices.to(device=device, dtype=torch.long)  # type: ignore[attr-defined]

    # Prefer task-side per-env visibility (incl. stochastic Bernoulli mode) so policy masking
    # matches the NaN semantics exactly.
    env_body_mask = getattr(motion_term, "env_body_mask", None)
    if env_body_mask is not None and hasattr(policy, "set_current_mask_from_env_body_mask"):
        policy._current_mode_indices = idx.clone()
        policy.set_current_mask_from_env_body_mask(env_body_mask.to(device=device, dtype=torch.float32))
        return True

    # Fallback: select precomputed mask-matrix row.
    policy._current_mask = mask_matrix[idx]
    policy._current_mode_indices = idx.clone()
    return True
