# SPDX-License-Identifier: BSD-3-Clause
"""Strict finiteness checks for tensors that feed neural networks."""

from __future__ import annotations

import torch


def require_all_finite(tensor: torch.Tensor, label: str) -> None:
    """Raise if ``tensor`` contains NaN or Inf.

    Use for policy / encoder inputs after any masking or NaN-padding from the environment
    has been resolved to finite values (see :func:`replace_nonfinite_with_zeros` on normalized
    observations in :class:`~rsl_rl.runners.on_policy_runner.OnPolicyRunner`).
    """
    if torch.isfinite(tensor).all():
        return
    bad = ~torch.isfinite(tensor)
    count = int(bad.sum().item())
    raise RuntimeError(
        f"Non-finite values in {label} (bad element count={count}). "
        "Policy inputs must be finite before the forward pass. "
        "If the environment emits NaN for masked keypoints, apply the visibility mask or "
        "sanitize (e.g. torch.where / nan_to_num) before calling the policy."
    )


def replace_nonfinite_with_zeros(tensor: torch.Tensor) -> torch.Tensor:
    """Turn NaN/Inf into 0 with **zero gradient** on non-finite positions.

    Apply to **normalized** actor/critic inputs after :class:`~rsl_rl.modules.EmpiricalNormalization`
    so running stats still ignore masked keypoint NaNs, while MLPs receive finite tensors.
    ``torch.where`` ensures masked dimensions do not receive input gradients.
    """
    return torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
