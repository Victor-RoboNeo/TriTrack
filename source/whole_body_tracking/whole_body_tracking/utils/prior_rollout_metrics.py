"""Streaming prior-rollout scalar metrics (no full-horizon [T×N×D] tensors)."""

from __future__ import annotations

import torch


def finalize_streaming_prior_metrics(
    *,
    num_envs: int,
    first_fall_step: torch.Tensor,
    action_sum: torch.Tensor,
    action_sumsq: torch.Tensor,
    action_count: int,
    sum_dp: float,
    sum_da: float,
    n_delta: int,
) -> dict[str, float]:
    """Build scalar report from pre-fall pooled streaming stats."""
    fell_any = first_fall_step > 0
    n_fell = int(fell_any.sum().item())
    steps_survived = torch.where(fell_any, (first_fall_step - 1).clamp(min=0).float(), first_fall_step.float())

    if action_count > 0:
        n = float(action_count)
        temporal_var = (action_sumsq / n) - (action_sum / n) ** 2
        temporal_std_mean = float(temporal_var.clamp(min=0.0).sqrt().mean().item())
    else:
        temporal_std_mean = 0.0

    mean_dp = sum_dp / max(n_delta, 1)
    mean_da = sum_da / max(n_delta, 1)
    ratio = mean_da / (mean_dp + 1e-8)
    
    metrics: dict[str, float] = {
        "prior_eval/fall_rate": n_fell / max(num_envs, 1),
        "prior_eval/steps_survived_mean": float(steps_survived.mean().item()),
        "prior_eval/action_temporal_std_mean": temporal_std_mean,
        "prior_eval/delta_proprio_l2_mean": float(mean_dp),
        "prior_eval/sensitivity_delta_a_over_delta_p": float(ratio),
    }
    return metrics
