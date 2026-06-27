from __future__ import annotations

from dataclasses import MISSING
from typing import Sequence

import torch
from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.mdp.commands import *  # noqa: F401, F403
from whole_body_tracking.tasks.tracking.mdp.commands import MultiMotionCommand, MultiMotionCommandCfg


class VRMultiMotionCommand(MultiMotionCommand):
    """Multi-motion command with command-manager keypoint masking modes.

    Reference goal positions/velocities use **NaN** on bodies hidden by the current mode (same idea as
    partial-mask whole-body tracking). Runners apply :func:`rsl_rl.utils.finite_checks.replace_nonfinite_with_zeros`
    **after** empirical normalization so the policy sees finite zeros with no gradient on masked dims.
    """

    cfg: "VRMultiMotionCommandCfg"

    def __init__(self, cfg: "VRMultiMotionCommandCfg", env):
        # Pin all envs to one mask mode during eval (e.g. play.py + video); None = stochastic sampling.
        self._eval_fixed_mode_idx: int | None = None
        super().__init__(cfg, env)
        self._compact_goal_observation = bool(getattr(cfg, "compact_goal_observation", True))
        self._mode_names = tuple(cfg.mask_mode_spec.keys())
        if len(self._mode_names) == 0:
            raise ValueError("mask_mode_spec must contain at least one mode.")

        probs = torch.tensor(list(cfg.mask_mode_probs), dtype=torch.float32, device=self.device)
        if probs.numel() != len(self._mode_names):
            raise ValueError(
                f"mask_mode_probs length ({probs.numel()}) must equal number of modes ({len(self._mode_names)})."
            )
        probs = torch.clamp(probs, min=0.0)
        if torch.all(probs <= 0):
            raise ValueError("mask_mode_probs must contain at least one positive value.")
        self._mode_probs = probs / probs.sum()

        self._mode_body_masks = torch.zeros((len(self._mode_names), len(self.cfg.body_names)), dtype=torch.float32, device=self.device)
        name_to_idx = {name: i for i, name in enumerate(self.cfg.body_names)}
        for mode_i, mode_name in enumerate(self._mode_names):
            for body_name in self.cfg.mask_mode_spec[mode_name]:
                if body_name not in name_to_idx:
                    raise ValueError(f"Body {body_name!r} in mode {mode_name!r} is not in cfg.body_names.")
                self._mode_body_masks[mode_i, name_to_idx[body_name]] = 1.0
        if self._compact_goal_observation:
            active_union = torch.any(self._mode_body_masks > 0.5, dim=0)
            self._obs_body_indices = torch.nonzero(active_union, as_tuple=False).squeeze(-1)
            if self._obs_body_indices.numel() == 0:
                raise ValueError(
                    "compact_goal_observation=True but no active keypoints exist in mask_mode_spec."
                )
        else:
            self._obs_body_indices = torch.arange(len(self.cfg.body_names), dtype=torch.long, device=self.device)

        self._env_mode_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._env_body_mask = self._mode_body_masks[self._env_mode_idx]
        self.metrics["vr_mode_index"] = self._env_mode_idx.to(torch.float32)
        self._ee_body_name_to_idx = {name: i for i, name in enumerate(self.cfg.body_names)}
        self._ee_metric_keys = {
            "left_ee": ("error_keypoint_pos_left_ee", "error_keypoint_vel_left_ee"),
            "right_ee": ("error_keypoint_pos_right_ee", "error_keypoint_vel_right_ee"),
            "both_ee": ("error_keypoint_pos_both_ee", "error_keypoint_vel_both_ee"),
        }
        self._ee_metric_body_indices = {
            "left_ee": self._resolve_metric_body_indices(("left_wrist_yaw_link",)),
            "right_ee": self._resolve_metric_body_indices(("right_wrist_yaw_link",)),
            "both_ee": self._resolve_metric_body_indices(("left_wrist_yaw_link", "right_wrist_yaw_link")),
        }
        self._mode_metric_group = self._build_mode_metric_group_map()
        for pos_key, vel_key in self._ee_metric_keys.values():
            self.metrics[pos_key] = torch.full((self.num_envs,), float("nan"), device=self.device)
            self.metrics[vel_key] = torch.full((self.num_envs,), float("nan"), device=self.device)

        probs_cpu = self._mode_probs.detach().cpu()
        lines = [
            "[VRMultiMotionCommand] EE keypoint mask modes (per-env resample; normalized probs):",
        ]
        for i, name in enumerate(self._mode_names):
            p = float(probs_cpu[i].item())
            bodies = ", ".join(cfg.mask_mode_spec[name])
            lines.append(f"  [{i}] {name!r}: p={p:.6f}  bodies=[{bodies}]")
        obs_body_names = [self.cfg.body_names[int(i)] for i in self._obs_body_indices.detach().cpu().tolist()]
        lines.append(
            f"  compact_goal_observation={self._compact_goal_observation}  "
            f"goal_keypoints={len(obs_body_names)}/{len(self.cfg.body_names)}"
        )
        lines.append(f"  goal_keypoint_names=[{', '.join(obs_body_names)}]")
        lines.append(f"  num_envs={self.num_envs}  num_bodies_in_command={len(self.cfg.body_names)}")
        print("\n".join(lines), flush=True)

    def _resolve_metric_body_indices(self, body_names: tuple[str, ...]) -> torch.Tensor | None:
        idx: list[int] = []
        for body_name in body_names:
            bi = self._ee_body_name_to_idx.get(body_name)
            if bi is None:
                return None
            idx.append(int(bi))
        return torch.tensor(idx, dtype=torch.long, device=self.device)

    def _build_mode_metric_group_map(self) -> dict[int, str]:
        mode_group: dict[int, str] = {}
        for mode_i, mode_name in enumerate(self._mode_names):
            bodies = set(self.cfg.mask_mode_spec[mode_name])
            has_left = "left_wrist_yaw_link" in bodies
            has_right = "right_wrist_yaw_link" in bodies
            if has_left and has_right:
                mode_group[mode_i] = "both_ee"
            elif has_left:
                mode_group[mode_i] = "left_ee"
            elif has_right:
                mode_group[mode_i] = "right_ee"
        return mode_group

    @property
    def env_mode_idx(self) -> torch.Tensor:
        return self._env_mode_idx

    @property
    def env_body_mask(self) -> torch.Tensor:
        return self._env_body_mask

    def _sample_modes(self, env_ids: torch.Tensor):
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = int(env_ids_t.numel())
        if n == 0:
            return
        if self._eval_fixed_mode_idx is not None:
            m = int(self._eval_fixed_mode_idx)
            sampled = torch.full((n,), m, dtype=torch.long, device=self.device)
        else:
            sampled = torch.multinomial(self._mode_probs, n, replacement=True)
        self._env_mode_idx[env_ids_t] = sampled
        self._env_body_mask[env_ids_t] = self._mode_body_masks[sampled]
        self.metrics["vr_mode_index"][env_ids_t] = sampled.to(torch.float32)

    def set_eval_fixed_mask_mode_idx(self, idx: int | None) -> None:
        """Pin every env to one VR mask mode during evaluation (e.g. play). None = resume stochastic sampling."""
        self._eval_fixed_mode_idx = None if idx is None else int(idx)

    def resample_all_mask_modes(self) -> None:
        """Re-draw VR mask modes for every env (e.g. after :meth:`set_eval_fixed_mask_mode_idx`)."""
        self._sample_modes(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() > 0:
            self._sample_modes(env_ids_t)
        super()._resample_command(env_ids)

    def ref_body_pos_robot_anchor_b(self) -> torch.Tensor:
        """Reference positions for compact keypoints; **NaN** where the current mode hides a body (not zeros)."""
        pos = super().ref_body_pos_robot_anchor_b().view(self.num_envs, len(self.cfg.body_names), 3)
        vis = self._env_body_mask.unsqueeze(-1)
        nan = torch.full_like(pos, float("nan"))
        masked = torch.where(vis > 0.5, pos, nan)
        return masked.index_select(1, self._obs_body_indices).reshape(self.num_envs, -1)

    def ref_body_lin_vel_robot_anchor_b(self) -> torch.Tensor:
        """Same NaN masking semantics as :meth:`ref_body_pos_robot_anchor_b`."""
        vel = super().ref_body_lin_vel_robot_anchor_b().view(self.num_envs, len(self.cfg.body_names), 3)
        vis = self._env_body_mask.unsqueeze(-1)
        nan = torch.full_like(vel, float("nan"))
        masked = torch.where(vis > 0.5, vel, nan)
        return masked.index_select(1, self._obs_body_indices).reshape(self.num_envs, -1)

    def _update_metrics(self):
        super()._update_metrics()

        pos_err_per_body = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1)
        vel_err_per_body = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1)

        for group, body_idx in self._ee_metric_body_indices.items():
            pos_key, vel_key = self._ee_metric_keys[group]
            if body_idx is None:
                self.metrics[pos_key].fill_(float("nan"))
                self.metrics[vel_key].fill_(float("nan"))
                continue

            mode_ids = [mid for mid, mg in self._mode_metric_group.items() if mg == group]
            if len(mode_ids) == 0:
                self.metrics[pos_key].fill_(float("nan"))
                self.metrics[vel_key].fill_(float("nan"))
                continue

            mode_ids_t = torch.tensor(mode_ids, dtype=torch.long, device=self.device)
            env_mask = (self._env_mode_idx[:, None] == mode_ids_t[None, :]).any(dim=1)
            if not bool(env_mask.any()):
                self.metrics[pos_key].fill_(float("nan"))
                self.metrics[vel_key].fill_(float("nan"))
                continue

            pos_group = pos_err_per_body.index_select(1, body_idx).mean(dim=-1)
            vel_group = vel_err_per_body.index_select(1, body_idx).mean(dim=-1)
            self.metrics[pos_key].fill_(float(pos_group[env_mask].mean().item()))
            self.metrics[vel_key].fill_(float(vel_group[env_mask].mean().item()))



@configclass
class VRMultiMotionCommandCfg(MultiMotionCommandCfg):
    class_type = VRMultiMotionCommand

    asset_name: str = MISSING
    motion: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    mask_mode_spec: dict[str, list[str]] = {
        "left_ee": ["left_wrist_yaw_link"],
        "right_ee": ["right_wrist_yaw_link"],
        "both_ee": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
    }
    mask_mode_probs: tuple[float, ...] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    # Keep only keypoints that are ever active in mask modes when building goal observations.
    compact_goal_observation: bool = True

