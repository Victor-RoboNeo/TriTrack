# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
from dataclasses import asdict

import torch
from torch.utils.tensorboard import SummaryWriter


def _is_finite_log_scalar(value) -> bool:
    """True if ``value`` is safe to log as a single scalar (finite, not NaN/Inf)."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool(torch.isfinite(value.float()).all().item())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, int):
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return True

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        # Slow networks / VPNs can exceed wandb's default 90s init timeout.
        init_timeout = float(os.environ.get("WANDB_INIT_TIMEOUT", "300"))
        wandb_settings = wandb.Settings(init_timeout=init_timeout)

        # Initialize wandb
        wandb.init(project=project, entity=entity, name=run_name, settings=wandb_settings)

        # Add log directory to wandb
        wandb.config.update({"log_dir": log_dir})

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        wandb.config.update({"runner_cfg": runner_cfg})
        wandb.config.update({"policy_cfg": policy_cfg})
        wandb.config.update({"alg_cfg": alg_cfg})
        try:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            wandb.config.update({"env_cfg": asdict(env_cfg)})

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        # Skip NaN/Inf so W&B (and TensorBoard) are not polluted; e.g. EE metrics when no env uses that mask bucket.
        if not _is_finite_log_scalar(scalar_value):
            return
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({self._map_path(tag): scalar_value}, step=global_step)

    def stop(self):
        wandb.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        wandb.save(path, base_path=os.path.dirname(path))

    """
    Private methods.
    """

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path
