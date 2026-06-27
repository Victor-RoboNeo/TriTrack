"""World-frame anchor + points-of-interest tracking metrics during rollouts.

Attaches to *any* motion command (``MotionCommand`` / ``MultiMotionCommand`` and
subclasses — it does NOT require the partial-masked ``_env_body_mask`` that
:mod:`kp_mode_metric_logger` needs) and accumulates a single global bucket of:

  - ``anchor_pos``      — world-frame L2 pos error of the anchor (torso) body
  - ``anchor_lin_vel``  — world-frame L2 lin-vel error of the anchor body
  - ``poi_pos``         — mean world-frame L2 pos error over the 5 points of
                          interest (``COTRAIN_KP5_BODIES``: torso + L/R wrist +
                          L/R ankle), measured against the RAW clip world
                          reference (``body_pos_w``), i.e. NOT re-anchored to the
                          robot (anchor/root drift counts).
  - ``poi_lin_vel``     — mean world-frame L2 lin-vel error over the same 5 POI

The anchor metrics are read straight from ``cmd.metrics`` (already world-frame
``||anchor_pos_w - robot_anchor_pos_w||`` etc.). The POI metrics are recomputed
here so they are world-frame and restricted to the 5 bodies regardless of the
command's default ``error_body_pos`` (which is anchor-RELATIVE and over all
tracked bodies). This is exactly the objective used by the world-frame POI
reward, so the metric tracks the deployment-true accuracy of the teacher.

Design mirrors :mod:`kp_mode_metric_logger`: monkey-patch ``cmd._update_metrics``
so the hook fires on every env step regardless of training/eval mode. Reset
accumulators at the iteration boundary via :meth:`flush`. Flushed keys are
prefixed ``eval_world/`` so they land in their own wandb panel.
"""
from __future__ import annotations

import torch


class WorldPoiMetricLogger:
    """Streaming accumulator for world-frame anchor + 5-POI tracking errors.

    One global bucket (no per-mode stratification — the JC teacher this targets
    has no keypoint mask). :meth:`flush` returns ``(means, steps)`` keyed by
    ``eval_world/<metric>`` and resets internal state.
    """

    def __init__(self, motion_command, poi_body_names: list[str] | None = None) -> None:
        self.cmd = motion_command
        cfg_body_names = list(getattr(getattr(motion_command, "cfg", None), "body_names", []) or [])
        if not cfg_body_names:
            raise RuntimeError("WorldPoiMetricLogger requires a motion command with cfg.body_names.")

        if poi_body_names is None:
            # The "five points scenario": torso + L/R wrist + L/R ankle (no pelvis).
            from whole_body_tracking.tasks.tracking.config.g1.mask_modes import COTRAIN_KP5_BODIES

            poi_body_names = list(COTRAIN_KP5_BODIES)

        # Resolve POI names to indices into the command's body tensors
        # (``body_pos_w`` / ``robot_body_pos_w`` are in cfg.body_names order).
        self.poi_names: list[str] = [n for n in poi_body_names if n in cfg_body_names]
        missing = [n for n in poi_body_names if n not in cfg_body_names]
        if missing:
            print(
                f"[WorldPoiMetricLogger] WARNING: POI bodies not tracked by this command, skipped: {missing}",
                flush=True,
            )
        if not self.poi_names:
            raise RuntimeError(
                f"WorldPoiMetricLogger: none of the POI bodies {poi_body_names} are in cfg.body_names."
            )
        device = motion_command.device
        self.device = device
        self._poi_idx = torch.tensor(
            [cfg_body_names.index(n) for n in self.poi_names], dtype=torch.long, device=device
        )

        self._sums: dict[str, torch.Tensor] = {
            "anchor_pos": torch.zeros((), device=device),
            "anchor_lin_vel": torch.zeros((), device=device),
            "poi_pos": torch.zeros((), device=device),
            "poi_lin_vel": torch.zeros((), device=device),
        }
        self._count = torch.zeros((), device=device, dtype=torch.long)
        self._original_update_metrics = None

    # ----- attach / detach -----

    def attach(self) -> None:
        if self._original_update_metrics is not None:
            raise RuntimeError("WorldPoiMetricLogger.attach() called twice.")
        original = self.cmd._update_metrics

        def wrapped():
            original()
            try:
                self.step()
            except Exception as e:
                print(f"[WorldPoiMetricLogger] step() failed, disabling: {e!r}", flush=True)
                self.cmd._update_metrics = original
                self._original_update_metrics = None

        self._original_update_metrics = original
        self.cmd._update_metrics = wrapped

    def detach(self) -> None:
        if self._original_update_metrics is not None:
            self.cmd._update_metrics = self._original_update_metrics
            self._original_update_metrics = None

    # ----- step (called per env step after cmd._update_metrics) -----

    def step(self) -> None:
        cmd = self.cmd

        # Anchor errors are already world-frame in cmd.metrics:
        #   error_anchor_pos      = ||anchor_pos_w - robot_anchor_pos_w||
        #   error_anchor_lin_vel  = ||anchor_lin_vel_w - robot_anchor_lin_vel_w||
        anchor_pos = cmd.metrics["error_anchor_pos"]  # [N]
        anchor_vel = cmd.metrics["error_anchor_lin_vel"]  # [N]

        # POI errors: TRUE world-frame, restricted to the 5 POI, against the raw
        # clip world reference (NOT body_pos_relative_w — drift must count). Mean
        # over POI then over envs.
        idx = self._poi_idx
        poi_pos_err = torch.norm(
            cmd.body_pos_w[:, idx] - cmd.robot_body_pos_w[:, idx], dim=-1
        ).mean(dim=-1)  # [N]
        poi_vel_err = torch.norm(
            cmd.body_lin_vel_w[:, idx] - cmd.robot_body_lin_vel_w[:, idx], dim=-1
        ).mean(dim=-1)  # [N]

        n = anchor_pos.shape[0]
        self._sums["anchor_pos"] += anchor_pos.sum().to(self._sums["anchor_pos"].dtype)
        self._sums["anchor_lin_vel"] += anchor_vel.sum().to(self._sums["anchor_lin_vel"].dtype)
        self._sums["poi_pos"] += poi_pos_err.sum().to(self._sums["poi_pos"].dtype)
        self._sums["poi_lin_vel"] += poi_vel_err.sum().to(self._sums["poi_lin_vel"].dtype)
        self._count += n

    # ----- flush -----

    def flush(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return (means, steps) keyed ``eval_world/<metric>``; reset state.

        Returns empty dicts when no observations were accumulated this iteration.
        """
        c = int(self._count.detach().cpu().item())
        means: dict[str, float] = {}
        steps: dict[str, float] = {}
        if c > 0:
            for metric, sum_t in self._sums.items():
                means[f"eval_world/{metric}"] = float(sum_t.item()) / c
            steps["eval_world/steps"] = float(c)

        for sum_t in self._sums.values():
            sum_t.zero_()
        self._count.zero_()
        return means, steps


def attach_to_motion_command(
    motion_command, poi_body_names: list[str] | None = None
) -> WorldPoiMetricLogger:
    """Convenience: construct and attach. Caller keeps the reference for ``flush()``."""
    logger = WorldPoiMetricLogger(motion_command, poi_body_names=poi_body_names)
    logger.attach()
    return logger
