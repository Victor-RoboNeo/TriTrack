"""Mode-stratified KP tracking metrics during training rollouts.

Attaches to a :class:`PartialMaskedMultiMotionCommand` and accumulates, per KP mask mode:
  - ``error_anchor_pos``         (anchor / torso world-frame L2 pos error; mode-independent signal)
  - ``error_anchor_lin_vel``     (anchor world-frame L2 lin-vel error; mode-independent signal)
  - ``error_body_pos_visible``   (mean per-body world-frame L2 pos error restricted to each env's
                                  currently-visible body subset — the "point of interest" error)

The anchor metrics are not mode-specific by definition; they're logged per-mode to surface how
each mask mode degrades torso tracking. ``error_body_pos_visible`` is the mode-relevant signal —
averaged only over the bodies that mode tells the encoder to track.

Design mirrors :mod:`per_clip_eval_logger`: monkey-patch ``cmd._update_metrics`` so the hook fires
on every env step regardless of training/eval mode. Reset accumulators at iteration boundary via
:meth:`flush`.
"""
from __future__ import annotations

import torch


class KpModeMetricLogger:
    """Per-mode streaming accumulator for KP tracking metrics.

    Bucketing key is ``cmd._env_mode_idx`` (shape ``[num_envs]``). Per step, each env contributes
    one sample to its current mode's running sum + count. :meth:`flush` returns ``(sums, counts)``
    keyed by ``kp_modes/<mode_name>/<metric>`` and resets internal state.
    """

    METRIC_ANCHOR_POS = "error_anchor_pos"
    METRIC_ANCHOR_LIN_VEL = "error_anchor_lin_vel"
    METRIC_BODY_POS_VISIBLE = "error_body_pos_visible"

    def __init__(self, motion_command) -> None:
        self.cmd = motion_command
        if not hasattr(motion_command, "_env_mode_idx") or not hasattr(motion_command, "_env_body_mask"):
            raise RuntimeError(
                "KpModeMetricLogger requires a PartialMaskedMultiMotionCommand (no _env_mode_idx / _env_body_mask)."
            )
        self.mode_names: tuple[str, ...] = tuple(motion_command._mode_names)
        self.num_modes = len(self.mode_names)
        device = motion_command.device
        self.device = device
        self._sums = {
            self.METRIC_ANCHOR_POS: torch.zeros(self.num_modes, device=device),
            self.METRIC_ANCHOR_LIN_VEL: torch.zeros(self.num_modes, device=device),
            self.METRIC_BODY_POS_VISIBLE: torch.zeros(self.num_modes, device=device),
        }
        self._counts = torch.zeros(self.num_modes, device=device, dtype=torch.long)
        self._original_update_metrics = None

    # ----- attach / detach -----

    def attach(self) -> None:
        if self._original_update_metrics is not None:
            raise RuntimeError("KpModeMetricLogger.attach() called twice.")
        original = self.cmd._update_metrics

        def wrapped():
            original()
            try:
                self.step()
            except Exception as e:
                print(f"[KpModeMetricLogger] step() failed, disabling: {e!r}", flush=True)
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
        env_mode = cmd._env_mode_idx  # [N]
        body_mask = cmd._env_body_mask  # [N, B]

        # Per-body TRUE world-frame L2 pos error (raw, no mask): [N, B].
        # Must use ``body_pos_w`` (raw clip world reference — anchor/root drift
        # COUNTS), NOT ``body_pos_relative_w`` (reference re-anchored to the robot,
        # drift removed). This mirrors the world-frame POI reward
        # (``motion_visible_kp_position_error_exp_world``) exactly, so the metric
        # tracks the reward's actual objective.
        per_body_pos_err = torch.norm(cmd.body_pos_w - cmd.robot_body_pos_w, dim=-1)
        visible_count = body_mask.sum(dim=-1).clamp(min=1.0)  # [N]; all-masked envs avoid div-by-zero
        visible_pos_err = (per_body_pos_err * body_mask).sum(dim=-1) / visible_count  # [N]

        anchor_pos = cmd.metrics[self.METRIC_ANCHOR_POS]  # [N]
        anchor_vel = cmd.metrics[self.METRIC_ANCHOR_LIN_VEL]  # [N]

        # Cast accumulators' dtype to match the float metrics for index_add_.
        self._sums[self.METRIC_ANCHOR_POS].index_add_(0, env_mode, anchor_pos.to(self._sums[self.METRIC_ANCHOR_POS].dtype))
        self._sums[self.METRIC_ANCHOR_LIN_VEL].index_add_(0, env_mode, anchor_vel.to(self._sums[self.METRIC_ANCHOR_LIN_VEL].dtype))
        self._sums[self.METRIC_BODY_POS_VISIBLE].index_add_(0, env_mode, visible_pos_err.to(self._sums[self.METRIC_BODY_POS_VISIBLE].dtype))

        ones = torch.ones_like(env_mode, dtype=torch.long)
        self._counts.index_add_(0, env_mode, ones)

    # ----- flush -----

    def flush(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return (per-mode means, per-mode step counts) keyed by ``kp_modes/<mode>/<metric>``.

        Resets internal state. Modes with zero observations this iteration are omitted.
        """
        counts_cpu = self._counts.detach().cpu()
        means: dict[str, float] = {}
        steps: dict[str, float] = {}
        for mode_i, mode_name in enumerate(self.mode_names):
            c = int(counts_cpu[mode_i].item())
            if c == 0:
                continue
            steps[f"kp_modes/{mode_name}/steps"] = float(c)
            for metric, sum_t in self._sums.items():
                v = float(sum_t[mode_i].item()) / max(c, 1)
                means[f"kp_modes/{mode_name}/{metric}"] = v

        # Reset.
        for sum_t in self._sums.values():
            sum_t.zero_()
        self._counts.zero_()
        return means, steps


def attach_to_motion_command(motion_command) -> KpModeMetricLogger:
    """Convenience: construct and attach. Caller keeps the reference for ``flush()``."""
    logger = KpModeMetricLogger(motion_command)
    logger.attach()
    return logger
