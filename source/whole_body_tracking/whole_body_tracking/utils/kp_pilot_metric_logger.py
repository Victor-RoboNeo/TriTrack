"""Per-pilot-modality KP tracking metrics for the MUSE co-train rollouts.

Buckets each env step into one of two pilots — JC or KP — based on
``policy._last_pilot_kp_mask`` (filled in by :meth:`LatentBottleneckMUSECoTrain._act_pilot`),
and accumulates per-bucket anchor-pos error + visible-body-pos error.

Termination attribution lives in :meth:`MuseCoTrainDistillation.process_env_step` because
``dones`` are only available at the runner / algorithm boundary, not inside the motion
command's ``_update_metrics`` hook. Runner aggregates the algorithm-side counters at log
time and writes ``pilot/{jc,kp}/success_rate`` from there.

Design mirrors :mod:`kp_mode_metric_logger`: monkey-patch ``cmd._update_metrics`` so the
hook fires on every env step. Reset at iteration boundary via :meth:`flush`.
"""
from __future__ import annotations

import torch


class KpPilotMetricLogger:
    """Per-pilot streaming accumulator for KP tracking metrics (2 buckets: jc, kp).

    Bucketing key is ``policy._last_pilot_kp_mask`` (shape ``[num_envs]``, bool).
    """

    METRIC_ANCHOR_POS = "error_anchor_pos"
    METRIC_BODY_POS_VISIBLE = "error_body_pos_visible"
    PILOT_NAMES: tuple[str, str] = ("jc", "kp")

    def __init__(self, motion_command, policy) -> None:
        if not hasattr(motion_command, "_env_body_mask"):
            raise RuntimeError(
                "KpPilotMetricLogger requires a PartialMaskedMultiMotionCommand (no _env_body_mask)."
            )
        self.cmd = motion_command
        self.policy = policy
        device = motion_command.device
        self.device = device
        # 2 pilot buckets (jc=0, kp=1).
        self._sums = {
            self.METRIC_ANCHOR_POS: torch.zeros(2, device=device),
            self.METRIC_BODY_POS_VISIBLE: torch.zeros(2, device=device),
        }
        self._counts = torch.zeros(2, device=device, dtype=torch.long)
        self._original_update_metrics = None

    # ----- attach / detach -----

    def attach(self) -> None:
        if self._original_update_metrics is not None:
            raise RuntimeError("KpPilotMetricLogger.attach() called twice.")
        original = self.cmd._update_metrics

        def wrapped():
            original()
            try:
                self.step()
            except Exception as e:
                print(f"[KpPilotMetricLogger] step() failed, disabling: {e!r}", flush=True)
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
        is_kp = getattr(self.policy, "_last_pilot_kp_mask", None)
        if is_kp is None:
            # No act() call yet (first iter pre-rollout); skip silently.
            return
        cmd = self.cmd
        body_mask = cmd._env_body_mask  # [N, B]
        # Per-env world-frame L2 pos error averaged over the env's currently-visible bodies.
        per_body_pos_err = torch.norm(cmd.body_pos_relative_w - cmd.robot_body_pos_w, dim=-1)  # [N, B]
        visible_count = body_mask.sum(dim=-1).clamp(min=1.0)  # [N]; all-masked envs avoid div-by-zero
        visible_pos_err = (per_body_pos_err * body_mask).sum(dim=-1) / visible_count  # [N]
        anchor_pos = cmd.metrics[self.METRIC_ANCHOR_POS]  # [N]

        pilot_idx = is_kp.to(device=self.device, dtype=torch.long)  # [N]; 0=jc, 1=kp
        self._sums[self.METRIC_ANCHOR_POS].index_add_(
            0, pilot_idx, anchor_pos.to(self._sums[self.METRIC_ANCHOR_POS].dtype)
        )
        self._sums[self.METRIC_BODY_POS_VISIBLE].index_add_(
            0, pilot_idx, visible_pos_err.to(self._sums[self.METRIC_BODY_POS_VISIBLE].dtype)
        )
        ones = torch.ones_like(pilot_idx, dtype=torch.long)
        self._counts.index_add_(0, pilot_idx, ones)

    # ----- flush -----

    def flush(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return ``(means, steps)`` dicts keyed by ``pilot/<jc|kp>/<metric>``. Resets state.

        Modes (pilots) with zero observations this iteration are omitted.
        """
        counts_cpu = self._counts.detach().cpu()
        means: dict[str, float] = {}
        steps: dict[str, float] = {}
        for pi, pilot in enumerate(self.PILOT_NAMES):
            c = int(counts_cpu[pi].item())
            if c == 0:
                continue
            steps[f"pilot/{pilot}/steps"] = float(c)
            for metric, sum_t in self._sums.items():
                means[f"pilot/{pilot}/{metric}"] = float(sum_t[pi].item()) / max(c, 1)

        for sum_t in self._sums.values():
            sum_t.zero_()
        self._counts.zero_()
        return means, steps


def attach_to_motion_command(motion_command, policy) -> KpPilotMetricLogger:
    """Convenience: construct and attach. Caller keeps the reference for ``flush()``."""
    logger = KpPilotMetricLogger(motion_command, policy)
    logger.attach()
    return logger
