"""Per-clip aggregation of tracking metrics during eval rollouts.

Attach to a ``MultiMotionCommand`` (or subclass) via :func:`attach_to_motion_command`.
After eval finishes, call :meth:`dump` to write CSV + JSON tables keyed by clip filename,
plus a single ``summary.json`` with the mean across all clips.

Design: monkey-patches the motion command's ``_update_metrics`` to call our step hook
after the original updates. Avoids touching ``commands.py``; isolation matters because
training uses the same class and we want zero risk of disturbing training paths.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Sequence

import torch


# Metrics aggregated per clip. All are per-env scalar tensors of shape [num_envs].
DEFAULT_METRIC_KEYS: tuple[str, ...] = (
    "error_anchor_pos",
    "error_anchor_rot",
    "error_anchor_lin_vel",
    "error_anchor_ang_vel",
    "error_body_pos",
    "error_body_rot",
    "error_joint_pos",
    "error_joint_vel",
)


@dataclass
class _PerClipState:
    sums: dict[str, torch.Tensor]  # metric -> [num_motions]
    counts: torch.Tensor           # [num_motions], step contributions
    fall_episodes: torch.Tensor    # [num_motions], terminated episodes that fell
    total_episodes: torch.Tensor   # [num_motions]


class PerClipEvalLogger:
    """Streaming per-clip aggregator.

    Reads ``cmd.metrics[<key>]`` (shape ``[num_envs]``) and ``cmd.env_motion_indices``
    (shape ``[num_envs]``, dtype long) each env step. Uses ``index_add_`` for cheap
    O(num_envs) updates.
    """

    def __init__(
        self,
        motion_command,
        metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
        fall_detector=None,
    ) -> None:
        self.cmd = motion_command
        self.metric_keys = tuple(metric_keys)
        self.fall_detector = fall_detector  # callable(env) -> [num_envs] bool, optional
        # ``motion_paths`` is the per-rank list (sharded). For per-clip CSV that's exactly
        # what we want — each rank dumps its own subset.
        self.motion_paths: list[str] = list(getattr(motion_command, "motion_paths", []))
        if len(self.motion_paths) == 0:
            raise RuntimeError(
                "PerClipEvalLogger: motion command has no ``motion_paths`` — was it constructed?"
            )
        device = motion_command.device
        n = len(self.motion_paths)
        self.state = _PerClipState(
            sums={k: torch.zeros(n, device=device) for k in self.metric_keys},
            counts=torch.zeros(n, device=device, dtype=torch.long),
            fall_episodes=torch.zeros(n, device=device, dtype=torch.long),
            total_episodes=torch.zeros(n, device=device, dtype=torch.long),
        )
        self._original_update_metrics = None  # set in attach()

    # ----- attach / detach -----

    def attach(self) -> None:
        """Monkey-patch ``cmd._update_metrics`` to call ``self.step()`` after the original."""
        if self._original_update_metrics is not None:
            raise RuntimeError("PerClipEvalLogger.attach() called twice.")
        original = self.cmd._update_metrics

        def wrapped():
            original()
            try:
                self.step()
            except Exception as e:
                # Don't let logger bugs kill the eval rollout — print and disable.
                print(f"[PerClipEvalLogger] step() failed, disabling: {e!r}", flush=True)
                self.cmd._update_metrics = original  # restore
                self._original_update_metrics = None

        self._original_update_metrics = original
        self.cmd._update_metrics = wrapped

    def detach(self) -> None:
        if self._original_update_metrics is not None:
            self.cmd._update_metrics = self._original_update_metrics
            self._original_update_metrics = None

    # ----- step (called per env step) -----

    def step(self) -> None:
        env_motion_idx = self.cmd.env_motion_indices  # [N], long
        if env_motion_idx is None:
            return
        # Step counts: every env contributes one step to its current clip.
        ones = torch.ones_like(env_motion_idx, dtype=torch.long)
        self.state.counts.index_add_(0, env_motion_idx, ones)
        # Metric sums.
        metrics = self.cmd.metrics
        for k in self.metric_keys:
            t = metrics.get(k, None)
            if t is None:
                continue
            self.state.sums[k].index_add_(0, env_motion_idx, t.to(dtype=self.state.sums[k].dtype))

    # ----- dump -----

    def dump(self, out_dir: str, tag: str = "") -> dict:
        """Write per-clip CSV + JSON to ``out_dir`` and return the in-memory summary dict.

        ``tag`` is appended to filenames (e.g. "kp__pelvis_only"). Files written:
          - ``per_clip_metrics{__tag}.csv``
          - ``per_clip_metrics{__tag}.json``
          - ``summary{__tag}.json``  (mean across all clips, weighted by step count)
        """
        os.makedirs(out_dir, exist_ok=True)
        counts = self.state.counts.detach().cpu()
        means: dict[str, torch.Tensor] = {}
        for k, s in self.state.sums.items():
            means[k] = torch.where(counts > 0, s.detach().cpu() / counts.clamp(min=1), torch.zeros_like(s.detach().cpu()))

        suffix = f"__{tag}" if tag else ""
        csv_path = os.path.join(out_dir, f"per_clip_metrics{suffix}.csv")
        json_path = os.path.join(out_dir, f"per_clip_metrics{suffix}.json")
        summary_path = os.path.join(out_dir, f"summary{suffix}.json")

        # CSV
        header = ["clip", "steps"] + list(self.metric_keys)
        rows: list[str] = [",".join(header)]
        per_clip_records: list[dict] = []
        for i, path in enumerate(self.motion_paths):
            clip_id = os.path.basename(path)
            steps_i = int(counts[i].item())
            row = [clip_id, str(steps_i)]
            record = {"clip": clip_id, "path": path, "steps": steps_i}
            for k in self.metric_keys:
                v = float(means[k][i].item()) if steps_i > 0 else float("nan")
                row.append(f"{v:.6g}")
                record[k] = v
            rows.append(",".join(row))
            per_clip_records.append(record)
        with open(csv_path, "w") as f:
            f.write("\n".join(rows) + "\n")
        with open(json_path, "w") as f:
            json.dump({"tag": tag, "per_clip": per_clip_records}, f, indent=2)

        # Step-weighted overall summary.
        total_steps = float(counts.sum().item())
        summary: dict = {"tag": tag, "total_steps": total_steps, "num_clips": len(self.motion_paths)}
        if total_steps > 0:
            for k in self.metric_keys:
                summary[k] = float(self.state.sums[k].detach().cpu().sum().item() / total_steps)
        else:
            for k in self.metric_keys:
                summary[k] = float("nan")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(
            f"[PerClipEvalLogger] dumped {len(self.motion_paths)} clips to:\n"
            f"  {csv_path}\n  {json_path}\n  {summary_path}",
            flush=True,
        )
        return summary


def attach_to_motion_command(
    motion_command,
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
) -> PerClipEvalLogger:
    """Convenience: construct and attach the logger to ``motion_command``."""
    logger = PerClipEvalLogger(motion_command, metric_keys=metric_keys)
    logger.attach()
    return logger
