"""MUSE-KP distillation: same training recipe as MUSE-Transformer, on a KP-token student.

The base loss (BC against the frozen MLP teacher + cosine smoothness on encoder μ), the optimizer
contract (``policy.student.parameters()``), the gradient-accumulation logic, and the
``forward_for_update(obs, sample_z)`` signature are all identical to :class:`MuseDistillation`.

KP-specific additions:

1. **Warmup freeze curriculum**: hold the policy in a partially-frozen mode (default
   ``decoder_only``) for the first ``warmup_freeze_iters`` updates, then transition to
   ``post_warmup_freeze_mode`` (default ``none``) and rebuild the optimizer over the new param set.
2. **Aux anchor prediction head** (optional, ``aux_lambda > 0``): the policy's :class:`AuxAnchorPredictor`
   consumes (z, last-frame proprio) and predicts the anchor's linear velocity (3D) and orientation
   (6D continuous-rot rep). Targets come from teacher_obs slicing (``ref_base_lin_vel_b`` and
   ``motion_anchor_ori_b`` at the last history frame). EMA running statistics normalize both
   targets to unit variance so a single ``aux_lambda`` controls both heads' contribution.
   ``aux_stop_grad`` controls Run A (probe; gradient detached on z+proprio) vs Run B
   (encoder-pressure; gradient flows through z and the proprio-proj path).

3. **Teacher-eval inspection mode** (``teacher_eval_mode=True``): a pure diagnostic. The teacher
   drives *every* env every step (no student forward, no pilot mix), and ``update()`` is a
   no-op that only clears rollout storage. Training is fully bypassed; the student is never
   touched. The runner's existing per-rollout logging (``Train/mean_reward``,
   ``Train/mean_episode_length``, ``Episode/mdp_termination_success_rate``) then measures the
   *teacher's* own task performance under this env config — the achievable ceiling / a sanity
   check on the loaded teacher checkpoint. Takes precedence over warmup-freeze and pilot anneal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rsl_rl.algorithms.muse_distillation import MuseDistillation


class MuseKpDistillation(MuseDistillation):
    """MuseDistillation + warmup freeze curriculum + optional aux anchor prediction head."""

    def __init__(
        self,
        policy,
        *,
        warmup_freeze_iters: int = 0,
        warmup_freeze_mode: str = "decoder_only",
        post_warmup_freeze_mode: str = "none",
        # ---- Aux anchor predictor (optional) -----------------------------------------------------
        aux_lambda: float = 0.0,
        aux_stop_grad: bool = True,
        aux_target_ema_decay: float = 0.99,
        # Teacher-obs layout: per-frame term sizes (in TEACHER obs order) + history length, plus
        # term indices (0-based, within the term list) for the aux targets. The algorithm uses
        # these to slice the targets out of ``privileged_observations`` at training time.
        teacher_term_sizes: tuple[int, ...] | None = None,
        teacher_history_length: int = 5,
        aux_ori_term_index: int = -1,
        aux_speed_term_index: int = -1,
        # ---- Passive→active pilot anneal (DAgger-style β schedule) -------------------------------
        # Who drives the env: per-env per-rollout Bernoulli(teacher_fraction). The supervision
        # target is ALWAYS the teacher (BC loss recomputes student action from stored obs, so the
        # executed action is irrelevant to the loss). teacher_fraction:
        #   iter < start                 → 1.0   (pure teacher-pilot bootstrap)
        #   start <= iter < end          → anneal 1.0 → floor, shape = (1-t)^anneal_power
        #   iter >= end                  → floor (steady trickle of clean states)
        pilot_anneal_enabled: bool = False,
        pilot_teacher_start_iter: int = 1500,
        pilot_teacher_end_iter: int = 4000,
        pilot_teacher_floor: float = 0.08,
        pilot_anneal_power: float = 2.0,
        # ---- Teacher-eval inspection mode (diagnostic) -------------------------------------------
        # When True: teacher drives every env, update() is a no-op (storage clear only). The
        # student is never touched; the runner's rollout logging measures the teacher's own
        # task performance. Overrides warmup-freeze and pilot anneal.
        teacher_eval_mode: bool = False,
        **kwargs,
    ):
        # If a warmup is requested, force the policy into the warmup mode BEFORE super().__init__
        # builds the optimizer — otherwise the optimizer would lock in the wrong param set.
        self.warmup_freeze_iters = int(warmup_freeze_iters)
        self.warmup_freeze_mode = str(warmup_freeze_mode)
        self.post_warmup_freeze_mode = str(post_warmup_freeze_mode)
        if self.warmup_freeze_iters > 0 and hasattr(policy, "set_freeze_mode"):
            if getattr(policy, "freeze_mode", None) != self.warmup_freeze_mode:
                policy.set_freeze_mode(self.warmup_freeze_mode)
                print(
                    f"[MuseKpDistillation] Warmup phase active for {self.warmup_freeze_iters} iters: "
                    f"policy freeze_mode -> {self.warmup_freeze_mode!r}"
                )
        super().__init__(policy, **kwargs)
        self._warmup_done = self.warmup_freeze_iters <= 0

        # ---- Aux predictor setup -----------------------------------------------------------------
        self.aux_lambda = float(aux_lambda)
        self.aux_stop_grad = bool(aux_stop_grad)
        self.aux_target_ema_decay = float(aux_target_ema_decay)
        self.aux_enabled = self.aux_lambda > 0.0 and getattr(policy, "aux_predictor", None) is not None
        if self.aux_enabled:
            if teacher_term_sizes is None or aux_ori_term_index < 0 or aux_speed_term_index < 0:
                raise ValueError(
                    "Aux predictor enabled (aux_lambda > 0) but teacher-obs layout is missing. "
                    "Provide teacher_term_sizes, teacher_history_length, aux_ori_term_index, "
                    "aux_speed_term_index."
                )
            self.teacher_term_sizes = tuple(int(s) for s in teacher_term_sizes)
            self.teacher_history_length = int(teacher_history_length)
            self.aux_ori_term_index = int(aux_ori_term_index)
            self.aux_speed_term_index = int(aux_speed_term_index)
            # Compute slice offsets into the (history-concatenated) teacher_obs tensor.
            # Layout: each term contributes ``term_size * history_length`` consecutive dims; within
            # a term block layout is history-major (frame 0 first, frame H-1 last) — same as
            # ``_split_history_terms`` in MUSE-Transformer.
            ori_dim = self.teacher_term_sizes[self.aux_ori_term_index]
            speed_dim = self.teacher_term_sizes[self.aux_speed_term_index]
            if ori_dim != 6 or speed_dim != 3:
                raise ValueError(
                    f"Aux target term dims must be 6 (ori) and 3 (speed); got "
                    f"ori_dim={ori_dim}, speed_dim={speed_dim}."
                )
            term_offsets = [0]
            for s in self.teacher_term_sizes:
                term_offsets.append(term_offsets[-1] + int(s) * self.teacher_history_length)
            H = self.teacher_history_length
            # Last frame within each term block:
            self._aux_ori_slice = (
                term_offsets[self.aux_ori_term_index] + (H - 1) * ori_dim,
                term_offsets[self.aux_ori_term_index] + H * ori_dim,
            )
            self._aux_speed_slice = (
                term_offsets[self.aux_speed_term_index] + (H - 1) * speed_dim,
                term_offsets[self.aux_speed_term_index] + H * speed_dim,
            )

            # EMA normalization stats. Initialized with rough priors: 6D rot entries are in [-1, 1]
            # so σ_R ≈ 0.5; anchor linear velocity is in m/s (typically ±2 m/s in walking) so σ_v ≈ 1.
            # Both means start at 0. Updated each forward (per-rank; close-enough across ranks for
            # a normalization constant — exact sync not load-bearing).
            self.register_aux_buffers(
                speed_dim=3, ori_dim=6, init_speed_std=1.0, init_ori_std=0.5
            )
            print(
                f"[MuseKpDistillation] Aux anchor predictor ENABLED: λ={self.aux_lambda}, "
                f"stop_grad={self.aux_stop_grad}, EMA decay={self.aux_target_ema_decay}, "
                f"ori_slice={self._aux_ori_slice}, speed_slice={self._aux_speed_slice}."
            )
        else:
            self.teacher_term_sizes = None
            self.teacher_history_length = 0
            self.aux_ori_term_index = -1
            self.aux_speed_term_index = -1
            self._aux_ori_slice = (0, 0)
            self._aux_speed_slice = (0, 0)
            self._aux_speed_mean = None
            self._aux_speed_std = None
            self._aux_ori_mean = None
            self._aux_ori_std = None
            print(
                f"[MuseKpDistillation] Aux anchor predictor DISABLED "
                f"(aux_lambda={self.aux_lambda}, policy aux_predictor present="
                f"{getattr(policy, 'aux_predictor', None) is not None})."
            )

        # ---- Pilot anneal setup ------------------------------------------------------------------
        self.pilot_anneal_enabled = bool(pilot_anneal_enabled)
        self.pilot_teacher_start_iter = int(pilot_teacher_start_iter)
        self.pilot_teacher_end_iter = int(pilot_teacher_end_iter)
        self.pilot_teacher_floor = float(pilot_teacher_floor)
        self.pilot_anneal_power = float(pilot_anneal_power)
        # Per-env teacher mask, resampled once per collection window (rollout). ``_new_rollout``
        # is True at the first act() of each window (set by update() at end of the previous one).
        self._pilot_teacher_env_mask: torch.Tensor | None = None
        self._new_rollout = True
        if self.pilot_anneal_enabled:
            if not (
                0 <= self.pilot_teacher_start_iter <= self.pilot_teacher_end_iter
                and 0.0 <= self.pilot_teacher_floor <= 1.0
            ):
                raise ValueError(
                    f"Invalid pilot schedule: start={self.pilot_teacher_start_iter}, "
                    f"end={self.pilot_teacher_end_iter}, floor={self.pilot_teacher_floor}."
                )
            print(
                f"[MuseKpDistillation] Pilot anneal ENABLED: teacher_fraction 1.0 until iter "
                f"{self.pilot_teacher_start_iter}, anneal (power={self.pilot_anneal_power}) to "
                f"floor={self.pilot_teacher_floor} by iter {self.pilot_teacher_end_iter}, per-env "
                f"per-rollout. Supervision target stays the teacher throughout."
            )
        else:
            print("[MuseKpDistillation] Pilot anneal DISABLED (student pilots from iter 0).")

        # ---- Teacher-eval inspection mode --------------------------------------------------------
        self.teacher_eval_mode = bool(teacher_eval_mode)
        # Running teacher-action diagnostics (filled in act(), drained in update()).
        self._teval_act_norm_sum = 0.0
        self._teval_act_absmax = 0.0
        self._teval_steps = 0
        if self.teacher_eval_mode:
            print(
                "\n"
                "============================================================\n"
                "[MuseKpDistillation] TEACHER-EVAL MODE — NOT TRAINING.\n"
                "  The teacher drives every env every step; the student is\n"
                "  never updated (update() clears storage only). Read the\n"
                "  teacher's ceiling from the runner's rollout metrics:\n"
                "    Train/mean_reward, Train/mean_episode_length,\n"
                "    Episode/mdp_termination_success_rate.\n"
                "  warmup-freeze and pilot anneal are overridden (ignored).\n"
                "  NOTE: runner checkpoints saved this run are the UNTRAINED\n"
                "  student — discard them.\n"
                "============================================================\n"
            )

    def register_aux_buffers(
        self, speed_dim: int, ori_dim: int, init_speed_std: float, init_ori_std: float
    ) -> None:
        """Create EMA buffers on this algorithm instance for normalizing aux targets in the loss."""
        device = self.device
        self._aux_speed_mean = torch.zeros(speed_dim, device=device)
        self._aux_speed_std = torch.full((speed_dim,), float(init_speed_std), device=device)
        self._aux_ori_mean = torch.zeros(ori_dim, device=device)
        self._aux_ori_std = torch.full((ori_dim,), float(init_ori_std), device=device)

    def _update_aux_ema(self, mean_buf: torch.Tensor, std_buf: torch.Tensor, target: torch.Tensor) -> None:
        """EMA update for target normalization stats. ``target`` shape: [..., dim]; we reduce over
        all leading dims. Stop-grad: stats are buffers, target is detached at the call site.
        """
        d = self.aux_target_ema_decay
        t = target.detach().reshape(-1, target.shape[-1])
        batch_mean = t.mean(dim=0)
        # Use batch std (with eps floor) — bessel correction not load-bearing for normalization.
        batch_std = t.std(dim=0).clamp(min=1e-3)
        mean_buf.mul_(d).add_(batch_mean, alpha=1.0 - d)
        std_buf.mul_(d).add_(batch_std, alpha=1.0 - d)

    def _extract_aux_targets(self, teacher_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice (speed_target [..., 3], ori_target [..., 6]) from teacher_obs at the latest frame."""
        s_lo, s_hi = self._aux_speed_slice
        o_lo, o_hi = self._aux_ori_slice
        speed_target = teacher_obs[..., s_lo:s_hi]
        ori_target = teacher_obs[..., o_lo:o_hi]
        return speed_target, ori_target

    # ---- Aux-loss running diagnostics (filled inside _extra_loss_term, drained by _extra_log_dict).
    _aux_speed_loss_sum: float = 0.0
    _aux_ori_loss_sum: float = 0.0
    _aux_contribution_sum: float = 0.0
    _aux_steps: int = 0

    def _extra_loss_term(
        self, obs: torch.Tensor, privileged_obs: torch.Tensor | None
    ) -> torch.Tensor | None:
        if not self.aux_enabled or privileged_obs is None:
            return None
        # Run the aux predictor with the configured stop-grad mode. Note: this does a SECOND encode
        # pass on the encoder — wasteful in compute but isolates the aux gradient path cleanly and
        # avoids broadening forward_for_update's return signature for a feature-gated head.
        speed_pred, ori_pred = self.policy.forward_aux(obs, stop_grad=self.aux_stop_grad)
        speed_target, ori_target = self._extract_aux_targets(privileged_obs)

        # Detach targets and update EMA stats. Targets are env-side observations — no gradient.
        speed_target_d = speed_target.detach()
        ori_target_d = ori_target.detach()
        self._update_aux_ema(self._aux_speed_mean, self._aux_speed_std, speed_target_d)
        self._update_aux_ema(self._aux_ori_mean, self._aux_ori_std, ori_target_d)

        # Normalize prediction AND target by the EMA stats (target std/mean detached buffers).
        speed_std = self._aux_speed_std.clamp(min=1e-3)
        ori_std = self._aux_ori_std.clamp(min=1e-3)
        speed_pred_n = (speed_pred - self._aux_speed_mean) / speed_std
        speed_tgt_n = (speed_target_d - self._aux_speed_mean) / speed_std
        ori_pred_n = (ori_pred - self._aux_ori_mean) / ori_std
        ori_tgt_n = (ori_target_d - self._aux_ori_mean) / ori_std

        # Smooth-L1 (= Huber with β=1) — robust to occasional outlier velocity spikes.
        speed_loss = F.smooth_l1_loss(speed_pred_n, speed_tgt_n)
        ori_loss = F.smooth_l1_loss(ori_pred_n, ori_tgt_n)
        aux_loss = speed_loss + ori_loss
        contribution = self.aux_lambda * aux_loss

        # Stash diagnostics (use .item() so we don't pin the autograd graph).
        self._aux_speed_loss_sum += float(speed_loss.detach().item())
        self._aux_ori_loss_sum += float(ori_loss.detach().item())
        self._aux_contribution_sum += float(contribution.detach().item())
        self._aux_steps += 1
        return contribution

    def _extra_log_dict(self) -> dict:
        if not self.aux_enabled or self._aux_steps == 0:
            base = {
                "aux_speed_loss": 0.0,
                "aux_ori_loss": 0.0,
                "aux_contribution": 0.0,
                "aux_to_behavior_ratio": 0.0,
                "aux_lambda": self.aux_lambda,
                "aux_stop_grad": float(self.aux_stop_grad),
            }
            # Reset counters for next update.
            self._aux_speed_loss_sum = 0.0
            self._aux_ori_loss_sum = 0.0
            self._aux_contribution_sum = 0.0
            self._aux_steps = 0
            return base
        n = self._aux_steps
        mean_speed = self._aux_speed_loss_sum / n
        mean_ori = self._aux_ori_loss_sum / n
        mean_contrib = self._aux_contribution_sum / n
        # Reset for next update.
        self._aux_speed_loss_sum = 0.0
        self._aux_ori_loss_sum = 0.0
        self._aux_contribution_sum = 0.0
        self._aux_steps = 0
        return {
            "aux_speed_loss": mean_speed,
            "aux_ori_loss": mean_ori,
            "aux_contribution": mean_contrib,
            "aux_lambda": self.aux_lambda,
            "aux_stop_grad": float(self.aux_stop_grad),
        }

    # ---- Pilot anneal --------------------------------------------------------------------------

    def _current_teacher_fraction(self) -> float:
        """teacher_fraction(iter) per the configured schedule. ``iter`` = completed-update count
        (≈ current learning iteration; off-by-one vs the true iter is negligible against the
        coarse start/end breakpoints).
        """
        if not self.pilot_anneal_enabled:
            return 0.0
        it = self.num_updates
        s, e = self.pilot_teacher_start_iter, self.pilot_teacher_end_iter
        fl, p = self.pilot_teacher_floor, self.pilot_anneal_power
        if it < s:
            return 1.0
        if it >= e:
            return fl
        t = (it - s) / max(1, e - s)  # [0, 1]
        # Convex when p > 1: stays teacher-heavy through the first part of the window, then
        # drops faster — banks more clean nominal behavior before self-recovery practice.
        return fl + (1.0 - fl) * ((1.0 - t) ** p)

    def act(self, obs, teacher_obs):
        """Per-env per-rollout pilot mix. The SUPERVISION target (privileged_actions) is always
        the teacher; only the action that steps the env is mixed teacher/student. The MUSE BC
        loss recomputes the student action from stored obs, so the executed action never enters
        the loss — this is pure DAgger-style state-distribution control.

        ``transition.actions`` stores the EXECUTED (mixed) action to keep the storage convention
        (stored action == action taken); it is unused by the MUSE update either way.
        """
        if self.teacher_eval_mode:
            # Diagnostic: teacher drives every env. Skip the student forward entirely (the
            # untrained student is irrelevant and update() never reads it). Still populate the
            # transition fields RolloutStorage.add_transitions() expects.
            teacher_a = self.policy.evaluate(teacher_obs).detach()
            self.transition.actions = teacher_a
            self.transition.privileged_actions = teacher_a
            self.transition.observations = obs
            self.transition.privileged_observations = teacher_obs
            self._teval_act_norm_sum += float(teacher_a.norm(dim=-1).mean().item())
            self._teval_act_absmax = max(
                self._teval_act_absmax, float(teacher_a.abs().max().item())
            )
            self._teval_steps += 1
            return teacher_a

        student_a = self.policy.act(obs).detach()
        teacher_a = self.policy.evaluate(teacher_obs).detach()
        self.transition.privileged_actions = teacher_a
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs

        if not self.pilot_anneal_enabled:
            self.transition.actions = student_a
            return student_a

        num_envs = obs.shape[0]
        # Resample the per-env teacher mask once per collection window (rollout). ``_new_rollout``
        # is set True by update() at the end of the previous window; also resample if the env
        # count changed or no mask exists yet (first act).
        if (
            self._new_rollout
            or self._pilot_teacher_env_mask is None
            or self._pilot_teacher_env_mask.shape[0] != num_envs
        ):
            frac = self._current_teacher_fraction()
            self._pilot_teacher_env_mask = torch.rand(num_envs, device=obs.device) < frac
            self._new_rollout = False

        mask = self._pilot_teacher_env_mask.unsqueeze(-1)  # [N, 1]
        stepped = torch.where(mask, teacher_a, student_a)
        self.transition.actions = stepped
        return stepped

    def _rebuild_optimizer(self) -> None:
        """Recreate Adam over the (newly updated) ``policy.student.parameters()`` group.

        Rebuilds adaptive-reg / adaptive-KL param groups too. This drops Adam state for params that
        carried it through the warmup phase — acceptable; KP-only layers will pick up new momentum
        within a few updates and the just-unfrozen decoder weights had no Adam state anyway.
        """
        self.optimizer = optim.Adam(self.policy.student.parameters(), lr=self.learning_rate)
        if self.use_adaptive_regularization and self.log_var_reg is not None:
            self.optimizer.add_param_group(
                {"params": [self.log_var_reg], "lr": self.learning_rate}
            )
        if self.use_adaptive_kl and self.log_var_kl is not None:
            self.optimizer.add_param_group(
                {"params": [self.log_var_kl], "lr": self.learning_rate}
            )

    def update(self, current_iter=None) -> dict:
        if self.teacher_eval_mode:
            # No-op: no gradient step, student untouched. Just clear the rollout storage so the
            # next collection window's add_transitions() doesn't overflow, and surface teacher
            # action diagnostics. The runner's own per-rollout logging carries the real signal
            # (reward / episode length / MDP success), now measured on the teacher.
            del current_iter
            self.num_updates += 1
            if self.storage is not None:
                self.storage.clear()
            n = max(1, self._teval_steps)
            out = {
                "teacher_eval_mode": 1.0,
                "teacher_action_norm_mean": self._teval_act_norm_sum / n,
                "teacher_action_absmax": self._teval_act_absmax,
                "behavior": 0.0,
            }
            self._teval_act_norm_sum = 0.0
            self._teval_act_absmax = 0.0
            self._teval_steps = 0
            return out

        out = super().update(current_iter=current_iter)
        if not self._warmup_done and self.num_updates >= self.warmup_freeze_iters:
            changed = self.policy.set_freeze_mode(self.post_warmup_freeze_mode)
            if changed:
                self._rebuild_optimizer()
                print(
                    f"[MuseKpDistillation] Warmup done at update={self.num_updates}: "
                    f"freeze_mode {self.warmup_freeze_mode!r} -> {self.post_warmup_freeze_mode!r}; "
                    f"optimizer rebuilt over new param set."
                )
            self._warmup_done = True
            out["warmup_freeze_transition"] = 1.0
        else:
            out["warmup_freeze_transition"] = 0.0
        # Always log the active freeze mode (1=decoder_plus_shared_encoder, 2=decoder_only, 3=none)
        # so the wandb chart shows the transition timestep cleanly.
        mode_to_int = {"decoder_plus_shared_encoder": 1, "decoder_only": 2, "none": 3}
        out["freeze_mode_int"] = float(
            mode_to_int.get(getattr(self.policy, "freeze_mode", "none"), 0)
        )
        # Pilot anneal: mark the next collection window for a fresh per-env mask resample, and log
        # the teacher_fraction that the next window will use (post-increment num_updates → matches
        # the iteration whose collection comes next).
        self._new_rollout = True
        out["pilot_teacher_fraction"] = (
            self._current_teacher_fraction() if self.pilot_anneal_enabled else 0.0
        )
        return out
