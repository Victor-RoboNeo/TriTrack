"""MUSE-Human-KP distillation: dual-modality (KP + human-motion) BC + smoothness + alignment.

Forward / loss contract mirrors :class:`MuseCoTrainDistillation` (the JC-KP cotrain) — only the
modality identities change (``jc`` → ``human``):

- ``policy.act(obs)`` does per-env 50/50 piloting (both modalities forward; partition selects
  which action drives ``env.step()``). Stored in the rollout for env state propagation only.
- ``policy.forward_for_update(obs)`` re-forwards both modalities on the full obs batch and returns
  ``{action_kp, action_human, mu_kp, mu_human, log_sigma_kp, log_sigma_human}``.

Loss (per transition, summed across all envs):
    L = MSE(action_kp, a_teacher) + MSE(action_human, a_teacher)            # 2× behavior
      + λ_reg_kp · (1 − cos(μ_kp^t, μ_kp^{t-1}))                            # KP smoothness
      + λ_reg_human · (1 − cos(μ_human^t, μ_human^{t-1}))                   # human smoothness
      + λ_align · (1 − cos(μ_kp^t, μ_human^t))                              # cross-modal alignment

The alignment term is the explicit "human description ≡ G1 description" signal the encoder needs to
make GENMO-driven inference work. Without it, the two modalities can drift apart even though they
share the decoder.

Warmup-freeze curriculum
------------------------
The intended workflow warmstarts from a MUSE-KP checkpoint where KP + shared backbone + decoder are
already trained, and only the human-specific layers (``human_proj``, ``human_body_id_emb``,
``modality_emb`` slot [2]) are new. ``warmup_freeze_iters`` + ``warmup_freeze_mode`` (default
``"human_only"``) hold the policy in human-only training for the first N updates, then transition
to ``post_warmup_freeze_mode`` (default ``"none"``) and rebuild the optimizer.

This mirrors :class:`MuseKpDistillation`'s warmup curriculum, with one difference: the warmup mode
here is ``"human_only"`` (everything frozen except human-specific layers) rather than
``"decoder_only"`` (only decoder frozen), since the KP encoder is already trained from warmstart.

Optimizer / gradient-accumulation / multi-GPU bookkeeping is identical to MuseDistillation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.muse_cotrain_distillation import _cross_modal_cosine_alignment
from rsl_rl.algorithms.muse_distillation import (
    MuseDistillation,
    _muse_cosine_smoothness_regularization,
    _muse_temporal_mu_regularization,
)


class MuseHumanKpDistillation(MuseDistillation):
    """MuseDistillation extended to co-train KP + human-motion modalities with cross-modal
    alignment, plus an optional warmup-freeze curriculum on the human-only layers.
    """

    def __init__(
        self,
        policy,
        *,
        weight_align: float = 0.1,
        weight_regularization_kp: float = 0.01,
        weight_regularization_human: float = 0.01,
        warmup_freeze_iters: int = 0,
        warmup_freeze_mode: str = "human_only",
        post_warmup_freeze_mode: str = "none",
        **kwargs,
    ):
        # The split per-modality smoothness weights below replace the legacy single
        # weight_regularization knob; never read by this class's update() path.
        kwargs.pop("weight_regularization", None)

        # Apply the warmup freeze mode BEFORE super().__init__ builds the optimizer, so the
        # optimizer locks in the warmup-phase param set.
        self.warmup_freeze_iters = int(warmup_freeze_iters)
        self.warmup_freeze_mode = str(warmup_freeze_mode)
        self.post_warmup_freeze_mode = str(post_warmup_freeze_mode)
        if self.warmup_freeze_iters > 0 and hasattr(policy, "set_freeze_mode"):
            if getattr(policy, "freeze_mode", None) != self.warmup_freeze_mode:
                policy.set_freeze_mode(self.warmup_freeze_mode)
                print(
                    f"[MuseHumanKpDistillation] Warmup phase active for {self.warmup_freeze_iters}"
                    f" iters: policy freeze_mode -> {self.warmup_freeze_mode!r}"
                )

        super().__init__(policy, **kwargs)

        self.weight_align = float(weight_align)
        self.weight_regularization_kp = float(weight_regularization_kp)
        self.weight_regularization_human = float(weight_regularization_human)

        # Per-pilot rollout counters (mirrors MuseCoTrainDistillation pattern). Layout:
        # 0 = KP pilot, 1 = human pilot. Eagerly allocated so flush() doesn't trip the
        # "inplace update to inference tensor" guard.
        try:
            _device = next(policy.parameters()).device
        except StopIteration:
            _device = torch.device("cpu")
        self._pilot_episode_end_counts: torch.Tensor = torch.zeros(2, device=_device, dtype=torch.long)
        self._pilot_failure_counts: torch.Tensor = torch.zeros(2, device=_device, dtype=torch.long)
        self._env_unwrapped = None  # plumbed in by the runner

        self._warmup_done = self.warmup_freeze_iters <= 0

    # ----- Warmup-freeze plumbing ----------------------------------------------------------

    def _rebuild_optimizer(self) -> None:
        """Recreate Adam over the newly-active ``policy.student.parameters()`` group."""
        self.optimizer = optim.Adam(self.policy.student.parameters(), lr=self.learning_rate)
        if self.use_adaptive_regularization and self.log_var_reg is not None:
            self.optimizer.add_param_group(
                {"params": [self.log_var_reg], "lr": self.learning_rate}
            )
        if self.use_adaptive_kl and self.log_var_kl is not None:
            self.optimizer.add_param_group(
                {"params": [self.log_var_kl], "lr": self.learning_rate}
            )

    # ----- Per-pilot success-rate tracking -------------------------------------------------

    def process_env_step(self, rewards, dones, infos) -> None:
        """Parent transition handling + per-pilot episode-end / failure accumulation."""
        is_human = getattr(self.policy, "_last_pilot_human_mask", None)
        env_u = self._env_unwrapped
        if is_human is not None and env_u is not None and hasattr(env_u, "reset_terminated"):
            device = self._pilot_episode_end_counts.device
            done_mask = dones.bool().view(-1).to(device=device)
            if done_mask.any():
                pilot_idx = is_human.to(device=device, dtype=torch.long)  # 0=kp, 1=human
                fail_mask = env_u.reset_terminated.bool().view(-1).to(device=device)
                self._pilot_episode_end_counts.index_add_(
                    0, pilot_idx, done_mask.to(dtype=torch.long)
                )
                self._pilot_failure_counts.index_add_(
                    0, pilot_idx, (done_mask & fail_mask).to(dtype=torch.long)
                )
        super().process_env_step(rewards, dones, infos)

    def flush_pilot_counts(self) -> dict[str, float]:
        """Return per-pilot success rate + episode-end / failure totals, then reset."""
        ends = self._pilot_episode_end_counts.detach().cpu()
        fails = self._pilot_failure_counts.detach().cpu()
        out: dict[str, float] = {}
        for pi, pilot in enumerate(("kp", "human")):
            e = int(ends[pi].item())
            f = int(fails[pi].item())
            if e == 0:
                continue
            out[f"pilot/{pilot}/episode_ends"] = float(e)
            out[f"pilot/{pilot}/failures"] = float(f)
            out[f"pilot/{pilot}/success_rate"] = 1.0 - (float(f) / float(e))
        self._pilot_episode_end_counts.zero_()
        self._pilot_failure_counts.zero_()
        return out

    # ----- Update -------------------------------------------------------------------------

    def update(self, current_iter: int | None = None) -> dict:
        del current_iter
        self.num_updates += 1

        mean_behavior_kp = 0.0
        mean_behavior_human = 0.0
        mean_reg_kp = 0.0
        mean_reg_human = 0.0
        mean_reg_kp_contribution = 0.0
        mean_reg_human_contribution = 0.0
        mean_align = 0.0
        mean_align_contribution = 0.0

        mu_norm_kp_sum = 0.0
        mu_norm_human_sum = 0.0
        mu_stats_count = 0

        cnt = 0
        total_loss_accumulator: torch.Tensor | None = None

        for _epoch in range(self.num_learning_epochs):
            prev_mu_kp: torch.Tensor | None = None
            prev_mu_human: torch.Tensor | None = None
            prev_dones: torch.Tensor | None = None
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()

            for _t, (obs, _, _, privileged_actions, dones) in enumerate(
                self.storage.generator()
            ):
                fwd = self.policy.forward_for_update(obs, sample_z=True)
                action_kp = fwd["action_kp"]
                action_human = fwd["action_human"]
                mu_kp = fwd["mu_kp"]
                mu_human = fwd["mu_human"]

                # 1. Behavior loss (both modalities vs same teacher target).
                behavior_kp = self.loss_fn(action_kp, privileged_actions)
                behavior_human = self.loss_fn(action_human, privileged_actions)
                mean_behavior_kp += behavior_kp.item()
                mean_behavior_human += behavior_human.item()
                total_loss = behavior_kp + behavior_human

                # μ-scale logging.
                with torch.no_grad():
                    mu_kp_flat = mu_kp.detach().reshape(-1, mu_kp.shape[-1])
                    mu_h_flat = mu_human.detach().reshape(-1, mu_human.shape[-1])
                    if mu_kp_flat.shape[0] > 1:
                        mu_norm_kp_sum += float(mu_kp_flat.norm(dim=-1).mean().item())
                        mu_norm_human_sum += float(mu_h_flat.norm(dim=-1).mean().item())
                        mu_stats_count += 1

                # 2. Per-modality temporal smoothness.
                smoothness_fn = (
                    _muse_cosine_smoothness_regularization
                    if self.smoothness_type == "cosine"
                    else _muse_temporal_mu_regularization
                )
                reg_kp_loss = torch.zeros((), device=obs.device, dtype=behavior_kp.dtype)
                reg_human_loss = torch.zeros((), device=obs.device, dtype=behavior_kp.dtype)
                reg_valid = False
                if (
                    prev_mu_kp is not None
                    and prev_mu_human is not None
                    and prev_dones is not None
                ):
                    reg_kp_loss, valid_kp = smoothness_fn(
                        mu_kp, prev_mu_kp, prev_dones, behavior_kp
                    )
                    reg_human_loss, valid_human = smoothness_fn(
                        mu_human, prev_mu_human, prev_dones, behavior_kp
                    )
                    reg_valid = valid_kp and valid_human

                reg_kp_contribution_value = 0.0
                reg_human_contribution_value = 0.0
                if self.use_adaptive_regularization:
                    if reg_valid:
                        s = self.log_var_reg.clamp(
                            self.regularization_log_var_min,
                            self.regularization_log_var_max,
                        )
                        eff_weight = torch.exp(-s)
                        # One Kendall scalar for the SUM of the two reg terms (same convention as
                        # MuseCoTrainDistillation).
                        adaptive_term = (
                            eff_weight * (reg_kp_loss + reg_human_loss)
                            + self.regularization_alpha * s
                        )
                        total_loss = total_loss + adaptive_term
                        reg_kp_contribution_value = float((eff_weight * reg_kp_loss).item())
                        reg_human_contribution_value = float(
                            (eff_weight * reg_human_loss).item()
                        )
                else:
                    if self.weight_regularization_kp > 0:
                        total_loss = total_loss + self.weight_regularization_kp * reg_kp_loss
                        reg_kp_contribution_value = float(
                            (self.weight_regularization_kp * reg_kp_loss).item()
                        )
                    if self.weight_regularization_human > 0:
                        total_loss = (
                            total_loss + self.weight_regularization_human * reg_human_loss
                        )
                        reg_human_contribution_value = float(
                            (self.weight_regularization_human * reg_human_loss).item()
                        )
                mean_reg_kp += reg_kp_loss.item()
                mean_reg_human += reg_human_loss.item()
                mean_reg_kp_contribution += reg_kp_contribution_value
                mean_reg_human_contribution += reg_human_contribution_value

                # 3. Cross-modal alignment.
                align_loss = _cross_modal_cosine_alignment(mu_kp, mu_human)
                align_contribution_value = 0.0
                if self.weight_align > 0:
                    total_loss = total_loss + self.weight_align * align_loss
                    align_contribution_value = float((self.weight_align * align_loss).item())
                mean_align += float(align_loss.item())
                mean_align_contribution += align_contribution_value

                if total_loss_accumulator is None:
                    total_loss_accumulator = total_loss
                else:
                    total_loss_accumulator = total_loss_accumulator + total_loss
                cnt += 1

                # Gradient step.
                just_optimized = False
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    total_loss_accumulator.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    nn.utils.clip_grad_norm_(
                        self.policy.student.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    total_loss_accumulator = None
                    just_optimized = True

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))
                if just_optimized:
                    prev_mu_kp = None
                    prev_mu_human = None
                    prev_dones = None
                else:
                    prev_mu_kp = mu_kp.detach()
                    prev_mu_human = mu_human.detach()
                    prev_dones = dones.detach()

        if total_loss_accumulator is not None:
            self.optimizer.zero_grad()
            total_loss_accumulator.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.policy.detach_hidden_states()

        if cnt > 0:
            mean_behavior_kp /= cnt
            mean_behavior_human /= cnt
            mean_reg_kp /= cnt
            mean_reg_human /= cnt
            mean_reg_kp_contribution /= cnt
            mean_reg_human_contribution /= cnt
            mean_align /= cnt
            mean_align_contribution /= cnt

        if mu_stats_count > 0:
            mu_norm_kp_mean = mu_norm_kp_sum / mu_stats_count
            mu_norm_human_mean = mu_norm_human_sum / mu_stats_count
        else:
            mu_norm_kp_mean = 0.0
            mu_norm_human_mean = 0.0

        if self.use_adaptive_regularization:
            log_var_value = float(self.log_var_reg.item())
            reg_effective_weight = float(
                torch.exp(
                    -self.log_var_reg.clamp(
                        self.regularization_log_var_min, self.regularization_log_var_max
                    )
                ).item()
            )
            reg_effective_weight_kp = reg_effective_weight
            reg_effective_weight_human = reg_effective_weight
        else:
            log_var_value = 0.0
            reg_effective_weight = float(self.weight_regularization_kp)
            reg_effective_weight_kp = float(self.weight_regularization_kp)
            reg_effective_weight_human = float(self.weight_regularization_human)

        mean_behavior_total = mean_behavior_kp + mean_behavior_human
        align_to_behavior_ratio = (
            mean_align_contribution / mean_behavior_total
            if mean_behavior_total > 1e-12 else 0.0
        )

        # Warmup-freeze transition.
        warmup_transition = 0.0
        if not self._warmup_done and self.num_updates >= self.warmup_freeze_iters:
            changed = self.policy.set_freeze_mode(self.post_warmup_freeze_mode)
            if changed:
                self._rebuild_optimizer()
                print(
                    f"[MuseHumanKpDistillation] Warmup done at update={self.num_updates}: "
                    f"freeze_mode {self.warmup_freeze_mode!r} -> "
                    f"{self.post_warmup_freeze_mode!r}; optimizer rebuilt."
                )
            self._warmup_done = True
            warmup_transition = 1.0

        # Log the active freeze mode as int for clean wandb charting.
        mode_to_int = {
            "decoder_plus_shared_encoder": 1,
            "decoder_only": 2,
            "none": 3,
            "human_only": 4,
        }
        freeze_mode_int = float(
            mode_to_int.get(getattr(self.policy, "freeze_mode", "none"), 0)
        )

        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        return {
            # Behavior (averaged for runner-side rolling-mean compatibility).
            "behavior": (mean_behavior_kp + mean_behavior_human) / 2.0,
            "behavior_kp": mean_behavior_kp,
            "behavior_human": mean_behavior_human,
            # Smoothness.
            "regularization": (mean_reg_kp + mean_reg_human) / 2.0,
            "reg_kp": mean_reg_kp,
            "reg_human": mean_reg_human,
            "reg_contribution": mean_reg_kp_contribution + mean_reg_human_contribution,
            "reg_kp_contribution": mean_reg_kp_contribution,
            "reg_human_contribution": mean_reg_human_contribution,
            "reg_log_var": log_var_value,
            "reg_effective_weight": reg_effective_weight,
            "reg_effective_weight_kp": reg_effective_weight_kp,
            "reg_effective_weight_human": reg_effective_weight_human,
            # Cross-modal alignment.
            "align": mean_align,
            "align_contribution": mean_align_contribution,
            "align_weight": self.weight_align,
            "align_to_behavior_ratio": align_to_behavior_ratio,
            # μ scale.
            "mu_norm_kp_mean": mu_norm_kp_mean,
            "mu_norm_human_mean": mu_norm_human_mean,
            # Warmup state.
            "warmup_freeze_transition": warmup_transition,
            "freeze_mode_int": freeze_mode_int,
        }
