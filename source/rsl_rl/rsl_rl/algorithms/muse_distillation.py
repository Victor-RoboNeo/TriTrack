"""MUSE distillation: BC + temporal latent regularization. No KL term (no separate prior).

Trains encoder E and decoder D with:
- Action loss: student action vs teacher (PHC+) action.
- Regularization: masked temporal smoothness on encoder means μ^e — same statistic as in PULSE.

The regularization can be weighted three ways:
  (a) Adaptive on  (``use_adaptive_regularization=True``): Kendall-style ``exp(-s)·reg + α·s``,
      with ``s`` learned. Reg's contribution to total loss equilibrates near α regardless of raw
      L_reg scale; behavior loss keeps weight 1, so encoder/decoder gradient scale is unchanged
      from the unweighted recipe.
  (b) Adaptive off, ``weight_regularization > 0``: legacy fixed-weight ``w·reg`` term.
  (c) Adaptive off, ``weight_regularization == 0``: diagnostic-only (term computed and logged but
      contributes no gradient).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.storage import RolloutStorage


def _policy_latent_log_sigma_bounds(policy) -> tuple[float, float]:
    """Encoder log-σ clamp bounds (for logging clamp rates)."""
    lmin = getattr(policy, "latent_log_sigma_min", None)
    lmax = getattr(policy, "latent_log_sigma_max", None)
    if lmin is not None and lmax is not None:
        return float(lmin), float(lmax)
    return math.log(0.001), math.log(1.0)


def _muse_kl_against_standard_normal(
    mu: torch.Tensor, log_sigma: torch.Tensor
) -> torch.Tensor:
    """Closed-form KL(N(μ, diag(σ²)) || N(0, I)), reduced to a scalar (mean over batch, sum over dims).

    Per-element: 0.5 * (μ² + σ² − 1 − 2·log σ). Minimum 0 at μ=0, σ=1. Bounded below ≥ 0; the −2·log σ
    term diverges as σ → 0 and σ² diverges as σ → ∞, so KL anchors both ends. With Kendall weighting,
    this gives a stable equilibrium near the prior while still permitting per-input variation.
    """
    sigma_sq = torch.exp(2.0 * log_sigma)
    per_elem = 0.5 * (mu.pow(2) + sigma_sq - 1.0 - 2.0 * log_sigma)
    # Sum over latent dim (keep the natural "nats per sample" unit), then mean over batch.
    return per_elem.sum(dim=-1).mean()


def _muse_temporal_mu_regularization(
    mu_t: torch.Tensor,
    mu_prev: torch.Tensor,
    prev_dones: torch.Tensor,
    fallback: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Masked mean of sum_latent (μ_t - μ_{t-1})^2; mirrors PULSE's encoder-regu reduction.

    NOTE: Absolute-delta smoothness is NOT scale-invariant. The smoothing gradient pulls μ_t→μ_{t-1},
    forcing the encoder to push harder per step to discriminate goals, which grows ‖μ‖. With Kendall
    on top, the runaway is self-reinforcing (raw reg ↑ → eff weight ↓ → constraint disengages →
    BC amplifies → reg ↑). Use ``_muse_cosine_smoothness_regularization`` for the scale-invariant
    alternative.

    Returns (loss, valid). ``valid=False`` means placeholder zero (shape mismatch or every env reset
    this step) — caller MUST NOT add the Kendall ``+α·s`` term on top without a real loss to balance.
    """
    mask = 1.0 - prev_dones.float()
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1)
    if mask.shape[0] != mu_t.shape[0]:
        return torch.zeros((), device=fallback.device, dtype=fallback.dtype), False
    masked_count = mask.sum()
    if masked_count.item() <= 0:
        return torch.zeros((), device=fallback.device, dtype=fallback.dtype), False
    diff = (mu_t - mu_prev) * mask
    return diff.pow(2).sum(dim=-1).sum() / masked_count, True


def _muse_cosine_smoothness_regularization(
    mu_t: torch.Tensor,
    mu_prev: torch.Tensor,
    prev_dones: torch.Tensor,
    fallback: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, bool]:
    """Scale-invariant smoothness: L = (1 - cos(μ_t, μ_{t-1})), masked-mean over valid pairs.

    Range [0, 2]. Penalizes directional change of the latent trajectory while leaving scale free.
    Fixes the absolute-delta failure mode where smoothing pressure incentivizes the encoder to grow
    ‖μ‖ to amplify per-step BC signal. Bounded ⇒ Kendall has a well-defined equilibrium that can't
    run away.
    """
    mask = 1.0 - prev_dones.float()
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1)
    if mask.shape[0] != mu_t.shape[0]:
        return torch.zeros((), device=fallback.device, dtype=fallback.dtype), False
    masked_count = mask.sum()
    if masked_count.item() <= 0:
        return torch.zeros((), device=fallback.device, dtype=fallback.dtype), False
    cos_sim = nn.functional.cosine_similarity(mu_t, mu_prev, dim=-1, eps=eps)  # [B]
    loss_per_sample = (1.0 - cos_sim).unsqueeze(-1)  # [B, 1]
    loss = (loss_per_sample * mask).sum() / masked_count
    return loss, True


class MuseDistillation:
    """Distillation for MUSE: E and D updated with action loss + temporal latent regularization."""

    def __init__(
        self,
        policy,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        loss_type: str = "mse",
        weight_regularization: float = 0.005,
        use_adaptive_regularization: bool = False,
        regularization_alpha: float = 0.01,
        smoothness_type: str = "l2",
        regularization_log_var_init: float = 3.0,
        regularization_log_var_min: float = -3.0,
        regularization_log_var_max: float = 8.0,
        weight_kl: float = 0.0,
        use_adaptive_kl: bool = False,
        kl_alpha: float = 0.01,
        kl_log_var_init: float = 3.0,
        kl_log_var_min: float = -3.0,
        kl_log_var_max: float = 4.0,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None

        self.policy = policy
        self.policy.to(self.device)
        self.storage: RolloutStorage | None = None
        self.optimizer = optim.Adam(self.policy.student.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None

        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.weight_regularization = weight_regularization

        # Smoothness term selector. "l2" = absolute-delta `‖μ_t-μ_{t-1}‖²` (legacy, NOT
        # scale-invariant — incentivizes the encoder to grow ‖μ‖ to amplify per-step BC signal).
        # "cosine" = `1 - cos(μ_t, μ_{t-1})` (scale-invariant, bounded in [0, 2], well-behaved
        # under Kendall). Use cosine unless you have a reason not to.
        if smoothness_type not in ("l2", "cosine"):
            raise ValueError(f"smoothness_type must be 'l2' or 'cosine', got {smoothness_type!r}")
        self.smoothness_type = str(smoothness_type)

        # Adaptive (Kendall-style) weighting for the temporal latent regularization. When enabled,
        # the contribution of reg to total loss equilibrates at α (regardless of raw L_reg scale),
        # and the behavior loss keeps weight 1 — gradient scale on the encoder/decoder is
        # unchanged from the unweighted MUSE recipe.
        self.use_adaptive_regularization = bool(use_adaptive_regularization)
        self.regularization_alpha = float(regularization_alpha)
        self.regularization_log_var_min = float(regularization_log_var_min)
        self.regularization_log_var_max = float(regularization_log_var_max)
        if self.use_adaptive_regularization:
            self.log_var_reg = nn.Parameter(
                torch.tensor(float(regularization_log_var_init), device=self.device)
            )
            # Separate param group so we can give it a different LR later without touching
            # encoder/decoder LR. Sync across ranks is handled in reduce_parameters().
            self.optimizer.add_param_group({"params": [self.log_var_reg], "lr": learning_rate})
        else:
            self.log_var_reg = None

        # Kendall-style adaptive weighting for the KL-against-N(0, I) term. The mechanism that
        # broke Kendall on temporal-reg (μ-drift loop) does not apply here: KL's gradient direction
        # is always corrective (toward μ=0, σ=1), so even when Kendall lowers the effective weight,
        # the encoder is still being pulled to a fixed equilibrium. The two adaptive terms cooperate
        # — KL anchors absolute scale, temporal-reg anchors smoothness; both are satisfied at
        # μ = const = 0 with σ ≈ 1.
        self.weight_kl = float(weight_kl)
        self.use_adaptive_kl = bool(use_adaptive_kl)
        self.kl_alpha = float(kl_alpha)
        self.kl_log_var_min = float(kl_log_var_min)
        self.kl_log_var_max = float(kl_log_var_max)
        if self.use_adaptive_kl:
            self.log_var_kl = nn.Parameter(
                torch.tensor(float(kl_log_var_init), device=self.device)
            )
            self.optimizer.add_param_group({"params": [self.log_var_kl], "lr": learning_rate})
        else:
            self.log_var_kl = None

        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported: mse, huber")

        self.num_updates = 0

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        student_obs_shape: list,
        teacher_obs_shape: list,
        actions_shape: list,
    ) -> None:
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
        )

    def act(self, obs: torch.Tensor, teacher_obs: torch.Tensor) -> torch.Tensor:
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(teacher_obs).detach()
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        return self.transition.actions

    def process_env_step(
        self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict
    ) -> None:
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self, current_iter: int | None = None) -> dict:
        del current_iter  # MUSE has no KL annealing or iteration-dependent terms.
        self.num_updates += 1
        log_sig_min, log_sig_max = _policy_latent_log_sigma_bounds(self.policy)
        mean_behavior_loss = 0.0
        mean_regularization_loss = 0.0
        mean_reg_contribution = 0.0
        mean_kl_loss = 0.0
        mean_kl_contribution = 0.0
        encoder_std_sum = 0.0
        encoder_std_count = 0
        encoder_std_clamped_lower_count = 0
        encoder_std_clamped_upper_count = 0
        # Encoder μ scale diagnostics. ``mu_norm_sum`` tracks absolute scale (does ‖μ‖ drift over
        # training?); ``mu_per_dim_std_*`` tracks anisotropy across latent dims (max/min ratio
        # close to 1 ⇒ isotropic; >> 1 ⇒ some dims dominate).
        mu_norm_sum = 0.0
        mu_per_dim_std_sum = 0.0
        mu_per_dim_std_max_sum = 0.0
        mu_per_dim_std_min_sum = 0.0
        mu_stats_count = 0
        cnt = 0
        total_loss_accumulator: torch.Tensor | None = None

        for _epoch in range(self.num_learning_epochs):
            prev_mu_e: torch.Tensor | None = None
            prev_dones: torch.Tensor | None = None
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()

            for _t, (obs, privileged_obs, _, privileged_actions, dones) in enumerate(
                self.storage.generator()
            ):
                # Forward: encoder -> sample z -> decoder.
                action, mu_e, log_sigma_e = self.policy.forward_for_update(obs, sample_z=True)

                # 1. Action loss: student action vs teacher action.
                behavior_loss = self.loss_fn(action, privileged_actions)
                mean_behavior_loss += behavior_loss.item()

                # Encoder σ statistics (logging only).
                sigma_e = torch.exp(log_sigma_e)
                encoder_std_sum += float(sigma_e.sum().item())
                encoder_std_count += int(sigma_e.numel())
                encoder_std_clamped_lower_count += int((log_sigma_e <= log_sig_min).sum().item())
                encoder_std_clamped_upper_count += int((log_sigma_e >= log_sig_max).sum().item())

                # Encoder μ scale diagnostics (logging only). All ops on detached μ to avoid
                # extra graph nodes; computed per-step and averaged across the update.
                with torch.no_grad():
                    mu_flat = mu_e.detach().reshape(-1, mu_e.shape[-1])  # [B*, latent_dim]
                    if mu_flat.shape[0] > 1:
                        mu_norm_sum += float(mu_flat.norm(dim=-1).mean().item())
                        per_dim_std = mu_flat.std(dim=0)  # [latent_dim]
                        mu_per_dim_std_sum += float(per_dim_std.mean().item())
                        mu_per_dim_std_max_sum += float(per_dim_std.max().item())
                        mu_per_dim_std_min_sum += float(per_dim_std.min().item())
                        mu_stats_count += 1

                # 2. Temporal latent regularization. Three modes:
                #    (a) adaptive on   → exp(-s)·reg + α·s, both terms gated on reg_valid.
                #    (b) adaptive off, weight_regularization > 0 → fixed-weight term (legacy).
                #    (c) adaptive off, weight_regularization == 0 → diagnostic-only.
                # ``reg_valid=False`` means reg_loss is a placeholder zero (no prev step or all envs
                # reset). Gating the adaptive ``+α·s`` term on validity is load-bearing — without
                # signal to balance against, that term would pull s monotonically and the effective
                # weight would run away.
                total_loss = behavior_loss
                reg_loss = torch.zeros((), device=obs.device, dtype=behavior_loss.dtype)
                reg_valid = False
                if prev_mu_e is not None and prev_dones is not None:
                    smoothness_fn = (
                        _muse_cosine_smoothness_regularization
                        if self.smoothness_type == "cosine"
                        else _muse_temporal_mu_regularization
                    )
                    reg_loss, reg_valid = smoothness_fn(
                        mu_e, prev_mu_e, prev_dones, behavior_loss
                    )

                reg_contribution_value = 0.0
                if self.use_adaptive_regularization:
                    if reg_valid:
                        s = self.log_var_reg.clamp(
                            self.regularization_log_var_min, self.regularization_log_var_max
                        )
                        eff_weight = torch.exp(-s)
                        adaptive_term = eff_weight * reg_loss + self.regularization_alpha * s
                        total_loss = total_loss + adaptive_term
                        reg_contribution_value = float((eff_weight * reg_loss).item())
                elif self.weight_regularization > 0:
                    total_loss = total_loss + self.weight_regularization * reg_loss
                    reg_contribution_value = float((self.weight_regularization * reg_loss).item())

                mean_regularization_loss += reg_loss.item()
                mean_reg_contribution += reg_contribution_value

                # 3. KL-against-N(0, I) (closed-form). Anchors μ at 0 and σ at 1 — addresses the
                # μ-norm drift that temporal-reg-alone can't (temporal-reg constrains derivatives,
                # not magnitudes). Three modes mirror the reg term:
                #    (a) adaptive on   → exp(-s_KL)·KL + α_KL·s_KL.
                #    (b) adaptive off, weight_kl > 0 → fixed-weight β·KL.
                #    (c) adaptive off, weight_kl == 0 → diagnostic-only (no gradient).
                kl_loss = _muse_kl_against_standard_normal(mu_e, log_sigma_e)
                kl_contribution_value = 0.0
                if self.use_adaptive_kl:
                    s_kl = self.log_var_kl.clamp(self.kl_log_var_min, self.kl_log_var_max)
                    eff_w_kl = torch.exp(-s_kl)
                    total_loss = total_loss + eff_w_kl * kl_loss + self.kl_alpha * s_kl
                    kl_contribution_value = float((eff_w_kl * kl_loss).item())
                elif self.weight_kl > 0:
                    total_loss = total_loss + self.weight_kl * kl_loss
                    kl_contribution_value = float((self.weight_kl * kl_loss).item())

                mean_kl_loss += float(kl_loss.item())
                mean_kl_contribution += kl_contribution_value

                # 4. Subclass-injected extra loss (e.g. aux anchor prediction in MuseKpDistillation).
                # Default :meth:`_extra_loss_term` returns a zero tensor — no effect on this class.
                extra_loss = self._extra_loss_term(obs, privileged_obs)
                if extra_loss is not None:
                    total_loss = total_loss + extra_loss

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
                    nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    total_loss_accumulator = None
                    # Skip one regularization pair after each optimizer step: ||μ_t - μ_{t-1}||^2
                    # would otherwise pair latents from before/after the weight update.
                    just_optimized = True

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))
                if just_optimized:
                    prev_mu_e = None
                    prev_dones = None
                else:
                    prev_mu_e = mu_e.detach()
                    prev_dones = dones.detach()

        # Backward any leftover accumulated loss.
        if total_loss_accumulator is not None:
            self.optimizer.zero_grad()
            total_loss_accumulator.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.policy.detach_hidden_states()

        if cnt > 0:
            mean_behavior_loss /= cnt
            mean_regularization_loss /= cnt
            mean_reg_contribution /= cnt
            mean_kl_loss /= cnt
            mean_kl_contribution /= cnt
        encoder_std_avg_magnitude = (
            encoder_std_sum / encoder_std_count if encoder_std_count > 0 else 0.0
        )
        encoder_std_clamp_lower_rate = (
            encoder_std_clamped_lower_count / encoder_std_count
            if encoder_std_count > 0 else 0.0
        )
        encoder_std_clamp_upper_rate = (
            encoder_std_clamped_upper_count / encoder_std_count
            if encoder_std_count > 0 else 0.0
        )

        if mu_stats_count > 0:
            mu_norm_mean = mu_norm_sum / mu_stats_count
            mu_per_dim_std_mean = mu_per_dim_std_sum / mu_stats_count
            mu_per_dim_std_max = mu_per_dim_std_max_sum / mu_stats_count
            mu_per_dim_std_min = mu_per_dim_std_min_sum / mu_stats_count
            mu_per_dim_std_ratio = (
                mu_per_dim_std_max / mu_per_dim_std_min
                if mu_per_dim_std_min > 1e-8 else float("inf")
            )
        else:
            mu_norm_mean = 0.0
            mu_per_dim_std_mean = 0.0
            mu_per_dim_std_max = 0.0
            mu_per_dim_std_min = 0.0
            mu_per_dim_std_ratio = 0.0

        # Reg-vs-behavior diagnostics. ``reg_contribution`` is what reg adds to total loss (the
        # actual gradient pressure on encoder/decoder weights); ``reg_to_behavior_ratio`` answers
        # "how loud is reg relative to behavior" directly. Under adaptive weighting, the ratio
        # should hover near α / mean_behavior_loss.
        reg_to_behavior_ratio = (
            mean_reg_contribution / mean_behavior_loss if mean_behavior_loss > 1e-12 else 0.0
        )
        if self.use_adaptive_regularization:
            log_var_value = float(self.log_var_reg.item())
            reg_effective_weight = float(
                torch.exp(
                    -self.log_var_reg.clamp(
                        self.regularization_log_var_min, self.regularization_log_var_max
                    )
                ).item()
            )
        else:
            log_var_value = 0.0
            reg_effective_weight = float(self.weight_regularization)

        if self.use_adaptive_kl:
            kl_log_var_value = float(self.log_var_kl.item())
            kl_effective_weight = float(
                torch.exp(
                    -self.log_var_kl.clamp(self.kl_log_var_min, self.kl_log_var_max)
                ).item()
            )
        else:
            kl_log_var_value = 0.0
            kl_effective_weight = float(self.weight_kl)
        kl_to_behavior_ratio = (
            mean_kl_contribution / mean_behavior_loss if mean_behavior_loss > 1e-12 else 0.0
        )

        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        log_dict = {
            "behavior": mean_behavior_loss,
            "regularization": mean_regularization_loss,
            "reg_contribution": mean_reg_contribution,
            "reg_to_behavior_ratio": reg_to_behavior_ratio,
            "reg_log_var": log_var_value,
            "reg_effective_weight": reg_effective_weight,
            "kl": mean_kl_loss,
            "kl_contribution": mean_kl_contribution,
            "kl_to_behavior_ratio": kl_to_behavior_ratio,
            "kl_log_var": kl_log_var_value,
            "kl_effective_weight": kl_effective_weight,
            "encoder_std_avg_magnitude": encoder_std_avg_magnitude,
            "encoder_std_clamp_lower_rate": encoder_std_clamp_lower_rate,
            "encoder_std_clamp_upper_rate": encoder_std_clamp_upper_rate,
            "mu_norm_mean": mu_norm_mean,
            "mu_per_dim_std_mean": mu_per_dim_std_mean,
            "mu_per_dim_std_max": mu_per_dim_std_max,
            "mu_per_dim_std_min": mu_per_dim_std_min,
            "mu_per_dim_std_ratio": mu_per_dim_std_ratio,
        }
        log_dict.update(self._extra_log_dict())
        return log_dict

    def _extra_loss_term(
        self, obs: torch.Tensor, privileged_obs: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Hook for subclasses to add extra loss terms to ``total_loss`` inside the update loop.

        Default returns ``None`` (no extra loss). Subclasses return a scalar tensor; it is added to
        ``total_loss`` before accumulation. Use this to inject auxiliary heads (e.g.
        :class:`MuseKpDistillation`'s anchor predictor) without copying the full update logic.

        Diagnostics for the extra loss should be cached on ``self`` by the subclass and merged into
        the return dict via :meth:`_extra_log_dict`.
        """
        del obs, privileged_obs
        return None

    def _extra_log_dict(self) -> dict:
        """Hook for subclasses to add per-update diagnostics to the ``update()`` return dict."""
        return {}

    def broadcast_parameters(self) -> None:
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        # log_var_reg / log_var_kl live on the algorithm (not the policy), so the
        # policy.parameters() iteration would miss them and ranks would drift. Splice them into
        # the all-reduce explicitly (in a fixed order so rank-0 and rank-k unpack identically).
        reg_grad_present = (
            self.use_adaptive_regularization
            and self.log_var_reg is not None
            and self.log_var_reg.grad is not None
        )
        kl_grad_present = (
            self.use_adaptive_kl
            and self.log_var_kl is not None
            and self.log_var_kl.grad is not None
        )
        grads = [
            param.grad.view(-1)
            for param in self.policy.parameters()
            if param.grad is not None
        ]
        if reg_grad_present:
            grads.append(self.log_var_reg.grad.view(-1))
        if kl_grad_present:
            grads.append(self.log_var_kl.grad.view(-1))
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(
                    all_grads[offset : offset + numel].view_as(param.grad.data)
                )
                offset += numel
        if reg_grad_present:
            numel = self.log_var_reg.numel()
            self.log_var_reg.grad.data.copy_(
                all_grads[offset : offset + numel].view_as(self.log_var_reg.grad.data)
            )
            offset += numel
        if kl_grad_present:
            numel = self.log_var_kl.numel()
            self.log_var_kl.grad.data.copy_(
                all_grads[offset : offset + numel].view_as(self.log_var_kl.grad.data)
            )
            offset += numel
