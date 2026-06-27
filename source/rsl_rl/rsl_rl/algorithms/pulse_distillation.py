"""PULSE-style distillation : BC + KL(encoder || prior) + regularization.

Trains encoder E, decoder D, and prior R with:
- Action loss: student action vs teacher (PHC+) action
- KL loss: encoder distribution q(z|s^p,s^g) toward prior r(z|s^p) so encoder produces legitimate latents
- regularization: masked temporal smoothness on encoder means μ^e (in the loss if ``weight_regularization`` > 0)
- prior regularization (logging only): same statistic on prior means μ^p, not added to the optimization objective
"""

from __future__ import annotations

# torch
import math
import torch
import torch.nn as nn
import torch.optim as optim

# rsl-rl
from rsl_rl.storage import RolloutStorage

# Diagonal Gaussian KL uses log(sigma_p/sigma_e); clamp both stds so sigma_e cannot
# underflow (huge log ratio) when only one side was clamped in older code.
_KL_SIGMA_MIN = 1e-6


def _policy_latent_log_sigma_bounds(policy) -> tuple[float, float]:
    """Encoder/prior clamp bounds (log space) for logging; matches :class:`LatentBottleneckPULSE`."""
    lmin = getattr(policy, "latent_log_sigma_min", None)
    lmax = getattr(policy, "latent_log_sigma_max", None)
    if lmin is not None and lmax is not None:
        return float(lmin), float(lmax)
    return math.log(0.01), math.log(1.0)


def _kl_gaussian(
    mu_e: torch.Tensor,
    log_sigma_e: torch.Tensor,
    mu_p: torch.Tensor,
    log_sigma_p: torch.Tensor,
) -> torch.Tensor:
    """KL( N(mu_e, sigma_e) || N(mu_p, sigma_p) ). Sum over latent_dim, mean over batch.

    Closed form for factorized Gaussians (same as ``torch.distributions.kl_divergence``).
    """
    posterior_var = torch.exp(2.0 * log_sigma_e)
    prior_var = torch.exp(2.0 * log_sigma_p)
    squared_mean_diff = (mu_e - mu_p) ** 2
    kl = log_sigma_p - log_sigma_e
    kl += (posterior_var + squared_mean_diff) / (2.0 * prior_var)
    kl -= 0.5
    return kl.sum(dim=-1).mean()


def _mean_mu_e_mu_p_l2(mu_e: torch.Tensor, mu_p: torch.Tensor) -> torch.Tensor:
    """Mean per-sample Euclidean distance ||μ_e - μ_p||_2 (L2 over latent dims, mean over batch)."""
    return ((mu_e - mu_p) ** 2).sum(dim=-1).sqrt().mean()


def _pulse_temporal_mu_regularization(
    mu_t: torch.Tensor,
    mu_prev: torch.Tensor,
    prev_dones: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Masked mean of sum_latent (μ_t - μ_{t-1})^2; same reduction as encoder regu in ``PulseDistillation.update``."""
    mask = 1.0 - prev_dones.float()
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1)
    if mask.shape[0] != mu_t.shape[0]:
        return torch.zeros((), device=fallback.device, dtype=fallback.dtype)
    masked_count = mask.sum()
    if masked_count.item() <= 0:
        return torch.zeros((), device=fallback.device, dtype=fallback.dtype)
    diff = (mu_t - mu_prev) * mask
    return diff.pow(2).sum(dim=-1).sum() / masked_count


class PulseDistillation:
    """Distillation for PULSE: E, D, R updated with action loss + KL(encoder||prior) + regularization ."""

    def __init__(
        self,
        policy,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        loss_type: str = "mse",
        weight_kl: float = 0.01,
        weight_kl_end: float = 0.001,
        kl_anneal_start_iter: int = 0,
        kl_anneal_end_iter: int = 0,
        weight_regularization: float = 0.005,
        kl_loss_upper_bound: float | None = 10.0,
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
        self.weight_kl = weight_kl
        self.weight_kl_end = weight_kl_end
        self.kl_anneal_start_iter = kl_anneal_start_iter
        self.kl_anneal_end_iter = kl_anneal_end_iter
        self.weight_regularization = weight_regularization
        self.kl_loss_upper_bound = kl_loss_upper_bound

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

    def _kl_weight(self, current_iter: int | None) -> float:
        """Current β for L_KL; paper anneals from 0.01 to 0.001. Uses iter (not sample count)."""
        if self.kl_anneal_end_iter <= self.kl_anneal_start_iter:
            return self.weight_kl
        iter_val = current_iter if current_iter is not None else self.num_updates
        if iter_val <= self.kl_anneal_start_iter:
            return self.weight_kl
        if iter_val >= self.kl_anneal_end_iter:
            return self.weight_kl_end
        progress = (iter_val - self.kl_anneal_start_iter) / (self.kl_anneal_end_iter - self.kl_anneal_start_iter)
        return self.weight_kl + progress * (self.weight_kl_end - self.weight_kl)

    def update(self, current_iter: int | None = None) -> dict:
        self.num_updates += 1
        kl_weight = self._kl_weight(current_iter)
        log_sig_min, log_sig_max = _policy_latent_log_sigma_bounds(self.policy)
        mean_behavior_loss = 0.0
        mean_kl_loss = 0.0
        mean_mu_e_mu_p_l2 = 0.0
        mean_regularization_loss = 0.0
        mean_regularization_prior_loss = 0.0
        prior_std_sum = 0.0
        prior_std_count = 0
        prior_std_clamped_lower_count = 0
        prior_std_clamped_upper_count = 0
        encoder_std_sum = 0.0
        encoder_std_count = 0
        kl_loss_clamped_count = 0
        cnt = 0
        total_loss_accumulator: torch.Tensor | None = None

        for epoch in range(self.num_learning_epochs):
            prev_mu_e: torch.Tensor | None = None
            prev_mu_p: torch.Tensor | None = None
            prev_dones: torch.Tensor | None = None
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()

            for t, (obs, _, _, privileged_actions, dones) in enumerate(
                self.storage.generator()
            ):
                # Forward (Algo 1: encode -> sample z -> decode; prior from proprio)
                action, mu_e, log_sigma_e, mu_p, log_sigma_p = self.policy.forward_for_update(
                    obs, sample_z=True
                )
                # 1. Action loss: decoder output vs teacher action
                behavior_loss = self.loss_fn(action, privileged_actions)
                mean_behavior_loss += behavior_loss.item()

                # 2. KL(encoder || prior); β annealed from weight_kl to weight_kl_end over iters
                kl_loss = _kl_gaussian(mu_e, log_sigma_e, mu_p, log_sigma_p)
                mean_kl_loss += kl_loss.item()
                mean_mu_e_mu_p_l2 += _mean_mu_e_mu_p_l2(mu_e, mu_p).item()
                kl_for_loss = (
                    kl_loss
                    if self.kl_loss_upper_bound is None
                    else torch.clamp(kl_loss, max=self.kl_loss_upper_bound)
                )
                # metrics for logging:
                if self.kl_loss_upper_bound is not None and float(kl_loss.item()) > float(
                    self.kl_loss_upper_bound
                ):
                    kl_loss_clamped_count += 1
                sigma_p = torch.exp(log_sigma_p)
                prior_std_sum += float(sigma_p.sum().item())
                prior_std_count += int(sigma_p.numel())
                clamped_lower_mask = log_sigma_p <= log_sig_min
                clamped_upper_mask = log_sigma_p >= log_sig_max
                prior_std_clamped_lower_count += int(clamped_lower_mask.sum().item())
                prior_std_clamped_upper_count += int(clamped_upper_mask.sum().item())
                sigma_e = torch.exp(log_sigma_e)
                encoder_std_sum += float(sigma_e.sum().item())
                encoder_std_count += int(sigma_e.numel())

                # 3. L_regu = masked sum_latent (μ_t - μ_{t-1})^2 for encoder (in loss if weight > 0).
                #    Same statistic for prior μ^p is logged only (not in total_loss).
                total_loss = behavior_loss + kl_weight * kl_for_loss
                reg_loss = torch.zeros((), device=obs.device, dtype=behavior_loss.dtype)
                reg_loss_prior = torch.zeros((), device=obs.device, dtype=behavior_loss.dtype)
                if self.weight_regularization > 0 and prev_mu_e is not None and prev_dones is not None:
                    reg_loss = _pulse_temporal_mu_regularization(
                        mu_e, prev_mu_e, prev_dones, behavior_loss
                    )
                    assert prev_mu_p is not None
                    reg_loss_prior = _pulse_temporal_mu_regularization(
                        mu_p.detach(), prev_mu_p, prev_dones, behavior_loss
                    )

                total_loss = total_loss + self.weight_regularization * reg_loss
                mean_regularization_loss += reg_loss.item()
                mean_regularization_prior_loss += reg_loss_prior.item()

                if total_loss_accumulator is None:
                    total_loss_accumulator = total_loss
                else:
                    total_loss_accumulator = total_loss_accumulator + total_loss
                cnt += 1

                # Gradient step
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
                    # Skip one regularization pair after each optimizer step: otherwise ||μ_t - μ_{t-1}||^2
                    # pairs latents from the encoder before the weight update with latents after it,
                    # which spikes the loss and destabilizes training (periodic jitter every gradient_length).
                    just_optimized = True

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))
                if just_optimized:
                    prev_mu_e = None
                    prev_mu_p = None
                    prev_dones = None
                else:
                    prev_mu_e = mu_e.detach()
                    prev_mu_p = mu_p.detach()
                    prev_dones = dones.detach()

        # Backward any leftover accumulated loss (when batch size doesn't divide gradient_length)
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
            mean_kl_loss /= cnt
            mean_mu_e_mu_p_l2 /= cnt
            mean_regularization_loss /= cnt
            mean_regularization_prior_loss /= cnt
        prior_std_avg_magnitude = (
            prior_std_sum / prior_std_count if prior_std_count > 0 else 0.0
        )
        encoder_std_avg_magnitude = (
            encoder_std_sum / encoder_std_count if encoder_std_count > 0 else 0.0
        )
        prior_std_clamp_lower_rate = (
            prior_std_clamped_lower_count / prior_std_count
            if prior_std_count > 0
            else 0.0
        )
        prior_std_clamp_upper_rate = (
            prior_std_clamped_upper_count / prior_std_count
            if prior_std_count > 0
            else 0.0
        )
        kl_loss_clamp_rate = (
            kl_loss_clamped_count / cnt
            if cnt > 0 and self.kl_loss_upper_bound is not None
            else 0.0
        )

        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        return {
            "behavior": mean_behavior_loss,
            "kl": mean_kl_loss,
            "mu_e_mu_p_l2": mean_mu_e_mu_p_l2,
            "kl_weight": kl_weight,
            "kl_loss_upper_bound": self.kl_loss_upper_bound,
            "kl_loss_clamp_rate": kl_loss_clamp_rate,
            "regularization": mean_regularization_loss,
            "regularization_prior": mean_regularization_prior_loss,
            "prior_std_avg_magnitude": prior_std_avg_magnitude,
            "encoder_std_avg_magnitude": encoder_std_avg_magnitude,
            "prior_std_clamp_lower_rate": prior_std_clamp_lower_rate,
            "prior_std_clamp_upper_rate": prior_std_clamp_upper_rate,
        }

    def broadcast_parameters(self) -> None:
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        grads = [
            param.grad.view(-1)
            for param in self.policy.parameters()
            if param.grad is not None
        ]
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


class PriorOnlyPulseDistillation(PulseDistillation):
    """Prior-only PULSE: train only R(s^p) to match encoder latents; E and D are not optimized.

    Minimizes KL(q_encoder || p_prior) with respect to prior parameters only (encoder outputs
    detached). Uses the same rollout buffer as :class:`PulseDistillation`; replaces the joint
    optimizer with Adam over ``policy.prior`` only.
    """

    def __init__(
        self,
        policy,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        loss_type: str = "mse",
        weight_kl: float = 0.01,
        weight_kl_end: float = 0.001,
        kl_anneal_start_iter: int = 0,
        kl_anneal_end_iter: int = 0,
        weight_regularization: float = 0.005,
        weight_prior_fit: float = 1.0,
        num_prior_fit_epochs: int = 1,
        prior_fit_learning_rate: float | None = None,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            policy,
            num_learning_epochs=num_learning_epochs,
            gradient_length=gradient_length,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            loss_type=loss_type,
            weight_kl=weight_kl,
            weight_kl_end=weight_kl_end,
            kl_anneal_start_iter=kl_anneal_start_iter,
            kl_anneal_end_iter=kl_anneal_end_iter,
            weight_regularization=weight_regularization,
            device=device,
            multi_gpu_cfg=multi_gpu_cfg,
            **kwargs,
        )
        self.weight_prior_fit = float(weight_prior_fit)
        self.num_prior_fit_epochs = max(1, int(num_prior_fit_epochs))
        self.prior_fit_learning_rate = (
            float(prior_fit_learning_rate)
            if prior_fit_learning_rate is not None
            else float(self.learning_rate)
        )
        self.optimizer = optim.Adam(
            self.policy.prior.parameters(), lr=self.prior_fit_learning_rate
        )

    def update_prior(
        self,
        current_iter: int | None = None,
        *,
        clear_storage: bool = True,
    ) -> dict:
        """Update only the prior toward the encoder: KL(q_e || p) w.r.t. prior; encoder detached."""
        mean_kl_prior = 0.0
        mean_mu_e_mu_p_l2 = 0.0
        log_sig_min, log_sig_max = _policy_latent_log_sigma_bounds(self.policy)
        prior_std_sum = 0.0
        prior_std_count = 0
        prior_std_clamped_lower_count = 0
        prior_std_clamped_upper_count = 0
        encoder_std_sum = 0.0
        encoder_std_count = 0
        kl_loss_clamped_count = 0
        cnt = 0
        total_loss_accumulator: torch.Tensor | None = None

        for _epoch in range(self.num_prior_fit_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()

            for _t, (obs, _, _, _privileged_actions, dones) in enumerate(
                self.storage.generator()
            ):
                _, mu_e, log_sigma_e, mu_p, log_sigma_p = self.policy.forward_for_update(
                    obs, sample_z=False
                )
                kl_prior = _kl_gaussian(
                    mu_e.detach(),
                    log_sigma_e.detach(),
                    mu_p,
                    log_sigma_p,
                )
                mean_kl_prior += kl_prior.item()
                mean_mu_e_mu_p_l2 += _mean_mu_e_mu_p_l2(mu_e.detach(), mu_p).item()
                kl_prior_for_loss = (
                    kl_prior
                    if self.kl_loss_upper_bound is None
                    else torch.clamp(kl_prior, max=self.kl_loss_upper_bound)
                )
                if self.kl_loss_upper_bound is not None and float(kl_prior.item()) > float(
                    self.kl_loss_upper_bound
                ):
                    kl_loss_clamped_count += 1
                sigma_p = torch.exp(log_sigma_p)
                prior_std_sum += float(sigma_p.sum().item())
                prior_std_count += int(sigma_p.numel())
                clamped_lower_mask = log_sigma_p <= log_sig_min
                clamped_upper_mask = log_sigma_p >= log_sig_max
                prior_std_clamped_lower_count += int(clamped_lower_mask.sum().item())
                prior_std_clamped_upper_count += int(clamped_upper_mask.sum().item())
                sigma_e = torch.exp(log_sigma_e)
                encoder_std_sum += float(sigma_e.sum().item())
                encoder_std_count += int(sigma_e.numel())
                total_loss = self.weight_prior_fit * kl_prior_for_loss
                cnt += 1

                if total_loss_accumulator is None:
                    total_loss_accumulator = total_loss
                else:
                    total_loss_accumulator = total_loss_accumulator + total_loss

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    total_loss_accumulator.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    nn.utils.clip_grad_norm_(self.policy.prior.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    total_loss_accumulator = None

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        if total_loss_accumulator is not None:
            self.optimizer.zero_grad()
            total_loss_accumulator.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.prior.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.policy.detach_hidden_states()

        if cnt > 0:
            mean_kl_prior /= cnt
            mean_mu_e_mu_p_l2 /= cnt
        prior_std_avg_magnitude = (
            prior_std_sum / prior_std_count if prior_std_count > 0 else 0.0
        )
        encoder_std_avg_magnitude = (
            encoder_std_sum / encoder_std_count if encoder_std_count > 0 else 0.0
        )
        prior_std_clamp_lower_rate = (
            prior_std_clamped_lower_count / prior_std_count
            if prior_std_count > 0
            else 0.0
        )
        prior_std_clamp_upper_rate = (
            prior_std_clamped_upper_count / prior_std_count
            if prior_std_count > 0
            else 0.0
        )
        kl_loss_clamp_rate = (
            kl_loss_clamped_count / cnt
            if cnt > 0 and self.kl_loss_upper_bound is not None
            else 0.0
        )

        if clear_storage:
            self.storage.clear()
            self.last_hidden_states = self.policy.get_hidden_states()
            self.policy.detach_hidden_states()

        return {
            "kl_prior_fit": mean_kl_prior,
            "mu_e_mu_p_l2": mean_mu_e_mu_p_l2,
            "weight_prior_fit": self.weight_prior_fit,
            "kl_loss_upper_bound": self.kl_loss_upper_bound,
            "kl_loss_clamp_rate": kl_loss_clamp_rate,
            "prior_std_avg_magnitude": prior_std_avg_magnitude,
            "encoder_std_avg_magnitude": encoder_std_avg_magnitude,
            "prior_std_clamp_lower_rate": prior_std_clamp_lower_rate,
            "prior_std_clamp_upper_rate": prior_std_clamp_upper_rate,
        }

    def update(self, current_iter: int | None = None) -> dict:
        self.num_updates += 1
        return self.update_prior(current_iter=current_iter, clear_storage=True)
