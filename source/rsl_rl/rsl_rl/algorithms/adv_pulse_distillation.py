"""PULSE distillation plus a diagnostic action discriminator (encoder vs prior decode).

The discriminator outputs **logits**; training uses ``binary_cross_entropy_with_logits``, which applies
sigmoid internally—so the MLP does **not** need a 0–1 output (that would double-sigmoid and hurt training).
For metrics, interpret probability of class "prior" as ``sigmoid(logit)``.

**Isolation from E/D/R:** The student optimizer only steps ``policy.student`` (encoder, decoder, prior).
``action_e`` / ``action_p`` fed to the discriminator are ``.detach()``'d, and ``total_loss`` never includes
the discriminator loss—so PULSE parameters receive no gradients from D (pure diagnostic signal).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rsl_rl.algorithms.pulse_distillation import (
    PulseDistillation,
    _kl_gaussian,
    _mean_mu_e_mu_p_l2,
    _policy_latent_log_sigma_bounds,
    _pulse_temporal_mu_regularization,
)


class AdvPulseDistillation(PulseDistillation):
    """Same student objective as :class:`PulseDistillation`; D trains on detached actions with a separate optimizer."""

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
        disc_lr: float = 1e-3,
        disc_update_interval: int = 1,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "AdvPulseDistillation.__init__ got unexpected arguments (ignored): "
                + str(list(kwargs.keys()))
            )
        self.disc_lr = float(disc_lr)
        self.disc_update_interval = max(1, int(disc_update_interval))
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
            kl_loss_upper_bound=kl_loss_upper_bound,
            device=device,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        if not hasattr(self.policy, "action_discriminator"):
            raise TypeError(
                "AdvPulseDistillation requires a policy with action_discriminator "
                f"(e.g. LatentBottleneckPULSEAdv), got {type(self.policy).__name__}."
            )
        self.disc_optimizer = optim.Adam(
            self.policy.action_discriminator.parameters(),
            lr=self.disc_lr,
        )

    def _reduce_grads_for_params(self, params: list[torch.nn.Parameter]) -> None:
        grads = [p.grad.view(-1) for p in params if p.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for p in params:
            if p.grad is not None:
                n = p.numel()
                p.grad.data.copy_(all_grads[offset : offset + n].view_as(p.grad.data))
                offset += n

    def _reduce_student_grads(self) -> None:
        student_params = list(self.policy.student.parameters())
        self._reduce_grads_for_params(student_params)

    def _reduce_disc_grads(self) -> None:
        disc_params = list(self.policy.action_discriminator.parameters())
        self._reduce_grads_for_params(disc_params)

    def _discriminator_step(
        self,
        obs: torch.Tensor,
        action_e: torch.Tensor,
        action_p: torch.Tensor,
    ) -> tuple[float, float, float, float, float]:
        """Train D on detached actions. Returns loss, acc, logit_gap, prob misclass encoder, prob misclass prior."""
        proprio = self.policy.student_core.get_proprio(obs).detach()
        ae = action_e.detach()
        ap = action_p.detach()
        disc = self.policy.action_discriminator
        logit_e = disc(ae, proprio)
        logit_p = disc(ap, proprio)
        tgt_e = torch.zeros_like(logit_e)
        tgt_p = torch.ones_like(logit_p)
        loss_e = F.binary_cross_entropy_with_logits(logit_e, tgt_e)
        loss_p = F.binary_cross_entropy_with_logits(logit_p, tgt_p)
        loss_d = 0.5 * (loss_e + loss_p)

        with torch.no_grad():
            acc_e = (logit_e < 0.0).float().mean().item()
            acc_p = (logit_p > 0.0).float().mean().item()
            acc = 0.5 * (acc_e + acc_p)
            logit_gap = (logit_e.mean() - logit_p.mean()).item()
            # P(D predicts "prior" | encoder sample) — wrong if > 0.5
            prob_misclass_encoder = torch.sigmoid(logit_e).mean().item()
            # P(D predicts "encoder" | prior sample) = 1 - sigmoid(logit_p)
            prob_misclass_prior = torch.sigmoid(-logit_p).mean().item()

        self.disc_optimizer.zero_grad(set_to_none=True)
        loss_d.backward()
        if self.is_multi_gpu:
            self._reduce_disc_grads()
        self.disc_optimizer.step()
        self.disc_optimizer.zero_grad(set_to_none=True)

        return (
            float(loss_d.item()),
            float(acc),
            float(logit_gap),
            float(prob_misclass_encoder),
            float(prob_misclass_prior),
        )

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
        mean_disc_loss = 0.0
        mean_disc_acc = 0.0
        mean_logit_gap = 0.0
        mean_prob_misclass_encoder = 0.0
        mean_prob_misclass_prior = 0.0
        disc_steps = 0
        cnt = 0
        total_loss_accumulator: torch.Tensor | None = None

        for _epoch in range(self.num_learning_epochs):
            prev_mu_e: torch.Tensor | None = None
            prev_mu_p: torch.Tensor | None = None
            prev_dones: torch.Tensor | None = None
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()

            for _t, (obs, _, _, privileged_actions, dones) in enumerate(
                self.storage.generator()
            ):
                action_e, action_p, mu_e, log_sigma_e, mu_p, log_sigma_p = (
                    self.policy.forward_for_update_dual_actions(obs, sample_z=True)
                )

                if cnt % self.disc_update_interval == 0:
                    d_loss, d_acc, d_gap, p_enc, p_pri = self._discriminator_step(
                        obs, action_e, action_p
                    )
                    mean_disc_loss += d_loss
                    mean_disc_acc += d_acc
                    mean_logit_gap += d_gap
                    mean_prob_misclass_encoder += p_enc
                    mean_prob_misclass_prior += p_pri
                    disc_steps += 1

                behavior_loss = self.loss_fn(action_e, privileged_actions)
                mean_behavior_loss += behavior_loss.item()

                kl_loss = _kl_gaussian(mu_e, log_sigma_e, mu_p, log_sigma_p)
                mean_kl_loss += kl_loss.item()
                mean_mu_e_mu_p_l2 += _mean_mu_e_mu_p_l2(mu_e, mu_p).item()
                kl_for_loss = (
                    kl_loss
                    if self.kl_loss_upper_bound is None
                    else torch.clamp(kl_loss, max=self.kl_loss_upper_bound)
                )
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

                just_optimized = False
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                    total_loss_accumulator.backward()
                    if self.is_multi_gpu:
                        self._reduce_student_grads()
                    nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    total_loss_accumulator = None
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

        if total_loss_accumulator is not None:
            self.optimizer.zero_grad(set_to_none=True)
            total_loss_accumulator.backward()
            if self.is_multi_gpu:
                self._reduce_student_grads()
            nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.policy.detach_hidden_states()

        if cnt > 0:
            mean_behavior_loss /= cnt
            mean_kl_loss /= cnt
            mean_mu_e_mu_p_l2 /= cnt
            mean_regularization_loss /= cnt
            mean_regularization_prior_loss /= cnt
        if disc_steps > 0:
            mean_disc_loss /= disc_steps
            mean_disc_acc /= disc_steps
            mean_logit_gap /= disc_steps
            mean_prob_misclass_encoder /= disc_steps
            mean_prob_misclass_prior /= disc_steps

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

        out = {
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
            "adv_pulse/disc_loss": mean_disc_loss,
            "adv_pulse/disc_acc": mean_disc_acc,
            "adv_pulse/logit_gap": mean_logit_gap,
            "adv_pulse/prob_misclass_encoder": mean_prob_misclass_encoder,
            "adv_pulse/prob_misclass_prior": mean_prob_misclass_prior,
            "adv_pulse/bce_chance": math.log(2.0),
        }
        return out

    def reduce_parameters(self) -> None:
        """Unused in this subclass (we call _reduce_student_grads / _reduce_disc_grads explicitly)."""
        self._reduce_student_grads()

    def broadcast_parameters(self) -> None:
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
