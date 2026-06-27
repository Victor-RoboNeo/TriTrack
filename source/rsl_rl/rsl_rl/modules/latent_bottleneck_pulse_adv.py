"""PULSE student policy with an action-space discriminator (encoder vs prior decode paths).

Rollout and BC still use the encoder path only; ``action_discriminator`` is unused at inference and only
trained in :class:`~rsl_rl.algorithms.adv_pulse_distillation.AdvPulseDistillation` on detached tensors.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from rsl_rl.modules.latent_bottleneck_pulse import LatentBottleneckPULSE
from rsl_rl.utils.finite_checks import require_all_finite
from rsl_rl.modules.pulse_action_discriminator import PulseActionDiscriminator


class LatentBottleneckPULSEAdv(LatentBottleneckPULSE):
    """Same as :class:`LatentBottleneckPULSE` plus ``action_discriminator`` and dual-action forward for training."""

    def __init__(
        self,
        num_student_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        *,
        proprio_dim: int,
        latent_dim: int = 64,
        encoder_hidden_dims: Sequence[int] = (512, 512),
        decoder_hidden_dims: Sequence[int] = (512, 512),
        prior_hidden_dims: Sequence[int] = (256, 256),
        teacher_hidden_dims: Sequence[int] = (256, 256, 256),
        activation: str = "elu",
        initialize_std: float = -1.0,
        init_noise_std: float = 0.1,
        latent_sigma_min: float = 0.01,
        latent_sigma_max: float = 1.0,
        fixed_prior_std: float | None = None,
        fixed_encoder_std: float | None = None,
        decoder_use_proprio: bool = True,
        disc_hidden_dims: Sequence[int] = (256, 128),
        **kwargs: Any,
    ):
        if kwargs:
            print(
                "LatentBottleneckPULSEAdv.__init__ got unexpected arguments (ignored): "
                + str(list(kwargs.keys()))
            )
        super().__init__(
            num_student_obs=num_student_obs,
            num_teacher_obs=num_teacher_obs,
            num_actions=num_actions,
            proprio_dim=proprio_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            prior_hidden_dims=prior_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            initialize_std=initialize_std,
            init_noise_std=init_noise_std,
            latent_sigma_min=latent_sigma_min,
            latent_sigma_max=latent_sigma_max,
            fixed_prior_std=fixed_prior_std,
            fixed_encoder_std=fixed_encoder_std,
            decoder_use_proprio=decoder_use_proprio,
        )
        disc_in = int(self.num_actions) + int(self.proprio_dim)
        self.action_discriminator = PulseActionDiscriminator(
            disc_in,
            list(disc_hidden_dims),
            activation=activation,
        )
        print(
            f"[LatentBottleneckPULSEAdv] disc_hidden_dims={list(disc_hidden_dims)}, disc_in={disc_in}"
        )

    def forward_for_update_dual_actions(
        self,
        observations: torch.Tensor,
        sample_z: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Encoder-path and prior-path decoded actions; KL statistics match :meth:`forward_for_update`.

        Returns:
            action_e, action_p, mu_e, log_sigma_e, mu_p, log_sigma_p
        """
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Student obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        require_all_finite(observations, "LatentBottleneckPULSEAdv forward_for_update_dual_actions observations")
        proprio = self.student_core.get_proprio(observations)
        mu_e, log_sigma_e = self.student_core.encode(observations)
        mu_p, log_sigma_p = self.prior(proprio)
        if sample_z and self.training:
            z_e = self.student_core.reparameterize(mu_e, log_sigma_e)
            z_p = self.student_core.reparameterize(mu_p, log_sigma_p)
        else:
            z_e = mu_e
            z_p = mu_p
        action_e = self._ensure_finite(
            self.student_core.decode(z_e, proprio), "decoder action encoder path"
        )
        with torch.no_grad(): #TODO: check if this is correct
            action_p = self._ensure_finite(
                self.student_core.decode(z_p, proprio), "decoder action prior path"
            )
        return action_e, action_p, mu_e, log_sigma_e, mu_p, log_sigma_p

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Allow warm-start from plain PULSE checkpoints without discriminator weights."""
        has_disc = any(str(k).startswith("action_discriminator") for k in state_dict.keys())
        if strict and not has_disc:
            strict = False
        return super().load_state_dict(state_dict, strict=strict)
