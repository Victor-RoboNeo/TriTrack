"""MUSE student-teacher policy: encoder + decoder, no separate prior.

Policy structure (no prior network):
- Encoder E: (s^p, s^g) -> μ^e, log σ^e   where s^g may be randomly masked at training time
- Decoder D: (z, s^p) -> action

Supervised update uses action loss + temporal latent regularization (no KL term).
The "encoder-as-prior" semantics is implicit: when s^g is masked (delta-command zero,
anchor-ori identity), the encoder learns to produce a default latent that the decoder
maps to a stable behaviour, because masking is sampled with probability ``p_mask`` during
training. Because there is no KL term to compete with, the encoder's σ head is always
learned (no ``fixed_encoder_std`` knob).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.utils.finite_checks import require_all_finite


def _muse_log_sigma_bounds(
    latent_predict_std_min: float,
    latent_predict_std_max: float,
) -> tuple[float, float]:
    """Return (log σ_min, log σ_max) for the encoder log-σ head from std bounds."""
    smin = max(float(latent_predict_std_min), 1e-12)
    smax = max(float(latent_predict_std_max), smin + 1e-12)
    return math.log(smin), math.log(smax)


class _MUSELatentBottleneck(nn.Module):
    """Self-contained encoder-decoder bottleneck for MUSE.

    obs = [goal, proprio]  (goal_dim = obs_dim - proprio_dim)
    - Encoder: obs -> μ, log σ. With ``fixed_encoder_std`` set, σ is broadcast as a constant per
      latent dim and the σ head is omitted entirely (no useless parameters). With it ``None``,
      σ is learned and clamped to ``[σ_min, σ_max]`` — but note that without a KL term the σ head
      has no incentive to remain non-minimal and tends to collapse to the lower clamp.
    - Decoder: [z, proprio] -> action  (or [z] -> action if ``decoder_use_proprio=False``)
    """

    def __init__(
        self,
        obs_dim: int,
        proprio_dim: int,
        num_actions: int,
        latent_dim: int,
        encoder_hidden: Sequence[int],
        decoder_hidden: Sequence[int],
        activation: str = "elu",
        *,
        latent_sigma_min: float = 0.001,
        latent_sigma_max: float = 1.0,
        fixed_encoder_std: float | None = None,
        decoder_use_proprio: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.proprio_dim = int(proprio_dim)
        self.goal_dim = self.obs_dim - self.proprio_dim
        if self.goal_dim <= 0:
            raise ValueError(
                f"_MUSELatentBottleneck: obs_dim ({obs_dim}) must be > proprio_dim ({proprio_dim})."
            )
        self.latent_dim = int(latent_dim)
        self.num_actions = int(num_actions)
        self.decoder_use_proprio = bool(decoder_use_proprio)
        self.fixed_encoder_std = (
            None if fixed_encoder_std is None else float(fixed_encoder_std)
        )

        lmin, lmax = _muse_log_sigma_bounds(latent_sigma_min, latent_sigma_max)
        self._log_sigma_min = lmin
        self._log_sigma_max = lmax

        act_cls = getattr(nn, activation.upper(), nn.ELU)

        # Encoder body: [goal, proprio] -> hidden  ->  (μ, log σ) heads
        enc_in = self.obs_dim
        enc_layers: list[nn.Module] = []
        for h in encoder_hidden:
            enc_layers.append(nn.Linear(enc_in, h))
            enc_layers.append(act_cls())
            enc_in = h
        self.encoder_body = nn.Sequential(*enc_layers)
        self.encoder_mu = nn.Linear(enc_in, self.latent_dim)
        if self.fixed_encoder_std is None:
            self.encoder_log_sigma = nn.Linear(enc_in, self.latent_dim)
        else:
            self.encoder_log_sigma = None

        # Decoder: [z, proprio] -> action  (or [z] -> action)
        dec_in = self.latent_dim + self.proprio_dim if self.decoder_use_proprio else self.latent_dim
        dec_layers: list[nn.Module] = []
        for h in decoder_hidden:
            dec_layers.append(nn.Linear(dec_in, h))
            dec_layers.append(act_cls())
            dec_in = h
        dec_layers.append(nn.Linear(dec_in, self.num_actions))
        self.decoder = nn.Sequential(*dec_layers)

    def get_proprio(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[..., -self.proprio_dim:]

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.shape[-1] != self.obs_dim:
            raise RuntimeError(
                f"MUSE encoder obs dim mismatch: got {obs.shape[-1]}, expected {self.obs_dim}."
            )
        h = self.encoder_body(obs)
        mu = self.encoder_mu(h)
        if self.encoder_log_sigma is not None:
            log_sigma = self.encoder_log_sigma(h).clamp(
                min=self._log_sigma_min, max=self._log_sigma_max
            )
        else:
            assert self.fixed_encoder_std is not None
            std = max(float(self.fixed_encoder_std), 1.0e-6)
            log_sigma = torch.full_like(mu, math.log(std))
        return mu, log_sigma

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu, device=mu.device)
        return mu + sigma * eps

    def decode(self, z: torch.Tensor, proprio: torch.Tensor | None = None) -> torch.Tensor:
        if self.decoder_use_proprio:
            if proprio is None:
                raise RuntimeError("MUSE decoder expected proprio but got None.")
            x = torch.cat([z, proprio], dim=-1)
        else:
            x = z
        return self.decoder(x)

    def forward(
        self,
        obs: torch.Tensor,
        sample_z: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        proprio = self.get_proprio(obs)
        mu, log_sigma = self.encode(obs)
        if sample_z and self.training:
            z = self.reparameterize(mu, log_sigma)
        else:
            z = mu
        action = self.decode(z, proprio)
        return action, mu, log_sigma, z


class LatentBottleneckMUSE(nn.Module):
    """MUSE student-teacher: standalone encoder + decoder + teacher MLP. No prior."""

    is_recurrent = False

    def __init__(
        self,
        num_student_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        *,
        proprio_dim: int,
        latent_dim: int = 16,
        encoder_hidden_dims: Sequence[int] = (512, 256, 128),
        decoder_hidden_dims: Sequence[int] = (512, 256, 128),
        teacher_hidden_dims: Sequence[int] = (1024, 1024, 512, 512, 256, 256),
        activation: str = "elu",
        initialize_std: float = -1.0,
        init_noise_std: float = 0.1,
        latent_predict_std_min: float = 0.001,
        latent_predict_std_max: float = 1.0,
        fixed_encoder_std: float | None = None,
        decoder_use_proprio: bool = True,
        **kwargs: Any,
    ):
        if kwargs:
            print(
                "LatentBottleneckMUSE.__init__ got unexpected arguments (ignored): "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        act_fn = resolve_nn_activation(activation)

        self.num_student_obs = int(num_student_obs)
        self.num_actions = int(num_actions)
        self.proprio_dim = int(proprio_dim)
        if self.proprio_dim >= self.num_student_obs:
            raise ValueError(
                f"LatentBottleneckMUSE: proprio_dim ({proprio_dim}) must be < num_student_obs ({num_student_obs})."
            )

        self.student_core = _MUSELatentBottleneck(
            obs_dim=self.num_student_obs,
            proprio_dim=self.proprio_dim,
            num_actions=self.num_actions,
            latent_dim=int(latent_dim),
            encoder_hidden=list(encoder_hidden_dims),
            decoder_hidden=list(decoder_hidden_dims),
            activation=activation,
            latent_sigma_min=latent_predict_std_min,
            latent_sigma_max=latent_predict_std_max,
            fixed_encoder_std=None if fixed_encoder_std is None else float(fixed_encoder_std),
            decoder_use_proprio=bool(decoder_use_proprio),
        )

        self._initialize_std_head(initialize_std)
        # Distillation algorithm reads ``policy.student.parameters()`` for the optimizer.
        self.student: nn.Module = nn.ModuleList([self.student_core])

        # Teacher MLP (PHC+-style stage-1 actor) over privileged observations.
        teacher_layers: list[nn.Module] = []
        hidden = list(teacher_hidden_dims) or [256, 256]
        teacher_layers.append(nn.Linear(num_teacher_obs, hidden[0]))
        teacher_layers.append(act_fn)
        for i in range(len(hidden)):
            if i == len(hidden) - 1:
                teacher_layers.append(nn.Linear(hidden[i], self.num_actions))
            else:
                teacher_layers.append(nn.Linear(hidden[i], hidden[i + 1]))
                teacher_layers.append(act_fn)
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()
        self.loaded_teacher = False

        # Action distribution (logging / interface compatibility only — distillation, not RL).
        init_std = max(float(init_noise_std), 1.0e-6)
        self.std = nn.Parameter(init_std * torch.ones(self.num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args = False  # type: ignore[assignment]
        self._warned_nonfinite = False

        # Expose log-σ bounds for algorithm-side logging (matches the PULSE convention).
        self.latent_log_sigma_min = self.student_core._log_sigma_min
        self.latent_log_sigma_max = self.student_core._log_sigma_max

        print(
            f"[LatentBottleneckMUSE] num_student_obs={self.num_student_obs}, "
            f"proprio_dim={self.proprio_dim}, goal_dim={self.num_student_obs - self.proprio_dim}, "
            f"latent_dim={latent_dim}, latent_sigma_min={latent_predict_std_min}, "
            f"latent_sigma_max={latent_predict_std_max}, "
            f"fixed_encoder_std={self.student_core.fixed_encoder_std}, "
            f"decoder_use_proprio={self.student_core.decoder_use_proprio}, "
            f"encoder_hidden={list(encoder_hidden_dims)}, decoder_hidden={list(decoder_hidden_dims)}, "
            f"teacher_hidden={list(teacher_hidden_dims)}"
        )

    def _initialize_std_head(self, initialize_std: float) -> None:
        if float(initialize_std) == -1.0:
            return
        if self.student_core.encoder_log_sigma is None:
            return  # σ is fixed; no head to initialize.
        init_std_val = max(float(initialize_std), 1.0e-3)
        init_log_std = float(torch.tensor(init_std_val).log().item())
        nn.init.normal_(self.student_core.encoder_log_sigma.weight, mean=0.0, std=1.0e-3)
        nn.init.constant_(self.student_core.encoder_log_sigma.bias, init_log_std)

    def reset(self, dones=None, hidden_states=None) -> None:
        return

    def _ensure_finite(self, tensor: torch.Tensor, label: str) -> torch.Tensor:
        if torch.isfinite(tensor).all():
            return tensor
        if not self._warned_nonfinite:
            self._warned_nonfinite = True
            warnings.warn(f"LatentBottleneckMUSE: non-finite values in {label}; replacing with 0.", stacklevel=2)
        return torch.nan_to_num(tensor)

    def update_distribution(self, observations: torch.Tensor) -> None:
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Student obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        require_all_finite(observations, "LatentBottleneckMUSE update_distribution observations")
        actions_mean, _, _, _ = self.student_core(observations, sample_z=False)
        std = self.std.expand_as(actions_mean).clamp(min=1.0e-6)
        self.distribution = Normal(actions_mean, std)

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Student obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        require_all_finite(observations, "LatentBottleneckMUSE act observations")
        use_mean = bool(getattr(self, "deterministic_latent", False)) or not self.training
        actions, _, _, _ = self.student_core(observations, sample_z=not use_mean)
        return actions

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Student obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        require_all_finite(observations, "LatentBottleneckMUSE act_inference observations")
        actions, _, _, _ = self.student_core(observations, sample_z=False)
        return actions

    def forward_for_update(
        self,
        observations: torch.Tensor,
        sample_z: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Supervised update: returns (action, μ^e, log σ^e). No prior outputs."""
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Student obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        require_all_finite(observations, "LatentBottleneckMUSE forward_for_update observations")
        proprio = self.student_core.get_proprio(observations)
        mu_e, log_sigma_e = self.student_core.encode(observations)
        if sample_z and self.training:
            z = self.student_core.reparameterize(mu_e, log_sigma_e)
        else:
            z = mu_e
        action = self._ensure_finite(self.student_core.decode(z, proprio), "decoder action")
        return action, mu_e, log_sigma_e

    def evaluate(self, teacher_observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.teacher(teacher_observations)

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call act(...) first.")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call act(...) first.")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call act(...) first.")
        return self.distribution.entropy().sum(dim=-1)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        # Stage-1 teacher checkpoint: keys prefixed with ``actor.``
        if any("actor." in key for key in state_dict.keys()):
            teacher_state_dict = {
                key.replace("actor.", ""): value
                for key, value in state_dict.items()
                if "actor." in key
            }
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            return False
        # Resume from MUSE checkpoint: ``student_core.*``, ``teacher.*``
        if any("student_core" in key or "teacher." in key for key in state_dict.keys()):
            super().load_state_dict(state_dict, strict=False)
            self.loaded_teacher = True
            self.teacher.eval()
            return True
        raise ValueError("state_dict does not contain actor, student_core, or teacher parameters.")

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None) -> None:
        return
