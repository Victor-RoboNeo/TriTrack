from __future__ import annotations

import math
import warnings
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.utils.finite_checks import require_all_finite

def _latent_log_sigma_bounds(
    latent_predict_std_min: float,
    latent_predict_std_max: float,
) -> tuple[float, float]:
    """Return (log σ_min, log σ_max) for encoder/prior log-sigma heads from std bounds."""
    smin = max(float(latent_predict_std_min), 1e-12)
    smax = max(float(latent_predict_std_max), smin + 1e-12)
    return math.log(smin), math.log(smax)


def _prior_rollout_log_sigma(
    log_sigma_p: torch.Tensor,
    fixed_latent_std: float | None,
) -> torch.Tensor:
    """Optional fixed diagonal std (broadcast) instead of network ``log_sigma_p`` (rollout / KL prior scale)."""
    if fixed_latent_std is None:
        return log_sigma_p
    std = max(float(fixed_latent_std), 1.0e-6)
    return torch.full_like(log_sigma_p, math.log(std))


class _PULSELatentBottleneck(nn.Module):
    """Encoder-decoder with latent bottleneck for PULSE student.

    Observation is split into imitation goal (s^g) and proprioception (s^p):
    obs = [goal, proprio], with goal_dim = obs_dim - proprio_dim.
    - Encoder (ε): goal (s^g) → μ, log σ (optional ``fixed_encoder_std``: σ is a constant per dim, like ``fixed_prior_std``)
    - Decoder (D): [z, proprio (s^p)] → action
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
        latent_sigma_min: float | None = None,
        latent_sigma_max: float | None = None,
        fixed_encoder_std: float | None = None,
        decoder_use_proprio: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.proprio_dim = int(proprio_dim)
        self.goal_dim = self.obs_dim - self.proprio_dim
        if self.goal_dim <= 0:
            raise ValueError(
                f"PULSE bottleneck: obs_dim ({obs_dim}) must be > proprio_dim ({proprio_dim})."
            )

        self.latent_dim = int(latent_dim)
        self.num_actions = int(num_actions)
        self.decoder_use_proprio = bool(decoder_use_proprio)
        if latent_sigma_min is None or latent_sigma_max is None:
            min_sigma = 0.01
            max_sigma = 1.0
        else: 
            min_sigma = float(latent_sigma_min)
            max_sigma = float(latent_sigma_max)
        lmin, lmax = _latent_log_sigma_bounds(min_sigma, max_sigma)
        self._log_sigma_min = lmin
        self._log_sigma_max = lmax
        self.fixed_encoder_std = (
            None if fixed_encoder_std is None else float(fixed_encoder_std)
        )

        act_cls = getattr(nn, activation.upper(), nn.ELU)

        # Encoder: [goal, proprio] -> μ, log σ
        enc_in = self.obs_dim
        enc_layers: list[nn.Module] = []
        for h in encoder_hidden:
            enc_layers.append(nn.Linear(enc_in, h))
            enc_layers.append(act_cls())
            enc_in = h
        self.encoder_body = nn.Sequential(*enc_layers)
        self.encoder_mu = nn.Linear(enc_in, self.latent_dim)
        if fixed_encoder_std is None:
            self.encoder_log_sigma = nn.Linear(enc_in, self.latent_dim)
        else:
            self.encoder_log_sigma = None

        # Decoder: [z, proprio] -> action (default) or z -> action when decoder_use_proprio=False.
        dec_in = self.latent_dim + self.proprio_dim if self.decoder_use_proprio else self.latent_dim
        dec_layers: list[nn.Module] = []
        for h in decoder_hidden:
            dec_layers.append(nn.Linear(dec_in, h))
            dec_layers.append(act_cls())
            dec_in = h
        dec_layers.append(nn.Linear(dec_in, self.num_actions))
        self.decoder = nn.Sequential(*dec_layers)

    def get_proprio(self, obs: torch.Tensor) -> torch.Tensor:
        """Return proprioceptive part of obs (last proprio_dim dimensions)."""
        return obs[..., -self.proprio_dim :]

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.shape[-1] != self.obs_dim:
            raise RuntimeError(
                f"PULSE obs dim mismatch: got {obs.shape[-1]}, expected {self.obs_dim}."
            )
        goal = obs[..., : self.goal_dim]
        proprio = obs[..., -self.proprio_dim :]
        return goal, proprio

    def encode(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.shape[-1] != self.obs_dim:
            raise RuntimeError(
                f"PULSE encoder obs dim mismatch: got {obs.shape[-1]}, expected {self.obs_dim}."
            )
        x = obs
        h = self.encoder_body(x)
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
                raise RuntimeError("PULSE decoder expected proprio but got None.")
            x = torch.cat([z, proprio], dim=-1)
        else:
            x = z
        return self.decoder(x)

    def forward(
        self,
        obs: torch.Tensor,
        sample_z: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        goal, proprio = self._split_obs(obs)
        mu, log_sigma = self.encode(obs)
        if sample_z and self.training:
            z = self.reparameterize(mu, log_sigma)
        else:
            z = mu
        action = self.decode(z, proprio)
        return action, mu, log_sigma, z


class _PULSEPrior(nn.Module):
    """Prior model R: proprioception (s^p) -> μ^p, log σ^p (predicted), log σ for KL / prior sampling.

    Conditioned only on proprioception so that the encoder E(s^p, s^g) is
    regularized toward this prior via KL during the supervised update .

    The MLP always predicts diagonal log σ unless ``fixed_prior_std`` is set; then ``forward`` returns
    ``μ`` and a broadcast ``log(fixed_prior_std)`` per latent dim (KL and default prior rollout).
    """

    def __init__(
        self,
        proprio_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "elu",
        *,
        latent_sigma_min: float | None = None,
        latent_sigma_max: float | None = None,
        fixed_prior_std: float | None = None,
    ):
        super().__init__()
        self.proprio_dim = int(proprio_dim)
        self.latent_dim = int(latent_dim)
        self.fixed_prior_std = (
            None if fixed_prior_std is None else float(fixed_prior_std)
        )
        if latent_sigma_min is None or latent_sigma_max is None:
            min_sigma = 0.01
            max_sigma = 1.0
        else: 
            min_sigma = float(latent_sigma_min)
            max_sigma = float(latent_sigma_max)
        lmin, lmax = _latent_log_sigma_bounds(min_sigma, max_sigma)
        self._log_sigma_min = lmin
        self._log_sigma_max = lmax

        act_cls = getattr(nn, activation.upper(), nn.ELU)
        layers: list[nn.Module] = []
        in_dim = self.proprio_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_cls())
            in_dim = h
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(in_dim, self.latent_dim)

        if fixed_prior_std is None: # predict log_sigma
            self.log_sigma = nn.Linear(in_dim, self.latent_dim)
        else: # use fixed prior std
            self.log_sigma = None

    def forward(
        self, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(proprio)
        mu = self.mu(h)

        if self.log_sigma is not None:
            log_sigma = self.log_sigma(h).clamp(
                min=self._log_sigma_min, max=self._log_sigma_max
                )
        else: # use fixed prior std
            assert self.fixed_prior_std is not None
            log_sigma = torch.full_like(mu, math.log(float(self.fixed_prior_std)))
        
        return mu,log_sigma


class LatentBottleneckPULSE(nn.Module):
    """
    PULSE student-teacher policy with encoder E, decoder D, and prior R .

    - Encoder E: (s^p, s^g) -> μ^e, σ^e  (optional ``fixed_encoder_std`` uses a constant σ^e per dim
      like ``fixed_prior_std`` for the prior)
    - Prior R: s^p -> μ^p, σ^p (MLP-predicted; optional ``fixed_prior_std`` on :class:`_PULSEPrior`
      uses that σ for KL / default prior rollout only)
    - Decoder D: (z, s^p) -> action
    Supervised update uses action loss + KL(encoder || prior) + optional regularization .
    """

    is_recurrent = False

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
        **kwargs: Any,
    ):
        if kwargs:
            print(
                "LatentBottleneckPULSE.__init__ got unexpected arguments (ignored): "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        act_fn = resolve_nn_activation(activation)

        self.num_student_obs = int(num_student_obs)
        self.num_actions = int(num_actions)
        self.proprio_dim = int(proprio_dim)
        if self.proprio_dim >= self.num_student_obs:
            raise ValueError(
                f"LatentBottleneckPULSE: proprio_dim ({proprio_dim}) must be < num_student_obs ({num_student_obs})."
            )

        # Student core: encoder on [goal, proprio], decoder on [z, proprio].
        self.student_core = _PULSELatentBottleneck(
            obs_dim=self.num_student_obs,
            proprio_dim=self.proprio_dim,
            num_actions=self.num_actions,
            latent_dim=int(latent_dim),
            encoder_hidden=list(encoder_hidden_dims),
            decoder_hidden=list(decoder_hidden_dims),
            activation=activation,
            latent_sigma_min=latent_sigma_min,
            latent_sigma_max=latent_sigma_max,
            fixed_encoder_std=None
            if fixed_encoder_std is None
            else float(fixed_encoder_std),
            decoder_use_proprio=bool(decoder_use_proprio),
        )
        # Prior R: proprio -> μ^p, σ^p (Algo 1; used in KL term with encoder output).
        self.prior = _PULSEPrior(
            proprio_dim=self.proprio_dim,
            latent_dim=int(latent_dim),
            hidden_dims=list(prior_hidden_dims),
            activation=activation,
            latent_sigma_min=latent_sigma_min,
            latent_sigma_max=latent_sigma_max,
            fixed_prior_std=None if fixed_prior_std is None else float(fixed_prior_std),
        )
        self._initialize_std_heads(initialize_std)
        # For compatibility with Distillation algorithm (uses policy.student.parameters()).
        # Student trainable params = encoder + decoder + prior (E, D, R all updated).
        self.student: nn.Module = nn.ModuleList([self.student_core, self.prior])

        # Teacher: MLP over privileged observations.
        teacher_layers: list[nn.Module] = []
        in_dim = num_teacher_obs
        hidden = list(teacher_hidden_dims)
        if not hidden:
            hidden = [256, 256]
        teacher_layers.append(nn.Linear(in_dim, hidden[0]))
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

        # Action distribution (for logging / interface compatibility).
        init_std = max(float(init_noise_std), 1.0e-6)
        self.std = nn.Parameter(init_std * torch.ones(self.num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args = False  # type: ignore[assignment]
        self._warned_nonfinite = False

        print(
            f"[LatentBottleneckPULSE] num_student_obs={self.num_student_obs}, proprio_dim={self.proprio_dim}, "
            f"goal_dim={self.num_student_obs - self.proprio_dim}, latent_dim={latent_dim}, "
            f"latent_sigma_min={latent_sigma_min}, latent_sigma_max={latent_sigma_max}, "
            f"fixed_encoder_std={self.student_core.fixed_encoder_std}, "
            f"fixed_prior_std={self.prior.fixed_prior_std}, "
            f"decoder_use_proprio={self.student_core.decoder_use_proprio}, "
            f"encoder_hidden={list(encoder_hidden_dims)}, decoder_hidden={list(decoder_hidden_dims)}, "
            f"prior_hidden={list(prior_hidden_dims)}, teacher_hidden={list(teacher_hidden_dims)}"
        )

    def _initialize_std_heads(self, initialize_std: float) -> None:
        """Optionally initialize encoder/prior log-sigma heads for small starting std."""
        if float(initialize_std) == -1.0:
            return

        init_std = max(float(initialize_std), 1.0e-3)
        init_log_std = float(torch.tensor(init_std).log().item())
        
        # 1. initialize encoder log_sigma if not using fixed encoder std
        if self.student_core.encoder_log_sigma is not None:
            nn.init.normal_(
                self.student_core.encoder_log_sigma.weight, mean=0.0, std=1.0e-3
            )
            nn.init.constant_(self.student_core.encoder_log_sigma.bias, init_log_std)

        # 2. initialize prior log_sigma if not using fixed prior std
        if self.prior.log_sigma is not None:
            nn.init.normal_(self.prior.log_sigma.weight, mean=0.0, std=1.0e-3)
            nn.init.constant_(self.prior.log_sigma.bias, init_log_std)


    def reset(self, dones=None, hidden_states=None) -> None:  # noqa: D401
        """No recurrent state."""
        return

    def _ensure_finite(self, tensor: torch.Tensor, label: str) -> torch.Tensor:
        """Replace NaN/inf for stable decoder / loss (same idea as advisor ``StudentTeacherVAE``)."""
        if torch.isfinite(tensor).all():
            return tensor
        if not self._warned_nonfinite:
            bad = ~torch.isfinite(tensor)
            bad_count = int(bad.sum().item())
            finite_vals = torch.abs(tensor[torch.isfinite(tensor)])
            max_abs = float(finite_vals.max().item()) if finite_vals.numel() > 0 else 0.0
            warnings.warn(
                f"LatentBottleneckPULSE: non-finite values in {label}; count={bad_count}, max_abs={max_abs}.",
                stacklevel=2,
            )
            self._warned_nonfinite = True
        return torch.nan_to_num(tensor)

    def update_distribution(self, observations: torch.Tensor) -> None:
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Student obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        require_all_finite(observations, "LatentBottleneckPULSE update_distribution observations")
        actions_mean, _, _, _ = self.student_core(
            observations, sample_z=False
        )
        std = self.std.expand_as(actions_mean).clamp(min=1.0e-6)
        self.distribution = Normal(actions_mean, std)

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        """Student action for environment rollouts.

        Uses reparameterized latent sampling when training and ``deterministic_latent`` is false;
        otherwise decodes with encoder mean μ (no sampling). Eval mode (``self.training`` False)
        also uses μ unless training with sampling enabled.

        If ``test_prior_quality`` is True (set by training CLI), bypasses the encoder and uses
        prior R(proprio) → decode only; optional ``test_prior_quality_fixed_std`` overrides σ.
        """
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Student obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        require_all_finite(observations, "LatentBottleneckPULSE act observations")
        if getattr(self, "test_prior_quality", False):
            return self.act_prior_sample(
                observations,
                sample_latent=True,
                fixed_latent_std=getattr(self, "test_prior_quality_fixed_std", None),
            )
        use_mean = bool(getattr(self, "deterministic_latent", False)) or not self.training
        actions, _, _, _ = self.student_core(observations, sample_z=not use_mean)
        return actions

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Deterministic student action used in distillation loss."""
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Student obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        require_all_finite(observations, "LatentBottleneckPULSE act_inference observations")
        if getattr(self, "test_prior_quality", False):
            return self.act_prior_sample(
                observations,
                sample_latent=True,
                fixed_latent_std=getattr(self, "test_prior_quality_fixed_std", None),
            )
        actions, _, _, _ = self.student_core(
            observations, sample_z=False
        )
        return actions

    def act_prior_sample(
        self,
        observations: torch.Tensor,
        *,
        sample_latent: bool = True,
        fixed_latent_std: float | None = None,
    ) -> torch.Tensor:
        """Use prior R(proprio) and decode to action (encoder bypassed).

        If ``sample_latent`` is True (default), draws ``z ~ N(μ_p, σ_p)`` via reparameterization.
        If False, uses ``z = μ_p`` (deterministic prior mean), e.g. for repeatable prior videos.

        If ``fixed_latent_std`` is set, uses that standard deviation for every latent dimension
        instead of the prior MLP's predicted ``σ_p`` (still uses ``μ_p`` from the network).
        If ``fixed_latent_std`` is None and the policy was constructed with ``fixed_prior_std``,
        that policy default is used for rollout sampling.
        """
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Student obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        require_all_finite(observations, "LatentBottleneckPULSE act_prior_sample observations")
        with torch.no_grad():
            proprio = self.student_core.get_proprio(observations)
            mu_p, log_sigma_p = self.prior(proprio)
            log_sigma_rollout = _prior_rollout_log_sigma(log_sigma_p, fixed_latent_std)
            if sample_latent:
                z = self.student_core.reparameterize(mu_p, log_sigma_rollout)
            else:
                z = mu_p
            actions = self.student_core.decode(z, proprio)
        return actions

    def act_prior_sample_with_stats(
        self,
        observations: torch.Tensor,
        *,
        sample_latent: bool = True,
        fixed_latent_std: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Same as ``act_prior_sample`` but returns tensors for rollout metrics (entropy, proprio, etc.).

        ``sigma_p`` is ``exp(log_sigma_p)`` from the prior MLP (predicted per-latent std), even when
        rollout uses a fixed std (``fixed_latent_std`` or policy ``fixed_prior_std``).
        """
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Student obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        require_all_finite(observations, "LatentBottleneckPULSE act_prior_sample_with_stats observations")
        with torch.no_grad():
            proprio = self.student_core.get_proprio(observations)
            mu_p, log_sigma_p = self.prior(proprio)
            log_sigma_rollout = _prior_rollout_log_sigma(log_sigma_p, fixed_latent_std)
            if sample_latent:
                z = self.student_core.reparameterize(mu_p, log_sigma_rollout)
            else:
                z = mu_p
            actions = self.student_core.decode(z, proprio)
            # Differential entropy of factorized Gaussian used for rollout (sigma = exp(log_sigma_rollout)).
            log_2pi_e = math.log(2.0 * math.pi * math.e)
            prior_entropy = (0.5 * (log_2pi_e + 2.0 * log_sigma_rollout)).sum(dim=-1)
            # Prior MLP predicted σ (sampling may use fixed_latent_std instead via log_sigma_rollout).
            sigma_p = torch.exp(log_sigma_p)
        return {
            "actions": actions,
            "proprio": proprio,
            "mu_p": mu_p,
            "log_sigma_p": log_sigma_p,
            "sigma_p": sigma_p,
            "prior_entropy": prior_entropy,
        }

    def forward_for_update(
        self,
        observations: torch.Tensor,
        sample_z: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Supervised update (Eq.3): action, μ^e, log σ^e, μ^p, log σ^p for KL.

        ``log_sigma_e`` uses a broadcast constant if ``fixed_encoder_std`` is set (encoder log-σ head omitted).
        ``log_sigma_p`` is the prior log-std used in ``KL(encoder || prior)`` (fixed if ``fixed_prior_std``).
        """
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Student obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        require_all_finite(observations, "LatentBottleneckPULSE forward_for_update observations")
        proprio = self.student_core.get_proprio(observations)
        mu_e, log_sigma_e = self.student_core.encode(observations)
        if sample_z and self.training:
            z = self.student_core.reparameterize(mu_e, log_sigma_e)
        else:
            z = mu_e
        action = self._ensure_finite(
            self.student_core.decode(z, proprio), "decoder action"
        )
        mu_p, log_sigma_p = self.prior(proprio)
        return action, mu_e, log_sigma_e, mu_p, log_sigma_p

    def evaluate(self, teacher_observations: torch.Tensor) -> torch.Tensor:
        """Teacher action from privileged observations."""
        with torch.no_grad():
            actions = self.teacher(teacher_observations)
        return actions

    # Boilerplate for Distillation / runner interfaces

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
        if any("student_core" in key or "teacher." in key or "prior." in key for key in state_dict.keys()):
            # Load all present; prior may be absent in old checkpoints (left at init)
            super().load_state_dict(state_dict, strict=False)
            self.loaded_teacher = True
            self.teacher.eval()
            return True
        raise ValueError("state_dict does not contain actor, student_core, prior, or teacher parameters.")

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None) -> None:
        return
