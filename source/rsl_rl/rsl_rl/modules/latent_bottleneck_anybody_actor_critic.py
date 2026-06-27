"""PPO actor–critic: PULSE-style prior + residual latent MLP + decoder (no PULSE encoder teacher)."""

from __future__ import annotations

import os
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.latent_bottleneck_pulse import _PULSEPrior
from rsl_rl.utils import resolve_nn_activation
from rsl_rl.utils.finite_checks import require_all_finite


def _strip_prefix_keys(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    plen = len(prefix)
    return {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}


class LatentBottleneckAnyBodyActorCritic(nn.Module):
    """Joint-space Gaussian policy with the same mean path as :class:`LatentBottleneckAnyBody` (no ``teacher_core``).

    Mean action: ``decoder(mu_prior(proprio) + residual_scale * latent_actor(obs), proprio)``.
    PPO explores with diagonal Gaussian noise in **joint** space (``num_actions == num joints``).

    By default **prior**, **latent_actor** (residual encoder), and **decoder** are all trainable for finetuning.
    Use ``train_prior=False`` / ``train_decoder=False`` to freeze subsets.

    Initialization:
    - ``init_from_pulse_checkpoint``: load ``prior.*`` and ``student_core.decoder.*`` from a PULSE-style .pt.
    - ``init_from_latent_rl_checkpoint``: load ``actor.*`` (MLP latent residual), ``critic.*``, and optionally ``std``.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        proprio_dim: int,
        latent_dim: int = 16,
        encoder_hidden_dims: Sequence[int] = (512, 256, 128),
        decoder_hidden_dims: Sequence[int] = (512, 256, 128),
        prior_hidden_dims: Sequence[int] = (512, 256, 128),
        critic_hidden_dims: Sequence[int] = (1024, 1024, 512, 512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 0.5,
        noise_std_type: str = "scalar",
        latent_predict_std_min: float | None = None,
        latent_predict_std_max: float | None = None,
        fixed_prior_std: float | None = None,
        residual_scale: float = 1.0,
        train_prior: bool = True,
        train_decoder: bool = True,
        **kwargs: Any,
    ):
        if kwargs:
            print(
                "LatentBottleneckAnyBodyActorCritic.__init__ got unexpected arguments (ignored): "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.num_actor_obs = int(num_actor_obs)
        self.num_student_obs = int(num_actor_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)
        self.latent_dim = int(latent_dim)
        self.proprio_dim = int(proprio_dim)
        self.residual_scale = float(residual_scale)
        self._train_prior = bool(train_prior)
        self._train_decoder = bool(train_decoder)

        if self.proprio_dim >= self.num_actor_obs:
            raise ValueError(
                f"proprio_dim ({self.proprio_dim}) must be < num_actor_obs ({self.num_actor_obs})."
            )

        act_fn = resolve_nn_activation(activation)

        self.prior = _PULSEPrior(
            proprio_dim=self.proprio_dim,
            latent_dim=self.latent_dim,
            hidden_dims=list(prior_hidden_dims),
            activation=activation,
            latent_sigma_min=latent_predict_std_min,
            latent_sigma_max=latent_predict_std_max,
            fixed_prior_std=fixed_prior_std,
        )

        dec_in = self.latent_dim + self.proprio_dim
        dec_layers: list[nn.Module] = []
        for h in decoder_hidden_dims:
            dec_layers.append(nn.Linear(dec_in, h))
            dec_layers.append(act_fn)
            dec_in = h
        dec_layers.append(nn.Linear(dec_in, self.num_actions))
        self.decoder = nn.Sequential(*dec_layers)

        enc_in = self.num_actor_obs
        la_layers: list[nn.Module] = []
        for h in encoder_hidden_dims:
            la_layers.append(nn.Linear(enc_in, h))
            la_layers.append(act_fn)
            enc_in = h
        la_layers.append(nn.Linear(enc_in, self.latent_dim))
        self.latent_actor = nn.Sequential(*la_layers)

        critic_layers: list[nn.Module] = []
        prev_c = self.num_critic_obs
        if critic_hidden_dims:
            critic_layers.append(nn.Linear(prev_c, critic_hidden_dims[0]))
            critic_layers.append(act_fn)
            for i in range(len(critic_hidden_dims)):
                in_dim = critic_hidden_dims[i]
                if i == len(critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(in_dim, 1))
                else:
                    out_dim = critic_hidden_dims[i + 1]
                    critic_layers.append(nn.Linear(in_dim, out_dim))
                    critic_layers.append(act_fn)
        else:
            critic_layers.append(nn.Linear(prev_c, 1))
        self.critic = nn.Sequential(*critic_layers)

        self.noise_std_type = str(noise_std_type)
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")

        self.distribution: Normal | None = None
        Normal.set_default_validate_args = False  # type: ignore[assignment]

        self._sync_parameter_requires_grad()

        print(
            f"[LatentBottleneckAnyBodyActorCritic] num_actor_obs={self.num_actor_obs}, "
            f"num_critic_obs={self.num_critic_obs}, proprio_dim={self.proprio_dim}, latent_dim={self.latent_dim}, "
            f"num_actions={self.num_actions}, residual_scale={self.residual_scale}, "
            f"train_prior={self._train_prior}, train_decoder={self._train_decoder}"
        )

    def _sync_parameter_requires_grad(self) -> None:
        for p in self.prior.parameters():
            p.requires_grad = self._train_prior
        for p in self.decoder.parameters():
            p.requires_grad = self._train_decoder
        for p in self.latent_actor.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if not self._train_prior:
            self.prior.eval()
        if not self._train_decoder:
            self.decoder.eval()
        return self

    def reset(self, dones=None) -> None:
        return

    def _split_student_obs(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        goal = observations[..., : self.num_student_obs - self.proprio_dim]
        proprio = observations[..., -self.proprio_dim :]
        return goal, proprio

    def joint_action_mean(self, observations: torch.Tensor) -> torch.Tensor:
        """Deterministic joint mean (prior + residual MLP + decoder); gradients flow when modules are trainable."""
        require_all_finite(observations, "LatentBottleneckAnyBodyActorCritic observations")
        goal, proprio = self._split_student_obs(observations)

        mu_p, _ = self.prior(proprio)
        delta = self.latent_actor(observations)
        mu_z = mu_p + self.residual_scale * delta
        x = torch.cat([mu_z, proprio], dim=-1)
        mean_joints = self.decoder(x)
        return mean_joints

    def update_distribution(self, observations: torch.Tensor) -> None:
        require_all_finite(observations, "LatentBottleneckAnyBodyActorCritic update_distribution")
        mean = self.joint_action_mean(observations)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        if self.noise_std_type == "scalar":
            std = torch.clamp_min(self.std, 1e-6).expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(self.noise_std_type)
        std = torch.nan_to_num(std, nan=1e-3, posinf=1e3, neginf=1e-3)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.update_distribution(observations)
        assert self.distribution is not None
        sampled_actions = self.distribution.sample()
        return sampled_actions

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        require_all_finite(observations, "LatentBottleneckAnyBodyActorCritic act_inference")
        return self.joint_action_mean(observations)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.critic(critic_observations)

    def forward(self) -> None:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized.")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized.")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized.")
        return self.distribution.entropy().sum(dim=-1)

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None) -> None:
        return

    def init_from_pulse_checkpoint(self, path: str) -> None:
        """Load ``prior.*`` and ``student_core.decoder.*`` from a PULSE / student checkpoint."""
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"pulse checkpoint not found: {path!r}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(sd, dict):
            raise ValueError("pulse checkpoint must contain a dict model_state_dict or be a raw state dict.")
        if any(k.startswith("module.") for k in sd):
            sd = {k.removeprefix("module."): v for k, v in sd.items()}

        prior_sd = _strip_prefix_keys(sd, "prior.")
        if not prior_sd:
            raise ValueError("pulse checkpoint missing prior.*")
        self.prior.load_state_dict(prior_sd, strict=True)
        self.prior.eval()

        dec_sd = {k[len("student_core.decoder.") :]: v for k, v in sd.items() if k.startswith("student_core.decoder.")}
        if not dec_sd:
            raise ValueError("pulse checkpoint missing student_core.decoder.*")
        self.decoder.load_state_dict(dec_sd, strict=True)
        self.decoder.eval()
        self._sync_parameter_requires_grad()
        print(f"[LatentBottleneckAnyBodyActorCritic] Loaded prior+decoder from {path!r}")

    def init_from_latent_rl_checkpoint(self, path: str, *, load_critic: bool = True, load_std: bool = True) -> None:
        """Load VR latent PPO ``ActorCritic`` weights: actor -> ``latent_actor``, critic -> ``critic``."""
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"latent RL checkpoint not found: {path!r}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        if not isinstance(sd, dict):
            raise ValueError("latent RL checkpoint must contain model_state_dict or be a raw state dict.")
        if any(k.startswith("module.") for k in sd):
            sd = {k.removeprefix("module."): v for k, v in sd.items()}

        actor_sd = _strip_prefix_keys(sd, "actor.")
        if not actor_sd:
            raise ValueError("latent RL checkpoint missing actor.* (expected VR latent ActorCritic).")
        self.latent_actor.load_state_dict(actor_sd, strict=True)
        print(f"[LatentBottleneckAnyBodyActorCritic] Loaded latent_actor from {path!r}")

        if load_critic:
            critic_sd = _strip_prefix_keys(sd, "critic.")
            if critic_sd:
                self.critic.load_state_dict(critic_sd, strict=True)
                print(f"[LatentBottleneckAnyBodyActorCritic] Loaded critic from {path!r}")


    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Resume training from a checkpoint saved from this module."""
        super().load_state_dict(state_dict, strict=strict)
        self._sync_parameter_requires_grad()
        return True
