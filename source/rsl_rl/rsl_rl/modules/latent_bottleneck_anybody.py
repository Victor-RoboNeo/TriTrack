from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.latent_bottleneck_pulse import _PULSELatentBottleneck, _PULSEPrior
from rsl_rl.utils import resolve_nn_activation
from rsl_rl.utils.finite_checks import require_all_finite



class LatentBottleneckAnyBody(nn.Module):
    """Residual-latent distillation policy with frozen PULSE prior/decoder.

    The student predicts a residual latent:
        mu_student = mu_prior + delta_mu(obs)
    where prior and decoder are loaded from pretrained PULSE and frozen.
    """

    is_recurrent = False

    def __init__(
        self,
        num_student_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        *,
        latent_dim: int = 64,
        encoder_hidden_dims: Sequence[int] = (512, 512),
        decoder_hidden_dims: Sequence[int] = (512, 512),
        prior_hidden_dims: Sequence[int] = (256, 256),
        activation: str = "elu",
        init_noise_std: float = 0.1,
        proprio_dim: int = 450,
        teacher_encoder_obs_dim: int | None = None,
        teacher_encoder_hidden_dims: Sequence[int] | None = None,
        latent_predict_std_min: float | None = None,
        latent_predict_std_max: float | None = None,
        fixed_encoder_std: float | None = None,
        fixed_prior_std: float | None = None,
        
        **kwargs: Any,
    ):
        if kwargs:
            print(
                "LatentBottleneckAnyBody.__init__ got unexpected arguments (ignored): "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.num_student_obs = int(num_student_obs)
        self.num_teacher_obs = int(num_teacher_obs)
        self.num_actions = int(num_actions)
        self.latent_dim = int(latent_dim)
        self.proprio_dim = int(proprio_dim)
        self.loaded_teacher = False
        self._decoder_frozen = False

        if self.proprio_dim >= self.num_teacher_obs:
            raise ValueError(
                f"proprio_dim ({self.proprio_dim}) must be < num_teacher_obs ({self.num_teacher_obs})."
            )
        _te = int(teacher_encoder_obs_dim) if teacher_encoder_obs_dim is not None else self.num_teacher_obs
        if _te <= self.proprio_dim:
            raise ValueError(
                f"teacher_encoder_obs_dim ({_te}) must be > proprio_dim ({self.proprio_dim})."
            )
        self.teacher_encoder_obs_dim = _te
        self._teacher_goal_in_dim = self.num_teacher_obs - self.proprio_dim
        self._teacher_goal_out_dim = self.teacher_encoder_obs_dim - self.proprio_dim
        if self._teacher_goal_in_dim <= 0 or self._teacher_goal_out_dim <= 0:
            raise ValueError("Invalid goal/proprio split for teacher observations.")
        self._teacher_goal_adapter: nn.Module | None
        if self._teacher_goal_in_dim != self._teacher_goal_out_dim:
            self._teacher_goal_adapter = nn.Linear(self._teacher_goal_in_dim, self._teacher_goal_out_dim)
            nn.init.xavier_uniform_(self._teacher_goal_adapter.weight)
            nn.init.zeros_(self._teacher_goal_adapter.bias)
            for p in self._teacher_goal_adapter.parameters():
                p.requires_grad = False
        else:
            self._teacher_goal_adapter = None

        # Action noise head (for logging/interface compatibility).
        self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args = False  # type: ignore[assignment]

        if self.proprio_dim >= self.num_student_obs:
            raise ValueError(
                f"Residual latent policy requires proprio_dim ({self.proprio_dim}) < num_student_obs ({self.num_student_obs})."
            )
        act_fn = resolve_nn_activation(activation)

        # Human/keypoint dimensionality (non-proprio part of student obs).
        self.human_dim = self.num_student_obs - self.proprio_dim

        # Frozen teacher core from PULSE checkpoint: encoder/decoder widths match the pretrained run
        # (often smaller than the trainable residual encoder below).
        _teacher_enc_h = (
            list(teacher_encoder_hidden_dims)
            if teacher_encoder_hidden_dims is not None
            else list(encoder_hidden_dims)
        )
        self.teacher_core = _PULSELatentBottleneck(
            obs_dim=self.teacher_encoder_obs_dim,
            proprio_dim=self.proprio_dim,
            num_actions=self.num_actions,
            latent_dim=self.latent_dim,
            encoder_hidden=_teacher_enc_h,
            decoder_hidden=list(decoder_hidden_dims),
            activation=activation,
            latent_sigma_min=latent_predict_std_min,
            latent_sigma_max=latent_predict_std_max,
            fixed_encoder_std=fixed_encoder_std,
        )
        # Frozen prior R(proprio)->(mu_p, log_sigma_p).
        self.prior = _PULSEPrior(
            proprio_dim=self.proprio_dim,
            latent_dim=self.latent_dim,
            hidden_dims=list(prior_hidden_dims),
            activation=activation,
            latent_sigma_min=latent_predict_std_min,
            latent_sigma_max=latent_predict_std_max,
            fixed_prior_std=fixed_prior_std,
        )
        # Frozen decoder D([z, proprio])->action.
        dec_in = self.latent_dim + self.proprio_dim
        dec_layers: list[nn.Module] = []
        for h in decoder_hidden_dims:
            dec_layers.append(nn.Linear(dec_in, h))
            dec_layers.append(act_fn)
            dec_in = h
        dec_layers.append(nn.Linear(dec_in, self.num_actions))
        self.decoder = nn.Sequential(*dec_layers)

        # Trainable residual encoder: delta_z = E_residual(obs).
        enc_in = self.num_student_obs
        body_layers: list[nn.Module] = []
        for h in encoder_hidden_dims:
            body_layers.append(nn.Linear(enc_in, h))
            body_layers.append(act_fn)
            enc_in = h
        self.residual_encoder_body = nn.Sequential(*body_layers)
        self.residual_mu = nn.Linear(enc_in, self.latent_dim)

        # Only residual encoder is optimized in this mode.
        self.student = nn.ModuleList([self.residual_encoder_body, self.residual_mu])
        self.freeze_decoder()
        self._freeze_teacher_prior()

        print(
            f"[LatentBottleneckAnyBody] num_student_obs={self.num_student_obs}, "
            f"num_teacher_obs={self.num_teacher_obs}, teacher_encoder_obs_dim={self.teacher_encoder_obs_dim}, "
            f"proprio_dim={self.proprio_dim}, latent_dim={self.latent_dim}, "
            f"student_residual_encoder_hidden={list(encoder_hidden_dims)}, "
            f"teacher_core_encoder_hidden={_teacher_enc_h}, decoder_hidden={list(decoder_hidden_dims)}, "
            f"prior_hidden={list(prior_hidden_dims)}"
        )

    def _project_teacher_obs_for_core(self, teacher_observations: torch.Tensor) -> torch.Tensor:
        """Map env teacher obs to the size expected by ``teacher_core`` (goal block only; proprio tail unchanged)."""
        if teacher_observations.shape[-1] != self.num_teacher_obs:
            raise RuntimeError(
                f"Teacher obs dim mismatch: got {teacher_observations.shape[-1]}, expected {self.num_teacher_obs}."
            )
        if self._teacher_goal_adapter is None:
            return teacher_observations
        pd = self.proprio_dim
        goal_in = teacher_observations[..., :-pd]
        proprio = teacher_observations[..., -pd:]
        goal_out = self._teacher_goal_adapter(goal_in)
        return torch.cat([goal_out, proprio], dim=-1)

    def _freeze_teacher_prior(self) -> None:
        if self._teacher_goal_adapter is not None:
            self._teacher_goal_adapter.eval()
        if hasattr(self, "teacher_core"):
            for p in self.teacher_core.parameters():
                p.requires_grad = False
            self.teacher_core.eval()
        if hasattr(self, "prior"):
            for p in self.prior.parameters():
                p.requires_grad = False
            self.prior.eval()

    def _enforce_frozen_modules(self) -> None:
        """Keep teacher/prior/decoder frozen and in eval mode."""
        if self._teacher_goal_adapter is not None:
            self._teacher_goal_adapter.eval()
        for module in (self.teacher_core, self.prior, self.decoder):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    def freeze_decoder(self) -> None:
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.decoder.eval()
        self._decoder_frozen = True

    def unfreeze_decoder(self) -> None:
        # Decoder remains frozen by design; kept for algorithm compatibility.
        self._decoder_frozen = True

    def train(self, mode: bool = True):
        # Preserve standard behavior for trainable residual encoder while
        # forcing frozen modules to remain eval/frozen under runner.train_mode().
        super().train(mode)
        self._enforce_frozen_modules()
        return self

    @property
    def decoder_frozen(self) -> bool:
        return bool(self._decoder_frozen)

    def _split_student_obs(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Student obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        goal = observations[..., : self.num_student_obs - self.proprio_dim]
        proprio = observations[..., -self.proprio_dim :]
        return goal, proprio


    def reset(self, dones=None, hidden_states=None) -> None:
        return

    def encode_residual_latent(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Student obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        proprio = observations[..., -self.proprio_dim :]
        with torch.no_grad():
            mu_p, log_sigma_p = self.prior(proprio)
        require_all_finite(observations, "LatentBottleneckAnyBody student observations")
        #pdb.set_trace()
        h = self.residual_encoder_body(observations)
        delta_mu = self.residual_mu(h)
        mu_student = mu_p + delta_mu
        return mu_student, mu_p, log_sigma_p

    def _decode_with_frozen_decoder(self, z: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, proprio], dim=-1)
        return self.decoder(x)

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        _, proprio = self._split_student_obs(observations)
        mu_student, _, _ = self.encode_residual_latent(observations)
        actions = self._decode_with_frozen_decoder(mu_student, proprio)
        return actions

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        _, proprio = self._split_student_obs(observations)
        #observations[:,90:93] = 0.0 # TODO: remove this # manually set the base velocity to 0
        mu_student, _, _ = self.encode_residual_latent(observations)
        return self._decode_with_frozen_decoder(mu_student, proprio)

    def get_teacher_targets(
        self,
        teacher_observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            z = self._project_teacher_obs_for_core(teacher_observations)
            actions, mu, _, _ = self.teacher_core(z, sample_z=False)
        return mu.detach(), actions.detach()

    def evaluate(self, teacher_observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self._project_teacher_obs_for_core(teacher_observations)
            actions, _, _, _ = self.teacher_core(z, sample_z=False)
        return actions

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
        # Full residual-student checkpoint.
        if any(k.startswith("teacher_core.") or k.startswith("residual_encoder_body.") for k in state_dict.keys()):
            super().load_state_dict(state_dict, strict=strict)
            self._freeze_teacher_prior()
            self.freeze_decoder()
            self.loaded_teacher = True
            return True

        # PULSE checkpoint: student_core.* (+ optional prior.*).
        if any(k.startswith("student_core.") for k in state_dict.keys()):
            teacher_sd = {
                k[len("student_core.") :]: v for k, v in state_dict.items() if k.startswith("student_core.")
            }
            try:
                self.teacher_core.load_state_dict(teacher_sd, strict=True)
            except RuntimeError as e:
                raise RuntimeError(
                    "Failed to load PULSE ``student_core`` into ``teacher_core``. "
                    "If the teacher used ``fixed_encoder_std`` / ``fixed_prior_std`` (no σ heads in the "
                    "checkpoint), set the same ``fixed_encoder_std`` / ``fixed_prior_std`` and "
                    "``latent_predict_std_{min,max}`` on ``RslRlAnyBodyLatentDistillationCfg`` as in the "
                    "teacher run's ``params/agent.yaml``."
                ) from e

            # Prior parameters were stored separately during PULSE training.
            if any(k.startswith("prior.") for k in state_dict.keys()):
                prior_sd = {
                    k[len("prior.") :]: v for k, v in state_dict.items() if k.startswith("prior.")
                }
                try:
                    self.prior.load_state_dict(prior_sd, strict=True)
                except RuntimeError as e:
                    raise RuntimeError(
                        "Failed to load PULSE ``prior``. "
                        "If the teacher used ``fixed_prior_std``, set ``fixed_prior_std`` (and σ bounds) "
                        "on the two-bottleneck policy cfg to match the teacher run."
                    ) from e
            self.decoder.load_state_dict(self.teacher_core.decoder.state_dict(), strict=True)
            self._freeze_teacher_prior()
            self.freeze_decoder()
            self.loaded_teacher = True
            # False => checkpoint was teacher init, not residual student resume.
            return False

        raise ValueError(
            "state_dict must contain either full residual-student keys or PULSE student_core keys."
        )

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None) -> None:
        return


