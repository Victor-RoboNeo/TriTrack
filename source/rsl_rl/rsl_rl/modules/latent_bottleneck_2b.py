"""Two-bottleneck distillation: teacher = PULSE student_core (frozen), student = encoder + decoder.

Teacher and student both use the same observation (policy obs). No mask.
- Stage 1: decoder loaded from teacher and frozen; encoder trained to catch up (action loss).
- Stage 2: decoder unfrozen; train both with strong action loss + weak latent loss (teacher collects).
- Stage 3: online distillation (student collects; targets still from teacher).
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.latent_bottleneck_pulse import _PULSELatentBottleneck


class LatentBottleneck2B(nn.Module):
    """
    Two-bottleneck policy: teacher = loaded PULSE student_core (frozen), student = same arch.

    Both teacher and student consume the same observation (policy / student obs). No mask.
    - teacher_core: PULSE student_core loaded from checkpoint, frozen. Used for action and latent targets.
    - student_core: Same architecture; decoder can be copied from teacher and frozen (stage 1), then unfrozen (stage 2).
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
        activation: str = "elu",
        init_noise_std: float = 0.1,
        teacher_encoder_obs_dim: int | None = None,
        teacher_encoder_hidden_dims: Sequence[int] | None = None,
        **kwargs: Any,
    ):
        if kwargs:
            print(
                "LatentBottleneck2B.__init__ got unexpected arguments (ignored): "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.num_student_obs = int(num_student_obs)
        self.num_teacher_obs = int(num_teacher_obs)
        self.num_actions = int(num_actions)
        self.latent_dim = int(latent_dim)
        self.proprio_dim = int(proprio_dim)
        if self.proprio_dim >= self.num_student_obs:
            raise ValueError(
                f"LatentBottleneck2B: proprio_dim ({proprio_dim}) must be < num_student_obs ({num_student_obs})."
            )
        if self.proprio_dim >= self.num_teacher_obs:
            raise ValueError(
                f"LatentBottleneck2B: proprio_dim ({proprio_dim}) must be < num_teacher_obs ({num_teacher_obs})."
            )

        _te = int(teacher_encoder_obs_dim) if teacher_encoder_obs_dim is not None else self.num_teacher_obs
        if _te <= self.proprio_dim:
            raise ValueError(
                f"LatentBottleneck2B: teacher_encoder_obs_dim ({_te}) must be > proprio_dim ({self.proprio_dim})."
            )
        self.teacher_encoder_obs_dim = _te
        self._teacher_goal_in_dim = self.num_teacher_obs - self.proprio_dim
        self._teacher_goal_out_dim = self.teacher_encoder_obs_dim - self.proprio_dim
        if self._teacher_goal_in_dim <= 0 or self._teacher_goal_out_dim <= 0:
            raise ValueError("LatentBottleneck2B: invalid goal/proprio split for teacher observations.")
        self._teacher_goal_adapter: nn.Module | None
        if self._teacher_goal_in_dim != self._teacher_goal_out_dim:
            self._teacher_goal_adapter = nn.Linear(self._teacher_goal_in_dim, self._teacher_goal_out_dim)
            nn.init.xavier_uniform_(self._teacher_goal_adapter.weight)
            nn.init.zeros_(self._teacher_goal_adapter.bias)
            for p in self._teacher_goal_adapter.parameters():
                p.requires_grad = False
        else:
            self._teacher_goal_adapter = None

        _teacher_enc_h = (
            list(teacher_encoder_hidden_dims)
            if teacher_encoder_hidden_dims is not None
            else list(encoder_hidden_dims)
        )

        # Teacher: PULSE-style bottleneck (encoder=[goal, proprio], decoder=[z, proprio]), loaded from checkpoint, frozen.
        self.teacher_core = _PULSELatentBottleneck(
            obs_dim=self.teacher_encoder_obs_dim,
            proprio_dim=self.proprio_dim,
            num_actions=self.num_actions,
            latent_dim=self.latent_dim,
            encoder_hidden=_teacher_enc_h,
            decoder_hidden=list(decoder_hidden_dims),
            activation=activation,
        )

        # Student: wider encoder ok; decoder init often copied from teacher after load.
        self.student_core = _PULSELatentBottleneck(
            obs_dim=self.num_student_obs,
            proprio_dim=self.proprio_dim,
            num_actions=self.num_actions,
            latent_dim=self.latent_dim,
            encoder_hidden=list(encoder_hidden_dims),
            decoder_hidden=list(decoder_hidden_dims),
            activation=activation,
        )
        self.student: nn.Module = self.student_core

        self.loaded_teacher = False
        self._decoder_frozen = False  # True during stage 1 (encoder catch-up).

        # Action distribution (for logging / interface compatibility).
        self.std = nn.Parameter(init_noise_std * torch.ones(self.num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args = False  # type: ignore[assignment]

        print(
            f"[LatentBottleneck2B] num_student_obs={self.num_student_obs}, num_teacher_obs={self.num_teacher_obs}, "
            f"teacher_encoder_obs_dim={self.teacher_encoder_obs_dim}, "
            f"proprio_dim={self.proprio_dim}, latent_dim={latent_dim}, "
            f"student_encoder_hidden={list(encoder_hidden_dims)}, teacher_encoder_hidden={_teacher_enc_h}, "
            f"decoder_hidden={list(decoder_hidden_dims)}"
        )

    def _project_teacher_obs_for_core(self, teacher_observations: torch.Tensor) -> torch.Tensor:
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

    def freeze_decoder(self) -> None:
        """Freeze student decoder (stage 1: encoder catch-up)."""
        for p in self.student_core.decoder.parameters():
            p.requires_grad = False
        self._decoder_frozen = True

    def unfreeze_decoder(self) -> None:
        """Unfreeze student decoder (stage 2: joint tuning)."""
        for p in self.student_core.decoder.parameters():
            p.requires_grad = True
        self._decoder_frozen = False

    @property
    def decoder_frozen(self) -> bool:
        return self._decoder_frozen

    def get_teacher_targets(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (teacher_action, teacher_mu) for the given obs. No grad."""
        with torch.no_grad():
            z = self._project_teacher_obs_for_core(obs)
            action, mu, _log_sigma, _z = self.teacher_core(z, sample_z=False)
        return action.detach(), mu.detach()

    def reset(self, dones=None, hidden_states=None) -> None:
        return

    def update_distribution(self, observations: torch.Tensor) -> None:
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Student obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        actions_mean, _, _, _ = self.student_core(observations, sample_z=False)
        std = self.std.expand_as(actions_mean)
        self.distribution = Normal(actions_mean, std)

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Student obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        actions, _, _, _ = self.student_core(observations, sample_z=True)
        self.update_distribution(observations)
        return actions

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.num_student_obs:
            raise RuntimeError(
                f"Student obs dim mismatch: got {observations.shape[-1]}, expected {self.num_student_obs}."
            )
        actions, _, _, _ = self.student_core(observations, sample_z=False)
        return actions

    def evaluate(self, teacher_observations: torch.Tensor) -> torch.Tensor:
        """Teacher action; for 2b teacher uses same obs as student (policy obs). Unused arg for API compatibility."""
        with torch.no_grad():
            z = self._project_teacher_obs_for_core(teacher_observations)
            action, _, _, _ = self.teacher_core(z, sample_z=False)
        return action

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
        """
        - PULSE checkpoint (student_core.*, teacher.*): load student_core into teacher_core; then copy
          teacher_core decoder into student_core and freeze student decoder. Student encoder stays random.
        - 2b checkpoint (teacher_core.*, student_core.*): load full state, return True.
        """
        # Case 1: full 2b checkpoint (or freshly initialized 2b model in DDP broadcast).
        # Contains both teacher_core.* and student_core.* keys – just load directly.
        if any("teacher_core." in key for key in state_dict.keys()):
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            # Teacher is always frozen.
            for p in self.teacher_core.parameters():
                p.requires_grad = False
            self.teacher_core.eval()
            return True

        # Case 2: PULSE checkpoint – only student_core.* keys.
        if any(key.startswith("student_core.") for key in state_dict.keys()):
            # PULSE checkpoint: student_core -> our teacher_core; then init student from teacher (decoder copy + freeze).
            teacher_sd = {}
            for k, v in state_dict.items():
                if k.startswith("student_core."):
                    teacher_sd["teacher_core." + k[len("student_core.") :]] = v
            self.loaded_teacher = True
            self.teacher_core.load_state_dict(
                {k.replace("teacher_core.", ""): v for k, v in teacher_sd.items()},
                strict=True,
            )
            for p in self.teacher_core.parameters():
                p.requires_grad = False
            self.teacher_core.eval()

            # Copy teacher decoder into student decoder; leave student encoder as-is (random init).
            for st_name, th_name in [
                ("decoder", "decoder"),
            ]:
                st_mod = getattr(self.student_core, st_name)
                th_mod = getattr(self.teacher_core, st_name)
                st_mod.load_state_dict(th_mod.state_dict(), strict=True)
            self.freeze_decoder()
            return True

        raise ValueError(
            "state_dict does not contain teacher_core, student_core, or (PULSE) student_core parameters."
        )

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None) -> None:
        return
