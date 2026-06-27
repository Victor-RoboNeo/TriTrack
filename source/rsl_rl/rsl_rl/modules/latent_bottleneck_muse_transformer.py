"""MUSE transformer encoder: 1-step command token + H proprio tokens, env-driven goal masking.

Layout (per timestep): the encoder consumes one un-masked command token
(``delta_command + motion_anchor_ori_b``) plus ``H`` per-frame proprio tokens
(``joint_pos + joint_vel + base_ang_vel + last_action``). With ``H=5`` that is ``1+5=6``
modality tokens plus a learned ``[CLS]`` -> 7 tokens total.

Mask routing is env-driven (mirrors SAGE-II): the env's ``goal_mask_history`` (1-step here)
threshold > 0 recovers the boolean. When set, the command token is dropped from attention via
``key_padding_mask`` *and* zeroed pre-projection (defense in depth) so neither the K/V projections
nor downstream attention can be poisoned. The mask probability is set on the env side via the
curriculum (``goal_mask_p``); the policy just consumes whatever bit the env supplies.

Decoder takes ``[z, proprio]``: full proprio history flattened (450-d for G1) by default, or just
the last frame if ``decoder_one_step_proprio=True``.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils.finite_checks import require_all_finite


def _muse_log_sigma_bounds(s_min: float, s_max: float) -> tuple[float, float]:
    smin = max(float(s_min), 1e-12)
    smax = max(float(s_max), smin + 1e-12)
    return math.log(smin), math.log(smax)


def _split_history_terms(
    obs: torch.Tensor, term_sizes: Sequence[int], history_length: int, start_offset: int
) -> tuple[torch.Tensor, int]:
    """Slice ``obs[..., start_offset : ...]`` into a ``[..., history_length, sum(term_sizes)]`` tensor.

    Isaac Lab stacks history per term: each term contributes a contiguous block of
    ``history_length * term_size`` values. Within a term's block the layout is
    ``[t_0_dim_0, ..., t_0_dim_size-1, t_1_dim_0, ...]`` (history-major within the term).
    """
    parts: list[torch.Tensor] = []
    offset = start_offset
    for size in term_sizes:
        size = int(size)
        block = obs[..., offset : offset + size * history_length]
        # block: [..., history_length * size]  ->  [..., history_length, size]
        block = block.reshape(*block.shape[:-1], history_length, size)
        parts.append(block)
        offset += size * history_length
    if len(parts) == 1:
        return parts[0], offset
    return torch.cat(parts, dim=-1), offset


class _MUSETransformerEncoder(nn.Module):
    """Encoder: 1 command token + H proprio tokens -> transformer -> [CLS] -> μ. σ is fixed.

    Mask routing is env-driven. ``mask_per_frame[..., 0] > 0`` recovers the env-side boolean
    (post-normalization 0/1 -> negative/positive). When True, the command token is dropped from
    attention via ``key_padding_mask`` and zeroed pre-projection.
    """

    def __init__(
        self,
        *,
        goal_term_sizes: Sequence[int],
        proprio_term_sizes: Sequence[int],
        mask_term_size: int,
        history_length: int,
        latent_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ffn_dim: int,
        activation: str,
        latent_sigma_min: float,
        latent_sigma_max: float,
        fixed_encoder_std: float,
        predict_sigma: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.goal_term_sizes = tuple(int(s) for s in goal_term_sizes)
        self.proprio_term_sizes = tuple(int(s) for s in proprio_term_sizes)
        self.mask_term_size = int(mask_term_size)
        self.history_length = int(history_length)  # proprio history length
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)
        self.predict_sigma = bool(predict_sigma)

        self.per_frame_goal_dim = sum(self.goal_term_sizes)
        self.per_frame_proprio_dim = sum(self.proprio_term_sizes)

        # Per-modality token embeddings.
        self.goal_proj = nn.Linear(self.per_frame_goal_dim, self.d_model)
        self.proprio_proj = nn.Linear(self.per_frame_proprio_dim, self.d_model)

        # Learned positional embeddings — only proprio is history-rolled, so positional indexing
        # is over the proprio frames. The single goal token gets its own learned embedding via
        # a dedicated entry; modality embedding distinguishes goal vs proprio.
        self.proprio_pos_emb = nn.Parameter(torch.zeros(self.history_length, self.d_model))
        nn.init.normal_(self.proprio_pos_emb, std=0.02)
        self.goal_pos_emb = nn.Parameter(torch.zeros(1, self.d_model))
        nn.init.normal_(self.goal_pos_emb, std=0.02)
        self.modality_emb = nn.Parameter(torch.zeros(2, self.d_model))  # 0=goal, 1=proprio
        nn.init.normal_(self.modality_emb, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))

        self.mu_head = nn.Linear(self.d_model, self.latent_dim)

        # σ: either predicted via a learned head or broadcast as a fixed constant.
        # Predicted-σ requires a regularizer (KL or similar) — without one, σ collapses to the lower
        # clamp and wastes the head. The Kendall-KL term in *_distillation handles that.
        self.fixed_encoder_std = float(fixed_encoder_std)
        lmin, lmax = _muse_log_sigma_bounds(latent_sigma_min, latent_sigma_max)
        self._log_sigma_min = lmin
        self._log_sigma_max = lmax
        self._log_sigma_const = math.log(max(self.fixed_encoder_std, 1e-6))
        if self.predict_sigma:
            self.log_sigma_head = nn.Linear(self.d_model, self.latent_dim)
            # Bias-init log_σ near the fixed-σ default so early-training noise level matches the
            # fixed-σ recipe and BC loss has a comparable starting trajectory.
            nn.init.zeros_(self.log_sigma_head.weight)
            nn.init.constant_(self.log_sigma_head.bias, self._log_sigma_const)
        else:
            self.log_sigma_head = None

        # Obs layout (concat order): [goal (1 step), proprio (H steps), mask (1 step)].
        # Goal/mask have NO Isaac history rolling; only proprio is history-rolled to H.
        self._goal_block_dim = self.per_frame_goal_dim
        self._proprio_block_dim = self.per_frame_proprio_dim * self.history_length
        self._mask_block_dim = self.mask_term_size
        self._expected_obs_dim = (
            self._goal_block_dim + self._proprio_block_dim + self._mask_block_dim
        )

        # Sequence length: [CLS, goal, proprio_0..proprio_{H-1}].
        self.seq_len = 2 + self.history_length

    def expected_obs_dim(self) -> int:
        return self._expected_obs_dim

    def split_obs(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (goal [..., G], proprio_per_frame [..., H, P], mask [..., mask_term_size])."""
        if obs.shape[-1] != self._expected_obs_dim:
            raise RuntimeError(
                f"_MUSETransformerEncoder obs dim mismatch: got {obs.shape[-1]}, "
                f"expected {self._expected_obs_dim}."
            )
        # Goal block (1 step) -> [..., 1, G] -> squeeze to [..., G].
        goal_h, off1 = _split_history_terms(
            obs, self.goal_term_sizes, history_length=1, start_offset=0
        )
        goal = goal_h.squeeze(-2)
        # Proprio (H frames).
        proprio_per_frame, off2 = _split_history_terms(
            obs, self.proprio_term_sizes, history_length=self.history_length, start_offset=off1
        )
        # Mask bit (1 step) -> [..., 1, mask_term_size] -> squeeze to [..., mask_term_size].
        mask_h, _ = _split_history_terms(
            obs, (self.mask_term_size,), history_length=1, start_offset=off2
        )
        mask = mask_h.squeeze(-2)
        return goal, proprio_per_frame, mask

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (μ, log_σ, proprio_per_frame [..., H, P]). log_σ is the broadcast fixed-σ constant.

        The per-frame proprio history is returned so the outer policy can decide whether the
        decoder consumes only the last frame (90-d) or the flattened full history (5×90 = 450-d).
        """
        goal, proprio_per_frame, mask = self.split_obs(obs)
        require_all_finite(proprio_per_frame, "MUSETransformer proprio_per_frame")

        batch_shape = obs.shape[:-1]

        # Env-side mask: ``mask`` is post-normalization (raw 0/1 -> negative/positive after
        # subtracting the running mean). Threshold > 0 recovers the original boolean.
        mask_bool = (mask[..., 0] > 0)  # [...]

        # Defense in depth: zero goal pre-projection on masked steps so the K/V projections of
        # the goal slot cannot poison attention even if key_padding_mask were misapplied.
        goal_clean = torch.where(mask_bool.unsqueeze(-1), torch.zeros_like(goal), goal)

        goal_token = self.goal_proj(goal_clean).unsqueeze(-2)  # [..., 1, D]
        proprio_tokens = self.proprio_proj(proprio_per_frame)  # [..., H, D]

        # Add positional + modality embeddings (broadcast across batch).
        goal_token = goal_token + self.goal_pos_emb.unsqueeze(0) + self.modality_emb[0].view(1, 1, -1)
        proprio_pos = self.proprio_pos_emb.unsqueeze(0)  # [1, H, D]
        proprio_tokens = proprio_tokens + proprio_pos + self.modality_emb[1].view(1, 1, -1)

        # Prepend [CLS]: shape [..., 1, D]
        cls = self.cls_token.expand(*batch_shape, 1, self.d_model)
        tokens = torch.cat([cls, goal_token, proprio_tokens], dim=-2)  # [..., 2+H, D]

        # Build key_padding_mask: True where attention should be skipped.
        # Layout: [CLS, goal, proprio_0..proprio_{H-1}].
        H = self.history_length
        cls_pad = torch.zeros(*batch_shape, 1, dtype=torch.bool, device=obs.device)
        goal_pad = mask_bool.unsqueeze(-1)  # [..., 1]
        proprio_pad = torch.zeros(*batch_shape, H, dtype=torch.bool, device=obs.device)
        key_padding_mask = torch.cat([cls_pad, goal_pad, proprio_pad], dim=-1)  # [..., 2+H]

        # Flatten leading dims for transformer (it expects [B, S, D]).
        tokens_flat = tokens.reshape(-1, self.seq_len, self.d_model)
        kpm_flat = key_padding_mask.reshape(-1, self.seq_len)
        out_flat = self.transformer(tokens_flat, src_key_padding_mask=kpm_flat)
        out = out_flat.reshape(*batch_shape, self.seq_len, self.d_model)

        cls_out = out[..., 0, :]  # [..., D]
        mu = self.mu_head(cls_out)  # [..., latent_dim]
        if self.log_sigma_head is not None:
            log_sigma = self.log_sigma_head(cls_out).clamp(self._log_sigma_min, self._log_sigma_max)
        else:
            log_sigma = torch.full_like(mu, self._log_sigma_const)

        return mu, log_sigma, proprio_per_frame  # proprio_per_frame: [..., H, per_frame_proprio_dim]

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)
        return mu + sigma * eps


class LatentBottleneckMUSETransformer(nn.Module):
    """MUSE student-teacher with transformer encoder + MLP decoder. No prior.

    Encoder layout: 1 un-masked command token + H proprio tokens (+ [CLS]). Env's
    ``goal_mask_history`` (1-step) drops the command token from attention via key_padding_mask
    when set. Curriculum-driven masking probability lives on the env side.
    """

    is_recurrent = False

    def __init__(
        self,
        num_student_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        *,
        proprio_dim: int,  # full history proprio dim (kept for cfg compatibility; not used here)
        history_length: int = 5,
        goal_term_sizes: Sequence[int] = (58, 6),
        proprio_term_sizes: Sequence[int] = (29, 29, 3, 29),
        mask_term_size: int = 1,
        latent_dim: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        decoder_hidden_dims: Sequence[int] = (256, 128),
        teacher_hidden_dims: Sequence[int] = (1024, 1024, 512, 512, 256, 256),
        activation: str = "gelu",
        init_noise_std: float = 0.1,
        latent_predict_std_min: float = 0.001,
        latent_predict_std_max: float = 1.0,
        fixed_encoder_std: float = 0.3,
        predict_sigma: bool = False,
        encoder_dropout: float = 0.0,
        decoder_one_step_proprio: bool = False,
        latent_normalize: bool = False,
        deterministic_encoder: bool = False,
        **kwargs: Any,
    ):
        if kwargs:
            print(
                "LatentBottleneckMUSETransformer.__init__ got unexpected arguments (ignored): "
                + str(list(kwargs.keys()))
            )
        super().__init__()
        del proprio_dim  # not used by the transformer encoder; current-frame proprio is sliced internally.

        self.num_student_obs = int(num_student_obs)
        self.num_actions = int(num_actions)
        self.history_length = int(history_length)
        # Project z onto the unit hypersphere before the decoder. Deterministic regime: decouples
        # direction from magnitude so the decoder can't cheat via scale; makes cosine the natural
        # metric for the smoothness regularizer.
        self.latent_normalize = bool(latent_normalize)
        # Skip the reparameterization step entirely — z := μ on every forward path. Pairs naturally
        # with multi-modal-input framing where the goal is conditioning richness, not output
        # sampling diversity. ``fixed_encoder_std`` is still surfaced for logging only.
        self.deterministic_encoder = bool(deterministic_encoder)

        # Encoder.
        self.transformer_encoder = _MUSETransformerEncoder(
            goal_term_sizes=goal_term_sizes,
            proprio_term_sizes=proprio_term_sizes,
            mask_term_size=mask_term_size,
            history_length=self.history_length,
            latent_dim=int(latent_dim),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            ffn_dim=int(ffn_dim),
            activation=activation,
            latent_sigma_min=float(latent_predict_std_min),
            latent_sigma_max=float(latent_predict_std_max),
            fixed_encoder_std=float(fixed_encoder_std),
            predict_sigma=bool(predict_sigma),
            dropout=float(encoder_dropout),
        )
        # Sanity check obs dim.
        if self.transformer_encoder.expected_obs_dim() != self.num_student_obs:
            raise ValueError(
                f"LatentBottleneckMUSETransformer: expected_obs_dim "
                f"({self.transformer_encoder.expected_obs_dim()}) != num_student_obs "
                f"({self.num_student_obs}). Check goal_term_sizes={goal_term_sizes}, "
                f"proprio_term_sizes={proprio_term_sizes}, mask_term_size={mask_term_size}, "
                f"history_length={history_length} (proprio-only; goal/mask are 1-step)."
            )

        # Decoder input depends on ``decoder_one_step_proprio``:
        #   False (default): full proprio history flattened to (H * per_frame_proprio_dim) — matches
        #     the MLP-MUSE decoder shape, so encoder + decoder split is the same as MLP-MUSE.
        #   True : last-frame proprio only (per_frame_proprio_dim). Latent z carries history info.
        per_frame_proprio_dim = self.transformer_encoder.per_frame_proprio_dim
        self.per_frame_proprio_dim = per_frame_proprio_dim
        self.decoder_one_step_proprio = bool(decoder_one_step_proprio)
        decoder_proprio_dim = (
            per_frame_proprio_dim if self.decoder_one_step_proprio
            else per_frame_proprio_dim * self.history_length
        )
        self.decoder_proprio_dim = decoder_proprio_dim

        act_cls = getattr(nn, activation.upper(), nn.GELU)
        dec_layers: list[nn.Module] = []
        dec_in = int(latent_dim) + decoder_proprio_dim
        for h in decoder_hidden_dims:
            dec_layers.append(nn.Linear(dec_in, h))
            dec_layers.append(act_cls())
            dec_in = h
        dec_layers.append(nn.Linear(dec_in, self.num_actions))
        self.decoder = nn.Sequential(*dec_layers)

        # Algorithm reads ``policy.student.parameters()`` for the optimizer.
        self.student: nn.Module = nn.ModuleList([self.transformer_encoder, self.decoder])

        # Teacher MLP.
        teacher_layers: list[nn.Module] = []
        hidden = list(teacher_hidden_dims) or [256, 256]
        teacher_act = getattr(nn, "elu".upper())
        teacher_layers.append(nn.Linear(num_teacher_obs, hidden[0]))
        teacher_layers.append(teacher_act())
        for i in range(len(hidden)):
            if i == len(hidden) - 1:
                teacher_layers.append(nn.Linear(hidden[i], self.num_actions))
            else:
                teacher_layers.append(nn.Linear(hidden[i], hidden[i + 1]))
                teacher_layers.append(teacher_act())
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()
        self.loaded_teacher = False

        init_std = max(float(init_noise_std), 1.0e-6)
        self.std = nn.Parameter(init_std * torch.ones(self.num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args = False  # type: ignore[assignment]

        # Expose log-σ bounds for algorithm-side logging.
        self.latent_log_sigma_min = self.transformer_encoder._log_sigma_min
        self.latent_log_sigma_max = self.transformer_encoder._log_sigma_max

        print(
            f"[LatentBottleneckMUSETransformer] num_student_obs={self.num_student_obs}, "
            f"history_length={self.history_length} (proprio; goal/mask are 1-step), "
            f"goal_term_sizes={list(goal_term_sizes)}, proprio_term_sizes={list(proprio_term_sizes)}, "
            f"mask_term_size={mask_term_size}, latent_dim={latent_dim}, d_model={d_model}, "
            f"nhead={nhead}, num_layers={num_layers}, ffn_dim={ffn_dim}, "
            f"decoder_hidden={list(decoder_hidden_dims)}, teacher_hidden={list(teacher_hidden_dims)}, "
            f"predict_sigma={predict_sigma}, fixed_encoder_std={fixed_encoder_std}, "
            f"per_frame_proprio_dim={per_frame_proprio_dim}, "
            f"decoder_one_step_proprio={self.decoder_one_step_proprio}, decoder_proprio_dim={decoder_proprio_dim}, "
            f"latent_normalize={self.latent_normalize}, deterministic_encoder={self.deterministic_encoder}"
        )

    def reset(self, dones=None, hidden_states=None) -> None:
        return

    def _maybe_normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        if self.latent_normalize:
            return nn.functional.normalize(z, p=2.0, dim=-1, eps=1e-8)
        return z

    def _shape_proprio_for_decoder(self, proprio_per_frame: torch.Tensor) -> torch.Tensor:
        """Slice last frame (90-d) or flatten full history (450-d) per ``decoder_one_step_proprio``."""
        if self.decoder_one_step_proprio:
            return proprio_per_frame[..., -1, :]
        # Flatten the last two dims: [..., H, P] -> [..., H*P].
        return proprio_per_frame.reshape(*proprio_per_frame.shape[:-2], -1)

    def _decode(self, z: torch.Tensor, proprio_per_frame: torch.Tensor) -> torch.Tensor:
        proprio = self._shape_proprio_for_decoder(proprio_per_frame)
        return self.decoder(torch.cat([z, proprio], dim=-1))

    def update_distribution(self, observations: torch.Tensor) -> None:
        mu, log_sigma, proprio_per_frame = self.transformer_encoder.encode(observations)
        z = self._maybe_normalize_latent(mu)
        actions_mean = self._decode(z, proprio_per_frame)
        std = self.std.expand_as(actions_mean).clamp(min=1.0e-6)
        self.distribution = Normal(actions_mean, std)

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        mu, log_sigma, proprio_per_frame = self.transformer_encoder.encode(observations)
        use_mean = (
            self.deterministic_encoder
            or bool(getattr(self, "deterministic_latent", False))
            or not self.training
        )
        if use_mean:
            z = mu
        else:
            z = self.transformer_encoder.reparameterize(mu, log_sigma)
        z = self._maybe_normalize_latent(z)
        return self._decode(z, proprio_per_frame)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        mu, _, proprio_per_frame = self.transformer_encoder.encode(observations)
        z = self._maybe_normalize_latent(mu)
        return self._decode(z, proprio_per_frame)

    def forward_for_update(
        self,
        observations: torch.Tensor,
        sample_z: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, μ, log_σ) — no prior. μ returned post-normalization when enabled, so the
        algorithm's smoothness regularizer operates on the same latent geometry the decoder sees."""
        mu, log_sigma, proprio_per_frame = self.transformer_encoder.encode(observations)
        if self.deterministic_encoder or not (sample_z and self.training):
            z = mu
        else:
            z = self.transformer_encoder.reparameterize(mu, log_sigma)
        z = self._maybe_normalize_latent(z)
        mu = self._maybe_normalize_latent(mu)
        action = self._decode(z, proprio_per_frame)
        if not torch.isfinite(action).all():
            warnings.warn("LatentBottleneckMUSETransformer: non-finite action; replacing with 0.")
            action = torch.nan_to_num(action)
        return action, mu, log_sigma

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
        # Stage-1 teacher checkpoint: ``actor.*`` keys.
        if any("actor." in key for key in state_dict.keys()):
            teacher_sd = {
                key.replace("actor.", ""): value
                for key, value in state_dict.items()
                if "actor." in key
            }
            self.teacher.load_state_dict(teacher_sd, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            return False
        # Resume: full module checkpoint.
        if any(
            k.startswith("transformer_encoder.") or k.startswith("decoder.") or k.startswith("teacher.")
            for k in state_dict.keys()
        ):
            super().load_state_dict(state_dict, strict=False)
            self.loaded_teacher = True
            self.teacher.eval()
            return True
        raise ValueError(
            "state_dict must contain actor.*, or transformer_encoder.*/decoder.*/teacher.* keys."
        )

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None) -> None:
        return
