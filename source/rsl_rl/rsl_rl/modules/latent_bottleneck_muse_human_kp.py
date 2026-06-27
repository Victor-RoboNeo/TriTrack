"""MUSE-Human-KP partial tracker: KP + human-motion modalities sharing a MUSE transformer body.

Built on :class:`LatentBottleneckMUSEKp`. Adds a third input modality — human-motion (SOMA-style
joint positions packed per-body the same way KP is) — alongside the existing G1 KP modality. The
shared transformer backbone + MLP decoder + frozen MLP teacher are unchanged.

Token sequence (length 1 + N_kp + J_h + H, e.g. 1 + 14 + 22 + 5 = 42):

    [CLS, kp_0, ..., kp_{N_kp-1}, human_0, ..., human_{J_h-1}, proprio_0, ..., proprio_{H-1}]

Modality slots in ``modality_emb`` (3 rows): 0 = KP, 1 = proprio, 2 = human. Slots 0 / 1 are
shape-identical to the MUSE-KP encoder so checkpoints warmstart cleanly; slot 2 is new.

Obs layout (per env):
  [0 .. kp_block)             kp_lookahead          (L * N_kp * 3; row-major [L, N_kp, 3]; NaN at masked bodies)
  [..kp_mask_block)           kp_mask_lookahead     (L * N_kp; 1.0=masked, 0.0=visible)
  [..human_block)             human_lookahead       (L * J_h * 3; same packing as kp)
  [..human_mask_block)        human_mask_lookahead  (L * J_h; same semantics as kp_mask)
  [..proprio_block)           proprio history-H     (MUSE layout: joint_pos | joint_vel | base_ang_vel | actions)

Modality override for paired training
-------------------------------------
:meth:`encode` accepts ``modality_override``:

- ``None`` (default): both modalities use env-supplied masks (i.e. partial visibility per modality).
- ``"kp"``: force-mask all human tokens (encoder only sees KP).
- ``"human"``: force-mask all KP tokens (encoder only sees human).

This is how the paired-forward training in :class:`MuseHumanKpDistillation` produces two latents
``z_kp`` and ``z_human`` from the same obs for the cross-modality alignment loss.

Warmstart
---------
``load_state_dict`` accepts (in order of preference):

1. **MUSE-Human-KP resume** — full load (contains ``transformer_encoder.human_proj.*``).
2. **MUSE-KP checkpoint** — copies KP layers + shared backbone + decoder + teacher; human-specific
   layers (``human_proj``, ``human_body_id_emb``, ``modality_emb`` slot [2]) keep random init.
   The ``modality_emb`` of shape (2, D) in the source is copied into the first two rows of the
   (3, D) target; row 2 keeps its init.
3. **MUSE-Transformer / PHC+ ``actor.*``** — same handling as :class:`LatentBottleneckMUSEKp`.

Freeze modes
------------
Extends :class:`LatentBottleneckMUSEKp`'s 3 modes with one new mode tailored to this branch's
warmup phase:

- ``"none"``: all encoder + decoder train.
- ``"decoder_only"``: decoder frozen; encoder (KP + human + shared) trains.
- ``"decoder_plus_shared_encoder"``: decoder + shared encoder pieces frozen; KP-specific +
  human-specific encoder layers train.
- ``"human_only"``: decoder + shared + KP-specific layers frozen; ONLY human-specific layers
  (``human_proj``, ``human_body_id_emb``, ``modality_emb`` slot [2]) train. Use for the warmup
  phase after warmstarting from a MUSE-KP checkpoint.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.latent_bottleneck_muse_transformer import (
    _muse_log_sigma_bounds,
    _split_history_terms,
)
from rsl_rl.utils.finite_checks import require_all_finite


_VALID_FREEZE_MODES = (
    "none",
    "decoder_only",
    "decoder_plus_shared_encoder",
    "human_only",
)

ModalityOverride = Literal["kp", "human", None]


class _MUSEHumanKpTransformerEncoder(nn.Module):
    """Two-modality (KP + human) token encoder with MUSE-Transformer body.

    Shared encoder pieces (``transformer``, ``proprio_proj``, ``proprio_pos_emb``, ``cls_token``,
    ``mu_head``) are shape-identical to :class:`_MUSEKpTransformerEncoder` so MUSE-KP weights warmstart
    via ``load_state_dict``. KP-specific layers (``kp_proj``, ``body_id_emb``) also shape-match the
    MUSE-KP names so they load directly. The new human-specific layers (``human_proj``,
    ``human_body_id_emb``) have no warmstart source.
    """

    def __init__(
        self,
        *,
        kp_n_bodies: int,
        kp_lookahead_steps: int,
        human_n_joints: int,
        human_lookahead_steps: int,
        proprio_term_sizes: Sequence[int],
        proprio_history_length: int,
        latent_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ffn_dim: int,
        activation: str,
        latent_sigma_min: float,
        latent_sigma_max: float,
        fixed_encoder_std: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.kp_n_bodies = int(kp_n_bodies)
        self.kp_lookahead_steps = int(kp_lookahead_steps)
        self.human_n_joints = int(human_n_joints)
        self.human_lookahead_steps = int(human_lookahead_steps)
        self.proprio_term_sizes = tuple(int(s) for s in proprio_term_sizes)
        self.proprio_history_length = int(proprio_history_length)
        self.per_frame_proprio_dim = sum(self.proprio_term_sizes)
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)

        # KP-specific layers (mirror _MUSEKpTransformerEncoder names so warmstart loads cleanly).
        self.kp_proj = nn.Linear(self.kp_lookahead_steps * 3, self.d_model)
        nn.init.normal_(self.kp_proj.weight, std=0.02)
        nn.init.zeros_(self.kp_proj.bias)
        self.body_id_emb = nn.Embedding(self.kp_n_bodies, self.d_model)
        nn.init.normal_(self.body_id_emb.weight, std=0.02)

        # Human-specific layers (NEW — no warmstart source).
        self.human_proj = nn.Linear(self.human_lookahead_steps * 3, self.d_model)
        nn.init.normal_(self.human_proj.weight, std=0.02)
        nn.init.zeros_(self.human_proj.bias)
        self.human_body_id_emb = nn.Embedding(self.human_n_joints, self.d_model)
        nn.init.normal_(self.human_body_id_emb.weight, std=0.02)

        # Shared encoder pieces (names match _MUSEKpTransformerEncoder). modality_emb expanded
        # from 2 rows -> 3 rows: 0 = KP, 1 = proprio, 2 = human. Warmstart copies rows [0, 1] from
        # the 2-row source; row [2] keeps the small random init.
        self.proprio_proj = nn.Linear(self.per_frame_proprio_dim, self.d_model)
        self.proprio_pos_emb = nn.Parameter(torch.zeros(self.proprio_history_length, self.d_model))
        nn.init.normal_(self.proprio_pos_emb, std=0.02)
        self.modality_emb = nn.Parameter(torch.zeros(3, self.d_model))
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

        self.fixed_encoder_std = float(fixed_encoder_std)
        lmin, lmax = _muse_log_sigma_bounds(latent_sigma_min, latent_sigma_max)
        self._log_sigma_min = lmin
        self._log_sigma_max = lmax
        self._log_sigma_const = math.log(max(self.fixed_encoder_std, 1e-6))

        # Obs offsets: [kp, kp_mask, human, human_mask, proprio].
        self._kp_block_dim = self.kp_lookahead_steps * self.kp_n_bodies * 3
        self._kp_mask_dim = self.kp_lookahead_steps * self.kp_n_bodies
        self._human_block_dim = self.human_lookahead_steps * self.human_n_joints * 3
        self._human_mask_dim = self.human_lookahead_steps * self.human_n_joints
        self._proprio_block_dim = self.per_frame_proprio_dim * self.proprio_history_length
        self._expected_obs_dim = (
            self._kp_block_dim
            + self._kp_mask_dim
            + self._human_block_dim
            + self._human_mask_dim
            + self._proprio_block_dim
        )

        self.seq_len = (
            1 + self.kp_n_bodies + self.human_n_joints + self.proprio_history_length
        )

        body_ids = torch.arange(self.kp_n_bodies, dtype=torch.long)
        human_ids = torch.arange(self.human_n_joints, dtype=torch.long)
        self.register_buffer("_body_ids", body_ids.contiguous(), persistent=False)
        self.register_buffer("_human_ids", human_ids.contiguous(), persistent=False)

    def expected_obs_dim(self) -> int:
        return self._expected_obs_dim

    def split_obs(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (kp [..., N_kp, L*3], kp_mask [..., N_kp], human [..., J_h, L_h*3],
        human_mask [..., J_h], proprio_per_frame [..., H, P]).

        Each mask block is collapsed to per-token (length N_kp or J_h) by taking the first lookahead
        frame's row — the env-side mask broadcaster ensures all L frames carry the same per-token bit.
        """
        if obs.shape[-1] != self._expected_obs_dim:
            raise RuntimeError(
                f"_MUSEHumanKpTransformerEncoder obs dim mismatch: got {obs.shape[-1]}, "
                f"expected {self._expected_obs_dim} "
                f"(kp_block={self._kp_block_dim}, kp_mask={self._kp_mask_dim}, "
                f"human_block={self._human_block_dim}, human_mask={self._human_mask_dim}, "
                f"proprio_block={self._proprio_block_dim})."
            )
        batch_shape = obs.shape[:-1]
        Lk, Nk = self.kp_lookahead_steps, self.kp_n_bodies
        Lh, Jh = self.human_lookahead_steps, self.human_n_joints

        off = 0
        # KP lookahead: row-major [L, N, 3] -> per-body [N, L*3].
        kp_flat = obs[..., off : off + self._kp_block_dim]
        kp_lhn = kp_flat.reshape(*batch_shape, Lk, Nk, 3)
        kp_per_body = kp_lhn.transpose(-3, -2).reshape(*batch_shape, Nk, Lk * 3)
        off += self._kp_block_dim

        # KP mask: [L, N] -> per-body [N] (first frame's row).
        kp_mask_flat = obs[..., off : off + self._kp_mask_dim]
        kp_mask = kp_mask_flat.reshape(*batch_shape, Lk, Nk)[..., 0, :]
        off += self._kp_mask_dim

        # Human lookahead: [L_h, J_h, 3] -> per-joint [J_h, L_h*3].
        human_flat = obs[..., off : off + self._human_block_dim]
        human_lhn = human_flat.reshape(*batch_shape, Lh, Jh, 3)
        human_per_joint = human_lhn.transpose(-3, -2).reshape(*batch_shape, Jh, Lh * 3)
        off += self._human_block_dim

        # Human mask.
        human_mask_flat = obs[..., off : off + self._human_mask_dim]
        human_mask = human_mask_flat.reshape(*batch_shape, Lh, Jh)[..., 0, :]
        off += self._human_mask_dim

        # Proprio.
        proprio_per_frame, _ = _split_history_terms(
            obs, self.proprio_term_sizes, self.proprio_history_length, start_offset=off
        )
        return kp_per_body, kp_mask, human_per_joint, human_mask, proprio_per_frame

    def encode(
        self,
        obs: torch.Tensor,
        modality_override: ModalityOverride = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (μ, log_σ, proprio_per_frame). ``modality_override`` forces full-masking of the
        inactive modality's tokens (see class docstring)."""
        if modality_override not in (None, "kp", "human"):
            raise ValueError(
                f"modality_override must be None | 'kp' | 'human', got {modality_override!r}"
            )
        kp, kp_mask, human, human_mask, proprio_per_frame = self.split_obs(obs)

        if modality_override == "kp":
            human_mask = torch.ones_like(human_mask)
        elif modality_override == "human":
            kp_mask = torch.ones_like(kp_mask)

        kp_clean = torch.nan_to_num(kp, 0.0)
        human_clean = torch.nan_to_num(human, 0.0)
        require_all_finite(proprio_per_frame, "MUSEHumanKp proprio_per_frame")

        batch_shape = obs.shape[:-1]
        D = self.d_model
        Nk, Jh, H = self.kp_n_bodies, self.human_n_joints, self.proprio_history_length

        # KP tokens.
        kp_tokens = self.kp_proj(kp_clean)  # [..., Nk, D]
        kp_tokens = kp_tokens + self.body_id_emb(self._body_ids) + self.modality_emb[0].view(1, -1)

        # Human tokens.
        human_tokens = self.human_proj(human_clean)  # [..., Jh, D]
        human_tokens = (
            human_tokens
            + self.human_body_id_emb(self._human_ids)
            + self.modality_emb[2].view(1, -1)
        )

        # Proprio tokens.
        proprio_tokens = (
            self.proprio_proj(proprio_per_frame)
            + self.proprio_pos_emb.unsqueeze(0)
            + self.modality_emb[1].view(1, -1)
        )

        cls = self.cls_token.expand(*batch_shape, 1, D)
        tokens = torch.cat([cls, kp_tokens, human_tokens, proprio_tokens], dim=-2)

        cls_pad = torch.zeros(*batch_shape, 1, dtype=torch.bool, device=obs.device)
        kp_pad = (kp_mask > 0)
        human_pad = (human_mask > 0)
        proprio_pad = torch.zeros(*batch_shape, H, dtype=torch.bool, device=obs.device)
        key_padding_mask = torch.cat(
            [cls_pad, kp_pad, human_pad, proprio_pad], dim=-1
        )  # [..., 1+Nk+Jh+H]

        tokens_flat = tokens.reshape(-1, self.seq_len, D)
        kpm_flat = key_padding_mask.reshape(-1, self.seq_len)
        out_flat = self.transformer(tokens_flat, src_key_padding_mask=kpm_flat)
        out = out_flat.reshape(*batch_shape, self.seq_len, D)

        cls_out = out[..., 0, :]
        mu = self.mu_head(cls_out)
        log_sigma = torch.full_like(mu, self._log_sigma_const)
        return mu, log_sigma, proprio_per_frame

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)
        return mu + sigma * eps


class LatentBottleneckMUSEHumanKp(nn.Module):
    """MUSE-Human-KP student: 2-modality (KP + human) token transformer encoder + MUSE-style
    decoder + frozen MLP teacher.

    Forward contract differs from :class:`LatentBottleneckMUSEKp` only by the additional
    ``modality_override`` argument on the encode path. ``act`` / ``act_inference`` /
    ``update_distribution`` default to ``modality_override=None`` (both modalities visible per env
    masks); the algorithm's paired-forward update passes ``"kp"`` and ``"human"`` explicitly to
    extract per-modality latents for the alignment loss.
    """

    is_recurrent = False

    def __init__(
        self,
        num_student_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        *,
        proprio_dim: int,  # placeholder for cfg compatibility; unused.
        history_length: int = 5,
        proprio_term_sizes: Sequence[int] = (29, 29, 3, 29),
        # KP encoder.
        kp_n_bodies: int = 14,
        kp_lookahead_steps: int = 10,
        # Human encoder (NEW).
        human_n_joints: int = 22,
        human_lookahead_steps: int = 10,
        # Transformer body.
        latent_dim: int = 16,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 768,
        decoder_hidden_dims: Sequence[int] = (1024, 512, 256, 128),
        teacher_hidden_dims: Sequence[int] = (1024, 1024, 512, 512, 256, 256),
        activation: str = "gelu",
        init_noise_std: float = 1.0,
        latent_predict_std_min: float = 0.001,
        latent_predict_std_max: float = 10.0,
        fixed_encoder_std: float = 1.0,
        encoder_dropout: float = 0.0,
        decoder_one_step_proprio: bool = False,
        latent_normalize: bool = True,
        deterministic_encoder: bool = True,
        freeze_mode: str = "none",
        pilot_human_fraction: float = 0.5,
        **kwargs: Any,
    ):
        if kwargs:
            print(
                "LatentBottleneckMUSEHumanKp.__init__ got unexpected arguments (ignored): "
                + str(list(kwargs.keys()))
            )
        super().__init__()
        del proprio_dim

        if freeze_mode not in _VALID_FREEZE_MODES:
            raise ValueError(
                f"freeze_mode must be one of {_VALID_FREEZE_MODES}, got {freeze_mode!r}"
            )
        self.freeze_mode = str(freeze_mode)

        self.num_student_obs = int(num_student_obs)
        self.num_teacher_obs = int(num_teacher_obs)
        self.num_actions = int(num_actions)
        self.history_length = int(history_length)
        self.latent_normalize = bool(latent_normalize)
        self.deterministic_encoder = bool(deterministic_encoder)
        if not 0.0 <= float(pilot_human_fraction) <= 1.0:
            raise ValueError(
                f"pilot_human_fraction must be in [0, 1], got {pilot_human_fraction!r}"
            )
        self.pilot_human_fraction = float(pilot_human_fraction)
        # Stashed by act() so the runner/algorithm can attribute per-step metrics + termination
        # events to whichever modality actually drove the step. ``None`` until first act().
        self._last_pilot_human_mask: torch.Tensor | None = None

        self.transformer_encoder = _MUSEHumanKpTransformerEncoder(
            kp_n_bodies=int(kp_n_bodies),
            kp_lookahead_steps=int(kp_lookahead_steps),
            human_n_joints=int(human_n_joints),
            human_lookahead_steps=int(human_lookahead_steps),
            proprio_term_sizes=proprio_term_sizes,
            proprio_history_length=self.history_length,
            latent_dim=int(latent_dim),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            ffn_dim=int(ffn_dim),
            activation=activation,
            latent_sigma_min=float(latent_predict_std_min),
            latent_sigma_max=float(latent_predict_std_max),
            fixed_encoder_std=float(fixed_encoder_std),
            dropout=float(encoder_dropout),
        )
        if self.transformer_encoder.expected_obs_dim() != self.num_student_obs:
            raise ValueError(
                f"LatentBottleneckMUSEHumanKp: expected_obs_dim "
                f"({self.transformer_encoder.expected_obs_dim()}) != num_student_obs "
                f"({self.num_student_obs}). Check kp_n_bodies={kp_n_bodies}, "
                f"kp_lookahead_steps={kp_lookahead_steps}, human_n_joints={human_n_joints}, "
                f"human_lookahead_steps={human_lookahead_steps}, "
                f"proprio_term_sizes={proprio_term_sizes}, history_length={history_length}."
            )

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

        teacher_layers: list[nn.Module] = []
        hidden = list(teacher_hidden_dims) or [256, 256]
        teacher_act = nn.ELU
        teacher_layers.append(nn.Linear(self.num_teacher_obs, hidden[0]))
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

        self.latent_log_sigma_min = self.transformer_encoder._log_sigma_min
        self.latent_log_sigma_max = self.transformer_encoder._log_sigma_max

        self._build_student_group()

        print(
            f"[LatentBottleneckMUSEHumanKp] num_student_obs={self.num_student_obs}, "
            f"num_teacher_obs={self.num_teacher_obs}, num_actions={self.num_actions}, "
            f"history_length={self.history_length}, "
            f"kp(n_bodies={kp_n_bodies}, lookahead={kp_lookahead_steps}), "
            f"human(n_joints={human_n_joints}, lookahead={human_lookahead_steps}), "
            f"transformer(d_model={d_model}, layers={num_layers}, heads={nhead}, ffn={ffn_dim}), "
            f"latent_dim={latent_dim}, decoder_hidden={list(decoder_hidden_dims)}, "
            f"teacher_hidden={list(teacher_hidden_dims)}, "
            f"per_frame_proprio_dim={per_frame_proprio_dim}, "
            f"decoder_one_step_proprio={self.decoder_one_step_proprio}, "
            f"decoder_proprio_dim={decoder_proprio_dim}, "
            f"latent_normalize={self.latent_normalize}, "
            f"deterministic_encoder={self.deterministic_encoder}, "
            f"freeze_mode={self.freeze_mode!r}"
        )

    # ----- Param grouping ----------------------------------------------------------------------

    def _kp_only_params(self) -> list[nn.Parameter]:
        enc = self.transformer_encoder
        return list(enc.kp_proj.parameters()) + list(enc.body_id_emb.parameters())

    def _human_only_params(self) -> list[nn.Parameter]:
        enc = self.transformer_encoder
        return list(enc.human_proj.parameters()) + list(enc.human_body_id_emb.parameters())

    def _shared_encoder_modules(self) -> list[nn.Module]:
        enc = self.transformer_encoder
        return [enc.transformer, enc.proprio_proj, enc.mu_head]

    def _build_student_group(self) -> None:
        """Wire ``self.student`` for the chosen freeze mode."""
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        enc = self.transformer_encoder

        if self.freeze_mode == "none":
            for p in enc.parameters():
                p.requires_grad_(True)
            for p in self.decoder.parameters():
                p.requires_grad_(True)
            self.student = nn.ModuleList([enc, self.decoder])

        elif self.freeze_mode == "decoder_only":
            for p in enc.parameters():
                p.requires_grad_(True)
            for p in self.decoder.parameters():
                p.requires_grad_(False)
            self.student = nn.ModuleList([enc])

        elif self.freeze_mode == "decoder_plus_shared_encoder":
            # Freeze decoder + shared backbone; KP-specific + human-specific encoder layers train.
            for p in self.decoder.parameters():
                p.requires_grad_(False)
            for p in enc.parameters():
                p.requires_grad_(False)
            for p in self._kp_only_params():
                p.requires_grad_(True)
            for p in self._human_only_params():
                p.requires_grad_(True)
            # modality_emb (all 3 rows) trains with the per-modality layers.
            enc.modality_emb.requires_grad_(True)

            class _KpHumanTrainable(nn.Module):
                def __init__(self, encoder: _MUSEHumanKpTransformerEncoder):
                    super().__init__()
                    self.kp_proj = encoder.kp_proj
                    self.body_id_emb = encoder.body_id_emb
                    self.human_proj = encoder.human_proj
                    self.human_body_id_emb = encoder.human_body_id_emb
                    self.modality_emb = nn.ParameterList([encoder.modality_emb])

            self.student = nn.ModuleList([_KpHumanTrainable(enc)])

        elif self.freeze_mode == "human_only":
            # Freeze decoder + shared backbone + KP-specific; ONLY human-specific layers train.
            # Use case: warmstart from a MUSE-KP checkpoint where everything except human is trained.
            for p in self.decoder.parameters():
                p.requires_grad_(False)
            for p in enc.parameters():
                p.requires_grad_(False)
            for p in self._human_only_params():
                p.requires_grad_(True)
            # modality_emb: all 3 rows trainable so slot 2 (human) can move; slots 0/1 will receive
            # zero grad anyway (the KP/proprio tokens are present but the loss only flows through
            # the human modality's contribution to the CLS output). Keeping the parameter trainable
            # is simpler than slicing to slot 2.
            enc.modality_emb.requires_grad_(True)

            class _HumanOnlyTrainable(nn.Module):
                def __init__(self, encoder: _MUSEHumanKpTransformerEncoder):
                    super().__init__()
                    self.human_proj = encoder.human_proj
                    self.human_body_id_emb = encoder.human_body_id_emb
                    self.modality_emb = nn.ParameterList([encoder.modality_emb])

            self.student = nn.ModuleList([_HumanOnlyTrainable(enc)])

        else:
            raise AssertionError(f"unhandled freeze_mode {self.freeze_mode!r}")

    def set_freeze_mode(self, mode: str) -> bool:
        if mode not in _VALID_FREEZE_MODES:
            raise ValueError(f"freeze_mode must be one of {_VALID_FREEZE_MODES}, got {mode!r}")
        if mode == self.freeze_mode:
            return False
        self.freeze_mode = str(mode)
        self._build_student_group()
        self.train(self.training)
        return True

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        if self.freeze_mode in (
            "decoder_only",
            "decoder_plus_shared_encoder",
            "human_only",
        ):
            self.decoder.eval()
        if self.freeze_mode in ("decoder_plus_shared_encoder", "human_only"):
            for m in self._shared_encoder_modules():
                m.eval()
        return self

    def reset(self, dones=None, hidden_states=None) -> None:
        return

    # ----- Forward helpers --------------------------------------------------------------------

    def _maybe_normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        if self.latent_normalize:
            return nn.functional.normalize(z, p=2.0, dim=-1, eps=1e-8)
        return z

    def _shape_proprio_for_decoder(self, proprio_per_frame: torch.Tensor) -> torch.Tensor:
        if self.decoder_one_step_proprio:
            return proprio_per_frame[..., -1, :]
        return proprio_per_frame.reshape(*proprio_per_frame.shape[:-2], -1)

    def _decode(self, z: torch.Tensor, proprio_per_frame: torch.Tensor) -> torch.Tensor:
        proprio = self._shape_proprio_for_decoder(proprio_per_frame)
        return self.decoder(torch.cat([z, proprio], dim=-1))

    # ----- Public interface -------------------------------------------------------------------

    def update_distribution(self, observations: torch.Tensor) -> None:
        action = self._act_pilot(observations, deterministic=True)
        std = self.std.expand_as(action).clamp(min=1.0e-6)
        self.distribution = Normal(action, std)

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        """Training-time rollout action. Dual-forward (KP-only + human-only) + 50/50 partition.

        Sets ``self._last_pilot_human_mask`` (per-env bool tensor) so the algorithm can attribute
        per-step termination events to whichever modality drove the step.
        """
        return self._act_pilot(observations, deterministic=False)

    def act_inference(
        self,
        observations: torch.Tensor,
        modality: str = "human",
    ) -> torch.Tensor:
        """Single-modality forward for eval / demo. ``modality`` ∈ {"kp", "human"}.

        Inference target is ``"human"`` by default — at deployment, GENMO drives the human modality
        and KP is not available. Use ``"kp"`` to evaluate the standalone KP modality.
        """
        if modality not in ("kp", "human"):
            raise ValueError(f"modality must be 'kp' or 'human', got {modality!r}")
        mu, _, proprio_per_frame = self.transformer_encoder.encode(
            observations, modality_override=modality
        )
        z = self._maybe_normalize_latent(mu)
        return self._decode(z, proprio_per_frame)

    def _act_pilot(self, observations: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """Dual-modality forward + 50/50 partition. ``self._last_pilot_human_mask`` is set."""
        action_kp, _, _ = self._forward_modality(
            observations, modality="kp", deterministic=deterministic
        )
        action_human, _, _ = self._forward_modality(
            observations, modality="human", deterministic=deterministic
        )

        N = action_kp.shape[0]
        device = action_kp.device
        if self.pilot_human_fraction <= 0.0:
            is_human = torch.zeros(N, device=device, dtype=torch.bool)
            self._last_pilot_human_mask = is_human
            return action_kp
        if self.pilot_human_fraction >= 1.0:
            is_human = torch.ones(N, device=device, dtype=torch.bool)
            self._last_pilot_human_mask = is_human
            return action_human
        is_human = (
            torch.rand(N, device=device, dtype=action_kp.dtype) < self.pilot_human_fraction
        )
        self._last_pilot_human_mask = is_human.detach()
        return torch.where(is_human.unsqueeze(-1), action_human, action_kp)

    def _forward_modality(
        self,
        observations: torch.Tensor,
        modality: ModalityOverride,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-modality forward through encoder + decoder. Returns (action, μ, log_σ).
        μ is returned post-normalize."""
        mu, log_sigma, proprio_per_frame = self.transformer_encoder.encode(
            observations, modality_override=modality
        )
        use_mean = self.deterministic_encoder or deterministic or not self.training
        z = mu if use_mean else self.transformer_encoder.reparameterize(mu, log_sigma)
        z = self._maybe_normalize_latent(z)
        mu = self._maybe_normalize_latent(mu)
        action = self._decode(z, proprio_per_frame)
        if not torch.isfinite(action).all():
            warnings.warn(
                f"LatentBottleneckMUSEHumanKp: non-finite action in modality={modality!r}; "
                f"replacing with 0."
            )
            action = torch.nan_to_num(action)
        return action, mu, log_sigma

    def forward_for_update(
        self,
        observations: torch.Tensor,
        sample_z: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Algorithm-side entry. Runs both modalities on the FULL obs batch and returns the dict
        ``{action_kp, action_human, mu_kp, mu_human, log_sigma_kp, log_sigma_human}``.

        μs are post-normalize so downstream regularizers (cosine smoothness, cross-modal alignment)
        operate on the geometry the decoder sees. ``sample_z`` is respected for both modalities.
        """
        deterministic = not (sample_z and self.training) or self.deterministic_encoder
        action_kp, mu_kp, log_sigma_kp = self._forward_modality(
            observations, modality="kp", deterministic=deterministic
        )
        action_human, mu_human, log_sigma_human = self._forward_modality(
            observations, modality="human", deterministic=deterministic
        )
        return {
            "action_kp": action_kp,
            "action_human": action_human,
            "mu_kp": mu_kp,
            "mu_human": mu_human,
            "log_sigma_kp": log_sigma_kp,
            "log_sigma_human": log_sigma_human,
        }

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

    # ----- Checkpoint loading ----------------------------------------------------------------

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Accepted source checkpoints:

        1. **MUSE-Human-KP resume** (has ``transformer_encoder.human_proj.*``): full load.
        2. **MUSE-KP checkpoint** (has ``transformer_encoder.kp_proj.*`` but no ``human_proj``):
           copy KP + shared backbone + decoder + teacher. Special-case ``modality_emb``: source is
           (2, D), target is (3, D); copy rows [0, 1], leave row [2] at init.
        3. **MUSE-Transformer / PHC+ actor.***: same as :class:`LatentBottleneckMUSEKp`.

        Returns ``True`` if this is a human-KP (or KP) checkpoint (i.e. the encoder was at least
        partially warmstarted), ``False`` for the teacher-only case.
        """
        keys = list(state_dict.keys())
        has_human = any(k.startswith("transformer_encoder.human_proj.") for k in keys)
        has_kp = any(k.startswith("transformer_encoder.kp_proj.") for k in keys)
        has_muse_modules = any(
            k.startswith("transformer_encoder.") or k.startswith("decoder.") or k.startswith("teacher.")
            for k in keys
        )
        has_actor_only = any("actor." in k for k in keys) and not has_muse_modules

        if has_actor_only:
            teacher_sd = {
                key.replace("actor.", ""): value
                for key, value in state_dict.items()
                if "actor." in key
            }
            self.teacher.load_state_dict(teacher_sd, strict=strict)
            self.teacher.eval()
            self.loaded_teacher = True
            return False

        if has_human or has_kp or has_muse_modules:
            self_sd = self.state_dict()
            filtered: dict[str, torch.Tensor] = {}
            skipped: list[str] = []
            modality_emb_key = "transformer_encoder.modality_emb"
            for k, v in state_dict.items():
                if k == modality_emb_key and k in self_sd and self_sd[k].shape != v.shape:
                    # Source (2, D) -> target (3, D): copy first 2 rows.
                    target = self_sd[k].clone()
                    if v.dim() == 2 and target.dim() == 2 and v.shape[1] == target.shape[1] \
                            and v.shape[0] <= target.shape[0]:
                        target[: v.shape[0]] = v
                        filtered[k] = target
                    else:
                        skipped.append(k)
                elif k in self_sd and self_sd[k].shape == v.shape:
                    filtered[k] = v
                else:
                    skipped.append(k)
            if skipped:
                print(
                    f"[LatentBottleneckMUSEHumanKp] Skipping {len(skipped)} ckpt keys "
                    f"(shape mismatch or absent): {skipped[:5]}"
                    f"{'...' if len(skipped) > 5 else ''}"
                )
            super().load_state_dict(filtered, strict=False)
            self.loaded_teacher = True
            self.teacher.eval()
            self._build_student_group()
            return has_human or has_kp

        raise ValueError(
            "state_dict must contain actor.*, or transformer_encoder.*/decoder.*/teacher.* keys."
        )

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None) -> None:
        return
