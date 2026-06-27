"""MUSE co-train: dual-frontend encoder + shared transformer backbone + shared decoder.

Two control modalities are co-trained against the same MLP teacher:

- Joint-cmd (JC, privileged): MUSE-Transformer-style goal token
  (``delta_command + motion_anchor_ori_b + motion_anchor_pos_b``). ``motion_anchor_pos_b`` is the
  privileged world-frame anchor position — JC is a latent-bottlenecked teacher whose jobs are
  (a) anchoring the shared decoder to the full motion suite, (b) a gold align target for KP.
- Keypoints (KP): per-body tokens packing one sparse log-spaced reference-position slot pack
  (``kp_layout``, e.g. ``log_0_5s`` = 12 slots: 3 history + 1 abs + 8 future, cap 0.5 s). Slot 0
  is the absolute ref pos; the others are deltas ``ref_pos_{t+k} − robot_pos_t`` — so the current
  actual position is encoded implicitly (no separate FK ``current_actual`` token). Mirrors
  ``_MUSEKpTransformerEncoder`` exactly so a cotrained KP encoder transfers to the canonical
  MUSE-Kp deployment config. N bodies → N KP tokens.

Both modalities share **everything** except the first projection layer:

    SHARED  : transformer body, proprio_proj, proprio_pos_emb, cls_token, mu_head, decoder, teacher
    JC ONLY : goal_proj, goal_pos_emb, modality_emb[0]
    KP ONLY : kp_proj, body_id_emb, modality_emb[1]
    SHARED  : modality_emb[2] (proprio)

Token sequences (different per pass, same backbone weights):
- JC pass : ``[CLS, goal_token, proprio_0..H-1]``  → μ_jc
- KP pass : ``[CLS, kp_0..N-1, proprio_0..H-1]``   → μ_kp

Both μs are unit-normalized then fed through the same shared decoder with the same proprio.

Piloting (env-step action): 50/50 random partition each step — half the envs are driven by
``decoder(z_jc, proprio)``, half by ``decoder(z_kp, proprio)``. Loss is computed across **all**
envs for **both** modalities (each encoder forwards on the full obs batch), so each modality
sees the state distribution induced by the other.

Combined obs (one tensor, per-term contiguous, in this order):
    [ JC.delta_command (58) | JC.motion_anchor_ori_b (6) | JC.motion_anchor_pos_b (3) |
      JC.goal_mask_history (1) |
      KP.logspaced (L*N*3) | KP.mask (L*N) |
      proprio history-H (per-term contiguous) ]
where L = len(kp_layout) (e.g. 12 for log_0_5s) and N = num_bodies.

Init contract: v1 trains encoder + decoder **from scratch**. ``load_state_dict`` still accepts an
``actor.*``-only checkpoint to load (and freeze) the MLP teacher — the one non-optional load,
since it is the distillation target. The MUSE-Transformer enc/dec warmstart path remains
supported by the loader but is unused in v1 (no clean ckpt).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.latent_bottleneck_muse_transformer import (
    _muse_log_sigma_bounds,
    _split_history_terms,
)
from rsl_rl.modules.latent_bottleneck_muse_kp import kp_layout_by_name
from rsl_rl.utils.finite_checks import require_all_finite


class _MUSECoTrainEncoder(nn.Module):
    """Dual-frontend encoder. ``encode(obs, modality)`` runs the matching front-end through the
    shared transformer body and returns (μ, log_σ, proprio_per_frame)."""

    def __init__(
        self,
        *,
        # JC modality.
        jc_goal_term_sizes: Sequence[int],
        jc_mask_term_size: int,
        # KP modality.
        kp_n_bodies: int,
        kp_lookahead_steps: int,
        kp_layout: str,
        # Shared proprio.
        proprio_term_sizes: Sequence[int],
        proprio_history_length: int,
        # Backbone.
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

        # JC modality dims.
        self.jc_goal_term_sizes = tuple(int(s) for s in jc_goal_term_sizes)
        self.jc_mask_term_size = int(jc_mask_term_size)
        self.per_frame_jc_goal_dim = sum(self.jc_goal_term_sizes)

        # KP modality dims. Sparse log-spaced slot pack mirroring _MUSEKpTransformerEncoder: one
        # Linear over L*3 (L = len(kp_layout)); slot 0 = abs ref pos, others = deltas, so the
        # current actual position is encoded implicitly (no separate FK token).
        self.kp_n_bodies = int(kp_n_bodies)
        self.kp_lookahead_steps = int(kp_lookahead_steps)
        self.kp_layout_name = str(kp_layout)
        self.kp_slot_offsets: tuple[int, ...] = kp_layout_by_name(self.kp_layout_name)
        if self.kp_lookahead_steps != len(self.kp_slot_offsets):
            raise ValueError(
                f"_MUSECoTrainEncoder: kp_lookahead_steps={self.kp_lookahead_steps} != "
                f"len(layout {self.kp_layout_name!r})={len(self.kp_slot_offsets)}."
            )
        self.kp_token_input_dim = self.kp_lookahead_steps * 3

        # Shared proprio.
        self.proprio_term_sizes = tuple(int(s) for s in proprio_term_sizes)
        self.proprio_history_length = int(proprio_history_length)
        self.per_frame_proprio_dim = sum(self.proprio_term_sizes)

        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)

        # ---- Per-modality first projections + their dedicated embeddings -----------------------

        # JC: single goal token (matches MUSE-Transformer naming).
        self.goal_proj = nn.Linear(self.per_frame_jc_goal_dim, self.d_model)
        self.goal_pos_emb = nn.Parameter(torch.zeros(1, self.d_model))
        nn.init.normal_(self.goal_pos_emb, std=0.02)

        # KP: per-body tokens.
        self.kp_proj = nn.Linear(self.kp_token_input_dim, self.d_model)
        nn.init.normal_(self.kp_proj.weight, std=0.02)
        nn.init.zeros_(self.kp_proj.bias)
        self.body_id_emb = nn.Embedding(self.kp_n_bodies, self.d_model)
        nn.init.normal_(self.body_id_emb.weight, std=0.02)

        # Modality embedding: [0]=JC goal, [1]=KP body, [2]=proprio.
        self.modality_emb = nn.Parameter(torch.zeros(3, self.d_model))
        nn.init.normal_(self.modality_emb, std=0.02)

        # ---- Shared backbone ------------------------------------------------------------------

        self.proprio_proj = nn.Linear(self.per_frame_proprio_dim, self.d_model)
        self.proprio_pos_emb = nn.Parameter(torch.zeros(self.proprio_history_length, self.d_model))
        nn.init.normal_(self.proprio_pos_emb, std=0.02)
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

        # σ is fixed (deterministic-encoder regime; matches MUSE-T / MUSE-KP).
        self.fixed_encoder_std = float(fixed_encoder_std)
        lmin, lmax = _muse_log_sigma_bounds(latent_sigma_min, latent_sigma_max)
        self._log_sigma_min = lmin
        self._log_sigma_max = lmax
        self._log_sigma_const = math.log(max(self.fixed_encoder_std, 1e-6))

        # ---- Obs offsets / sequence lengths ---------------------------------------------------

        # JC blocks (each 1-step at the obs source — no Isaac history rolling).
        self._jc_goal_block_dim = self.per_frame_jc_goal_dim
        self._jc_mask_block_dim = self.jc_mask_term_size

        # KP blocks (log-spaced slot pack; baked inside the term, no Isaac rolling).
        self._kp_block_dim = self.kp_lookahead_steps * self.kp_n_bodies * 3
        self._kp_mask_block_dim = self.kp_lookahead_steps * self.kp_n_bodies

        # Shared proprio block (Isaac history-rolled, per-term contiguous).
        self._proprio_block_dim = self.per_frame_proprio_dim * self.proprio_history_length

        # Combined obs layout (in concat order): JC.goal | JC.mask | KP.logspaced | KP.mask | proprio.
        self._jc_goal_off = 0
        self._jc_mask_off = self._jc_goal_off + self._jc_goal_block_dim
        self._kp_off = self._jc_mask_off + self._jc_mask_block_dim
        self._kp_mask_off = self._kp_off + self._kp_block_dim
        self._proprio_off = self._kp_mask_off + self._kp_mask_block_dim
        self._expected_obs_dim = self._proprio_off + self._proprio_block_dim

        # Sequence lengths per modality pass.
        self.seq_len_jc = 1 + 1 + self.proprio_history_length  # CLS + goal + proprio
        self.seq_len_kp = 1 + self.kp_n_bodies + self.proprio_history_length  # CLS + N_kp + proprio

        # Pre-built body-id grid.
        body_ids = torch.arange(self.kp_n_bodies, dtype=torch.long)
        self.register_buffer("_body_ids", body_ids.contiguous(), persistent=False)

        # Records the slot offsets kp_proj was trained for (parity with _MUSEKpTransformerEncoder;
        # written into checkpoints for downstream slot-aware loaders).
        self.register_buffer(
            "kp_slot_offsets_buf",
            torch.tensor(self.kp_slot_offsets, dtype=torch.long).contiguous(),
            persistent=True,
        )

    def expected_obs_dim(self) -> int:
        return self._expected_obs_dim

    # ----- Obs splitting ----------------------------------------------------------------------

    def split_obs(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Slice the combined obs into per-modality blocks. All KP/proprio blocks are reshaped to
        their natural per-token / per-frame layouts; raw 1-step blocks are squeezed."""
        if obs.shape[-1] != self._expected_obs_dim:
            raise RuntimeError(
                f"_MUSECoTrainEncoder obs dim mismatch: got {obs.shape[-1]}, "
                f"expected {self._expected_obs_dim} "
                f"(jc_goal={self._jc_goal_block_dim}, jc_mask={self._jc_mask_block_dim}, "
                f"kp={self._kp_block_dim}, kp_mask={self._kp_mask_block_dim}, "
                f"proprio={self._proprio_block_dim})."
            )
        batch_shape = obs.shape[:-1]
        N = self.kp_n_bodies
        L = self.kp_lookahead_steps

        # JC goal — split per-term (each is 1-step), concat → [..., per_frame_jc_goal_dim].
        jc_goal_h, _ = _split_history_terms(
            obs, self.jc_goal_term_sizes, history_length=1, start_offset=self._jc_goal_off
        )
        jc_goal = jc_goal_h.squeeze(-2)

        # JC mask — 1 step.
        jc_mask = obs[..., self._jc_mask_off : self._jc_mask_off + self._jc_mask_block_dim]

        # KP log-spaced slot pack: row-major [L, N, 3] → per-body sequence [N, L*3].
        kp_flat = obs[..., self._kp_off : self._kp_off + self._kp_block_dim]
        kp_lhn = kp_flat.reshape(*batch_shape, L, N, 3)
        kp_per_body = kp_lhn.transpose(-3, -2).reshape(*batch_shape, N, L * 3)

        # KP mask: [L, N] → per-body (broadcast across slots, take first).
        kp_mask_flat = obs[..., self._kp_mask_off : self._kp_mask_off + self._kp_mask_block_dim]
        kp_mask_ln = kp_mask_flat.reshape(*batch_shape, L, N)
        kp_mask_per_body = kp_mask_ln[..., 0, :]  # [..., N]

        # Proprio history (per-term contiguous → [..., H, P]).
        proprio_per_frame, _ = _split_history_terms(
            obs, self.proprio_term_sizes, self.proprio_history_length, start_offset=self._proprio_off
        )

        return {
            "jc_goal": jc_goal,
            "jc_mask": jc_mask,
            "kp_per_body": kp_per_body,
            "kp_mask_per_body": kp_mask_per_body,
            "proprio_per_frame": proprio_per_frame,
        }

    # ----- Encoders --------------------------------------------------------------------------

    def _encode_jc(
        self,
        jc_goal: torch.Tensor,
        jc_mask: torch.Tensor,
        proprio_per_frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_shape = jc_goal.shape[:-1]
        D = self.d_model
        H = self.proprio_history_length

        # Mask bit (post-EmpiricalNormalization 0/1 → sign threshold recovers boolean).
        mask_bool = (jc_mask[..., 0] > 0)  # [...]
        # Defense in depth: zero the goal pre-projection on masked steps.
        goal_clean = torch.where(mask_bool.unsqueeze(-1), torch.zeros_like(jc_goal), jc_goal)

        goal_token = self.goal_proj(goal_clean).unsqueeze(-2)  # [..., 1, D]
        goal_token = goal_token + self.goal_pos_emb.unsqueeze(0) + self.modality_emb[0].view(1, 1, -1)

        proprio_tokens = (
            self.proprio_proj(proprio_per_frame)
            + self.proprio_pos_emb.unsqueeze(0)
            + self.modality_emb[2].view(1, 1, -1)
        )

        cls = self.cls_token.expand(*batch_shape, 1, D)
        tokens = torch.cat([cls, goal_token, proprio_tokens], dim=-2)  # [..., 1+1+H, D]

        cls_pad = torch.zeros(*batch_shape, 1, dtype=torch.bool, device=jc_goal.device)
        goal_pad = mask_bool.unsqueeze(-1)
        proprio_pad = torch.zeros(*batch_shape, H, dtype=torch.bool, device=jc_goal.device)
        key_padding_mask = torch.cat([cls_pad, goal_pad, proprio_pad], dim=-1)

        tokens_flat = tokens.reshape(-1, self.seq_len_jc, D)
        kpm_flat = key_padding_mask.reshape(-1, self.seq_len_jc)
        out_flat = self.transformer(tokens_flat, src_key_padding_mask=kpm_flat)
        out = out_flat.reshape(*batch_shape, self.seq_len_jc, D)

        cls_out = out[..., 0, :]
        mu = self.mu_head(cls_out)
        log_sigma = torch.full_like(mu, self._log_sigma_const)
        return mu, log_sigma

    def _encode_kp(
        self,
        kp_per_body: torch.Tensor,
        kp_mask_per_body: torch.Tensor,
        proprio_per_frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_shape = kp_per_body.shape[:-2]
        D = self.d_model
        N, H = self.kp_n_bodies, self.proprio_history_length

        # Per-body log-spaced slot pack [..., N, L*3]; NaN at hidden bodies → nan_to_num before
        # any projection so K/V can't be poisoned (defense in depth).
        kp_token_input = torch.nan_to_num(kp_per_body, 0.0)
        require_all_finite(proprio_per_frame, "MUSECoTrain proprio_per_frame")

        kp_tokens = self.kp_proj(kp_token_input)  # [..., N, D]
        body_token = self.body_id_emb(self._body_ids)  # [N, D]
        kp_tokens = kp_tokens + body_token + self.modality_emb[1].view(1, -1)

        proprio_tokens = (
            self.proprio_proj(proprio_per_frame)
            + self.proprio_pos_emb.unsqueeze(0)
            + self.modality_emb[2].view(1, 1, -1)
        )

        cls = self.cls_token.expand(*batch_shape, 1, D)
        tokens = torch.cat([cls, kp_tokens, proprio_tokens], dim=-2)

        cls_pad = torch.zeros(*batch_shape, 1, dtype=torch.bool, device=kp_per_body.device)
        kp_pad = (kp_mask_per_body > 0)
        proprio_pad = torch.zeros(*batch_shape, H, dtype=torch.bool, device=kp_per_body.device)
        key_padding_mask = torch.cat([cls_pad, kp_pad, proprio_pad], dim=-1)

        tokens_flat = tokens.reshape(-1, self.seq_len_kp, D)
        kpm_flat = key_padding_mask.reshape(-1, self.seq_len_kp)
        out_flat = self.transformer(tokens_flat, src_key_padding_mask=kpm_flat)
        out = out_flat.reshape(*batch_shape, self.seq_len_kp, D)

        cls_out = out[..., 0, :]
        mu = self.mu_head(cls_out)
        log_sigma = torch.full_like(mu, self._log_sigma_const)
        return mu, log_sigma

    def encode(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward both modality passes once. Returns (μ_jc, log_σ_jc, μ_kp, log_σ_kp, proprio_per_frame).

        Both encoders share the proprio block and run sequentially on the same flattened batch —
        the transformer body sees different first tokens but identical proprio tokens (with the
        same proprio modality embedding), so the proprio_proj weights get gradient signal from
        both passes.
        """
        blocks = self.split_obs(obs)
        proprio_per_frame = blocks["proprio_per_frame"]
        mu_jc, log_sigma_jc = self._encode_jc(blocks["jc_goal"], blocks["jc_mask"], proprio_per_frame)
        mu_kp, log_sigma_kp = self._encode_kp(
            blocks["kp_per_body"],
            blocks["kp_mask_per_body"],
            proprio_per_frame,
        )
        return mu_jc, log_sigma_jc, mu_kp, log_sigma_kp, proprio_per_frame

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)
        return mu + sigma * eps


class LatentBottleneckMUSECoTrain(nn.Module):
    """Co-train MUSE student: dual-frontend encoder + shared backbone + shared decoder + frozen
    MLP teacher.

    Loss / training contract (computed by :class:`MuseCoTrainDistillation`):
      L = L_BC^jc + L_BC^kp
        + λ_reg · (L_smooth^jc + L_smooth^kp)
        + λ_align · L_align(μ_jc, μ_kp)

    All five terms operate on the FULL env batch every transition. The 50/50 piloting partition
    only chooses which encoder's action drives env.step() per env per step — it does not gate
    which encoder receives gradient.
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
        # JC modality (privileged: delta_command + motion_anchor_ori_b + motion_anchor_pos_b).
        jc_goal_term_sizes: Sequence[int] = (58, 6, 3),
        jc_mask_term_size: int = 1,
        # KP modality (sparse log-spaced slot pack; mirrors _MUSEKpTransformerEncoder).
        kp_n_bodies: int = 14,
        kp_lookahead_steps: int = 12,
        kp_layout: str = "log_0_5s",
        # Backbone (must match MUSE-Transformer ckpt for warmstart to load cleanly).
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
        # Piloting.
        pilot_kp_fraction: float = 0.5,
        **kwargs: Any,
    ):
        if kwargs:
            print(
                "LatentBottleneckMUSECoTrain.__init__ got unexpected arguments (ignored): "
                + str(list(kwargs.keys()))
            )
        super().__init__()
        del proprio_dim

        self.num_student_obs = int(num_student_obs)
        self.num_teacher_obs = int(num_teacher_obs)
        self.num_actions = int(num_actions)
        self.history_length = int(history_length)
        self.latent_normalize = bool(latent_normalize)
        self.deterministic_encoder = bool(deterministic_encoder)
        self.pilot_kp_fraction = float(pilot_kp_fraction)
        # Filled in by ``_act_pilot`` every env step; read by the per-pilot metric logger and by
        # ``MuseCoTrainDistillation.process_env_step`` for per-modality success-rate attribution.
        self._last_pilot_kp_mask: torch.Tensor | None = None
        if not (0.0 <= self.pilot_kp_fraction <= 1.0):
            raise ValueError(
                f"pilot_kp_fraction must be in [0, 1], got {self.pilot_kp_fraction!r}"
            )

        self.transformer_encoder = _MUSECoTrainEncoder(
            jc_goal_term_sizes=jc_goal_term_sizes,
            jc_mask_term_size=int(jc_mask_term_size),
            kp_n_bodies=int(kp_n_bodies),
            kp_lookahead_steps=int(kp_lookahead_steps),
            kp_layout=str(kp_layout),
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
                f"LatentBottleneckMUSECoTrain: expected_obs_dim "
                f"({self.transformer_encoder.expected_obs_dim()}) != num_student_obs "
                f"({self.num_student_obs}). Check jc_goal_term_sizes={jc_goal_term_sizes}, "
                f"jc_mask_term_size={jc_mask_term_size}, kp_n_bodies={kp_n_bodies}, "
                f"kp_lookahead_steps={kp_lookahead_steps}, kp_layout={kp_layout!r}, "
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

        # MLP teacher (frozen). Same shape as MUSE-Transformer's teacher slot.
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
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.loaded_teacher = False

        # Action distribution: ONE shared, FIXED std (buffer, not Parameter — locked-in choice).
        init_std = max(float(init_noise_std), 1.0e-6)
        self.register_buffer("std", init_std * torch.ones(self.num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args = False  # type: ignore[assignment]

        self.latent_log_sigma_min = self.transformer_encoder._log_sigma_min
        self.latent_log_sigma_max = self.transformer_encoder._log_sigma_max

        # Optimizer-visible group: encoder (with both frontends) + decoder.
        self.student = nn.ModuleList([self.transformer_encoder, self.decoder])

        print(
            f"[LatentBottleneckMUSECoTrain] num_student_obs={self.num_student_obs}, "
            f"num_teacher_obs={self.num_teacher_obs}, num_actions={self.num_actions}, "
            f"history_length={self.history_length}, "
            f"jc(goal_term_sizes={list(jc_goal_term_sizes)}, mask_term_size={jc_mask_term_size}), "
            f"kp(n_bodies={kp_n_bodies}, lookahead={kp_lookahead_steps}, layout={kp_layout!r}, "
            f"token_in_dim={self.transformer_encoder.kp_token_input_dim}), "
            f"transformer(d_model={d_model}, layers={num_layers}, heads={nhead}, ffn={ffn_dim}), "
            f"latent_dim={latent_dim}, decoder_hidden={list(decoder_hidden_dims)}, "
            f"teacher_hidden={list(teacher_hidden_dims)}, "
            f"per_frame_proprio_dim={per_frame_proprio_dim}, "
            f"decoder_one_step_proprio={self.decoder_one_step_proprio}, "
            f"decoder_proprio_dim={decoder_proprio_dim}, "
            f"latent_normalize={self.latent_normalize}, "
            f"deterministic_encoder={self.deterministic_encoder}, "
            f"pilot_kp_fraction={self.pilot_kp_fraction}"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Teacher always eval.
        self.teacher.eval()
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

    # ----- Public interface (matches MUSE-Transformer / MUSE-KP for runner compatibility) -----

    def update_distribution(self, observations: torch.Tensor) -> None:
        action = self._act_pilot(observations, deterministic=True)
        std = self.std.expand_as(action).clamp(min=1.0e-6)
        self.distribution = Normal(action, std)

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        return self._act_pilot(observations, deterministic=False)

    def act_inference(self, observations: torch.Tensor, modality: str = "jc") -> torch.Tensor:
        """Single-modality forward for eval / demo. ``modality`` ∈ {"jc", "kp"}."""
        mu_jc, _, mu_kp, _, proprio_per_frame = self.transformer_encoder.encode(observations)
        if modality == "jc":
            z = self._maybe_normalize_latent(mu_jc)
        elif modality == "kp":
            z = self._maybe_normalize_latent(mu_kp)
        else:
            raise ValueError(f"modality must be 'jc' or 'kp', got {modality!r}")
        return self._decode(z, proprio_per_frame)

    def _act_pilot(self, observations: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """Run both encoders, then 50/50 (configurable) partition picks the action source per env.

        Sample-z behavior follows MUSE-Transformer / MUSE-KP: deterministic_encoder forces
        z := μ; otherwise sample for training, take mean for eval.
        """
        mu_jc, log_sigma_jc, mu_kp, log_sigma_kp, proprio_per_frame = (
            self.transformer_encoder.encode(observations)
        )
        use_mean = self.deterministic_encoder or deterministic or not self.training
        if use_mean:
            z_jc, z_kp = mu_jc, mu_kp
        else:
            z_jc = self.transformer_encoder.reparameterize(mu_jc, log_sigma_jc)
            z_kp = self.transformer_encoder.reparameterize(mu_kp, log_sigma_kp)
        z_jc = self._maybe_normalize_latent(z_jc)
        z_kp = self._maybe_normalize_latent(z_kp)

        action_jc = self._decode(z_jc, proprio_per_frame)
        action_kp = self._decode(z_kp, proprio_per_frame)

        # Partition mask: True → pilot with KP; False → pilot with JC.
        # Random per-env per-step; runs on the action's device for cheap broadcast.
        # Stashed on the module as ``_last_pilot_kp_mask`` so the runner/algorithm can attribute
        # per-step metrics + termination events to the modality that actually drove the step.
        N = action_jc.shape[0]
        if self.pilot_kp_fraction <= 0.0:
            is_kp = torch.zeros(N, device=action_jc.device, dtype=torch.bool)
            self._last_pilot_kp_mask = is_kp
            return action_jc
        if self.pilot_kp_fraction >= 1.0:
            is_kp = torch.ones(N, device=action_jc.device, dtype=torch.bool)
            self._last_pilot_kp_mask = is_kp
            return action_kp
        is_kp = (
            torch.rand(N, device=action_jc.device, dtype=action_jc.dtype)
            < self.pilot_kp_fraction
        )  # [N], bool
        self._last_pilot_kp_mask = is_kp.detach()
        return torch.where(is_kp.unsqueeze(-1), action_kp, action_jc)

    def forward_for_update(
        self, observations: torch.Tensor, sample_z: bool = True
    ) -> dict[str, torch.Tensor]:
        """Run both encoders + decoder on the full obs batch.

        Returns a dict with action / μ / log_σ for both modalities. μs are returned post-
        normalization (when enabled) so downstream regularizers operate on the geometry the
        decoder sees. This is the algorithm-side entry point — algorithm computes the 5 loss
        terms from these tensors.
        """
        mu_jc, log_sigma_jc, mu_kp, log_sigma_kp, proprio_per_frame = (
            self.transformer_encoder.encode(observations)
        )
        if self.deterministic_encoder or not (sample_z and self.training):
            z_jc, z_kp = mu_jc, mu_kp
        else:
            z_jc = self.transformer_encoder.reparameterize(mu_jc, log_sigma_jc)
            z_kp = self.transformer_encoder.reparameterize(mu_kp, log_sigma_kp)
        z_jc = self._maybe_normalize_latent(z_jc)
        z_kp = self._maybe_normalize_latent(z_kp)
        mu_jc = self._maybe_normalize_latent(mu_jc)
        mu_kp = self._maybe_normalize_latent(mu_kp)

        action_jc = self._decode(z_jc, proprio_per_frame)
        action_kp = self._decode(z_kp, proprio_per_frame)
        if not torch.isfinite(action_jc).all():
            warnings.warn("LatentBottleneckMUSECoTrain: non-finite action_jc; replacing with 0.")
            action_jc = torch.nan_to_num(action_jc)
        if not torch.isfinite(action_kp).all():
            warnings.warn("LatentBottleneckMUSECoTrain: non-finite action_kp; replacing with 0.")
            action_kp = torch.nan_to_num(action_kp)

        # KP visible fraction per sample, for the algorithm's mask-stratified latent telemetry.
        # split_obs is pure indexing/reshape (no NN), so the second call is cheap.
        with torch.no_grad():
            kp_mask_pb = self.transformer_encoder.split_obs(observations)["kp_mask_per_body"]
            kp_visible_frac = 1.0 - (kp_mask_pb > 0).float().mean(dim=-1)

        return {
            "action_jc": action_jc,
            "action_kp": action_kp,
            "mu_jc": mu_jc,
            "mu_kp": mu_kp,
            "log_sigma_jc": log_sigma_jc,
            "log_sigma_kp": log_sigma_kp,
            "kp_visible_frac": kp_visible_frac,
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
        """Three accepted shapes:

        1. **PHC+ stage-1** (``actor.*``): only fills the MLP teacher slot.
        2. **MUSE-Transformer checkpoint** (no ``transformer_encoder.kp_proj.*``): copies the
           shared backbone + JC ``goal_proj`` / ``goal_pos_emb`` + decoder + teacher. KP-only
           layers (``kp_proj``, ``body_id_emb``) keep their random init. ``modality_emb`` is
           partially copied: MUSE's slot [0] (goal) → ours [0] (JC); MUSE's [1] (proprio) →
           ours [2] (proprio); ours [1] (KP) keeps random init.
        3. **MUSE-CoTrain resume** (contains ``transformer_encoder.kp_proj.*``): full load.
        """
        keys = list(state_dict.keys())
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
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.loaded_teacher = True
            return False

        if has_kp or has_muse_modules:
            self_sd = self.state_dict()
            filtered: dict[str, torch.Tensor] = {}
            skipped: list[str] = []
            modality_key = "transformer_encoder.modality_emb"
            for k, v in state_dict.items():
                if k == modality_key:
                    target = self_sd.get(k, None)
                    # Special case: MUSE-T [2, D] → CoTrain [3, D] partial copy. Only fires when
                    # the source modality_emb has 2 rows; the matching-shape DDP/resume case falls
                    # through to the normal shape-match path below.
                    if (
                        target is not None
                        and v.shape[0] == 2
                        and target.shape[0] == 3
                        and v.shape[1] == target.shape[1]
                    ):
                        new_emb = target.clone()
                        new_emb[0] = v[0]  # MUSE goal → JC
                        new_emb[2] = v[1]  # MUSE proprio → proprio
                        # new_emb[1] (KP) keeps the random init from `target`.
                        filtered[k] = new_emb
                        continue
                    # else: fall through to the normal shape-match check.
                if k in self_sd and self_sd[k].shape == v.shape:
                    filtered[k] = v
                else:
                    skipped.append(k)
            if skipped:
                print(
                    f"[LatentBottleneckMUSECoTrain] Skipping {len(skipped)} ckpt keys (shape "
                    f"mismatch or absent in current model): {skipped[:8]}"
                    f"{'...' if len(skipped) > 8 else ''}"
                )
            super().load_state_dict(filtered, strict=False)
            self.loaded_teacher = True
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            return has_kp

        raise ValueError(
            "state_dict must contain actor.*, or transformer_encoder.*/decoder.*/teacher.* keys."
        )

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None) -> None:
        return
