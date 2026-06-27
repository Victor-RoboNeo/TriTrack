"""MUSE-KP partial keypoint tracker: KP token encoder + MUSE proprio/transformer/decoder warmstart.

Architecture
------------
Built on top of the stabilized MUSE-Transformer recipe (deterministic encoder + unit-norm latent +
cosine smoothness regularization on μ). Replaces MUSE's single goal token with N_kp tokens, one per
body, where each token packs that body's L-frame position sequence (in the current robot anchor frame)
into D via a single Linear projection. Body identity is supplied by ``body_id_emb``.

Token sequence (length 1 + N_kp + H, e.g. 1 + 14 + 5 = 20 for G1 with H=5 proprio history):

    [CLS, kp_0, ..., kp_{N_kp-1}, proprio_0, ..., proprio_{H-1}]

Per-body KP token: ``kp_proj(R^{L*3} → R^D) + body_id_emb[i] + modality_emb[0]``.
Per-frame proprio token: ``proprio_proj + proprio_pos_emb[t] + modality_emb[1]`` — unchanged from
:class:`LatentBottleneckMUSETransformer`.

Masking — defense in depth
--------------------------
- Pre-projection: NaN at masked body positions is ``nan_to_num(0)``'d before ``kp_proj`` so K/V
  projections cannot be poisoned even if the attention mask were misapplied.
- Attention: ``src_key_padding_mask`` True at masked KP slots; the transformer softmax skips them.

Distillation
------------
This is a thin variant of the MUSE-Transformer recipe — the algorithm is the MUSE distillation
loss (BC against the MLP teacher + cosine smoothness on μ), there is **no latent matching** loss.
The MLP teacher is loaded into ``self.teacher`` from a MUSE checkpoint and used as the action
target source via ``evaluate(teacher_obs)``.

Warmstart
---------
``load_state_dict`` accepts a MUSE-Transformer checkpoint (or stage-1 PHC+ ``actor.*`` or a
MUSE-KP resume). When loading from MUSE, the shape-compatible subset of the MUSE encoder
(``transformer.*``, ``proprio_proj``, ``proprio_pos_emb``, ``modality_emb`` slot [1], ``cls_token``,
``mu_head``) and the full ``decoder``/``teacher`` are copied; KP-specific layers (``kp_proj``,
``body_id_emb``, ``modality_emb`` slot [0]) keep their small random init.

Freeze modes
------------
``freeze_mode`` controls which parameters the optimizer touches via the ``self.student`` group:

- ``"none"`` (default): full encoder + decoder train.
- ``"decoder_only"``: decoder frozen; encoder (KP + shared) trains.
- ``"decoder_plus_shared_encoder"``: decoder + shared encoder pieces frozen; only KP-specific layers
  train. Useful as a warmstart phase to bring KP-input layers up to speed before unfreezing.

The ``teacher`` MLP is always frozen and held in ``eval()``.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.latent_bottleneck_muse_transformer import (
    _MUSETransformerEncoder,
    _muse_log_sigma_bounds,
    _split_history_terms,
)
from rsl_rl.utils.finite_checks import require_all_finite


_VALID_FREEZE_MODES = ("none", "decoder_only", "decoder_plus_shared_encoder")


# -------- KP slot layouts (env-step offsets at 50Hz; 0 = abs, negative = history, positive = future).
# All layouts include exactly one abs slot (offset 0) to preserve OmniGrasp-style packing semantics:
# offset == 0 → ref_pos_t (absolute, anchor frame); offset != 0 → ref_pos_{t+k} − robot_pos_t (delta).
# The warmstart loader in :meth:`LatentBottleneckMUSEKp.load_state_dict` remaps ``kp_proj`` columns
# by slot-offset identity, so a checkpoint trained on one layout warmstarts cleanly into another.

# Legacy 33-dim/body layout (yesterday's MUSE-Kp run): 1 abs + 10 future deltas, dense at 50Hz.
KP_LAYOUT_LEGACY: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

# 0.5s cap log-spaced layout (12 slots → 36 dims/body). 3 history + 1 abs + 8 future.
# Future spacings dense near t (control fidelity), sparser at the tail (speed disambiguation).
KP_LAYOUT_0_5S: tuple[int, ...] = (-20, -10, -3, 0, 1, 2, 3, 5, 8, 13, 20, 25)

# 1.0s cap log-spaced layout (13 slots → 39 dims/body). Same 3 history + 1 abs; 9 future to ~1s.
# Fallback for scenarios with a longer reliable cursor-future buffer.
KP_LAYOUT_1_0S: tuple[int, ...] = (-20, -10, -3, 0, 1, 2, 4, 7, 11, 17, 25, 40, 50)

# 0.5s cap symmetric sparse layout (15 slots → 45 dims/body): {0} ∪ ±{1,3,6,10,15,20,25}.
# The canonical KP5 layout (used by the KP5 latent-distill + latent-RL tasks).
KP_LAYOUT_SYM_SPARSE_0P5S: tuple[int, ...] = (
    -25, -20, -15, -10, -6, -3, -1, 0, 1, 3, 6, 10, 15, 20, 25
)  # {0} ∪ ±{1,3,6,10,15,20,25}, ±0.5s, sparse — 15 slots

# Named layouts, exposed by name so env config + cfg-runner can stay in sync without duplication.
KP_LAYOUTS: dict[str, tuple[int, ...]] = {
    "legacy": KP_LAYOUT_LEGACY,
    "log_0_5s": KP_LAYOUT_0_5S,
    "log_1_0s": KP_LAYOUT_1_0S,
    "sym_sparse_0p5s": KP_LAYOUT_SYM_SPARSE_0P5S,
}


def kp_layout_by_name(name: str) -> tuple[int, ...]:
    if name not in KP_LAYOUTS:
        raise KeyError(f"Unknown KP layout {name!r}. Known: {sorted(KP_LAYOUTS)}.")
    return KP_LAYOUTS[name]


class AuxAnchorPredictor(nn.Module):
    """Predicts motion anchor signals from (latent z, last-frame proprio).

    Two heads, shared trunk:
      - ``speed_head`` → 3D anchor linear velocity (matches ``mdp.ref_base_lin_vel_b``).
      - ``ori_head`` → 6D anchor orientation (continuous rotation rep; matches ``mdp.motion_anchor_ori_b``).

    Targets are sourced from the env-side teacher obs by the algorithm; this module only consumes
    (z, proprio) and emits unnormalized predictions. The algorithm handles target normalization
    statistics (EMA buffers) and computes the loss in normalized space.

    Stop-grad behavior: not enforced here. The algorithm chooses whether to detach ``z`` before
    calling :meth:`forward`. Set ``aux_stop_grad=True`` (Run A, probe) → z.detach() upstream;
    set ``aux_stop_grad=False`` (Run B) → encoder receives aux gradient.
    """

    def __init__(self, latent_dim: int, per_frame_proprio_dim: int):
        super().__init__()
        in_dim = int(latent_dim) + int(per_frame_proprio_dim)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 512), nn.SiLU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.SiLU(),
        )
        self.speed_head = nn.Linear(128, 3)
        self.ori_head = nn.Linear(128, 6)

    def forward(self, z: torch.Tensor, proprio_last: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([z, proprio_last], dim=-1))
        return self.speed_head(h), self.ori_head(h)


class _MUSEKpTransformerEncoder(nn.Module):
    """KP-token encoder with MUSE-Transformer body. Shared encoder pieces are shape-identical to
    :class:`_MUSETransformerEncoder` so MUSE-Transformer weights can be loaded into them via
    ``load_state_dict`` (with the per-body KP layers added on top).

    Obs layout (per env):
      [0 .. L*N*3)        kp_lookahead     (N=14 bodies, 3 dims, L=10 lookahead frames; row-major
                                            ``[L, N, 3]``; NaN at masked bodies)
      [..L*N)             kp_mask_lookahead (1.0 = masked, 0.0 = visible; row-major ``[L, N]``)
      [..proprio_block)   proprio history-H in MUSE layout (joint_pos | joint_vel | base_ang_vel | actions)
    """

    def __init__(
        self,
        *,
        kp_n_bodies: int,
        kp_lookahead_steps: int,
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
        self.proprio_term_sizes = tuple(int(s) for s in proprio_term_sizes)
        self.proprio_history_length = int(proprio_history_length)
        self.per_frame_proprio_dim = sum(self.proprio_term_sizes)
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)

        # KP-specific layers (new — small init, trainable in all freeze modes that include encoder).
        self.kp_proj = nn.Linear(self.kp_lookahead_steps * 3, self.d_model)
        nn.init.normal_(self.kp_proj.weight, std=0.02)
        nn.init.zeros_(self.kp_proj.bias)
        self.body_id_emb = nn.Embedding(self.kp_n_bodies, self.d_model)
        nn.init.normal_(self.body_id_emb.weight, std=0.02)

        # Shared encoder pieces — names + shapes deliberately match _MUSETransformerEncoder so MUSE
        # checkpoints load cleanly. ``modality_emb`` slot [0] is the "KP modality" (re-purposed; the
        # MUSE-Transformer goal-modality value is overwritten with a small random init at load time).
        # Slot [1] is "proprio modality" — semantics carry over from MUSE.
        self.proprio_proj = nn.Linear(self.per_frame_proprio_dim, self.d_model)
        self.proprio_pos_emb = nn.Parameter(torch.zeros(self.proprio_history_length, self.d_model))
        nn.init.normal_(self.proprio_pos_emb, std=0.02)
        self.modality_emb = nn.Parameter(torch.zeros(2, self.d_model))  # 0 = KP, 1 = proprio
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

        # σ is fixed — matches MUSE-Transformer deterministic-encoder regime.
        self.fixed_encoder_std = float(fixed_encoder_std)
        lmin, lmax = _muse_log_sigma_bounds(latent_sigma_min, latent_sigma_max)
        self._log_sigma_min = lmin
        self._log_sigma_max = lmax
        self._log_sigma_const = math.log(max(self.fixed_encoder_std, 1e-6))

        # Obs offsets.
        self._kp_block_dim = self.kp_lookahead_steps * self.kp_n_bodies * 3
        self._mask_block_dim = self.kp_lookahead_steps * self.kp_n_bodies
        self._proprio_block_dim = self.per_frame_proprio_dim * self.proprio_history_length
        self._expected_obs_dim = (
            self._kp_block_dim + self._mask_block_dim + self._proprio_block_dim
        )

        # Sequence: [CLS, kp_0..kp_{N-1}, proprio_0..proprio_{H-1}].
        self.seq_len = 1 + self.kp_n_bodies + self.proprio_history_length

        # Pre-built body-id grid (constant per forward — register so .to(device) follows the module).
        body_ids = torch.arange(self.kp_n_bodies, dtype=torch.long)
        self.register_buffer("_body_ids", body_ids.contiguous(), persistent=False)

        # Persistent buffer recording the slot offsets the kp_proj weight was trained for. Written
        # into checkpoints so warmstart loaders can discover the source layout without external
        # bookkeeping. Default is the legacy dense offset list ``(0..kp_lookahead_steps-1)``; the
        # outer policy module overwrites this to the resolved layout immediately after construction.
        default_offsets = torch.arange(self.kp_lookahead_steps, dtype=torch.long)
        self.register_buffer("kp_slot_offsets_buf", default_offsets.contiguous(), persistent=True)

    def expected_obs_dim(self) -> int:
        return self._expected_obs_dim

    def split_obs(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (kp [..., N, L*3], kp_mask [..., N], proprio_per_frame [..., H, P]).

        ``kp_mask`` is collapsed to per-body (length N) by taking the first lookahead frame's row —
        :func:`mdp.partial_kp_mask_lookahead` broadcasts the per-body mask across all L frames, so
        any frame slice carries the same bit per body. We keep the full mask block in obs for layout
        compatibility with the SAGE-KP env config; only the per-body collapse is consumed here.
        """
        if obs.shape[-1] != self._expected_obs_dim:
            raise RuntimeError(
                f"_MUSEKpTransformerEncoder obs dim mismatch: got {obs.shape[-1]}, "
                f"expected {self._expected_obs_dim} "
                f"(kp_block={self._kp_block_dim}, mask_block={self._mask_block_dim}, "
                f"proprio_block={self._proprio_block_dim})."
            )
        batch_shape = obs.shape[:-1]
        L, N = self.kp_lookahead_steps, self.kp_n_bodies

        off = 0
        # KP lookahead: row-major [L, N, 3] -> per-body sequence [N, L*3] (transpose L<->N then flatten).
        kp_flat = obs[..., off : off + self._kp_block_dim]
        kp_lhn = kp_flat.reshape(*batch_shape, L, N, 3)
        kp_per_body = kp_lhn.transpose(-3, -2).reshape(*batch_shape, N, L * 3)
        off += self._kp_block_dim

        # Mask block: [L, N] -> take first frame's mask (per-body).
        mask_flat = obs[..., off : off + self._mask_block_dim]
        mask_ln = mask_flat.reshape(*batch_shape, L, N)
        kp_mask = mask_ln[..., 0, :]  # [..., N]
        off += self._mask_block_dim

        # Proprio history: per-term contiguous (term-major) -> [..., H, P].
        proprio_per_frame, _ = _split_history_terms(
            obs, self.proprio_term_sizes, self.proprio_history_length, start_offset=off
        )
        return kp_per_body, kp_mask, proprio_per_frame

    def encode(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (μ, log_σ, proprio_per_frame [..., H, P]). log_σ is the broadcast fixed constant."""
        kp, kp_mask, proprio_per_frame = self.split_obs(obs)
        # Defense in depth: zero NaN values BEFORE any linear projection so K/V can't be poisoned.
        kp_clean = torch.nan_to_num(kp, 0.0)
        require_all_finite(proprio_per_frame, "MUSEKp proprio_per_frame")

        batch_shape = obs.shape[:-1]
        D = self.d_model
        N, H = self.kp_n_bodies, self.proprio_history_length

        # KP tokens: [..., N, D].
        kp_tokens = self.kp_proj(kp_clean)  # [..., N, D]
        body_token = self.body_id_emb(self._body_ids)  # [N, D]
        kp_tokens = kp_tokens + body_token + self.modality_emb[0].view(1, -1)  # broadcast

        # Proprio tokens: [..., H, D] (unchanged from MUSE-Transformer).
        proprio_tokens = self.proprio_proj(proprio_per_frame) + self.proprio_pos_emb.unsqueeze(0) \
            + self.modality_emb[1].view(1, -1)

        # CLS: [..., 1, D].
        cls = self.cls_token.expand(*batch_shape, 1, D)

        tokens = torch.cat([cls, kp_tokens, proprio_tokens], dim=-2)  # [..., 1+N+H, D]

        # key_padding_mask: True where attention should skip. CLS / proprio never masked; KP per-body.
        cls_pad = torch.zeros(*batch_shape, 1, dtype=torch.bool, device=obs.device)
        kp_pad = (kp_mask > 0)  # post-EmpiricalNormalization 0/1 -> sign threshold recovers boolean
        proprio_pad = torch.zeros(*batch_shape, H, dtype=torch.bool, device=obs.device)
        key_padding_mask = torch.cat([cls_pad, kp_pad, proprio_pad], dim=-1)  # [..., 1+N+H]

        # Flatten leading dims for transformer.
        tokens_flat = tokens.reshape(-1, self.seq_len, D)
        kpm_flat = key_padding_mask.reshape(-1, self.seq_len)
        out_flat = self.transformer(tokens_flat, src_key_padding_mask=kpm_flat)
        out = out_flat.reshape(*batch_shape, self.seq_len, D)

        cls_out = out[..., 0, :]  # [..., D]
        mu = self.mu_head(cls_out)
        log_sigma = torch.full_like(mu, self._log_sigma_const)
        return mu, log_sigma, proprio_per_frame

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)
        return mu + sigma * eps


class LatentBottleneckMUSEKp(nn.Module):
    """MUSE-KP student: KP-token transformer encoder + MUSE-style decoder + frozen MLP teacher.

    Loss / training contract: identical to :class:`LatentBottleneckMUSETransformer` (BC against
    teacher MLP + cosine smoothness on μ; deterministic encoder + unit-norm latent). The only change
    is the encoder input modality.
    """

    is_recurrent = False

    def __init__(
        self,
        num_student_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        *,
        proprio_dim: int,  # placeholder for cfg compatibility; unused (per-frame proprio derived).
        history_length: int = 5,
        proprio_term_sizes: Sequence[int] = (29, 29, 3, 29),
        # KP encoder (per-body sequence packing).
        kp_n_bodies: int = 14,
        kp_lookahead_steps: int = 10,
        kp_layout: str = "legacy",
        # Transformer body (must match the MUSE-Transformer ckpt for warmstart to load cleanly).
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
        aux_predictor_enabled: bool = False,
        **kwargs: Any,
    ):
        if kwargs:
            print(
                "LatentBottleneckMUSEKp.__init__ got unexpected arguments (ignored): "
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

        # Resolve KP slot layout. ``kp_layout`` selects a named layout from :data:`KP_LAYOUTS`;
        # the resolved tuple is stored so the warmstart loader can remap by slot identity. The
        # encoder's per-body projection consumes len(layout) * 3 dims; ``kp_lookahead_steps``
        # remains the per-body packed slot count for naming compatibility.
        self.kp_layout_name = str(kp_layout)
        self.kp_slot_offsets: tuple[int, ...] = kp_layout_by_name(self.kp_layout_name)
        if int(kp_lookahead_steps) != len(self.kp_slot_offsets):
            raise ValueError(
                f"LatentBottleneckMUSEKp: kp_lookahead_steps={kp_lookahead_steps} != "
                f"len(layout {self.kp_layout_name!r})={len(self.kp_slot_offsets)}. "
                f"Set kp_lookahead_steps to match the layout's slot count."
            )

        # KP encoder (with shared MUSE-Transformer encoder pieces inside).
        self.transformer_encoder = _MUSEKpTransformerEncoder(
            kp_n_bodies=int(kp_n_bodies),
            kp_lookahead_steps=int(kp_lookahead_steps),
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
        # Write the resolved slot offsets into the encoder's persistent buffer so ckpts saved from
        # this run carry the layout for downstream slot-aware warmstart.
        self.transformer_encoder.kp_slot_offsets_buf.copy_(
            torch.tensor(self.kp_slot_offsets, dtype=torch.long)
        )
        if self.transformer_encoder.expected_obs_dim() != self.num_student_obs:
            raise ValueError(
                f"LatentBottleneckMUSEKp: expected_obs_dim "
                f"({self.transformer_encoder.expected_obs_dim()}) != num_student_obs "
                f"({self.num_student_obs}). Check kp_n_bodies={kp_n_bodies}, "
                f"kp_lookahead_steps={kp_lookahead_steps}, proprio_term_sizes={proprio_term_sizes}, "
                f"history_length={history_length}."
            )

        # Decoder — shape mirrors LatentBottleneckMUSETransformer for warmstart compatibility.
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

        # Aux anchor predictor (Run A probe / Run B encoder-pressure). Off by default; enabled
        # by ``aux_predictor_enabled=True``. The algorithm decides stop-grad vs not via the
        # ``aux_stop_grad`` cfg flag — the predictor itself does not. Trained jointly via the
        # algorithm's update step; not used at inference.
        self.aux_predictor_enabled = bool(aux_predictor_enabled)
        if self.aux_predictor_enabled:
            self.aux_predictor: AuxAnchorPredictor | None = AuxAnchorPredictor(
                latent_dim=int(latent_dim),
                per_frame_proprio_dim=per_frame_proprio_dim,
            )
        else:
            self.aux_predictor = None

        # MLP teacher (always frozen). Same shape as MUSE-Transformer's teacher slot.
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

        # Action distribution std (matches MUSE convention).
        init_std = max(float(init_noise_std), 1.0e-6)
        self.std = nn.Parameter(init_std * torch.ones(self.num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args = False  # type: ignore[assignment]

        self.latent_log_sigma_min = self.transformer_encoder._log_sigma_min
        self.latent_log_sigma_max = self.transformer_encoder._log_sigma_max

        # Build the trainable-parameter group based on freeze_mode.
        self._build_student_group()

        print(
            f"[LatentBottleneckMUSEKp] num_student_obs={self.num_student_obs}, "
            f"num_teacher_obs={self.num_teacher_obs}, num_actions={self.num_actions}, "
            f"history_length={self.history_length}, "
            f"kp(n_bodies={kp_n_bodies}, lookahead={kp_lookahead_steps}), "
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

    # ----- Frozen-module bookkeeping ------------------------------------------------------------

    def _kp_only_params(self) -> list[nn.Parameter]:
        enc = self.transformer_encoder
        return list(enc.kp_proj.parameters()) + list(enc.body_id_emb.parameters()) + [enc.modality_emb]

    def _shared_encoder_modules(self) -> list[nn.Module]:
        """Modules in the encoder that are shape-shared with MUSE-Transformer (warmstart targets)."""
        enc = self.transformer_encoder
        return [enc.transformer, enc.proprio_proj, enc.mu_head]

    def _shared_encoder_params(self) -> list[nn.Parameter]:
        enc = self.transformer_encoder
        params: list[nn.Parameter] = []
        for m in self._shared_encoder_modules():
            params += list(m.parameters())
        # Standalone parameters (Parameter, not Module): proprio_pos_emb, cls_token. modality_emb is
        # KP-only by convention (slot [0] is KP, slot [1] is proprio — but they share storage; for
        # simplicity we group it with KP-only so it trains in the warmstart phase too).
        params += [enc.proprio_pos_emb, enc.cls_token]
        return params

    def _build_student_group(self) -> None:
        """Wire ``self.student`` to the trainable submodules per ``freeze_mode``; freeze the rest.

        ``self.student.parameters()`` is what the algorithm passes to the optimizer; only those
        weights move. Non-student weights additionally have ``requires_grad_(False)`` set so they
        don't accumulate grads even if mistakenly referenced.
        """
        # Always: teacher frozen.
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        if self.freeze_mode == "none":
            # Encoder + decoder train.
            for p in self.transformer_encoder.parameters():
                p.requires_grad_(True)
            for p in self.decoder.parameters():
                p.requires_grad_(True)
            student_modules: list[nn.Module] = [self.transformer_encoder, self.decoder]
        elif self.freeze_mode == "decoder_only":
            for p in self.transformer_encoder.parameters():
                p.requires_grad_(True)
            for p in self.decoder.parameters():
                p.requires_grad_(False)
            # Wrapping only the encoder in student so the optimizer skips decoder params.
            student_modules = [self.transformer_encoder]
        elif self.freeze_mode == "decoder_plus_shared_encoder":
            # Freeze decoder + shared encoder pieces; only KP-only encoder layers train.
            for p in self.decoder.parameters():
                p.requires_grad_(False)
            # First freeze all encoder params, then re-enable KP-only ones below.
            for p in self.transformer_encoder.parameters():
                p.requires_grad_(False)
            for p in self._kp_only_params():
                p.requires_grad_(True)
            # Build a thin Module that exposes only KP-only params via .parameters().
            class _KpTrainable(nn.Module):
                def __init__(self, encoder: _MUSEKpTransformerEncoder):
                    super().__init__()
                    self.kp_proj = encoder.kp_proj
                    self.body_id_emb = encoder.body_id_emb
                    # modality_emb is a single Parameter — wrap as ParameterList to keep .parameters() clean.
                    self.modality_emb = nn.ParameterList([encoder.modality_emb])
            student_modules = [_KpTrainable(self.transformer_encoder)]
        else:
            raise AssertionError(f"unhandled freeze_mode {self.freeze_mode!r}")

        # Aux predictor always trains when enabled, independent of freeze_mode. Its loss is small
        # (9 scalars total) and the comparison contract requires the predictor to converge in both
        # Run A (stop-grad probe) and Run B (encoder-pressure) so its loss curves are comparable.
        if self.aux_predictor is not None:
            for p in self.aux_predictor.parameters():
                p.requires_grad_(True)
            student_modules.append(self.aux_predictor)

        self.student = nn.ModuleList(student_modules)

    def set_freeze_mode(self, mode: str) -> bool:
        """Switch ``freeze_mode`` at runtime; returns ``True`` if the mode actually changed.

        Used by the algorithm-level warmup curriculum (e.g. start in ``decoder_only`` for the first
        N iters, then unfreeze to ``none``). The caller MUST rebuild the optimizer afterward — this
        method only updates ``self.student`` + ``requires_grad`` / ``eval`` flags; the optimizer
        still references the old param list until rebuilt.
        """
        if mode not in _VALID_FREEZE_MODES:
            raise ValueError(f"freeze_mode must be one of {_VALID_FREEZE_MODES}, got {mode!r}")
        if mode == self.freeze_mode:
            return False
        self.freeze_mode = str(mode)
        self._build_student_group()
        # Re-apply the train()-mode bookkeeping so newly frozen modules drop into eval immediately.
        self.train(self.training)
        return True

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen modules stay in eval regardless of parent train(). Decoder freezes in two of three
        # modes; teacher always.
        self.teacher.eval()
        if self.freeze_mode in ("decoder_only", "decoder_plus_shared_encoder"):
            self.decoder.eval()
        if self.freeze_mode == "decoder_plus_shared_encoder":
            # Shared encoder pieces in eval too (e.g. dropout disabled).
            for m in self._shared_encoder_modules():
                m.eval()
        return self

    def reset(self, dones=None, hidden_states=None) -> None:
        return

    # ----- Forward helpers ----------------------------------------------------------------------

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

    # ----- Public interface (matches LatentBottleneckMUSETransformer) ---------------------------

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
        z = mu if use_mean else self.transformer_encoder.reparameterize(mu, log_sigma)
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
        """Returns (action, μ, log_σ). μ returned post-normalization when enabled — same contract
        as :meth:`LatentBottleneckMUSETransformer.forward_for_update` so the algorithm's smoothness
        regularizer operates on the geometry the decoder sees.

        Side effect when ``aux_predictor`` is enabled: caches the post-normalized z + last-frame
        proprio (sharing the SAME autograd graph as the behavior loss) so a subsequent
        :meth:`forward_aux` call can skip a second encoder forward pass. This is load-bearing for
        Run B's memory budget — running a second forward holds the transformer encoder's
        activations twice, ~2× peak memory.
        """
        mu, log_sigma, proprio_per_frame = self.transformer_encoder.encode(observations)
        if self.deterministic_encoder or not (sample_z and self.training):
            z = mu
        else:
            z = self.transformer_encoder.reparameterize(mu, log_sigma)
        z = self._maybe_normalize_latent(z)
        mu = self._maybe_normalize_latent(mu)
        action = self._decode(z, proprio_per_frame)
        if not torch.isfinite(action).all():
            warnings.warn("LatentBottleneckMUSEKp: non-finite action; replacing with 0.")
            action = torch.nan_to_num(action)
        # Cache for forward_aux reuse. Overwritten on each call; previous cache is freed when its
        # owning autograd graph is freed (by the algorithm's backward + step at gradient_length).
        if self.aux_predictor is not None:
            self._aux_z_cache = z
            self._aux_proprio_last_cache = proprio_per_frame[..., -1, :]
        return action, mu, log_sigma

    def evaluate(self, teacher_observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.teacher(teacher_observations)

    def forward_aux(
        self, observations: torch.Tensor, stop_grad: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the aux anchor predictor: returns (speed_pred [..., 3], ori_pred [..., 6]).

        ``stop_grad=True`` detaches z and the last-frame proprio before the predictor sees them →
        encoder + proprio_proj receive no aux gradient (Run A probe). ``stop_grad=False`` lets the
        gradient flow into the encoder via z (Run B encoder-pressure). Either way, the predictor's
        own parameters receive aux gradient and train.

        Memory contract: when a cached (z, proprio_last) from the most recent
        :meth:`forward_for_update` is available, reuses it — the algorithm calls this immediately
        after forward_for_update inside the same gradient-accumulation window, so the cache shares
        the SAME autograd graph as the behavior loss. This avoids a duplicate encoder forward pass
        (which would otherwise hold all encoder activations twice → ~2× peak memory under Run B).

        Falls back to a fresh encode if cache is absent (e.g. called standalone outside the
        algorithm loop, for diagnostics).

        Raises if ``aux_predictor`` was not enabled at construction.
        """
        if self.aux_predictor is None:
            raise RuntimeError("forward_aux called but aux_predictor_enabled=False.")
        cached_z = getattr(self, "_aux_z_cache", None)
        cached_p = getattr(self, "_aux_proprio_last_cache", None)
        if cached_z is not None and cached_p is not None:
            z, proprio_last = cached_z, cached_p
        else:
            mu, _, proprio_per_frame = self.transformer_encoder.encode(observations)
            z = self._maybe_normalize_latent(mu)
            proprio_last = proprio_per_frame[..., -1, :]
        if stop_grad:
            z = z.detach()
            proprio_last = proprio_last.detach()
        return self.aux_predictor(z, proprio_last)

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

    # ----- Checkpoint loading -------------------------------------------------------------------

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Three accepted shapes:

        1. **MUSE-Transformer checkpoint** (``transformer_encoder.*`` / ``decoder.*`` / ``teacher.*``,
           no ``transformer_encoder.kp_proj.*``): copy the shape-compatible subset into the KP
           encoder + full decoder + full teacher. KP-only layers (``kp_proj``, ``body_id_emb``,
           ``modality_emb`` slot [0]) keep their small random init. Skipped MUSE keys (e.g.
           ``goal_proj.*``, ``goal_pos_emb``) are reported.
        2. **PHC+ stage-1** (``actor.*``): only fills the MLP teacher slot.
        3. **MUSE-KP resume** (contains ``transformer_encoder.kp_proj.*``): full load.
           When the ckpt's ``kp_proj.weight`` dim differs from the current model (e.g. shifting
           from the legacy 33-dim layout to a 36-dim log-spaced layout), the loader does a
           **slot-aware column remap** by matching slot offsets: shared slots copy their (D, 3)
           column block; new slots stay at the current model's init (small random); dropped slots
           are discarded. Requires the ckpt to carry a ``kp_slot_offsets`` buffer; otherwise the
           legacy 33-dim layout is assumed (compat with pre-layout-aware checkpoints).
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
            self.loaded_teacher = True
            return False

        if has_kp or has_muse_modules:
            # Slot-aware kp_proj remap: if the ckpt's kp_proj.weight column count differs from the
            # current model, rebuild the weight by slot-offset identity and substitute into the
            # state_dict before the parent load_state_dict pass (which would otherwise filter the
            # mismatched key out and leave kp_proj at random init for the matching columns too).
            state_dict = self._maybe_remap_kp_proj(state_dict)

            # Shape-filter to drop incompatible keys (e.g. MUSE's goal_proj which doesn't exist here).
            self_sd = self.state_dict()
            filtered: dict[str, torch.Tensor] = {}
            skipped: list[str] = []
            for k, v in state_dict.items():
                if k in self_sd and self_sd[k].shape == v.shape:
                    filtered[k] = v
                else:
                    skipped.append(k)
            if skipped:
                print(
                    f"[LatentBottleneckMUSEKp] Skipping {len(skipped)} ckpt keys (shape mismatch or "
                    f"absent in current model): {skipped[:5]}{'...' if len(skipped) > 5 else ''}"
                )
            super().load_state_dict(filtered, strict=False)
            self.loaded_teacher = True
            self.teacher.eval()
            # Re-apply freeze_mode bookkeeping: load_state_dict can re-enable requires_grad on some
            # params via assignment; redo the freezing pass to keep the contract intact.
            self._build_student_group()
            return has_kp

        raise ValueError(
            "state_dict must contain actor.*, or transformer_encoder.*/decoder.*/teacher.* keys."
        )

    def _maybe_remap_kp_proj(self, state_dict: dict) -> dict:
        """If the ckpt's kp_proj.weight has a different per-body dim than the current model, return
        a copy of ``state_dict`` with the kp_proj.weight remapped by slot-offset identity.

        Matching rule: each NEW slot's 3 weight columns come from the OLD slot with the same offset
        if it exists; otherwise stays at the current model's init (small random). Bias is unchanged
        if shapes match. The remap **mutates the in-memory state_dict copy**; the original is left
        intact for downstream consumers.

        Layout discovery: the ckpt is expected to carry a buffer at
        ``transformer_encoder.kp_slot_offsets_buf`` (registered in :class:`_MUSEKpTransformerEncoder`)
        encoding the OLD layout. When that buffer is absent (pre-layout-aware checkpoints), the
        legacy dense layout :data:`KP_LAYOUT_LEGACY` is assumed — this is the documented contract
        for migrating yesterday's MUSE-Kp run forward.
        """
        kp_w_key = "transformer_encoder.kp_proj.weight"
        kp_b_key = "transformer_encoder.kp_proj.bias"
        if kp_w_key not in state_dict:
            return state_dict

        old_W = state_dict[kp_w_key]  # [d_model, L_old * 3]
        target_W = self.transformer_encoder.kp_proj.weight  # [d_model, L_new * 3]
        if old_W.shape == target_W.shape:
            return state_dict  # nothing to remap

        d_model = target_W.shape[0]
        if old_W.shape[0] != d_model:
            print(
                f"[LatentBottleneckMUSEKp] kp_proj d_model mismatch (ckpt={old_W.shape[0]}, "
                f"model={d_model}); skipping slot-aware remap, parent loader will drop the key."
            )
            return state_dict
        if old_W.shape[1] % 3 != 0:
            print(
                f"[LatentBottleneckMUSEKp] kp_proj ckpt column count {old_W.shape[1]} not "
                f"divisible by 3; skipping slot-aware remap."
            )
            return state_dict

        L_old = old_W.shape[1] // 3
        L_new = target_W.shape[1] // 3

        # Try to discover the OLD layout from the ckpt; fall back to KP_LAYOUT_LEGACY.
        old_offsets_key = "transformer_encoder.kp_slot_offsets_buf"
        if old_offsets_key in state_dict:
            old_offsets = tuple(int(x) for x in state_dict[old_offsets_key].tolist())
        elif L_old == len(KP_LAYOUT_LEGACY):
            old_offsets = KP_LAYOUT_LEGACY
            print(
                f"[LatentBottleneckMUSEKp] kp_slot_offsets_buf missing from ckpt; assuming legacy "
                f"layout {KP_LAYOUT_LEGACY} (L_old={L_old})."
            )
        else:
            print(
                f"[LatentBottleneckMUSEKp] kp_slot_offsets_buf missing AND L_old={L_old} doesn't "
                f"match legacy ({len(KP_LAYOUT_LEGACY)}); skipping slot-aware remap, kp_proj will "
                f"stay at random init."
            )
            return state_dict
        new_offsets = self.kp_slot_offsets

        # Build the remapped weight: start from the current model's init (preserves random init
        # on new slots); overwrite columns whose slot offset exists in the old layout.
        new_W = target_W.detach().clone()
        old_to_idx = {off: i for i, off in enumerate(old_offsets)}
        copied = 0
        for new_idx, off in enumerate(new_offsets):
            if off in old_to_idx:
                old_idx = old_to_idx[off]
                new_W[:, new_idx * 3 : (new_idx + 1) * 3] = (
                    old_W[:, old_idx * 3 : (old_idx + 1) * 3]
                )
                copied += 1

        out = dict(state_dict)
        out[kp_w_key] = new_W
        # Bias is per-d_model (not per-slot), so it copies straight if shapes line up.
        if kp_b_key in state_dict:
            ckpt_b = state_dict[kp_b_key]
            if ckpt_b.shape == self.transformer_encoder.kp_proj.bias.shape:
                out[kp_b_key] = ckpt_b
        # Drop the OLD layout buffer from the ckpt so the parent loader doesn't try to fit it into
        # the NEW buffer (which has a different length); the new buffer keeps its current value.
        out.pop(old_offsets_key, None)
        print(
            f"[LatentBottleneckMUSEKp] kp_proj slot-aware remap: ckpt L_old={L_old} "
            f"({old_offsets}) → model L_new={L_new} ({new_offsets}); copied {copied}/{L_new} slot "
            f"blocks; remaining {L_new - copied} new slots stay at random init."
        )
        return out

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None) -> None:
        return
