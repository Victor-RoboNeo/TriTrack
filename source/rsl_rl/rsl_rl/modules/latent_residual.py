# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Residual latent corrector (``g_φ``) for the MUSE-Kp latent-RL `residual`
adapter (M4).

Decision D2 (locked): ``g_φ`` is a **full-capacity parallel corrector** that
sees the encoder's input *and* its answer, and emits a correction ``Δz``; the
corrected mean latent is ``normalize(μ̂ + α·Δz)``. Encoder + decoder stay
frozen — only ``g_φ`` (+ the latent Gaussian std + critic) trains. The output
head is small-gain initialized so ``Δz ≈ 0`` at step 0 ⇒ the policy starts
exactly at the distilled policy (safe start; ResMimic-style).

**Why a (shallow) transformer, not an MLP** (see plan §3.3 / the design
discussion): a flat MLP over the 738-d KP obs spends ~70% of its params on a
``obs_dim×hidden`` first layer and must *learn* the combinatorial visibility
handling from scratch. A per-body-token transformer instead (a) handles masked
keypoints **structurally** via ``src_key_padding_mask`` — no NaN scrubbing, no
"learn to ignore garbage" — exactly like the MUSE-Kp encoder, and (b) at a
shallow/narrow size is ~10× *fewer* params than the MLP (input proj decouples
from the flattened obs size). Param efficiency is a bonus; correct masking +
body equivariance is the reason.

Tokenization mirrors :class:`_MUSEKpTransformerEncoder` but with its OWN small
weights (a true *parallel* corrector per D2). The frozen encoder's unit-norm
latent ``μ̂`` is injected as one extra conditioning token (the "sees the
encoder's attempt" property). The caller passes already-split tensors
(``kp``/``kp_mask``/``proprio``) so this module needs no obs-layout knowledge.

Default size: ``d_model=64, num_layers=1, nhead=4, ffn=128`` (~45k params,
~12× under the MLP). For more upper-body-under-mask capacity, a documented
larger combo is ``d_model=96, num_layers=2, nhead=4, ffn=256`` (~191k, still
~3× under the MLP) — set via the ``residual_*`` policy-cfg knobs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualLatentCorrector(nn.Module):
    """Shallow per-body-token transformer ``g_φ``: ``(kp, kp_mask, proprio, μ̂)
    -> Δz``.

    Token sequence: ``[CLS, kp_0..N-1, proprio_0..H-1, μ̂]``. Masked KP bodies
    are excluded via ``src_key_padding_mask`` (and NaN-scrubbed before the
    projection, defense-in-depth, exactly like the encoder). Returns the raw
    ``Δz`` only — the caller does ``normalize(μ̂ + α·Δz)`` so the unit-norm
    latent geometry stays centralized with the rest of the policy.

    Args:
        kp_n_bodies: N — number of KP body tokens.
        kp_feat_dim: per-body packed KP feature dim (= ``kp_lookahead_steps*3``).
        proprio_feat_dim: per-frame proprio dim P.
        proprio_history_length: H proprio frames.
        latent_dim: MUSE-Kp latent dim (= μ̂ dim = Δz dim).
        d_model / num_layers / nhead / ffn: transformer size.
        last_layer_gain: Xavier gain on the Δz head (small ⇒ Δz≈0 at init).
        alpha: scalar on Δz in ``μ̂ + α·Δz`` (exposed as ``self.alpha``).
    """

    def __init__(
        self,
        kp_n_bodies: int,
        kp_feat_dim: int,
        proprio_feat_dim: int,
        proprio_history_length: int,
        latent_dim: int,
        obstacle_n: int = 0,
        obstacle_feat_dim: int = 0,
        d_model: int = 64,
        num_layers: int = 1,
        nhead: int = 4,
        ffn: int = 128,
        last_layer_gain: float = 0.01,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.kp_n_bodies = int(kp_n_bodies)
        self.proprio_history_length = int(proprio_history_length)
        self.obstacle_n = int(obstacle_n)
        d = int(d_model)
        self.d_model = d

        # Own (fresh) tokenizer — parallel to, not shared with, the frozen encoder.
        self.kp_proj = nn.Linear(int(kp_feat_dim), d)
        self.proprio_proj = nn.Linear(int(proprio_feat_dim), d)
        self.mu_proj = nn.Linear(int(latent_dim), d)
        self.body_id_emb = nn.Embedding(self.kp_n_bodies, d)
        # Optional obstacle tokens (one per box): the corrector sees the obstacles structurally
        # (invalid boxes masked via key_padding, exactly like masked KP), so Δz can condition on
        # them. obstacle_n=0 (default) -> no obstacle head -> byte-identical to the writing/general
        # residual policy.
        if self.obstacle_n > 0:
            self.obstacle_proj = nn.Linear(int(obstacle_feat_dim), d)
        # modality_emb slots: [0]=kp, [1]=proprio, [2]=μ̂, [3]=obstacle (only if obstacle_n>0).
        self.modality_emb = nn.Parameter(torch.zeros(4 if self.obstacle_n > 0 else 3, d))
        self.proprio_pos_emb = nn.Parameter(torch.zeros(self.proprio_history_length, d))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        for p in (self.modality_emb, self.proprio_pos_emb, self.cls_token):
            nn.init.normal_(p, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(nhead),
            dim_feedforward=int(ffn),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(num_layers))

        self.out_head = nn.Linear(d, int(latent_dim))
        # Small-gain init => Δz ≈ 0 at step 0 (safe start: policy == distilled).
        nn.init.xavier_uniform_(self.out_head.weight, gain=float(last_layer_gain))
        nn.init.zeros_(self.out_head.bias)

        self.register_buffer(
            "_body_ids", torch.arange(self.kp_n_bodies, dtype=torch.long), persistent=False
        )

    def forward(
        self,
        kp: torch.Tensor,            # [B, N, kp_feat_dim] (NaN at masked bodies)
        kp_mask: torch.Tensor,       # [B, N] (1.0 = masked, 0.0 = visible)
        proprio: torch.Tensor,       # [B, H, proprio_feat_dim]
        mu_hat: torch.Tensor,        # [B, latent_dim] (frozen encoder's unit-norm latent)
        obstacle_feat: torch.Tensor | None = None,  # [B, K, obstacle_feat_dim] (per-box feature)
        obstacle_mask: torch.Tensor | None = None,  # [B, K] (True = invalid/empty box -> excluded)
    ) -> torch.Tensor:
        B = kp.shape[0]
        d = self.d_model
        dev = kp.device

        kp_clean = torch.nan_to_num(kp, 0.0)  # defense-in-depth (mirrors encoder)
        kp_tok = (
            self.kp_proj(kp_clean)
            + self.body_id_emb(self._body_ids)
            + self.modality_emb[0].view(1, 1, d)
        )  # [B, N, d]
        prop_tok = (
            self.proprio_proj(proprio)
            + self.proprio_pos_emb.unsqueeze(0)
            + self.modality_emb[1].view(1, 1, d)
        )  # [B, H, d]
        mu_tok = (self.mu_proj(mu_hat) + self.modality_emb[2]).unsqueeze(1)  # [B, 1, d]
        cls = self.cls_token.expand(B, 1, d)  # [B, 1, d]

        # key_padding_mask True = excluded. CLS / proprio / μ̂ never masked; KP per-body and
        # obstacle per-box from their visibility/validity masks — structural masking.
        token_list = [cls, kp_tok, prop_tok, mu_tok]
        pad_list = [
            torch.zeros(B, 1, dtype=torch.bool, device=dev),                     # cls
            kp_mask > 0,                                                         # kp (1=masked)
            torch.zeros(B, proprio.shape[1], dtype=torch.bool, device=dev),     # proprio
            torch.zeros(B, 1, dtype=torch.bool, device=dev),                    # μ̂
        ]
        if self.obstacle_n > 0 and obstacle_feat is not None:
            obs_tok = self.obstacle_proj(torch.nan_to_num(obstacle_feat, 0.0)) + self.modality_emb[3].view(1, 1, d)
            token_list.append(obs_tok)  # [B, K, d]
            if obstacle_mask is None:
                obstacle_mask = torch.zeros(B, obs_tok.shape[1], dtype=torch.bool, device=dev)
            pad_list.append(obstacle_mask.to(torch.bool))

        tokens = torch.cat(token_list, dim=1)
        key_padding_mask = torch.cat(pad_list, dim=1)

        out = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        return self.out_head(out[:, 0, :])  # CLS pool -> Δz [B, latent_dim]


def build_residual_corrector(
    *,
    kp_n_bodies: int,
    kp_feat_dim: int,
    proprio_feat_dim: int,
    proprio_history_length: int,
    latent_dim: int,
    obstacle_n: int = 0,
    obstacle_feat_dim: int = 0,
    d_model: int = 64,
    num_layers: int = 1,
    nhead: int = 4,
    ffn: int = 128,
    last_layer_gain: float = 0.01,
    alpha: float = 1.0,
) -> ResidualLatentCorrector:
    return ResidualLatentCorrector(
        kp_n_bodies=kp_n_bodies,
        kp_feat_dim=kp_feat_dim,
        proprio_feat_dim=proprio_feat_dim,
        proprio_history_length=proprio_history_length,
        latent_dim=latent_dim,
        obstacle_n=obstacle_n,
        obstacle_feat_dim=obstacle_feat_dim,
        d_model=d_model,
        num_layers=num_layers,
        nhead=nhead,
        ffn=ffn,
        last_layer_gain=last_layer_gain,
        alpha=alpha,
    )
