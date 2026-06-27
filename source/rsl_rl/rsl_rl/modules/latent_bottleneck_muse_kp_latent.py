"""MUSE-Kp latent-distillation student: KP encoder + frozen JC MUSE-Transformer teacher.

Extends :class:`LatentBottleneckMUSEKp` with a **frozen JC MUSE-Transformer encoder**
(``self.jc_encoder``) that produces the latent target μ_jc from the privileged (full, unmasked)
joint-command obs. The shared decoder is loaded from the same JC checkpoint and frozen
(``freeze_mode`` keeps it out of the optimizer). The KP encoder is warmstarted from the JC
backbone and trained to match μ_jc in latent space.

Latent is **unit-normalized** (``latent_normalize=True``, matching the JC MUSE-T cosine
recipe, 2026-05-18): the frozen decoder consumes the L2-normalized z, so the latent-alignment
target is z_jc = normalize(μ_jc) and the student is matched as z_kp = normalize(μ_kp) — both
on the unit sphere, the same geometry the decoder sees.

Single-checkpoint contract: pass the JC MUSE-Transformer ``.pt`` via ``--teacher_checkpoint``.
The runner's teacher-load calls :meth:`load_state_dict` with that ckpt's ``model_state_dict``
(keys ``transformer_encoder.*`` / ``decoder.*`` / ``teacher.*``, no ``jc_encoder.*``). This
class then (1) replicates ``transformer_encoder.*`` onto ``jc_encoder.*`` (the frozen teacher),
(2) lets the base loader warmstart the KP backbone + load the shared decoder, and freezes the
JC teacher encoder. A native/full state_dict (resume, multi-GPU broadcast) already carries
``jc_encoder.*`` and is loaded verbatim — the remap is gated on the absence of ``jc_encoder.*``
so a broadcast can never overwrite the frozen teacher with the student backbone.
"""

from __future__ import annotations

from typing import Any

import torch

from .latent_bottleneck_muse_kp import LatentBottleneckMUSEKp
from .latent_bottleneck_muse_transformer import _MUSETransformerEncoder


class LatentBottleneckMUSEKpLatent(LatentBottleneckMUSEKp):
    """KP latent-distillation student with a frozen JC MUSE-Transformer teacher + shared decoder."""

    def __init__(
        self,
        num_student_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        *,
        jc_goal_term_sizes: tuple[int, ...] = (58, 6, 3, 3, 3),
        jc_proprio_term_sizes: tuple[int, ...] = (29, 29, 3, 29),
        jc_mask_term_size: int = 1,
        **kwargs: Any,
    ):
        # JC MUSE-T recipe (2026-05-18 update): deterministic encoder + UNIT-NORM latent. The
        # JC teacher now trains with latent_normalize=True (cosine-smoothness recipe), so the
        # frozen decoder consumes the L2-normalized z — the latent target (z_jc), the student
        # latent (z_kp) and the decoder input must ALL be unit-normed. Forced here so a play.py
        # restore / cfg drift can't desync the student from the frozen decoder's input contract.
        kwargs["latent_normalize"] = True
        kwargs["deterministic_encoder"] = True

        # Capture backbone hyper-params (consumed by super().__init__) so the JC teacher encoder is
        # built shape-identical to the KP encoder's shared backbone + the JC checkpoint.
        history_length = int(kwargs.get("history_length", 5))
        latent_dim = int(kwargs.get("latent_dim", 16))
        d_model = int(kwargs.get("d_model", 192))
        nhead = int(kwargs.get("nhead", 4))
        num_layers = int(kwargs.get("num_layers", 2))
        ffn_dim = int(kwargs.get("ffn_dim", 768))
        activation = str(kwargs.get("activation", "gelu"))
        latent_predict_std_min = float(kwargs.get("latent_predict_std_min", 0.001))
        latent_predict_std_max = float(kwargs.get("latent_predict_std_max", 10.0))
        fixed_encoder_std = float(kwargs.get("fixed_encoder_std", 1.0))
        encoder_dropout = float(kwargs.get("encoder_dropout", 0.0))

        super().__init__(num_student_obs, num_teacher_obs, num_actions, **kwargs)

        # Expose latent_dim on the POLICY (base only sets it on transformer_encoder). The runner
        # sizes the teacher_latent rollout buffer from getattr(policy, "latent_dim", 64) — without
        # this it defaults to 64 and mismatches μ_jc (16) in storage.add_transitions.
        self.latent_dim = int(latent_dim)

        self.jc_goal_term_sizes = tuple(int(s) for s in jc_goal_term_sizes)
        self.jc_proprio_term_sizes = tuple(int(s) for s in jc_proprio_term_sizes)
        self.jc_mask_term_size = int(jc_mask_term_size)
        # Offset of the proprio block within the JC teacher obs (= sum of JC goal terms). Used by
        # the runner to copy the proprio normalizer slice from the JC teacher (proprio is NOT the
        # teacher obs tail — a trailing mask bit follows it).
        self.jc_goal_dim = sum(self.jc_goal_term_sizes)
        # Real proprio block dim (split-normalizer "proprio tail"); base __init__ del'd proprio_dim.
        self.proprio_dim = self.per_frame_proprio_dim * self.history_length

        # Frozen JC MUSE-Transformer teacher encoder: full unmasked JC obs -> μ_jc.
        self.jc_encoder = _MUSETransformerEncoder(
            goal_term_sizes=self.jc_goal_term_sizes,
            proprio_term_sizes=self.jc_proprio_term_sizes,
            mask_term_size=self.jc_mask_term_size,
            history_length=history_length,
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            activation=activation,
            latent_sigma_min=latent_predict_std_min,
            latent_sigma_max=latent_predict_std_max,
            fixed_encoder_std=fixed_encoder_std,
            predict_sigma=False,
            dropout=encoder_dropout,
        )
        for p in self.jc_encoder.parameters():
            p.requires_grad_(False)
        self.jc_encoder.eval()

        exp = self.jc_encoder.expected_obs_dim()
        if exp != self.num_teacher_obs:
            print(
                f"[LatentBottleneckMUSEKpLatent] WARNING: jc_encoder expected teacher obs dim "
                f"{exp} != num_teacher_obs {self.num_teacher_obs}. Check jc_goal_term_sizes="
                f"{self.jc_goal_term_sizes}, jc_proprio_term_sizes={self.jc_proprio_term_sizes}, "
                f"jc_mask_term_size={self.jc_mask_term_size}, history_length={history_length}."
            )
        print(
            f"[LatentBottleneckMUSEKpLatent] JC teacher encoder built (expected_obs_dim={exp}, "
            f"jc_goal_dim={self.jc_goal_dim}, proprio_dim={self.proprio_dim}); FROZEN."
        )

    # ----- Latent-distillation interface --------------------------------------------------------

    @torch.no_grad()
    def get_teacher_targets(
        self, teacher_observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(z_jc, action_jc) from the frozen JC teacher + frozen shared decoder.

        z_jc = L2-normalize(μ_jc) (latent_normalize=True — the JC teacher's unit-norm recipe);
        action_jc = decoder([z_jc, proprio_jc]) through the same frozen decoder the student uses,
        so a latent match (z_kp ≈ z_jc on the unit sphere) ⟹ action match at convergence. The
        returned latent target is the NORMALIZED z_jc so the algorithm's latent loss lives in
        the same geometry the decoder consumes.
        """
        mu_jc, _, proprio_per_frame = self.jc_encoder.encode(teacher_observations)
        z_jc = self._maybe_normalize_latent(mu_jc)  # unit-norm (latent_normalize=True)
        action_jc = self._decode(z_jc, proprio_per_frame)
        return z_jc, action_jc

    def encode_latent(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(z_kp, log_σ, proprio_per_frame) from the trainable KP encoder.

        z_kp = L2-normalize(μ_kp) (latent_normalize=True) so the algorithm's latent loss
        compares z_kp vs the teacher's z_jc on the unit sphere — the SAME geometry the frozen
        decoder consumes. Falls back to raw μ if latent_normalize=False (kept generic).
        """
        mu, log_sigma, proprio_per_frame = self.transformer_encoder.encode(observations)
        return self._maybe_normalize_latent(mu), log_sigma, proprio_per_frame

    # ----- Frozen-module bookkeeping ------------------------------------------------------------

    def train(self, mode: bool = True):
        super().train(mode)
        # JC teacher encoder is always frozen / eval (no grad, no dropout drift).
        self.jc_encoder.eval()
        return self

    # ----- Checkpoint loading -------------------------------------------------------------------

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        keys = list(state_dict.keys())
        has_jc = any(k.startswith("jc_encoder.") for k in keys)
        if not has_jc:
            # Raw JC MUSE-Transformer ckpt: encoder keys are ``transformer_encoder.*``. Replicate
            # them onto ``jc_encoder.*`` (frozen teacher) IN ADDITION to the base loader's KP
            # backbone warmstart + shared-decoder load. Gated on absence of ``jc_encoder.*`` so a
            # full/native state_dict (resume / DDP broadcast) never clobbers the teacher.
            merged = dict(state_dict)
            for k, v in state_dict.items():
                if k.startswith("transformer_encoder."):
                    merged["jc_encoder." + k[len("transformer_encoder.") :]] = v
            ret = super().load_state_dict(merged, strict=False)
        else:
            ret = super().load_state_dict(state_dict, strict=strict)
        # Latent-distillation contract: the JC teacher encoder is always frozen.
        for p in self.jc_encoder.parameters():
            p.requires_grad_(False)
        self.jc_encoder.eval()
        return ret
