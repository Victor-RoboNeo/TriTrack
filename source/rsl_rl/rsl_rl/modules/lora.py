# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LoRA adapters for the MUSE-Kp encoder (M3 of the latent-RL plan).

Implemented via ``torch.nn.utils.parametrize`` so the effective weight
``W_eff = W_base + (alpha/r)·B·A`` is computed transparently inside the existing
module forward — no need to reimplement attention/FFN. ``W_base`` (now stored at
``module.parametrizations.<name>.original``) is frozen; only ``A``/``B`` train.
``B`` is zero-initialized so the adapter is a **no-op at step 0** (safe start —
the policy is exactly the distilled one until LoRA learns).

Target tokens (decision: attention q/k/v + o primary, ``mu_head`` default,
FFN/kp_proj ablatable):
  - ``attn_qkv`` : fused ``self_attn.in_proj_weight`` ([3D, D]) of every block
  - ``attn_out`` : ``self_attn.out_proj.weight`` of every block
  - ``ffn``      : ``linear1.weight`` + ``linear2.weight`` of every block
  - ``mu_head``  : the final latent projection ``mu_head.weight``
  - ``kp_proj``  : the per-body KP input projection ``kp_proj.weight``

Injection is deferred until AFTER the distilled warmstart is loaded (the loader
matches checkpoint keys by name, and parametrization renames ``weight`` ->
``parametrizations.weight.original``). :class:`LatentRLActorCritic` calls
:func:`apply_lora_to_encoder` post-warmstart; see its ``_ensure_lora_applied``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


VALID_LORA_TARGETS = ("attn_qkv", "attn_out", "ffn", "mu_head", "kp_proj")


class _LoRADelta(nn.Module):
    """Parametrization: ``W -> W + (alpha/r)·(B @ A)``. ``W`` (the registered
    original) is frozen by the caller; ``A``/``B`` are the trainable adapter."""

    def __init__(self, out_features: int, in_features: int, r: int, alpha: float):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {r}")
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        # A ~ kaiming (so the adapter has well-scaled gradients once B leaves 0);
        # B = 0  =>  delta = 0 at init  =>  exact distilled policy at step 0.
        self.A = nn.Parameter(torch.empty(self.r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, self.r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return W + self.scaling * (self.B @ self.A)


def _resolve_lora_linears(encoder: nn.Module, targets: Sequence[str]):
    """Yield ``(module, weight_attr_name)`` pairs to LoRA-parametrize.

    ``encoder`` is a ``_MUSEKpTransformerEncoder``: ``.transformer`` is an
    ``nn.TransformerEncoder`` whose ``.layers`` are ``nn.TransformerEncoderLayer``
    (``.self_attn`` = ``nn.MultiheadAttention`` with fused ``in_proj_weight`` +
    ``out_proj`` Linear; ``.linear1`` / ``.linear2`` FFN Linears).
    """
    targets = list(targets)
    bad = [t for t in targets if t not in VALID_LORA_TARGETS]
    if bad:
        raise ValueError(f"Unknown LoRA target(s) {bad}; valid: {VALID_LORA_TARGETS}")

    for layer in encoder.transformer.layers:
        if "attn_qkv" in targets:
            sa = layer.self_attn
            if getattr(sa, "in_proj_weight", None) is not None:
                yield sa, "in_proj_weight"
            else:
                # separate q/k/v projections (kdim/vdim != embed_dim) — not our
                # config, but handle gracefully.
                for n in ("q_proj_weight", "k_proj_weight", "v_proj_weight"):
                    if getattr(sa, n, None) is not None:
                        yield sa, n
        if "attn_out" in targets:
            yield layer.self_attn.out_proj, "weight"
        if "ffn" in targets:
            yield layer.linear1, "weight"
            yield layer.linear2, "weight"
    if "mu_head" in targets:
        yield encoder.mu_head, "weight"
    if "kp_proj" in targets:
        yield encoder.kp_proj, "weight"


def apply_lora_to_encoder(
    encoder: nn.Module,
    targets: Sequence[str],
    r: int,
    alpha: float,
) -> int:
    """Freeze every encoder parameter, then attach a :class:`_LoRADelta`
    parametrization to each resolved target weight. Returns the number of
    parametrized tensors. Idempotent-safe: skips weights already parametrized.
    """
    for p in encoder.parameters():
        p.requires_grad_(False)

    n = 0
    for module, wname in _resolve_lora_linears(encoder, targets):
        if parametrize.is_parametrized(module, wname):
            continue
        W = getattr(module, wname)
        out_features, in_features = W.shape[0], W.shape[1]
        delta = _LoRADelta(out_features, in_features, r=r, alpha=alpha).to(
            device=W.device, dtype=W.dtype
        )
        parametrize.register_parametrization(module, wname, delta)
        # Frozen base (the registered original) + trainable adapter.
        module.parametrizations[wname].original.requires_grad_(False)
        delta.A.requires_grad_(True)
        delta.B.requires_grad_(True)
        n += 1
    return n
