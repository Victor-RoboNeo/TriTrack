#!/usr/bin/env python3
"""Interaction-conditioned latent residual. No terrain IDs / scan / experts / gate.

Token is packed from existing 750-D Stage-2 obs + frozen z_nom only:
    z0 (16) + e (9) + de (9) + current proprio frame (90) = 124-D
Contact / projected gravity are logged in eval, never fed to the adapter.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.modules.terrain_residual import apply_tangent_correction

LATENT = 16
E_DIM = 9
PROPRIO_FRAME = 90  # joint_pos 29 + joint_vel 29 + ang_vel 3 + action 29
TOKEN_DIM = LATENT + E_DIM + E_DIM + PROPRIO_FRAME  # 124
DEFAULT_R_MAX = math.tan(math.radians(5.0))


def pack_token(
    z_nom: torch.Tensor,
    e: torch.Tensor,
    e_dot: torch.Tensor,
    proprio_per_frame: torch.Tensor,
) -> torch.Tensor:
    """Causal token. ``proprio_per_frame`` is [B, H, P]; use the last frame only."""
    prop = proprio_per_frame[:, -1, :].reshape(z_nom.shape[0], -1)
    if int(prop.shape[-1]) != PROPRIO_FRAME:
        # pad / trim to keep a fixed GRU input size
        out = z_nom.new_zeros(z_nom.shape[0], PROPRIO_FRAME)
        n = min(PROPRIO_FRAME, int(prop.shape[-1]))
        out[:, :n] = prop[:, :n]
        prop = out
    return torch.cat([z_nom, e, e_dot, prop], dim=-1)


class InstantResidualMLP(nn.Module):
    """M1: current token → Δz. Last Linear zero-init so π(0)=Stage2."""

    def __init__(self, token_dim: int = TOKEN_DIM, latent_dim: int = LATENT, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.net(token)


class HistoryResidualGRU(nn.Module):
    """M2/M3: causal GRU over a token ring → Δz. Last Linear zero-init."""

    def __init__(
        self,
        token_dim: int = TOKEN_DIM,
        latent_dim: int = LATENT,
        hidden: int = 128,
        layers: int = 1,
    ):
        super().__init__()
        self.gru = nn.GRU(token_dim, hidden, num_layers=int(layers), batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        """hist [B, T, D] → Δz [B, latent]."""
        y, _h = self.gru(hist)
        return self.head(y[:, -1])


class InteractionResidual(nn.Module):
    def __init__(
        self,
        encoder: str = "mlp",
        history_len: int = 16,
        tangent: bool = True,
        beta: float = 1.0,
        r_max: float = DEFAULT_R_MAX,
        hidden: int = 128,
        token_dim: int = TOKEN_DIM,
        latent_dim: int = LATENT,
    ):
        super().__init__()
        self.encoder_name = str(encoder)
        self.history_len = int(history_len)
        self.tangent = bool(tangent)
        self.beta = float(beta)
        self.r_max = float(r_max)
        self.token_dim = int(token_dim)
        self.latent_dim = int(latent_dim)
        if self.encoder_name == "gru":
            self.body = HistoryResidualGRU(token_dim, latent_dim, hidden=hidden)
        else:
            self.body = InstantResidualMLP(token_dim, latent_dim, hidden=hidden)
        self.register_buffer("_hist", torch.zeros(1, self.history_len, self.token_dim), persistent=False)
        self.register_buffer("_len", torch.zeros(1, dtype=torch.long), persistent=False)

    def extra_repr(self) -> str:
        return (
            f"enc={self.encoder_name} T={self.history_len} tangent={self.tangent} "
            f"beta={self.beta} r_max={self.r_max:.4f}"
        )

    def _ensure(self, n: int, device, dtype) -> None:
        if int(self._hist.shape[0]) == n and self._hist.device == device:
            return
        self._hist = torch.zeros(n, self.history_len, self.token_dim, device=device, dtype=dtype)
        self._len = torch.zeros(n, dtype=torch.long, device=device)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if dones is None:
            self._hist.zero_()
            self._len.zero_()
            return
        d = dones.reshape(-1).to(device=self._hist.device, dtype=torch.bool)
        if int(d.shape[0]) != int(self._hist.shape[0]):
            return
        self._hist[d] = 0
        self._len[d] = 0

    def push(self, token: torch.Tensor) -> torch.Tensor:
        n = int(token.shape[0])
        self._ensure(n, token.device, token.dtype)
        self._hist = torch.cat([self._hist[:, 1:], token.unsqueeze(1)], dim=1)
        self._len = (self._len + 1).clamp(max=self.history_len)
        return self._hist

    def delta(self, token: torch.Tensor) -> torch.Tensor:
        if self.encoder_name != "gru":
            return self.body(token)
        # Rollout (no grad): causal ring. PPO update (grad): do not mutate the
        # ring with shuffled minibatches — one-step GRU keeps the buffer intact.
        if token.requires_grad or torch.is_grad_enabled():
            y, _h = self.body.gru(token.unsqueeze(1))
            return self.body.head(y[:, -1])
        hist = self.push(token)
        return self.body(hist)

    def apply(self, z_nom: torch.Tensor, token: torch.Tensor) -> dict[str, torch.Tensor]:
        dz = self.delta(token)
        z = F.normalize(z_nom, dim=-1, eps=1e-8)
        if self.tangent:
            z_exec, dz_bar = apply_tangent_correction(
                z, dz, r_max=self.r_max, alpha=z.new_full((z.shape[0],), self.beta)
            )
        else:
            z_exec = F.normalize(z + self.beta * dz, dim=-1, eps=1e-8)
            dz_bar = dz
        return {
            "z_exec": z_exec,
            "dz": dz,
            "dz_bar": dz_bar,
            "dz_norm": dz.norm(dim=-1),
            "dz_bar_norm": dz_bar.norm(dim=-1),
        }
