# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""P2-D gated terrain residual ``h_η(H_t) → u ∈ R^{16}``.

Nominal controller ``z_nom = Norm(z_base + g_{φ,50000})`` stays frozen.
``h_η`` sees **only** the height scan (no ``z_nom`` conditioning). Last Linear
is zero-initialized so ``u=0`` at step 0 and ``z_exec = z_nom``.

Analytic severity gate ``α(H)`` with ``α(0)=0``:

    s_slope = √(a²+b²)     local-plane gradient magnitude
    s_rough = P_95(|H − Ĥ|)
    s      = w_s s_slope + w_r s_rough
    α      = G(s),   G(0)=0,  dead-zone + shifted sigmoid

Tangent cap + evidence-dependent authority:

    u_perp = u − (uᵀ z_nom) z_nom
    ū      = Cap(u_perp, r_max = tan 10°)
    z_exec = Norm(z_nom + α(H) ū)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# tan(10°) — P2-D max latent angular correction.
DEFAULT_R_MAX = math.tan(math.radians(10.0))


def _grid_xy(nx: int, ny: int, size_xy: tuple[float, float], device=None, dtype=None) -> torch.Tensor:
    """Isaac ``GridPatternCfg(ordering='xy')`` coordinates, flattened ``[N, 2]``."""
    sx, sy = float(size_xy[0]), float(size_xy[1])
    res_x = sx / max(int(nx) - 1, 1)
    res_y = sy / max(int(ny) - 1, 1)
    x = torch.arange(int(nx), device=device, dtype=dtype) * res_x - 0.5 * sx
    y = torch.arange(int(ny), device=device, dtype=dtype) * res_y - 0.5 * sy
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)


class TerrainSeverityGate(nn.Module):
    """Analytic ``α(H)∈[0,1]`` from a local plane fit. No learned parameters.

    Physical heights are ``h = scan * clip_abs`` (obs scan is already /clip_abs).
    ``α(0)=0`` by shifted-sigmoid construction; a dead zone zeros tiny evidence.
    """

    def __init__(
        self,
        scan_dim: int = 187,
        nx: int = 17,
        ny: int = 11,
        size_xy: tuple[float, float] = (1.6, 1.0),
        clip_abs: float = 0.5,
        w_s: float = 1.0,
        w_r: float = 2.5,
        s0: float = 0.06,
        tau: float = 0.025,
        s_dead: float = 0.02,
    ):
        super().__init__()
        self.scan_dim = int(scan_dim)
        self.nx = int(nx)
        self.ny = int(ny)
        self.clip_abs = float(clip_abs)
        self.w_s = float(w_s)
        self.w_r = float(w_r)
        self.s0 = float(s0)
        self.tau = float(max(tau, 1e-8))
        self.s_dead = float(s_dead)
        self.sig0 = float(1.0 / (1.0 + math.exp(self.s0 / self.tau)))
        xy = _grid_xy(self.nx, self.ny, size_xy, dtype=torch.float32)
        if int(xy.shape[0]) != self.scan_dim:
            # Fallback: still fit a plane if the flattening length mismatches.
            n = self.scan_dim
            t = torch.linspace(-0.5, 0.5, n)
            xy = torch.stack([t * size_xy[0], torch.zeros(n)], dim=-1)
        ones = torch.ones(xy.shape[0], 1)
        a_mat = torch.cat([xy, ones], dim=-1)  # [N, 3]  →  z ≈ ax+by+c
        pinv = torch.linalg.pinv(a_mat)
        self.register_buffer("design", a_mat, persistent=False)
        self.register_buffer("pinv", pinv, persistent=False)

    def forward(self, scan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(α, s, s_slope, s_rough)``, each ``[B]``."""
        h = scan[..., : self.design.shape[0]] * self.clip_abs
        coef = h @ self.pinv.transpose(0, 1)  # [B, 3]
        a, b = coef[..., 0], coef[..., 1]
        s_slope = torch.sqrt(a * a + b * b)
        hat = coef @ self.design.transpose(0, 1)
        abs_r = (h - hat).abs()
        k = max(int(round(0.95 * (abs_r.shape[-1] - 1))), 0)
        s_rough = abs_r.kthvalue(k + 1, dim=-1).values
        s = self.w_s * s_slope + self.w_r * s_rough
        sig = torch.sigmoid((s - self.s0) / self.tau)
        denom = max(1.0 - self.sig0, 1e-8)
        alpha = ((sig - self.sig0) / denom).clamp(min=0.0)
        alpha = torch.where(s < self.s_dead, torch.zeros_like(alpha), alpha)
        return alpha, s, s_slope, s_rough


class TerrainResidualMLP(nn.Module):
    """``h_η(H)``: 187 → 128 → 64 → 64 → 16. Last layer zeros. No ``z_nom``."""

    def __init__(
        self,
        scan_dim: int = 187,
        latent_dim: int = 16,
        scan_hidden: tuple[int, int] = (128, 64),
        head_hidden: int = 64,
    ):
        super().__init__()
        h1, h2 = int(scan_hidden[0]), int(scan_hidden[1])
        self.scan_dim = int(scan_dim)
        self.latent_dim = int(latent_dim)
        self.scan_mlp = nn.Sequential(
            nn.Linear(self.scan_dim, h1),
            nn.ELU(),
            nn.Linear(h1, h2),
            nn.ELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(h2, int(head_hidden)),
            nn.ELU(),
            nn.Linear(int(head_hidden), self.latent_dim),
        )
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, height_scan: torch.Tensor, z_nom: torch.Tensor | None = None) -> torch.Tensor:
        del z_nom  # P2-D: scan-only. Signature kept so older call sites still type-check.
        return self.head(self.scan_mlp(height_scan))


def apply_tangent_correction(
    z_nom: torch.Tensor,
    u: torch.Tensor,
    r_max: float = DEFAULT_R_MAX,
    alpha: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project ``u`` onto the tangent of unit ``z_nom``, cap, scale by ``α``, re-normalize.

    Returns ``(z_exec, u_bar)`` where ``u_bar`` is the *capped* tangent residual
    **before** the gate (so ``Δz = α u_bar``).
    """
    z = F.normalize(z_nom, dim=-1, eps=1e-8)
    u_par = (u * z).sum(dim=-1, keepdim=True) * z
    u_perp = u - u_par
    n = u_perp.norm(dim=-1, keepdim=True)
    cap = float(r_max)
    scale = torch.ones_like(n)
    if cap > 0.0:
        scale = torch.clamp(n, max=cap) / n.clamp_min(1e-8)
        scale = torch.where(n > 0, scale, torch.ones_like(scale))
    u_bar = u_perp * scale
    if torch.is_tensor(alpha):
        a = alpha.to(dtype=z.dtype, device=z.device).reshape(-1, 1)
    else:
        a = z.new_full((z.shape[0], 1), float(alpha))
    z_exec = F.normalize(z + a * u_bar, dim=-1, eps=1e-8)
    return z_exec, u_bar
