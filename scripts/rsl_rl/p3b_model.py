#!/usr/bin/env python3
"""P3-B intent-metric MLP. Instantaneous state only. No history/terrain."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

Z_DIM = 16
EPS_C = 1e-8
N_TRIL = Z_DIM * (Z_DIM + 1) // 2  # 136


def tril_indices(device=None):
    i, j = torch.tril_indices(Z_DIM, Z_DIM, device=device)
    return i, j


class IntentMetricMLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(512, 512, 256), out_dim: int = N_TRIL):
        super().__init__()
        self.in_ln = nn.LayerNorm(in_dim)
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())
            d = h
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(d, out_dim)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.backbone(self.in_ln(x)))


def tangent_P_batch(z: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z, dim=-1, eps=1e-8)
    eye = torch.eye(Z_DIM, device=z.device, dtype=z.dtype).unsqueeze(0)
    return eye - z.unsqueeze(-1) * z.unsqueeze(-2)


def trace_norm(C: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    Pt = tangent_P_batch(z)
    C = 0.5 * (C + C.transpose(-1, -2))
    C = torch.bmm(Pt, torch.bmm(C, Pt))
    tr = C.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=0.0)
    return C / (tr / 15.0 + EPS_C).view(-1, 1, 1)


def chol_to_C(lvec: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    n = lvec.shape[0]
    L = lvec.new_zeros(n, Z_DIM, Z_DIM)
    i, j = tril_indices(device=lvec.device)
    L[:, i, j] = lvec
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    L = L - torch.diag_embed(diag) + torch.diag_embed(F.softplus(diag) + 1e-4)
    C = torch.bmm(L, L.transpose(-1, -2))
    return trace_norm(C, z)


def lowrank_to_C(uvec: torch.Tensor, z: torch.Tensor, rank: int) -> torch.Tensor:
    n = uvec.shape[0]
    U = uvec.view(n, Z_DIM, rank)
    C = torch.bmm(U, U.transpose(-1, -2))
    C = C + 1e-4 * torch.eye(Z_DIM, device=uvec.device, dtype=uvec.dtype).unsqueeze(0)
    return trace_norm(C, z)


def soft_P(C: torch.Tensor, z: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    eye = torch.eye(Z_DIM, device=C.device, dtype=C.dtype).unsqueeze(0)
    P = torch.linalg.inv(eye + float(lam) * C)
    Pt = tangent_P_batch(z)
    return torch.bmm(Pt, torch.bmm(P, Pt))


def random_tangent(z: torch.Tensor, n_dir: int, generator=None) -> torch.Tensor:
    n = z.shape[0]
    v = torch.randn(n, n_dir, Z_DIM, device=z.device, dtype=z.dtype, generator=generator)
    z = F.normalize(z, dim=-1, eps=1e-8)
    v = v - (v * z.unsqueeze(1)).sum(-1, keepdim=True) * z.unsqueeze(1)
    return F.normalize(v, dim=-1, eps=1e-8)


def sample_ucr_dirs(z: torch.Tensor, pool: torch.Tensor, n_dir: int, generator=None, noise_std: float = 0.0):
    n = z.shape[0]
    if pool is None or pool.numel() == 0:
        return random_tangent(z, n_dir, generator)
    idx = torch.randint(0, pool.shape[0], (n, n_dir), device=z.device, generator=generator)
    v = pool[idx.reshape(-1)].view(n, n_dir, Z_DIM)
    if noise_std and float(noise_std) > 0:
        xi = torch.randn(n, n_dir, Z_DIM, device=z.device, dtype=z.dtype, generator=generator)
        v = v + float(noise_std) * xi
    z = F.normalize(z, dim=-1, eps=1e-8)
    v = v - (v * z.unsqueeze(1)).sum(-1, keepdim=True) * z.unsqueeze(1)
    return F.normalize(v, dim=-1, eps=1e-8)


def losses(C_hat, C_gt, z, v_rand, v_ucr, lam=1.0):
    lf = ((C_hat - C_gt) ** 2).sum(dim=(-1, -2)).mean()
    v = torch.cat([v_rand, v_ucr], dim=1)
    s_h = torch.einsum("bdi,bij,bdj->bd", v, C_hat, v)
    s_g = torch.einsum("bdi,bij,bdj->bd", v, C_gt, v)
    lq = ((s_h - s_g) ** 2).mean()
    Ph = soft_P(C_hat, z, lam)
    Pg = soft_P(C_gt, z, lam)
    ph = torch.einsum("bij,bdj->bdi", Ph, v)
    pg = torch.einsum("bij,bdj->bdi", Pg, v)
    lp = ((ph - pg) ** 2).sum(-1).mean()
    return lf, lq, lp, 0.5 * lf + 1.0 * lq + 2.0 * lp


def conservative_losses(C_hat, C_gt, z, v, under_w=3.0, kappa=1.25, lam=1.0):
    """B2-1: asymmetric sensitivity + one-sided oracle leak. λ_pred=λ_oracle=1."""
    lf = ((C_hat - C_gt) ** 2).sum(dim=(-1, -2)).mean()
    s_h = torch.einsum("bdi,bij,bdj->bd", v, C_hat, v)
    s_g = torch.einsum("bdi,bij,bdj->bd", v, C_gt, v)
    w = torch.where(s_h < s_g, torch.full_like(s_h, float(under_w)), torch.ones_like(s_h))
    lq = (w * (s_h - s_g) ** 2).mean()
    Ph = soft_P(C_hat, z, lam)
    Pg = soft_P(C_gt, z, 1.0)
    ph = torch.einsum("bij,bdj->bdi", Ph, v)
    pg = torch.einsum("bij,bdj->bdi", Pg, v)
    lp = ((ph - pg) ** 2).sum(-1).mean()
    d_pred = F.normalize(ph, dim=-1, eps=1e-8)
    d_or = F.normalize(pg, dim=-1, eps=1e-8)
    s_pred_gt = torch.einsum("bdi,bij,bdj->bd", d_pred, C_gt, d_pred)
    s_or_gt = torch.einsum("bdi,bij,bdj->bd", d_or, C_gt, d_or)
    lleak = (F.relu(s_pred_gt - float(kappa) * s_or_gt) ** 2).mean()
    total = 0.25 * lf + 1.0 * lq + 2.0 * lp + 3.0 * lleak
    return lf, lq, lp, lleak, total
