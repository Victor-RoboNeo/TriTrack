"""Tiny residual MLP: u_G1 = u_human + f(u_human). ~19k params."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 9),
        )

    def forward(self, u_h: torch.Tensor) -> torch.Tensor:
        return u_h + self.net(u_h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
