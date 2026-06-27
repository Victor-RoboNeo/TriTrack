"""Binary discriminator on concat(action, proprio) for diagnostic encoder vs prior action paths."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class PulseActionDiscriminator(nn.Module):
    """MLP: concat(a, s^p) -> scalar **logit** (label 0 = encoder path, 1 = prior path).

    Use with ``binary_cross_entropy_with_logits``; class probabilities are ``sigmoid(logit)`` for "prior".
    Do not add a final sigmoid in the network.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: str = "elu",
    ):
        super().__init__()
        act_fn = resolve_nn_activation(activation)
        layers: list[nn.Module] = []
        d = int(in_dim)
        hlist = list(hidden_dims)
        if not hlist:
            hlist = [256, 128]
        for h in hlist:
            layers.append(nn.Linear(d, int(h)))
            layers.append(act_fn)
            d = int(h)
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, actions: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Return unbounded logits of shape ``[batch]`` (not probabilities)."""
        x = torch.cat([actions, proprio], dim=-1)
        return self.net(x).squeeze(-1)
