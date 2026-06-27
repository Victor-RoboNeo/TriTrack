from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


def parse_hidden_dims(spec: str) -> List[int]:
    """Parse a comma-separated hidden layer spec into a list of ints."""
    return [int(x) for x in spec.split(",") if x.strip()]


@dataclass
class LatentBottleneckConfig:
    human_dim: int
    proprio_dim: int
    num_actions: int
    latent_dim: int = 64
    encoder_hidden_dims: List[int] | None = None
    decoder_hidden_dims: List[int] | None = None
    activation: str = "elu"


class LatentBottleneckPolicy(nn.Module):
    """Encoder-decoder policy with a latent bottleneck (VAE-style encoder).

    - Encoder: (masked reference motion, proprioception) → μ, log σ.
    - Sampling: z ~ N(μ, σ²) via reparameterization during training.
    - Decoder: (z, proprioception) → action.
    """

    def __init__(
        self,
        human_dim: int,
        proprio_dim: int,
        num_actions: int,
        latent_dim: int,
        encoder_hidden: List[int],
        decoder_hidden: List[int],
        activation: str = "elu",
    ):
        super().__init__()
        self.human_dim = human_dim
        self.proprio_dim = proprio_dim
        self.latent_dim = latent_dim
        self.num_actions = num_actions

        act_cls = getattr(nn, activation.upper(), nn.ELU)

        # Encoder: [human, mask, proprio], dim: (human_dim*2 + proprio_dim)
        # We always allocate for mask so that we can pass mask=None and use zeros for "all visible"
        enc_in = human_dim * 2 + proprio_dim
        enc_layers: list[nn.Module] = []
        for h in encoder_hidden:
            enc_layers.append(nn.Linear(enc_in, h))
            enc_layers.append(act_cls())
            enc_in = h
        self.encoder_body = nn.Sequential(*enc_layers)
        self.encoder_mu = nn.Linear(enc_in, latent_dim)
        self.encoder_log_sigma = nn.Linear(enc_in, latent_dim)

        # Decoder: [z, proprio] -> action
        dec_in = latent_dim + proprio_dim
        dec_layers: list[nn.Module] = []
        for h in decoder_hidden:
            dec_layers.append(nn.Linear(dec_in, h))
            dec_layers.append(act_cls())
            dec_in = h
        dec_layers.append(nn.Linear(dec_in, num_actions))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(
        self,
        human: torch.Tensor,
        proprio: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode (human, proprio) into latent distribution (mu, log_sigma).

        If mask is provided, masked-out keypoints are zeroed and mask is concatenated
        so the encoder sees which dims are masked (not truly zero).
        Mask can be per-keypoint (last dim 42) or same as human (e.g. 210 for 5 steps).
        If human is multi-step (e.g. 210 = 5*42) and mask is per-keypoint (42), we expand
        mask by repeating along the last dim so one visibility applies to all steps.
        """
        if mask is not None:
            if mask.shape[-1] != human.shape[-1]:
                # Multi-step obs: human (..., 42*num_steps), mask (..., 42). Expand mask.
                n_steps = human.shape[-1] // mask.shape[-1]
                if human.shape[-1] % mask.shape[-1] != 0:
                    raise ValueError(
                        f"human last dim {human.shape[-1]} must be a multiple of mask last dim {mask.shape[-1]}"
                    )
                mask = mask.repeat_interleave(n_steps, dim=-1)
            human_masked = human * mask
            x = torch.cat([human_masked, mask, proprio], dim=-1)
        else:
            # All visible: same as human with mask of ones
            mask_ones = torch.ones_like(human, device=human.device, dtype=human.dtype)
            x = torch.cat([human, mask_ones, proprio], dim=-1)
        h = self.encoder_body(x)
        mu = self.encoder_mu(h)
        log_sigma = self.encoder_log_sigma(h).clamp(max=2.0)
        return mu, log_sigma

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu, device=mu.device)
        return mu + sigma * eps

    def decode(self, z: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, proprio], dim=-1)
        return self.decoder(x)

    def forward(
        self,
        human: torch.Tensor,
        proprio: torch.Tensor,
        sample_z: bool = True,
        mask: torch.Tensor | None = None,
    ) -> tuple:
        mu, log_sigma = self.encode(human, proprio, mask=mask)
        if sample_z and self.training:
            z = self.reparameterize(mu, log_sigma)
        else:
            z = mu
        action = self.decode(z, proprio)
        return action, mu, log_sigma, z

