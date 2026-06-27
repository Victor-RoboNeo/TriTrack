# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .distillation import Distillation
from .adv_pulse_distillation import AdvPulseDistillation
from .muse_cotrain_distillation import MuseCoTrainDistillation
from .muse_distillation import MuseDistillation
from .muse_kp_distillation import MuseKpDistillation
from .muse_kp_latent_distillation import MuseKpLatentDistillation
from .muse_human_kp_distillation import MuseHumanKpDistillation
from .pulse_distillation import PulseDistillation
from .anybody_latent_distillation import AnyBodyLatentDistillation
from .ppo import PPO
from .latent_ppo import LatentPPO

__all__ = [
    "PPO",
    "LatentPPO",
    "Distillation",
    "AdvPulseDistillation",
    "MuseCoTrainDistillation",
    "MuseDistillation",
    "MuseKpDistillation",
    "MuseKpLatentDistillation",
    "MuseHumanKpDistillation",
    "PulseDistillation",
    "AnyBodyLatentDistillation",
]
