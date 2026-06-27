# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .normalizer import EmpiricalNormalization
from .rnd import RandomNetworkDistillation
from .student_teacher import StudentTeacher
from .latent_bottleneck_anybody import LatentBottleneckAnyBody
from .latent_bottleneck_anybody_actor_critic import LatentBottleneckAnyBodyActorCritic
from .latent_bottleneck_muse import LatentBottleneckMUSE
from .latent_bottleneck_muse_cotrain import LatentBottleneckMUSECoTrain
from .latent_bottleneck_muse_transformer import LatentBottleneckMUSETransformer
from .latent_bottleneck_muse_kp import LatentBottleneckMUSEKp
from .latent_bottleneck_muse_kp_latent import LatentBottleneckMUSEKpLatent
from .latent_bottleneck_muse_human_kp import LatentBottleneckMUSEHumanKp
from .latent_bottleneck_pulse import LatentBottleneckPULSE
from .latent_bottleneck_pulse_adv import LatentBottleneckPULSEAdv
from .pulse_action_discriminator import PulseActionDiscriminator
from .latent_bottleneck_2b import LatentBottleneck2B
from .residual_actor_critic import ResidualActorCritic
from .latent_rl_actor_critic import LatentRLActorCritic
from .velocity_estimator import VelocityEstimator
from .velocity_estimator_transformer import VelocityEstimatorTransformer

__all__ = [
    "ActorCritic",
    "EmpiricalNormalization",
    "RandomNetworkDistillation",
    "StudentTeacher",
    "LatentBottleneckAnyBody",
    "LatentBottleneckAnyBodyActorCritic",
    "LatentBottleneckMUSE",
    "LatentBottleneckMUSECoTrain",
    "LatentBottleneckMUSETransformer",
    "LatentBottleneckMUSEKp",
    "LatentBottleneckMUSEKpLatent",
    "LatentBottleneckMUSEHumanKp",
    "LatentBottleneckPULSE",
    "LatentBottleneckPULSEAdv",
    "PulseActionDiscriminator",
    "LatentBottleneck2B",
    "ResidualActorCritic",
    "LatentRLActorCritic",
    "VelocityEstimator",
    "VelocityEstimatorTransformer",
]
