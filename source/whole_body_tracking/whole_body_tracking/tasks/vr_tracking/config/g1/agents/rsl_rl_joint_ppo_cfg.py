from __future__ import annotations

from typing import Optional

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlLatentBottleneckAnyBodyJointActorCriticCfg(RslRlPpoActorCriticCfg):
    """Policy kwargs for :class:`rsl_rl.modules.latent_bottleneck_anybody_actor_critic.LatentBottleneckAnyBodyActorCritic`."""

    class_name: str = "LatentBottleneckAnyBodyActorCritic"
    # Overwritten in train.py from ``VRJointPositionAction._proprio_dim`` when possible.
    proprio_dim: int = 450
    latent_dim: int = 16
    encoder_hidden_dims: list[int] = [512, 256, 128]
    decoder_hidden_dims: list[int] = [512, 256, 128]
    prior_hidden_dims: list[int] = [512, 256, 128]
    latent_predict_std_min: float = 0.001
    latent_predict_std_max: float = 1.0
    fixed_prior_std: Optional[float] = 0.5
    residual_scale: float = 1.0
    train_prior: bool = True
    train_decoder: bool = True


@configclass
class G1FlatVRTrackingJointPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO on joints with PULSE prior+decoder and latent VR warm-started residual MLP + critic."""

    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 1000
    experiment_name = "g1_flat_vr_tracking_joint_pulse"
    empirical_normalization = True
    enable_rl_split_obs_normalizer = True
    seed = 42

    policy = RslRlLatentBottleneckAnyBodyJointActorCriticCfg(
        init_noise_std=0.05,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[1024, 1024, 512, 512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
