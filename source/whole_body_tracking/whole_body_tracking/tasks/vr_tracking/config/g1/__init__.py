import gymnasium as gym

from . import agents, flat_env_cfg, joint_flat_env_cfg


gym.register(
    id="VR-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatVRTrackingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatVRTrackingPPORunnerCfg",
    },
)

gym.register(
    id="VR-Tracking-Joint-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_flat_env_cfg.G1FlatVRTrackingJointEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_joint_ppo_cfg:G1FlatVRTrackingJointPPORunnerCfg",
    },
)

