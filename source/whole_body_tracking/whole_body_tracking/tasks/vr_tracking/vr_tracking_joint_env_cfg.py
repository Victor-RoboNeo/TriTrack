"""VR tracking MDP in joint action space (PPO on decoded joints; structure matches latent VR obs/rewards)."""

from __future__ import annotations

from isaaclab.envs.mdp import action_rate_l2 as isaac_action_rate_l2
from isaaclab.envs.mdp import last_action as isaac_last_action
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.vr_tracking.mdp as mdp
from whole_body_tracking.tasks.vr_tracking.vr_tracking_env_cfg import RewardsCfg, VRTrackingEnvCfg


@configclass
class VRTrackingJointSpaceRewardsCfg(RewardsCfg):
    """Match latent VR rewards except action smoothness: use Isaac Lab rate on raw joint actions."""

    action_rate_l2 = RewTerm(func=isaac_action_rate_l2, weight=-1e-1)


@configclass
class VRTrackingJointSpaceObservationsCfg:
    """Same as latent VR observations, but ``last_action`` uses Isaac Lab's term (raw joint actions)."""

    @configclass
    class PolicyCfg(ObsGroup):
        body_pos = ObsTerm(func=mdp.ref_body_pos_robot_anchor_b, params={"command_name": "motion"})
        body_lin_vel = ObsTerm(func=mdp.ref_body_lin_vel_robot_anchor_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        ref_base_lin_vel_b = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        actions = ObsTerm(func=isaac_last_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class CriticCfg(ObsGroup):
        body_pos = ObsTerm(func=mdp.ref_body_pos_robot_anchor_b, params={"command_name": "motion"})
        ref_body_lin_vel = ObsTerm(func=mdp.ref_body_lin_vel_robot_anchor_b, params={"command_name": "motion"})
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        ref_base_lin_vel_b = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        actions = ObsTerm(func=isaac_last_action, params={"action_name": "joint_pos"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class VRTrackingJointSpaceEnvCfg(VRTrackingEnvCfg):
    """Same task rewards as latent VR; joint actions; Isaac ``action_rate_l2`` / ``last_action`` on raw joints."""

    observations: VRTrackingJointSpaceObservationsCfg = VRTrackingJointSpaceObservationsCfg()
    rewards: VRTrackingJointSpaceRewardsCfg = VRTrackingJointSpaceRewardsCfg()

    @configclass
    class ActionsCfg:
        joint_pos = mdp.VRJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            use_default_offset=True,
            proprio_history_length=5,
        )

    actions: ActionsCfg = ActionsCfg()
