from __future__ import annotations

from dataclasses import MISSING
from typing import Union

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from rsl_rl.modules.latent_bottleneck_muse_kp import KP_LAYOUT_0_5S, KP_LAYOUT_1_0S
from whole_body_tracking.tasks.tracking.config.g1 import mask_modes

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

# Default partial-mask mode probabilities (11 modes); align with ``mask_cfg.mode_probs`` in the runner.
# Ordering matches ``PARTIAL_MASKED_2B_G1_MODE_NAMES`` (incl. trailing ``bernoulli``).
_G1_PARTIAL_MASK_MODE_PROBS = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

from isaaclab.terrains import TerrainGeneratorCfg, MeshPlaneTerrainCfg, HfRandomUniformTerrainCfg
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #         project_uvw=True,
    #     ),
    # )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator", 
        terrain_generator=TerrainGeneratorCfg(
            seed=42,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=10,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            sub_terrains={
                "flat": MeshPlaneTerrainCfg(proportion=0.5),
                "slightly_rough": HfRandomUniformTerrainCfg(
                    proportion=0.5,
                    noise_range=(0.01, 0.03),
                    noise_step=0.01,
                ),
            },
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


##
# MDP settings
##


@configclass
class SingleMotionCommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )

@configclass
class MultiMotionCommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MultiMotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        resample_motions_every_s =50, #TODO: finetune this 
        motion_sampling_warmup_s=1.0e9, #1000,
        motion_sampling_ramp_s=2000,
        motion_sampling_schedule="cosine",
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class PartialMaskedMultiMotionCommandsCfg:
    """Command specs with :class:`PartialMaskedMultiMotionCommand` for task-side keypoint masking."""

    motion = mdp.PartialMaskedMultiMotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        resample_motions_every_s=1.0e9,
        motion_sampling_warmup_s=1.0e9,
        motion_sampling_ramp_s=2000,
        motion_sampling_schedule="cosine",
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        mask_mode_spec=mask_modes.partial_masked_2b_g1_mode_spec(),
        mask_mode_probs=_G1_PARTIAL_MASK_MODE_PROBS,
        duplicate_for_ref_body_lin_vel=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        # def __post_init__(self):
        #     self.history_length = 5

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

@configclass
class ObservationsExpertCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        
        
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # self.history_length = 5

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        # def __post_init__(self):
        #     self.history_length = 5

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )

@configclass
class RewardsExpertCfg:
    """
    Expert reward configuration - MOSAIC.
    """

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 3.14},
    )
    motion_anchor_lin_vel = RewTerm(
        func=mdp.motion_anchor_linear_velocity_error_exp,
        weight=1.0,  # 2*1.0
        params={"command_name": "motion", "std": 1.0},
    )

    teleop_body_position_extend = RewTerm(
        func=mdp.teleop_body_position_extend,
        weight=1.0,
        params={
            "command_name": "motion",
            "upper_body_std": 0.5, 
            "lower_body_std": 0.5,  
            "upper_weight": 1.0,
            "lower_weight": 1.0,
        }
    )
    teleop_vr_3point = RewTerm(
        func=mdp.teleop_vr_3point,
        weight=0.5,
        params={"command_name": "motion", "std": 0.5}  
    )
    teleop_body_position_feet = RewTerm(
        func=mdp.teleop_body_position_feet,
        weight=1,  # 1.5*1
        params={"command_name": "motion", "std": 0.5} 
    )
    teleop_body_rotation_extend = RewTerm(
        func=mdp.teleop_body_rotation_extend,
        weight=0.5,
        params={"command_name": "motion", "std": 0.5} 
    )
    teleop_body_ang_velocity_extend = RewTerm(
        func=mdp.teleop_body_ang_velocity_extend,
        weight=0.5,
        params={"command_name": "motion", "std": 3.14}
    )
    teleop_body_velocity_extend = RewTerm(
        func=mdp.teleop_body_velocity_extend,
        weight=0.5,
        params={"command_name": "motion", "std": 1.0} 
    )

    # ===== Penalty terms (same as base) =====
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)  # 2*1e-1
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # -2*2.5e-7
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)  # -2*1e-5
    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    motion_end = DoneTerm(func=mdp.motion_end, params={"command_name": "motion"}, time_out=True)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class PartialMaskedVaeDistillationCurriculumCfg(CurriculumCfg):
    """Curriculum that updates partial keypoint mask sampling probabilities (command-manager mode)."""

    keypoint_mask_mode = CurrTerm(
        func=mdp.curriculums.keypoint_mask_mode_curriculum,
        params={
            "command_name": "motion",
            # Bounds are global PPO iteration indices; runner maps them to env steps (see OnPolicyRunner.learn).
            "phase_until_learning_iterations": (None,),
            # Per-phase schedule dict: includes both mode probs and Bernoulli p_start/p_end.
            "mask_phases": (
                {"mode_probs": _G1_PARTIAL_MASK_MODE_PROBS, "p_start": 1.0, "p_end": 0.5},
            ),
        },
    )


##
# Environment configuration
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: SingleMotionCommandsCfg = SingleMotionCommandsCfg()
    
    # MDP settings
    rewards: RewardsExpertCfg = RewardsExpertCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 15 * 2**17
        # viewer settings
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.viewer.origin_type = "env"
        self.viewer.asset_name = "robot"
    
@configclass
class GeneralTrackingEnvCfg(TrackingEnvCfg):
    """Configuration for the general tracking environment."""

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()
        
@configclass
class ExpertGeneralTrackingEnvCfg(GeneralTrackingEnvCfg):
    """
    Expert general tracking environment.
    """

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()
    observations: ObservationsExpertCfg = ObservationsExpertCfg()
    rewards: RewardsExpertCfg = RewardsExpertCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()


# 1. teacher model: 
@configclass
class OneStageTrackingEnvCfg(GeneralTrackingEnvCfg):
    """
    Teacher-student distillation environment configuration.
    """

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    @configclass
    class OneStageObservationsCfg:
        """Observation specifications for distillation."""

        @configclass
        class PolicyCfg(ObsGroup):
            """Observations for policy group."""
            # 1. obs-goal:
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"}) # J*2 (ref_joint_pos, ref_joint_vel)
            motion_anchor_ori_b = ObsTerm(
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05) # 6
            )

            # 2. obs-proprio: joint_pos, joint_vel, base_ang_vel, last_action
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)) # J
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)) # J
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) # 3
            actions = ObsTerm(func=mdp.last_action) # J

            # 3. privileged-goal (added 2026-05-13): goal anchor position in robot-anchor frame.
            # Appended at the END so existing checkpoints can be warm-started by zero-padding the
            # actor's first-layer input columns (history_length=5 means this adds 3*5=15 cols at
            # the tail of the flattened obs; columns [0:770] keep their semantic meaning).
            motion_anchor_pos_b = ObsTerm(
                func=mdp.motion_anchor_pos_b, params={"command_name": "motion"},
                noise=Unoise(n_min=-0.01, n_max=0.01),  # 3
            )

            # 4. privileged-velocity (added 2026-05-13 round 2): own base linear velocity + the
            # reference base linear velocity from the motion clip. Both 3-dim, both appended at
            # the END for clean zero-pad warmstart (history=5 → +30 columns, 785 → 815).
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))  # 3
            ref_base_lin_vel = ObsTerm(
                func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"},
                noise=Unoise(n_min=-0.01, n_max=0.01),  # 3
            )

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 5

        @configclass
        class PrivilegedCfg(ObsGroup):
            # 1. obs-goal: 
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
            ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})

            # 2. obs-proprio: 
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
            body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            actions = ObsTerm(func=mdp.last_action)

        # observation groups
        policy: PolicyCfg = PolicyCfg()
        critic: PrivilegedCfg = PrivilegedCfg()

    observations: OneStageObservationsCfg = OneStageObservationsCfg()
    rewards: RewardsExpertCfg = RewardsExpertCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()
        #self.commands.motion.enable_dense_joint_error_metrics = True
        # Anchor-pos reward bump (2026-05-13): the teacher's policy obs gained motion_anchor_pos_b
        # so it can now actually act on goal-anchor information. Lift the anchor-pos weight 0.5→2.0
        # to push world-frame anchor tracking; relative body terms (weight 1.0) still dominate per
        # body but anchor is now the heaviest single term — by design.
        self.rewards.motion_global_anchor_pos.weight = 2.0
        # Anchor-lin-vel reward bump (2026-05-13 round 2): now that the policy sees its own
        # base_lin_vel + ref_base_lin_vel it can act on world-frame anchor velocity. Bump 1.0→3.0
        # so anchor velocity matches the proportional emphasis on anchor pos (2× the body terms).
        self.rewards.motion_anchor_lin_vel.weight = 3.0
        self.rewards.motion_global_anchor_ori.weight = 1.0


# 2. student model: 
@configclass
class DistillationTrackingEnvCfg(GeneralTrackingEnvCfg):
    """
    Student-teacher distillation environment configuration.
    """

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    @configclass
    class DistillationObservationsCfg:
        """Observation specifications for distillation."""

        @configclass
        class PolicyCfg(ObsGroup):
            """Observations for policy group."""
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
            )
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 5

        @configclass
        class TeacherCfg(ObsGroup):
            """Teacher observations - teacehr information."""
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
            body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
            body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        @configclass
        class PrivilegedCfg(ObsGroup):
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
            body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
            body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            actions = ObsTerm(func=mdp.last_action)

            # def __post_init__(self):
            #     self.history_length = 5

        @configclass
        class RefVelEstimatorCfg(ObsGroup):
            """Observations for reference velocity estimator."""
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})  # [58] = [joint_pos(29), joint_vel(29)]
            ref_projected_gravity = ObsTerm(func=mdp.ref_projected_gravity, params={"command_name": "motion"})  # [3] using ANCHOR BODY quaternion

            def __post_init__(self):
                self.enable_corruption = False 
                self.concatenate_terms = True
                self.history_length = 5

        # observation groups
        policy: PolicyCfg = PolicyCfg()
        teacher: TeacherCfg = TeacherCfg()
        critic: PrivilegedCfg = PrivilegedCfg()
        ref_vel_estimator: RefVelEstimatorCfg = RefVelEstimatorCfg() 

    observations: DistillationObservationsCfg = DistillationObservationsCfg()
    rewards: RewardsExpertCfg = RewardsExpertCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()

@configclass
class PULSEDistillationTrackingEnvCfg(GeneralTrackingEnvCfg):
    """
    PULSE-style teacher-student distillation environment.

    Both student (policy) and teacher receive the same observation structure
    (imitation goal + proprioception). The only difference is that the
    student's proprioceptive channels are corrupted with noise, while the
    teacher sees clean inputs—matching the PULSE setting illustrated in the
    diagram.
    """

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    @configclass
    class PULSEObservationsCfg:
        """Observation specifications for PULSE distillation."""

        @configclass
        class PolicyCfg(ObsGroup):
            # 1. obs-goal: 
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"}) # J*2 (ref_joint_pos, ref_joint_vel)
            motion_anchor_ori_b = ObsTerm(
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05) # 6
            )

            # 2. obs-proprio: joint_pos, joint_vel, base_ang_vel, last_action
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)) # J
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)) # J
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) # 3
            actions = ObsTerm(func=mdp.last_action) # J

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 5

        @configclass
        class TeacherCfg(ObsGroup):
            # 1. obs-goal: 
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"}) # J*2 (ref_joint_pos, ref_joint_vel)
            motion_anchor_ori_b = ObsTerm(
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"} # 6
            )

            # 2. obs-proprio: joint_pos, joint_vel, base_ang_vel, last_action
            joint_pos = ObsTerm(func=mdp.joint_pos_rel) # J
            joint_vel = ObsTerm(func=mdp.joint_vel_rel) # J
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel) # 3
            actions = ObsTerm(func=mdp.last_action) # J

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True
                self.history_length = 5

        # Observation groups
        policy: PolicyCfg = PolicyCfg()
        teacher: TeacherCfg = TeacherCfg()

    observations: PULSEObservationsCfg = PULSEObservationsCfg()
    rewards: RewardsExpertCfg = RewardsExpertCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()
        #self.commands.motion.enable_dense_joint_error_metrics = True


@configclass
class MUSEDistillationCurriculumCfg(CurriculumCfg):
    """Curriculum that ramps the MUSE goal-block mask probability ``p_mask`` over training."""

    goal_mask_p = CurrTerm(
        func=mdp.curriculums.goal_mask_probability_curriculum,
        params={
            "command_name": "motion",
            "p_start": 0.0,
            "p_end": 0.8,
            "ramp_start_iter": 2000,
            "ramp_end_iter": 8000,
            "schedule": "linear",
        },
    )


@configclass
class MUSEDistillationTrackingEnvCfg(GeneralTrackingEnvCfg):
    """MUSE distillation environment.

    Mirrors :class:`PULSEDistillationTrackingEnvCfg` but:
    - Policy goal block uses ``delta_command`` (delta = ref − current) and ``motion_anchor_ori_b_maskable``.
    - Both policy goal terms are jointly masked per env per timestep with probability ``p_mask`` (curriculum).
    - Teacher obs unchanged: still ``command`` (absolute) + ``motion_anchor_ori_b`` (matches stage-1 teacher's
      training distribution, never masked).
    """

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    @configclass
    class MUSEObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            # 1. Goal block: delta-command + maskable anchor orientation.
            command = ObsTerm(func=mdp.delta_command, params={"command_name": "motion"})  # J*2
            motion_anchor_ori_b = ObsTerm(
                func=mdp.motion_anchor_ori_b_maskable,
                params={"command_name": "motion"},
                noise=Unoise(n_min=-0.05, n_max=0.05),  # 6
            )
            # 2. Proprio.
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 5

        @configclass
        class TeacherCfg(ObsGroup):
            # Teacher matches stage-1 PHC+ training: absolute command, real anchor ori, no masking, no noise.
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True
                self.history_length = 5

        policy: PolicyCfg = PolicyCfg()
        teacher: TeacherCfg = TeacherCfg()

    observations: MUSEObservationsCfg = MUSEObservationsCfg()
    rewards: RewardsExpertCfg = RewardsExpertCfg()
    curriculum: MUSEDistillationCurriculumCfg = MUSEDistillationCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()


@configclass
class MUSETransformerDistillationCurriculumCfg(CurriculumCfg):
    """Curriculum for the transformer-MUSE: ramps env's p_mask. Curriculum drives the encoder's
    1-step ``goal_mask_history`` bit, which the policy routes to its goal-token
    ``key_padding_mask``."""

    goal_mask_p = CurrTerm(
        func=mdp.curriculums.goal_mask_probability_curriculum,
        params={
            "command_name": "motion",
            "p_start": 0.0,
            "p_end": 0.0,
            "ramp_start_iter": 1000,
            "ramp_end_iter": 2000,
            "schedule": "linear",
        },
    )


@configclass
class MUSETransformerDistillationTrackingEnvCfg(GeneralTrackingEnvCfg):
    """Transformer-MUSE env: 1-step un-masked command + proprio history-5 + 1-step goal_mask.

    Layout mirrors :class:`SAGEIIDistillationTrackingEnvCfg` but with proprio H=5 (vs 10):
    - Goal terms use :func:`mdp.delta_command_real` and :func:`mdp.motion_anchor_ori_b` — the
      encoder always sees real values at the obs source. Masking lives entirely in the policy's
      transformer (the goal token is dropped from attention via ``key_padding_mask`` when the
      env-side mask bit is set).
    - Per-term ``history_length``: goal/mask = 0 (current step only); proprio = 5.
    - Group-level ``history_length`` left unset so per-term values pass through.

    Teacher obs unchanged: stage-1 PHC+ contract (absolute command, real anchor ori, no masking).
    """

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    @configclass
    class MUSETransformerObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            """MUSE-Transformer student input. Goal/mask are 1-step (no Isaac rolling); proprio
            terms are history-rolled to 5 frames via per-term ``history_length=5``. Group-level
            ``history_length`` is left unset so per-term values pass through.
            """

            # 1. Goal block (1 step, never masked at the obs source — masking is encoder-side).
            command = ObsTerm(
                func=mdp.delta_command_real, params={"command_name": "motion"}
            )  # J*2 = 58
            motion_anchor_ori_b = ObsTerm(
                func=mdp.motion_anchor_ori_b,
                params={"command_name": "motion"},
                #noise=Unoise(n_min=-0.05, n_max=0.05),
            )  # 6
            motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})

            # 2. Proprio history-5.
            joint_pos = ObsTerm(
                func=mdp.joint_pos_rel,
                #noise=Unoise(n_min=-0.01, n_max=0.01),
                history_length=5,
            )
            joint_vel = ObsTerm(
                func=mdp.joint_vel_rel,
                #noise=Unoise(n_min=-0.5, n_max=0.5),
                history_length=5,
            )
            base_ang_vel = ObsTerm(
                func=mdp.base_ang_vel,
                #noise=Unoise(n_min=-0.2, n_max=0.2),
                history_length=5,
            )
            actions = ObsTerm(func=mdp.last_action, history_length=5)
            # 3. Mask bit (1 step) -> routed to encoder's goal-token key_padding_mask.
            goal_mask_history = ObsTerm(
                func=mdp.goal_mask_history, params={"command_name": "motion"}
            )

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                # Do NOT set self.history_length — preserves per-term values
                # (goal/mask = 0, proprio = 5).

        @configclass
        class TeacherCfg(ObsGroup):
            """Teacher: stage-1 PHC+ contract (absolute command + real anchor ori, no masking).

            Order must match the teacher's training obs schema EXACTLY — the MLP's first-layer
            weights are positional. Today's teacher (2026-05-13 round 2 sonic_55k, 163 dims/frame
            × 5 history = 815) ends with ref_base_lin_vel; see OneStageTrackingEnvCfg.PolicyCfg
            for the source-of-truth ordering."""

            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            actions = ObsTerm(func=mdp.last_action)
            # Appended 2026-05-13 round 1.
            motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
            # Appended 2026-05-13 round 2 (sonic_55k teacher). Keep base_lin_vel and ref_base_lin_vel
            # as the LAST two terms in this order for teacher-ckpt compatibility.
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True
                self.history_length = 5

        policy: PolicyCfg = PolicyCfg()
        teacher: TeacherCfg = TeacherCfg()

    observations: MUSETransformerObservationsCfg = MUSETransformerObservationsCfg()
    rewards: RewardsExpertCfg = RewardsExpertCfg()
    curriculum: MUSETransformerDistillationCurriculumCfg = MUSETransformerDistillationCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()


# 3. keypoint-based tracker:
@configclass
class VAEDistillationTrackingEnvCfg(GeneralTrackingEnvCfg):
    """
    VAE-style teacher-student distillation environment.
    """

    commands: MultiMotionCommandsCfg = MultiMotionCommandsCfg()

    @configclass
    class VAEDistillationObservationsCfg:
        """Observation specifications for VAE distillation."""

        @configclass
        class PolicyCfg(ObsGroup):
            """Student observations: reference motion keypoints + body linear velocities + proprio."""

            # 1. Reference motion body positions in robot-anchor frame (motion clip targets, not robot FK).
            body_pos = ObsTerm(func=mdp.ref_body_pos_robot_anchor_b, params={"command_name": "motion"}) # N * 3
            body_lin_vel = ObsTerm(func=mdp.ref_body_lin_vel_robot_anchor_b, params={"command_name": "motion"}) # N * 3
            motion_anchor_ori_b = ObsTerm(
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}) # 6
            ref_base_lin_vel_b = ObsTerm(
                func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"}) # 3
            

            # 2. Proprioception: velocity and orientation (needed for tracking; deployable via IMU/estimator)
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)) # J
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)) # J
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) # 3
            actions = ObsTerm(func=mdp.last_action) # J

            def __post_init__(self):
                # Student should be robust to corruption, and all terms concatenated.
                self.enable_corruption = True
                self.concatenate_terms = True
                self.history_length = 5

        @configclass
        class TeacherCfg(ObsGroup):
            
            # 1. Reference motion
            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})

            # 2. Proprioception
            joint_pos = ObsTerm(func=mdp.joint_pos_rel) # J
            joint_vel = ObsTerm(func=mdp.joint_vel_rel) # J
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel) # 3
            actions = ObsTerm(func=mdp.last_action) # J

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True
                self.history_length = 5

        # Observation groups
        policy: PolicyCfg = PolicyCfg()
        teacher: TeacherCfg = TeacherCfg()
        
    observations: VAEDistillationObservationsCfg = VAEDistillationObservationsCfg()
    rewards: RewardsExpertCfg = RewardsExpertCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands = MultiMotionCommandsCfg()


# 5. MUSE-Kp distillation: KP-token student warmstarted from MUSE-Transformer.
@configclass
class MUSEKpDistillationCurriculumCfg(CurriculumCfg):
    """4-phase KP6 MUSE-Kp curriculum: pure-bernoulli p_see ramp → OOD-avoidance mask mix.

    Mode spec is now the 5-mode :func:`mask_modes.muse_kp6_ood_mix_mode_spec`
    (``bernoulli`` + the 4 single-point deploy modes: L/R wrist, torso, pelvis), so every
    phase's ``mode_probs`` is a length-5 tuple. Phases 1-3 keep the original pure-bernoulli
    behaviour — ``mode_probs=(1.0, 0, 0, 0, 0)`` (bernoulli only) with the same p_see ramp.
    Phase 4 switches to the 5-way mask-mode sampling to remove the single-point deploy OOD.

    Phases (p_see = bernoulli-mode keep-prob; only applied while bernoulli prob > 0):
    - Phase 1 (iter 0..1000):    bernoulli-only, p_see 1.0 — all 6 KP points visible.
    - Phase 2 (iter 1000..2000): bernoulli-only, p_see ramps 1.0 → 0.4 (linear).
    - Phase 3 (iter 2000..5000): bernoulli-only, p_see held at 0.4.
    - Phase 4 (iter 5000..end):  mask-mode sampling — ``(0.2, 0.2, 0.2, 0.2, 0.2)`` over
      {bernoulli (p_see=0.4), L wrist, R wrist, torso, pelvis}. Avoids test-time OOD on the
      single-point-visible interactive-drag deploy distribution.

    The headline run uses JC encoder-decoder warmstart only (no student resume), so a fresh
    run executes all 4 phases in order: phases 1-3 are the bernoulli warmup over iter
    0..5000, then phase 4 (mask-mode mix) from iter 5000 on. (If a student resume is opted
    back in via ``RESUME_CHECKPOINT``, the iter-5000 phase-4 boundary still lets a
    ``model_5000.pt`` resume land directly in the mask-mode-sampling phase.)

    Used only by :class:`MUSEKpDistillationTrackingEnvCfg` (the
    ``MUSE-Kp-Distill-General-Tracking-Flat-G1-v0`` task).
    """

    keypoint_mask_mode = CurrTerm(
        func=mdp.curriculums.keypoint_mask_mode_curriculum,
        params={
            "command_name": "motion",
            "phase_until_learning_iterations": (1500, 4500, None), #(2000, 4000, None),
            "mask_phases": (
                {"mode_probs": mask_modes.MUSE_KP6_UNIFIED_BERNOULLI_PROBS, "p_start": 1.0, "p_end": 1.0},
                {"mode_probs": mask_modes.MUSE_KP6_UNIFIED_BERNOULLI_PROBS, "p_start": 1.0, "p_end": 0.4},
                {"mode_probs": mask_modes.MUSE_KP6_UNIFIED_BERNOULLI_PROBS, "p_start": 0.4, "p_end": 0.4},
                #{"mode_probs": mask_modes.MUSE_KP6_UNIFIED_MIX_PROBS, "p_start": 0.4, "p_end": 0.4},
            ),
        },
    )


@configclass
class MUSEKpLatentDemoCurriculumCfg(CurriculumCfg):
    """8-mode latent-distill curriculum: bernoulli p_see warmup → 8-mode deploy-shaped mix.

    For :class:`G1MUSEKpLatentDistillationTrackingEnvCfg` (5-body KP5 set; the 8 deploy modes
    from :func:`mask_modes.muse_kp5_latent_demo_mode_spec` — full / vr / torso / L,R wrist /
    wrists / ankles / bernoulli). Unlike the shared :class:`MUSEKpDistillationCurriculumCfg`
    (whose final mix phase is commented out, so it trains bernoulli-only), this one ENABLES
    the demo mix in its last phase, so every explicit deploy mode is in-distribution.

    Phases (``bernoulli`` is found by name; the p_see ramp gates only while its prob > 0):
    - Phase 1 (iter 0..1500):    bernoulli-only, p_see 1.0 — all 5 KP points visible.
    - Phase 2 (iter 1500..4500): bernoulli-only, p_see ramps 1.0 → 0.4 (linear).
    - Phase 3 (iter 4500..end):  8-way demo mix (``MUSE_KP5_LATENT_DEMO_MIX_PROBS``), bernoulli
      held at p_see=0.4. Removes the test-time OOD on the single-/multi-point deploy modes.

    Length-8 ``mode_probs`` tuples — paired ONLY with the 8-mode latent-demo spec. The KP6
    envs keep :class:`MUSEKpDistillationCurriculumCfg` (length-6) and are unaffected.
    """

    keypoint_mask_mode = CurrTerm(
        func=mdp.curriculums.keypoint_mask_mode_curriculum,
        params={
            "command_name": "motion",
            "phase_until_learning_iterations": (1500, 6500, None),
            "mask_phases": (
                {"mode_probs": mask_modes.MUSE_KP5_LATENT_DEMO_BERNOULLI_PROBS, "p_start": 1.0, "p_end": 1.0},
                {"mode_probs": mask_modes.MUSE_KP5_LATENT_DEMO_BERNOULLI_PROBS, "p_start": 1.0, "p_end": 0.4},
                {"mode_probs": mask_modes.MUSE_KP5_LATENT_DEMO_MIX_PROBS, "p_start": 0.4, "p_end": 0.4},
            ),
        },
    )


@configclass
class MUSEKpDistillationTrackingEnvCfg(GeneralTrackingEnvCfg):
    """MUSE-Kp distillation env: KP-token student on the 5-body cotrain KP set
    (:data:`mask_modes.COTRAIN_KP5_BODIES`) using an OmniGrasp-style per-body packing
    (1 absolute current slot + H delta-from-current-robot-pos future slots, all in the
    current robot-anchor frame) + proprio history-5 + MUSE-shape teacher obs.

    Per-body KP packing along the slot axis (L = 1 + H):
      - slot 0 (3D): absolute ref_pos at the CURRENT step (anchor frame).
      - slots 1..H (3D each): delta = ref_pos_{t+k} - robot_body_pos_{t} (anchor frame).
    With H_lookahead=10 → L=11 → 33 dims per body. Mask is per-body (broadcast across L slots).

    Teacher obs matches the **MUSE-Transformer teacher obs** (stage-1 PHC+ contract: absolute
    command + real anchor ori, no masking). The frozen MLP teacher loaded from the MUSE
    checkpoint produces the action target — no SAGE encoder in the loop.
    """

    commands: PartialMaskedMultiMotionCommandsCfg = PartialMaskedMultiMotionCommandsCfg()

    @configclass
    class MUSEKpObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            """KP-student input. ``kp_lookahead`` and ``kp_mask_lookahead`` bake L_pack=len(layout)
            slots inside the term (no Isaac history rolling). Proprio terms are history-rolled to 5
            frames in the MUSE layout so the warmstarted decoder slot matches.

            Layout: log-spaced ``KP_LAYOUT_0_5S`` (12 slots: 3 history + 1 abs + 8 future, cap=0.5s
            lookahead). Per-slot semantics:
              - offset == 0: absolute ``ref_pos_t`` in anchor frame.
              - offset != 0: delta ``ref_pos_{t+offset} − robot_body_pos_t`` in anchor frame.
            Edge handling: motion clip ends pad with the nearest available frame (clamped by the
            motion loader's ``compute_global_indices``).
            """

            # 1. KP goal block: log-spaced layout in current anchor frame, NaN at masked bodies;
            #    row-major [L, n_bodies, 3]. Must match the policy cfg's :attr:`kp_layout`.
            kp_lookahead = ObsTerm(
                func=mdp.ref_body_pos_robot_anchor_b_logspaced,
                params={"command_name": "motion", "slot_offsets": KP_LAYOUT_0_5S},
            )
            # Mask block has the same L slot dimension so the encoder's
            # ``mask.reshape(L, N)[..., 0, :]`` per-body collapse stays correct.
            kp_mask_lookahead = ObsTerm(
                func=mdp.partial_kp_mask_logspaced,
                params={"command_name": "motion", "num_slots": len(KP_LAYOUT_0_5S)},
            )

            # 2. Proprio history-5 (matches MUSE-Transformer decoder's expected per-frame layout = 90 dims).
            joint_pos = ObsTerm(
                func=mdp.joint_pos_rel,
                noise=Unoise(n_min=-0.01, n_max=0.01),
                history_length=5,
            )
            joint_vel = ObsTerm(
                func=mdp.joint_vel_rel,
                noise=Unoise(n_min=-0.5, n_max=0.5),
                history_length=5,
            )
            base_ang_vel = ObsTerm(
                func=mdp.base_ang_vel,
                noise=Unoise(n_min=-0.2, n_max=0.2),
                history_length=5,
            )
            actions = ObsTerm(func=mdp.last_action, history_length=5)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True
                # Per-term history_length governs (kp_*=0, proprio=5).

        @configclass
        class TeacherCfg(ObsGroup):
            """MLP teacher input: PHC+ stage-1 contract. Order must match the teacher's training obs
            schema EXACTLY — the MLP's first-layer weights are positional. Today's teacher
            (2026-05-13 round 2 sonic_55k, 163 dims/frame × 5 history = 815) ends with
            ref_base_lin_vel; mirrors the cotrain env's teacher slot. See OneStageTrackingEnvCfg.PolicyCfg
            for the source-of-truth ordering."""

            command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            actions = ObsTerm(func=mdp.last_action)
            # Appended 2026-05-13 round 1.
            motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
            # Appended 2026-05-13 round 2 (sonic_55k teacher). Keep base_lin_vel and ref_base_lin_vel
            # as the LAST two terms in this order for teacher-ckpt compatibility.
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            ref_base_lin_vel = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True
                self.history_length = 5

        policy: PolicyCfg = PolicyCfg()
        teacher: TeacherCfg = TeacherCfg()

    observations: MUSEKpObservationsCfg = MUSEKpObservationsCfg()
    rewards: RewardsExpertCfg = RewardsExpertCfg()
    curriculum: MUSEKpDistillationCurriculumCfg = MUSEKpDistillationCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands = PartialMaskedMultiMotionCommandsCfg()
        # Reward-scale alignment with the teacher's training env (OneStageTrackingEnvCfg).
        # The sonic_55k teacher was optimized for these tuned anchor weights (anchor_pos 0.5→2.0,
        # anchor_lin_vel 1.0→3.0, anchor_ori 0.5→1.0); RewardsExpertCfg ships the un-tuned
        # defaults. MUSE distillation is pure BC (reward is logging-only, never in the loss), so
        # this changes NOTHING in training — it only makes the teacher-pilot Episode_Reward
        # directly comparable to the teacher's known training reward, so a correctly-loaded
        # teacher reads at its real performance during the 100%-teacher-pilot bootstrap.
        
        #self.rewards.motion_global_anchor_pos.weight = 2.0
        #self.rewards.motion_anchor_lin_vel.weight = 3.0
        #self.rewards.motion_global_anchor_ori.weight = 1.0

        # Masked-KP fairness + distillation data exploitation: drop the WRISTS from the
        # ``ee_body_pos`` early-termination (keep only ankles). Under the p_see curriculum
        # (final p_see=0.4) a wrist keypoint is masked ~60% of the time, so the student gets
        # no wrist target — terminating on wrist-Z divergence then (1) unfairly fails the
        # episode for not tracking something it cannot see, and (2) truncates the clip
        # mid-rollout, discarding the frozen teacher's remaining BC targets (every step is
        # supervision in pure-BC distillation). Empirically this was ~all of the upper /
        # loco_upper failures at p_see=0.4 (ee_body_pos ≫ anchor_*; see per-category eval
        # 2026-05-16). Ankles stay: ankle-Z divergence ≈ fall / instability — a genuinely
        # bad state worth resetting, and unrelated to KP-mask fairness.
        #
        # NOTE: ``mdp_termination_success_rate`` no longer reflects arm tracking after this
        # (it measures "didn't fall + ankles tracked"); monitor upper-body quality via
        # ``Metrics/motion/error_body_pos`` and ``kp_modes/bernoulli/error_body_pos_visible``
        # instead. Scoped to this class + its subclasses (MUSEKpAux, MUSEKp5FromScratch);
        # the base TerminationsCfg used by the PHC+ teacher and other tasks is untouched.
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]


@configclass
class MUSEKp5FromScratchCurriculumCfg(CurriculumCfg):
    """2-phase mask curriculum for the 5-body-native from-scratch pilot-anneal run.

    Mode order matches :data:`mask_modes.MUSE_KP5_CURRICULUM_MODE_NAMES` (length 5):
      0 ``bernoulli`` · 1 ``kp5_full`` · 2 ``kp5_torso`` · 3 ``kp5_wrists`` · 4 ``kp5_ankles``.

    - **M1** (iter 0..1500): 100% ``bernoulli`` over the 5 kp5 bodies; per-body ``p_keep``
      ramps 1.0 → 0.4 (full visibility → sparse).
    - **M2** (iter 1500..∞): ``p_keep`` held at 0.4; mode sampling
      ``0.4 bernoulli + 0.15 each of {kp5_full, kp5_torso, kp5_wrists, kp5_ankles}`` —
      broad coverage plus the explicit interactive-drag demo modes.

    Aligned with the algorithm's pilot anneal: ``pilot_teacher_start_iter`` should equal the
    M1→M2 boundary (1500) so the mask reaches final difficulty *before* the pilot starts
    shifting teacher→student (one new hardness at a time; mask ramped under teacher-pilot).
    """

    keypoint_mask_mode = CurrTerm(
        func=mdp.curriculums.keypoint_mask_mode_curriculum,
        params={
            "command_name": "motion",
            "phase_until_learning_iterations": (1500, None),
            # CANONICAL unified 6-mode spec (single-source constants). M1: bernoulli-only,
            # p_see 1.0 → 0.4. M2: 0.5 bernoulli + 0.1 each deploy mode, p_see held 0.4.
            "mask_phases": (
                {"mode_probs": mask_modes.MUSE_KP6_UNIFIED_BERNOULLI_PROBS, "p_start": 1.0, "p_end": 0.4},
                {"mode_probs": mask_modes.MUSE_KP6_UNIFIED_MIX_PROBS, "p_start": 0.4, "p_end": 0.4},
            ),
        },
    )


@configclass
class MUSEKp5FromScratchTrackingEnvCfg(MUSEKpDistillationTrackingEnvCfg):
    """6-body-native (KP5 demo bodies + pelvis), 1s-lookahead, from-scratch (no MUSE-Kp
    warmstart) env for the passive→active pilot-anneal run.

    Differences vs :class:`MUSEKpDistillationTrackingEnvCfg`:
      - KP obs uses the **log_1_0s** layout (13 slots, future to +1.0 s) instead of log_0_5s.
      - 2-phase curriculum (:class:`MUSEKp5FromScratchCurriculumCfg`, 5 modes) instead of the
        14-body 4-phase dense+demo-mix.
    Body set / mask spec are wired G1-side in :class:`G1MUSEKp5FromScratchTrackingEnvCfg`
    (``body_names = KP6_NATIVE_BODIES``, ``mask_mode_spec = muse_kp6_curriculum_mode_spec()``).
    Pelvis is included (FIRST) so the motion-command reset writes the robot root-link
    reference — the old KP5 set dropped pelvis and led with torso, spawning the base at the
    torso pose every reset (a near-perfect teacher then read ~0.80 not ~0.99). Pelvis is a
    maskable root-pose token; the demo modes stay pelvis-free so deployment is unchanged.

    The runner cfg pairs this with ``kp_n_bodies=6``, ``kp_layout="log_1_0s"``,
    ``kp_lookahead_steps=13``, aux predictor off, pilot anneal on, freeze-warmup off, and the
    JC MUSE-Transformer ckpt as backbone+decoder warmstart.
    """

    @configclass
    class MUSEKp5Log1sObservationsCfg(MUSEKpDistillationTrackingEnvCfg.MUSEKpObservationsCfg):
        @configclass
        class PolicyCfg(MUSEKpDistillationTrackingEnvCfg.MUSEKpObservationsCfg.PolicyCfg):
            # Override only the two KP terms to the 1s log-spaced layout; proprio terms inherit.
            kp_lookahead = ObsTerm(
                func=mdp.ref_body_pos_robot_anchor_b_logspaced,
                params={"command_name": "motion", "slot_offsets": KP_LAYOUT_1_0S},
            )
            kp_mask_lookahead = ObsTerm(
                func=mdp.partial_kp_mask_logspaced,
                params={"command_name": "motion", "num_slots": len(KP_LAYOUT_1_0S)},
            )

        policy: PolicyCfg = PolicyCfg()
        teacher: MUSEKpDistillationTrackingEnvCfg.MUSEKpObservationsCfg.TeacherCfg = (
            MUSEKpDistillationTrackingEnvCfg.MUSEKpObservationsCfg.TeacherCfg()
        )

    observations: MUSEKp5Log1sObservationsCfg = MUSEKp5Log1sObservationsCfg()
    curriculum: MUSEKp5FromScratchCurriculumCfg = MUSEKp5FromScratchCurriculumCfg()
