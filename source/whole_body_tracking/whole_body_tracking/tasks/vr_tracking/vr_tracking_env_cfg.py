from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from whole_body_tracking.tasks.tracking.tracking_env_cfg import MySceneCfg, VELOCITY_RANGE
import whole_body_tracking.tasks.vr_tracking.mdp as mdp

EE_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]


@configclass
class CommandsCfg:
    motion = mdp.VRMultiMotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        resample_motions_every_s=50.0,
        motion_sampling_warmup_s=1.0e9,
        motion_sampling_ramp_s=2000.0,
        motion_sampling_schedule="cosine",
        debug_vis=True,
        compact_goal_observation=True,
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
class ActionsCfg:
    joint_pos = mdp.ResidualLatentActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        use_default_offset=True,
        tanh_actions=False,
        residual_scale=1.0,
        #clip_actions=True,
        #clip_action_min=-3.0,
        #clip_action_max=3.0,
        #history_action_clip=1.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):

        # 1. Reference motion body positions in robot-anchor frame (motion clip targets, not robot FK).
        body_pos = ObsTerm(func=mdp.ref_body_pos_robot_anchor_b, params={"command_name": "motion"}) # N * 3
        body_lin_vel = ObsTerm(func=mdp.ref_body_lin_vel_robot_anchor_b, params={"command_name": "motion"}) # N * 3
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}) # 6
        ref_base_lin_vel_b = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"}) # 3

        # 2. Proprioception: velocity and orientation (needed for tracking; deployable via IMU/estimator)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)) # J
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)) # J
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) # 3
        actions = ObsTerm(func=mdp.last_action) # J

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class CriticCfg(ObsGroup):
        
        # 1. obs-goal: 
        body_pos = ObsTerm(func=mdp.ref_body_pos_robot_anchor_b, params={"command_name": "motion"})
        ref_body_lin_vel = ObsTerm(func=mdp.ref_body_lin_vel_robot_anchor_b, params={"command_name": "motion"})
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        ref_base_lin_vel_b = ObsTerm(func=mdp.ref_base_lin_vel_b, params={"command_name": "motion"})
        
        # 2. obs-proprio: 
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
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
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:

    # ===== Reward terms =====
    # 1. Anchor:
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=1.0,  # 0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_anchor_lin_vel = RewTerm(
        func=mdp.motion_anchor_linear_velocity_error_exp,
        weight=1.0,  # 2*1.0
        params={"command_name": "motion", "std": 1.0},
    )

    # 2. Active body:
    motion_active_ee_pos = RewTerm(
        func=mdp.motion_active_body_position_error_exp,
        weight=8.0, #4.0,
        params={"command_name": "motion", "std": 0.2},
    )
    motion_active_ee_vel = RewTerm(
        func=mdp.motion_active_body_linear_velocity_error_exp,
        weight=6.0,
        params={"command_name": "motion", "std": 0.75},
    )

    # ===== Penalty terms =====
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:

    # 1. timeout:
    motion_end = DoneTerm(func=mdp.motion_end, params={"command_name": "motion"}, time_out=True)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 2. anchor orientation:
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only, 
        params={"command_name": "motion", "threshold": 0.25}
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_active_body_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class VRTrackingEnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 15 * 2**17
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.viewer.origin_type = "env"
        self.viewer.asset_name = "robot"

