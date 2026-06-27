"""ObstacleReach command + keep-out reward + obstacle/lookahead observations.

TRUE PER-RESET sampling (no clip pool). The command subclasses
:class:`PartialMaskedMultiMotionCommand` only to reuse its masked-KP + reset machinery; it
loads a SINGLE standing-seed clip (for the reset root/joint pose + the standing pose of the
non-wrist bodies). Every reset it samples a fresh scene with
:func:`~...mdp.obstacle_reach.sample_phase` (phase from ``OBSTACLE_REACH_PHASE``) — a reach
point P + obstacle OBBs — or, for eval, loads ONE authored scene JSON
(``OBSTACLE_REACH_SCENE``) shared by all envs.

The reach target P is injected by OVERRIDING ``body_pos_w`` at the right-wrist slot. Because
the target is STATIC, the keypoint-lookahead at every future slot equals the current
reference, so the ObstacleReach lookahead obs below build the packing from ``body_pos_w``
(mirroring ``ref_body_pos_robot_anchor_b_logspaced`` exactly) — the frozen encoder sees the
identical values it would for a constant clip, with no pre-generated pool. See
[[obstacle-reach-task-design]].
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .commands import PartialMaskedMultiMotionCommand, PartialMaskedMultiMotionCommandCfg
from .obstacle_reach import (
    ObstacleScene,
    num_boxes,
    phase_box_half_sizes,
    quat_rotate_inverse,
    sample_phase,
    scene_from_json,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_REACH_BODY = "right_wrist_yaw_link"
_OBSTACLE_COLOR = (0.40, 0.55, 0.75)


def make_obstacle_collider_cfgs(phase: int) -> dict[str, RigidObjectCfg]:
    """RigidObjectCfg for each FIXED-size, KINEMATIC box collider of ``phase`` (option 2:
    SOLID obstacles the robot physically cannot pass through). Keys ``obstacle_0..N-1`` match
    the slot order of :func:`~...obstacle_reach.sample_phase`; the command writes their poses
    per reset. Empty dict for the free-reach phase. Kinematic ⇒ the box is immovable by
    contact (robot collides, box stays put) and is teleported via ``write_root_pose_to_sim``.
    The cfgs are attached to the scene in the env's ``__post_init__`` (phase from env var)."""
    cfgs: dict[str, RigidObjectCfg] = {}
    for i, (hx, hy, hz) in enumerate(phase_box_half_sizes(phase)):
        cfgs[f"obstacle_{i}"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Obstacle_%d" % i,
            spawn=sim_utils.CuboidCfg(
                size=(2.0 * hx, 2.0 * hy, 2.0 * hz),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_OBSTACLE_COLOR),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            ),
            # Parked below ground at spawn; the command teleports it on the first reset.
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
        )
    return cfgs


class ObstacleReachCommand(PartialMaskedMultiMotionCommand):
    """Per-reset obstacle-reach command: samples P + obstacles each reset and injects the
    constant reach target at the right-wrist slot."""

    cfg: "ObstacleReachCommandCfg"

    def __init__(self, cfg: "ObstacleReachCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._wrist_slot = list(self.cfg.body_names).index(_REACH_BODY)
        # Phase / fixed-scene from env vars (the cfg is the shared masked cfg; env-var config
        # matches the repo's KP-layout / LATENTRL_REF_W style and avoids cfg surgery).
        self._phase = int(os.environ.get("OBSTACLE_REACH_PHASE", "1"))
        # Number of SOLID rigid box colliders the env spawned for this phase (option 2). The
        # env's __post_init__ attaches obstacle_0..N-1 to the scene with FIXED sizes; we set
        # their poses per reset. 0 for the free-reach phase.
        self._n_boxes = num_boxes(self._phase)
        scene_path = os.environ.get("OBSTACLE_REACH_SCENE", "").strip()
        self._fixed: ObstacleScene | None = (
            scene_from_json(scene_path, 1, self.device) if scene_path else None
        )
        mode = f"fixed scene {scene_path!r}" if self._fixed is not None else f"phase {self._phase}"
        print(f"[ObstacleReachCommand] per-reset sampling ({mode}); reach body = {_REACH_BODY}; "
              f"solid colliders = {self._n_boxes}")

        self.obstacle_scene = ObstacleScene.empty(self.num_envs, self.device)
        self._reach_target_rel = torch.zeros(self.num_envs, 3, device=self.device)  # env-relative P
        # Consecutive in-tolerance steps -> the success-on-hold termination. Reset per episode.
        self.reach_hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._sample_scene(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    # ---- per-reset scene sampling --------------------------------------------------------
    def _sample_scene(self, env_ids: torch.Tensor) -> None:
        """Sample (or broadcast the fixed) scene for ``env_ids``; store P (env-relative) and
        the obstacle scene in WORLD frame (env-relative + env origin)."""
        n = int(env_ids.numel())
        if self._fixed is not None:
            sc = self._fixed
            centers = sc.centers.expand(n, -1, -1)
            half = sc.half.expand(n, -1, -1)
            quat = sc.quat.expand(n, -1, -1)
            valid = sc.valid.expand(n, -1)
            target = sc.target.expand(n, -1)
        else:
            sc = sample_phase(self._phase, n, self.device)
            centers, half, quat, valid, target = sc.centers, sc.half, sc.quat, sc.valid, sc.target
        origin = self._env.scene.env_origins[env_ids]
        self._reach_target_rel[env_ids] = target
        self.obstacle_scene.centers[env_ids] = centers + origin[:, None, :]
        self.obstacle_scene.half[env_ids] = half
        self.obstacle_scene.quat[env_ids] = quat
        self.obstacle_scene.valid[env_ids] = valid
        self.obstacle_scene.target[env_ids] = target + origin

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Standing-seed body positions (world) with the right wrist replaced by the per-env
        reach target P. Single source of the target — drives the reach reward, the success
        metric, and (via the ObstacleReach lookahead obs) the policy/critic obs."""
        base = super().body_pos_w.clone()
        base[:, self._wrist_slot] = self._reach_target_rel + self._env.scene.env_origins
        return base

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)  # robot reset to standing seed + mask sample
        if len(env_ids) > 0:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._sample_scene(ids)
            self.reach_hold_steps[ids] = 0
            self._place_obstacle_colliders(ids)

    # ---- SOLID obstacles: teleport the kinematic box colliders to the sampled poses --------
    def _place_obstacle_colliders(self, env_ids: torch.Tensor) -> None:
        """Write each rigid box's world pose for the reset envs (kinematic ⇒ stays put until
        the next reset; the robot physically collides). The boxes are obstacle_0..N-1, matching
        the slots :func:`sample_phase` fills; ``obstacle_scene.centers`` is already world-frame."""
        if self._n_boxes == 0:
            return
        for i in range(self._n_boxes):
            asset = self._env.scene[f"obstacle_{i}"]
            pose = torch.cat(
                [self.obstacle_scene.centers[env_ids, i], self.obstacle_scene.quat[env_ids, i]], dim=-1
            )
            asset.write_root_pose_to_sim(pose, env_ids=env_ids)


@configclass
class ObstacleReachCommandCfg(PartialMaskedMultiMotionCommandCfg):
    """Like the masked multi-motion cfg, but instantiates :class:`ObstacleReachCommand`
    (phase / fixed-scene come from ``OBSTACLE_REACH_PHASE`` / ``OBSTACLE_REACH_SCENE``)."""

    class_type: type = ObstacleReachCommand


# ====================================================================================
# Observations: ObstacleReach keypoint lookahead, built from the CONSTANT reach target.
# Mirrors mdp.ref_body_pos_robot_anchor_b_logspaced / ref_single_body_..._logspaced exactly
# (offset 0 -> abs ref in anchor frame; offset != 0 -> delta from robot's current body pos),
# but sources the reference from command.body_pos_w (constant over time) instead of gathering
# clip frames. For a static target this is identical to the gather version, so the FROZEN
# encoder sees the values it was distilled on — with no pre-generated clip pool.
# ====================================================================================
def obstacle_reach_kp_lookahead(
    env: "ManagerBasedRLEnv", command_name: str, slot_offsets: tuple[int, ...]
) -> torch.Tensor:
    """Policy KP lookahead (all tracked bodies, NaN at masked bodies). Returns [N, L*n_bodies*3]."""
    command: ObstacleReachCommand = env.command_manager.get_term(command_name)
    n_bodies = len(command.cfg.body_names)
    N, L = env.num_envs, len(slot_offsets)
    offsets_t = torch.as_tensor(slot_offsets, device=command.device, dtype=torch.long)

    ref_w = command.body_pos_w[:, None, :, :].expand(N, L, n_bodies, 3)  # constant over slots
    anchor_pos_w = command.robot_anchor_pos_w
    anchor_quat_w = command.robot_anchor_quat_w
    fc = N * L * n_bodies
    rel_ref = (ref_w - anchor_pos_w[:, None, None, :]).reshape(fc, 3)
    quat_ref = anchor_quat_w[:, None, None, :].expand(N, L, n_bodies, 4).reshape(fc, 4)
    ref_pos_b = quat_rotate_inverse(quat_ref, rel_ref).view(N, L, n_bodies, 3)

    robot_w = command.robot_body_pos_w  # [N, n_bodies, 3]
    rel_robot = (robot_w - anchor_pos_w[:, None, :]).reshape(N * n_bodies, 3)
    quat_robot = anchor_quat_w[:, None, :].expand(N, n_bodies, 4).reshape(N * n_bodies, 4)
    robot_pos_b = quat_rotate_inverse(quat_robot, rel_robot).view(N, n_bodies, 3)

    is_abs = (offsets_t == 0).view(1, L, 1, 1)
    packed = torch.where(is_abs, ref_pos_b, ref_pos_b - robot_pos_b.unsqueeze(1))
    mask = getattr(command, "_env_body_mask", None)
    if mask is not None:
        vis = (mask > 0.5).view(N, 1, n_bodies, 1)
        packed = torch.where(vis, packed, torch.full_like(packed, float("nan")))
    return packed.reshape(N, L * n_bodies * 3)


def obstacle_reach_rwrist_lookahead(
    env: "ManagerBasedRLEnv", command_name: str, body_name: str, slot_offsets: tuple[int, ...]
) -> torch.Tensor:
    """Privileged single-body (right-wrist) lookahead, unmasked. Returns [N, L*3]."""
    command: ObstacleReachCommand = env.command_manager.get_term(command_name)
    body_idx = list(command.cfg.body_names).index(body_name)
    N, L = env.num_envs, len(slot_offsets)
    offsets_t = torch.as_tensor(slot_offsets, device=command.device, dtype=torch.long)

    ref_w = command.body_pos_w[:, body_idx, :][:, None, :].expand(N, L, 3)
    anchor_pos_w = command.robot_anchor_pos_w
    anchor_quat_w = command.robot_anchor_quat_w
    rel_ref = (ref_w - anchor_pos_w[:, None, :]).reshape(N * L, 3)
    quat_ref = anchor_quat_w[:, None, :].expand(N, L, 4).reshape(N * L, 4)
    ref_pos_b = quat_rotate_inverse(quat_ref, rel_ref).view(N, L, 3)

    robot_w = command.robot_body_pos_w[:, body_idx, :]
    robot_pos_b = quat_rotate_inverse(anchor_quat_w, robot_w - anchor_pos_w)

    is_abs = (offsets_t == 0).view(1, L, 1)
    packed = torch.where(is_abs, ref_pos_b, ref_pos_b - robot_pos_b[:, None, :])
    return packed.reshape(N, L * 3)


# ====================================================================================
# Reward: obstacle keep-out (OBB penetration penalty)
# ====================================================================================
def obstacle_keepout_penalty(
    env: "ManagerBasedRLEnv", command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Sum of OBB penetration depth (m) over ``asset_cfg`` bodies. Positive = inside an
    obstacle; configure with a NEGATIVE weight. 0 when all listed bodies are clear."""
    command: ObstacleReachCommand = env.command_manager.get_term(command_name)
    asset = env.scene[asset_cfg.name]
    pts = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (N, B, 3) world
    return command.obstacle_scene.penetration(pts).sum(dim=-1)


# ====================================================================================
# Observation: obstacle OBBs in the robot-anchor frame (policy + critic)
# ====================================================================================
def obstacle_params_robot_anchor_b(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """Per-obstacle [center_b(3), half(3), valid(1)] in the robot-anchor frame, flattened
    to ``MAX_OBSTACLES * 7``. Invalid slots are zeroed."""
    command: ObstacleReachCommand = env.command_manager.get_term(command_name)
    scene = command.obstacle_scene
    N, K = scene.centers.shape[0], scene.centers.shape[1]
    anchor_pos = command.robot_anchor_pos_w[:, None, :]            # (N,1,3)
    anchor_quat = command.robot_anchor_quat_w[:, None, :].expand(N, K, 4)
    center_b = quat_rotate_inverse(anchor_quat, scene.centers - anchor_pos)  # (N,K,3)
    valid = scene.valid.to(center_b.dtype)[..., None]             # (N,K,1)
    feat = torch.cat([center_b, scene.half, valid.expand(N, K, 1)], dim=-1)  # (N,K,7)
    feat = feat * valid                                           # zero invalid slots
    return feat.reshape(N, K * 7)


# ====================================================================================
# Termination: reach SUCCESS (held). Use with time_out=True so the value is bootstrapped
# (a held reach is a HIGH-value state, not a zero-value terminal — bootstrapping it as a
# truncation is correct; a `terminated` cut would teach the critic the goal is worthless).
# ====================================================================================
def obstacle_reach_succeeded(
    env: "ManagerBasedRLEnv", command_name: str, success_tol: float = 0.08, hold_steps: int = 100
) -> torch.Tensor:
    """End the episode once the visible-POI (the right wrist) world error has stayed under
    ``success_tol`` (m) for ``hold_steps`` consecutive steps (~hold_steps * step_dt seconds)."""
    command: ObstacleReachCommand = env.command_manager.get_term(command_name)
    err = command.metrics["error_body_pos_w_visible"]  # world-frame visible-POI error, updated this step
    in_tol = err < success_tol
    command.reach_hold_steps = torch.where(
        in_tol, command.reach_hold_steps + 1, torch.zeros_like(command.reach_hold_steps)
    )
    return command.reach_hold_steps >= int(hold_steps)
