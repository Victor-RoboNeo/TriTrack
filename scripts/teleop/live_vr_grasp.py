#!/usr/bin/env python3
"""Live VR teleop + grasp-reach smoke test: Mapper-B + H2R + Stage-2 (model_14000), Isaac Lab.

Closes the loop for real: XRoboToolkit headset/controllers -> H2R-0 analytic
canonicalizer -> Mapper-B future-slot prediction -> frozen AnyBody Stage-2 ->
G1 in a flat-plane Isaac Lab scene with one graspable cube in front of the
robot. No RL policy, no reward/termination machinery -- this is TriTrack's
own deployment controller (`tritrack.controller.SparseIntentController`)
closing the loop through `tritrack.robot.isaaclab_bridge.IsaacLabG1Bridge`,
exactly the "mapper-B + H2R + stage-2" combo already in use, just with a
real headset in front of a simulated body instead of a UDP replay/echo robot.

"Grasping" here means the tracked right- (or left-) wrist keypoint reaching
within `--grasp_radius` of the cube for `--dwell_ticks` consecutive control
ticks. The AnyBody G1 rig has no fingers/gripper -- there is no attach
physics -- so this is a reach proxy for "the hand got to the object", not a
pick-up. Good enough to sanity-check the pipeline end to end; not a
manipulation benchmark.

Usage (two terminals):
  1) Start the XRoboToolkit PC Service, connect the headset (companion app ->
     enter this machine's IP under "PC Service:" -> Status: WORKING).
  2) conda activate isaaclab && cd /data/home/chenxiangyu/robotics/Anybody
     python scripts/teleop/live_vr_grasp.py                      # real headset
     python scripts/teleop/live_vr_grasp.py --mock               # no headset; synthetic wave, smoke test

Stand neutral (arms relaxed, facing the robot's forward direction) for the
first ~1.5 s after the window opens -- that's the H2R calibration window.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

from isaaclab.app import AppLauncher

TASK = "MUSE-Kp-LatentRL-Kp5-HeadHands-Locomani-G1-v0"
# Reset-pose donor clips only (see _apply_eval_motion) -- our controller ignores their future
# trajectory entirely. logs/tritrack/infer_clips/p1/loco (eval_isaac.py's default) has been
# cleaned up on this machine; batch4_locomani is a small, currently-present set of symlinks
# into datasets/SONIC_npzs/g1/npz_splits_loco_manip/train that still resolve.
CLIP_LOCO = Path("/data/home/chenxiangyu/robotics/Anybody/logs/tritrack/infer_clips/batch4_locomani")
MAPPER_B = Path("/data/home/chenxiangyu/victor/TriTrack/runs/mapper_b_intent72/mapper_best.pt")
TRITRACK_ROOT = Path("/data/home/chenxiangyu/victor/TriTrack")
if str(TRITRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRITRACK_ROOT))  # no-op if tritrack is already pip-installed

parser = argparse.ArgumentParser(description="Live VR teleop + grasp-reach smoke test.")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seconds", type=float, default=120.0)
parser.add_argument("--start_frame", type=int, default=10, help="reset-pose donor clip frame (standing).")
parser.add_argument("--mock", action="store_true", default=False, help="synthetic wave source, no headset needed.")
parser.add_argument("--no_mapper", action="store_true", default=False, help="disable Mapper-B (hold-current future slots).")
parser.add_argument("--grasp_hand", type=str, default="right", choices=("left", "right"))
parser.add_argument("--cube_forward", type=float, default=0.35, help="m, in front of the robot's initial heading.")
parser.add_argument("--cube_lateral", type=float, default=0.0, help="m, +left of the robot's initial heading.")
parser.add_argument("--cube_height", type=float, default=0.85, help="m, world z (flat ground assumed z=0).")
parser.add_argument("--cube_size", type=float, default=0.08, help="m, cube edge length.")
parser.add_argument("--grasp_radius", type=float, default=0.15, help="m, wrist-to-cube-center dwell threshold.")
parser.add_argument("--dwell_ticks", type=int, default=15, help="consecutive ticks under grasp_radius to latch GRASPED.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import whole_body_tracking.tasks  # noqa: E402, F401

from tritrack.anybody.stage2 import AnyBodyStage2  # noqa: E402
from tritrack.controller import SparseIntentController  # noqa: E402
from tritrack.intent.h2r_canonicalizer import H2RCanonicalizer  # noqa: E402
from tritrack.intent.se3_utils import quat_rotate  # noqa: E402
from tritrack.intent.state import SE3, IntentState  # noqa: E402
from tritrack.robot.g1_constants import CONTROL_HZ  # noqa: E402
from tritrack.robot.isaaclab_bridge import IsaacLabG1Bridge  # noqa: E402


def _apply_eval_motion(env_cfg, motion: str, start_frame: int) -> None:
    """Pin a single standing-ish clip for the RESET pose only. Our controller ignores the
    clip's own future trajectory entirely; this just avoids the known motion_group_sampling_
    ratios crash and gives a clean standing spawn. See scripts/sirac/eval_isaac.py."""
    env_cfg.commands.motion.motion = motion
    if hasattr(env_cfg.commands.motion, "motion_groups"):
        env_cfg.commands.motion.motion_groups = None
    if hasattr(env_cfg.commands.motion, "motion_group_sampling_ratios"):
        env_cfg.commands.motion.motion_group_sampling_ratios = None
    if hasattr(env_cfg.commands.motion, "start_frame"):
        env_cfg.commands.motion.start_from_beginning = True
        env_cfg.commands.motion.start_frame = start_frame
    if hasattr(env_cfg.commands.motion, "random_init_frame"):
        env_cfg.commands.motion.random_init_frame = False
    if hasattr(env_cfg.commands.motion, "resample_motions_every_s"):
        env_cfg.commands.motion.resample_motions_every_s = 0.0
    zero = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    if hasattr(env_cfg.commands.motion, "pose_range"):
        env_cfg.commands.motion.pose_range = dict(zero)
    if hasattr(env_cfg.commands.motion, "velocity_range"):
        env_cfg.commands.motion.velocity_range = dict(zero)
    if hasattr(env_cfg.commands.motion, "joint_position_range"):
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    if hasattr(env_cfg.commands.motion, "motion_dataset_shard_across_gpus"):
        env_cfg.commands.motion.motion_dataset_shard_across_gpus = False
    if hasattr(env_cfg.commands.motion, "max_active_motions"):
        env_cfg.commands.motion.max_active_motions = None


def _apply_terrain(env_cfg) -> None:
    env_cfg.scene.terrain.terrain_type = "plane"


def _disable_eval_hooks(env_cfg) -> None:
    """No RL reward/termination/curriculum/randomization machinery -- we drive actions
    ourselves every tick and only care about the physics + the grasp-cube prim."""
    if hasattr(env_cfg, "observations"):
        for group_name in ("policy", "teacher", "critic"):
            group = getattr(env_cfg.observations, group_name, None)
            if group is not None and hasattr(group, "enable_corruption"):
                group.enable_corruption = False
    if hasattr(env_cfg, "events"):
        env_cfg.events = None
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations = None
    if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
        for name in ("keypoint_mask_mode", "motion_group_ratio", "terrain_levels"):
            if hasattr(env_cfg.curriculum, name):
                setattr(env_cfg.curriculum, name, None)
    if hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = 1.0e6  # this is an interactive session, not an RL episode


def _make_grasp_cube_cfg(size: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspCube",
        spawn=sim_utils.CuboidCfg(
            size=(size, size, size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15)),
        ),
        # Parked below ground at spawn; teleported once we know the robot's reset pose.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
    )


class MockVRSource:
    """Synthetic head-sways-hands-circle stream (same math as scripts/mock_vr_sender.py,
    in-process, no UDP/headset). For smoke-testing the pipeline without hardware."""

    def __init__(self):
        self._t0 = time.time()
        self._ident = np.array([1.0, 0.0, 0.0, 0.0])

    def poll(self) -> IntentState | None:
        t = time.time() - self._t0
        head = SE3([0.05 * math.sin(0.5 * t), 0.0, 1.60 + 0.02 * math.sin(1.1 * t)], self._ident)
        lh = SE3([0.25 + 0.10 * math.sin(1.5 * t), 0.30, 1.05 + 0.10 * math.cos(1.5 * t)], self._ident)
        rh = SE3(
            [0.25 + 0.10 * math.sin(1.5 * t + math.pi), -0.30, 1.05 + 0.10 * math.cos(1.5 * t + math.pi)],
            self._ident,
        )
        return IntentState(time.time(), head, lh, rh)

    @property
    def stale(self) -> bool:
        return False

    def close(self) -> None:
        pass


def _recalibrate_listener(flag: dict) -> None:
    """Background thread: pressing Enter on stdin schedules a recalibration at the next tick."""
    while not flag.get("stop"):
        try:
            line = sys.stdin.readline()
        except Exception:
            return
        if not line:
            return
        flag["recalibrate"] = True
        print("[teleop] recalibrating on next tick -- stand neutral now", flush=True)


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1, use_fabric=True)
    n_files = len(list(CLIP_LOCO.rglob("*.npz")))
    if n_files == 0:
        raise FileNotFoundError(
            f"no reset-pose donor clips under {CLIP_LOCO} (only used for the initial standing "
            "pose -- pass --motion-equivalent by editing CLIP_LOCO to any directory of G1 "
            "tracking npz clips, e.g. datasets/SONIC_npzs/g1/npz_splits_loco_manip/train/<shard>)"
        )
    _apply_eval_motion(env_cfg, str(CLIP_LOCO), args_cli.start_frame)
    _apply_terrain(env_cfg)
    _disable_eval_hooks(env_cfg)
    setattr(env_cfg.scene, "grasp_cube", _make_grasp_cube_cfg(args_cli.cube_size))

    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    env.reset()

    asset = env.unwrapped.scene["robot"]
    cube = env.unwrapped.scene["grasp_cube"]
    bridge = IsaacLabG1Bridge(asset)

    rs0 = bridge.robot_state()
    fwd_w = quat_rotate(rs0.anchor_quat_w, np.array([1.0, 0.0, 0.0]))
    fwd_w[2] = 0.0
    fwd_w = fwd_w / max(np.linalg.norm(fwd_w), 1e-6)
    left_w = np.array([-fwd_w[1], fwd_w[0], 0.0])
    # anchor_pos_w is already absolute world (env origin baked in) -- no extra offset needed.
    cube_xy = rs0.anchor_pos_w[:2] + args_cli.cube_forward * fwd_w[:2] + args_cli.cube_lateral * left_w[:2]
    cube_world = np.array([cube_xy[0], cube_xy[1], args_cli.cube_height])
    pose = torch.zeros(1, 7, device=env.unwrapped.device)
    pose[0, :3] = torch.as_tensor(cube_world, dtype=torch.float32, device=env.unwrapped.device)
    pose[0, 3] = 1.0  # identity quat wxyz
    cube.write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=env.unwrapped.device))
    print(f"[teleop] grasp cube placed at world {cube_world.round(3).tolist()} (hand={args_cli.grasp_hand})", flush=True)

    stage2 = AnyBodyStage2.from_config(device=args_cli.device)  # default configs/anybody.yaml (model_14000)
    mapper = None
    if not args_cli.no_mapper:
        from tritrack.intent.mapper import FutureIntentMapper

        mapper, _ = FutureIntentMapper.from_checkpoint(MAPPER_B, device=args_cli.device)
    canon = H2RCanonicalizer(control_hz=CONTROL_HZ)
    source = MockVRSource() if args_cli.mock else None
    if source is None:
        from tritrack.sources.xrobotoolkit import XRoboToolkitSource

        source = XRoboToolkitSource()

    print("[teleop] waiting for the first intent packet ...", flush=True)
    intent = None
    t_wait = time.time()
    while intent is None:
        intent = source.poll()
        if intent is None:
            time.sleep(0.01)
            if time.time() - t_wait > 5.0:
                print("[teleop] still waiting -- is the headset connected / PC Service running?", flush=True)
                t_wait = time.time()

    ctrl = SparseIntentController(stage2, source, canonicalizer=canon, adapter=None, mapper=mapper, device=args_cli.device)
    ctrl.reset(bridge.robot_state(), first_intent=intent)
    print("[teleop] stream up; calibrating H2R on current operator pose (stand neutral ~1.5s) ...", flush=True)

    flag = {"stop": False, "recalibrate": False}
    listener = threading.Thread(target=_recalibrate_listener, args=(flag,), daemon=True)
    listener.start()
    print("[teleop] press ENTER at any time to recalibrate (stand neutral first).", flush=True)

    hand_idx = 2 if args_cli.grasp_hand == "right" else 1
    dwell = 0
    grasped_latched = False
    dt = 1.0 / CONTROL_HZ
    n = 0
    t_end = time.time() + args_cli.seconds
    while simulation_app.is_running() and time.time() < t_end:
        t0 = time.perf_counter()
        if flag["recalibrate"]:
            flag["recalibrate"] = False
            ctrl.reset(bridge.robot_state())

        robot_state = bridge.robot_state()
        action = ctrl.step(robot_state)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=env.unwrapped.device).unsqueeze(0)
        env.step(action_t)

        hand_w = robot_state.kp_body_pos_w[hand_idx]
        dist = float(np.linalg.norm(hand_w - cube_world))
        if dist < args_cli.grasp_radius:
            dwell += 1
        else:
            dwell = 0
        newly_grasped = dwell >= args_cli.dwell_ticks
        if newly_grasped and not grasped_latched:
            print(f"[teleop] GRASPED (t={n/CONTROL_HZ:5.1f}s, dist={dist*100:.1f}cm)", flush=True)
        elif grasped_latched and dwell == 0:
            print(f"[teleop] released (t={n/CONTROL_HZ:5.1f}s)", flush=True)
        grasped_latched = newly_grasped

        n += 1
        if n % int(CONTROL_HZ) == 0:
            print(
                f"[{n/CONTROL_HZ:6.1f}s] stale={getattr(source, 'stale', False)} "
                f"{args_cli.grasp_hand}_wrist_to_cube={dist*100:5.1f}cm "
                f"calibrated={getattr(canon, 'calibrated', True)}",
                flush=True,
            )
        time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

    flag["stop"] = True
    source.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
