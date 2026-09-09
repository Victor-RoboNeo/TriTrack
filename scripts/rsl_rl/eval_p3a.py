#!/usr/bin/env python3
"""P3-A oracle hard/soft intent projector. Frozen Stage-2. No terrain IDs."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DT = 0.02
WARMUP = 40
STRIDE = 5
N_AXES = 15
Z_DIM = 16
FAIL_ANCHOR_Z = 0.25
FAIL_ANCHOR_ORI = 0.8
FAIL_FALL_Z = 0.4
CONTACT_N = 10.0
POS_SCALE = 0.05
KP_VIS = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
ANKLE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
MASK_VIS = {
    "torso": (True, False, False),
    "vr": (True, True, True),
    "head_right": (True, False, True),
    "head_left": (True, True, False),
}
LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
WAIST_JOINT_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
PARENT_CKPT = (
    "/data/home/chenxiangyu/robotics/Anybody/logs/rsl_rl/"
    "g1_flat_muse_kp_latent_rl/2026-08-26_00-38-10_tritrack_headhands_locomani_from35000/"
    "model_50000.pt"
)
MAPPER_B = "/data/home/chenxiangyu/victor/TriTrack/runs/mapper_b_intent72/mapper_best.pt"
CLIP_ROOT = "/data/home/chenxiangyu/robotics/Anybody/logs/tritrack/infer_clips/p1"
BASIS_TRAIN_TERRAINS = ("plane", "slope", "slope_down", "light_rough", "steps")
HELD_OUT_TERRAINS = ("slip",)
TASK_SOURCES = ("loco", "stoop", "reach", "carry")
S_ENABLED_TASKS = ("stoop", "reach", "carry")
OUT_DEFAULT = "results/p3_intent_projected_adaptation"
IFS_ROOT = "/data/home/chenxiangyu/robotics/Anybody/results/intent_free_space"
SOFT_LAMBDAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
HARD_LAMBDAS = (1e-3, 1e-2, 1e-1)
EPS_C = 1e-8


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="P3-A oracle hard/soft intent projector. Frozen Stage-2.")
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--task", type=str, default="MUSE-Kp-LatentRL-Kp5-HeadHands-Locomani-G1-v0")
parser.add_argument("--motion", type=str, default=f"{CLIP_ROOT}/loco")
parser.add_argument("--mask_modes", type=str, default="torso")
parser.add_argument("--task_source", type=str, default="loco", choices=TASK_SOURCES)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out", type=str, default=OUT_DEFAULT)
parser.add_argument("--start_frame", type=int, default=10)
parser.add_argument("--hint_iter", type=int, default=50000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
parser.add_argument("--terrain", type=str, default="plane")
parser.add_argument("--terrains", type=str, default="plane")
parser.add_argument("--keep_headhands_spec", action="store_true", default=True)
parser.add_argument("--mapper_path", type=str, default=MAPPER_B)
parser.add_argument("--mapper_b_path", type=str, default=MAPPER_B)
parser.add_argument("--residual_alpha", type=float, default=1.0)
parser.add_argument("--s_enabled", action="store_true", default=False)
parser.add_argument("--no_s_enabled", action="store_true", default=False)
parser.add_argument("--stage", type=str, default="p3a", choices=("p3a",))
parser.add_argument("--ifs_root", type=str, default=IFS_ROOT)
parser.add_argument("--stride", type=int, default=STRIDE)
parser.add_argument("--eps_deg", type=float, default=1.0)
parser.add_argument("--jac_horizon", type=int, default=1)
parser.add_argument("--eval_deg", type=str, default="1,5")
parser.add_argument("--k_max", type=int, default=6)
parser.add_argument("--train_quota", type=int, default=400)
parser.add_argument("--val_quota", type=int, default=100)
parser.add_argument("--test_quota", type=int, default=100)
parser.add_argument("--persist", type=int, default=3)
parser.add_argument("--max_snaps_rec", type=int, default=2)
parser.add_argument("--e_rec", type=float, default=0.13)
parser.add_argument("--ucr_burst", type=int, default=5)
parser.add_argument("--ucr_horizon", type=int, default=25)
parser.add_argument("--ucr_deg", type=float, default=5.0)
parser.add_argument("--lambda_rel", type=float, default=1.0e-2)
parser.add_argument("--random_eval_seeds", type=int, default=3)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False
if bool(getattr(args_cli, "no_s_enabled", False)):
    args_cli.s_enabled = False
elif str(args_cli.task_source) in S_ENABLED_TASKS:
    args_cli.s_enabled = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import yaml  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import whole_body_tracking.tasks  # noqa: E402, F401
from causal_future import CausalFutureInjector  # noqa: E402
from whole_body_tracking.tasks.tracking.mdp.rewards import _ankle_mean_z_offset  # noqa: E402


def project_tangent(d: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z, dim=-1, eps=1e-8)
    d = d - (d * z).sum(-1, keepdim=True) * z
    return F.normalize(d, dim=-1, eps=1e-8)


def tangent_basis(z: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z, dim=-1, eps=1e-8)
    n, d = z.shape
    eye = torch.eye(d, device=z.device, dtype=z.dtype).unsqueeze(0).expand(n, -1, -1).clone()
    eye[:, :, 0] = z
    q, _r = torch.linalg.qr(eye)
    sign = torch.sign((q[:, :, 0] * z).sum(-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign.unsqueeze(-1)
    return q[:, :, 1:]


def geodesic_z(z: torch.Tensor, v: torch.Tensor, eps: float) -> torch.Tensor:
    z = F.normalize(z, dim=-1, eps=1e-8)
    v = project_tangent(v, z)
    c = math.cos(float(eps))
    s = math.sin(float(eps))
    return F.normalize(c * z + s * v, dim=-1, eps=1e-8)


def qr_subspace(P_B: torch.Tensor) -> torch.Tensor:
    """Orthonormalize last dim. P_B: (n, 16, k) -> (n, 16, k)."""
    q, _r = torch.linalg.qr(P_B)
    return q


def tangent_project_basis(z: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """B: (16, k) or (n, 16, k). Return state-wise QR(P_T B)."""
    z = F.normalize(z, dim=-1, eps=1e-8)
    n, d = z.shape
    if B.dim() == 2:
        B = B.unsqueeze(0).expand(n, -1, -1)
    P = torch.eye(d, device=z.device, dtype=z.dtype).unsqueeze(0) - z.unsqueeze(-1) * z.unsqueeze(-2)
    return qr_subspace(P @ B)


def _restore_adapter(agent_cfg, resume_path: str) -> None:
    params = Path(resume_path).parent / "params" / "agent.yaml"
    if not params.exists():
        return
    cfg = yaml.safe_load(params.read_text()) or {}
    policy_cfg = cfg.get("policy") or {}
    for key in (
        "adapter", "lora_rank", "lora_alpha", "lora_targets",
        "residual_d_model", "residual_num_layers", "residual_nhead", "residual_ffn",
        "residual_last_layer_gain", "residual_alpha", "latent_std_min", "latent_std_max",
        "init_latent_std", "terrain_scan_dim", "terrain_r_max", "terrain_scan_zero",
        "terrain_gate", "intent_recovery", "intent_recovery_aux_dim", "intent_recovery_s_enabled",
    ):
        if key in policy_cfg and hasattr(agent_cfg.policy, key):
            setattr(agent_cfg.policy, key, policy_cfg[key])


def _apply_eval_motion(env_cfg, motion: str, start_frame: int) -> None:
    env_cfg.commands.motion.motion = motion
    if hasattr(env_cfg.commands.motion, "motion_groups"):
        env_cfg.commands.motion.motion_groups = None
    if hasattr(env_cfg.commands.motion, "motion_group_sampling_ratios"):
        env_cfg.commands.motion.motion_group_sampling_ratios = None
    if hasattr(env_cfg.commands.motion, "start_frame"):
        env_cfg.commands.motion.start_from_beginning = True
        env_cfg.commands.motion.start_frame = start_frame
    zero = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    if hasattr(env_cfg.commands.motion, "pose_range"):
        env_cfg.commands.motion.pose_range = dict(zero)
    if hasattr(env_cfg.commands.motion, "velocity_range"):
        env_cfg.commands.motion.velocity_range = dict(zero)
    if hasattr(env_cfg.commands.motion, "joint_position_range"):
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    if hasattr(env_cfg.commands.motion, "motion_dataset_shard_across_gpus"):
        env_cfg.commands.motion.motion_dataset_shard_across_gpus = False
    if hasattr(env_cfg.commands.motion, "resample_motions_every_s"):
        env_cfg.commands.motion.resample_motions_every_s = 0.0
    if hasattr(env_cfg.commands.motion, "max_active_motions"):
        env_cfg.commands.motion.max_active_motions = None
    if hasattr(env_cfg.commands.motion, "random_init_frame"):
        env_cfg.commands.motion.random_init_frame = False


def _apply_terrain(env_cfg, terrain: str) -> None:
    mat = getattr(env_cfg.scene.terrain, "physics_material", None)
    if mat is not None:
        mat.static_friction = 1.0
        mat.dynamic_friction = 1.0
    if terrain == "plane":
        env_cfg.scene.terrain.terrain_type = "plane"
        return
    if terrain == "slip":
        env_cfg.scene.terrain.terrain_type = "plane"
        if mat is not None:
            mat.static_friction = 0.25
            mat.dynamic_friction = 0.20
        return
    from isaaclab.terrains import (
        HfInvertedPyramidSlopedTerrainCfg,
        HfPyramidSlopedTerrainCfg,
        HfPyramidStairsTerrainCfg,
        HfRandomUniformTerrainCfg,
        TerrainGeneratorCfg,
    )

    env_cfg.scene.terrain.terrain_type = "generator"
    if terrain == "light_rough":
        subs = {"slightly_rough": HfRandomUniformTerrainCfg(proportion=1.0, noise_range=(0.01, 0.03), noise_step=0.01)}
    elif terrain == "slope":
        subs = {"slope": HfPyramidSlopedTerrainCfg(proportion=1.0, slope_range=(0.087, 0.176), platform_width=2.0)}
    elif terrain == "slope_down":
        subs = {"slope_inv": HfInvertedPyramidSlopedTerrainCfg(proportion=1.0, slope_range=(0.087, 0.176), platform_width=2.0)}
    elif terrain == "steps":
        subs = {
            "stairs": HfPyramidStairsTerrainCfg(
                proportion=1.0, step_height_range=(0.03, 0.08), step_width=0.4, platform_width=2.0
            )
        }
    else:
        raise ValueError(terrain)
    env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=42, size=(8.0, 8.0), border_width=20.0, num_rows=1, num_cols=1,
        horizontal_scale=0.1, vertical_scale=0.005, curriculum=False, sub_terrains=subs,
    )
    env_cfg.scene.terrain.max_init_terrain_level = None
    if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
        if hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None


def _disable_eval_hooks(env_cfg) -> None:
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


def _pin_clips(env, mask_name: str | None = None) -> tuple[int, list[str]]:
    cmd = env.unwrapped.command_manager.get_term("motion")
    if hasattr(cmd, "p_mask"):
        cmd.p_mask = 0.0
    paths = list(cmd.motion_dir_loader.motion_paths)
    n_envs = int(env.unwrapped.num_envs)
    use = min(n_envs, len(paths))
    with torch.inference_mode():
        env.reset()
        ids = torch.arange(use, device=cmd.device, dtype=torch.long)
        cmd.env_motion_indices[:use] = ids
        if hasattr(cmd, "env_motion_groups"):
            cmd.env_motion_groups[:use] = 0
        if hasattr(cmd, "_env_remap_version") and hasattr(cmd, "_remap_version"):
            cmd._env_remap_version[:use] = cmd._remap_version
        if mask_name and hasattr(cmd, "set_eval_fixed_mask_mode_idx"):
            names = [str(x) for x in getattr(cmd, "_mode_names", ())]
            if mask_name in names:
                cmd.set_eval_fixed_mask_mode_idx(names.index(mask_name))
        cmd._resample_command(ids)
        if hasattr(cmd, "resample_all_mask_modes"):
            cmd.resample_all_mask_modes()
        try:
            env.unwrapped.sim.forward()
        except Exception:
            pass
    return use, [str(p) for p in paths[:use]]


def _body_e(cmd, vis_i):
    delta = cmd.robot_body_pos_w - cmd.body_pos_w
    return (
        torch.linalg.norm(delta[:, vis_i[0]], dim=-1),
        torch.linalg.norm(delta[:, vis_i[1]], dim=-1),
        torch.linalg.norm(delta[:, vis_i[2]], dim=-1),
    )


def _visible_e(cmd, vis_i, vis_flags) -> torch.Tensor:
    et, el, er = _body_e(cmd, vis_i)
    acc = torch.zeros_like(et)
    n = torch.zeros_like(et)
    for flag, e in zip(vis_flags, (et, el, er)):
        if flag:
            acc = acc + e
            n = n + 1
    return acc / n.clamp(min=1.0)


@torch.inference_mode()
def _z_and_proprio(policy, runner, obs: torch.Tensor):
    obs_n = runner._normalize_student_obs(obs)
    enc, _scan, feat, mask, _aux = policy._split_policy_obs(obs_n)
    z, proprio = policy._nominal_mean_latent(enc, feat, mask)
    return z, proprio, enc


@torch.inference_mode()
def _decode(policy, z: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
    z = policy.muse._maybe_normalize_latent(z)
    return policy.muse._decode(z, proprio)


def _official_fail(cmd, asset, vis_i, use, pelvis_i, gravity) -> torch.Tensor:
    offset = _ankle_mean_z_offset(cmd)[:use]
    adj_z = (cmd.anchor_pos_w[:use, -1] - cmd.robot_anchor_pos_w[:use, -1]) + offset
    fail_z = adj_z.abs() > FAIL_ANCHOR_Z
    mot_g = quat_rotate_inverse(cmd.anchor_quat_w, gravity)[:use]
    rob_g = quat_rotate_inverse(cmd.robot_anchor_quat_w, gravity)[:use]
    fail_ori = (mot_g[:, 2] - rob_g[:, 2]).abs() > FAIL_ANCHOR_ORI
    torso_z = cmd.robot_body_pos_w[:use, vis_i[0], 2]
    fall = torso_z < FAIL_FALL_Z
    if pelvis_i is not None:
        fall = fall | (asset.data.body_pos_w[:use, pelvis_i, 2] < FAIL_FALL_Z)
    return fall | fail_z | fail_ori


def _hist_snapshot(env) -> dict:
    om = env.unwrapped.observation_manager
    out = {}
    for g, terms in om._group_obs_term_history_buffer.items():
        out[g] = {}
        for name, buf in terms.items():
            if buf._buffer is None:
                out[g][name] = {"chron": None, "num_pushes": buf._num_pushes.detach().cpu().clone()}
            else:
                out[g][name] = {
                    "chron": buf.buffer.detach().cpu().clone(),
                    "num_pushes": buf._num_pushes.detach().cpu().clone(),
                }
    return out


def _slice_hist(hist: dict, env_id: int) -> dict:
    out = {}
    for g, terms in hist.items():
        out[g] = {}
        for name, rec in terms.items():
            chron = rec["chron"]
            npsh = rec["num_pushes"]
            out[g][name] = {
                "chron": None if chron is None else chron[env_id : env_id + 1].clone(),
                "num_pushes": npsh[env_id : env_id + 1].clone(),
            }
    return out


def _hist_restore(env, snaps_hist: list[dict], n: int) -> None:
    om = env.unwrapped.observation_manager
    device = env.unwrapped.device
    for g, terms in om._group_obs_term_history_buffer.items():
        for name, buf in terms.items():
            chron0 = snaps_hist[0][g][name]["chron"]
            if chron0 is None:
                continue
            max_len = int(chron0.shape[1])
            feat = chron0.shape[2:]
            if buf._buffer is None:
                buf._buffer = torch.zeros((max_len, buf.batch_size, *feat), device=device, dtype=chron0.dtype)
            buf._pointer = max_len - 1
            for i in range(n):
                chron = snaps_hist[i][g][name]["chron"].to(device=device)
                frame = chron[0] if chron.shape[0] == 1 else chron[i]
                buf._buffer[:, i] = frame
                npsh = snaps_hist[i][g][name]["num_pushes"]
                buf._num_pushes[i] = int(npsh.reshape(-1)[0].item())


def _capture_snap(env, env_id: int, hist, extra: dict) -> dict:
    uw = env.unwrapped
    robot = uw.scene["robot"]
    cmd = uw.command_manager.get_term("motion")
    am = uw.action_manager
    origin = uw.scene.env_origins[env_id].detach().cpu().clone()
    root = robot.data.root_state_w[env_id].detach().cpu().clone()
    root_local = root.clone()
    root_local[0] -= origin[0]
    root_local[1] -= origin[1]
    snap = {
        "env_id": int(env_id),
        "origin": origin,
        "root_local": root_local,
        "joint_pos": robot.data.joint_pos[env_id].detach().cpu().clone(),
        "joint_vel": robot.data.joint_vel[env_id].detach().cpu().clone(),
        "time_steps": int(cmd.time_steps[env_id].item()),
        "motion_idx": int(cmd.env_motion_indices[env_id].item()),
        "action": am._action[env_id].detach().cpu().clone(),
        "prev_action": am._prev_action[env_id].detach().cpu().clone(),
        "hist": _slice_hist(hist, env_id),
        "episode_length": int(uw.episode_length_buf[env_id].item()),
    }
    if hasattr(cmd, "env_motion_groups"):
        snap["motion_group"] = int(cmd.env_motion_groups[env_id].item())
    snap.update(extra)
    return snap


def _restore_batch_inner(env, snaps: list[dict], n: int) -> int:
    uw = env.unwrapped
    robot = uw.scene["robot"]
    cmd = uw.command_manager.get_term("motion")
    am = uw.action_manager
    device = uw.device
    ids = torch.arange(n, device=device, dtype=torch.long)
    roots = []
    for i, s in enumerate(snaps):
        root = s["root_local"].to(device=device)
        origin = uw.scene.env_origins[i]
        root = root.clone()
        root[0] += origin[0]
        root[1] += origin[1]
        roots.append(root)
    robot.write_root_state_to_sim(torch.stack(roots, dim=0), env_ids=ids)
    robot.write_joint_state_to_sim(
        torch.stack([s["joint_pos"].to(device) for s in snaps], dim=0),
        torch.stack([s["joint_vel"].to(device) for s in snaps], dim=0),
        env_ids=ids,
    )
    cmd.time_steps[ids] = torch.tensor([s["time_steps"] for s in snaps], device=device, dtype=cmd.time_steps.dtype)
    cmd.env_motion_indices[ids] = torch.tensor(
        [s["motion_idx"] for s in snaps], device=device, dtype=cmd.env_motion_indices.dtype
    )
    if hasattr(cmd, "env_motion_groups") and "motion_group" in snaps[0]:
        cmd.env_motion_groups[ids] = torch.tensor(
            [s["motion_group"] for s in snaps], device=device, dtype=cmd.env_motion_groups.dtype
        )
    am._action[ids] = torch.stack([s["action"].to(device) for s in snaps], dim=0)
    am._prev_action[ids] = torch.stack([s["prev_action"].to(device) for s in snaps], dim=0)
    uw.episode_length_buf[ids] = torch.tensor(
        [s["episode_length"] for s in snaps], device=device, dtype=uw.episode_length_buf.dtype
    )
    _hist_restore(env, [s["hist"] for s in snaps], n)
    uw.scene.write_data_to_sim()
    uw.sim.forward()
    return n


def _restore_batch(env, snaps: list[dict]) -> int:
    n = len(snaps)
    with torch.inference_mode():
        return _restore_batch_inner(env, snaps, n)


def _pad_snaps(snaps: list[dict], n_envs: int) -> tuple[list[dict], int]:
    if not snaps:
        return snaps, 0
    n = len(snaps)
    if n >= n_envs:
        return snaps[:n_envs], n_envs
    padded = list(snaps) + [snaps[-1]] * (n_envs - n)
    return padded, n


def _make_env(env_cfg, agent_cfg, resume_path: str, motion_dir: str, terrain: str):
    n_files = len(list(Path(motion_dir).rglob("*.npz")))
    n_want = int(args_cli.num_envs) if args_cli.num_envs and int(args_cli.num_envs) > 0 else n_files
    env_cfg.scene.num_envs = min(n_want, n_files)
    env_cfg.seed = int(args_cli.seed)
    agent_cfg.seed = int(args_cli.seed)
    _apply_eval_motion(env_cfg, motion_dir, args_cli.start_frame)
    _apply_terrain(env_cfg, terrain)
    _disable_eval_hooks(env_cfg)
    from whole_body_tracking.tasks.tracking.config.g1.mask_modes import eval_single_mode_spec

    spec, probs = eval_single_mode_spec("torso")
    if not bool(args_cli.keep_headhands_spec):
        env_cfg.commands.motion.mask_mode_spec = spec
        env_cfg.commands.motion.mask_mode_probs = probs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    from whole_body_tracking.tasks.tracking.mdp.curriculums import attach_curriculum_rollout_hints

    attach_curriculum_rollout_hints(
        env, spe=int(agent_cfg.num_steps_per_env), learning_iteration=int(args_cli.hint_iter)
    )
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False, load_critic=True)
    policy = runner.alg.policy
    policy.eval()
    rc = getattr(policy, "residual_corrector", None)
    if rc is not None:
        rc.alpha = float(args_cli.residual_alpha)
    mapper_paths = {"mapper": args_cli.mapper_path, "mapper_b": args_cli.mapper_b_path or args_cli.mapper_path}
    injector = CausalFutureInjector(mappers=mapper_paths, device=env.device)
    return env, runner, policy, injector


def _named_joint_idx(joint_names: list[str], wanted: tuple[str, ...]) -> list[int]:
    out = []
    for name in wanted:
        if name in joint_names:
            out.append(joint_names.index(name))
        else:
            print(f"[ifs] missing joint {name}", flush=True)
    return out


def _feature_layout(joint_names: list[str]) -> dict:
    leg_i = _named_joint_idx(joint_names, LEG_JOINT_NAMES)
    waist_i = _named_joint_idx(joint_names, WAIST_JOINT_NAMES)
    names = []
    for i in leg_i:
        names.append(f"leg_q:{joint_names[i]}")
    for i in leg_i:
        names.append(f"leg_qd:{joint_names[i]}")
    for i in waist_i:
        names.append(f"waist_q:{joint_names[i]}")
    for i in waist_i:
        names.append(f"waist_qd:{joint_names[i]}")
    names.extend(["root_lin_x", "root_lin_y", "root_lin_z", "root_ang_x", "root_ang_y", "root_ang_z"])
    names.extend(["pelvis_height", "pelvis_roll", "pelvis_pitch"])
    return {"leg_i": leg_i, "waist_i": waist_i, "names": names, "dim": len(names)}


@torch.inference_mode()
def _e_intent_vec(cmd, vis_i, vis_flags) -> torch.Tensor:
    chunks = []
    for k, flag in enumerate(vis_flags):
        if not flag:
            continue
        chunks.append(cmd.robot_body_pos_w[:, vis_i[k]] - cmd.body_pos_w[:, vis_i[k]])
    if not chunks:
        return cmd.robot_body_pos_w[:, vis_i[0]] * 0.0
    return torch.cat(chunks, dim=-1)


def _y_intent(cmd, vis_i, vis_flags, origins) -> torch.Tensor:
    """FK of currently controlled sparse points, env-local xy."""
    chunks = []
    for k, flag in enumerate(vis_flags):
        if not flag:
            continue
        p = cmd.robot_body_pos_w[:, vis_i[k]].clone()
        p[:, 0] = p[:, 0] - origins[:, 0]
        p[:, 1] = p[:, 1] - origins[:, 1]
        chunks.append(p)
    if not chunks:
        return cmd.robot_body_pos_w[:, vis_i[0]] * 0.0
    return torch.cat(chunks, dim=-1)


@torch.inference_mode()
def _y_body(asset, cmd, layout, pelvis_i, ankle_i, n: int) -> torch.Tensor:
    jp = asset.data.joint_pos[:n]
    jv = asset.data.joint_vel[:n]
    parts = [
        jp[:, layout["leg_i"]],
        jv[:, layout["leg_i"]],
        jp[:, layout["waist_i"]],
        jv[:, layout["waist_i"]],
        asset.data.root_lin_vel_b[:n],
        asset.data.root_ang_vel_b[:n],
    ]
    root_z = asset.data.root_pos_w[:n, 2]
    if ankle_i:
        pel_h = root_z - asset.data.body_pos_w[:n][:, ankle_i, 2].mean(dim=-1)
    else:
        pel_h = root_z
    roll, pitch, _yaw = euler_xyz_from_quat(asset.data.root_quat_w[:n])
    parts.extend([pel_h.unsqueeze(-1), roll.unsqueeze(-1), pitch.unsqueeze(-1)])
    return torch.cat(parts, dim=-1)


@torch.inference_mode()
def _contact2(env, ankle_i, n: int) -> torch.Tensor:
    device = env.unwrapped.device
    out = torch.zeros(n, 2, device=device)
    try:
        cf = env.unwrapped.scene["contact_forces"]
    except Exception:
        return out
    if not ankle_i or not hasattr(cf.data, "net_forces_w"):
        return out
    nf = cf.data.net_forces_w[:n]
    mag = nf[:, ankle_i, :].norm(dim=-1)
    out[:, : mag.shape[1]] = mag
    return out


@torch.inference_mode()
def _collect_snaps(env, policy, runner, injector, steps, seed, terrain, vis_i, vis_flags, mask_name, layout):
    n_envs = int(env.unwrapped.num_envs)
    use, paths = _pin_clips(env, mask_name=mask_name)
    cmd = env.unwrapped.command_manager.get_term("motion")
    device = env.unwrapped.device
    asset = env.unwrapped.scene["robot"]
    names = list(asset.data.body_names)
    pelvis_i = names.index("pelvis") if "pelvis" in names else None
    ankle_i = [names.index(n) for n in ANKLE_BODIES if n in names]
    origins = env.unwrapped.scene.env_origins
    e_vis_prev = None
    n_rec = [0] * use
    last_snap_t = [-999] * use
    persist_n = [0] * use
    snaps = []
    stride = max(1, int(args_cli.stride))
    for t in range(steps):
        obs, _ = env.get_observations()
        obs = injector.patch_policy_obs(obs, env, "mapper")
        z_nom, proprio, _enc = _z_and_proprio(policy, runner, obs)
        e_vis = _visible_e(cmd, vis_i, vis_flags)[:use]
        de_s = torch.zeros_like(e_vis) if e_vis_prev is None else (e_vis - e_vis_prev)
        if t >= WARMUP:
            hist_now = _hist_snapshot(env)
            yI = _y_intent(cmd, vis_i, vis_flags, origins)[:use]
            yB = _y_body(asset, cmd, layout, pelvis_i, ankle_i, use)
            c2 = _contact2(env, ankle_i, use)
            for i in range(use):
                if float(e_vis[i]) >= float(args_cli.e_rec):
                    persist_n[i] += 1
                else:
                    persist_n[i] = 0
                take_nom = ((t - WARMUP) % stride) == 0
                rec = persist_n[i] >= max(1, int(args_cli.persist)) and float(de_s[i]) > 0
                kind = None
                if rec and n_rec[i] < int(args_cli.max_snaps_rec) and (t - last_snap_t[i]) >= 40:
                    kind = "recovery"
                    n_rec[i] += 1
                    last_snap_t[i] = t
                elif take_nom:
                    kind = "nominal"
                if kind is None:
                    continue
                extra = {
                    "obs": obs[i].detach().cpu().clone(),
                    "z_nom": z_nom[i].detach().cpu().clone(),
                    "y_I": yI[i].detach().cpu().clone(),
                    "y_B": yB[i].detach().cpu().clone(),
                    "contact": c2[i].detach().cpu().clone(),
                    "e0": float(e_vis[i].item()),
                    "de0": float(de_s[i].item()),
                    "t": int(t),
                    "seed": int(seed),
                    "terrain": terrain,
                    "window": kind,
                    "clip": Path(paths[i]).name if i < len(paths) else f"env{i}",
                    "task": str(args_cli.task_source),
                    "episode_id": f"{terrain}|{seed}|{Path(paths[i]).name if i < len(paths) else i}|{i}",
                }
                snaps.append(_capture_snap(env, i, hist_now, extra))
        env.step(_decode(policy, z_nom, proprio))
        e_vis_prev = e_vis.clone()
        if (t + 1) % 50 == 0:
            n_nom = sum(1 for s in snaps if s["window"] == "nominal")
            n_r = sum(1 for s in snaps if s["window"] == "recovery")
            print(
                f"[ifs] walk {terrain} s{seed} {t+1}/{steps} e={float(e_vis.mean()):.3f} "
                f"nom={n_nom} rec={n_r}",
                flush=True,
            )
    print(f"[ifs] collected {len(snaps)} snaps terrain={terrain} seed={seed}", flush=True)
    return snaps, pelvis_i, ankle_i, use, paths


def _episode_split(episode_ids: list[str], held_out: bool, seed: int = 2026) -> dict[str, str]:
    uniq = sorted(set(episode_ids))
    if held_out:
        return {k: "test" for k in uniq}
    rng = np.random.RandomState(seed)
    order = uniq[:]
    rng.shuffle(order)
    n = len(order)
    n_tr = int(round(0.60 * n))
    n_va = int(round(0.20 * n))
    out = {}
    for i, k in enumerate(order):
        if i < n_tr:
            out[k] = "train"
        elif i < n_tr + n_va:
            out[k] = "val"
        else:
            out[k] = "test"
    return out


def _cap_by_split(snaps: list[dict], split_map: dict[str, str], quotas: dict[str, int], seed: int) -> list[dict]:
    rng = np.random.RandomState(seed)
    grouped: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for s in snaps:
        if s["window"] != "nominal":
            continue
        sp = split_map.get(s["episode_id"], "test")
        s = dict(s)
        s["split"] = sp
        grouped[sp].append(s)
    out = []
    for sp, quota in quotas.items():
        pool = grouped[sp]
        if len(pool) > quota:
            idx = rng.choice(len(pool), quota, replace=False)
            idx.sort()
            pool = [pool[i] for i in idx]
        out.extend(pool)
    rec = []
    for s in snaps:
        if s["window"] == "recovery":
            s2 = dict(s)
            s2["split"] = "recovery"
            rec.append(s2)
    out.extend(rec)
    return out


@torch.inference_mode()
def _roll_features(
    env,
    policy,
    runner,
    injector,
    snaps,
    n_valid,
    v,
    eps,
    horizon,
    vis_i,
    vis_flags,
    layout,
    pelvis_i,
    ankle_i,
    burst: int | None = None,
    record_e: bool = False,
):
    n_envs = int(env.unwrapped.num_envs)
    padded, _ = _pad_snaps(snaps, n_envs)
    _restore_batch(env, padded)
    cmd = env.unwrapped.command_manager.get_term("motion")
    asset = env.unwrapped.scene["robot"]
    origins = env.unwrapped.scene.env_origins
    device = env.unwrapped.device
    v_pad = None
    if v is not None:
        v_pad = torch.zeros(n_envs, Z_DIM, device=device, dtype=torch.float32)
        v_pad[:n_valid] = v
    e_hist = []
    evec_hist = []
    yI = yB = None
    obs_stack = torch.stack([s["obs"].to(device) for s in padded], dim=0)
    e0_vec = _e_intent_vec(cmd, vis_i, vis_flags).detach()
    e0_s = _visible_e(cmd, vis_i, vis_flags).detach()
    yB0 = _y_body(asset, cmd, layout, pelvis_i, ankle_i, n_envs)
    yI0 = _y_intent(cmd, vis_i, vis_flags, origins)
    for t in range(horizon):
        obs = obs_stack if t == 0 else injector.patch_policy_obs(env.get_observations()[0], env, "mapper")
        z_nom, proprio, _enc = _z_and_proprio(policy, runner, obs)
        apply = v_pad is not None and (burst is None or t < int(burst))
        z_exec = geodesic_z(z_nom, v_pad, eps) if apply else z_nom
        env.step(_decode(policy, z_exec, proprio))
        yI = _y_intent(cmd, vis_i, vis_flags, origins)
        yB = _y_body(asset, cmd, layout, pelvis_i, ankle_i, n_envs)
        if record_e:
            e_hist.append(_visible_e(cmd, vis_i, vis_flags).detach())
            evec_hist.append(_e_intent_vec(cmd, vis_i, vis_flags).detach())
    ei = torch.stack(e_hist, dim=-1) if e_hist else None
    ev = torch.stack(evec_hist, dim=1) if evec_hist else None
    extra = {
        "e0_vec": e0_vec[:n_valid],
        "e0": e0_s[:n_valid],
        "yI0": yI0[:n_valid],
        "yB0": yB0[:n_valid],
        "evec": ev[:n_valid] if ev is not None else None,
    }
    return yI[:n_valid].detach(), yB[:n_valid].detach(), ei[:n_valid].detach() if ei is not None else None, extra


@torch.inference_mode()
def _compute_jacobians(env, policy, runner, injector, snaps, vis_i, vis_flags, layout, pelvis_i, ankle_i):
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    eps = math.radians(float(args_cli.eps_deg))
    L = max(1, int(args_cli.jac_horizon))
    rows = []
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[ifs] jac {start}:{start + n_valid}/{len(snaps)} L={L} eps={args_cli.eps_deg}deg", flush=True)
        z0 = torch.stack([s["z_nom"] for s in batch], dim=0).to(device)
        V = tangent_basis(z0)
        ny = int(batch[0]["y_I"].shape[0])
        nb = int(batch[0]["y_B"].shape[0])
        J_tan_I = torch.zeros(n_valid, ny, N_AXES, device=device)
        J_tan_B = torch.zeros(n_valid, nb, N_AXES, device=device)
        for ax in range(N_AXES):
            v = V[:, :, ax]
            yIp, yBp, _, _ = _roll_features(
                env, policy, runner, injector, batch, n_valid, v, eps, L,
                vis_i, vis_flags, layout, pelvis_i, ankle_i,
            )
            yIm, yBm, _, _ = _roll_features(
                env, policy, runner, injector, batch, n_valid, v, -eps, L,
                vis_i, vis_flags, layout, pelvis_i, ankle_i,
            )
            J_tan_I[:, :, ax] = (yIp - yIm) / (2.0 * eps)
            J_tan_B[:, :, ax] = (yBp - yBm) / (2.0 * eps)
            if ax == 0 or (ax + 1) % 5 == 0:
                print(f"[ifs]   axis {ax + 1}/{N_AXES}", flush=True)
        J_I = torch.bmm(J_tan_I, V.transpose(1, 2))
        J_B = torch.bmm(J_tan_B, V.transpose(1, 2))
        for i in range(n_valid):
            s = batch[i]
            rows.append(
                {
                    "z0": z0[i].detach().cpu().numpy().astype(np.float32),
                    "J_I": J_I[i].detach().cpu().numpy().astype(np.float32),
                    "J_B": J_B[i].detach().cpu().numpy().astype(np.float32),
                    "y_I": s["y_I"].numpy().astype(np.float32),
                    "y_B": s["y_B"].numpy().astype(np.float32),
                    "contact": s["contact"].numpy().astype(np.float32),
                    "episode_id": s["episode_id"],
                    "split": s.get("split", "test"),
                    "window": s["window"],
                    "terrain": s["terrain"],
                    "seed": int(s["seed"]),
                    "t": int(s["t"]),
                    "clip": s["clip"],
                    "e0": float(s["e0"]),
                    "env_id": int(s["env_id"]),
                }
            )
    return rows


def _dump_jac(path: Path, rows: list[dict], layout: dict, terrain: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        z0=np.stack([r["z0"] for r in rows]),
        J_I=np.stack([r["J_I"] for r in rows]),
        J_B=np.stack([r["J_B"] for r in rows]),
        y_I=np.stack([r["y_I"] for r in rows]),
        y_B=np.stack([r["y_B"] for r in rows]),
        contact=np.stack([r["contact"] for r in rows]),
        episode_id=np.asarray([r["episode_id"] for r in rows]),
        split=np.asarray([r["split"] for r in rows]),
        window=np.asarray([r["window"] for r in rows]),
        terrain=np.asarray([r["terrain"] for r in rows]),
        seed=np.asarray([r["seed"] for r in rows], dtype=np.int32),
        t=np.asarray([r["t"] for r in rows], dtype=np.int32),
        clip=np.asarray([r["clip"] for r in rows]),
        e0=np.asarray([r["e0"] for r in rows], dtype=np.float32),
        env_id=np.asarray([r["env_id"] for r in rows], dtype=np.int32),
        body_names=np.asarray(layout["names"]),
        terrain_tag=np.asarray(terrain),
        eps_deg=np.asarray(float(args_cli.eps_deg)),
        jac_horizon=np.asarray(int(args_cli.jac_horizon)),
        pos_scale=np.asarray(POS_SCALE),
        no_terrain_in_model=np.asarray(True),
    )


def _intent_projector(J_bar: torch.Tensor, lam_rel: float) -> torch.Tensor:
    """J_bar: (n, ny, 16) -> P_I (n, 16, 16). λ_I = lam_rel * mean(diag(JJ^T))."""
    ny = J_bar.shape[1]
    jjt = torch.bmm(J_bar, J_bar.transpose(1, 2))
    scale = jjt.diagonal(dim1=-2, dim2=-1).mean(-1).clamp(min=1e-8)
    lam = (float(lam_rel) * scale).view(-1, 1, 1)
    eye = torch.eye(ny, device=J_bar.device, dtype=J_bar.dtype).unsqueeze(0)
    inv = torch.linalg.solve(jjt + lam * eye, J_bar)
    return torch.eye(Z_DIM, device=J_bar.device, dtype=J_bar.dtype).unsqueeze(0) - torch.bmm(J_bar.transpose(1, 2), inv)


def _np_gep(C_B: np.ndarray, C_I: np.ndarray, eta_rel: float, k: int) -> np.ndarray:
    d = C_I.shape[0]
    eta = float(eta_rel) * float(np.trace(C_I)) / float(d)
    A = 0.5 * (C_I + C_I.T) + eta * np.eye(d)
    w, Q = np.linalg.eigh(A)
    Aminv_h = Q * (1.0 / np.sqrt(np.maximum(w, 1e-12)))
    M = Aminv_h.T @ (0.5 * (C_B + C_B.T)) @ Aminv_h
    evals, U = np.linalg.eigh(0.5 * (M + M.T))
    evecs = Aminv_h @ U
    order = np.argsort(evals)[::-1]
    return evecs[:, order[:k]].astype(np.float32)


def _local_free(J_I_bar: torch.Tensor, J_B_bar: torch.Tensor, k: int, eta_rel: float = 1e-3) -> torch.Tensor:
    """Per-state GEP. Returns (n, 16, k)."""
    n = J_I_bar.shape[0]
    cols = []
    for i in range(n):
        JI = J_I_bar[i].detach().cpu().numpy()
        JB = J_B_bar[i].detach().cpu().numpy()
        cols.append(_np_gep(JB.T @ JB, JI.T @ JI, eta_rel, k))
    return torch.as_tensor(np.stack(cols, axis=0), device=J_I_bar.device, dtype=J_I_bar.dtype)


@torch.inference_mode()
def _eval_spaces(env, policy, runner, injector, snaps, vis_i, vis_flags, layout, pelvis_i, ankle_i, bases, J_map, scales):
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    eval_degs = [float(x) for x in str(args_cli.eval_deg).split(",") if x.strip()]
    k_max = int(args_cli.k_max)
    mu_b = torch.as_tensor(scales["mu_B"], device=device, dtype=torch.float32)
    sig_b = torch.as_tensor(scales["sig_B"], device=device, dtype=torch.float32).clamp(min=1e-8)
    pos_scale = float(scales.get("pos_scale", POS_SCALE))
    rows = []
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[ifs] eval spaces {start}:{start + n_valid}/{len(snaps)}", flush=True)
        z0 = torch.stack([s["z_nom"] for s in batch], dim=0).to(device)
        J_I = []
        J_B = []
        have_j = True
        for s in batch:
            key = (s["episode_id"], int(s["t"]), s["window"])
            if key not in J_map:
                have_j = False
                break
            J_I.append(J_map[key]["J_I"])
            J_B.append(J_map[key]["J_B"])
        J_I_t = torch.as_tensor(np.stack(J_I), device=device) if have_j else None
        J_B_t = torch.as_tensor(np.stack(J_B), device=device) if have_j else None
        J_I_bar = J_I_t / pos_scale if J_I_t is not None else None
        J_B_bar = None
        if J_B_t is not None:
            J_B_bar = J_B_t / sig_b.view(1, -1, 1)
        rec = {name: {} for name in bases}
        for name, Bglob in bases.items():
            if Bglob is None and name != "S4_local":
                continue
            if name == "S3_PIUCR":
                if J_I_bar is None:
                    continue
                P_I = _intent_projector(J_I_bar, float(args_cli.lambda_rel))
                Buse = P_I @ Bglob.unsqueeze(0).expand(n_valid, -1, -1)
                Bt = tangent_project_basis(z0, Buse)
            elif name == "S4_local":
                if J_I_bar is None or J_B_bar is None:
                    continue
                Buse = _local_free(J_I_bar, J_B_bar, k_max)
                Bt = tangent_project_basis(z0, Buse)
            else:
                Bt = tangent_project_basis(z0, Bglob)
            k_use = min(k_max, int(Bt.shape[-1]))
            rec[name]["k"] = k_use
            rec[name]["DI"] = {}
            rec[name]["DB"] = {}
            for deg in eval_degs:
                eps = math.radians(deg)
                di = torch.zeros(n_valid, k_use, device=device)
                db = torch.zeros(n_valid, k_use, device=device)
                for ki in range(k_use):
                    v = Bt[:, :, ki]
                    yIp, yBp, _, _ = _roll_features(
                        env, policy, runner, injector, batch, n_valid, v, eps, 1,
                        vis_i, vis_flags, layout, pelvis_i, ankle_i,
                    )
                    yIm, yBm, _, _ = _roll_features(
                        env, policy, runner, injector, batch, n_valid, v, -eps, 1,
                        vis_i, vis_flags, layout, pelvis_i, ankle_i,
                    )
                    dI = (yIp - yIm) / 2.0
                    dB = (yBp - yBm) / 2.0
                    di[:, ki] = torch.linalg.norm(dI / pos_scale, dim=-1)
                    db[:, ki] = torch.linalg.norm((dB) / sig_b.view(1, -1), dim=-1)
                rec[name]["DI"][str(deg)] = di.detach().cpu().numpy().astype(np.float32)
                rec[name]["DB"][str(deg)] = db.detach().cpu().numpy().astype(np.float32)
            print(f"[ifs]   space {name}", flush=True)
        for i in range(n_valid):
            s = batch[i]
            item = {
                "episode_id": s["episode_id"],
                "split": s.get("split", "test"),
                "window": s["window"],
                "terrain": s["terrain"],
                "seed": int(s["seed"]),
                "t": int(s["t"]),
                "clip": s["clip"],
                "e0": float(s["e0"]),
                "spaces": {},
            }
            for name, payload in rec.items():
                if "DI" not in payload:
                    continue
                item["spaces"][name] = {
                    "DI": {k: v[i].tolist() for k, v in payload["DI"].items()},
                    "DB": {k: v[i].tolist() for k, v in payload["DB"].items()},
                }
            rows.append(item)
    return rows


@torch.inference_mode()
def _eval_ucr_proj(env, policy, runner, injector, snaps, vis_i, vis_flags, layout, pelvis_i, ankle_i, bases, J_map, scales):
    if not snaps:
        return [], None
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    burst = int(args_cli.ucr_burst)
    horizon = int(args_cli.ucr_horizon)
    deg = float(args_cli.ucr_deg)
    eps = math.radians(deg)
    pos_scale = float(scales.get("pos_scale", POS_SCALE))
    k_max = int(args_cli.k_max)
    rows = []
    e_curves = {k: [] for k in ["parent", "full_ucr"] + list(bases.keys())}
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[ifs] ucr-proj {start}:{start + n_valid}/{len(snaps)} burst={burst} H={horizon}", flush=True)
        z0 = torch.stack([s["z_nom"] for s in batch], dim=0).to(device)
        V = tangent_basis(z0)
        _, _, e_parent, _ = _roll_features(
            env, policy, runner, injector, batch, n_valid, None, 0.0, horizon,
            vis_i, vis_flags, layout, pelvis_i, ankle_i, burst=None, record_e=True,
        )
        a_axes = torch.zeros(n_valid, N_AXES, 2, device=device)
        for ax in range(N_AXES):
            for si, sgn in enumerate((1.0, -1.0)):
                _, _, eh, _ = _roll_features(
                    env, policy, runner, injector, batch, n_valid, V[:, :, ax], float(sgn) * eps, horizon,
                    vis_i, vis_flags, layout, pelvis_i, ankle_i, burst=burst, record_e=True,
                )
                a_axes[:, ax, si] = e_parent[:, -1] - eh[:, -1]
        best_a, best_flat = a_axes.reshape(n_valid, -1).max(dim=-1)
        best_ax = best_flat // 2
        best_sg = best_flat % 2
        d_star = torch.zeros(n_valid, Z_DIM, device=device)
        e_full = torch.zeros_like(e_parent)
        for i in range(n_valid):
            sgn = 1.0 if int(best_sg[i]) == 0 else -1.0
            d_star[i] = sgn * V[i, :, int(best_ax[i])]
        _, _, e_full, _ = _roll_features(
            env, policy, runner, injector, batch, n_valid, d_star, eps, horizon,
            vis_i, vis_flags, layout, pelvis_i, ankle_i, burst=burst, record_e=True,
        )
        e_curves["parent"].append(e_parent.detach().cpu().numpy())
        e_curves["full_ucr"].append(e_full.detach().cpu().numpy())
        A_full = (e_parent[:, -1] - e_full[:, -1]).detach()
        proj_e = {}
        for name, Bglob in bases.items():
            if name == "S4_local":
                continue
            if name == "S3_PIUCR":
                J_I = []
                ok = True
                for s in batch:
                    key = (s["episode_id"], int(s["t"]), s["window"])
                    if key not in J_map:
                        ok = False
                        break
                    J_I.append(J_map[key]["J_I"])
                if not ok:
                    continue
                J_I_bar = torch.as_tensor(np.stack(J_I), device=device) / pos_scale
                P_I = _intent_projector(J_I_bar, float(args_cli.lambda_rel))
                Buse = P_I @ Bglob.unsqueeze(0).expand(n_valid, -1, -1)
                Bt = tangent_project_basis(z0, Buse)
            else:
                Bt = tangent_project_basis(z0, Bglob)
            k_use = min(k_max, int(Bt.shape[-1]))
            Bt = Bt[:, :, :k_use]
            d_tan = project_tangent(d_star, z0)
            coef = torch.bmm(Bt.transpose(1, 2), d_tan.unsqueeze(-1)).squeeze(-1)
            d_s = torch.bmm(Bt, coef.unsqueeze(-1)).squeeze(-1)
            d_s = project_tangent(d_s, z0)
            _, _, eh, _ = _roll_features(
                env, policy, runner, injector, batch, n_valid, d_s, eps, horizon,
                vis_i, vis_flags, layout, pelvis_i, ankle_i, burst=burst, record_e=True,
            )
            proj_e[name] = eh
            e_curves[name].append(eh.detach().cpu().numpy())
            print(f"[ifs]   ucr {name}", flush=True)
        for i in range(n_valid):
            s = batch[i]
            item = {
                "episode_id": s["episode_id"],
                "terrain": s["terrain"],
                "seed": int(s["seed"]),
                "t": int(s["t"]),
                "clip": s["clip"],
                "e0": float(s["e0"]),
                "A_full": float(A_full[i].item()),
                "e_parent": float(e_parent[i, -1].item()),
                "e_full": float(e_full[i, -1].item()),
                "A_proj": {},
                "R_A": {},
                "e_proj": {},
            }
            for name, eh in proj_e.items():
                A = float((e_parent[i, -1] - eh[i, -1]).item())
                item["A_proj"][name] = A
                item["e_proj"][name] = float(eh[i, -1].item())
                item["R_A"][name] = A / (float(A_full[i].item()) + 1e-6)
            rows.append(item)
    stacked = {}
    for k, chunks in e_curves.items():
        if chunks:
            stacked[k] = np.concatenate(chunks, axis=0)
    return rows, stacked


def _load_jac_map(path: Path) -> dict:
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=True)
    out = {}
    for i in range(int(z["z0"].shape[0])):
        key = (str(z["episode_id"][i]), int(z["t"][i]), str(z["window"][i]))
        out[key] = {"J_I": z["J_I"][i], "J_B": z["J_B"][i], "z0": z["z0"][i]}
    return out


def _load_bases(out_root: Path, k_max: int, device) -> dict:
    bdir = out_root / "basis"
    bases = {}
    bf = bdir / "B_free.npz"
    bu = bdir / "B_ucr.npz"
    br = bdir / "random_bases.npz"
    if bf.exists():
        B = np.load(bf)["B"]
        bases["S2_free"] = torch.as_tensor(B[:, :k_max], device=device, dtype=torch.float32)
    if bu.exists():
        B = np.load(bu)["B"]
        bases["S1_UCR"] = torch.as_tensor(B[:, :k_max], device=device, dtype=torch.float32)
        bases["S3_PIUCR"] = bases["S1_UCR"]
    if br.exists():
        R = np.load(br)["B"]
        bases["S0_random"] = torch.as_tensor(R[0, :, :k_max], device=device, dtype=torch.float32)
    bases["S4_local"] = None
    return bases


def _load_scales(out_root: Path) -> dict:
    p = out_root / "basis" / "body_scale.npz"
    if p.exists():
        z = np.load(p)
        return {"mu_B": z["mu"], "sig_B": z["scale"], "pos_scale": float(z["pos_scale"]) if "pos_scale" in z.files else POS_SCALE}
    return {"mu_B": np.zeros(1, dtype=np.float32), "sig_B": np.ones(1, dtype=np.float32), "pos_scale": POS_SCALE}


def _run_collect_jac(env_cfg, agent_cfg, resume_path, motion_dir, terrain, out_root: Path):
    cell = out_root / "dataset_a" / terrain
    cell.mkdir(parents=True, exist_ok=True)
    jac_p = cell / "jac.npz"
    meta_p = cell / "meta.json"
    snaps_p = cell / "snaps_eval.pt"
    if jac_p.exists() and snaps_p.exists():
        print(f"[ifs] skip collect_jac {terrain}", flush=True)
        return json.loads(meta_p.read_text()) if meta_p.exists() else {}
    env, runner, policy, injector = _make_env(env_cfg, agent_cfg, resume_path, motion_dir, terrain)
    cmd = env.unwrapped.command_manager.get_term("motion")
    body_names = list(cmd.cfg.body_names)
    vis_i = [body_names.index(n) for n in KP_VIS]
    mask_name = str(args_cli.mask_modes).split(",")[0].strip() or "torso"
    vis_flags = MASK_VIS.get(mask_name, (True, True, True))
    asset = env.unwrapped.scene["robot"]
    layout = _feature_layout(list(asset.data.joint_names))
    seeds = [int(s) for s in args_cli.seeds.split(",") if s.strip()] or [int(args_cli.seed)]
    all_snaps = []
    pelvis_i = ankle_i = None
    for seed in seeds:
        print(f"[ifs] ===== collect {args_cli.task_source} {terrain} seed={seed} =====", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        snaps, pelvis_i, ankle_i, _use, _paths = _collect_snaps(
            env, policy, runner, injector, int(args_cli.steps), seed, terrain, vis_i, vis_flags, mask_name, layout
        )
        all_snaps.extend(snaps)
    held = terrain in HELD_OUT_TERRAINS
    nom = [s for s in all_snaps if s["window"] == "nominal"]
    rec = [s for s in all_snaps if s["window"] == "recovery"]
    split_map = _episode_split([s["episode_id"] for s in nom], held_out=held)
    quotas = {
        "train": 0 if held else int(args_cli.train_quota),
        "val": int(args_cli.val_quota) if not held else 0,
        "test": int(args_cli.test_quota) if not held else max(int(args_cli.test_quota), 250),
    }
    if held:
        quotas = {"train": 0, "val": 0, "test": min(len(nom), 500)}
    kept = _cap_by_split(all_snaps, split_map, quotas, seed=2026 + (sum(ord(c) for c in terrain) % 997))
    print(
        f"[ifs] kept {len(kept)} (nom={sum(1 for s in kept if s['window']=='nominal')} "
        f"rec={sum(1 for s in kept if s['window']=='recovery')}) terrain={terrain}",
        flush=True,
    )
    rows = _compute_jacobians(env, policy, runner, injector, kept, vis_i, vis_flags, layout, pelvis_i, ankle_i)
    _dump_jac(jac_p, rows, layout, terrain)
    eval_snaps = [s for s in kept if s.get("split") in ("val", "test", "recovery")]
    torch.save(eval_snaps, snaps_p)
    counts = {}
    for s in kept:
        key = f"{s.get('split')}:{s['window']}"
        counts[key] = counts.get(key, 0) + 1
    meta = {
        "terrain": terrain,
        "held_out": held,
        "n_walk": len(all_snaps),
        "n_kept": len(kept),
        "n_jac": len(rows),
        "n_eval_snaps": len(eval_snaps),
        "counts": counts,
        "body_names": layout["names"],
        "y_I_dim": int(kept[0]["y_I"].shape[0]) if kept else 0,
        "stride": int(args_cli.stride),
        "eps_deg": float(args_cli.eps_deg),
        "jac_horizon": int(args_cli.jac_horizon),
        "no_terrain_in_model": True,
        "seeds": seeds,
    }
    meta_p.write_text(json.dumps(_sanitize(meta), indent=2), encoding="utf-8")
    print(f"[ifs] wrote {jac_p} and {snaps_p}", flush=True)
    try:
        env.close()
    except Exception:
        pass
    return meta


def _tan_raw(d: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z, dim=-1, eps=1e-8)
    return d - (d * z).sum(-1, keepdim=True) * z


def _tilde_C_torch(J_I: torch.Tensor, z: torch.Tensor, pos_scale: float) -> torch.Tensor:
    J = J_I / float(pos_scale)
    z = F.normalize(z, dim=-1, eps=1e-8)
    eye = torch.eye(Z_DIM, device=z.device, dtype=z.dtype).unsqueeze(0)
    Pt = eye - z.unsqueeze(-1) * z.unsqueeze(-2)
    C = torch.bmm(J.transpose(1, 2), J)
    C = torch.bmm(Pt, torch.bmm(C, Pt))
    tr = C.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=0.0)
    return C / (tr / 15.0 + EPS_C).view(-1, 1, 1)


def _hard_P_torch(J_I: torch.Tensor, z: torch.Tensor, lam_rel: float, pos_scale: float) -> torch.Tensor:
    J = J_I / float(pos_scale)
    jjt = torch.bmm(J, J.transpose(1, 2))
    scale = jjt.diagonal(dim1=-2, dim2=-1).mean(-1).clamp(min=1e-8)
    ny = int(J.shape[1])
    eye_y = torch.eye(ny, device=J.device, dtype=J.dtype).unsqueeze(0)
    inv = torch.linalg.solve(jjt + (float(lam_rel) * scale).view(-1, 1, 1) * eye_y, J)
    P = torch.eye(Z_DIM, device=J.device, dtype=J.dtype).unsqueeze(0) - torch.bmm(J.transpose(1, 2), inv)
    z = F.normalize(z, dim=-1, eps=1e-8)
    Pt = torch.eye(Z_DIM, device=z.device, dtype=z.dtype).unsqueeze(0) - z.unsqueeze(-1) * z.unsqueeze(-2)
    return torch.bmm(Pt, torch.bmm(P, Pt))


def _soft_P_torch(Ctil: torch.Tensor, lam: float, z: torch.Tensor) -> torch.Tensor:
    P = torch.linalg.inv(torch.eye(Z_DIM, device=Ctil.device, dtype=Ctil.dtype).unsqueeze(0) + float(lam) * Ctil)
    z = F.normalize(z, dim=-1, eps=1e-8)
    Pt = torch.eye(Z_DIM, device=z.device, dtype=z.dtype).unsqueeze(0) - z.unsqueeze(-1) * z.unsqueeze(-2)
    return torch.bmm(Pt, torch.bmm(P, Pt))


def _apply_P_torch(P: torch.Tensor, d: torch.Tensor, z: torch.Tensor, suppress_eps: float = 1e-6):
    dT = _tan_raw(d, z)
    n0 = torch.linalg.norm(dT, dim=-1)
    dP = torch.bmm(P, dT.unsqueeze(-1)).squeeze(-1)
    dPT = _tan_raw(dP, z)
    n1 = torch.linalg.norm(dPT, dim=-1)
    rd = n1 / (n0 + 1e-12)
    suppressed = n1 < suppress_eps
    d_out = torch.where(suppressed.unsqueeze(-1), torch.zeros_like(dPT), dPT / n1.clamp(min=1e-12).unsqueeze(-1))
    return d_out, rd, suppressed


def _di_from_evec(e0: torch.Tensor, evec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """evec: (n, H, d). peak and integrated ||e_t - e0||."""
    delta = evec - e0.unsqueeze(1)
    nrm = torch.linalg.norm(delta, dim=-1)
    peak = nrm.max(dim=-1).values
    integ = nrm.sum(dim=-1) * DT
    return peak, integ


def _run_p3a(env_cfg, agent_cfg, resume_path, motion_dir, terrain, out_root: Path):
    ifs = Path(args_cli.ifs_root)
    out_json = out_root / "p3a_oracle_projector" / f"{terrain}.json"
    if out_json.exists():
        print(f"[p3a] skip {terrain}", flush=True)
        return json.loads(out_json.read_text())
    snaps_p = ifs / "dataset_a" / terrain / "snaps_eval.pt"
    jac_p = ifs / "dataset_a" / terrain / "jac.npz"
    if not snaps_p.exists() or not jac_p.exists():
        raise FileNotFoundError(f"need {snaps_p} and {jac_p}")
    env, runner, policy, injector = _make_env(env_cfg, agent_cfg, resume_path, motion_dir, terrain)
    cmd = env.unwrapped.command_manager.get_term("motion")
    body_names = list(cmd.cfg.body_names)
    vis_i = [body_names.index(n) for n in KP_VIS]
    mask_name = str(args_cli.mask_modes).split(",")[0].strip() or "torso"
    vis_flags = MASK_VIS.get(mask_name, (True, True, True))
    asset = env.unwrapped.scene["robot"]
    layout = _feature_layout(list(asset.data.joint_names))
    names = list(asset.data.body_names)
    pelvis_i = names.index("pelvis") if "pelvis" in names else None
    ankle_i = [names.index(n) for n in ANKLE_BODIES if n in names]
    device = env.unwrapped.device
    snaps = [s for s in torch.load(snaps_p, map_location="cpu") if s.get("window") == "recovery"]
    J_map = _load_jac_map(jac_p)
    scales = _load_scales(ifs)
    pos_scale = float(scales.get("pos_scale", POS_SCALE))
    sig_b = torch.as_tensor(scales["sig_B"], device=device, dtype=torch.float32).clamp(min=1e-8)
    if int(sig_b.numel()) == 1:
        sig_b = torch.ones(int(snaps[0]["y_B"].shape[0]), device=device)
    burst = int(args_cli.ucr_burst)
    horizon = int(args_cli.ucr_horizon)
    deg = float(args_cli.ucr_deg)
    eps = math.radians(deg)
    eps5 = math.radians(5.0)
    n_envs = int(env.unwrapped.num_envs)
    methods = [("A0_stage2", "stage2", 0.0), ("A1_full", "full", 0.0)]
    for lam in HARD_LAMBDAS:
        methods.append((f"A2_hard_{lam:g}", "hard", float(lam)))
    for lam in SOFT_LAMBDAS:
        methods.append((f"soft_{lam:g}", "soft", float(lam)))
    rows = []
    e_store = {m[0]: [] for m in methods}
    kw = dict(vis_i=vis_i, vis_flags=vis_flags, layout=layout, pelvis_i=pelvis_i, ankle_i=ankle_i)
    dstar_cache = {}
    cache_p = out_root / "p3a_oracle_projector" / f"{terrain}_dstars.npz"
    if cache_p.exists():
        zc = np.load(cache_p, allow_pickle=True)
        for i, ep in enumerate(zc["episode_id"]):
            dstar_cache[(str(ep), int(zc["t"][i]))] = zc["d"][i]
        print(f"[p3a] loaded {len(dstar_cache)} cached d*", flush=True)
    new_d = []
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[p3a] {terrain} {start}:{start+n_valid}/{len(snaps)}", flush=True)
        z0 = torch.stack([s["z_nom"] for s in batch], dim=0).to(device)
        V = tangent_basis(z0)
        _, _, e_parent, extra_p = _roll_features(
            env, policy, runner, injector, batch, n_valid, None, 0.0, horizon, **kw, burst=None, record_e=True
        )
        need_search = []
        d_star = torch.zeros(n_valid, Z_DIM, device=device)
        for i, s in enumerate(batch):
            key = (s["episode_id"], int(s["t"]))
            if key in dstar_cache:
                d_star[i] = torch.as_tensor(dstar_cache[key], device=device)
            else:
                need_search.append(i)
        if need_search:
            a_axes = torch.zeros(n_valid, N_AXES, 2, device=device)
            for ax in range(N_AXES):
                for si, sgn in enumerate((1.0, -1.0)):
                    _, _, eh, _ = _roll_features(
                        env, policy, runner, injector, batch, n_valid, V[:, :, ax], float(sgn) * eps,
                        horizon, **kw, burst=burst, record_e=True,
                    )
                    a_axes[:, ax, si] = e_parent[:, -1] - eh[:, -1]
            best_flat = a_axes.reshape(n_valid, -1).argmax(dim=-1)
            for i in need_search:
                ax = int(best_flat[i] // 2)
                sgn = 1.0 if int(best_flat[i] % 2) == 0 else -1.0
                d_star[i] = sgn * V[i, :, ax]
                new_d.append((batch[i]["episode_id"], int(batch[i]["t"]), d_star[i].detach().cpu().numpy()))
            print(f"[p3a]   searched d* for {len(need_search)} states", flush=True)
        J_list = []
        ok_j = True
        for s in batch:
            key = (s["episode_id"], int(s["t"]), s["window"])
            if key not in J_map:
                ok_j = False
                break
            J_list.append(J_map[key]["J_I"])
        if not ok_j:
            print("[p3a] missing J_I, skip batch", flush=True)
            continue
        J_I = torch.as_tensor(np.stack(J_list), device=device, dtype=torch.float32)
        Ctil = _tilde_C_torch(J_I, z0, pos_scale)
        method_out = {}
        for name, kind, lam in methods:
            if kind == "stage2":
                d_exec = None
                rd = torch.zeros(n_valid, device=device)
                supp = torch.ones(n_valid, dtype=torch.bool, device=device)
                eh = e_parent
                extra = extra_p
            elif kind == "full":
                d_exec = project_tangent(d_star, z0)
                rd = torch.ones(n_valid, device=device)
                supp = torch.zeros(n_valid, dtype=torch.bool, device=device)
            elif kind == "hard":
                P = _hard_P_torch(J_I, z0, lam, pos_scale)
                d_exec, rd, supp = _apply_P_torch(P, d_star, z0)
            else:
                P = _soft_P_torch(Ctil, lam, z0)
                d_exec, rd, supp = _apply_P_torch(P, d_star, z0)
            if kind != "stage2":
                d_use = torch.where(supp.unsqueeze(-1), torch.zeros_like(d_exec), d_exec)
                yIp, yBp, eh, extra = _roll_features(
                    env, policy, runner, injector, batch, n_valid,
                    d_use, eps, horizon, **kw, burst=burst, record_e=True,
                )
                yI_f, yB_f = yIp, yBp
            peak, integ = _di_from_evec(extra["e0_vec"], extra["evec"])
            yI5p, yB5p, _, _ = _roll_features(
                env, policy, runner, injector, batch, n_valid,
                torch.zeros_like(z0) if kind == "stage2" else (d_exec if d_exec is not None else torch.zeros_like(z0)),
                0.0 if kind == "stage2" else eps5, 1, **kw,
            )
            yI5m, yB5m, _, _ = _roll_features(
                env, policy, runner, injector, batch, n_valid,
                torch.zeros_like(z0) if kind == "stage2" else (d_exec if d_exec is not None else torch.zeros_like(z0)),
                0.0 if kind == "stage2" else -eps5, 1, **kw,
            )
            di5 = torch.linalg.norm((yI5p - yI5m) / 2.0 / pos_scale, dim=-1)
            db5 = torch.linalg.norm((yB5p - yB5m) / 2.0 / sig_b.view(1, -1), dim=-1)
            A = (e_parent[:, -1] - eh[:, -1]).detach()
            method_out[name] = {
                "A": A.cpu().numpy(),
                "e_term": eh[:, -1].detach().cpu().numpy(),
                "e0": extra["e0"].cpu().numpy(),
                "DI_peak": peak.cpu().numpy(),
                "DI_int": integ.cpu().numpy(),
                "DI_5deg": di5.detach().cpu().numpy(),
                "DB_5deg": db5.detach().cpu().numpy(),
                "Rd": rd.detach().cpu().numpy(),
                "suppressed": supp.detach().cpu().numpy().astype(np.int32),
                "e_hist": eh.detach().cpu().numpy(),
            }
            e_store[name].append(eh.detach().cpu().numpy())
            print(f"[p3a]   {name}", flush=True)
        for i in range(n_valid):
            s = batch[i]
            item = {
                "episode_id": s["episode_id"],
                "terrain": s["terrain"],
                "seed": int(s["seed"]),
                "t": int(s["t"]),
                "clip": s["clip"],
                "e0": float(s["e0"]),
                "held_out": terrain in HELD_OUT_TERRAINS,
                "methods": {},
            }
            for name, _k, _l in methods:
                mo = method_out[name]
                item["methods"][name] = {
                    "A": float(mo["A"][i]),
                    "e_term": float(mo["e_term"][i]),
                    "e_parent": float(e_parent[i, -1].item()),
                    "DI_peak": float(mo["DI_peak"][i]),
                    "DI_int": float(mo["DI_int"][i]),
                    "DI_5deg": float(mo["DI_5deg"][i]),
                    "DB_5deg": float(mo["DB_5deg"][i]),
                    "Rd": float(mo["Rd"][i]),
                    "suppressed": int(mo["suppressed"][i]),
                    "dE": float(mo["e0"][i] - mo["e_term"][i]),
                }
            rows.append(item)
    if new_d:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        old_ep = list(dstar_cache.keys())
        eps_all = [k[0] for k in old_ep] + [x[0] for x in new_d]
        ts_all = [k[1] for k in old_ep] + [x[1] for x in new_d]
        ds_all = [dstar_cache[k] for k in old_ep] + [x[2] for x in new_d]
        np.savez_compressed(
            cache_p,
            episode_id=np.asarray(eps_all),
            t=np.asarray(ts_all, dtype=np.int32),
            d=np.stack(ds_all).astype(np.float32),
        )
    payload = {
        "terrain": terrain,
        "held_out": terrain in HELD_OUT_TERRAINS,
        "n": len(rows),
        "ucr_burst": burst,
        "ucr_horizon": horizon,
        "ucr_deg": deg,
        "methods": [m[0] for m in methods],
        "rows": rows,
        "no_terrain_in_model": True,
        "A_def": "E_parent - E_pert at t+0.5s, positive is better",
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    e_dir = out_root / "p3a_oracle_projector" / "e_hist"
    e_dir.mkdir(parents=True, exist_ok=True)
    packed = {k: np.concatenate(v, axis=0) for k, v in e_store.items() if v}
    if packed:
        np.savez_compressed(e_dir / f"{terrain}.npz", **packed)
    print(f"[p3a] wrote {out_json} n={len(rows)}", flush=True)
    try:
        env.close()
    except Exception:
        pass
    return payload


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    direct = getattr(agent_cfg, "resume_checkpoint_path", None)
    resume_path = os.path.abspath(str(direct)) if direct else (
        PARENT_CKPT if Path(PARENT_CKPT).is_file() else get_checkpoint_path(
            log_root, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    )
    print(f"[p3a] ckpt {resume_path}", flush=True)
    _restore_adapter(agent_cfg, resume_path)
    if hasattr(agent_cfg.policy, "intent_recovery"):
        agent_cfg.policy.intent_recovery = False
    if hasattr(agent_cfg.policy, "interaction_recovery"):
        agent_cfg.policy.interaction_recovery = False
    if hasattr(agent_cfg.policy, "terrain_scan_dim"):
        agent_cfg.policy.terrain_scan_dim = 0
    if hasattr(agent_cfg.policy, "adapter"):
        agent_cfg.policy.adapter = "residual"
    terrains = [t.strip() for t in args_cli.terrains.split(",") if t.strip()] or [args_cli.terrain]
    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for terrain in terrains:
        _run_p3a(env_cfg, agent_cfg, resume_path, args_cli.motion, terrain, out_dir)
    print("[p3a] done", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        os._exit(0)
