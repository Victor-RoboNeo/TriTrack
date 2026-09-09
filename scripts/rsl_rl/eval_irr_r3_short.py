#!/usr/bin/env python3
"""IRR R3-short: 1–2 step ±ε probe → direction of 5° recovery.

Eval-only recovery window (e>=0.13 persist 3). Trigger is not a deployment
mechanism. No GRU / PPO / terrain labels. Frozen Stage-2.
"""
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
HIST = 16
K_HORIZON = 10
H_CORR = 25  # 0.5 s endpoint burst at 50 Hz for P4-B4B
GAMMA = 0.95
LAM_S = 1.0
N_AXES = 15
ANGLES = (1.0, 2.0)
TOKEN = 128  # z16 + e9 + de9 + prop90 + contact2 + roll + pitch
Y_DIM = 12
PROBE_LS = (1, 2, 3, 5)
EPS_DEG = 1.0
EVAL_DEG = 5.0
B_PATH = "/data/home/chenxiangyu/robotics/Anybody/results/irr_response_recovery/B_recovery.npz"
FAIL_ANCHOR_Z = 0.25
FAIL_ANCHOR_ORI = 0.8
FAIL_FALL_Z = 0.4
CONTACT_N = 10.0
KP_VIS = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
ANKLE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
MASK_VIS = {
    "torso": (True, False, False),
    "vr": (True, True, True),
    "head_right": (True, False, True),
    "head_left": (True, True, False),
}
PARENT_CKPT = (
    "/data/home/chenxiangyu/robotics/Anybody/logs/rsl_rl/"
    "g1_flat_muse_kp_latent_rl/2026-08-26_00-38-10_tritrack_headhands_locomani_from35000/"
    "model_50000.pt"
)
MAPPER_B = "/data/home/chenxiangyu/victor/TriTrack/runs/mapper_b_intent72/mapper_best.pt"
CLIP_ROOT = "/data/home/chenxiangyu/robotics/Anybody/logs/tritrack/infer_clips/p1"
TERRAINS = ("plane", "slope", "slope_down", "light_rough", "steps", "slip")
TASK_SOURCES = ("loco", "stoop", "reach", "carry")
S_ENABLED_TASKS = ("stoop", "reach", "carry")


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

parser = argparse.ArgumentParser(description="IRR R3-short active probe. Frozen Stage-2.")
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--task", type=str, default="MUSE-Kp-LatentRL-Kp5-HeadHands-Locomani-G1-v0")
parser.add_argument("--motion", type=str, default=f"{CLIP_ROOT}/loco")
parser.add_argument("--mask_modes", type=str, default="torso")
parser.add_argument("--task_source", type=str, default="loco", choices=TASK_SOURCES)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out", type=str, default="results/irr_response_recovery")
parser.add_argument("--start_frame", type=int, default=10)
parser.add_argument("--hint_iter", type=int, default=50000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--seeds", type=str, default="42,43,44")
parser.add_argument("--terrain", type=str, default="plane")
parser.add_argument("--terrains", type=str, default="plane")
parser.add_argument("--keep_headhands_spec", action="store_true", default=True)
parser.add_argument("--mapper_path", type=str, default=MAPPER_B)
parser.add_argument("--mapper_b_path", type=str, default=MAPPER_B)
parser.add_argument("--residual_alpha", type=float, default=1.0)
parser.add_argument("--s_enabled", action="store_true", default=False)
parser.add_argument("--no_s_enabled", action="store_true", default=False)
parser.add_argument("--horizon", type=int, default=K_HORIZON)
parser.add_argument("--n_axes", type=int, default=N_AXES)
parser.add_argument("--angles", type=str, default="1,2")
parser.add_argument("--r0_tag", type=str, default="r3_short")
parser.add_argument("--r3_tag", type=str, default="r3_short")
parser.add_argument("--b_path", type=str, default=B_PATH)
parser.add_argument("--persist", type=int, default=3)
parser.add_argument("--max_snaps_rec", type=int, default=2)
parser.add_argument("--max_snaps_nom", type=int, default=0)
parser.add_argument("--e_rec", type=float, default=0.13)
parser.add_argument("--e_nom", type=float, default=0.06)
parser.add_argument("--push_dv", type=float, default=0.0)
parser.add_argument("--push_step", type=int, default=100)
parser.add_argument("--rec_on_plane", action="store_true", default=False)
parser.add_argument("--p4a0", action="store_true", default=False)
parser.add_argument("--p4a1", action="store_true", default=False)
parser.add_argument("--p4b", action="store_true", default=False)
parser.add_argument("--p4b2", action="store_true", default=False)
parser.add_argument("--p4b3j", action="store_true", default=False)
parser.add_argument("--p4b4a", action="store_true", default=False)
parser.add_argument("--p4b4b", action="store_true", default=False)
parser.add_argument("--p4b_ns", type=str, default="6,8,10")
parser.add_argument("--p4b_code_seed", type=int, default=2026)
parser.add_argument("--p4b_dir", type=str, default="/data/home/chenxiangyu/robotics/Anybody/results/p4b_tube_online_id")
parser.add_argument("--tube_guard", type=float, default=0.1)
parser.add_argument("--p4a1_dir", type=str, default="/data/home/chenxiangyu/robotics/Anybody/results/p4a_intent_tube_interface")
parser.add_argument("--tube_fracs", type=str, default="0,0.1,0.2,0.4,0.6,1.0")
parser.add_argument("--bucr_path", type=str, default="/data/home/chenxiangyu/robotics/Anybody/results/intent_free_space/basis/B_ucr.npz")
parser.add_argument("--p4a0_k", type=int, default=3)
parser.add_argument("--pos_scale", type=float, default=0.05)
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


def _e9(cmd, vis_i) -> torch.Tensor:
    delta = cmd.robot_body_pos_w - cmd.body_pos_w
    return torch.cat([delta[:, vis_i[k]] for k in range(3)], dim=-1)


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
    return list(snaps) + [snaps[-1]] * (n_envs - n), n


def _angle_list() -> tuple[float, ...]:
    raw = str(getattr(args_cli, "angles", "1,2"))
    return tuple(float(x) for x in raw.split(",") if x.strip())


def _candidate_table(n_axes: int):
    rows = [("zero", 0, -1, 0.0)]
    for axis in range(int(n_axes)):
        for deg in _angle_list():
            rows.append((f"+a{axis}_{deg:g}", 1, axis, float(deg)))
            rows.append((f"-a{axis}_{deg:g}", -1, axis, float(deg)))
    return rows


@torch.inference_mode()
def _roll_j(env, policy, runner, injector, snaps, n_valid, dz, vis_i, vis_flags, pelvis_i, horizon):
    n_envs = int(env.unwrapped.num_envs)
    padded, _ = _pad_snaps(snaps, n_envs)
    _restore_batch(env, padded)
    cmd = env.unwrapped.command_manager.get_term("motion")
    asset = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    gravity = asset.data.GRAVITY_VEC_W
    if gravity.ndim == 1:
        gravity = gravity.unsqueeze(0).expand(cmd.num_envs, -1)
    failed = torch.zeros(n_envs, dtype=torch.bool, device=device)
    j = torch.zeros(n_envs, device=device, dtype=torch.float32)
    obs_stack = torch.stack([s["obs"].to(device) for s in padded], dim=0)
    dz_pad = torch.zeros(n_envs, 16, device=device, dtype=torch.float32)
    if dz is not None:
        dz_pad[:n_valid] = dz
    for t in range(horizon):
        obs = obs_stack if t == 0 else injector.patch_policy_obs(env.get_observations()[0], env, "mapper")
        z_nom, proprio, _enc = _z_and_proprio(policy, runner, obs)
        z_exec = F.normalize(z_nom + dz_pad, dim=-1, eps=1e-8) if dz is not None else z_nom
        env.step(_decode(policy, z_exec, proprio))
        e = _visible_e(cmd, vis_i, vis_flags)
        fail = _official_fail(cmd, asset, vis_i, n_envs, pelvis_i, gravity)
        failed = failed | fail
        j = j + (GAMMA ** t) * (e + LAM_S * fail.to(dtype=e.dtype))
    return j[:n_valid].detach(), failed[:n_valid].detach()


def _read_y(env, vis_i, vis_flags, pelvis_i, ankle_i, cf, e_prev):
    """y = [e, de, roll, pitch, ω(3), v_root(3), contact(2)]. No terrain."""
    cmd = env.unwrapped.command_manager.get_term("motion")
    asset = env.unwrapped.scene["robot"]
    n = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    e = _visible_e(cmd, vis_i, vis_flags)
    de = torch.zeros_like(e) if e_prev is None else (e - e_prev) / DT
    roll, pitch, _yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    omega = getattr(asset.data, "root_ang_vel_b", None)
    if omega is None:
        omega = asset.data.root_ang_vel_w
    vel = getattr(asset.data, "root_lin_vel_b", None)
    if vel is None:
        vel = asset.data.root_lin_vel_w
    contact2 = torch.zeros(n, 2, device=device)
    if cf is not None and ankle_i and hasattr(cf.data, "net_forces_w"):
        nf = cf.data.net_forces_w
        contact2 = (nf[:, ankle_i, :].norm(dim=-1) > CONTACT_N).float()
    parts = [
        e.unsqueeze(-1),
        de.unsqueeze(-1),
        roll.unsqueeze(-1),
        pitch.unsqueeze(-1),
        omega[:, :3],
        vel[:, :3],
        contact2,
    ]
    y = torch.cat(parts, dim=-1)
    if int(y.shape[-1]) < Y_DIM:
        pad = y.new_zeros(n, Y_DIM - int(y.shape[-1]))
        y = torch.cat([y, pad], dim=-1)
    return y[:, :Y_DIM], e.detach()


@torch.inference_mode()
def _roll_y(env, policy, runner, injector, snaps, n_valid, dz, vis_i, vis_flags, pelvis_i, ankle_i, cf, horizon):
    n_envs = int(env.unwrapped.num_envs)
    padded, _ = _pad_snaps(snaps, n_envs)
    _restore_batch(env, padded)
    device = env.unwrapped.device
    obs_stack = torch.stack([s["obs"].to(device) for s in padded], dim=0)
    dz_pad = torch.zeros(n_envs, 16, device=device, dtype=torch.float32)
    if dz is not None:
        dz_pad[:n_valid] = dz
    y0, e0 = _read_y(env, vis_i, vis_flags, pelvis_i, ankle_i, cf, None)
    e_prev = e0
    for t in range(max(int(horizon), 1)):
        obs = obs_stack if t == 0 else injector.patch_policy_obs(env.get_observations()[0], env, "mapper")
        z_nom, proprio, _enc = _z_and_proprio(policy, runner, obs)
        z_exec = F.normalize(z_nom + dz_pad, dim=-1, eps=1e-8) if dz is not None else z_nom
        env.step(_decode(policy, z_exec, proprio))
        yL, e_prev = _read_y(env, vis_i, vis_flags, pelvis_i, ankle_i, cf, e_prev)
    return (yL - y0)[:n_valid].detach(), y0[:n_valid].detach(), e0[:n_valid].detach()


@torch.inference_mode()
def _roll_y_seq(env, policy, runner, injector, snaps, n_valid, dz_pos, dz_neg, vis_i, vis_flags, pelvis_i, ankle_i, cf, horizon):
    """Sequential +ε then −ε from the same restored state. No clone between signs."""
    n_envs = int(env.unwrapped.num_envs)
    padded, _ = _pad_snaps(snaps, n_envs)
    _restore_batch(env, padded)
    device = env.unwrapped.device
    obs_stack = torch.stack([s["obs"].to(device) for s in padded], dim=0)
    dz_p = torch.zeros(n_envs, 16, device=device, dtype=torch.float32)
    dz_m = torch.zeros(n_envs, 16, device=device, dtype=torch.float32)
    dz_p[:n_valid] = dz_pos
    dz_m[:n_valid] = dz_neg
    y0, e0 = _read_y(env, vis_i, vis_flags, pelvis_i, ankle_i, cf, None)
    e_prev = e0
    obs = obs_stack
    for t in range(max(int(horizon), 1)):
        if t > 0:
            obs = injector.patch_policy_obs(env.get_observations()[0], env, "mapper")
        z_nom, proprio, _enc = _z_and_proprio(policy, runner, obs)
        env.step(_decode(policy, F.normalize(z_nom + dz_p, dim=-1, eps=1e-8), proprio))
        yp, e_prev = _read_y(env, vis_i, vis_flags, pelvis_i, ankle_i, cf, e_prev)
    for t in range(max(int(horizon), 1)):
        obs = injector.patch_policy_obs(env.get_observations()[0], env, "mapper")
        z_nom, proprio, _enc = _z_and_proprio(policy, runner, obs)
        env.step(_decode(policy, F.normalize(z_nom + dz_m, dim=-1, eps=1e-8), proprio))
        ym, e_prev = _read_y(env, vis_i, vis_flags, pelvis_i, ankle_i, cf, e_prev)
    dy_p = yp - y0
    dy_m = ym - yp
    return dy_p[:n_valid].detach(), dy_m[:n_valid].detach()


def _tangent_B(z: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Project shared B (16,k) onto current tangent and re-orthonormalize. [n,16,k]."""
    n = z.shape[0]
    k = int(B.shape[1])
    z = F.normalize(z, dim=-1, eps=1e-8)
    cols = []
    for i in range(k):
        cols.append(project_tangent(B[:, i].unsqueeze(0).expand(n, -1), z))
    M = torch.stack(cols, dim=-1)
    q, _r = torch.linalg.qr(M)
    return q


def _axis_dz(z: torch.Tensor, B_t: torch.Tensor, ki: int, sgn: float, tan_a: float) -> torch.Tensor:
    """Unit tangent along UCR axis ki, scaled to geodesic tan(deg)."""
    v = _tangent_B(z, B_t)[:, :, int(ki)]
    v = project_tangent(v, z)
    nrm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return float(sgn) * float(tan_a) * v / nrm


def _yI_visible(cmd, vis_i, vis_flags) -> torch.Tensor:
    delta = cmd.robot_body_pos_w - cmd.body_pos_w
    parts = [delta[:, vis_i[k]] for k, flag in enumerate(vis_flags) if flag]
    if not parts:
        return delta[:, vis_i[0]]
    return torch.cat(parts, dim=-1)


def _roll_yI(env, policy, runner, injector, snaps, n_valid, dz, vis_i, vis_flags, horizon=1):
    n_envs = int(env.unwrapped.num_envs)
    padded, _ = _pad_snaps(snaps, n_envs)
    _restore_batch(env, padded)
    cmd = env.unwrapped.command_manager.get_term("motion")
    device = env.unwrapped.device
    obs_stack = torch.stack([s["obs"].to(device) for s in padded], dim=0)
    dz_pad = torch.zeros(n_envs, 16, device=device, dtype=torch.float32)
    if dz is not None:
        dz_pad[:n_valid] = dz
    for _t in range(max(int(horizon), 1)):
        obs = obs_stack if _t == 0 else injector.patch_policy_obs(env.get_observations()[0], env, "mapper")
        z_nom, proprio, _enc = _z_and_proprio(policy, runner, obs)
        z_exec = F.normalize(z_nom + dz_pad, dim=-1, eps=1e-8) if dz is not None else z_nom
        env.step(_decode(policy, z_exec, proprio))
    return _yI_visible(cmd, vis_i, vis_flags)[:n_valid].detach()


def _oracle_P_from_JI(J_I: torch.Tensor, z: torch.Tensor, pos_scale: float) -> torch.Tensor:
    J = J_I / float(pos_scale)
    z = F.normalize(z, dim=-1, eps=1e-8)
    eye = torch.eye(16, device=z.device, dtype=z.dtype).unsqueeze(0)
    Pt = eye - z.unsqueeze(-1) * z.unsqueeze(-2)
    C = torch.bmm(J.transpose(1, 2), J)
    C = torch.bmm(Pt, torch.bmm(C, Pt))
    tr = C.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=0.0)
    C = C / (tr / 15.0 + 1e-8).view(-1, 1, 1)
    P = torch.linalg.inv(eye + C)
    return torch.bmm(Pt, torch.bmm(P, Pt))


@torch.inference_mode()
def _fd_JI(env, policy, runner, injector, batch, n_valid, z0, vis_i, vis_flags):
    V = tangent_basis(z0)
    eps = math.tan(math.radians(EPS_DEG))
    y0 = _roll_yI(env, policy, runner, injector, batch, n_valid, None, vis_i, vis_flags, horizon=1)
    ny = int(y0.shape[-1])
    Jtan = torch.zeros(n_valid, ny, 15, device=z0.device, dtype=z0.dtype)
    for ax in range(15):
        v = V[:, :, ax]
        yp = _roll_yI(env, policy, runner, injector, batch, n_valid, eps * v, vis_i, vis_flags, horizon=1)
        ym = _roll_yI(env, policy, runner, injector, batch, n_valid, -eps * v, vis_i, vis_flags, horizon=1)
        Jtan[:, :, ax] = (yp - ym) / (2.0 * eps)
    return torch.bmm(Jtan, V.transpose(1, 2))


@torch.inference_mode()
def _probe_p4a0(env, policy, runner, injector, snaps, vis_i, vis_flags, pelvis_i, B_raw_np):
    """Clone 1° L=1, 5° labels. RAW vs oracle-shielded QR(P_T P_I B_UCR). No sequential."""
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    asset = env.unwrapped.scene["robot"]
    names = list(asset.data.body_names)
    ankle_i = [names.index(n) for n in ANKLE_BODIES if n in names]
    try:
        cf = env.unwrapped.scene["contact_forces"]
    except Exception:
        cf = None
    eps = math.tan(math.radians(EPS_DEG))
    tan5 = math.tan(math.radians(EVAL_DEG))
    k = int(args_cli.p4a0_k)
    B_raw = torch.as_tensor(B_raw_np[:, :k], device=device, dtype=torch.float32)
    pos_scale = float(args_cli.pos_scale)
    rows = []
    kwj = dict(vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i)
    kwy = dict(vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i, ankle_i=ankle_i, cf=cf)
    spaces = ("raw", "oracle_safe")
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[p4a0] batch {start}:{start + n_valid}/{len(snaps)}", flush=True)
        z0 = torch.stack([s["z_nom"] for s in batch], dim=0).to(device=device)
        J_I = _fd_JI(env, policy, runner, injector, batch, n_valid, z0, vis_i, vis_flags)
        P_I = _oracle_P_from_JI(J_I, z0, pos_scale)
        Bt_raw = _tangent_B(z0, B_raw)
        B_sh = torch.bmm(P_I, B_raw.unsqueeze(0).expand(n_valid, -1, -1))
        z_n = F.normalize(z0, dim=-1, eps=1e-8)
        PtB = B_sh - z_n.unsqueeze(-1) * (z_n.unsqueeze(1) @ B_sh)
        Bt_safe, _ = torch.linalg.qr(PtB)
        bases = {"raw": Bt_raw, "oracle_safe": Bt_safe}
        j0, _ = _roll_j(env, policy, runner, injector, batch, n_valid, None, horizon=K_HORIZON, **kwj)
        pack = {}
        for name, Bt in bases.items():
            a5 = torch.zeros(n_valid, k, 2, device=device)
            r_e = torch.zeros(n_valid, k, device=device)
            di = torch.zeros(n_valid, k, device=device)
            for ki in range(k):
                v = Bt[:, :, ki]
                for si, sgn in enumerate((1.0, -1.0)):
                    j5, _ = _roll_j(
                        env, policy, runner, injector, batch, n_valid, float(sgn) * tan5 * v,
                        horizon=K_HORIZON, **kwj,
                    )
                    a5[:, ki, si] = j0 - j5
                dyp, y0, e0_b = _roll_y(env, policy, runner, injector, batch, n_valid, eps * v, horizon=1, **kwy)
                dym, _, _ = _roll_y(env, policy, runner, injector, batch, n_valid, -eps * v, horizon=1, **kwy)
                r_e[:, ki] = (dyp[:, 0] - dym[:, 0]) / (2.0 * eps)
                di[:, ki] = 0.5 * (dyp[:, 0].abs() + dym[:, 0].abs())
            pack[name] = {"a5": a5, "r_e": r_e, "di": di, "Bt": Bt}
            print(f"[p4a0]   {name} k={k}", flush=True)
        for i in range(n_valid):
            s = batch[i]
            item = {
                "clip": s["clip"],
                "seed": int(s["seed"]),
                "t": int(s["t"]),
                "terrain": s["terrain"],
                "window": s["window"],
                "e0": float(s["e0"]),
                "j0": float(j0[i].item()),
                "z_nom": s["z_nom"].numpy().astype(np.float32),
            }
            for name in spaces:
                item[f"a5_{name}"] = pack[name]["a5"][i].detach().cpu().numpy().astype(np.float32)
                item[f"r_e_{name}"] = pack[name]["r_e"][i].detach().cpu().numpy().astype(np.float32)
                item[f"di_{name}"] = pack[name]["di"][i].detach().cpu().numpy().astype(np.float32)
            rows.append(item)
    return rows


def _C_intent(J_I: torch.Tensor, z: torch.Tensor, pos_scale: float) -> torch.Tensor:
    """C = P_T (J/s)^T (J/s) P_T. No trace-norm. d^T C d is (DI per unit tan)^2."""
    J = J_I / float(pos_scale)
    z = F.normalize(z, dim=-1, eps=1e-8)
    eye = torch.eye(16, device=z.device, dtype=z.dtype).unsqueeze(0)
    Pt = eye - z.unsqueeze(-1) * z.unsqueeze(-2)
    C = torch.bmm(J.transpose(1, 2), J)
    C = torch.bmm(Pt, torch.bmm(C, Pt))
    return 0.5 * (C + C.transpose(1, 2))


def _hard_null_B(C: torch.Tensor, z: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """QR of ker(C) projection of B. Strict nullspace, f=0."""
    n, k = B.shape[0], B.shape[2]
    z = F.normalize(z, dim=-1, eps=1e-8)
    eye = torch.eye(16, device=z.device, dtype=z.dtype).unsqueeze(0)
    Pt = eye - z.unsqueeze(-1) * z.unsqueeze(-2)
    out = []
    for i in range(n):
        w, U = torch.linalg.eigh(C[i])
        null = U[:, w < 1e-8]
        if null.shape[1] < 1:
            null = U[:, :1]
        P = null @ null.T
        M = P @ B[i]
        M = Pt[i] @ M
        q, _r = torch.linalg.qr(M)
        if q.shape[1] < k:
            pad = torch.zeros(16, k - q.shape[1], device=q.device, dtype=q.dtype)
            q = torch.cat([q, pad], dim=1)
        out.append(q[:, :k])
    return torch.stack(out, dim=0)


def _tube_column(d_raw: np.ndarray, C: np.ndarray, V: np.ndarray, eps2: float) -> tuple[np.ndarray, float, float]:
    """Closest tangent d to d_raw with d^T C d <= eps2 and ||d||<=1, then unitize for probe."""
    from scipy.optimize import minimize

    d_raw = np.asarray(d_raw, dtype=np.float64).reshape(-1)
    C = np.asarray(C, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    a0 = V.T @ d_raw
    n0 = float(np.linalg.norm(a0))
    if n0 < 1e-12:
        return np.zeros(16), 0.0, 1.0
    a0 = a0 / n0
    Chat = V.T @ C @ V
    Chat = 0.5 * (Chat + Chat.T)
    s0 = float(a0 @ Chat @ a0)
    if s0 <= float(eps2) + 1e-10:
        d = V @ a0
        d = d / (np.linalg.norm(d) + 1e-12)
        return d, abs(float(np.dot(d, d_raw))), float(np.linalg.norm(d - d_raw))

    def fun(a):
        return float(np.sum((a - a0) ** 2))

    cons = (
        {"type": "ineq", "fun": lambda a: float(eps2) - float(a @ Chat @ a)},
        {"type": "ineq", "fun": lambda a: 1.0 - float(np.sum(a * a))},
    )
    res = minimize(fun, a0, method="SLSQP", constraints=cons, bounds=[(-1.0, 1.0)] * 15, options={"maxiter": 80, "ftol": 1e-9})
    a = np.asarray(res.x, dtype=np.float64) if res.success else a0.copy()
    # 2D mix fallback / cleanup
    w, U = np.linalg.eigh(Chat)
    w = np.maximum(w, 0.0)
    sens = w > 1e-8
    b = U.T @ a
    if float(eps2) <= 1e-16:
        b[sens] = 0.0
        a = U @ b
    else:
        s = float(a @ Chat @ a)
        if s > float(eps2) + 1e-8:
            bs = (U.T @ a0)
            lam = w[sens]
            xs = bs[sens]
            s_s = float(np.dot(lam * xs, xs))
            if s_s > 1e-16:
                t = math.sqrt(float(eps2) / s_s)
                bs = bs.copy()
                bs[sens] = t * xs
                a = U @ bs
    nrm = float(np.linalg.norm(a))
    if nrm < 1e-12:
        w2, U2 = np.linalg.eigh(Chat)
        a = U2[:, 0]
        nrm = float(np.linalg.norm(a))
    a = a / max(nrm, 1e-12)
    d = V @ a
    d = d / (np.linalg.norm(d) + 1e-12)
    return d, abs(float(np.dot(d, d_raw))), float(np.linalg.norm(d - d_raw))


def _tube_B(Bt_raw: torch.Tensor, C: torch.Tensor, z: torch.Tensor, frac: float, tan5: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns B_eps [n,16,k], Rd [n,k], dist [n,k]. Constraint: linearized 5° DI <= frac."""
    n, _, k = Bt_raw.shape
    device = Bt_raw.device
    V = tangent_basis(z)
    eps2 = 0.0 if frac <= 0 else (float(frac) / float(tan5)) ** 2
    B_np = Bt_raw.detach().cpu().numpy()
    C_np = C.detach().cpu().numpy()
    V_np = V.detach().cpu().numpy()
    out = np.zeros((n, 16, k), dtype=np.float64)
    rd = np.zeros((n, k), dtype=np.float64)
    dist = np.zeros((n, k), dtype=np.float64)
    if frac <= 0:
        Bh = _hard_null_B(C, z, Bt_raw)
        for i in range(n):
            for ki in range(k):
                d = Bh[i, :, ki].detach().cpu().numpy()
                dr = B_np[i, :, ki]
                out[i, :, ki] = d
                rd[i, ki] = abs(float(np.dot(d, dr)))
                dist[i, ki] = float(np.linalg.norm(d - dr))
        return (
            torch.as_tensor(out, device=device, dtype=Bt_raw.dtype),
            torch.as_tensor(rd, device=device, dtype=Bt_raw.dtype),
            torch.as_tensor(dist, device=device, dtype=Bt_raw.dtype),
        )
    for i in range(n):
        for ki in range(k):
            d, r, dst = _tube_column(B_np[i, :, ki], C_np[i], V_np[i], eps2)
            out[i, :, ki] = d
            rd[i, ki] = r
            dist[i, ki] = dst
    return (
        torch.as_tensor(out, device=device, dtype=Bt_raw.dtype),
        torch.as_tensor(rd, device=device, dtype=Bt_raw.dtype),
        torch.as_tensor(dist, device=device, dtype=Bt_raw.dtype),
    )


@torch.inference_mode()
def _probe_signed(env, policy, runner, injector, batch, n_valid, v, tan_a, horizon, kwj):
    a = torch.zeros(n_valid, 2, device=v.device)
    j0, _ = _roll_j(env, policy, runner, injector, batch, n_valid, None, horizon=horizon, **kwj)
    for si, sgn in enumerate((1.0, -1.0)):
        j5, _ = _roll_j(
            env, policy, runner, injector, batch, n_valid, float(sgn) * tan_a * v,
            horizon=horizon, **kwj,
        )
        a[:, si] = j0 - j5
    return j0, a


@torch.inference_mode()
def _probe_p4a1(env, policy, runner, injector, snaps, vis_i, vis_flags, pelvis_i, B_raw_np, fracs):
    """Intent-tube clone probe. RAW + tube fracs. P-only and P+C 5° labels. No sequential."""
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    eps = math.tan(math.radians(EPS_DEG))
    tan5 = math.tan(math.radians(EVAL_DEG))
    k = int(args_cli.p4a0_k)
    B_raw = torch.as_tensor(B_raw_np[:, :k], device=device, dtype=torch.float32)
    pos_scale = float(args_cli.pos_scale)
    kwj = dict(vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i)
    rows = []
    levels = ["raw"] + [f"f{frac:g}" for frac in fracs]
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[p4a1] batch {start}:{start + n_valid}/{len(snaps)}", flush=True)
        z0 = torch.stack([s["z_nom"] for s in batch], dim=0).to(device=device)
        J_I = _fd_JI(env, policy, runner, injector, batch, n_valid, z0, vis_i, vis_flags)
        C = _C_intent(J_I, z0, pos_scale)
        Bt_raw = _tangent_B(z0, B_raw)
        y_par = _roll_yI(env, policy, runner, injector, batch, n_valid, None, vis_i, vis_flags, horizon=1)
        j0, _ = _roll_j(env, policy, runner, injector, batch, n_valid, None, horizon=K_HORIZON, **kwj)
        pack = {}
        a5_raw = torch.zeros(n_valid, k, 2, device=device)
        r_raw = torch.zeros(n_valid, k, device=device)
        di_p_raw = torch.zeros(n_valid, k, device=device)
        di_c_raw = torch.zeros(n_valid, k, device=device)
        for ki in range(k):
            v = Bt_raw[:, :, ki]
            for si, sgn in enumerate((1.0, -1.0)):
                j5, _ = _roll_j(
                    env, policy, runner, injector, batch, n_valid, float(sgn) * tan5 * v,
                    horizon=K_HORIZON, **kwj,
                )
                a5_raw[:, ki, si] = j0 - j5
                y5 = _roll_yI(env, policy, runner, injector, batch, n_valid, float(sgn) * tan5 * v, vis_i, vis_flags, 1)
                if si == 0:
                    y5p = y5
                else:
                    y5m = y5
            yp = _roll_yI(env, policy, runner, injector, batch, n_valid, eps * v, vis_i, vis_flags, 1)
            ym = _roll_yI(env, policy, runner, injector, batch, n_valid, -eps * v, vis_i, vis_flags, 1)
            r_raw[:, ki] = (yp.norm(dim=-1) - ym.norm(dim=-1)) / (2.0 * eps)
            # sign protocol: tracking error, same as P4-A0 channel-0 of Δy which is Δe
            # use ||y_I|| change as e-proxy for torso; also store true DI
            di_p_raw[:, ki] = 0.5 * ((yp - y_par).norm(dim=-1) + (ym - y_par).norm(dim=-1)) / pos_scale
            di_c_raw[:, ki] = 0.5 * ((y5p - y_par).norm(dim=-1) + (y5m - y_par).norm(dim=-1)) / pos_scale
        pack["raw"] = {
            "B": Bt_raw, "a5": a5_raw, "r_e": r_raw, "di_probe": di_p_raw, "di_corr": di_c_raw,
            "Rd": torch.ones(n_valid, k, device=device), "dist": torch.zeros(n_valid, k, device=device),
        }
        print("[p4a1]   raw", flush=True)
        for frac in fracs:
            Bt, Rd, dist = _tube_B(Bt_raw, C, z0, frac, tan5)
            a5 = torch.zeros(n_valid, k, 2, device=device)
            r_e = torch.zeros(n_valid, k, device=device)
            di_p = torch.zeros(n_valid, k, device=device)
            di_c = torch.zeros(n_valid, k, device=device)
            for ki in range(k):
                v = Bt[:, :, ki]
                for si, sgn in enumerate((1.0, -1.0)):
                    j5, _ = _roll_j(
                        env, policy, runner, injector, batch, n_valid, float(sgn) * tan5 * v,
                        horizon=K_HORIZON, **kwj,
                    )
                    a5[:, ki, si] = j0 - j5
                    y5 = _roll_yI(env, policy, runner, injector, batch, n_valid, float(sgn) * tan5 * v, vis_i, vis_flags, 1)
                    if si == 0:
                        y5p = y5
                    else:
                        y5m = y5
                yp = _roll_yI(env, policy, runner, injector, batch, n_valid, eps * v, vis_i, vis_flags, 1)
                ym = _roll_yI(env, policy, runner, injector, batch, n_valid, -eps * v, vis_i, vis_flags, 1)
                r_e[:, ki] = (yp.norm(dim=-1) - ym.norm(dim=-1)) / (2.0 * eps)
                di_p[:, ki] = 0.5 * ((yp - y_par).norm(dim=-1) + (ym - y_par).norm(dim=-1)) / pos_scale
                di_c[:, ki] = 0.5 * ((y5p - y_par).norm(dim=-1) + (y5m - y_par).norm(dim=-1)) / pos_scale
            key = f"f{frac:g}"
            pack[key] = {"B": Bt, "a5": a5, "r_e": r_e, "di_probe": di_p, "di_corr": di_c, "Rd": Rd, "dist": dist}
            print(f"[p4a1]   frac={frac:g}", flush=True)
        for i in range(n_valid):
            s = batch[i]
            item = {
                "clip": s["clip"],
                "seed": int(s["seed"]),
                "t": int(s["t"]),
                "terrain": s["terrain"],
                "window": s["window"],
                "e0": float(s["e0"]),
                "j0": float(j0[i].item()),
                "z_nom": s["z_nom"].numpy().astype(np.float32),
                "J_I": J_I[i].detach().cpu().numpy().astype(np.float32),
            }
            for name in levels:
                p = pack[name]
                item[f"a5_{name}"] = p["a5"][i].detach().cpu().numpy().astype(np.float32)
                item[f"r_e_{name}"] = p["r_e"][i].detach().cpu().numpy().astype(np.float32)
                item[f"di_probe_{name}"] = p["di_probe"][i].detach().cpu().numpy().astype(np.float32)
                item[f"di_corr_{name}"] = p["di_corr"][i].detach().cpu().numpy().astype(np.float32)
                item[f"Rd_{name}"] = p["Rd"][i].detach().cpu().numpy().astype(np.float32)
                item[f"dist_{name}"] = p["dist"][i].detach().cpu().numpy().astype(np.float32)
                item[f"B_{name}"] = p["B"][i].detach().cpu().numpy().astype(np.float32)
            rows.append(item)
    return rows, levels


def _jac_alpha(J_e: torch.Tensor, e0: torch.Tensor, lam=1e-3) -> torch.Tensor:
    """α* = -(J^T J + λI)^{-1} J^T e , J_e [n,k] is ∂e/∂α."""
    n, k = J_e.shape
    eye = torch.eye(k, device=J_e.device, dtype=J_e.dtype).unsqueeze(0).expand(n, -1, -1)
    gram = torch.bmm(J_e.unsqueeze(-1), J_e.unsqueeze(1)) + lam * eye
    rhs = (J_e * e0.unsqueeze(-1)).unsqueeze(-1)
    try:
        alpha = -torch.linalg.solve(gram, rhs).squeeze(-1)
    except Exception:
        alpha = -J_e * e0.unsqueeze(-1)
    return alpha


def _clip_tan(dz: torch.Tensor, z: torch.Tensor, deg=EVAL_DEG) -> torch.Tensor:
    z = F.normalize(z, dim=-1, eps=1e-8)
    d = dz - (dz * z).sum(-1, keepdim=True) * z
    cap = math.tan(math.radians(float(deg)))
    nrm = d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return d * torch.clamp(cap / nrm, max=1.0)


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


def _token(z, e9, de9, proprio, contact2, roll, pitch) -> torch.Tensor:
    prop = proprio[:, -1, :].reshape(z.shape[0], -1)
    if int(prop.shape[-1]) != 90:
        out = z.new_zeros(z.shape[0], 90)
        n = min(90, int(prop.shape[-1]))
        out[:, :n] = prop[:, :n]
        prop = out
    extras = torch.stack([contact2[:, 0], contact2[:, 1], roll, pitch], dim=-1)
    return torch.cat([z, e9, de9, prop, extras], dim=-1)


@torch.inference_mode()
def _collect_snaps(env, policy, runner, injector, steps, seed, terrain, vis_i, vis_flags, mask_name):
    n_envs = int(env.unwrapped.num_envs)
    use, paths = _pin_clips(env, mask_name=mask_name)
    cmd = env.unwrapped.command_manager.get_term("motion")
    device = env.unwrapped.device
    asset = env.unwrapped.scene["robot"]
    am = env.unwrapped.action_manager
    names = list(asset.data.body_names)
    pelvis_i = names.index("pelvis") if "pelvis" in names else None
    ankle_i = [names.index(n) for n in ANKLE_BODIES if n in names]
    try:
        cf = env.unwrapped.scene["contact_forces"]
    except Exception:
        cf = None
    ring = torch.zeros(use, HIST, TOKEN, device=device)
    e9_prev = None
    e_vis_prev = None
    n_rec = [0] * use
    n_nom = [0] * use
    last_snap_t = [-999] * use
    persist_n = [0] * use
    snaps = []
    for t in range(steps):
        obs, _ = env.get_observations()
        obs = injector.patch_policy_obs(obs, env, "mapper")
        if float(args_cli.push_dv) > 0 and t == int(args_cli.push_step):
            root = asset.data.root_state_w.clone()
            root[:, 8] += float(args_cli.push_dv)
            asset.write_root_state_to_sim(root)
            env.unwrapped.sim.forward()
            obs, _ = env.get_observations()
            obs = injector.patch_policy_obs(obs, env, "mapper")
        z_nom, proprio, _enc = _z_and_proprio(policy, runner, obs)
        e9 = _e9(cmd, vis_i)[:use]
        de9 = torch.zeros_like(e9) if e9_prev is None else (e9 - e9_prev) / DT
        e_vis = _visible_e(cmd, vis_i, vis_flags)[:use]
        de_s = torch.zeros_like(e_vis) if e_vis_prev is None else (e_vis - e_vis_prev)
        roll, pitch, _yaw = euler_xyz_from_quat(asset.data.root_quat_w[:use])
        contact2 = torch.zeros(use, 2, device=device)
        if cf is not None and ankle_i and hasattr(cf.data, "net_forces_w"):
            nf = cf.data.net_forces_w[:use]
            contact2 = (nf[:, ankle_i, :].norm(dim=-1) > CONTACT_N).float()
        tok = _token(z_nom[:use], e9, de9, proprio[:use], contact2, roll, pitch)
        ring = torch.cat([ring[:, 1:], tok.unsqueeze(1)], dim=1)
        e_th = float(args_cli.e_rec)
        persist_need = max(1, int(args_cli.persist))
        allow_plane_rec = bool(args_cli.rec_on_plane) or float(args_cli.push_dv) > 0
        if t >= WARMUP:
            hist_now = _hist_snapshot(env)
            for i in range(use):
                if float(e_vis[i]) >= e_th:
                    persist_n[i] += 1
                else:
                    persist_n[i] = 0
                if t - last_snap_t[i] < 40:
                    continue
                rec = bool(
                    persist_n[i] >= persist_need
                    and de_s[i] > 0
                    and (terrain != "plane" or allow_plane_rec)
                )
                nom = bool(terrain == "plane" and e_vis[i] <= float(args_cli.e_nom) and t in (120, 240))
                kind = None
                if rec and n_rec[i] < int(args_cli.max_snaps_rec):
                    kind = "recovery"
                    n_rec[i] += 1
                elif nom and n_nom[i] < int(args_cli.max_snaps_nom):
                    kind = "nominal"
                    n_nom[i] += 1
                if kind is None:
                    continue
                last_snap_t[i] = t
                extra = {
                    "obs": obs[i].detach().cpu().clone(),
                    "z_nom": z_nom[i].detach().cpu().clone(),
                    "hist_tok": ring[i].detach().cpu().clone(),
                    "e0": float(e_vis[i].item()),
                    "de0": float(de_s[i].item()),
                    "t": int(t),
                    "seed": int(seed),
                    "terrain": terrain,
                    "window": kind,
                    "clip": Path(paths[i]).name if i < len(paths) else f"env{i}",
                    "task": str(args_cli.task_source),
                    "trigger_eval_only": kind,
                }
                snaps.append(_capture_snap(env, i, hist_now, extra))
        env.step(_decode(policy, z_nom, proprio))
        e9_prev = e9.clone()
        e_vis_prev = e_vis.clone()
        if (t + 1) % 50 == 0:
            print(
                f"[irr-r0] walk {terrain} s{seed} {t+1}/{steps} e={float(e_vis.mean()):.3f} snaps={len(snaps)}",
                flush=True,
            )
    print(f"[irr-r0] collected {len(snaps)} snaps terrain={terrain} seed={seed}", flush=True)
    return snaps, pelvis_i, use, paths


@torch.inference_mode()
def _probe_on_snaps(env, policy, runner, injector, snaps, vis_i, vis_flags, pelvis_i, n_axes, B_shared):
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    asset = env.unwrapped.scene["robot"]
    names = list(asset.data.body_names)
    ankle_i = [names.index(n) for n in ANKLE_BODIES if n in names]
    try:
        cf = env.unwrapped.scene["contact_forces"]
    except Exception:
        cf = None
    eps = math.tan(math.radians(EPS_DEG))
    tan5 = math.tan(math.radians(EVAL_DEG))
    n_l = len(PROBE_LS)
    k_max = int(B_shared.shape[1]) if B_shared is not None else 4
    B_t = torch.as_tensor(B_shared, device=device, dtype=torch.float32) if B_shared is not None else None
    rows = []
    kwj = dict(vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i)
    kwy = dict(vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i, ankle_i=ankle_i, cf=cf)
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[irr-r3] batch {start}:{start + n_valid}/{len(snaps)} axes={n_axes}", flush=True)
        z0 = torch.stack([s["z_nom"] for s in batch], dim=0).to(device=device)
        basis = tangent_basis(z0)
        j0, _f0 = _roll_j(env, policy, runner, injector, batch, n_valid, None, horizon=K_HORIZON, **kwj)
        a5 = torch.zeros(n_valid, n_axes, 2, device=device)
        a1 = torch.zeros(n_valid, n_axes, 2, device=device)
        r_clone = torch.zeros(n_valid, n_l, n_axes, Y_DIM, device=device)
        r_seq = torch.zeros(n_valid, n_l, n_axes, Y_DIM, device=device)
        e0_y = torch.zeros(n_valid, device=device)
        for ax in range(n_axes):
            v = basis[:, :, ax]
            for si, sgn in enumerate((1.0, -1.0)):
                dz5 = float(sgn) * tan5 * v
                dz1 = float(sgn) * eps * v
                j5, _ = _roll_j(env, policy, runner, injector, batch, n_valid, dz5, horizon=K_HORIZON, **kwj)
                j1, _ = _roll_j(env, policy, runner, injector, batch, n_valid, dz1, horizon=K_HORIZON, **kwj)
                a5[:, ax, si] = j0 - j5
                a1[:, ax, si] = j0 - j1
            print(f"[irr-r3]   labels axis {ax + 1}/{n_axes}", flush=True)
            for li, L in enumerate(PROBE_LS):
                dyp, y0, e0_b = _roll_y(env, policy, runner, injector, batch, n_valid, eps * v, horizon=L, **kwy)
                dym, _, _ = _roll_y(env, policy, runner, injector, batch, n_valid, -eps * v, horizon=L, **kwy)
                r_clone[:, li, ax] = (dyp - dym) / (2.0 * eps)
                if li == 0 and ax == 0:
                    e0_y = e0_b
                dyp_s, dym_s = _roll_y_seq(
                    env, policy, runner, injector, batch, n_valid, eps * v, -eps * v, horizon=L, **kwy
                )
                r_seq[:, li, ax] = (dyp_s - dym_s) / (2.0 * eps)
        # shared-B probes + Jacobian + SPSA
        a_jac_c = torch.zeros(n_valid, n_l, 3, device=device)  # k=2,3,4
        a_jac_s = torch.zeros(n_valid, n_l, 3, device=device)
        a_spsa_c = torch.zeros(n_valid, n_l, device=device)
        a_spsa_s = torch.zeros(n_valid, n_l, device=device)
        rB_c = torch.zeros(n_valid, n_l, k_max, Y_DIM, device=device)
        rB_s = torch.zeros(n_valid, n_l, k_max, Y_DIM, device=device)
        aB = torch.zeros(n_valid, k_max, 2, device=device)
        if B_t is not None:
            Bt = _tangent_B(z0, B_t)
            for ki in range(k_max):
                v = Bt[:, :, ki]
                for si, sgn in enumerate((1.0, -1.0)):
                    j5, _ = _roll_j(
                        env, policy, runner, injector, batch, n_valid, float(sgn) * tan5 * v,
                        horizon=K_HORIZON, **kwj,
                    )
                    aB[:, ki, si] = j0 - j5
            print("[irr-r3]   B 5° labels", flush=True)
            Bt = _tangent_B(z0, B_t)
            for li, L in enumerate(PROBE_LS):
                for ki in range(k_max):
                    v = Bt[:, :, ki]
                    dyp, _, _ = _roll_y(env, policy, runner, injector, batch, n_valid, eps * v, horizon=L, **kwy)
                    dym, _, _ = _roll_y(env, policy, runner, injector, batch, n_valid, -eps * v, horizon=L, **kwy)
                    rB_c[:, li, ki] = (dyp - dym) / (2.0 * eps)
                    dyp_s, dym_s = _roll_y_seq(
                        env, policy, runner, injector, batch, n_valid, eps * v, -eps * v, horizon=L, **kwy
                    )
                    rB_s[:, li, ki] = (dyp_s - dym_s) / (2.0 * eps)
                for kk, kdim in enumerate((2, 3, 4)):
                    if kdim > k_max:
                        continue
                    Je = rB_c[:, li, :kdim, 0]
                    alpha = _jac_alpha(Je, e0_y)
                    dz = torch.zeros(n_valid, 16, device=device)
                    for ki in range(kdim):
                        dz = dz + alpha[:, ki : ki + 1] * Bt[:, :, ki]
                    dz = _clip_tan(dz, z0)
                    jx, _ = _roll_j(env, policy, runner, injector, batch, n_valid, dz, horizon=K_HORIZON, **kwj)
                    a_jac_c[:, li, kk] = j0 - jx
                    Je_s = rB_s[:, li, :kdim, 0]
                    alpha_s = _jac_alpha(Je_s, e0_y)
                    dzs = torch.zeros(n_valid, 16, device=device)
                    for ki in range(kdim):
                        dzs = dzs + alpha_s[:, ki : ki + 1] * Bt[:, :, ki]
                    dzs = _clip_tan(dzs, z0)
                    jxs, _ = _roll_j(env, policy, runner, injector, batch, n_valid, dzs, horizon=K_HORIZON, **kwj)
                    a_jac_s[:, li, kk] = j0 - jxs
                print(f"[irr-r3]   B/jac L={L}", flush=True)
        rng_seed = 1000 + start
        torch.manual_seed(rng_seed)
        for li, L in enumerate(PROBE_LS):
            u = torch.randn(n_valid, 16, device=device, dtype=z0.dtype)
            u = project_tangent(u, z0)
            dyp, _, _ = _roll_y(env, policy, runner, injector, batch, n_valid, eps * u, horizon=L, **kwy)
            dym, _, _ = _roll_y(env, policy, runner, injector, batch, n_valid, -eps * u, horizon=L, **kwy)
            g = (dyp[:, 0] - dym[:, 0]) / (2.0 * eps)
            dz = _clip_tan((-g).unsqueeze(-1) * u, z0)
            js, _ = _roll_j(env, policy, runner, injector, batch, n_valid, dz, horizon=K_HORIZON, **kwj)
            a_spsa_c[:, li] = j0 - js
            dyp_s, dym_s = _roll_y_seq(
                env, policy, runner, injector, batch, n_valid, eps * u, -eps * u, horizon=L, **kwy
            )
            gs = (dyp_s[:, 0] - dym_s[:, 0]) / (2.0 * eps)
            dzs = _clip_tan((-gs).unsqueeze(-1) * u, z0)
            jss, _ = _roll_j(env, policy, runner, injector, batch, n_valid, dzs, horizon=K_HORIZON, **kwj)
            a_spsa_s[:, li] = j0 - jss
            print(f"[irr-r3]   spsa L={L}", flush=True)
        for i in range(n_valid):
            s = batch[i]
            rows.append(
                {
                    "clip": s["clip"],
                    "seed": int(s["seed"]),
                    "t": int(s["t"]),
                    "terrain": s["terrain"],
                    "window": s["window"],
                    "task": s["task"],
                    "e0": float(s["e0"]),
                    "j0": float(j0[i].item()),
                    "a5": a5[i].detach().cpu().numpy().astype(np.float32),
                    "a1": a1[i].detach().cpu().numpy().astype(np.float32),
                    "r_clone": r_clone[i].detach().cpu().numpy().astype(np.float32),
                    "r_seq": r_seq[i].detach().cpu().numpy().astype(np.float32),
                    "rB_clone": rB_c[i].detach().cpu().numpy().astype(np.float32),
                    "rB_seq": rB_s[i].detach().cpu().numpy().astype(np.float32),
                    "a_jac_clone": a_jac_c[i].detach().cpu().numpy().astype(np.float32),
                    "a_jac_seq": a_jac_s[i].detach().cpu().numpy().astype(np.float32),
                    "a_spsa_clone": a_spsa_c[i].detach().cpu().numpy().astype(np.float32),
                    "a_spsa_seq": a_spsa_s[i].detach().cpu().numpy().astype(np.float32),
                    "aB": aB[i].detach().cpu().numpy().astype(np.float32),
                    "z_nom": s["z_nom"].numpy().astype(np.float32),
                }
            )
    return rows


def _poly_code(n_step: int, k: int = 3, seed: int = 2026):
    """U (N,k): columns orthonormal in P^perp, P=[1,t,t^2], then scale max row-norm to 1."""
    rng = np.random.RandomState(int(seed) + int(n_step) * 17)
    t = np.linspace(0.0, 1.0, int(n_step), dtype=np.float64)
    P = np.stack([np.ones(n_step), t, t * t], axis=1)
    qp, _ = np.linalg.qr(P, mode="reduced")
    rnk = min(3, int(qp.shape[1]))
    null = np.eye(n_step) - qp[:, :rnk] @ qp[:, :rnk].T
    raw = null @ rng.randn(n_step, k)
    U, _ = np.linalg.qr(raw, mode="reduced")
    U = U[:, :k]
    rn = np.linalg.norm(U, axis=1)
    U = U / max(float(rn.max()), 1e-8)
    return U.astype(np.float64), P.astype(np.float64)


def _fit_design(Y: np.ndarray, U: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Y (N,dy) = P β + U J^T → J (dy,k). Non-finite rows dropped."""
    n, dy = Y.shape
    k = int(U.shape[1])
    X = np.hstack([P[:n], U[:n]])
    J = np.zeros((dy, k), dtype=np.float64)
    for j in range(dy):
        yj = np.asarray(Y[:, j], dtype=np.float64)
        m = np.isfinite(yj) & np.isfinite(X).all(axis=1)
        if int(m.sum()) < 2:
            continue
        coef, *_ = np.linalg.lstsq(X[m], yj[m], rcond=None)
        J[j] = coef[P.shape[1] : P.shape[1] + k]
    return J


def _read_p4b_y(env, vis_i, vis_flags, e_prev):
    cmd = env.unwrapped.command_manager.get_term("motion")
    asset = env.unwrapped.scene["robot"]
    e = _visible_e(cmd, vis_i, vis_flags)
    de = torch.zeros_like(e) if e_prev is None else (e - e_prev) / DT
    vel = getattr(asset.data, "root_lin_vel_b", asset.data.root_lin_vel_w)
    omega = getattr(asset.data, "root_ang_vel_b", asset.data.root_ang_vel_w)
    g_w = asset.data.GRAVITY_VEC_W
    if g_w.ndim == 1:
        g_w = g_w.unsqueeze(0).expand(e.shape[0], -1)
    g = quat_rotate_inverse(asset.data.root_quat_w, g_w)
    y = torch.cat([e.unsqueeze(-1), de.unsqueeze(-1), vel[:, :3], omega[:, :3], g[:, :2]], dim=-1)
    return y, e.detach()


def _dstar_from_g(g: torch.Tensor, z: torch.Tensor, B_t: torch.Tensor) -> torch.Tensor:
    """α* = -g/||g||, d* = tangent-QR(B) α*. g (n,k), z (n,16). Zeros stay zero."""
    z = F.normalize(z, dim=-1, eps=1e-8)
    nrm = g.norm(dim=-1, keepdim=True)
    alpha = torch.where(nrm > 1e-10, -g / nrm.clamp(min=1e-10), torch.zeros_like(g))
    Bt = _tangent_B(z, B_t)
    d = torch.bmm(Bt, alpha.unsqueeze(-1)).squeeze(-1)
    d = d - (d * z).sum(-1, keepdim=True) * z
    dn = d.norm(dim=-1, keepdim=True)
    return torch.where(dn > 1e-8, d / dn.clamp(min=1e-8), torch.zeros_like(d))


@torch.inference_mode()
def _seq_coded_probe(env, policy, runner, injector, batch, n_valid, B, U, vis_i, vis_flags, tube_abs):
    """Continuous N-step coded multiplex. Combined geodesic ≤ 1°. Tube guard on ΔE_I."""
    n_envs = int(env.unwrapped.num_envs)
    padded, _ = _pad_snaps(batch, n_envs)
    _restore_batch(env, padded)
    device = env.unwrapped.device
    n_step = int(U.shape[0])
    dy = 10
    B_t = torch.as_tensor(B, device=device, dtype=torch.float32)
    U_t = torch.as_tensor(U, device=device, dtype=torch.float32)
    tan1 = math.tan(math.radians(EPS_DEG))
    obs_stack = torch.stack([s["obs"].to(device) for s in padded], dim=0)
    Y = torch.full((n_valid, n_step, dy), float("nan"), device=device)
    E = torch.zeros(n_valid, n_step, device=device)
    abort = torch.zeros(n_valid, dtype=torch.bool, device=device)
    contact = torch.zeros(n_valid, n_step, 2, device=device)
    _y0, e0 = _read_p4b_y(env, vis_i, vis_flags, None)
    e_prev = e0
    names = list(env.unwrapped.scene["robot"].data.body_names)
    ankle_i = [names.index(n) for n in ANKLE_BODIES if n in names]
    try:
        cf = env.unwrapped.scene["contact_forces"]
    except Exception:
        cf = None
    for t in range(n_step):
        was_abort = abort.clone()
        obs = obs_stack if t == 0 else injector.patch_policy_obs(env.get_observations()[0], env, "mapper")
        z_nom, proprio, _enc = _z_and_proprio(policy, runner, obs)
        Bt = _tangent_B(z_nom, B_t)
        u = U_t[t]
        v = torch.einsum("njk,k->nj", Bt, u)
        v = project_tangent(v, z_nom)
        nrm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        mag = tan1 * min(1.0, float(np.linalg.norm(U[t])))
        z_exec = F.normalize(z_nom + mag * v / nrm, dim=-1, eps=1e-8)
        ab = torch.zeros(n_envs, 1, device=device, dtype=torch.bool)
        ab[:n_valid] = was_abort.view(-1, 1)
        z_use = torch.where(ab, z_nom, z_exec)
        env.step(_decode(policy, z_use, proprio))
        y, e = _read_p4b_y(env, vis_i, vis_flags, e_prev)
        keep = ~was_abort
        Y[keep, t] = y[:n_valid][keep]
        E[:, t] = e[:n_valid]
        if cf is not None and ankle_i and hasattr(cf.data, "net_forces_w"):
            contact[:, t] = (cf.data.net_forces_w[:n_valid][:, ankle_i, :].norm(dim=-1) > CONTACT_N).float()
        abort = abort | ((e[:n_valid] - e0[:n_valid]) > float(tube_abs))
        e_prev = e
    return Y.detach(), E.detach(), abort.detach(), e0[:n_valid].detach(), contact.detach()


@torch.inference_mode()
def _anet_traj(env, policy, runner, injector, batch, n_valid, B, U, dstar, vis_i, vis_flags, pelvis_i, n_step, horizon):
    """N coded probe then H-step 5° correction along dstar. Parent is Stage-2 for N+H."""
    n_envs = int(env.unwrapped.num_envs)
    padded, _ = _pad_snaps(batch, n_envs)
    device = env.unwrapped.device
    B_t = torch.as_tensor(B, device=device, dtype=torch.float32)
    U_t = torch.as_tensor(U, device=device, dtype=torch.float32)
    tan1 = math.tan(math.radians(EPS_DEG))
    tan5 = math.tan(math.radians(EVAL_DEG))
    kwj = dict(vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i)
    j_par, _ = _roll_j(env, policy, runner, injector, batch, n_valid, None, horizon=n_step + horizon, **kwj)
    _restore_batch(env, padded)
    cmd = env.unwrapped.command_manager.get_term("motion")
    asset = env.unwrapped.scene["robot"]
    gravity = asset.data.GRAVITY_VEC_W
    if gravity.ndim == 1:
        gravity = gravity.unsqueeze(0).expand(cmd.num_envs, -1)
    failed = torch.zeros(n_envs, dtype=torch.bool, device=device)
    j = torch.zeros(n_envs, device=device, dtype=torch.float32)
    obs_stack = torch.stack([s["obs"].to(device) for s in padded], dim=0)
    dpad = torch.zeros(n_envs, 16, device=device)
    if dstar is not None:
        dpad[:n_valid] = dstar
    e_hist = []
    for t in range(n_step + horizon):
        obs = obs_stack if t == 0 else injector.patch_policy_obs(env.get_observations()[0], env, "mapper")
        z_nom, proprio, _enc = _z_and_proprio(policy, runner, obs)
        if t < n_step:
            Bt = _tangent_B(z_nom, B_t)
            u = U_t[t]
            v = torch.einsum("njk,k->nj", Bt, u)
            v = project_tangent(v, z_nom)
            nrm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            mag = tan1 * min(1.0, float(np.linalg.norm(U[t])))
            z_exec = F.normalize(z_nom + mag * v / nrm, dim=-1, eps=1e-8)
        else:
            z_exec = F.normalize(z_nom + tan5 * dpad, dim=-1, eps=1e-8)
        env.step(_decode(policy, z_exec, proprio))
        e = _visible_e(cmd, vis_i, vis_flags)
        fail = _official_fail(cmd, asset, vis_i, n_envs, pelvis_i, gravity)
        failed = failed | fail
        j = j + (GAMMA ** t) * (e + LAM_S * fail.to(dtype=e.dtype))
        e_hist.append(e[:n_valid].detach())
    return j_par, j[:n_valid].detach(), torch.stack(e_hist, dim=1)


@torch.inference_mode()
def _probe_p4b(env, policy, runner, injector, snaps, vis_i, vis_flags, pelvis_i, B_raw_np, ns, terrain):
    """Coded multiplex online ID. No projector. No sequential ±pair. No networks."""
    del terrain
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    k = int(args_cli.p4a0_k)
    B = np.asarray(B_raw_np[:, :k], dtype=np.float64)
    B_t = torch.as_tensor(B, device=device, dtype=torch.float32)
    tube_abs = float(args_cli.tube_guard) * float(args_cli.pos_scale)
    tan5 = math.tan(math.radians(EVAL_DEG))
    kwj = dict(vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i)
    code_seed = int(getattr(args_cli, "p4b_code_seed", 2026))
    codes = {int(n): _poly_code(int(n), k, seed=code_seed) for n in ns}
    estimators = ("E1", "E2", "scalar")
    rows = []
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[p4b] batch {start}:{start + n_valid}/{len(snaps)}", flush=True)
        z0 = torch.stack([s["z_nom"] for s in batch], dim=0).to(device)
        j0, _ = _roll_j(env, policy, runner, injector, batch, n_valid, None, horizon=K_HORIZON, **kwj)
        Bt0 = _tangent_B(z0, B_t)
        a5_gt = torch.zeros(n_valid, k, 2, device=device)
        for ki in range(k):
            v = Bt0[:, :, ki]
            for si, sgn in enumerate((1.0, -1.0)):
                j5, _ = _roll_j(
                    env, policy, runner, injector, batch, n_valid, float(sgn) * tan5 * v,
                    horizon=K_HORIZON, **kwj,
                )
                a5_gt[:, ki, si] = j0 - j5
        rec = {int(n): {est: {} for est in estimators} for n in ns}
        di_max = {int(n): np.zeros(n_valid, dtype=np.float64) for n in ns}
        Eseq = {}
        for n_step in ns:
            n_step = int(n_step)
            U, P = codes[n_step]
            Y, E, abort, e0, _c = _seq_coded_probe(
                env, policy, runner, injector, batch, n_valid, B, U, vis_i, vis_flags, tube_abs
            )
            Yn, En, abn = Y.cpu().numpy(), E.cpu().numpy(), abort.cpu().numpy().astype(bool)
            e0n = e0.cpu().numpy()
            dE = En - e0n[:, None]
            with np.errstate(all="ignore"):
                mx = np.nanmax(dE, axis=1)
            di_max[n_step] = np.where(np.isfinite(mx), mx, 0.0)
            Eseq[n_step] = dE.astype(np.float32)
            g_np = {est: np.zeros((n_valid, k), dtype=np.float64) for est in estimators}
            t_inc = np.linspace(0.0, 1.0, max(n_step - 1, 1), dtype=np.float64)
            P2 = np.stack([np.ones_like(t_inc), t_inc, t_inc * t_inc], axis=1)
            for i in range(n_valid):
                J1 = _fit_design(Yn[i], U, P)
                g1 = J1[0]
                c = Yn[i, :, 0:1]
                g_s = _fit_design(c, U, P)[0]
                if n_step >= 3:
                    dY = np.diff(Yn[i], axis=0)
                    J2 = _fit_design(dY, U[:-1], P2)
                    g2 = J2[0]
                else:
                    g2 = np.zeros(k, dtype=np.float64)
                g_np["E1"][i] = g1
                g_np["E2"][i] = g2
                g_np["scalar"][i] = g_s
            for est in estimators:
                g_t = torch.as_tensor(g_np[est], device=device, dtype=torch.float32)
                dstar = _dstar_from_g(g_t, z0, B_t)
                ab_t = torch.as_tensor(abn, device=device)
                dstar = torch.where(ab_t.view(-1, 1), torch.zeros_like(dstar), dstar)
                weak = dstar.norm(dim=-1) < 1e-8
                rec[n_step][est]["g"] = g_np[est]
                rec[n_step][est]["d"] = dstar.detach().cpu().numpy()
                rec[n_step][est]["abort"] = abn
                if est == "scalar":
                    rec[n_step][est]["A"] = rec[n_step]["E1"]["A"]
                    rec[n_step][est]["A_net"] = rec[n_step]["E1"]["A_net"]
                    rec[n_step][est]["d"] = rec[n_step]["E1"]["d"]
                    continue
                j5, _ = _roll_j(env, policy, runner, injector, batch, n_valid, tan5 * dstar, horizon=K_HORIZON, **kwj)
                a_clone = (j0 - j5).detach().cpu().numpy()
                a_clone[abn | weak.cpu().numpy()] = 0.0
                rec[n_step][est]["A"] = a_clone
                jpar, jnet, _eh = _anet_traj(
                    env, policy, runner, injector, batch, n_valid, B, U, dstar,
                    vis_i, vis_flags, pelvis_i, n_step, K_HORIZON,
                )
                a_net = (jpar - jnet).detach().cpu().numpy()
                a_net[abn] = 0.0
                rec[n_step][est]["A_net"] = a_net
            print(f"[p4b]   N={n_step} abort={int(abn.sum())}/{n_valid}", flush=True)
        a5n = a5_gt.detach().cpu().numpy().astype(np.float32)
        for i in range(n_valid):
            s = batch[i]
            item = {
                "clip": s["clip"],
                "seed": int(s["seed"]),
                "t": int(s["t"]),
                "terrain": s["terrain"],
                "window": s.get("window", "recovery"),
                "e0": float(s["e0"]),
                "j0": float(j0[i].item()),
                "z_nom": s["z_nom"].numpy().astype(np.float32),
                "a5_gt": a5n[i],
                "ns": {},
            }
            for n_step in ns:
                n_step = int(n_step)
                blk = {"di_max": float(di_max[n_step][i]), "dE": Eseq[n_step][i], "est": {}}
                for est in estimators:
                    blk["est"][est] = {
                        "g": rec[n_step][est]["g"][i].astype(np.float32),
                        "d": rec[n_step][est]["d"][i].astype(np.float32),
                        "A": float(rec[n_step][est]["A"][i]),
                        "A_net": float(rec[n_step][est]["A_net"][i]),
                        "abort": bool(rec[n_step][est]["abort"][i]),
                    }
                item["ns"][str(n_step)] = blk
            rows.append(item)
    return rows, codes


def _load_p4b_catalog(terrain: str):
    """Reuse P4-B snapshot IDs, a5_gt, and exact coded U/P."""
    root = Path(getattr(args_cli, "p4b_dir", "/data/home/chenxiangyu/robotics/Anybody/results/p4b_tube_online_id"))
    p = root / "p4b" / "loco" / terrain / "p4b.npz"
    if not p.is_file():
        raise FileNotFoundError(f"P4-B2 requires P4-B catalog {p}")
    z = np.load(p, allow_pickle=True)
    keys, a5, e0, znom = [], {}, {}, {}
    for i in range(int(z["e0"].shape[0])):
        key = (int(z["seed"][i]), int(z["t"][i]), str(z["clip"][i]))
        keys.append(key)
        a5[key] = z["a5_gt"][i].astype(np.float32)
        e0[key] = float(z["e0"][i])
        znom[key] = z["z_nom"][i].astype(np.float32)
    codes = {}
    ns = [int(x) for x in z["ns"].tolist()] if "ns" in z.files else [6, 8, 10]
    for n_step in ns:
        U = np.asarray(z[f"U_{n_step}"], dtype=np.float64)
        P = np.asarray(z[f"P_{n_step}"], dtype=np.float64)
        codes[int(n_step)] = (U, P)
    return keys, a5, e0, znom, codes


def _align_p4b_snaps(snaps: list[dict], terrain: str):
    keys, a5, e0, znom, codes = _load_p4b_catalog(terrain)
    by = {}
    for s in snaps:
        by[(int(s["seed"]), int(s["t"]), str(s["clip"]))] = s
    out, miss = [], []
    for key in keys:
        if key not in by:
            miss.append(key)
            continue
        s = by[key]
        s = dict(s)
        s["a5_gt"] = a5[key]
        s["e0_p4b"] = e0[key]
        s["z_p4b"] = znom[key]
        out.append(s)
    if miss:
        raise RuntimeError(f"P4-B2 missing {len(miss)}/{len(keys)} snapshot IDs on {terrain}: {miss[:5]}")
    if len(out) != len(keys):
        raise RuntimeError(f"P4-B2 aligned {len(out)} != catalog {len(keys)} on {terrain}")
    return out, codes


def _state_fp(env, n_valid: int) -> dict:
    uw = env.unwrapped
    robot = uw.scene["robot"]
    cmd = uw.command_manager.get_term("motion")
    return {
        "q": robot.data.joint_pos[:n_valid].detach().clone(),
        "dq": robot.data.joint_vel[:n_valid].detach().clone(),
        "root": robot.data.root_state_w[:n_valid].detach().clone(),
        "time_steps": cmd.time_steps[:n_valid].detach().clone(),
        "motion_idx": cmd.env_motion_indices[:n_valid].detach().clone(),
    }


def _fp_err(a: dict, b: dict) -> dict:
    out = {}
    for k in a:
        da, db = a[k].float(), b[k].float()
        out[k] = float((da - db).abs().max().item())
    return out


def _coded_z(z_nom, B_t, U_t, U_np, t: int):
    Bt = _tangent_B(z_nom, B_t)
    v = torch.einsum("njk,k->nj", Bt, U_t[t])
    v = project_tangent(v, z_nom)
    nrm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    mag = math.tan(math.radians(EPS_DEG)) * min(1.0, float(np.linalg.norm(U_np[t])))
    return F.normalize(z_nom + mag * v / nrm, dim=-1, eps=1e-8)


@torch.inference_mode()
def _roll_ye(env, policy, runner, injector, batch, n_valid, n_step, vis_i, vis_flags, mode, B=None, U=None):
    """Stage2 twin or coded probe. Never aborts. Returns Y, E after each step, e0 at t0."""
    n_envs = int(env.unwrapped.num_envs)
    padded, _ = _pad_snaps(batch, n_envs)
    _restore_batch(env, padded)
    device = env.unwrapped.device
    dy = 10
    B_t = torch.as_tensor(B, device=device, dtype=torch.float32) if B is not None else None
    U_t = torch.as_tensor(U, device=device, dtype=torch.float32) if U is not None else None
    obs_stack = torch.stack([s["obs"].to(device) for s in padded], dim=0)
    Y = torch.zeros(n_valid, n_step, dy, device=device)
    E = torch.zeros(n_valid, n_step, device=device)
    _y0, e0 = _read_p4b_y(env, vis_i, vis_flags, None)
    e_prev = e0
    z0_live = None
    for t in range(int(n_step)):
        obs = obs_stack if t == 0 else injector.patch_policy_obs(env.get_observations()[0], env, "mapper")
        z_nom, proprio, _enc = _z_and_proprio(policy, runner, obs)
        if t == 0:
            z0_live = z_nom[:n_valid].detach()
        if mode == "probe":
            z_exec = _coded_z(z_nom, B_t, U_t, U, t)
        else:
            z_exec = z_nom
        env.step(_decode(policy, z_exec, proprio))
        y, e = _read_p4b_y(env, vis_i, vis_flags, e_prev)
        Y[:, t] = y[:n_valid]
        E[:, t] = e[:n_valid]
        e_prev = e
    return Y.detach(), E.detach(), e0[:n_valid].detach(), z0_live


@torch.inference_mode()
def _probe_p4b2(env, policy, runner, injector, snaps, vis_i, vis_flags, pelvis_i, B_raw_np, ns, codes):
    """Twin-referenced online ID. No abort. Full n. Same E1/E2/scalar and U as P4-B."""
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    k = int(args_cli.p4a0_k)
    B = np.asarray(B_raw_np[:, :k], dtype=np.float64)
    B_t = torch.as_tensor(B, device=device, dtype=torch.float32)
    tan5 = math.tan(math.radians(EVAL_DEG))
    tube_abs = float(args_cli.tube_guard) * float(args_cli.pos_scale)
    kwj = dict(vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i)
    estimators = ("E1", "E2", "scalar")
    g_eps = 1e-10
    rows = []
    twin_assert = []
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[p4b2] batch {start}:{start + n_valid}/{len(snaps)}", flush=True)
        padded, _ = _pad_snaps(batch, n_envs)
        _restore_batch(env, padded)
        fp1 = _state_fp(env, n_valid)
        _restore_batch(env, padded)
        fp2 = _state_fp(env, n_valid)
        ferr = _fp_err(fp1, fp2)
        twin_assert.append(ferr)
        print(f"[p4b2]   restore max-abs {ferr}", flush=True)
        z0 = torch.stack([s["z_nom"] for s in batch], dim=0).to(device)
        j0, _ = _roll_j(env, policy, runner, injector, batch, n_valid, None, horizon=K_HORIZON, **kwj)
        rec = {int(n): {est: {} for est in estimators} for n in ns}
        traj = {}
        for n_step in ns:
            n_step = int(n_step)
            U, P = codes[n_step]
            Ya, Ea, e0a, _z = _roll_ye(
                env, policy, runner, injector, batch, n_valid, n_step, vis_i, vis_flags, "twin"
            )
            Yb, Eb, e0b, _zb = _roll_ye(
                env, policy, runner, injector, batch, n_valid, n_step, vis_i, vis_flags, "twin"
            )
            Yp, Ep, e0p, zlive = _roll_ye(
                env, policy, runner, injector, batch, n_valid, n_step, vis_i, vis_flags, "probe", B, U
            )
            e0n = e0p.cpu().numpy()
            Et = Ea.cpu().numpy()
            Etb = Eb.cpu().numpy()
            Epn = Ep.cpu().numpy()
            Yn = Yp.cpu().numpy()
            excess = Epn - Et
            excess_pos = np.maximum(excess, 0.0)
            d_old = Epn - e0n[:, None]
            d_nom = Et - e0n[:, None]
            peak_old = np.max(d_old, axis=1)
            peak_ex = np.max(excess_pos, axis=1)
            peak_nom = np.max(np.abs(d_nom), axis=1)
            auc_ex = np.sum(excess_pos, axis=1) * DT
            auc_nom = np.sum(np.abs(d_nom), axis=1) * DT
            auc_noise = np.sum(np.abs(Et - Etb), axis=1) * DT
            peak_noise = np.max(np.abs(Et - Etb), axis=1)
            r_probe = auc_ex / (auc_nom + 1e-8)
            old_step = np.full(n_valid, -1, dtype=np.int32)
            twin_step = np.full(n_valid, -1, dtype=np.int32)
            for i in range(n_valid):
                w = np.where(d_old[i] > tube_abs)[0]
                old_step[i] = int(w[0]) if len(w) else -1
                w2 = np.where(excess_pos[i] > tube_abs)[0]
                twin_step[i] = int(w2[0]) if len(w2) else -1
            traj[n_step] = {
                "E_twin": Et.astype(np.float32),
                "E_twinB": Etb.astype(np.float32),
                "E_probe": Epn.astype(np.float32),
                "Y_probe": Yn.astype(np.float32),
                "e0": e0n.astype(np.float32),
                "peak_old": peak_old.astype(np.float32),
                "peak_ex": peak_ex.astype(np.float32),
                "peak_nom": peak_nom.astype(np.float32),
                "peak_noise": peak_noise.astype(np.float32),
                "auc_ex": auc_ex.astype(np.float32),
                "auc_nom": auc_nom.astype(np.float32),
                "auc_noise": auc_noise.astype(np.float32),
                "r_probe": r_probe.astype(np.float32),
                "old_step": old_step,
                "twin_step": twin_step,
                "e0_twin_err": (e0a - e0p).abs().cpu().numpy().astype(np.float32),
                "e0_twinB_err": (e0b - e0p).abs().cpu().numpy().astype(np.float32),
            }
            t_inc = np.linspace(0.0, 1.0, max(n_step - 1, 1), dtype=np.float64)
            P2 = np.stack([np.ones_like(t_inc), t_inc, t_inc * t_inc], axis=1)
            g_np = {est: np.zeros((n_valid, k), dtype=np.float64) for est in estimators}
            valid = {est: np.zeros(n_valid, dtype=bool) for est in estimators}
            for i in range(n_valid):
                J1 = _fit_design(Yn[i], U, P)
                g1 = J1[0]
                g_s = _fit_design(Yn[i, :, 0:1], U, P)[0]
                if n_step >= 3:
                    g2 = _fit_design(np.diff(Yn[i], axis=0), U[:-1], P2)[0]
                else:
                    g2 = np.zeros(k, dtype=np.float64)
                g_np["E1"][i] = g1
                g_np["E2"][i] = g2
                g_np["scalar"][i] = g_s
                valid["E1"][i] = bool(np.isfinite(g1).all() and float(np.linalg.norm(g1)) > g_eps)
                valid["E2"][i] = bool(np.isfinite(g2).all() and float(np.linalg.norm(g2)) > g_eps)
                valid["scalar"][i] = bool(np.isfinite(g_s).all() and float(np.linalg.norm(g_s)) > g_eps)
            for est in estimators:
                g_t = torch.as_tensor(g_np[est], device=device, dtype=torch.float32)
                dstar = _dstar_from_g(g_t, z0, B_t)
                inv = ~torch.as_tensor(valid[est], device=device)
                dstar = torch.where(inv.view(-1, 1), torch.zeros_like(dstar), dstar)
                rec[n_step][est]["g"] = g_np[est]
                rec[n_step][est]["d"] = dstar.detach().cpu().numpy()
                rec[n_step][est]["valid"] = valid[est]
                if est == "scalar":
                    rec[n_step][est]["A"] = rec[n_step]["E1"]["A"]
                    rec[n_step][est]["A_net"] = rec[n_step]["E1"]["A_net"]
                    rec[n_step][est]["d"] = rec[n_step]["E1"]["d"]
                    rec[n_step][est]["valid"] = valid["E1"] & valid["scalar"]
                    continue
                j5, _ = _roll_j(env, policy, runner, injector, batch, n_valid, tan5 * dstar, horizon=K_HORIZON, **kwj)
                a_clone = (j0 - j5).detach().cpu().numpy()
                a_clone[~valid[est]] = 0.0
                rec[n_step][est]["A"] = a_clone
                jpar, jnet, _eh = _anet_traj(
                    env, policy, runner, injector, batch, n_valid, B, U, dstar,
                    vis_i, vis_flags, pelvis_i, n_step, K_HORIZON,
                )
                rec[n_step][est]["A_net"] = (jpar - jnet).detach().cpu().numpy()
            print(
                f"[p4b2]   N={n_step} old_viol={(old_step>=0).mean():.3f} "
                f"twin_viol={(twin_step>=0).mean():.3f} "
                f"ex_med={1000*np.median(peak_ex):.2f}mm noise_med={1000*np.median(peak_noise):.4f}mm",
                flush=True,
            )
        for i in range(n_valid):
            s = batch[i]
            item = {
                "clip": s["clip"],
                "seed": int(s["seed"]),
                "t": int(s["t"]),
                "terrain": s["terrain"],
                "window": s.get("window", "recovery"),
                "e0": float(s["e0"]),
                "e0_p4b": float(s.get("e0_p4b", s["e0"])),
                "j0": float(j0[i].item()),
                "z_nom": s["z_nom"].numpy().astype(np.float32),
                "a5_gt": np.asarray(s["a5_gt"], dtype=np.float32),
                "ns": {},
            }
            for n_step in ns:
                n_step = int(n_step)
                tr = traj[n_step]
                blk = {
                    "E_twin": tr["E_twin"][i],
                    "E_twinB": tr["E_twinB"][i],
                    "E_probe": tr["E_probe"][i],
                    "Y_probe": tr["Y_probe"][i],
                    "peak_old": float(tr["peak_old"][i]),
                    "peak_ex": float(tr["peak_ex"][i]),
                    "peak_nom": float(tr["peak_nom"][i]),
                    "peak_noise": float(tr["peak_noise"][i]),
                    "auc_ex": float(tr["auc_ex"][i]),
                    "auc_nom": float(tr["auc_nom"][i]),
                    "auc_noise": float(tr["auc_noise"][i]),
                    "r_probe": float(tr["r_probe"][i]),
                    "old_step": int(tr["old_step"][i]),
                    "twin_step": int(tr["twin_step"][i]),
                    "est": {},
                }
                for est in estimators:
                    blk["est"][est] = {
                        "g": rec[n_step][est]["g"][i].astype(np.float32),
                        "d": rec[n_step][est]["d"][i].astype(np.float32),
                        "A": float(rec[n_step][est]["A"][i]),
                        "A_net": float(rec[n_step][est]["A_net"][i]),
                        "valid": bool(rec[n_step][est]["valid"][i]),
                    }
                item["ns"][str(n_step)] = blk
            rows.append(item)
    return rows, codes, twin_assert


@torch.inference_mode()
def _recapture(env, policy, runner, injector, orig, n_valid):
    hist = _hist_snapshot(env)
    obs = injector.patch_policy_obs(env.get_observations()[0], env, "mapper")
    z, _pr, _enc = _z_and_proprio(policy, runner, obs)
    out = []
    for i in range(n_valid):
        extra = {
            "obs": obs[i].detach().cpu().clone(),
            "z_nom": z[i].detach().cpu().clone(),
            "e0": orig[i]["e0"],
            "t": orig[i]["t"],
            "seed": orig[i]["seed"],
            "terrain": orig[i]["terrain"],
            "window": orig[i].get("window", "recovery"),
            "clip": orig[i]["clip"],
            "task": orig[i].get("task", "loco"),
        }
        out.append(_capture_snap(env, i, hist, extra))
    return out


@torch.inference_mode()
def _probe_p4b3j(env, policy, runner, injector, snaps, vis_i, vis_flags, pelvis_i, B_raw_np):
    """Clone 1-step g at t0, t0+2, t0+4 along Stage-2 twin. No probe."""
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    k = int(args_cli.p4a0_k)
    B = np.asarray(B_raw_np[:, :k], dtype=np.float64)
    B_t = torch.as_tensor(B, device=device, dtype=torch.float32)
    eps = math.tan(math.radians(EPS_DEG))
    asset = env.unwrapped.scene["robot"]
    names = list(asset.data.body_names)
    ankle_i = [names.index(n) for n in ANKLE_BODIES if n in names]
    try:
        cf = env.unwrapped.scene["contact_forces"]
    except Exception:
        cf = None
    kwy = dict(vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i, ankle_i=ankle_i, cf=cf)
    taus = (0, 2, 4)
    rows = []
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[p4b3j] batch {start}:{start + n_valid}/{len(snaps)}", flush=True)
        pack = {}
        for tau in taus:
            padded, _ = _pad_snaps(batch, n_envs)
            _restore_batch(env, padded)
            obs_stack = torch.stack([s["obs"].to(device) for s in padded], dim=0)
            for t in range(int(tau)):
                obs = obs_stack if t == 0 else injector.patch_policy_obs(env.get_observations()[0], env, "mapper")
                z_nom, proprio, _enc = _z_and_proprio(policy, runner, obs)
                env.step(_decode(policy, z_nom, proprio))
            rec = _recapture(env, policy, runner, injector, batch, n_valid)
            z0 = torch.stack([s["z_nom"] for s in rec], dim=0).to(device)
            Bt = _tangent_B(z0, B_t)
            g = torch.zeros(n_valid, k, device=device)
            for ki in range(k):
                v = Bt[:, :, ki]
                dyp, _, _ = _roll_y(env, policy, runner, injector, rec, n_valid, eps * v, horizon=1, **kwy)
                dym, _, _ = _roll_y(env, policy, runner, injector, rec, n_valid, -eps * v, horizon=1, **kwy)
                g[:, ki] = (dyp[:, 0] - dym[:, 0]) / (2.0 * eps)
            pack[tau] = g.detach().cpu().numpy().astype(np.float32)
            print(f"[p4b3j]   tau={tau} |g| med={float(np.median(np.linalg.norm(pack[tau], axis=1))):.4f}", flush=True)
        for i in range(n_valid):
            s = batch[i]
            rows.append({
                "clip": s["clip"],
                "seed": int(s["seed"]),
                "t": int(s["t"]),
                "terrain": s["terrain"],
                "e0": float(s["e0"]),
                "g_0": pack[0][i],
                "g_2": pack[2][i],
                "g_4": pack[4][i],
            })
    return rows


@torch.inference_mode()
def _probe_p4b4a(env, policy, runner, injector, snaps, vis_i, vis_flags, pelvis_i, B_raw_np, ns, codes):
    """P4-B4A: clone 5° labels at probe-end and A_net from t0. No decoder training."""
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    k = int(args_cli.p4a0_k)
    B = np.asarray(B_raw_np[:, :k], dtype=np.float64)
    B_t = torch.as_tensor(B, device=device, dtype=torch.float32)
    tan5 = math.tan(math.radians(EVAL_DEG))
    eps = math.tan(math.radians(EPS_DEG))
    H = int(K_HORIZON)
    kwj = dict(vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i)
    asset = env.unwrapped.scene["robot"]
    names = list(asset.data.body_names)
    ankle_i = [names.index(n) for n in ANKLE_BODIES if n in names]
    try:
        cf = env.unwrapped.scene["contact_forces"]
    except Exception:
        cf = None
    kwy = dict(vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i, ankle_i=ankle_i, cf=cf)
    rows = []
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[p4b4a] batch {start}:{start + n_valid}/{len(snaps)}", flush=True)
        pack = {}
        for n_step in ns:
            n_step = int(n_step)
            U, _P = codes[n_step]
            _Yp, _Ep, _e0p, _z0 = _roll_ye(
                env, policy, runner, injector, batch, n_valid, n_step, vis_i, vis_flags, "probe", B, U
            )
            rec = _recapture(env, policy, runner, injector, batch, n_valid)
            z_end = torch.stack([s["z_nom"] for s in rec], dim=0).to(device)
            j0_end, _ = _roll_j(env, policy, runner, injector, rec, n_valid, None, horizon=H, **kwj)
            a5_end = torch.zeros(n_valid, k, 2, device=device)
            for ki in range(k):
                for si, sgn in enumerate((1.0, -1.0)):
                    dz = _axis_dz(z_end, B_t, ki, sgn, tan5)
                    j5, _ = _roll_j(env, policy, runner, injector, rec, n_valid, dz, horizon=H, **kwj)
                    a5_end[:, ki, si] = j0_end - j5
            g_end = torch.zeros(n_valid, k, device=device)
            Bt_end = _tangent_B(z_end, B_t)
            for ki in range(k):
                v = Bt_end[:, :, ki]
                dyp, _, _ = _roll_y(env, policy, runner, injector, rec, n_valid, eps * v, horizon=1, **kwy)
                dym, _, _ = _roll_y(env, policy, runner, injector, rec, n_valid, -eps * v, horizon=1, **kwy)
                g_end[:, ki] = (dyp[:, 0] - dym[:, 0]) / (2.0 * eps)
            a_net = torch.zeros(n_valid, k, 2, device=device)
            for ki in range(k):
                for si, sgn in enumerate((1.0, -1.0)):
                    dstar = project_tangent(_axis_dz(z_end, B_t, ki, sgn, 1.0), z_end)
                    dn = dstar.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    dstar = dstar / dn
                    jpar, jnet, _eh = _anet_traj(
                        env, policy, runner, injector, batch, n_valid, B, U, dstar,
                        vis_i, vis_flags, pelvis_i, n_step, H,
                    )
                    a_net[:, ki, si] = jpar - jnet
            jpar0, jnet0, _eh0 = _anet_traj(
                env, policy, runner, injector, batch, n_valid, B, U, torch.zeros_like(z_end),
                vis_i, vis_flags, pelvis_i, n_step, H,
            )
            pack[n_step] = {
                "a5_end": a5_end.detach().cpu().numpy().astype(np.float32),
                "a_net": a_net.detach().cpu().numpy().astype(np.float32),
                "a_net0": (jpar0 - jnet0).detach().cpu().numpy().astype(np.float32),
                "g_end": g_end.detach().cpu().numpy().astype(np.float32),
                "j0_end": j0_end.detach().cpu().numpy().astype(np.float32),
            }
            dA = (a5_end[:, :, 0] - a5_end[:, :, 1]).detach().cpu().numpy()
            print(
                f"[p4b4a]   N={n_step} |Aend| med={float(np.median(np.abs(dA))):.4f} "
                f"Anet0 med={float(np.median(pack[n_step]['a_net0'])):.4f}",
                flush=True,
            )
        for i in range(n_valid):
            s = batch[i]
            item = {
                "clip": s["clip"],
                "seed": int(s["seed"]),
                "t": int(s["t"]),
                "terrain": s["terrain"],
                "e0": float(s["e0"]),
                "a5_t0": np.asarray(s["a5_gt"], dtype=np.float32),
            }
            for n_step in ns:
                n_step = int(n_step)
                for key, val in pack[n_step].items():
                    item[f"{key}_{n_step}"] = val[i]
            rows.append(item)
    return rows


def _run_terrain(env_cfg, agent_cfg, resume_path, motion_dir, terrain, out_root):
    p4b4b = bool(getattr(args_cli, "p4b4b", False))
    p4b4a = bool(getattr(args_cli, "p4b4a", False))
    p4b3j = bool(getattr(args_cli, "p4b3j", False))
    p4b2 = bool(getattr(args_cli, "p4b2", False))
    p4b = bool(getattr(args_cli, "p4b", False))
    p4a1 = bool(getattr(args_cli, "p4a1", False))
    p4a0 = bool(getattr(args_cli, "p4a0", False))
    tag = (
        "p4b4b" if p4b4b else (
            "p4b4a" if p4b4a else (
                "p4b3j" if p4b3j else (
                    "p4b2" if p4b2 else ("p4b" if p4b else ("p4a1" if p4a1 else ("p4a0" if p4a0 else str(args_cli.r3_tag))))
                )
            )
        )
    )
    cell = out_root / tag / str(args_cli.task_source) / terrain
    cell.mkdir(parents=True, exist_ok=True)
    done_p = cell / "summary.json"
    if done_p.exists():
        print(f"[irr-r3] skip {done_p}", flush=True)
        return json.loads(done_p.read_text())
    B = None
    bp = Path(args_cli.bucr_path if (p4a0 or p4a1 or p4b or p4b2 or p4b3j or p4b4a or p4b4b) else args_cli.b_path)
    if bp.is_file():
        B = np.load(bp)["B"].astype(np.float32)
        print(f"[irr-r3] B {tuple(B.shape)} from {bp}", flush=True)
    env, runner, policy, injector = _make_env(env_cfg, agent_cfg, resume_path, motion_dir, terrain)
    cmd = env.unwrapped.command_manager.get_term("motion")
    body_names = list(cmd.cfg.body_names)
    vis_i = [body_names.index(n) for n in KP_VIS]
    mask_name = str(args_cli.mask_modes).split(",")[0].strip() or "torso"
    vis_flags = MASK_VIS.get(mask_name, (True, True, True))
    seeds = [int(s) for s in args_cli.seeds.split(",") if s.strip()] or [int(args_cli.seed)]
    all_snaps = []
    pelvis_i = None
    for seed in seeds:
        print(f"[irr-r3] ===== walk {args_cli.task_source} {terrain} seed={seed} =====", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        snaps, pelvis_i, _use, _paths = _collect_snaps(
            env, policy, runner, injector, int(args_cli.steps), seed, terrain, vis_i, vis_flags, mask_name
        )
        all_snaps.extend(snaps)
    rec_snaps = [s for s in all_snaps if s.get("window") == "recovery"]
    if p4b4b:
        if B is None:
            raise FileNotFoundError("P4-B4B requires B_ucr")
        rec_snaps, _codes = _align_p4b_snaps(rec_snaps, terrain)
        from p4b4b_runtime import run_p4b4b, make_u4, make_u8
        print(f"[p4b4b] aligned n={len(rec_snaps)}", flush=True)
        rows, meta = run_p4b4b(
            env, policy, runner, injector, rec_snaps, vis_i, vis_flags, pelvis_i, B
        )
        payload = {
            "step": "P4-B4B",
            "name": "200 Hz intra-policy microprobe",
            "no_p4c": True,
            "no_virtual_twin": True,
            "physics_dt": meta.get("physics_dt"),
            "decimation": meta.get("decimation"),
            "horizon_corr": H_CORR,
            "eval_deg": EVAL_DEG,
            "n": len(rows),
            "terrain": terrain,
            "implementation": "exact_batched_latent_decode",
            "coverage": 1.0,
        }
        if rows:
            dump = {
                "clip": np.asarray([r["clip"] for r in rows]),
                "seed": np.asarray([r["seed"] for r in rows], dtype=np.int32),
                "t": np.asarray([r["t"] for r in rows], dtype=np.int32),
                "terrain": np.asarray([r["terrain"] for r in rows]),
                "e0": np.asarray([r["e0"] for r in rows], dtype=np.float32),
                "a5_gt": np.stack([np.asarray(r["a5_gt"], dtype=np.float32) for r in rows]),
                "order_subset": np.asarray([r["order_subset"] for r in rows], dtype=np.int32),
                "U4": make_u4().astype(np.float32),
                "U8": make_u8().astype(np.float32),
            }
            all_keys = sorted(set().union(*[set(r["cfgs"].keys()) for r in rows]))
            dump["cfg_keys"] = np.asarray(all_keys)
            for key in all_keys:
                present = np.asarray([key in r["cfgs"] for r in rows], dtype=np.bool_)
                dump[f"has_{key}"] = present
                sample = next(r["cfgs"][key] for r in rows if key in r["cfgs"])
                for kk, vv in sample.items():
                    if not isinstance(vv, np.ndarray):
                        dump[f"{key}__{kk}"] = np.asarray(
                            [r["cfgs"][key][kk] if key in r["cfgs"] else vv for r in rows]
                        )
                        continue
                    stacked = []
                    for r in rows:
                        if key in r["cfgs"]:
                            stacked.append(r["cfgs"][key][kk])
                        else:
                            stacked.append(np.full_like(vv, np.nan))
                    dump[f"{key}__{kk}"] = np.stack(stacked, axis=0)
            np.savez_compressed(cell / "p4b4b.npz", **dump)
        done_p.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
        print(f"[p4b4b] wrote {done_p} n={len(rows)}", flush=True)
        try:
            env.close()
        except Exception:
            pass
        return payload
    if p4b4a:
        if B is None:
            raise FileNotFoundError("P4-B4A requires B_ucr")
        rec_snaps, codes = _align_p4b_snaps(rec_snaps, terrain)
        ns = [int(x) for x in str(args_cli.p4b_ns).split(",") if x.strip()]
        print(f"[p4b4a] aligned n={len(rec_snaps)} ns={ns}", flush=True)
        rows = _probe_p4b4a(
            env, policy, runner, injector, rec_snaps, vis_i, vis_flags, pelvis_i, B, ns, codes
        )
        payload = {
            "step": "P4-B4A",
            "name": "Post-probe endpoint re-labeling",
            "no_p4c": True,
            "no_virtual_twin": True,
            "no_p4b4b": True,
            "horizon_A": K_HORIZON,
            "eval_deg": EVAL_DEG,
            "k": int(args_cli.p4a0_k),
            "ns": ns,
            "n": len(rows),
            "terrain": terrain,
            "note": "K_HORIZON=10 (0.2s) matches frozen t0 clone A, not 0.5s",
        }
        if rows:
            dump = {
                "clip": np.asarray([r["clip"] for r in rows]),
                "seed": np.asarray([r["seed"] for r in rows], dtype=np.int32),
                "t": np.asarray([r["t"] for r in rows], dtype=np.int32),
                "terrain": np.asarray([r["terrain"] for r in rows]),
                "e0": np.asarray([r["e0"] for r in rows], dtype=np.float32),
                "a5_t0": np.stack([r["a5_t0"] for r in rows]),
                "ns": np.asarray(ns, dtype=np.int32),
            }
            for n_step in ns:
                n_step = int(n_step)
                dump[f"a5_end_{n_step}"] = np.stack([r[f"a5_end_{n_step}"] for r in rows])
                dump[f"a_net_{n_step}"] = np.stack([r[f"a_net_{n_step}"] for r in rows])
                dump[f"a_net0_{n_step}"] = np.stack([r[f"a_net0_{n_step}"] for r in rows])
                dump[f"g_end_{n_step}"] = np.stack([r[f"g_end_{n_step}"] for r in rows])
                dump[f"j0_end_{n_step}"] = np.stack([r[f"j0_end_{n_step}"] for r in rows])
            np.savez_compressed(cell / "p4b4a.npz", **dump)
        done_p.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
        print(f"[p4b4a] wrote {done_p} n={len(rows)}", flush=True)
        try:
            env.close()
        except Exception:
            pass
        return payload
    if p4b3j:
        if B is None:
            raise FileNotFoundError("P4-B3j requires B_ucr")
        rec_snaps, _codes = _align_p4b_snaps(rec_snaps, terrain)
        rows = _probe_p4b3j(env, policy, runner, injector, rec_snaps, vis_i, vis_flags, pelvis_i, B)
        payload = {
            "step": "P4-B3j",
            "name": "Jacobian stationarity along Stage2 twin",
            "taus": [0, 2, 4],
            "n": len(rows),
            "terrain": terrain,
            "no_p4c": True,
        }
        if rows:
            np.savez_compressed(
                cell / "p4b3j.npz",
                clip=np.asarray([r["clip"] for r in rows]),
                seed=np.asarray([r["seed"] for r in rows], dtype=np.int32),
                t=np.asarray([r["t"] for r in rows], dtype=np.int32),
                terrain=np.asarray([r["terrain"] for r in rows]),
                e0=np.asarray([r["e0"] for r in rows], dtype=np.float32),
                g_0=np.stack([r["g_0"] for r in rows]),
                g_2=np.stack([r["g_2"] for r in rows]),
                g_4=np.stack([r["g_4"] for r in rows]),
            )
        done_p.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
        print(f"[p4b3j] wrote {done_p} n={len(rows)}", flush=True)
        try:
            env.close()
        except Exception:
            pass
        return payload
    if p4b2:
        if B is None:
            raise FileNotFoundError("P4-B2 requires B_ucr")
        rec_snaps, codes = _align_p4b_snaps(rec_snaps, terrain)
        ns = [int(x) for x in str(args_cli.p4b_ns).split(",") if x.strip()]
        print(f"[p4b2] aligned n={len(rec_snaps)} codes={sorted(codes.keys())}", flush=True)
        rows, codes, twin_assert = _probe_p4b2(
            env, policy, runner, injector, rec_snaps, vis_i, vis_flags, pelvis_i, B, ns, codes
        )
        estimators = ("E1", "E2", "scalar")
        payload = {
            "step": "P4-B2",
            "name": "Twin-Referenced Tube-Safe Online Interaction Identification",
            "no_abort": True,
            "no_p4c": True,
            "no_hard_nullspace": True,
            "no_projector": True,
            "reused_p4b_ids": True,
            "reused_p4b_U": True,
            "eps_deg": EPS_DEG,
            "eval_deg": EVAL_DEG,
            "k": int(args_cli.p4a0_k),
            "ns": ns,
            "tube_abs_m": float(args_cli.tube_guard) * float(args_cli.pos_scale),
            "dt": DT,
            "horizon_A": K_HORIZON,
            "terrain": terrain,
            "task": str(args_cli.task_source),
            "n": len(rows),
            "n_walk_snaps": len(all_snaps),
            "twin_restore_maxabs": twin_assert,
            "y": ["e_I", "de_I", "v_root(3)", "omega_root(3)", "g_xy(2)"],
            "estimators": list(estimators),
        }
        if rows:
            dump = {
                "clip": np.asarray([r["clip"] for r in rows]),
                "seed": np.asarray([r["seed"] for r in rows], dtype=np.int32),
                "t": np.asarray([r["t"] for r in rows], dtype=np.int32),
                "terrain": np.asarray([r["terrain"] for r in rows]),
                "window": np.asarray([r["window"] for r in rows]),
                "e0": np.asarray([r["e0"] for r in rows], dtype=np.float32),
                "e0_p4b": np.asarray([r["e0_p4b"] for r in rows], dtype=np.float32),
                "j0": np.asarray([r["j0"] for r in rows], dtype=np.float32),
                "z_nom": np.stack([r["z_nom"] for r in rows]),
                "a5_gt": np.stack([r["a5_gt"] for r in rows]),
                "ns": np.asarray(ns, dtype=np.int32),
            }
            for n_step, (U, P) in codes.items():
                dump[f"U_{n_step}"] = U.astype(np.float32)
                dump[f"P_{n_step}"] = P.astype(np.float32)
                dump[f"UtU_{n_step}"] = (U.T @ U).astype(np.float32)
                dump[f"PtU_{n_step}"] = (P.T @ U).astype(np.float32)
                dump[f"cond_UtU_{n_step}"] = np.array(float(np.linalg.cond(U.T @ U)), dtype=np.float32)
            for n_step in ns:
                dump[f"E_twin_{n_step}"] = np.stack([r["ns"][str(int(n_step))]["E_twin"] for r in rows])
                dump[f"E_twinB_{n_step}"] = np.stack([r["ns"][str(int(n_step))]["E_twinB"] for r in rows])
                dump[f"E_probe_{n_step}"] = np.stack([r["ns"][str(int(n_step))]["E_probe"] for r in rows])
                dump[f"Y_probe_{n_step}"] = np.stack([r["ns"][str(int(n_step))]["Y_probe"] for r in rows])
                dump[f"peak_old_{n_step}"] = np.asarray(
                    [r["ns"][str(int(n_step))]["peak_old"] for r in rows], dtype=np.float32
                )
                dump[f"peak_ex_{n_step}"] = np.asarray(
                    [r["ns"][str(int(n_step))]["peak_ex"] for r in rows], dtype=np.float32
                )
                dump[f"peak_nom_{n_step}"] = np.asarray(
                    [r["ns"][str(int(n_step))]["peak_nom"] for r in rows], dtype=np.float32
                )
                dump[f"peak_noise_{n_step}"] = np.asarray(
                    [r["ns"][str(int(n_step))]["peak_noise"] for r in rows], dtype=np.float32
                )
                dump[f"auc_ex_{n_step}"] = np.asarray(
                    [r["ns"][str(int(n_step))]["auc_ex"] for r in rows], dtype=np.float32
                )
                dump[f"auc_nom_{n_step}"] = np.asarray(
                    [r["ns"][str(int(n_step))]["auc_nom"] for r in rows], dtype=np.float32
                )
                dump[f"auc_noise_{n_step}"] = np.asarray(
                    [r["ns"][str(int(n_step))]["auc_noise"] for r in rows], dtype=np.float32
                )
                dump[f"r_probe_{n_step}"] = np.asarray(
                    [r["ns"][str(int(n_step))]["r_probe"] for r in rows], dtype=np.float32
                )
                dump[f"old_step_{n_step}"] = np.asarray(
                    [r["ns"][str(int(n_step))]["old_step"] for r in rows], dtype=np.int32
                )
                dump[f"twin_step_{n_step}"] = np.asarray(
                    [r["ns"][str(int(n_step))]["twin_step"] for r in rows], dtype=np.int32
                )
                for est in estimators:
                    dump[f"g_{est}_{n_step}"] = np.stack(
                        [r["ns"][str(int(n_step))]["est"][est]["g"] for r in rows]
                    )
                    dump[f"d_{est}_{n_step}"] = np.stack(
                        [r["ns"][str(int(n_step))]["est"][est]["d"] for r in rows]
                    )
                    dump[f"A_{est}_{n_step}"] = np.asarray(
                        [r["ns"][str(int(n_step))]["est"][est]["A"] for r in rows], dtype=np.float32
                    )
                    dump[f"Anet_{est}_{n_step}"] = np.asarray(
                        [r["ns"][str(int(n_step))]["est"][est]["A_net"] for r in rows], dtype=np.float32
                    )
                    dump[f"valid_{est}_{n_step}"] = np.asarray(
                        [r["ns"][str(int(n_step))]["est"][est]["valid"] for r in rows], dtype=np.bool_
                    )
            np.savez_compressed(cell / "p4b2.npz", **dump)
        done_p.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
        print(f"[p4b2] wrote {done_p} n={len(rows)}", flush=True)
        try:
            env.close()
        except Exception:
            pass
        return payload
    if p4b:
        if B is None:
            raise FileNotFoundError("P4-B requires B_ucr")
        ns = [int(x) for x in str(args_cli.p4b_ns).split(",") if x.strip()]
        rows, codes = _probe_p4b(
            env, policy, runner, injector, rec_snaps, vis_i, vis_flags, pelvis_i, B, ns, terrain
        )
        estimators = ("E1", "E2", "scalar")
        payload = {
            "step": "P4-B",
            "name": "Tube-Constrained Online Interaction Identification",
            "trigger_eval_only": True,
            "no_terrain_in_model": True,
            "no_learned_shield": True,
            "no_hard_nullspace": True,
            "no_projector": True,
            "no_sequential_fd": True,
            "probe": "coded_multiplex",
            "eps_deg": EPS_DEG,
            "eval_deg": EVAL_DEG,
            "k": int(args_cli.p4a0_k),
            "ns": ns,
            "tube_guard_frac": float(args_cli.tube_guard),
            "tube_abs_m": float(args_cli.tube_guard) * float(args_cli.pos_scale),
            "code_seed": int(getattr(args_cli, "p4b_code_seed", 2026)),
            "terrain": terrain,
            "task": str(args_cli.task_source),
            "n": len(rows),
            "n_walk_snaps": len(all_snaps),
            "B_shape": list(B.shape),
            "y": ["e_I", "de_I", "v_root(3)", "omega_root(3)", "g_xy(2)"],
            "estimators": list(estimators),
        }
        if rows:
            dump = {
                "clip": np.asarray([r["clip"] for r in rows]),
                "seed": np.asarray([r["seed"] for r in rows], dtype=np.int32),
                "t": np.asarray([r["t"] for r in rows], dtype=np.int32),
                "terrain": np.asarray([r["terrain"] for r in rows]),
                "window": np.asarray([r["window"] for r in rows]),
                "e0": np.asarray([r["e0"] for r in rows], dtype=np.float32),
                "j0": np.asarray([r["j0"] for r in rows], dtype=np.float32),
                "z_nom": np.stack([r["z_nom"] for r in rows]),
                "a5_gt": np.stack([r["a5_gt"] for r in rows]),
                "ns": np.asarray(ns, dtype=np.int32),
            }
            for n_step, (U, P) in codes.items():
                dump[f"U_{n_step}"] = U.astype(np.float32)
                dump[f"P_{n_step}"] = P.astype(np.float32)
                dump[f"UtU_{n_step}"] = (U.T @ U).astype(np.float32)
                dump[f"PtU_{n_step}"] = (P.T @ U).astype(np.float32)
            for n_step in ns:
                dump[f"di_max_{n_step}"] = np.asarray(
                    [r["ns"][str(int(n_step))]["di_max"] for r in rows], dtype=np.float32
                )
                dE = [r["ns"][str(int(n_step))]["dE"] for r in rows]
                dump[f"dE_{n_step}"] = np.stack(dE)
                for est in estimators:
                    dump[f"g_{est}_{n_step}"] = np.stack(
                        [r["ns"][str(int(n_step))]["est"][est]["g"] for r in rows]
                    )
                    dump[f"d_{est}_{n_step}"] = np.stack(
                        [r["ns"][str(int(n_step))]["est"][est]["d"] for r in rows]
                    )
                    dump[f"A_{est}_{n_step}"] = np.asarray(
                        [r["ns"][str(int(n_step))]["est"][est]["A"] for r in rows], dtype=np.float32
                    )
                    dump[f"Anet_{est}_{n_step}"] = np.asarray(
                        [r["ns"][str(int(n_step))]["est"][est]["A_net"] for r in rows], dtype=np.float32
                    )
                    dump[f"abort_{est}_{n_step}"] = np.asarray(
                        [r["ns"][str(int(n_step))]["est"][est]["abort"] for r in rows], dtype=np.bool_
                    )
            np.savez_compressed(cell / "p4b.npz", **dump)
        done_p.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
        print(f"[p4b] wrote {done_p} n={len(rows)}", flush=True)
        try:
            env.close()
        except Exception:
            pass
        return payload
    if p4a1:
        if B is None:
            raise FileNotFoundError("P4-A1 requires B_ucr")
        fracs = [float(x) for x in str(args_cli.tube_fracs).split(",") if x.strip()]
        rows, levels = _probe_p4a1(env, policy, runner, injector, rec_snaps, vis_i, vis_flags, pelvis_i, B, fracs)
        payload = {
            "step": "P4-A1",
            "trigger_eval_only": True,
            "no_terrain_in_model": True,
            "no_learned_shield": True,
            "probe_L": [1],
            "eps_deg": EPS_DEG,
            "eval_deg": EVAL_DEG,
            "k": int(args_cli.p4a0_k),
            "tube_fracs": fracs,
            "levels": levels,
            "constraint": "linearized 5deg DI <= frac of 5cm band",
            "terrain": terrain,
            "task": str(args_cli.task_source),
            "n": len(rows),
            "n_walk_snaps": len(all_snaps),
            "B_shape": list(B.shape),
        }
        done_p.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
        if rows:
            dump = {
                "clip": np.asarray([r["clip"] for r in rows]),
                "seed": np.asarray([r["seed"] for r in rows], dtype=np.int32),
                "t": np.asarray([r["t"] for r in rows], dtype=np.int32),
                "terrain": np.asarray([r["terrain"] for r in rows]),
                "window": np.asarray([r["window"] for r in rows]),
                "e0": np.asarray([r["e0"] for r in rows], dtype=np.float32),
                "j0": np.asarray([r["j0"] for r in rows], dtype=np.float32),
                "z_nom": np.stack([r["z_nom"] for r in rows]),
                "J_I": np.stack([r["J_I"] for r in rows]),
                "levels": np.asarray(levels),
                "fracs": np.asarray(fracs, dtype=np.float32),
            }
            for name in levels:
                dump[f"a5_{name}"] = np.stack([r[f"a5_{name}"] for r in rows])
                dump[f"r_e_{name}"] = np.stack([r[f"r_e_{name}"] for r in rows])
                dump[f"di_probe_{name}"] = np.stack([r[f"di_probe_{name}"] for r in rows])
                dump[f"di_corr_{name}"] = np.stack([r[f"di_corr_{name}"] for r in rows])
                dump[f"Rd_{name}"] = np.stack([r[f"Rd_{name}"] for r in rows])
                dump[f"dist_{name}"] = np.stack([r[f"dist_{name}"] for r in rows])
                dump[f"B_{name}"] = np.stack([r[f"B_{name}"] for r in rows])
            np.savez_compressed(cell / "p4a1.npz", **dump)
        print(f"[p4a1] wrote {done_p} n={len(rows)}", flush=True)
        try:
            env.close()
        except Exception:
            pass
        return payload
    if p4a0:
        if B is None:
            raise FileNotFoundError("P4-A0 requires B_ucr")
        rows = _probe_p4a0(env, policy, runner, injector, rec_snaps, vis_i, vis_flags, pelvis_i, B)
        payload = {
            "step": "P4-A0",
            "trigger_eval_only": True,
            "no_terrain_in_model": True,
            "oracle_shield_only": True,
            "no_learned_b1": True,
            "probe_L": [1],
            "eps_deg": EPS_DEG,
            "eval_deg": EVAL_DEG,
            "k": int(args_cli.p4a0_k),
            "terrain": terrain,
            "task": str(args_cli.task_source),
            "n": len(rows),
            "n_walk_snaps": len(all_snaps),
            "spaces": ["raw", "oracle_safe"],
            "B_shape": list(B.shape),
        }
        done_p.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
        if rows:
            np.savez_compressed(
                cell / "p4a0.npz",
                clip=np.asarray([r["clip"] for r in rows]),
                seed=np.asarray([r["seed"] for r in rows], dtype=np.int32),
                t=np.asarray([r["t"] for r in rows], dtype=np.int32),
                terrain=np.asarray([r["terrain"] for r in rows]),
                window=np.asarray([r["window"] for r in rows]),
                e0=np.asarray([r["e0"] for r in rows], dtype=np.float32),
                j0=np.asarray([r["j0"] for r in rows], dtype=np.float32),
                a5_raw=np.stack([r["a5_raw"] for r in rows]),
                a5_oracle_safe=np.stack([r["a5_oracle_safe"] for r in rows]),
                r_e_raw=np.stack([r["r_e_raw"] for r in rows]),
                r_e_oracle_safe=np.stack([r["r_e_oracle_safe"] for r in rows]),
                di_raw=np.stack([r["di_raw"] for r in rows]),
                di_oracle_safe=np.stack([r["di_oracle_safe"] for r in rows]),
                z_nom=np.stack([r["z_nom"] for r in rows]),
            )
        print(f"[p4a0] wrote {done_p} n={len(rows)}", flush=True)
        try:
            env.close()
        except Exception:
            pass
        return payload
    rows = _probe_on_snaps(
        env, policy, runner, injector, rec_snaps, vis_i, vis_flags, pelvis_i, int(args_cli.n_axes), B
    )
    payload = {
        "step": "IRR-R3-short",
        "trigger_eval_only": True,
        "no_terrain_in_model": True,
        "frozen_stage2": True,
        "probe_L": list(PROBE_LS),
        "eps_deg": EPS_DEG,
        "eval_deg": EVAL_DEG,
        "n_axes": int(args_cli.n_axes),
        "terrain": terrain,
        "task": str(args_cli.task_source),
        "n": len(rows),
        "n_walk_snaps": len(all_snaps),
        "ydim": Y_DIM,
        "B_shape": None if B is None else list(B.shape),
    }
    done_p.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    if rows:
        np.savez_compressed(
            cell / "r3.npz",
            clip=np.asarray([r["clip"] for r in rows]),
            seed=np.asarray([r["seed"] for r in rows], dtype=np.int32),
            t=np.asarray([r["t"] for r in rows], dtype=np.int32),
            terrain=np.asarray([r["terrain"] for r in rows]),
            window=np.asarray([r["window"] for r in rows]),
            e0=np.asarray([r["e0"] for r in rows], dtype=np.float32),
            j0=np.asarray([r["j0"] for r in rows], dtype=np.float32),
            a5=np.stack([r["a5"] for r in rows]),
            a1=np.stack([r["a1"] for r in rows]),
            r_clone=np.stack([r["r_clone"] for r in rows]),
            r_seq=np.stack([r["r_seq"] for r in rows]),
            rB_clone=np.stack([r["rB_clone"] for r in rows]),
            rB_seq=np.stack([r["rB_seq"] for r in rows]),
            a_jac_clone=np.stack([r["a_jac_clone"] for r in rows]),
            a_jac_seq=np.stack([r["a_jac_seq"] for r in rows]),
            a_spsa_clone=np.stack([r["a_spsa_clone"] for r in rows]),
            a_spsa_seq=np.stack([r["a_spsa_seq"] for r in rows]),
            aB=np.stack([r["aB"] for r in rows]),
            z_nom=np.stack([r["z_nom"] for r in rows]),
            probe_L=np.asarray(PROBE_LS, dtype=np.int32),
        )
    print(f"[irr-r3] wrote {done_p} n={len(rows)}", flush=True)
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
    print(f"[irr-r3] ckpt {resume_path}", flush=True)
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
        _run_terrain(env_cfg, agent_cfg, resume_path, args_cli.motion, terrain, out_dir)
    print("[irr-r3] done", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
