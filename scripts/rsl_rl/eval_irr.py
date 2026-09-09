#!/usr/bin/env python3
"""IRR R0: oracle candidate search on frozen Stage-2. No residual policy.

From cloned s_t, roll δz=0 and ±ε tangent candidates. Labels are
A = J(0) − J(δz). Terrain params never enter the model.
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
GAMMA = 0.95
LAM_S = 1.0
N_AXES = 6
ANGLES = (1.0, 2.0)
TOKEN = 128  # z16 + e9 + de9 + prop90 + contact2 + roll + pitch
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

parser = argparse.ArgumentParser(description="IRR R0 oracle candidate search. Frozen Stage-2.")
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
parser.add_argument("--r0_tag", type=str, default="r0")
parser.add_argument("--persist", type=int, default=1)
parser.add_argument("--max_snaps_rec", type=int, default=2)
parser.add_argument("--max_snaps_nom", type=int, default=1)
parser.add_argument("--e_rec", type=float, default=0.08)
parser.add_argument("--e_nom", type=float, default=0.06)
parser.add_argument("--push_dv", type=float, default=0.0)
parser.add_argument("--push_step", type=int, default=100)
parser.add_argument("--rec_on_plane", action="store_true", default=False)
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
def _oracle_on_snaps(env, policy, runner, injector, snaps, vis_i, vis_flags, pelvis_i, horizon, n_axes):
    table = _candidate_table(n_axes)
    n_c = len(table)
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    rows = []
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[irr-r0] clone {start}:{start + n_valid}/{len(snaps)} cands={n_c}", flush=True)
        z0 = torch.stack([s["z_nom"] for s in batch], dim=0).to(device=device)
        basis = tangent_basis(z0)
        j0, f0 = _roll_j(env, policy, runner, injector, batch, n_valid, None, vis_i, vis_flags, pelvis_i, horizon)
        j_all = [j0]
        names = ["zero"]
        dz_all = [torch.zeros_like(z0)]
        for name, sgn, axis, deg in table[1:]:
            v = basis[:, :, int(axis)]
            dz = float(sgn) * math.tan(math.radians(float(deg))) * v
            jc, _fc = _roll_j(env, policy, runner, injector, batch, n_valid, dz, vis_i, vis_flags, pelvis_i, horizon)
            j_all.append(jc)
            names.append(name)
            dz_all.append(dz)
            if len(j_all) == 2 or len(j_all) % 10 == 0 or len(j_all) == n_c:
                print(f"[irr-r0]   {name} ({len(j_all)}/{n_c})", flush=True)
        j_mat = torch.stack(j_all, dim=1)
        a_mat = j0.unsqueeze(1) - j_mat
        for i in range(n_valid):
            s = batch[i]
            a_i = a_mat[i].detach().cpu().numpy().astype(np.float32)
            j_i = j_mat[i].detach().cpu().numpy().astype(np.float32)
            dz_i = torch.stack([d[i] for d in dz_all], dim=0).detach().cpu().numpy().astype(np.float32)
            star = int(np.argmax(a_i))
            rows.append(
                {
                    "clip": s["clip"],
                    "seed": int(s["seed"]),
                    "t": int(s["t"]),
                    "terrain": s["terrain"],
                    "window": s["window"],
                    "task": s["task"],
                    "e0": float(s["e0"]),
                    "de0": float(s["de0"]),
                    "j0": float(j0[i].item()),
                    "fail0": int(bool(f0[i].item())),
                    "a": a_i,
                    "j": j_i,
                    "dz": dz_i,
                    "names": names,
                    "star": star,
                    "a_star": float(a_i[star]),
                    "a_zero": float(a_i[0]),
                    "hist_tok": s["hist_tok"].numpy().astype(np.float32),
                    "z_nom": s["z_nom"].numpy().astype(np.float32),
                }
            )
    return rows, names


def _run_terrain(env_cfg, agent_cfg, resume_path, motion_dir, terrain, out_root):
    cell = out_root / str(args_cli.r0_tag) / str(args_cli.task_source) / terrain
    cell.mkdir(parents=True, exist_ok=True)
    done_p = cell / "summary.json"
    if done_p.exists():
        print(f"[irr-r0] skip {done_p}", flush=True)
        return json.loads(done_p.read_text())
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
        print(f"[irr-r0] ===== walk {args_cli.task_source} {terrain} seed={seed} =====", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        snaps, pelvis_i, _use, _paths = _collect_snaps(
            env, policy, runner, injector, int(args_cli.steps), seed, terrain, vis_i, vis_flags, mask_name
        )
        all_snaps.extend(snaps)
    rows, names = _oracle_on_snaps(
        env, policy, runner, injector, all_snaps, vis_i, vis_flags, pelvis_i,
        int(args_cli.horizon), int(args_cli.n_axes),
    )
    a_star = np.asarray([r["a_star"] for r in rows], dtype=np.float64)
    windows = np.asarray([r["window"] for r in rows])
    rec = windows == "recovery"
    nom = windows == "nominal"

    a_all = np.stack([r["a"] for r in rows]) if rows else np.zeros((0, 1))

    def blk(mask):
        x = a_star[mask]
        if x.size == 0:
            return {
                "n": 0, "P_Agt0": None, "mean_Astar": None, "p50_Astar": None,
                "P_Agt0_02": None, "P_Agt0_05": None, "Astar_over_std": None,
            }
        std = a_all[mask][:, 1:].std(axis=1) if a_all.shape[1] > 1 else np.ones_like(x)
        return {
            "n": int(x.size),
            "P_Agt0": float((x > 0).mean()),
            "P_Agt0_02": float((x > 0.02).mean()),
            "P_Agt0_05": float((x > 0.05).mean()),
            "mean_Astar": float(x.mean()),
            "p50_Astar": float(np.median(x)),
            "Astar_over_std": float(np.median(x / np.maximum(std, 1e-9))),
        }

    payload = {
        "step": "IRR-R0",
        "no_terrain_in_model": True,
        "same_reference": True,
        "frozen_stage2": True,
        "candidates": names,
        "angles": list(_angle_list()),
        "r0_tag": str(args_cli.r0_tag),
        "horizon": int(args_cli.horizon),
        "n_axes": int(args_cli.n_axes),
        "terrain": terrain,
        "task": str(args_cli.task_source),
        "n": len(rows),
        "all": blk(np.ones(len(rows), dtype=bool)),
        "recovery": blk(rec),
        "nominal": blk(nom),
    }
    done_p.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    if rows:
        np.savez_compressed(
            cell / "r0.npz",
            clip=np.asarray([r["clip"] for r in rows]),
            seed=np.asarray([r["seed"] for r in rows], dtype=np.int32),
            t=np.asarray([r["t"] for r in rows], dtype=np.int32),
            terrain=np.asarray([r["terrain"] for r in rows]),
            window=windows,
            e0=np.asarray([r["e0"] for r in rows], dtype=np.float32),
            de0=np.asarray([r["de0"] for r in rows], dtype=np.float32),
            j0=np.asarray([r["j0"] for r in rows], dtype=np.float32),
            a=np.stack([r["a"] for r in rows]),
            j=np.stack([r["j"] for r in rows]),
            dz=np.stack([r["dz"] for r in rows]),
            star=np.asarray([r["star"] for r in rows], dtype=np.int32),
            a_star=np.asarray([r["a_star"] for r in rows], dtype=np.float32),
            hist_tok=np.stack([r["hist_tok"] for r in rows]),
            z_nom=np.stack([r["z_nom"] for r in rows]),
            names=np.asarray(names),
        )
    print(f"[irr-r0] wrote {done_p} n={len(rows)} rec={payload['recovery']}", flush=True)
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
    print(f"[irr-r0] ckpt {resume_path}", flush=True)
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
    print("[irr-r0] done", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
