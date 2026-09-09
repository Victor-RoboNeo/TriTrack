#!/usr/bin/env python3
"""Phase T0 — Full-task Human Intent Transfer matrix. Frozen eval. No training.

Parent: Mapper-B + g_phi,50000. Task SR = official Locomani termination success
(no anchor_z / anchor_ori / fall). SR@5 is secondary intent fidelity only.

R-M3 active single-burst: Loco + Stoop only.
Reach / Carry: Parent active, R-M3 shadow (adapter logged, z_exec = z_nom).
No PPO, no oracle, no adapter update, no Max-2.
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
Q50_E = 0.046
Q90_E = 0.130
WARMUP = 10
BURST_STEPS = 5
THETA_DEG = 5.0
HORIZON_PAIR = 25  # 0.5 s paired clone
H_FRAMES = {"dE25": 13, "dE50": 25, "dE100": 50}
FALL_H = {"fall_05": 25, "fall_10": 50, "fall_20": 100}
TERRAINS = ("plane", "light_rough", "slope", "steps")
TERRAIN_LABEL = {
    "plane": "Flat",
    "light_rough": "Light",
    "slope": "Slope",
    "steps": "Steps",
}
KP_VIS = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
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
CLIP_STOOP = "/data/home/chenxiangyu/robotics/Anybody/logs/tritrack/infer_clips/p1/stoop"
CLIP_LOCO = "/data/home/chenxiangyu/robotics/Anybody/logs/tritrack/infer_clips/p1/loco"
CLIP_REACH = "/data/home/chenxiangyu/robotics/Anybody/logs/tritrack/infer_clips/p1/reach"
CLIP_CARRY = "/data/home/chenxiangyu/robotics/Anybody/logs/tritrack/infer_clips/p1/carry"
STEP6S_CKPT = "/data/home/chenxiangyu/robotics/Anybody/results/p2r_step6s_full/model_best.pt"
ADAPTER_CKPT = (
    "/data/home/chenxiangyu/robotics/Anybody/results/"
    "rm_intent_conditioned_recovery/checkpoints/model_best.pt"
)
THETA_BINS = (2.5, 5.0, 7.5, 10.0)
FAIL_ANCHOR_Z = 0.25
FAIL_ANCHOR_ORI = 0.8
FAIL_FALL_Z = 0.4
REC_KINDS = ("none", "shared_adapt", "shadow")
TASK_SOURCES = ("loco", "stoop", "reach", "carry")
S_ENABLED_TASKS = ("stoop", "reach", "carry")
BODY_KEYS = ("torso", "lw", "rw")


def project_tangent(d: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z, dim=-1, eps=1e-8)
    d = d - (d * z).sum(-1, keepdim=True) * z
    return F.normalize(d, dim=-1, eps=1e-8)


def apply_angle(z_nom: torch.Tensor, d_raw: torch.Tensor, theta_deg) -> torch.Tensor:
    """Hold raw d; re-project onto current z_nom; apply tan(theta)."""
    d = project_tangent(d_raw, z_nom)
    if torch.is_tensor(theta_deg):
        th = theta_deg.to(device=z_nom.device, dtype=z_nom.dtype)
        tan_th = torch.tan(th * (math.pi / 180.0))
        if tan_th.ndim == 1:
            tan_th = tan_th.unsqueeze(-1)
    else:
        tan_th = math.tan(math.radians(float(theta_deg)))
    return F.normalize(z_nom + tan_th * d, dim=-1, eps=1e-8)


def _pct(x, qs=(25, 50, 75)) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{q}": float("nan") for q in qs} | {"n": 0, "mean": float("nan")}
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} | {"n": int(x.size), "mean": float(x.mean())}


def _rate(mask) -> float:
    m = np.asarray(mask, dtype=bool)
    return float(m.mean()) if m.size else float("nan")


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
    if obj is True or obj is False or obj is None:
        return obj
    return obj


from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="P2-R Phase T0 full-task frozen eval. No PPO.")
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--task", type=str, default="MUSE-Kp-LatentRL-Kp5-HeadHands-Locomani-G1-v0")
parser.add_argument("--motion", type=str, default=CLIP_LOCO)
parser.add_argument("--mask_modes", type=str, default="torso")
parser.add_argument("--task_source", type=str, default="loco", choices=TASK_SOURCES)
parser.add_argument("--rec_kind", type=str, default="none", choices=REC_KINDS)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out", type=str, default="results/rm3_shared_single_burst/loco/parent")
parser.add_argument("--start_frame", type=int, default=10)
parser.add_argument("--hint_iter", type=int, default=50000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
parser.add_argument("--terrain", type=str, default="plane", choices=TERRAINS)
parser.add_argument("--terrains", type=str, default="plane")
parser.add_argument("--keep_headhands_spec", action="store_true", default=True)
parser.add_argument("--mapper_path", type=str, default=MAPPER_B)
parser.add_argument("--mapper_b_path", type=str, default=MAPPER_B)
parser.add_argument("--mapper_a_path", type=str, default="")
parser.add_argument("--residual_alpha", type=float, default=1.0)
parser.add_argument("--step6s_ckpt", type=str, default=STEP6S_CKPT)
parser.add_argument("--adapter_ckpt", type=str, default=ADAPTER_CKPT)
parser.add_argument("--theta_deg", type=float, default=THETA_DEG)
parser.add_argument("--burst_steps", type=int, default=BURST_STEPS)
parser.add_argument("--max_bursts", type=int, default=1)
parser.add_argument("--s_enabled", action="store_true", default=False)
parser.add_argument("--no_s_enabled", action="store_true", default=False)
parser.add_argument("--z_subsample", type=int, default=20)
parser.add_argument("--max_z_samples", type=int, default=400)
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
from isaaclab.utils.math import quat_rotate_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.modules.intent_recovery import RecoveryRiskGate, extract_visible_task_error  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import whole_body_tracking.tasks  # noqa: E402, F401
from causal_future import CausalFutureInjector  # noqa: E402
from whole_body_tracking.tasks.tracking.mdp.rewards import _ankle_mean_z_offset  # noqa: E402


def _restore_adapter(agent_cfg, resume_path: str) -> None:
    params = Path(resume_path).parent / "params" / "agent.yaml"
    if not params.exists():
        return
    cfg = yaml.safe_load(params.read_text()) or {}
    policy_cfg = cfg.get("policy") or {}
    for key in (
        "adapter",
        "lora_rank",
        "lora_alpha",
        "lora_targets",
        "residual_d_model",
        "residual_num_layers",
        "residual_nhead",
        "residual_ffn",
        "residual_last_layer_gain",
        "residual_alpha",
        "latent_std_min",
        "latent_std_max",
        "init_latent_std",
        "terrain_scan_dim",
        "terrain_r_max",
        "terrain_scan_zero",
        "terrain_gate",
        "intent_recovery",
        "intent_recovery_r_max",
        "intent_recovery_r_off",
        "intent_recovery_r_full",
        "intent_recovery_persist_on",
        "intent_recovery_persist_off",
        "intent_recovery_q50_e",
        "intent_recovery_q90_e",
        "intent_recovery_q50_s",
        "intent_recovery_q90_s",
        "intent_recovery_aux_dim",
        "intent_recovery_s_enabled",
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
    if terrain == "plane":
        env_cfg.scene.terrain.terrain_type = "plane"
        print("[t0] terrain=plane", flush=True)
        return
    from isaaclab.terrains import (
        HfPyramidSlopedTerrainCfg,
        HfPyramidStairsTerrainCfg,
        HfRandomUniformTerrainCfg,
        TerrainGeneratorCfg,
    )

    env_cfg.scene.terrain.terrain_type = "generator"
    if terrain == "light_rough":
        subs = {
            "slightly_rough": HfRandomUniformTerrainCfg(
                proportion=1.0, noise_range=(0.01, 0.03), noise_step=0.01
            )
        }
    elif terrain == "slope":
        subs = {
            "slope": HfPyramidSlopedTerrainCfg(
                proportion=1.0, slope_range=(0.087, 0.176), platform_width=2.0
            )
        }
    elif terrain == "steps":
        subs = {
            "stairs": HfPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(0.03, 0.08),
                step_width=0.4,
                platform_width=2.0,
            )
        }
    else:
        raise ValueError(terrain)
    env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=42,
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        curriculum=False,
        sub_terrains=subs,
    )
    env_cfg.scene.terrain.max_init_terrain_level = None
    if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
        if hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None
    print(f"[t0] terrain={terrain} 1x1 generator", flush=True)


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
                print(f"[t0] pin mask {mask_name!r} idx={names.index(mask_name)}", flush=True)
        cmd._resample_command(ids)
        if hasattr(cmd, "resample_all_mask_modes"):
            cmd.resample_all_mask_modes()
        try:
            env.unwrapped.sim.forward()
        except Exception as exc:
            print(f"[t0] sim.forward skipped: {exc}", flush=True)
    return use, [str(p) for p in paths[:use]]


def _visible_e(cmd, vis_i, vis_flags) -> torch.Tensor:
    delta = cmd.robot_body_pos_w - cmd.body_pos_w
    parts = [torch.linalg.norm(delta[:, vis_i[k]], dim=-1) for k, flag in enumerate(vis_flags) if flag]
    stacked = torch.stack(parts, dim=-1)
    return stacked.mean(dim=-1)


def _body_e(cmd, vis_i) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = cmd.robot_body_pos_w - cmd.body_pos_w
    return (
        torch.linalg.norm(delta[:, vis_i[0]], dim=-1),
        torch.linalg.norm(delta[:, vis_i[1]], dim=-1),
        torch.linalg.norm(delta[:, vis_i[2]], dim=-1),
    )


def _series_stats(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "n": 0}
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "n": int(a.size),
    }


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


def _set_residual_alpha(policy_mod, alpha: float) -> None:
    rc = getattr(policy_mod, "residual_corrector", None)
    if rc is None:
        return
    rc.alpha = float(alpha)


def _normalize_obs(runner, obs: torch.Tensor) -> torch.Tensor:
    return runner._normalize_student_obs(obs)


@torch.inference_mode()
def _z_and_proprio(policy, runner, obs: torch.Tensor):
    obs_n = _normalize_obs(runner, obs)
    enc, _scan, feat, mask, _aux = policy._split_policy_obs(obs_n)
    z, proprio = policy._nominal_mean_latent(enc, feat, mask)
    return z, proprio


@torch.inference_mode()
def _decode(policy, z: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
    z = policy.muse._maybe_normalize_latent(z)
    return policy.muse._decode(z, proprio)


def _recovery_pack(
    obs: torch.Tensor, z_nom: torch.Tensor, e9_prev: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    from causal_future import N_BODIES, SLOT_OFFSETS

    n = obs.shape[0]
    kp = obs[:, :225].reshape(n, 15, N_BODIES, 3)
    mask = obs[:, 225:300].reshape(n, 15, N_BODIES)
    slot0 = int(SLOT_OFFSETS.index(0))
    e9, vis, _e_rms = extract_visible_task_error(kp, mask[:, slot0], slot0)
    if e9_prev is None:
        e_dot = torch.zeros_like(e9)
    else:
        e_dot = (e9 - e9_prev.to(device=e9.device, dtype=e9.dtype)) / DT
    proprio = obs[:, 300:750]
    return torch.cat([e9, e_dot, vis, proprio, z_nom], dim=-1), e9


def _load_6s_predictor(ckpt_path: str, device):
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from train_p2r_step6s import SupervisedRecoveryMLP

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SupervisedRecoveryMLP()
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mu = torch.as_tensor(blob["obs_mean"], device=device, dtype=torch.float32)
    sd = torch.as_tensor(blob["obs_std"], device=device, dtype=torch.float32)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    return model, mu, sd


def _load_rm2_adapter(ckpt_path: str, device):
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from train_rm2_intent_adapter import IntentConditionedRecoveryAdapter, THETA_BINS as BINS

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = IntentConditionedRecoveryAdapter()
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mu = torch.as_tensor(blob["obs_mean"], device=device, dtype=torch.float32)
    sd = torch.as_tensor(blob["obs_std"], device=device, dtype=torch.float32)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    bins = tuple(float(x) for x in (blob.get("theta_bins") or BINS))
    return model, mu, sd, bins


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
    _set_residual_alpha(policy, float(args_cli.residual_alpha))
    mapper_paths = {"mapper": args_cli.mapper_path, "mapper_b": args_cli.mapper_b_path or args_cli.mapper_path}
    injector = CausalFutureInjector(mappers=mapper_paths, device=env.device)
    return env, runner, policy, injector


def _want_snap(snaps: list[dict], ch: str, task: str, max_snaps: int) -> bool:
    if len(snaps) >= int(max_snaps):
        return False
    if task != "stoop":
        return True
    n_s = sum(1 for s in snaps if str(s.get("trigger_channel")) == "S")
    n_other = len(snaps) - n_s
    cap_s = max(int(round(max_snaps * 0.80)), 1)
    cap_o = max(max_snaps - cap_s, 1)
    if ch == "S":
        return n_s < cap_s
    return n_other < cap_o


def _horizon_fill(events: list[dict], e_series: dict[tuple[int, int], list[float]], t_end: dict[tuple[int, int], int]) -> None:
    for ev in events:
        key = (int(ev["seed"]), int(ev["env"]))
        series = e_series.get(key, [])
        end = t_end.get(key, max(0, len(series) - 1))
        t0 = int(ev["t0"])
        e0 = float(ev["e0"])
        if ev.get("released") and ev.get("t_rel") is not None:
            t_stop = min(end, int(ev["t_rel"]))
        else:
            t_stop = end
        ev["t_end"] = int(t_stop)
        ev["Trec"] = (ev["n_act"] * DT) if ev.get("released") else None
        lo = max(0, t0 - 25)
        hi = min(len(series), t0 + 51)
        ev["e_win_t0"] = int(t0)
        ev["e_win_lo"] = int(lo)
        ev["e_win"] = [float(x) for x in series[lo:hi]]
        for name, h in H_FRAMES.items():
            t_h = t0 + h
            survived = t_h <= end and t_h < len(series)
            ev[name + "_survived"] = bool(survived)
            if survived:
                ev[name + "_complete"] = float(series[t_h] - e0)
                ev[name + "_last"] = float(series[t_h] - e0)
                ev[name.replace("dE", "E")] = float(series[t_h])
            else:
                ev[name + "_complete"] = None
                t_use = min(end, max(t0, end))
                t_use = min(t_use, len(series) - 1) if series else 0
                ev[name + "_last"] = float(series[t_use] - e0) if series else float("nan")
                ev[name.replace("dE", "E")] = float(series[t_use]) if series else float("nan")
        ev["lived_1s"] = (t0 + H_FRAMES["dE100"]) <= end


def _theta_hist(events: list[dict]) -> dict:
    vals = []
    for e in events:
        for b in e.get("bursts") or []:
            if b.get("theta_deg") is not None:
                vals.append(float(b["theta_deg"]))
    hist = {str(th): int(sum(abs(v - th) < 1e-6 for v in vals)) for th in THETA_BINS}
    return {"n": len(vals), "hist": hist, "mean": float(np.mean(vals)) if vals else float("nan")}


def _channel_block(events: list[dict]) -> dict:
    n = len(events)
    n_fail = sum(1 for e in events if e.get("fail_after"))
    auc = [e["auc"] for e in events]
    vis = [e.get("median_vis_e") for e in events if e.get("median_vis_e") is not None]
    out = {
        "n": n,
        "fail_after_trigger": (n_fail / n) if n else float("nan"),
        "AUC_E": _pct(np.array(auc, dtype=np.float64)),
        "median_visible_kp_error": _pct(np.array(vis, dtype=np.float64)) if vis else _pct([]),
        "task_completion": _rate([not e.get("ep_fail", False) for e in events]) if n else float("nan"),
        "theta": _theta_hist(events),
    }
    for name, _h in FALL_H.items():
        out[name] = _rate([bool(e.get(name)) for e in events]) if n else float("nan")
    first = []
    for e in events:
        for b in e.get("bursts") or []:
            if int(b.get("burst_index", 0)) == 1 and b.get("dE_m") is not None:
                first.append(float(b["dE_m"]))
    dE1 = np.array(first, dtype=np.float64)
    out["first_burst"] = {
        "n": int(dE1.size),
        "median_dE_cm": float(np.median(dE1) * 100.0) if dE1.size else float("nan"),
        "mean_dE_cm": float(dE1.mean() * 100.0) if dE1.size else float("nan"),
        "U1_diagnostic": _rate(dE1 < 0.0) if dE1.size else float("nan"),
    }
    e0 = [e["e0"] for e in events]
    out["E_at_trigger"] = _pct(np.array(e0, dtype=np.float64) * 100.0)
    return out


def _agg_events(events: list[dict]) -> dict:
    n = len(events)
    n_rel = sum(1 for e in events if e.get("released"))
    out = _channel_block(events)
    out["n_events"] = n
    out["n_release"] = n_rel
    out["SR_recover"] = (n_rel / n) if n else float("nan")
    out["n_fail_after"] = sum(1 for e in events if e.get("fail_after"))
    n_bursts = [e.get("n_bursts", 0) for e in events]
    out["bursts_per_event"] = _pct(np.array(n_bursts, dtype=np.float64))
    out["active_duration_s"] = _pct(np.array([e["n_act"] * DT for e in events], dtype=np.float64))
    for name in H_FRAMES:
        c = [e[name + "_complete"] for e in events if e.get(name + "_complete") is not None]
        out[name + "_complete_m"] = _pct(np.array(c, dtype=np.float64)) | {"n": len(c)}
        out[name + "_complete_frac"] = (len(c) / n) if n else float("nan")
        out[name + "_coverage"] = out[name + "_complete_frac"]
        med_cm = (out[name + "_complete_m"]["p50"] * 100.0) if c else float("nan")
        out[name + "_complete_cm_p50"] = med_cm
    out["lived_1s_frac"] = (sum(1 for e in events if e.get("lived_1s")) / n) if n else float("nan")
    src = [str(e.get("trigger_channel") or "?") for e in events]
    out["trigger_channel"] = {
        k: _rate(np.array(src) == k) if n else float("nan") for k in ("E", "S", "both")
    }
    out["n_trigger_S"] = int(sum(s in ("S", "both") for s in src))
    out["n_trigger_E"] = int(sum(s in ("E", "both") for s in src))
    out["n_trigger_S_only"] = int(sum(s == "S" for s in src))
    out["S_first"] = _channel_block([e for e in events if e.get("trigger_channel") == "S"])
    out["E_first"] = _channel_block([e for e in events if e.get("trigger_channel") == "E"])
    out["both"] = _channel_block([e for e in events if e.get("trigger_channel") == "both"])
    return out


@torch.inference_mode()
def _predict_raw_theta(
    rec_kind: str,
    rec_pack: torch.Tensor,
    idxs: list[int],
    sixs,
    mu6,
    sd6,
    adapter,
    mua,
    sda,
    bins: tuple[float, ...],
    fixed_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx_t = torch.tensor(idxs, device=rec_pack.device, dtype=torch.long)
    pack = rec_pack.index_select(0, idx_t)
    if rec_kind == "6s":
        raw = sixs((pack - mu6) / sd6)
        th = torch.full((len(idxs),), float(fixed_theta), device=raw.device, dtype=torch.float32)
        return raw, th
    raw, logits = adapter((pack - mua) / sda)
    if rec_kind == "shared5":
        th = torch.full((len(idxs),), float(fixed_theta), device=raw.device, dtype=torch.float32)
    else:
        pred = logits.argmax(dim=-1)
        th = torch.tensor([float(bins[int(j)]) for j in pred.tolist()], device=raw.device, dtype=torch.float32)
    return raw, th


@torch.inference_mode()
def _rollout_probe(
    env,
    policy,
    runner,
    injector,
    snaps: list[dict],
    n_valid: int,
    *,
    d_raw: torch.Tensor | None,
    theta_deg,
    burst: int,
    horizon: int,
    vis_i,
    vis_flags,
) -> torch.Tensor:
    n_envs = int(env.unwrapped.num_envs)
    padded, _ = _pad_snaps(snaps, n_envs)
    _restore_batch(env, padded)
    cmd = env.unwrapped.command_manager.get_term("motion")
    device = env.unwrapped.device
    e_hist = []
    obs_stack = torch.stack([s["obs"].to(device) for s in padded], dim=0)
    for t in range(horizon):
        if t == 0:
            obs = obs_stack
        else:
            obs, _ = env.get_observations()
            obs = injector.patch_policy_obs(obs, env, "mapper")
        z_nom, proprio = _z_and_proprio(policy, runner, obs)
        if d_raw is not None and t < burst:
            z_exec = apply_angle(z_nom, d_raw, theta_deg)
        else:
            z_exec = z_nom
        joints = _decode(policy, z_exec, proprio)
        env.step(joints)
        e_hist.append(_visible_e(cmd, vis_i, vis_flags))
    return e_hist[-1][:n_valid].detach()


def _i_stats(vals_cm: np.ndarray) -> dict:
    x = np.asarray(vals_cm, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "P_I_lt_0": float("nan"), "P_I_lt_0.5cm": float("nan"), "P_I_lt_1cm": float("nan"),
                "median_cm": float("nan"), "p25_cm": float("nan"), "p75_cm": float("nan")}
    return {
        "n": int(x.size),
        "P_I_lt_0": _rate(x < 0),
        "P_I_lt_0.5cm": _rate(x < -0.5),
        "P_I_lt_1cm": _rate(x < -1.0),
        "median_cm": float(np.median(x)),
        "p25_cm": float(np.percentile(x, 25)),
        "p75_cm": float(np.percentile(x, 75)),
        "mean_cm": float(x.mean()),
    }


@torch.inference_mode()
def _paired_on_snaps(
    env,
    policy,
    runner,
    injector,
    snaps: list[dict],
    vis_i,
    vis_flags,
    sixs,
    mu6,
    sd6,
    adapter,
    mua,
    sda,
    bins: tuple[float, ...],
    task: str,
    terrain: str,
) -> dict:
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    rows: list[dict] = []
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[t0] paired batch {start}:{start + n_valid}/{len(snaps)} {task}/{terrain}", flush=True)
        e_p = _rollout_probe(
            env, policy, runner, injector, batch, n_valid,
            d_raw=None, theta_deg=0.0, burst=0, horizon=HORIZON_PAIR,
            vis_i=vis_i, vis_flags=vis_flags,
        )
        rec = torch.stack([s["rec_obs"] for s in batch], dim=0).to(device=device, dtype=torch.float32)
        z0 = torch.stack([s["z_nom"] for s in batch], dim=0).to(device=device, dtype=torch.float32)
        d_pad = torch.zeros(n_envs, 16, device=device, dtype=torch.float32)
        th_pad = torch.full((n_envs,), 5.0, device=device, dtype=torch.float32)

        raw6 = sixs((rec - mu6) / sd6)
        d_pad[:n_valid] = raw6
        th_pad[:n_valid] = 5.0
        e_6s = _rollout_probe(
            env, policy, runner, injector, batch, n_valid,
            d_raw=d_pad, theta_deg=th_pad, burst=BURST_STEPS, horizon=HORIZON_PAIR,
            vis_i=vis_i, vis_flags=vis_flags,
        )

        raw_s, _lg = adapter((rec - mua) / sda)
        d_pad[:n_valid] = raw_s
        th_pad[:n_valid] = 5.0
        e_s5 = _rollout_probe(
            env, policy, runner, injector, batch, n_valid,
            d_raw=d_pad, theta_deg=th_pad, burst=BURST_STEPS, horizon=HORIZON_PAIR,
            vis_i=vis_i, vis_flags=vis_flags,
        )

        raw_a, logits = adapter((rec - mua) / sda)
        pred = logits.argmax(dim=-1)
        th_a = torch.tensor([float(bins[int(j)]) for j in pred.tolist()], device=device, dtype=torch.float32)
        d_pad[:n_valid] = raw_a
        th_pad[:n_valid] = th_a
        e_ad = _rollout_probe(
            env, policy, runner, injector, batch, n_valid,
            d_raw=d_pad, theta_deg=th_pad, burst=BURST_STEPS, horizon=HORIZON_PAIR,
            vis_i=vis_i, vis_flags=vis_flags,
        )

        for i in range(n_valid):
            ep = float(e_p[i].item())
            e6 = float(e_6s[i].item())
            es = float(e_s5[i].item())
            ea = float(e_ad[i].item())
            rows.append(
                {
                    "task": task,
                    "terrain": terrain,
                    "seed": int(batch[i]["seed"]),
                    "clip": str(batch[i].get("clip")),
                    "t0": int(batch[i]["t0"]),
                    "trigger_channel": str(batch[i].get("trigger_channel") or "?"),
                    "E0": float(batch[i].get("E0", float("nan"))),
                    "E_parent": ep,
                    "E_loco6s": e6,
                    "E_shared5": es,
                    "E_adaptive": ea,
                    "I_loco6s_cm": (e6 - ep) * 100.0,
                    "I_shared5_cm": (es - ep) * 100.0,
                    "I_adaptive_cm": (ea - ep) * 100.0,
                    "theta_adaptive": float(th_a[i].item()),
                    "z_nom_l2": float(z0[i].norm().item()),
                }
            )
    by_ch: dict[str, list[dict]] = {"S": [], "E": [], "both": []}
    for r in rows:
        ch = str(r.get("trigger_channel") or "?")
        if ch in by_ch:
            by_ch[ch].append(r)

    def _method_block(rs: list[dict]) -> dict:
        out = {"n": len(rs)}
        for key, name in (("I_loco6s_cm", "loco6s"), ("I_shared5_cm", "shared5"), ("I_adaptive_cm", "adaptive")):
            out[name] = _i_stats(np.array([r[key] for r in rs], dtype=np.float64))
        thetas = [r["theta_adaptive"] for r in rs]
        out["theta_adaptive"] = {
            "hist": {str(th): int(sum(abs(v - th) < 1e-6 for v in thetas)) for th in THETA_BINS},
            "mean": float(np.mean(thetas)) if thetas else float("nan"),
        }
        by_th: dict[str, list[float]] = {str(th): [] for th in THETA_BINS}
        for r in rs:
            by_th[str(r["theta_adaptive"])].append(r["I_adaptive_cm"])
        out["theta_vs_I_adaptive"] = {k: _i_stats(np.array(v, dtype=np.float64)) for k, v in by_th.items()}
        e0 = np.array([r["E0"] for r in rs], dtype=np.float64)
        out["E0_cm"] = _pct(e0 * 100.0)
        return out

    summary = {
        "task": task,
        "terrain": terrain,
        "n": len(rows),
        "all": _method_block(rows),
        "S_first": _method_block(by_ch["S"]),
        "E_first": _method_block(by_ch["E"]),
        "both": _method_block(by_ch["both"]),
        "rows": rows,
    }
    print(
        f"[t0] paired {task}/{terrain} n={len(rows)} "
        f"adapt P(I<0)={summary['all']['adaptive']['P_I_lt_0']} "
        f"med={summary['all']['adaptive']['median_cm']}",
        flush=True,
    )
    return summary


@torch.inference_mode()
def _rollout_cl(
    env,
    policy,
    runner,
    injector,
    *,
    rec_kind: str,
    sixs,
    mu6,
    sd6,
    adapter,
    mua,
    sda,
    bins: tuple[float, ...],
    steps: int,
    seed: int,
    terrain: str,
    theta_deg: float,
    burst_len: int,
    max_bursts: int,
    vis_i,
    vis_flags,
    s_enabled: bool,
    mask_name: str,
    task_source: str,
    log_adapter: bool,
    z_subsample: int,
    max_z_samples: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    n_envs = int(env.unwrapped.num_envs)
    use, paths = _pin_clips(env, mask_name=mask_name)
    cmd = env.unwrapped.command_manager.get_term("motion")
    device = env.unwrapped.device
    asset = env.unwrapped.scene["robot"]
    gravity = asset.data.GRAVITY_VEC_W
    if gravity.ndim == 1:
        gravity = gravity.unsqueeze(0).expand(cmd.num_envs, -1)

    gate = RecoveryRiskGate(s_enabled=bool(s_enabled))
    gate.reset()
    active_prev = torch.zeros(n_envs, dtype=torch.bool, device=device)
    burst_left = torch.zeros(n_envs, dtype=torch.long, device=device)
    held_d = torch.zeros(n_envs, 16, dtype=torch.float32, device=device)
    held_theta = torch.full((n_envs,), float(theta_deg), dtype=torch.float32, device=device)
    event_bursted = torch.zeros(n_envs, dtype=torch.bool, device=device)
    failed = torch.zeros(n_envs, dtype=torch.bool, device=device)
    fail_step = [-1] * use
    fail_reason = [""] * use
    e9_prev = None
    apply_recovery = rec_kind == "shared_adapt"
    log_adapt = bool(log_adapter) and rec_kind in ("shared_adapt", "shadow")
    shadow_rows: list[dict] = []
    z_samples: list[dict] = []
    first_trig = [-1] * use
    body_series = {
        (seed, e): {"torso": [], "lw": [], "rw": [], "S_raw": []} for e in range(use)
    }

    events: list[dict] = []
    cur: list[dict | None] = [None] * use
    open_bursts: list[dict | None] = [None] * use
    e_post_series: dict[tuple[int, int], list[float]] = {(seed, e): [] for e in range(use)}
    t_end_map: dict[tuple[int, int], int] = {(seed, e): steps - 1 for e in range(use)}

    robot_names = list(asset.data.body_names)
    pelvis_i = robot_names.index("pelvis") if "pelvis" in robot_names else None

    events: list[dict] = []
    cur: list[dict | None] = [None] * use
    open_bursts: list[dict | None] = [None] * use
    e_post_series: dict[tuple[int, int], list[float]] = {(seed, e): [] for e in range(use)}
    t_end_map: dict[tuple[int, int], int] = {(seed, e): steps - 1 for e in range(use)}

    for t in range(steps):
        obs, _ = env.get_observations()
        obs = injector.patch_policy_obs(obs, env, "mapper")
        e_pre = _visible_e(cmd, vis_i, vis_flags)
        e_torso, e_lw, e_rw = _body_e(cmd, vis_i)
        s_in = asset.data.root_lin_vel_w[:, 2].abs().to(dtype=e_pre.dtype)
        gout = gate.step(e_pre, s_in if bool(s_enabled) else torch.zeros_like(e_pre))
        active = gout["active"]
        z_nom, proprio = _z_and_proprio(policy, runner, obs)
        rec_pack, e9 = _recovery_pack(obs, z_nom, e9_prev)

        rising = active & ~active_prev
        falling = (~active) & active_prev

        if z_subsample > 0 and t >= WARMUP and (t % int(z_subsample) == 0) and len(z_samples) < int(max_z_samples):
            vis3 = rec_pack[:, 18:21]
            for i in range(use):
                if len(z_samples) >= int(max_z_samples):
                    break
                z_samples.append(
                    {
                        "task": task_source,
                        "terrain": terrain,
                        "seed": int(seed),
                        "t": int(t),
                        "kind": "subsample",
                        "z_nom": z_nom[i].detach().cpu().tolist(),
                        "E": float(e_pre[i].item()),
                        "S_raw": float(s_in[i].item()),
                        "vis": vis3[i].detach().cpu().tolist(),
                    }
                )

        for i in range(use):
            if failed[i]:
                continue
            if bool(rising[i]) and t >= WARMUP:
                r_e0 = float(gout["R_E"][i].item())
                r_s0 = float(gout["R_S"][i].item())
                if r_e0 >= 1.0 and r_s0 >= 1.0:
                    ch = "both"
                elif r_s0 >= 1.0:
                    ch = "S"
                else:
                    ch = "E"
                ev = {
                    "env": int(i),
                    "seed": int(seed),
                    "terrain": terrain,
                    "clip": Path(paths[i]).name if i < len(paths) else f"env{i}",
                    "t0": int(t),
                    "e0": float(e_pre[i].item()),
                    "S0": float(gout["S"][i].item()) if "S" in gout else float(s_in[i].item()),
                    "R_E0": r_e0,
                    "R_S0": r_s0,
                    "R0": float(gout["R"][i].item()),
                    "trigger_channel": ch,
                    "released": False,
                    "t_rel": None,
                    "failed": False,
                    "fail_after": False,
                    "auc": 0.0,
                    "n_act": 0,
                    "bursts": [],
                    "n_bursts": 0,
                    "vis_e_acc": [],
                }
                events.append(ev)
                cur[i] = ev
                event_bursted[i] = False
                if first_trig[i] < 0:
                    first_trig[i] = int(t)
                if not log_adapt:
                    z_samples.append(
                        {
                            "task": task_source,
                            "terrain": terrain,
                            "seed": int(seed),
                            "t": int(t),
                            "kind": "trigger",
                            "z_nom": z_nom[i].detach().cpu().tolist(),
                            "E": float(e_pre[i].item()),
                            "S_raw": float(s_in[i].item()),
                            "vis": rec_pack[i, 18:21].detach().cpu().tolist(),
                        }
                    )
            if cur[i] is not None and (not cur[i]["released"]) and (not cur[i]["failed"]):
                if bool(active[i]):
                    cur[i]["n_act"] += 1
                    cur[i]["auc"] += float(e_pre[i].item()) * DT
                    cur[i]["vis_e_acc"].append(float(e_pre[i].item()))
                if bool(falling[i]):
                    cur[i]["released"] = True
                    cur[i]["t_rel"] = int(t)
                    event_bursted[i] = False

        if log_adapt:
            rise_idx = [
                i
                for i in range(use)
                if (not bool(failed[i])) and bool(rising[i]) and t >= WARMUP
            ]
            if rise_idx:
                raw, th = _predict_raw_theta(
                    "shared_adapt", rec_pack, rise_idx, sixs, mu6, sd6, adapter, mua, sda, bins, theta_deg
                )
                vis3 = rec_pack[:, 18:21]
                for j, i in enumerate(rise_idx):
                    d_hat = project_tangent(raw[j : j + 1], z_nom[i : i + 1])[0]
                    row = {
                        "task": task_source,
                        "terrain": terrain,
                        "seed": int(seed),
                        "env": int(i),
                        "clip": Path(paths[i]).name if i < len(paths) else f"env{i}",
                        "t0": int(t),
                        "trigger_channel": cur[i]["trigger_channel"] if cur[i] else "?",
                        "E0": float(e_pre[i].item()),
                        "S_raw": float(s_in[i].item()),
                        "R_E": float(gout["R_E"][i].item()),
                        "R_S": float(gout["R_S"][i].item()),
                        "R": float(gout["R"][i].item()),
                        "theta_pred": float(th[j].item()),
                        "d_raw_norm": float(raw[j].norm().item()),
                        "d_hat_norm": float(d_hat.norm().item()),
                        "z_nom": z_nom[i].detach().cpu().tolist(),
                        "d_pred": raw[j].detach().cpu().tolist(),
                        "vis": vis3[i].detach().cpu().tolist(),
                        "applied": bool(apply_recovery),
                    }
                    shadow_rows.append(row)
                    z_samples.append(
                        {
                            "task": task_source,
                            "terrain": terrain,
                            "seed": int(seed),
                            "t": int(t),
                            "kind": "trigger",
                            "z_nom": row["z_nom"],
                            "E": row["E0"],
                            "S_raw": row["S_raw"],
                            "vis": row["vis"],
                            "theta_pred": row["theta_pred"],
                        }
                    )
                    if cur[i] is not None:
                        cur[i]["theta_pred"] = row["theta_pred"]
                        cur[i]["d_raw_norm"] = row["d_raw_norm"]
            if cur[i] is not None and (not cur[i]["released"]) and (not cur[i]["failed"]):
                if bool(active[i]):
                    cur[i]["n_act"] += 1
                    cur[i]["auc"] += float(e_pre[i].item()) * DT
                    cur[i]["vis_e_acc"].append(float(e_pre[i].item()))
                if bool(falling[i]):
                    cur[i]["released"] = True
                    cur[i]["t_rel"] = int(t)
                    event_bursted[i] = False

        has_open = torch.zeros(n_envs, dtype=torch.bool, device=device)
        for i in range(use):
            ev = cur[i]
            if ev is not None and (not ev["released"]) and (not ev["failed"]):
                has_open[i] = True
        need = has_open & (burst_left == 0) & (~failed) & (~event_bursted)
        if int(max_bursts) > 0:
            for i in range(use):
                if cur[i] is not None and int(cur[i]["n_bursts"]) >= int(max_bursts):
                    need[i] = False
        if not apply_recovery:
            need[:] = False

        idxs = need.nonzero(as_tuple=False).view(-1).tolist()
        if idxs:
            raw, th = _predict_raw_theta(
                rec_kind, rec_pack, idxs, sixs, mu6, sd6, adapter, mua, sda, bins, theta_deg
            )
            for j, i in enumerate(idxs):
                held_d[i] = raw[j]
                held_theta[i] = th[j]
                burst_left[i] = int(burst_len)
                event_bursted[i] = True
                if cur[i] is not None:
                    cur[i]["n_bursts"] += 1
                    b = {
                        "burst_index": int(cur[i]["n_bursts"]),
                        "t_start": int(t),
                        "e_before": float(e_pre[i].item()),
                        "e_after": None,
                        "dE_m": None,
                        "theta_deg": float(th[j].item()),
                    }
                    cur[i]["bursts"].append(b)
                    open_bursts[i] = b

        in_burst = burst_left > 0
        z_exec = z_nom.clone()
        if bool(in_burst.any()):
            z_exec[in_burst] = apply_angle(z_nom[in_burst], held_d[in_burst], held_theta[in_burst])
        joints = _decode(policy, z_exec, proprio)
        env.step(joints)

        e_post = _visible_e(cmd, vis_i, vis_flags)
        e_torso_p, e_lw_p, e_rw_p = _body_e(cmd, vis_i)
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

        finishing = (burst_left == 1) & in_burst
        burst_left = torch.where(in_burst, burst_left - 1, burst_left)

        for i in range(use):
            e_post_series[(seed, i)].append(float(e_post[i].item()))
            body_series[(seed, i)]["torso"].append(float(e_torso_p[i].item()))
            body_series[(seed, i)]["lw"].append(float(e_lw_p[i].item()))
            body_series[(seed, i)]["rw"].append(float(e_rw_p[i].item()))
            body_series[(seed, i)]["S_raw"].append(float(s_in[i].item()))
            if bool(finishing[i]) and open_bursts[i] is not None:
                open_bursts[i]["e_after"] = float(e_post[i].item())
                open_bursts[i]["dE_m"] = float(e_post[i].item()) - float(open_bursts[i]["e_before"])
                open_bursts[i] = None
            if (not bool(failed[i])) and t >= WARMUP:
                if bool(fall[i]):
                    failed[i] = True
                    fail_step[i] = t
                    fail_reason[i] = "fall"
                elif bool(fail_z[i]):
                    failed[i] = True
                    fail_step[i] = t
                    fail_reason[i] = "anchor_z"
                elif bool(fail_ori[i]):
                    failed[i] = True
                    fail_step[i] = t
                    fail_reason[i] = "anchor_ori"
                if bool(failed[i]):
                    t_end_map[(seed, i)] = t
                    burst_left[i] = 0
                    if open_bursts[i] is not None:
                        open_bursts[i]["e_after"] = float(e_post[i].item())
                        open_bursts[i]["dE_m"] = float(e_post[i].item()) - float(open_bursts[i]["e_before"])
                        open_bursts[i]["truncated"] = True
                        open_bursts[i] = None
                    if cur[i] is not None and (not cur[i]["released"]) and (not cur[i]["failed"]):
                        cur[i]["failed"] = True
                        cur[i]["fail_after"] = True
                        cur[i]["t_fail"] = t

        e9_prev = e9.detach()
        active_prev = active.clone()
        if (t + 1) % 50 == 0:
            e_show = [f"{float(e_post[i]):.3f}" for i in range(use)]
            n_on = int(active[:use].sum().item())
            n_b = int(in_burst[:use].sum().item())
            print(
                f"[t0] {rec_kind} {task_source} {terrain} s{seed} step {t+1}/{steps} "
                f"e_vis={e_show} active={n_on} burst={n_b}",
                flush=True,
            )

    for i in range(use):
        if open_bursts[i] is not None and open_bursts[i].get("e_after") is None:
            series = e_post_series[(seed, i)]
            open_bursts[i]["e_after"] = float(series[-1]) if series else open_bursts[i]["e_before"]
            open_bursts[i]["dE_m"] = float(open_bursts[i]["e_after"]) - float(open_bursts[i]["e_before"])
            open_bursts[i]["truncated"] = True

    _horizon_fill(events, e_post_series, t_end_map)

    ep_rows = []
    for i in range(use):
        e_vis = np.asarray(e_post_series[(seed, i)], dtype=np.float64)
        fail = fail_step[i] >= 0
        ep_len = (fail_step[i] + 1) if fail else steps
        bs = body_series[(seed, i)]
        n_ev = sum(1 for ev in events if ev["env"] == i)
        t0 = first_trig[i]
        lead = None
        if fail and t0 >= 0:
            lead = (fail_step[i] - t0) * DT
        trig_before = bool(fail and t0 >= 0 and t0 <= fail_step[i])
        tax = "success"
        if fail:
            if fail_reason[i] in ("fall", "anchor_z", "anchor_ori"):
                tax = "instability"
            else:
                tax = "other"
        ep_rows.append(
            {
                "clip": Path(paths[i]).name if i < len(paths) else f"env{i}",
                "seed": int(seed),
                "terrain": terrain,
                "sr_task": int(not fail),
                "sr_5cm": float((e_vis < 0.05).mean()) if e_vis.size else float("nan"),
                "sr_2cm": float((e_vis < 0.02).mean()) if e_vis.size else float("nan"),
                "e_kp_mean": float(e_vis.mean()) if e_vis.size else float("nan"),
                "e_kp": _series_stats(e_vis.tolist()),
                "e_torso": _series_stats(bs["torso"] if vis_flags[0] else []),
                "e_lw": _series_stats(bs["lw"] if vis_flags[1] else []),
                "e_rw": _series_stats(bs["rw"] if vis_flags[2] else []),
                "S_raw": _series_stats(bs["S_raw"]),
                "fail": int(fail),
                "fail_reason": fail_reason[i] or "none",
                "fail_taxonomy": tax,
                "fail_step": int(fail_step[i]),
                "episode_length": int(ep_len),
                "n_events": n_ev,
                "any_trigger": int(n_ev > 0),
                "first_trigger_t": int(t0),
                "trigger_before_fail": int(trig_before),
                "lead_time_s": lead,
                "task_completion": int(not fail),
            }
        )
    fail_map = {(seed, i): fail_step[i] for i in range(use)}
    ep_fail_map = {(seed, i): bool(fail_step[i] >= 0) for i in range(use)}
    reason_map = {(seed, i): fail_reason[i] for i in range(use)}
    for ev in events:
        key = (int(ev["seed"]), int(ev["env"]))
        fs = int(fail_map.get(key, -1))
        ev["ep_fail"] = bool(ep_fail_map.get(key, False))
        acc = ev.pop("vis_e_acc", [])
        ev["median_vis_e"] = float(np.median(acc)) if acc else float("nan")
        if fs >= 0 and fs >= int(ev["t0"]):
            dt = (fs - int(ev["t0"])) * DT
            ev["fail_dt_s"] = float(dt)
            ev["fall_05"] = bool(dt <= 0.5)
            ev["fall_10"] = bool(dt <= 1.0)
            ev["fall_20"] = bool(dt <= 2.0)
            if not ev.get("fail_after"):
                ev["fail_after"] = True
                ev["t_fail"] = fs
        else:
            ev["fail_dt_s"] = None
            ev["fall_05"] = False
            ev["fall_10"] = False
            ev["fall_20"] = False
    for row in shadow_rows:
        key = (int(row["seed"]), int(row["env"]))
        row["ep_fail"] = bool(ep_fail_map.get(key, False))
        row["fail_reason"] = reason_map.get(key, "") or "none"
        fs = int(fail_map.get(key, -1))
        row["fail_step"] = fs
        if fs >= 0:
            row["lead_time_s"] = (fs - int(row["t0"])) * DT
        else:
            row["lead_time_s"] = None
    return events, ep_rows, shadow_rows, z_samples


def _ep_monitor(all_eps: list[dict]) -> dict:
    n = len(all_eps)
    failed = [r for r in all_eps if r.get("fail")]
    ok = [r for r in all_eps if not r.get("fail")]
    leads = [r["lead_time_s"] for r in failed if r.get("lead_time_s") is not None]
    recall = (
        float(np.mean([r.get("trigger_before_fail", 0) for r in failed])) if failed else float("nan")
    )
    reasons = {}
    tax = {}
    for r in all_eps:
        k = r.get("fail_reason") or "none"
        reasons[k] = reasons.get(k, 0) + 1
        t = r.get("fail_taxonomy") or ("success" if not r.get("fail") else "other")
        tax[t] = tax.get(t, 0) + 1
    return {
        "n_episodes": n,
        "sr_task": float(np.mean([r.get("sr_task", r.get("task_completion", 0)) for r in all_eps])) if n else float("nan"),
        "sr_5cm": float(np.mean([r["sr_5cm"] for r in all_eps])) if n else float("nan"),
        "fail_frac": float(np.mean([r["fail"] for r in all_eps])) if n else float("nan"),
        "fall_frac": float(np.mean([r.get("fail_reason") == "fall" for r in all_eps])) if n else float("nan"),
        "P_any_trigger": float(np.mean([r.get("any_trigger", 0) for r in all_eps])) if n else float("nan"),
        "events_per_episode": float(np.mean([r.get("n_events", 0) for r in all_eps])) if n else float("nan"),
        "trigger_before_failure_recall": recall,
        "n_failed": len(failed),
        "n_success": len(ok),
        "lead_time_s": _pct(np.array(leads, dtype=np.float64), qs=(10, 25, 50, 75, 90)),
        "fail_reasons": reasons,
        "taxonomy": {k: (v / n if n else 0.0) for k, v in tax.items()},
        "success_trigger_duty": float(np.mean([r.get("n_events", 0) for r in ok])) if ok else float("nan"),
        "fail_trigger_duty": float(np.mean([r.get("n_events", 0) for r in failed])) if failed else float("nan"),
        "e_kp": _pct(np.array([r["e_kp_mean"] for r in all_eps], dtype=np.float64) * 100.0),
    }


def _run_terrain(env_cfg, agent_cfg, resume_path: str, motion_dir: str, terrain: str, out_root: Path) -> dict:
    cell = out_root / terrain
    cell.mkdir(parents=True, exist_ok=True)
    done_p = cell / "summary.json"
    if done_p.exists():
        print(f"[t0] skip existing {done_p}", flush=True)
        return json.loads(done_p.read_text())

    task = str(args_cli.task_source)
    rec_kind = str(args_cli.rec_kind)
    env, runner, policy, injector = _make_env(env_cfg, agent_cfg, resume_path, motion_dir, terrain)
    adapter, mua, sda, bins = _load_rm2_adapter(args_cli.adapter_ckpt, env.device)
    sixs = mu6 = sd6 = None
    cmd = env.unwrapped.command_manager.get_term("motion")
    body_names = list(cmd.cfg.body_names)
    vis_i = [body_names.index(n) for n in KP_VIS]
    mask_name = str(args_cli.mask_modes).split(",")[0].strip() or ("torso" if task == "loco" else "vr")
    vis_flags = MASK_VIS.get(mask_name, (True, True, True))
    s_enabled = bool(args_cli.s_enabled)
    seeds = [int(s) for s in args_cli.seeds.split(",") if s.strip()] or [int(args_cli.seed)]
    all_events: list[dict] = []
    all_eps: list[dict] = []
    all_shadow: list[dict] = []
    all_z: list[dict] = []
    log_adapter = rec_kind in ("shared_adapt", "shadow")
    for seed in seeds:
        print(
            f"[t0] ===== kind={rec_kind} task={task} s={int(s_enabled)} "
            f"terrain={terrain} seed={seed} =====",
            flush=True,
        )
        torch.manual_seed(seed)
        np.random.seed(seed)
        events, ep_rows, shadow_rows, z_rows = _rollout_cl(
            env,
            policy,
            runner,
            injector,
            rec_kind=rec_kind,
            sixs=sixs,
            mu6=mu6,
            sd6=sd6,
            adapter=adapter,
            mua=mua,
            sda=sda,
            bins=bins,
            steps=int(args_cli.steps),
            seed=seed,
            terrain=terrain,
            theta_deg=float(args_cli.theta_deg),
            burst_len=int(args_cli.burst_steps),
            max_bursts=int(args_cli.max_bursts),
            vis_i=vis_i,
            vis_flags=vis_flags,
            s_enabled=s_enabled,
            mask_name=mask_name,
            task_source=task,
            log_adapter=log_adapter,
            z_subsample=int(args_cli.z_subsample),
            max_z_samples=int(args_cli.max_z_samples),
        )
        all_events.extend(events)
        all_eps.extend(ep_rows)
        all_shadow.extend(shadow_rows)
        all_z.extend(z_rows)
        print(
            f"[t0] seed={seed} events={len(events)} bursts={sum(e['n_bursts'] for e in events)} "
            f"shadow={len(shadow_rows)} z={len(z_rows)}",
            flush=True,
        )

    recov = _agg_events(all_events)
    mon = _ep_monitor(all_eps)
    payload = {
        "step": "T0",
        "no_ppo": True,
        "success_predicate": {
            "source": "tracking_env_cfg.TerminationsCfg + fall_to_ground diagnostic",
            "fail": "fall (torso/pelvis z<0.4) OR bad_anchor_pos_z_only_terrain_rel(0.25) OR bad_anchor_ori(0.8)",
            "success": "episode reaches 400-step eval horizon without those failures",
            "not_used": ["SR@5", "obstacle_reach_succeeded", "ee_body_pos"],
            "sr_5cm_role": "secondary sparse-intent fidelity, never SR_task",
        },
        "p1_task": task,
        "mask": mask_name,
        "s_enabled": s_enabled,
        "rec_kind": rec_kind,
        "apply_recovery": rec_kind == "shared_adapt",
        "shadow_only": rec_kind == "shadow",
        "burst_steps": int(args_cli.burst_steps),
        "max_bursts": int(args_cli.max_bursts),
        "ckpt": resume_path,
        "mapper": args_cli.mapper_path,
        "adapter_ckpt": args_cli.adapter_ckpt,
        "terrain": terrain,
        "seeds": seeds,
        "steps": int(args_cli.steps),
        "n_envs": int(env.unwrapped.num_envs),
        "n_episodes": len(all_eps),
        "sr_task": mon["sr_task"],
        "sr_5cm": mon["sr_5cm"],
        "fail_frac": mon["fail_frac"],
        "fall_frac": mon["fall_frac"],
        "task_completion": mon["sr_task"],
        "monitor": recov,
        "episode_monitor": mon,
        "episodes": all_eps,
        "events": all_events,
        "n_shadow_rows": len(all_shadow),
        "n_z_samples": len(all_z),
    }
    (cell / "events.json").write_text(json.dumps(_sanitize(all_events), indent=2), encoding="utf-8")
    (cell / "episodes.json").write_text(json.dumps(_sanitize(all_eps), indent=2), encoding="utf-8")
    (cell / "summary.json").write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    if all_shadow:
        (cell / "shadow.json").write_text(json.dumps(_sanitize(all_shadow), indent=2), encoding="utf-8")
    if all_z:
        np.savez_compressed(
            cell / "z_nom.npz",
            z=np.asarray([r["z_nom"] for r in all_z], dtype=np.float32),
            meta=np.array(
                [json.dumps({k: v for k, v in r.items() if k != "z_nom"}) for r in all_z]
            ),
        )
    print(
        f"[t0] {task}/{rec_kind}/{terrain} SR_task={mon['sr_task']:.3f} SR@5={mon['sr_5cm']:.3f} "
        f"fail={mon['fail_frac']:.3f} recall={mon['trigger_before_failure_recall']}",
        flush=True,
    )
    try:
        env.close()
    except Exception:
        pass
    return payload

def _write_pooled(out_root: Path, resume_path: str) -> dict | None:
    cells = {}
    for ter in TERRAINS:
        p = out_root / ter / "summary.json"
        if p.exists():
            cells[ter] = json.loads(p.read_text())
    if not cells:
        return None
    all_events = []
    all_eps = []
    by_terrain = {}
    for ter, cell in cells.items():
        all_events.extend(cell.get("events") or [])
        all_eps.extend(cell.get("episodes") or [])
        recov = cell.get("recovery") or {}
        by_terrain[ter] = {
            "label": TERRAIN_LABEL.get(ter, ter),
            "sr_task": cell.get("sr_task"),
            "sr_5cm": cell.get("sr_5cm"),
            "fail_frac": cell.get("fail_frac"),
            "fall_frac": cell.get("fall_frac"),
            "task_completion": cell.get("task_completion"),
            "n_events": recov.get("n_events"),
            "n_episodes": cell.get("n_episodes"),
            "episode_monitor": cell.get("episode_monitor"),
            "trigger_channel": recov.get("trigger_channel"),
        }
    recov_all = _agg_events(all_events)
    mon_all = _ep_monitor(all_eps)
    pooled = {
        "step": "T0",
        "no_ppo": True,
        "p1_task": str(args_cli.task_source),
        "rec_kind": str(args_cli.rec_kind),
        "s_enabled": bool(args_cli.s_enabled),
        "ckpt": resume_path,
        "adapter_ckpt": args_cli.adapter_ckpt,
        "max_bursts": int(args_cli.max_bursts),
        "mask": str(args_cli.mask_modes),
        "terrains": sorted(cells),
        "n_episodes": len(all_eps),
        "n_events": recov_all["n_events"],
        "sr_task_all": mon_all["sr_task"],
        "sr_5cm_all": mon_all["sr_5cm"],
        "fail_frac_all": mon_all["fail_frac"],
        "fall_frac_all": mon_all["fall_frac"],
        "episode_monitor_all": mon_all,
        "recovery_all": recov_all,
        "by_terrain": by_terrain,
        "note": "T0 frozen task matrix. SR_task = no official fail termination. No PPO.",
    }
    (out_root / "pooled.json").write_text(json.dumps(_sanitize(pooled), indent=2), encoding="utf-8")
    print(
        f"[t0] POOLED {args_cli.task_source}/{args_cli.rec_kind} n_ep={len(all_eps)} "
        f"SR_task={mon_all['sr_task']:.3f} SR@5={mon_all['sr_5cm']:.3f} fail={mon_all['fail_frac']:.3f}",
        flush=True,
    )
    return pooled


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    direct = getattr(agent_cfg, "resume_checkpoint_path", None)
    if direct:
        resume_path = os.path.abspath(str(direct))
    else:
        resume_path = PARENT_CKPT if Path(PARENT_CKPT).is_file() else get_checkpoint_path(
            log_root, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    print(f"[t0] ckpt {resume_path}", flush=True)
    print(
        f"[t0] task={args_cli.task_source} kind={args_cli.rec_kind} "
        f"theta={args_cli.theta_deg} burst={args_cli.burst_steps} max_bursts={args_cli.max_bursts} "
        f"s_enabled={args_cli.s_enabled} mask={args_cli.mask_modes} NO PPO NO TRAIN",
        flush=True,
    )
    _restore_adapter(agent_cfg, resume_path)
    if hasattr(agent_cfg.policy, "intent_recovery"):
        agent_cfg.policy.intent_recovery = False
    if hasattr(agent_cfg.policy, "terrain_scan_dim"):
        agent_cfg.policy.terrain_scan_dim = 0
    if hasattr(agent_cfg.policy, "adapter"):
        agent_cfg.policy.adapter = "residual"

    motion_dir = args_cli.motion
    terrains = [t.strip() for t in args_cli.terrains.split(",") if t.strip()] or [args_cli.terrain]
    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for terrain in terrains:
        print(f"[t0] ===== {args_cli.task_source}/{args_cli.rec_kind}/{terrain} =====", flush=True)
        _run_terrain(env_cfg, agent_cfg, resume_path, motion_dir, terrain, out_dir)
        _write_pooled(out_dir, resume_path)
    print("[t0] done", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
