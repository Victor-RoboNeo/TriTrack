#!/usr/bin/env python3
"""Phase R-M2 — cloned-state utility. No PPO. No closed-loop.

Held-out clones: Parent / frozen Loco 6S@5° / Shared@5° / Shared adaptive /
Oracle direction+magnitude. Probe-B 100 ms then Parent to 0.5 s.

I = E_method(t+0.5) - E_parent(t+0.5). Task-preserving visible E.
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

# ---------------------------------------------------------------------------
# Pure-torch helpers (usable before Isaac AppLauncher / --self-test)
# ---------------------------------------------------------------------------

DT = 0.02
Q50_E = 0.046
Q90_E = 0.130
Q50_S = 0.041
Q90_S = 0.241
WARMUP = 10
HORIZON = 25  # 0.5 s
BURST_B = 5  # 100 ms @ 50 Hz
EPS_DEG = 1.0
SWEEP_DEG = (1.0, 2.5, 5.0, 7.5, 10.0)
LOCO_THETA_DEG = 5.0
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
STEP6S_CKPT = "/data/home/chenxiangyu/robotics/Anybody/results/p2r_step6s_full/model_best.pt"
THETA_BINS = (2.5, 5.0, 7.5, 10.0)


def _r_e(e_m: torch.Tensor) -> torch.Tensor:
    return (e_m - Q50_E) / max(Q90_E - Q50_E, 1e-6)


def tangent_basis(z: torch.Tensor) -> torch.Tensor:
    """Orthonormal tangent frame. ``z [N,D]`` unit → ``B [N,D,D-1]``, ``B^T z = 0``."""
    z = F.normalize(z, dim=-1, eps=1e-8)
    n, d = z.shape
    eye = torch.eye(d, device=z.device, dtype=z.dtype).unsqueeze(0).expand(n, -1, -1).clone()
    eye[:, :, 0] = z
    q, _r = torch.linalg.qr(eye)
    sign = torch.sign((q[:, :, 0] * z).sum(-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign.unsqueeze(-1)
    return q[:, :, 1:]


def project_tangent(d: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z, dim=-1, eps=1e-8)
    d = d - (d * z).sum(-1, keepdim=True) * z
    return F.normalize(d, dim=-1, eps=1e-8)


def apply_angle(z_nom: torch.Tensor, d_raw: torch.Tensor, theta_deg: float) -> torch.Tensor:
    d = project_tangent(d_raw, z_nom)
    tan_th = math.tan(math.radians(float(theta_deg)))
    return F.normalize(z_nom + tan_th * d, dim=-1, eps=1e-8)


def _self_test() -> None:
    torch.manual_seed(0)
    z = F.normalize(torch.randn(8, 16), dim=-1)
    b = tangent_basis(z)
    gram = b.transpose(-1, -2) @ b
    err_i = (gram - torch.eye(15, device=z.device)).abs().max().item()
    err_z = (b.transpose(-1, -2) @ z.unsqueeze(-1)).abs().max().item()
    z2 = apply_angle(z, b[:, :, 0], 1.0)
    ang = torch.rad2deg(torch.acos((z * z2).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)))
    assert err_i < 1e-5, err_i
    assert err_z < 1e-5, err_z
    assert float(ang.mean()) < 1.05 and float(ang.mean()) > 0.95, float(ang.mean())
    print(f"[rm-stoop-ora] self-test OK  I-err={err_i:.2e}  z-err={err_z:.2e}  ang={float(ang.mean()):.4f}°", flush=True)


if "--self-test" in sys.argv:
    _self_test()
    raise SystemExit(0)

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="P2-R RM2 cloned-state utility. No PPO.")
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--task", type=str, default="MUSE-Kp-LatentRL-Kp5-HeadHands-Locomani-G1-v0")
parser.add_argument("--motion", type=str, default=CLIP_STOOP)
parser.add_argument("--mask_modes", type=str, default="vr")
parser.add_argument("--task_source", type=str, default="stoop", choices=("loco", "stoop"))
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out", type=str, default="results/rm_intent_conditioned_recovery/eval/cloned_utility")
parser.add_argument("--start_frame", type=int, default=10)
parser.add_argument("--hint_iter", type=int, default=50000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--seeds", type=str, default="")
parser.add_argument("--terrain", type=str, default="plane", choices=TERRAINS)
parser.add_argument("--terrains", type=str, default="plane,light_rough,slope,steps")
parser.add_argument("--keep_headhands_spec", action="store_true", default=True)
parser.add_argument("--mapper_path", type=str, default=MAPPER_B)
parser.add_argument("--mapper_b_path", type=str, default=MAPPER_B)
parser.add_argument("--mapper_a_path", type=str, default="")
parser.add_argument("--residual_alpha", type=float, default=1.0)
parser.add_argument(
    "--adapter_ckpt",
    type=str,
    default="/data/home/chenxiangyu/robotics/Anybody/results/rm_intent_conditioned_recovery/checkpoints/model_best.pt",
)
parser.add_argument("--step6s_ckpt", type=str, default=STEP6S_CKPT)
parser.add_argument(
    "--split_json",
    type=str,
    default="/data/home/chenxiangyu/robotics/Anybody/results/rm_intent_conditioned_recovery/data/merged/split.json",
)
parser.add_argument(
    "--records_json",
    type=str,
    default="/data/home/chenxiangyu/robotics/Anybody/results/rm_intent_conditioned_recovery/data/merged/test.json",
)
parser.add_argument("--eval_split", type=str, default="test", choices=("train", "val", "test"))
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.modules.intent_recovery import RecoveryRiskGate, extract_visible_task_error  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import whole_body_tracking.tasks  # noqa: E402, F401
from causal_future import CausalFutureInjector  # noqa: E402


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
        print("[rm-stoop-ora] terrain=plane", flush=True)
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
    print(f"[rm-stoop-ora] terrain={terrain} 1x1 generator", flush=True)


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
                print(f"[rm-stoop-ora] pin mask {mask_name!r} idx={names.index(mask_name)}", flush=True)
        cmd._resample_command(ids)
        if hasattr(cmd, "resample_all_mask_modes"):
            cmd.resample_all_mask_modes()
        try:
            env.unwrapped.sim.forward()
        except Exception as exc:
            print(f"[rm-stoop-ora] sim.forward skipped: {exc}", flush=True)
    return use, [str(p) for p in paths[:use]]


def _visible_e(cmd, vis_i, vis_flags) -> torch.Tensor:
    delta = cmd.robot_body_pos_w - cmd.body_pos_w
    parts = [torch.linalg.norm(delta[:, vis_i[k]], dim=-1) for k, flag in enumerate(vis_flags) if flag]
    stacked = torch.stack(parts, dim=-1)
    return stacked.mean(dim=-1)


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
                # Saved as [1, max_len, ...] (per-env slice) with most recent at the end.
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


def _restore_batch(env, snaps: list[dict]) -> int:
    n = len(snaps)
    with torch.inference_mode():
        return _restore_batch_inner(env, snaps, n)


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


@torch.inference_mode()
def _parent_joints(policy, runner, obs: torch.Tensor) -> torch.Tensor:
    z, proprio = _z_and_proprio(policy, runner, obs)
    return _decode(policy, z, proprio)


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


def _pad_snaps(snaps: list[dict], n_envs: int) -> tuple[list[dict], int]:
    if not snaps:
        return snaps, 0
    n = len(snaps)
    if n >= n_envs:
        return snaps[:n_envs], n_envs
    padded = list(snaps) + [snaps[-1]] * (n_envs - n)
    return padded, n


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
    theta_deg: float,
    burst: int,
    horizon: int,
    vis_i,
    vis_flags,
) -> torch.Tensor:
    """Return E at t+horizon for the first ``n_valid`` envs. ``d_raw`` None = parent."""
    n_envs = int(env.unwrapped.num_envs)
    padded, _ = _pad_snaps(snaps, n_envs)
    _restore_batch(env, padded)
    cmd = env.unwrapped.command_manager.get_term("motion")
    device = env.unwrapped.device
    e_hist = []
    obs0 = padded[0]["obs"].to(device)
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


def _severity(r_e: float) -> str:
    if r_e < 1.5:
        return "mild"
    if r_e < 2.0:
        return "mid"
    return "severe"


def _channel_of(r_e: float, r_s: float) -> str:
    if r_e >= 1.0 and r_s >= 1.0:
        return "both"
    if r_s >= 1.0:
        return "S"
    return "E"


def _collect_triggers(
    env,
    policy,
    runner,
    injector,
    *,
    steps: int,
    warmup: int,
    first_only: bool,
    terrain: str,
    seed: int,
    clip_paths: list[str],
    vis_i,
    vis_flags,
    s_enabled: bool = True,
) -> list[dict]:
    n_envs = int(env.unwrapped.num_envs)
    gate = RecoveryRiskGate(s_enabled=bool(s_enabled))
    gate.reset()
    active_prev = torch.zeros(n_envs, dtype=torch.bool, device=env.unwrapped.device)
    collected = torch.zeros(n_envs, dtype=torch.bool, device=env.unwrapped.device)
    e9_prev = None
    snaps: list[dict] = []
    cmd = env.unwrapped.command_manager.get_term("motion")
    asset = env.unwrapped.scene["robot"]
    counts = {"S": 0, "E": 0, "both": 0}
    for t in range(steps):
        with torch.inference_mode():
            obs, _ = env.get_observations()
            obs = injector.patch_policy_obs(obs, env, "mapper")
            e = _visible_e(cmd, vis_i, vis_flags)
            s_in = asset.data.root_lin_vel_w[:, 2].abs().to(dtype=e.dtype)
            gout = gate.step(e, s_in if bool(s_enabled) else torch.zeros_like(e))
            active = gout["active"]
            rising = active & ~active_prev
            z_nom, _prop = _z_and_proprio(policy, runner, obs)
            rec_pack, e9 = _recovery_pack(obs, z_nom, e9_prev)
            if t >= warmup:
                take = rising & (~collected if first_only else torch.ones_like(rising))
                idxs = take.nonzero(as_tuple=False).view(-1).tolist()
                if idxs:
                    hist = _hist_snapshot(env)
                    for i in idxs:
                        r_e = float(gout["R_E"][i].item())
                        r_s = float(gout["R_S"][i].item())
                        ch = _channel_of(r_e, r_s)
                        extra = {
                            "obs": obs[i].detach().cpu().clone(),
                            "z_nom": z_nom[i].detach().cpu().clone(),
                            "rec_obs": rec_pack[i].detach().cpu().clone(),
                            "E0": float(e[i].item()),
                            "S0": float(s_in[i].item()),
                            "R_E": r_e,
                            "R_S": r_s,
                            "R": float(gout["R"][i].item()),
                            "trigger_channel": ch,
                            "t0": int(t),
                            "terrain": terrain,
                            "seed": int(seed),
                            "clip": Path(clip_paths[i]).name if i < len(clip_paths) else f"env{i}",
                        }
                        snaps.append(_capture_snap(env, i, hist, extra))
                        collected[i] = True
                        counts[ch] = counts.get(ch, 0) + 1
            joints = _parent_joints(policy, runner, obs)
            env.step(joints)
            e9_prev = e9.detach()
            active_prev = active
    print(
        f"[rm-stoop-ora] collected {len(snaps)} triggers terrain={terrain} seed={seed} "
        f"S={counts['S']} E={counts['E']} both={counts['both']}",
        flush=True,
    )
    return snaps


def _cap_by_channel(snaps: list[dict]) -> list[dict]:
    want = {
        "S": int(getattr(args_cli, "max_states_s", 40)),
        "E": int(getattr(args_cli, "max_states_e", 15)),
        "both": int(getattr(args_cli, "max_states_both", 10)),
    }
    buckets: dict[str, list[dict]] = {"S": [], "E": [], "both": []}
    for s in snaps:
        ch = str(s.get("trigger_channel") or "E")
        buckets.setdefault(ch, []).append(s)
    keep: list[dict] = []
    for ch in ("S", "E", "both"):
        keep.extend(buckets.get(ch, [])[: want[ch]])
    hard = int(getattr(args_cli, "max_states", 65))
    if len(keep) > hard:
        keep = keep[:hard]
    print(
        f"[rm-stoop-ora] cap kept {len(keep)}/"
        f"{len(snaps)} S={sum(1 for x in keep if x.get('trigger_channel')=='S')} "
        f"E={sum(1 for x in keep if x.get('trigger_channel')=='E')} "
        f"both={sum(1 for x in keep if x.get('trigger_channel')=='both')}",
        flush=True,
    )
    return keep


def _knn_cos(rec_obs: np.ndarray, dstar: np.ndarray, k: int = 5) -> float:
    if rec_obs.shape[0] < k + 1:
        return float("nan")
    x = rec_obs.astype(np.float64)
    mu = x.mean(0)
    sd = x.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    x = (x - mu) / sd
    d = dstar.astype(np.float64)
    d = d / np.clip(np.linalg.norm(d, axis=-1, keepdims=True), 1e-8, None)
    dist = ((x[:, None] - x[None, :]) ** 2).sum(-1)
    np.fill_diagonal(dist, np.inf)
    nn = np.argpartition(dist, kth=min(k, dist.shape[1] - 1), axis=1)[:, :k]
    cos = (d[:, None, :] * d[nn]).sum(-1).mean()
    return float(cos)


def _pct(x, qs=(25, 50, 75)) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{q}": float("nan") for q in qs} | {"n": 0, "mean": float("nan")}
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} | {"n": int(x.size), "mean": float(x.mean())}


def _rate(mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    return float(m.mean()) if m.size else float("nan")


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}

    def _block(rs: list[dict]) -> dict:
        if not rs:
            return {"n": 0}
        io = np.array([x["I_oracle_cm"] for x in rs], dtype=np.float64)
        de_p = np.array([x["dE_parent_cm"] for x in rs], dtype=np.float64)
        de_o = np.array([x["dE_oracle_cm"] for x in rs], dtype=np.float64)
        cases = [x["case"] for x in rs]
        sweep = {th: _pct([x["A_sweep_cm"][str(th)] for x in rs if str(th) in x.get("A_sweep_cm", {})]) for th in SWEEP_DEG}
        best_angs = [x.get("best_angle_deg") for x in rs if x.get("best_angle_deg") is not None]
        i_loco = np.array([x["I_loco_5deg_cm"] for x in rs if x.get("I_loco_5deg_cm") is not None], dtype=np.float64)
        cos_l = np.array([x["cos_loco_dstar"] for x in rs if x.get("cos_loco_dstar") is not None], dtype=np.float64)
        out = {
            "n": len(rs),
            "P_I_lt_0": _rate(io < 0),
            "P_I_lt_1cm": _rate(io < -1.0),
            "P_I_lt_2cm": _rate(io < -2.0),
            "I_oracle_cm": _pct(io),
            "median_oracle_advantage_cm": float(np.median(io)) if io.size else float("nan"),
            "dE_parent_cm": _pct(de_p),
            "dE_oracle_cm": _pct(de_o),
            "P_oracle_abs_recover": _rate(de_o < 0),
            "P_parent_abs_recover": _rate(de_p < 0),
            "case_A": cases.count("A") / len(rs),
            "case_B": cases.count("B") / len(rs),
            "case_C": cases.count("C") / len(rs),
            "best_angle_deg": _pct(best_angs) if best_angs else {"n": 0},
            "best_angle_mode": float(max(set(best_angs), key=best_angs.count)) if best_angs else float("nan"),
            "I_loco_5deg_cm": _pct(i_loco) if i_loco.size else {"n": 0},
            "P_I_loco_lt_0": _rate(i_loco < 0) if i_loco.size else float("nan"),
            "cos_loco_dstar": _pct(cos_l) if cos_l.size else {"n": 0},
            "median_cos_loco_dstar": float(np.median(cos_l)) if cos_l.size else float("nan"),
            "A_sweep_cm": {str(th): sweep[th] for th in SWEEP_DEG},
        }
        rec = np.stack([x["rec_obs"] for x in rs], axis=0)
        dst = np.stack([x["d_star"] for x in rs], axis=0)
        out["C_dir_knn"] = _knn_cos(rec, dst, k=5)
        return out

    by_ter: dict[str, list[dict]] = {t: [] for t in TERRAINS}
    by_ch: dict[str, list[dict]] = {"S": [], "E": [], "both": []}
    for r in rows:
        by_ter.setdefault(r["terrain"], []).append(r)
        by_ch.setdefault(str(r.get("trigger_channel") or "?"), []).append(r)
    s_first = by_ch.get("S", [])
    e_first = by_ch.get("E", [])
    i_or = np.array([r["I_oracle_cm"] for r in rows], dtype=np.float64)
    case_votes = [r["case"] for r in rows]
    majority = max(("A", "B", "C"), key=case_votes.count)
    return {
        "n": len(rows),
        "all": _block(rows),
        "S_first": _block(s_first),
        "E_first": _block(e_first),
        "by_channel": {ch: _block(by_ch.get(ch, [])) for ch in ("S", "E", "both")},
        "by_terrain": {t: _block(by_ter[t]) for t in TERRAINS},
        "majority_case": majority,
        "P_I_lt_0": _rate(i_or < 0),
        "P_I_lt_1cm": _rate(i_or < -1.0),
        "P_I_lt_2cm": _rate(i_or < -2.0),
    }


def _assign_case(a_sweep: dict[str, float]) -> str:
    """A: 1–5° already helps; B: only 7.5/10°; C: even 10° no parent-relative gain."""
    le5 = [a_sweep[str(th)] for th in (1.0, 2.5, 5.0) if str(th) in a_sweep]
    gt5 = [a_sweep[str(th)] for th in (7.5, 10.0) if str(th) in a_sweep]
    allv = le5 + gt5
    if le5 and min(le5) < -1.0:
        return "A"
    if allv and min(allv) < -1.0:
        return "B"
    if allv and min(allv) < 0.0:
        return "B"
    return "C"


def _ep_key_snap(s: dict) -> str:
    return f"{s['terrain']}|{s['seed']}|{s['clip']}"


def _load_6s_predictor(ckpt_path: str, device):
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from train_p2r_step6s import SupervisedRecoveryMLP, _project_unit

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SupervisedRecoveryMLP()
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    mu = torch.as_tensor(blob["obs_mean"], device=device, dtype=torch.float32)
    sd = torch.as_tensor(blob["obs_std"], device=device, dtype=torch.float32)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    return model, mu, sd, _project_unit


def _oracle_index(step5_root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for ter in TERRAINS:
        p = step5_root / ter / "states.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            k = f"{r['terrain']}|{r['seed']}|{r['clip']}|{r['t0']}"
            out[k] = r
    return out


def _eval_6s_states(
    env,
    policy,
    runner,
    injector,
    snaps: list[dict],
    vis_i,
    vis_flags,
    *,
    ckpt_path: str,
    oracle_map: dict[str, dict],
    thetas: tuple[float, ...] = (1.0, 5.0),
) -> list[dict]:
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    model, mu, sd, project_unit = _load_6s_predictor(ckpt_path, device)
    rows: list[dict] = []
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[6s] utility batch {start}:{start + n_valid} / {len(snaps)}", flush=True)
        j0 = _rollout_probe(
            env, policy, runner, injector, batch, n_valid,
            d_raw=None, theta_deg=0.0, burst=0, horizon=HORIZON,
            vis_i=vis_i, vis_flags=vis_flags,
        )
        rec = torch.stack([s["rec_obs"] for s in batch], dim=0).to(device=device, dtype=torch.float32)
        z0 = torch.stack([s["z_nom"] for s in batch], dim=0).to(device=device, dtype=torch.float32)
        x = (rec - mu) / sd
        with torch.no_grad():
            d_pred = project_unit(model(x), z0)
        d_pad = torch.zeros(n_envs, 16, device=device, dtype=d_pred.dtype)
        d_pad[:n_valid] = d_pred
        jp_by: dict[float, torch.Tensor] = {}
        for th in thetas:
            jp_by[th] = _rollout_probe(
                env, policy, runner, injector, batch, n_valid,
                d_raw=d_pad, theta_deg=float(th), burst=BURST_B, horizon=HORIZON,
                vis_i=vis_i, vis_flags=vis_flags,
            )
            print(f"[6s]   burst {th}° done", flush=True)
        e0 = torch.tensor([s["E0"] for s in batch], device=device, dtype=j0.dtype)
        for i in range(n_valid):
            key = f"{batch[i]['terrain']}|{batch[i]['seed']}|{batch[i]['clip']}|{batch[i]['t0']}"
            ora = oracle_map.get(key)
            if ora is not None:
                d_star = F.normalize(
                    torch.as_tensor(ora["d_star"], device=device, dtype=d_pred.dtype), dim=-1
                )
                cos = float((d_pred[i] * d_star).sum().item())
                i_or = float(ora["I_oracle_cm"])
            else:
                cos = float("nan")
                i_or = float("nan")
            row = {
                "terrain": batch[i]["terrain"],
                "seed": batch[i]["seed"],
                "clip": batch[i]["clip"],
                "t0": batch[i]["t0"],
                "E0_cm": float(e0[i].item()) * 100.0,
                "R_E": float(batch[i]["R_E"]),
                "E_parent_05_cm": float(j0[i].item()) * 100.0,
                "cos_pred_dstar": cos,
                "I_oracle_cm": i_or,
                "matched_oracle": ora is not None,
                "pair_rec_l2": float(batch[i].get("pair_rec_l2", float("nan"))),
                "pair_dt": int(batch[i].get("pair_dt", -1)),
                "oracle_t0": int(batch[i].get("oracle_t0", batch[i]["t0"])),
            }
            for th in thetas:
                i_cm = float((jp_by[th][i] - j0[i]).item()) * 100.0
                key_th = str(th).replace(".0", "")
                row[f"E_pred_{key_th}deg_05_cm"] = float(jp_by[th][i].item()) * 100.0
                row[f"I_pred_{key_th}deg_cm"] = i_cm
                row[f"regret_{key_th}deg_cm"] = (i_cm - i_or) if ora is not None else float("nan")
            if 5.0 in thetas:
                row["I_pred_cm"] = row["I_pred_5deg_cm"]
                row["regret_cm"] = row["regret_5deg_cm"]
                row["E_pred_05_cm"] = row["E_pred_5deg_05_cm"]
            elif thetas:
                th0 = thetas[0]
                k0 = str(th0).replace(".0", "")
                row["I_pred_cm"] = row[f"I_pred_{k0}deg_cm"]
                row["regret_cm"] = row[f"regret_{k0}deg_cm"]
                row["E_pred_05_cm"] = row[f"E_pred_{k0}deg_05_cm"]
            rows.append(row)
    return rows


def _probe_states(
    env,
    policy,
    runner,
    injector,
    snaps: list[dict],
    vis_i,
    vis_flags,
    *,
    do_a: bool,
    do_cons: bool,
    do_sweep: bool,
    step6s_ckpt: str = "",
) -> list[dict]:
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    rows: list[dict] = []
    tan_eps = math.tan(math.radians(EPS_DEG))
    rec_model = obs_mu = obs_sd = project_unit = None
    if step6s_ckpt:
        rec_model, obs_mu, obs_sd, project_unit = _load_6s_predictor(step6s_ckpt, device)
        print(f"[rm-stoop-ora] loaded frozen 6S {step6s_ckpt}", flush=True)
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(
            f"[rm-stoop-ora] probe batch {start}:{start + n_valid} / {len(snaps)} terrain={batch[0]['terrain']}",
            flush=True,
        )
        padded, _ = _pad_snaps(batch, n_envs)
        j0 = _rollout_probe(
            env, policy, runner, injector, batch, n_valid,
            d_raw=None, theta_deg=0.0, burst=0, horizon=HORIZON,
            vis_i=vis_i, vis_flags=vis_flags,
        )
        z0 = torch.stack([s["z_nom"] for s in padded], dim=0).to(device)
        e0 = torch.tensor([s["E0"] for s in padded], device=device, dtype=j0.dtype)
        b = tangent_basis(z0)
        # Probe B finite differences
        j_plus = []
        j_minus = []
        for i in range(15):
            d = b[:, :, i]
            jp = _rollout_probe(
                env, policy, runner, injector, batch, n_valid,
                d_raw=d, theta_deg=EPS_DEG, burst=BURST_B, horizon=HORIZON,
                vis_i=vis_i, vis_flags=vis_flags,
            )
            jm = _rollout_probe(
                env, policy, runner, injector, batch, n_valid,
                d_raw=-d, theta_deg=EPS_DEG, burst=BURST_B, horizon=HORIZON,
                vis_i=vis_i, vis_flags=vis_flags,
            )
            j_plus.append(jp)
            j_minus.append(jm)
            print(f"[rm-stoop-ora]   B fd {i + 1}/15", flush=True)
        jp_t = torch.stack(j_plus, dim=-1)
        jm_t = torch.stack(j_minus, dim=-1)
        g_i = (jp_t - jm_t) / (2.0 * tan_eps)
        g_t = torch.einsum("ni,ndi->nd", g_i, b[:n_valid])
        d_star = project_tangent(-g_t, z0[:n_valid])
        d_star_pad = torch.zeros(n_envs, 16, device=device, dtype=d_star.dtype)
        d_star_pad[:n_valid] = d_star

        a_fd = torch.cat([jp_t - j0.unsqueeze(-1), jm_t - j0.unsqueeze(-1)], dim=-1)
        i_fd = a_fd.min(dim=-1).values

        a_sweep: dict[str, torch.Tensor] = {}
        if do_sweep:
            for th in SWEEP_DEG:
                js = _rollout_probe(
                    env, policy, runner, injector, batch, n_valid,
                    d_raw=d_star_pad, theta_deg=th, burst=BURST_B, horizon=HORIZON,
                    vis_i=vis_i, vis_flags=vis_flags,
                )
                a_sweep[str(th)] = js - j0
                print(f"[rm-stoop-ora]   B sweep {th}°", flush=True)

        i_a = None
        d_star_a = None
        if do_a:
            ap, am = [], []
            for i in range(15):
                d = b[:, :, i]
                jp = _rollout_probe(
                    env, policy, runner, injector, batch, n_valid,
                    d_raw=d, theta_deg=EPS_DEG, burst=1, horizon=HORIZON,
                    vis_i=vis_i, vis_flags=vis_flags,
                )
                jm = _rollout_probe(
                    env, policy, runner, injector, batch, n_valid,
                    d_raw=-d, theta_deg=EPS_DEG, burst=1, horizon=HORIZON,
                    vis_i=vis_i, vis_flags=vis_flags,
                )
                ap.append(jp)
                am.append(jm)
            ap_t = torch.stack(ap, dim=-1)
            am_t = torch.stack(am, dim=-1)
            g_a = (ap_t - am_t) / (2.0 * tan_eps)
            g_ta = torch.einsum("ni,ndi->nd", g_a, b[:n_valid])
            d_star_a = project_tangent(-g_ta, z0[:n_valid])
            i_a = torch.cat([ap_t - j0.unsqueeze(-1), am_t - j0.unsqueeze(-1)], dim=-1).min(dim=-1).values
            print("[rm-stoop-ora]   A fd done", flush=True)

        cos_plus5 = None
        d_star_p5 = None
        if do_cons:
            _restore_batch(env, padded)
            cmd = env.unwrapped.command_manager.get_term("motion")
            for t in range(BURST_B):
                if t == 0:
                    obs = torch.stack([s["obs"].to(device) for s in padded], dim=0)
                else:
                    obs, _ = env.get_observations()
                    obs = injector.patch_policy_obs(obs, env, "mapper")
                joints = _parent_joints(policy, runner, obs)
                env.step(joints)
            hist5 = _hist_snapshot(env)
            obs5, _ = env.get_observations()
            obs5 = injector.patch_policy_obs(obs5, env, "mapper")
            z5, _ = _z_and_proprio(policy, runner, obs5)
            snaps5 = []
            e5 = _visible_e(cmd, vis_i, vis_flags)
            for i in range(n_envs):
                extra = {
                    "obs": obs5[i].detach().cpu().clone(),
                    "z_nom": z5[i].detach().cpu().clone(),
                    "E0": float(e5[i].item()),
                    "R_E": 0.0,
                    "t0": 0,
                    "terrain": padded[i]["terrain"],
                    "seed": padded[i]["seed"],
                    "clip": padded[i]["clip"],
                    "rec_obs": padded[i]["rec_obs"],
                }
                snaps5.append(_capture_snap(env, i, hist5, extra))
            b5 = tangent_basis(z5)
            p5, m5 = [], []
            for i in range(15):
                d = b5[:, :, i]
                jp = _rollout_probe(
                    env, policy, runner, injector, snaps5[:n_valid], n_valid,
                    d_raw=d, theta_deg=EPS_DEG, burst=BURST_B, horizon=HORIZON,
                    vis_i=vis_i, vis_flags=vis_flags,
                )
                jm = _rollout_probe(
                    env, policy, runner, injector, snaps5[:n_valid], n_valid,
                    d_raw=-d, theta_deg=EPS_DEG, burst=BURST_B, horizon=HORIZON,
                    vis_i=vis_i, vis_flags=vis_flags,
                )
                p5.append(jp)
                m5.append(jm)
            g5 = (torch.stack(p5, dim=-1) - torch.stack(m5, dim=-1)) / (2.0 * tan_eps)
            d_star_p5 = project_tangent(-torch.einsum("ni,ndi->nd", g5, b5[:n_valid]), z5[:n_valid])
            cos_plus5 = (d_star * d_star_p5).sum(-1)
            print("[rm-stoop-ora]   consistency +100ms done", flush=True)

        i_sweep = None
        if a_sweep:
            i_sweep = torch.stack(list(a_sweep.values()), dim=-1).min(dim=-1).values
        i_oracle = i_fd.clone()
        if i_sweep is not None:
            i_oracle = torch.minimum(i_oracle, i_sweep)
        if i_a is not None:
            i_oracle = torch.minimum(i_oracle, i_a)

        # Best E among probed directions (for ΔE_oracle)
        j_best = j0 + i_oracle

        i_loco = None
        cos_loco = None
        d_loco = None
        if rec_model is not None:
            rec = torch.stack([s["rec_obs"] for s in batch], dim=0).to(device=device, dtype=torch.float32)
            x = (rec - obs_mu) / obs_sd
            with torch.no_grad():
                d_loco = project_unit(rec_model(x), z0[:n_valid])
            d_loco_pad = torch.zeros(n_envs, 16, device=device, dtype=d_loco.dtype)
            d_loco_pad[:n_valid] = d_loco
            j_loco = _rollout_probe(
                env, policy, runner, injector, batch, n_valid,
                d_raw=d_loco_pad, theta_deg=LOCO_THETA_DEG, burst=BURST_B, horizon=HORIZON,
                vis_i=vis_i, vis_flags=vis_flags,
            )
            i_loco = j_loco - j0
            cos_loco = (d_loco * d_star).sum(-1)
            print(f"[rm-stoop-ora]   Loco 6S {LOCO_THETA_DEG}° done", flush=True)

        for i in range(n_valid):
            sw = {k: float(v[i].item()) * 100.0 for k, v in a_sweep.items()}
            best_ang = None
            if sw:
                best_k = min(sw, key=sw.get)
                best_ang = float(best_k)
            rec_np = batch[i]["rec_obs"]
            if torch.is_tensor(rec_np):
                rec_np = rec_np.detach().cpu().numpy()
            z_np = batch[i]["z_nom"]
            if torch.is_tensor(z_np):
                z_np = z_np.detach().cpu().numpy()
            row = {
                "terrain": batch[i]["terrain"],
                "seed": batch[i]["seed"],
                "clip": batch[i]["clip"],
                "t0": batch[i]["t0"],
                "trigger_channel": batch[i].get("trigger_channel", "?"),
                "E0_cm": float(e0[i].item()) * 100.0,
                "S0": float(batch[i].get("S0", float("nan"))),
                "R_E": float(batch[i]["R_E"]),
                "R_S": float(batch[i].get("R_S", float("nan"))),
                "R": float(batch[i].get("R", batch[i]["R_E"])),
                "severity": _severity(float(batch[i]["R_E"])),
                "E_parent_05_cm": float(j0[i].item()) * 100.0,
                "dE_parent_cm": float((j0[i] - e0[i]).item()) * 100.0,
                "E_oracle_05_cm": float(j_best[i].item()) * 100.0,
                "dE_oracle_cm": float((j_best[i] - e0[i]).item()) * 100.0,
                "I_oracle_cm": float(i_oracle[i].item()) * 100.0,
                "I_fd_cm": float(i_fd[i].item()) * 100.0,
                "A_sweep_cm": sw,
                "best_angle_deg": best_ang,
                "I_best_sweep_cm": None if not sw else float(min(sw.values())),
                "case": _assign_case(sw) if sw else "C",
                "g_T_norm": float(g_t[i].norm().item()),
                "d_star": d_star[i].detach().cpu().numpy().tolist(),
                "z_nom": np.asarray(z_np, dtype=np.float32).tolist(),
                "rec_obs": np.asarray(rec_np, dtype=np.float32).tolist(),
                "cos_dstar_plus5": None if cos_plus5 is None else float(cos_plus5[i].item()),
                "I_probe_a_cm": None if i_a is None else float(i_a[i].item()) * 100.0,
                "I_loco_5deg_cm": None if i_loco is None else float(i_loco[i].item()) * 100.0,
                "cos_loco_dstar": None if cos_loco is None else float(cos_loco[i].item()),
            }
            if d_star_a is not None:
                row["cos_dstar_A_B"] = float((d_star_a[i] * d_star[i]).sum().item())
            if d_loco is not None:
                row["d_loco"] = d_loco[i].detach().cpu().numpy().tolist()
            rows.append(row)
    return rows


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

    spec, probs = eval_single_mode_spec(str(args_cli.mask_modes).split(",")[0].strip() or "vr")
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


def _sixs_mode() -> bool:
    # Frozen 6S is compared inside Probe-B. Never skip the FD oracle.
    return False


def _snap_key(s: dict) -> str:
    return f"{s['terrain']}|{s['seed']}|{s['clip']}|{s['t0']}"


def _snap_rec_np(s) -> np.ndarray:
    a = s["rec_obs"] if isinstance(s, dict) else s
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float32)


def _pair_snaps_to_oracles(snaps: list[dict], oracles: list[dict]) -> list[tuple[dict, dict]]:
    """Greedy 1-1: same episode, nearest rec_obs (then |Δt0|)."""
    remaining = list(snaps)
    paired: list[tuple[dict, dict]] = []
    for ora in oracles:
        ep = _ep_key_snap(ora)
        cands = [s for s in remaining if _ep_key_snap(s) == ep]
        if not cands:
            print(f"[6s] no snap for oracle {_snap_key(ora)}", flush=True)
            continue
        ro = _snap_rec_np(ora)
        t0 = int(ora["t0"])

        def _score(s: dict) -> tuple[int, float]:
            rec_d = float(np.linalg.norm(_snap_rec_np(s) - ro))
            dt = abs(int(s["t0"]) - t0)
            return dt, rec_d

        best = min(cands, key=_score)
        dt, rec_d = _score(best)
        remaining = [s for s in remaining if s is not best]
        tagged = dict(best)
        tagged["pair_rec_l2"] = rec_d
        tagged["pair_dt"] = dt
        tagged["oracle_t0"] = int(ora["t0"])
        paired.append((tagged, ora))
        print(
            f"[6s] pair {_snap_key(ora)} <- t0={best['t0']} rec_l2={rec_d:.4f} dt={dt}",
            flush=True,
        )
    return paired


def _load_rm2_adapter(ckpt_path: str, device):
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from train_rm2_intent_adapter import IntentConditionedRecoveryAdapter, THETA_BINS as BINS

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = IntentConditionedRecoveryAdapter()
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    mu = torch.as_tensor(blob["obs_mean"], device=device, dtype=torch.float32)
    sd = torch.as_tensor(blob["obs_std"], device=device, dtype=torch.float32)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    bins = tuple(float(x) for x in (blob.get("theta_bins") or BINS))
    return model, mu, sd, bins


def _records_as_oracles(records: list[dict], task: str, terrain: str) -> list[dict]:
    out = []
    for r in records:
        if r.get("task_source") != task:
            continue
        ter = r.get("terrain_label") or r.get("terrain")
        if ter != terrain:
            continue
        out.append(
            {
                "terrain": ter,
                "seed": int(r["seed"]),
                "clip": str(r["clip"]),
                "t0": int(r["t0"]),
                "rec_obs": r["rec_obs"],
                "d_star": r["d_oracle"],
                "z_nom": r["z_nom"],
                "best_angle": r.get("best_angle"),
                "theta_idx": r.get("theta_idx"),
                "trigger_channel": r.get("trigger_channel"),
                "task_source": task,
                "I_oracle_stored_cm": r.get("I_oracle_cm"),
                "utility_by_angle": r.get("utility_by_angle") or {},
            }
        )
    return out


def _run_terrain(env_cfg, agent_cfg, resume_path: str, motion_dir: str, terrain: str, out_dir: Path) -> list[dict]:
    task = str(args_cli.task_source)
    cell = out_dir / task / terrain
    done_p = cell / "utility.json"
    if done_p.exists():
        print(f"[rm2-util] skip existing {done_p}", flush=True)
        return json.loads(done_p.read_text())
    records = json.loads(Path(args_cli.records_json).read_text())
    oracles = _records_as_oracles(records, task, terrain)
    if not oracles:
        print(f"[rm2-util] no test records for {task}/{terrain}", flush=True)
        cell.mkdir(parents=True, exist_ok=True)
        done_p.write_text("[]")
        return []
    want_seeds = sorted({int(o["seed"]) for o in oracles})
    env, runner, policy, injector = _make_env(env_cfg, agent_cfg, resume_path, motion_dir, terrain)
    cmd = env.unwrapped.command_manager.get_term("motion")
    body_names = list(cmd.cfg.body_names)
    vis_i = [body_names.index(n) for n in KP_VIS]
    mask_name = str(args_cli.mask_modes).split(",")[0].strip() or ("torso" if task == "loco" else "vr")
    vis_flags = MASK_VIS.get(mask_name, (True, True, True) if task == "stoop" else (True, False, False))
    s_on = task == "stoop"
    device = env.unwrapped.device
    n_envs = int(env.unwrapped.num_envs)
    sixs, mu6, sd6, project_unit = _load_6s_predictor(str(args_cli.step6s_ckpt), device)
    adapter, mu_a, sd_a, bins = _load_rm2_adapter(str(args_cli.adapter_ckpt), device)
    all_snaps: list[dict] = []
    for seed in want_seeds:
        print(f"[rm2-util] collect {task} terrain={terrain} seed={seed}", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        _use, paths = _pin_clips(env, mask_name=mask_name)
        snaps = _collect_triggers(
            env, policy, runner, injector,
            steps=int(args_cli.steps),
            warmup=WARMUP,
            first_only=False,
            terrain=terrain,
            seed=seed,
            clip_paths=paths,
            vis_i=vis_i,
            vis_flags=vis_flags,
            s_enabled=s_on,
        )
        all_snaps.extend(snaps)
    paired = _pair_snaps_to_oracles(all_snaps, oracles)
    print(f"[rm2-util] {task}/{terrain}: snaps={len(all_snaps)} oracles={len(oracles)} paired={len(paired)}", flush=True)
    rows: list[dict] = []
    for start in range(0, len(paired), n_envs):
        chunk = paired[start : start + n_envs]
        batch = [s for s, _o in chunk]
        oras = [_o for _s, _o in chunk]
        n_valid = len(batch)
        print(f"[rm2-util] utility batch {start}:{start + n_valid} / {len(paired)}", flush=True)
        j0 = _rollout_probe(
            env, policy, runner, injector, batch, n_valid,
            d_raw=None, theta_deg=0.0, burst=0, horizon=HORIZON,
            vis_i=vis_i, vis_flags=vis_flags,
        )
        rec = torch.stack(
            [torch.as_tensor(s["rec_obs"], dtype=torch.float32) for s in batch], dim=0
        ).to(device)
        z0 = torch.stack(
            [torch.as_tensor(s["z_nom"], dtype=torch.float32) for s in batch], dim=0
        ).to(device)
        d_ora = torch.stack(
            [F.normalize(torch.as_tensor(o["d_star"], dtype=torch.float32), dim=-1) for o in oras],
            dim=0,
        ).to(device)
        with torch.no_grad():
            d_6s = project_unit(sixs((rec - mu6) / sd6), z0)
            raw_a, logits_a = adapter((rec - mu_a) / sd_a)
            z_n = F.normalize(z0, dim=-1, eps=1e-8)
            d_sh = F.normalize(raw_a - (raw_a * z_n).sum(-1, keepdim=True) * z_n, dim=-1, eps=1e-8)
            th_idx = logits_a.argmax(-1)
        th_pred = [float(bins[int(i)]) for i in th_idx.tolist()]
        th_ora = []
        for o in oras:
            th = o.get("best_angle")
            if th is None:
                th = 5.0
            th_ora.append(float(th))

        def _run_dir(d_vec: torch.Tensor, theta: float) -> torch.Tensor:
            pad = torch.zeros(n_envs, 16, device=device, dtype=d_vec.dtype)
            pad[:n_valid] = d_vec
            return _rollout_probe(
                env, policy, runner, injector, batch, n_valid,
                d_raw=pad, theta_deg=float(theta), burst=BURST_B, horizon=HORIZON,
                vis_i=vis_i, vis_flags=vis_flags,
            )

        j_6s = _run_dir(d_6s, 5.0)
        j_sh5 = _run_dir(d_sh, 5.0)
        # adaptive: one theta per env — split unique thetas
        j_sha = torch.zeros(n_valid, device=device, dtype=j0.dtype)
        for th in sorted(set(th_pred)):
            sel = [i for i, t in enumerate(th_pred) if t == th]
            js = _run_dir(d_sh, th)
            for i in sel:
                j_sha[i] = js[i]
        j_ora = torch.zeros(n_valid, device=device, dtype=j0.dtype)
        for th in sorted(set(th_ora)):
            sel = [i for i, t in enumerate(th_ora) if abs(t - th) < 1e-6]
            js = _run_dir(d_ora, th)
            for i in sel:
                j_ora[i] = js[i]
        cos_sh = (d_sh * d_ora).sum(-1)
        cos_6s = (d_6s * d_ora).sum(-1)
        e0 = torch.tensor([float(s["E0"]) for s in batch], device=device, dtype=j0.dtype)
        for i in range(n_valid):
            rows.append(
                {
                    "task_source": task,
                    "terrain": terrain,
                    "seed": batch[i]["seed"],
                    "clip": batch[i]["clip"],
                    "t0": batch[i]["t0"],
                    "trigger_channel": oras[i].get("trigger_channel") or batch[i].get("trigger_channel"),
                    "pair_rec_l2": float(batch[i].get("pair_rec_l2", float("nan"))),
                    "pair_dt": int(batch[i].get("pair_dt", -1)),
                    "E0_cm": float(e0[i].item()) * 100.0,
                    "E_parent_05_cm": float(j0[i].item()) * 100.0,
                    "I_parent_cm": 0.0,
                    "I_6s_5deg_cm": float((j_6s[i] - j0[i]).item()) * 100.0,
                    "I_shared_5deg_cm": float((j_sh5[i] - j0[i]).item()) * 100.0,
                    "I_shared_adapt_cm": float((j_sha[i] - j0[i]).item()) * 100.0,
                    "I_oracle_cm": float((j_ora[i] - j0[i]).item()) * 100.0,
                    "theta_pred": th_pred[i],
                    "theta_oracle": th_ora[i],
                    "cos_shared_oracle": float(cos_sh[i].item()),
                    "cos_6s_oracle": float(cos_6s[i].item()),
                }
            )
    cell.mkdir(parents=True, exist_ok=True)
    done_p.write_text(json.dumps(rows))
    env.close()
    print(f"[rm2-util] wrote {done_p} n={len(rows)}", flush=True)
    return rows


def _i_block(vals) -> dict:
    x = np.asarray(vals, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "P_I_lt_0": float((x < 0).mean()),
        "P_I_lt_0.5cm": float((x < -0.5).mean()),
        "P_I_lt_1cm": float((x < -1.0).mean()),
        "median": float(np.median(x)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "mean": float(x.mean()),
    }


def _summarize_utility(rows: list[dict]) -> dict:
    methods = {
        "parent": "I_parent_cm",
        "loco_6s": "I_6s_5deg_cm",
        "shared_5deg": "I_shared_5deg_cm",
        "shared_adapt": "I_shared_adapt_cm",
        "oracle": "I_oracle_cm",
    }

    def _group(rs: list[dict]) -> dict:
        if not rs:
            return {"n": 0}
        out = {"n": len(rs)}
        ora = np.array([r["I_oracle_cm"] for r in rs], dtype=np.float64)
        for name, key in methods.items():
            blk = _i_block([r[key] for r in rs])
            if name != "oracle" and ora.size:
                rg = np.array([r[key] for r in rs], dtype=np.float64) - ora
                blk["oracle_regret_median_cm"] = float(np.median(rg[np.isfinite(rg)])) if np.isfinite(rg).any() else float("nan")
            out[name] = blk
        out["cos_shared_oracle"] = _pct([r.get("cos_shared_oracle") for r in rs])
        out["mean_theta_pred"] = float(np.mean([r["theta_pred"] for r in rs]))
        out["mean_theta_oracle"] = float(np.mean([r["theta_oracle"] for r in rs]))
        return out

    by_task = {"loco": [], "stoop": []}
    by_ch = {"S": [], "E": [], "both": []}
    by_ter: dict[str, list] = {t: [] for t in TERRAINS}
    for r in rows:
        by_task.setdefault(r.get("task_source") or "?", []).append(r)
        by_ch.setdefault(str(r.get("trigger_channel") or "?"), []).append(r)
        by_ter.setdefault(r["terrain"], []).append(r)
    stoop = by_task.get("stoop") or []
    return {
        "n": len(rows),
        "all": _group(rows),
        "loco": _group(by_task.get("loco") or []),
        "stoop": _group(stoop),
        "stoop_s_first": _group([r for r in stoop if r.get("trigger_channel") == "S"]),
        "stoop_e_first": _group([r for r in stoop if r.get("trigger_channel") == "E"]),
        "by_terrain": {t: _group(by_ter[t]) for t in TERRAINS},
        "by_channel": {ch: _group(by_ch.get(ch) or []) for ch in ("S", "E", "both")},
    }


def _load_all_utility_rows(out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for task in ("loco", "stoop"):
        for ter in TERRAINS:
            p = out_dir / task / ter / "utility.json"
            if p.exists():
                rows.extend(json.loads(p.read_text()))
    return rows


def _write_summary(out_dir: Path, rows: list[dict], resume_path: str) -> dict:
    summary = {
        "step": "rm2-cloned-utility",
        "no_ppo": True,
        "ckpt": resume_path,
        "adapter": str(args_cli.adapter_ckpt),
        "step6s_ckpt": str(args_cli.step6s_ckpt),
        "metrics": _summarize_utility(rows),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    m = summary["metrics"]
    def _line(name, blk):
        if not blk or not blk.get("n"):
            return f"{name}: n=0"
        s = blk.get("shared_adapt") or {}
        b = blk.get("loco_6s") or {}
        return (
            f"{name} n={blk['n']} "
            f"shared P<0={s.get('P_I_lt_0')} med={s.get('median')} "
            f"6s P<0={b.get('P_I_lt_0')} med={b.get('median')}"
        )
    print("[rm2-util] " + _line("loco", m.get("loco")), flush=True)
    print("[rm2-util] " + _line("stoop-S", m.get("stoop_s_first")), flush=True)
    print("[rm2-util] " + _line("stoop-E", m.get("stoop_e_first")), flush=True)
    print(f"[rm2-util] wrote {out_dir / 'summary.json'}", flush=True)
    return summary


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    _self_test()
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    direct = getattr(agent_cfg, "resume_checkpoint_path", None)
    if direct:
        resume_path = os.path.abspath(str(direct))
    else:
        resume_path = PARENT_CKPT if Path(PARENT_CKPT).is_file() else get_checkpoint_path(
            log_root, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    print(f"[rm2-util] ckpt {resume_path} task_source={args_cli.task_source}", flush=True)
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
        print(f"[rm2-util] ===== {args_cli.task_source}/{terrain} =====", flush=True)
        _run_terrain(env_cfg, agent_cfg, resume_path, motion_dir, terrain, out_dir)
        _write_summary(out_dir, _load_all_utility_rows(out_dir), resume_path)
    _write_summary(out_dir, _load_all_utility_rows(out_dir), resume_path)
    print("[rm2-util] done", flush=True)



if __name__ == "__main__":
    main()
    simulation_app.close()
