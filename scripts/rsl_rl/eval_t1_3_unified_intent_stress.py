#!/usr/bin/env python3
"""Phase T1.3 — Unified coupled human-intent stress (time × space, before Mapper-B).

I_stress = S_space(S_time(I_original, s_t), s_x)
No task ID / terrain in the operator. Frozen Parent. No PPO.
Reach / Carry: R-M3 shadow only (z_exec = z_nom).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque
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
URA0_HIST = 15  # 300 ms history for ablation
UCR2P_HIST = 25  # 500 ms process ring

LAM_FAIL_DEFAULT = 1.0
EPS_I_REC = 0.0025  # 0.25 cm dead-zone, meters
UCR0_EPS_DEG = 1.0
UCR0_SWEEP_DEG = (1.0, 2.5, 5.0, 7.5, 10.0)
RE_ON_DEFAULT = 1.0
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
PAYLOAD_CAP_KG = 12.0
PUSH_STEP_DEFAULT = {"loco": 150, "stoop": 80, "reach": 100, "carry": 150}
WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
REC_KINDS = ("none", "shared_adapt", "shadow")
TASK_SOURCES = ("loco", "stoop", "reach", "carry")
S_ENABLED_TASKS = ("stoop", "reach", "carry")
BODY_KEYS = ("torso", "lw", "rw")


def project_tangent(d: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z, dim=-1, eps=1e-8)
    d = d - (d * z).sum(-1, keepdim=True) * z
    return F.normalize(d, dim=-1, eps=1e-8)


def tangent_basis(z: torch.Tensor) -> torch.Tensor:
    """Orthonormal tangent frame. z [N,D] unit → B [N,D,D-1], B^T z = 0."""
    z = F.normalize(z, dim=-1, eps=1e-8)
    n, d = z.shape
    eye = torch.eye(d, device=z.device, dtype=z.dtype).unsqueeze(0).expand(n, -1, -1).clone()
    eye[:, :, 0] = z
    q, _r = torch.linalg.qr(eye)
    sign = torch.sign((q[:, :, 0] * z).sum(-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign.unsqueeze(-1)
    return q[:, :, 1:]


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

parser = argparse.ArgumentParser(description="P2-R Phase T1.3 unified intent-stress frozen eval. No PPO.")
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
parser.add_argument("--stress_family", type=str, default="none",
                    choices=("none", "payload", "com", "workspace", "push"))
parser.add_argument("--payload_kg", type=float, default=0.0,
                    help="Total extra mass (kg) split equally on L/R wrist_yaw_link.")
parser.add_argument("--com_x", type=float, default=0.0, help="Wrist COM offset x (m), body frame.")
parser.add_argument("--com_y", type=float, default=0.0, help="Wrist COM offset y/lateral (m).")
parser.add_argument("--com_z", type=float, default=0.0, help="Wrist COM offset z (m).")
parser.add_argument("--ws_fwd", type=float, default=0.0, help="Reach target offset forward (clip +X, m).")
parser.add_argument("--ws_lat", type=float, default=0.0, help="Reach target offset lateral (clip +Y, m).")
parser.add_argument("--ws_z", type=float, default=0.0, help="Reach target offset vertical (clip +Z, m).")
parser.add_argument("--ws_bodies", type=str, default="right",
                    choices=("right", "left", "both", "visible"))
parser.add_argument("--push_dv", type=float, default=0.0, help="Single lateral/fwd Δv (m/s), robot frame.")
parser.add_argument("--push_axis", type=str, default="lat", choices=("lat", "fwd"))
parser.add_argument("--push_step", type=int, default=-1, help="-1 = task-aligned default.")
parser.add_argument("--severity_name", type=str, default="")
parser.add_argument("--intent_scale", type=float, default=1.0,
                    help="Scale p_rel = p_wrist - p_torso on human clip before Mapper-B.")
parser.add_argument("--intent_mode", type=str, default="radial",
                    choices=("radial", "lateral", "forward", "low", "cross"))
parser.add_argument("--intent_speed", type=float, default=1.0,
                    help="Unified temporal scale s_t. I(t)=I_original(s_t*t).")
parser.add_argument("--intent_spatial", type=float, default=1.0,
                    help="Unified spatial scale s_x on torso Δp and wrist Δr. No task ID.")
parser.add_argument("--lambda_id", type=float, default=-1.0,
                    help="Unified severity λ index if on the canonical ladder; else -1.")
parser.add_argument("--ufr_dump", action="store_true", default=False,
                    help="Write compact per-step UFR arrays. No control change.")
parser.add_argument("--ufr2_dump", action="store_true", default=False,
                    help="Write Parent-relative dump: q/dq/q_target/root. No control change.")
parser.add_argument("--ura0_clone", action="store_true", default=False,
                    help="URA-0: uniform Parent-state clones vs frozen R-M3 single burst. No closed-loop.")
parser.add_argument("--ura0_sample_ts", type=str, default="40,100,160,220,280,340",
                    help="Parent steps at which to clone (uniform, not RE/RS-triggered).")
parser.add_argument("--lam_fail", type=float, default=1.0,
                    help="Unified physical-failure penalty in J (meters-equivalent).")
parser.add_argument("--ucr0_clone", action="store_true", default=False,
                    help="UCR-0: clone only R_E-triggered states vs frozen R-M3 + local oracle.")
parser.add_argument("--ucr0_persist", type=int, default=3,
                    help="Consecutive frames with R_E>=re_on before capturing (unified persist).")
parser.add_argument("--re_on", type=float, default=RE_ON_DEFAULT,
                    help="Unified R_E trigger threshold. No RS. No task ID.")
parser.add_argument("--ucr0_max_snaps", type=int, default=48,
                    help="Cap RE-trigger clones per cell.")
parser.add_argument("--ucr1_dump", action="store_true", default=False,
                    help="UCR-1: also dump rec_obs/z_nom/d_star/j_by_angle + restorable snaps.pt.")
parser.add_argument("--ucr1_heldout", action="store_true", default=False,
                    help="UCR-1: restore test snaps and roll listed adapters only. No 400-step collect.")
parser.add_argument("--ucr1_snaps", type=str, default="",
                    help="Path to snaps.pt for --ucr1_heldout.")
parser.add_argument("--ucr1_adapters_json", type=str, default="",
                    help="JSON list of {name, path} adapters to roll on held-out snaps.")
parser.add_argument("--ucr1_test_clips", type=str, default="",
                    help="JSON list of clip filenames in the test split (held-out filter).")
parser.add_argument("--wrist_torso_max", type=float, default=0.85,
                    help="Skip first-trigger snap if wrist-torso distance exceeds this (m).")
parser.add_argument("--ucr1d_probe", action="store_true", default=False,
                    help="UCR-1D: restore test snaps and roll kNN-assigned (d,θ) plus ±1° FD set.")
parser.add_argument("--ucr1d_assign", type=str, default="",
                    help="Path to UCR-1D assignments.npz")
parser.add_argument("--ucr1d_fd", action="store_true", default=False,
                    help="Also roll 15-D ±1° FD for label-set width.")
parser.add_argument("--ucr2p_dump", action="store_true", default=False,
                    help="UCR-2P: dump 500ms causal process ring at first RE-trigger. No oracle, no closed-loop.")
parser.add_argument("--ucr2p_hist", type=int, default=25, help="Process frames at 50 Hz (25=500ms).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False
if bool(getattr(args_cli, "ucr1_dump", False)):
    args_cli.ucr0_clone = True
if bool(getattr(args_cli, "no_s_enabled", False)):
    args_cli.s_enabled = False
elif str(args_cli.task_source) in S_ENABLED_TASKS:
    args_cli.s_enabled = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import yaml  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_rotate_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.modules.intent_recovery import RecoveryRiskGate, extract_visible_task_error  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import whole_body_tracking.tasks  # noqa: E402, F401
from causal_future import (  # noqa: E402
    FUTURE_OFFSETS,
    N_VISIBLE,
    CausalFutureInjector,
    _lerp_horizon,
    gather_ref_abs_anchor,
)
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
        print("[t1] terrain=plane", flush=True)
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
    print(f"[t1] terrain={terrain} 1x1 generator", flush=True)


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


def _stress_meta() -> dict:
    task = str(args_cli.task_source)
    push_step = int(args_cli.push_step)
    if push_step < 0:
        push_step = int(PUSH_STEP_DEFAULT.get(task, 150))
    return {
        "family": str(args_cli.stress_family),
        "severity_name": str(args_cli.severity_name or args_cli.stress_family),
        "payload_kg": float(args_cli.payload_kg),
        "payload_cap_kg": PAYLOAD_CAP_KG,
        "com_xyz_m": [float(args_cli.com_x), float(args_cli.com_y), float(args_cli.com_z)],
        "ws_fwd_m": float(args_cli.ws_fwd),
        "ws_lat_m": float(args_cli.ws_lat),
        "ws_z_m": float(args_cli.ws_z),
        "ws_bodies": str(args_cli.ws_bodies),
        "push_dv_mps": float(args_cli.push_dv),
        "push_axis": str(args_cli.push_axis),
        "push_step": push_step,
        "intent_scale": 1.0,
        "intent_mode": "none",
        "intent_speed": float(args_cli.intent_speed),
        "intent_spatial": float(args_cli.intent_spatial),
        "lambda_id": float(args_cli.lambda_id),
        "unified_operator": "S_space(S_time(I,s_t),s_x); no task id; no terrain",
        "time_warp": {
            "formula": "I_s(t) = I_original(s * t)",
            "before_mapper_b": True,
            "position": "linear interpolation",
            "orientation": "quaternion SLERP (wxyz, shortest path)",
            "clip_end": "hold_final_valid_human_intent",
            "no_wrap": True,
            "no_repeat": True,
            "no_extrapolate": True,
            "robot_dt_unchanged": True,
            "policy_hz_unchanged": True,
        },
        "no_carry_object": True,
        "perception_stress": "not_applicable_to_current_parent",
        "g1_total_mass_kg": 35.12,
        "wrist_yaw_mass_kg": 0.085,
    }


def _wrist_cfg(env):
    cfg = SceneEntityCfg("robot", body_names=list(WRIST_BODIES))
    cfg.resolve(env.unwrapped.scene)
    return cfg


def _apply_payload_com(env, snapshot: dict | None) -> dict:
    """Add wrist mass + COM offset using Isaac Lab physx APIs. Re-apply from snapshot after reset."""
    from isaaclab.envs.mdp.events import randomize_rigid_body_mass
    from whole_body_tracking.tasks.tracking.mdp.events import randomize_rigid_body_com

    asset = env.unwrapped.scene["robot"]
    n = int(env.unwrapped.num_envs)
    env_ids = torch.arange(n, device="cpu")
    if snapshot is None:
        snapshot = {
            "masses": asset.root_physx_view.get_masses().clone(),
            "coms": asset.root_physx_view.get_coms().clone(),
            "inertias": asset.root_physx_view.get_inertias().clone(),
        }
    asset.root_physx_view.set_masses(snapshot["masses"].clone(), env_ids)
    asset.root_physx_view.set_coms(snapshot["coms"].clone(), env_ids)
    asset.root_physx_view.set_inertias(snapshot["inertias"].clone(), env_ids)

    extra = float(args_cli.payload_kg)
    if extra > PAYLOAD_CAP_KG + 1e-6:
        raise ValueError(f"payload_kg={extra} exceeds physical cap {PAYLOAD_CAP_KG} kg")
    per_wrist = extra / 2.0
    cfg = _wrist_cfg(env)
    if extra > 0.0:
        randomize_rigid_body_mass(
            env.unwrapped,
            env_ids=None,
            asset_cfg=cfg,
            mass_distribution_params=(per_wrist, per_wrist),
            operation="add",
            recompute_inertia=True,
        )
    cx, cy, cz = float(args_cli.com_x), float(args_cli.com_y), float(args_cli.com_z)
    if any(abs(v) > 1e-9 for v in (cx, cy, cz)):
        randomize_rigid_body_com(
            env.unwrapped,
            env_ids=None,
            com_range={"x": (cx, cx), "y": (cy, cy), "z": (cz, cz)},
            asset_cfg=cfg,
        )
    masses = asset.root_physx_view.get_masses()
    names = list(asset.data.body_names)
    wrist_mass = {}
    for name in WRIST_BODIES:
        if name in names:
            wrist_mass[name] = float(masses[0, names.index(name)].item())
    print(
        f"[t1] physics payload_kg={extra} per_wrist={per_wrist:.3f} "
        f"com=({cx},{cy},{cz}) wrist_mass={wrist_mass}",
        flush=True,
    )
    return snapshot


def _ws_body_names(mask_name: str) -> list[str]:
    mode = str(args_cli.ws_bodies)
    if mode == "right":
        return ["right_wrist_yaw_link"]
    if mode == "left":
        return ["left_wrist_yaw_link"]
    if mode == "both":
        return list(WRIST_BODIES)
    vis = MASK_VIS.get(mask_name, (True, True, True))
    out = []
    if vis[1]:
        out.append("left_wrist_yaw_link")
    if vis[2]:
        out.append("right_wrist_yaw_link")
    return out or ["right_wrist_yaw_link"]


def _quat_slerp_batch(q0: torch.Tensor, q1: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Batched shortest-path SLERP. q: [..., 4] wxyz. w: broadcastable to q[..., :1]."""
    q0 = F.normalize(q0, dim=-1, eps=1e-8)
    q1 = q1.clone()
    if w.ndim < q0.ndim:
        w = w.unsqueeze(-1)
    d = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(d < 0.0, -q1, q1)
    d = d.abs().clamp(max=1.0 - 1e-6)
    omega = torch.acos(d)
    sin_omega = torch.sin(omega)
    near = sin_omega.abs() < 1e-6
    s0 = torch.sin((1.0 - w) * omega) / sin_omega.clamp(min=1e-8)
    s1 = torch.sin(w * omega) / sin_omega.clamp(min=1e-8)
    out = F.normalize(s0 * q0 + s1 * q1, dim=-1, eps=1e-8)
    lerp = F.normalize((1.0 - w) * q0 + w * q1, dim=-1, eps=1e-8)
    return torch.where(near, lerp, out)


def _pct4(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "max": float("nan"), "n": 0}
    return {
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
        "n": int(x.size),
    }


def _kin_arrays(pos: torch.Tensor, fps: float) -> tuple[np.ndarray, np.ndarray]:
    p = pos.detach().float().cpu()
    dt = 1.0 / max(float(fps), 1e-6)
    if int(p.shape[0]) < 3:
        z = np.asarray([], dtype=np.float64)
        return z, z
    v = (p[1:] - p[:-1]) / dt
    a = (v[1:] - v[:-1]) / dt
    return torch.linalg.norm(v, dim=-1).numpy(), torch.linalg.norm(a, dim=-1).numpy()


def _kin_block(pos: torch.Tensor, fps: float) -> dict:
    """pos [T, 3] → linear speed / acc percentiles (m/s, m/s^2)."""
    sp, ac = _kin_arrays(pos, fps)
    return {"vel": _pct4(sp), "acc": _pct4(ac)}


def _kin_ang(quat: torch.Tensor, fps: float) -> dict:
    """quat [T, 4] wxyz → angular-speed percentiles (rad/s)."""
    q = F.normalize(quat.detach().float().cpu(), dim=-1, eps=1e-8)
    dt = 1.0 / max(float(fps), 1e-6)
    if int(q.shape[0]) < 2:
        return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "max": float("nan"), "n": 0}
    q0, q1 = q[:-1], q[1:]
    d = (q0 * q1).sum(dim=-1).abs().clamp(max=1.0)
    ang = (2.0 * torch.acos(d) / dt).numpy()
    x = np.asarray(ang, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "max": float("nan"), "n": 0}
    return {
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
        "n": int(x.size),
    }


def _apply_unified_intent_stress(env) -> dict:
    """I' = S_space(S_time(I, s_t), s_x) on MultiMotionLoader, before Mapper-B.

    Time: warped[t] = interpolate(original, s_t*t); hold last pose. SLERP quats.
    Space: p_T' = p_T0 + s_x Δp_T; r_j' = r_j0 + s_x Δr_j; p_W' = p_T' + r_j'.
    Same formula for every clip. No task id. Ankles untouched. Orientation not spatially amplified.
    """
    speed = float(args_cli.intent_speed)
    sx = float(args_cli.intent_spatial)
    info = {
        "applied": False,
        "api": "motion_dir_loader body_pos_w/quat resampled; Mapper-B gather unchanged",
        "before_mapper_b": True,
        "s_t": speed,
        "s_x": sx,
        "lambda_id": float(args_cli.lambda_id),
        "formula": "S_space(S_time(I,s_t),s_x)",
        "position": "linear time + torso/wrist excursion scale",
        "orientation": "quaternion_slerp_wxyz; no spatial amplification",
        "clip_end": "hold_final_valid_human_intent",
        "no_task_id": True,
        "no_wrap": True,
        "n_clips": 0,
        "median_hold_frac": None,
        "median_active_frac": None,
        "kinematics": {},
        "example": {},
        "out_of_plausible_range": False,
        "kinematic_impossible": False,
        "plausible_flags": {},
        "wrist_torso": {},
    }
    cmd = env.unwrapped.command_manager.get_term("motion")
    loader = getattr(cmd, "motion_dir_loader", None)
    if loader is None or not hasattr(loader, "body_pos_w"):
        raise RuntimeError("intent source missing: command.motion_dir_loader.body_pos_w")
    if getattr(loader, "_t13_stressed", False):
        return getattr(loader, "_t13_warp_info", info)
    body_names = list(cmd.cfg.body_names)
    if "torso_link" not in body_names:
        raise RuntimeError(f"no torso_link in {body_names}")
    idx = {
        "torso": body_names.index("torso_link"),
        "lw": body_names.index("left_wrist_yaw_link") if "left_wrist_yaw_link" in body_names else None,
        "rw": body_names.index("right_wrist_yaw_link") if "right_wrist_yaw_link" in body_names else None,
    }
    attrs_pos = ("body_pos_w", "root_pos_w")
    attrs_quat = ("body_quat_w", "root_quat_w")
    attrs_vel = ("body_lin_vel_w", "body_ang_vel_w", "root_lin_vel_w", "root_ang_vel_w", "joint_vel")
    attrs_lin = ("joint_pos",)
    src = {}
    for name in (*attrs_pos, *attrs_quat, *attrs_vel, *attrs_lin):
        if hasattr(loader, name):
            src[name] = getattr(loader, name).detach().clone()
    n_clips = int(loader.motion_lengths.numel())
    hold_fracs = []
    hold_starts = []
    path_lens = []
    wt_all = []
    example = {}
    pool = {k: {"vel": [], "acc": [], "ang": []} for k in ("torso", "lw", "rw")}
    device = src["body_pos_w"].device
    hold_start_t = torch.zeros(n_clips, dtype=torch.long, device=device)
    for ci in range(n_clips):
        off = int(loader.motion_offsets[ci].item())
        tlen = int(loader.motion_lengths[ci].item())
        if tlen <= 1:
            continue
        fps = float(loader.motion_fps[ci].item()) if hasattr(loader, "motion_fps") else 50.0
        t = torch.arange(tlen, device=device, dtype=torch.float32)
        tau = (speed * t).clamp(max=float(tlen - 1))
        held = tau >= (float(tlen - 1) - 1e-6)
        hold_fracs.append(float(held.float().mean().item()))
        if bool(held.any()):
            hs = int(held.nonzero(as_tuple=False)[0].item())
        else:
            hs = tlen
        hold_starts.append(hs)
        hold_start_t[ci] = hs
        i0 = tau.floor().long()
        i1 = (i0 + 1).clamp(max=tlen - 1)
        w = (tau - i0.float())
        sl = slice(off, off + tlen)

        def _take(attr, extra_dims):
            ten = src[attr][sl]
            wview = w.view(tlen, *([1] * extra_dims))
            out = (1.0 - wview) * ten[i0] + wview * ten[i1]
            if held.any() and extra_dims >= 0:
                last = ten[-1:].expand_as(out)
                hview = held.view(tlen, *([1] * extra_dims))
                out = torch.where(hview, last, out)
            return out

        new_pos = _take("body_pos_w", 2)
        if abs(sx - 1.0) > 1e-9:
            orig = new_pos.clone()
            ti = idx["torso"]
            p_t0 = orig[0, ti]
            p_t = orig[:, ti]
            new_pos[:, ti] = p_t0 + sx * (p_t - p_t0)
            for wkey in ("lw", "rw"):
                wi = idx[wkey]
                if wi is None:
                    continue
                r0 = orig[0, wi] - p_t0
                rel = orig[:, wi] - p_t
                new_pos[:, wi] = new_pos[:, ti] + r0 + sx * (rel - r0)
        new_quat = _quat_slerp_batch(src["body_quat_w"][sl][i0], src["body_quat_w"][sl][i1], w.view(tlen, 1, 1))
        if held.any():
            new_quat = torch.where(held.view(tlen, 1, 1), src["body_quat_w"][sl][-1:].expand_as(new_quat), new_quat)
        loader.body_pos_w[sl] = new_pos
        loader.body_quat_w[sl] = new_quat
        if "joint_pos" in src:
            loader.joint_pos[sl] = _take("joint_pos", 1)
        if "root_pos_w" in src:
            rp = src["root_pos_w"][sl]
            w1 = w.view(tlen, 1)
            nrp = (1.0 - w1) * rp[i0] + w1 * rp[i1]
            nrp = torch.where(held.view(tlen, 1), rp[-1:].expand_as(nrp), nrp)
            loader.root_pos_w[sl] = nrp
        if "root_quat_w" in src:
            rq = _quat_slerp_batch(src["root_quat_w"][sl][i0], src["root_quat_w"][sl][i1], w.view(tlen, 1))
            rq = torch.where(held.view(tlen, 1), src["root_quat_w"][sl][-1:].expand_as(rq), rq)
            loader.root_quat_w[sl] = rq
        for vname in attrs_vel:
            if vname not in src:
                continue
            extra = src[vname][sl].ndim - 1
            nv = speed * _take(vname, extra)
            if vname == "body_lin_vel_w" and abs(sx - 1.0) > 1e-9:
                for bi in (idx["torso"], idx["lw"], idx["rw"]):
                    if bi is not None:
                        nv[:, bi] = nv[:, bi] * sx
            if held.any():
                nv = torch.where(held.view(tlen, *([1] * extra)), torch.zeros_like(nv), nv)
            getattr(loader, vname)[sl] = nv

        ti = idx["torso"]
        path_lens.append(float(torch.linalg.norm(new_pos[1:, ti] - new_pos[:-1, ti], dim=-1).sum().item()))
        for wkey, wi in (("lw", idx["lw"]), ("rw", idx["rw"])):
            if wi is None:
                continue
            wt_all.append(torch.linalg.norm(new_pos[:, wi] - new_pos[:, ti], dim=-1).detach().cpu().numpy())
        for key, bi in (("torso", idx["torso"]), ("lw", idx["lw"]), ("rw", idx["rw"])):
            if bi is None:
                continue
            sv, sa = _kin_arrays(new_pos[:, bi], fps)
            pool[key]["vel"].append(sv)
            pool[key]["acc"].append(sa)
            pool[key]["ang"].append(np.asarray([_kin_ang(new_quat[:, bi], fps)["p50"]], dtype=np.float64))
        if ci == 0:
            step = max(1, tlen // 80)
            example = {
                "clip": Path(loader.motion_paths[ci]).name if hasattr(loader, "motion_paths") else f"clip{ci}",
                "T": tlen,
                "fps": fps,
                "s_t": speed,
                "s_x": sx,
                "t": t[::step].cpu().tolist(),
                "tau": tau[::step].cpu().tolist(),
                "torso_xyz": new_pos[::step, ti].cpu().tolist(),
                "torso_xyz_orig": src["body_pos_w"][sl][::step, ti].cpu().tolist(),
            }
            if idx["rw"] is not None:
                example["rw_xyz"] = new_pos[::step, idx["rw"]].cpu().tolist()
            if idx["lw"] is not None:
                example["lw_xyz"] = new_pos[::step, idx["lw"]].cpu().tolist()

    kin = {"by_clip_hold_frac": hold_fracs, "by_clip_hold_start": hold_starts}

    def _pool_kin(key: str, out_name: str) -> None:
        if not pool[key]["vel"]:
            return
        kin[out_name] = {
            "vel": _pct4(np.concatenate(pool[key]["vel"]) if pool[key]["vel"] else []),
            "acc": _pct4(np.concatenate(pool[key]["acc"]) if pool[key]["acc"] else []),
            "ang_vel": _pct4(np.concatenate(pool[key]["ang"]) if pool[key]["ang"] else []),
        }

    _pool_kin("torso", "torso")
    _pool_kin("lw", "left_wrist")
    _pool_kin("rw", "right_wrist")
    wrist_v99 = max(
        (kin.get("left_wrist") or {}).get("vel", {}).get("p99") or 0.0,
        (kin.get("right_wrist") or {}).get("vel", {}).get("p99") or 0.0,
    )
    wrist_a99 = max(
        (kin.get("left_wrist") or {}).get("acc", {}).get("p99") or 0.0,
        (kin.get("right_wrist") or {}).get("acc", {}).get("p99") or 0.0,
    )
    wt_cat = np.concatenate(wt_all) if wt_all else np.asarray([], dtype=np.float64)
    wt = _pct4(wt_cat)
    flags = {
        "wrist_vel_p99_mps": float(wrist_v99),
        "wrist_acc_p99_mps2": float(wrist_a99),
        "vel_threshold_mps": 6.0,
        "acc_threshold_mps2": 50.0,
        "impossible_wrist_torso_m": 0.85,
    }
    out_of = bool(wrist_v99 > 6.0 or wrist_a99 > 50.0)
    impossible = bool(np.isfinite(wt.get("max", np.nan)) and wt["max"] > 0.85)
    torso_travel = float(np.median(path_lens)) if path_lens else 0.0
    med_hold = float(np.median(hold_fracs)) if hold_fracs else None
    info.update({
        "applied": True,
        "n_clips": n_clips,
        "median_hold_frac": med_hold,
        "median_active_frac": (1.0 - med_hold) if med_hold is not None else None,
        "kinematics": kin,
        "example": example,
        "out_of_plausible_range": out_of,
        "kinematic_impossible": impossible,
        "plausible_flags": flags,
        "wrist_torso": wt,
        "torso_path_length_m": torso_travel,
        "temporal_stress_informative": bool(torso_travel > 0.15 or wrist_v99 > 0.20),
    })
    loader._t13_stressed = True
    loader._t13_warp_info = info
    loader._t13_hold_start = hold_start_t
    print(
        f"[t13] unified s_t={speed:.3f} s_x={sx:.3f} clips={n_clips} hold={med_hold} "
        f"wrist_v99={wrist_v99:.3f} wrist_a99={wrist_a99:.3f} wt_max={wt.get('max')} "
        f"oor={out_of} impossible={impossible}",
        flush=True,
    )
    return info
def _mapper_body_errors(env, injector, vis_flags) -> dict[str, torch.Tensor] | None:
    """Offline GT future vs Mapper-B. Never fed to the policy."""
    pred = injector.last_pred_abs
    if pred is None:
        return None
    gt = gather_ref_abs_anchor(env, FUTURE_OFFSETS)[:, :, :N_VISIBLE, :]
    err = torch.linalg.norm(pred - gt, dim=-1)  # [N, 7, 3]
    vis = [i for i, f in enumerate(vis_flags) if f] or [0]
    out: dict[str, torch.Tensor] = {
        "torso": err[:, :, 0],
        "lw": err[:, :, 1],
        "rw": err[:, :, 2],
        "agg": err[:, :, vis].mean(dim=-1),
    }
    for h in (0.1, 0.2, 0.4):
        p, g = _lerp_horizon(pred, gt, h)
        e = torch.linalg.norm(p - g, dim=-1)
        out[f"h{h}"] = e
        out[f"h{h}_agg"] = e[:, vis].mean(dim=-1)
        out[f"h{h}_torso"] = e[:, 0]
        out[f"h{h}_lw"] = e[:, 1]
        out[f"h{h}_rw"] = e[:, 2]
    return out


def _apply_push(asset, n_use: int, dv: float, axis: str) -> dict:
    """Robot-frame lateral/fwd Δv via write_root_velocity_to_sim (not force/impulse)."""
    q = asset.data.root_quat_w[:n_use]
    vel_before = asset.data.root_lin_vel_w[:n_use].clone()
    if abs(dv) < 1e-9:
        return {
            "applied": False,
            "command_dv": 0.0,
            "axis": axis,
            "api": "write_root_velocity_to_sim",
            "unit": "m/s",
            "frame": "robot_local_then_world",
            "body": "articulation_root",
            "duration_steps": 1,
            "measured_dv_mean": 0.0,
        }
    local = torch.zeros((n_use, 3), device=q.device, dtype=q.dtype)
    if axis == "fwd":
        local[:, 0] = float(dv)
    else:
        local[:, 1] = float(dv)
    delta = quat_apply(q, local)
    vel = asset.data.root_vel_w.clone()
    vel[:n_use, :3] = vel[:n_use, :3] + delta
    ids = torch.arange(n_use, device=asset.device, dtype=torch.long)
    asset.write_root_velocity_to_sim(vel[:n_use], env_ids=ids)
    vel_after = asset.data.root_lin_vel_w[:n_use]
    measured = (vel_after - vel_before).norm(dim=-1)
    info = {
        "applied": True,
        "command_dv": float(dv),
        "axis": axis,
        "api": "write_root_velocity_to_sim",
        "unit": "m/s",
        "frame": "robot_local_then_world",
        "body": "articulation_root",
        "duration_steps": 1,
        "measured_dv_mean": float(measured.mean().item()),
        "measured_dv_p50": float(measured.median().item()),
    }
    print(f"[t11] push {info}", flush=True)
    return info


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
                print(f"[t1] pin mask {mask_name!r} idx={names.index(mask_name)}", flush=True)
        cmd._resample_command(ids)
        if hasattr(cmd, "resample_all_mask_modes"):
            cmd.resample_all_mask_modes()
        try:
            env.unwrapped.sim.forward()
        except Exception as exc:
            print(f"[t1] sim.forward skipped: {exc}", flush=True)
    snap = getattr(env.unwrapped, "_t1_phys_snap", None)
    if snap is not None:
        env.unwrapped._t1_phys_snap = _apply_payload_com(env, snap)
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


def _poi_rates(xs: list[float]) -> dict:
    """Visible-POI tracking rates. Secondary only; never replaces Task SR."""
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "sr_5cm": float("nan"), "sr_2cm": float("nan"), "n": 0}
    return {
        "mean": float(a.mean()),
        "sr_5cm": float((a < 0.05).mean()),
        "sr_2cm": float((a < 0.02).mean()),
        "n": int(a.size),
    }


def _nanmean_key(rows: list[dict], getter) -> float:
    vals = []
    for r in rows:
        try:
            v = getter(r)
        except (TypeError, ValueError, KeyError):
            v = float("nan")
        if v is not None and np.isfinite(float(v)):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


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


UCR2P_D = 48
_UCR2P_ANKLE = ("left_ankle_roll_link", "right_ankle_roll_link")


def _ucr2p_frame(
    asset,
    am,
    origin,
    i: int,
    e_pre,
    e_torso,
    e_lw,
    e_rw,
    rec_pack,
    gout,
    z_nom,
    cf,
    ankle_ids: list[int],
) -> np.ndarray:
    """Causal compact process vector. No future GT."""
    v = np.zeros(UCR2P_D, dtype=np.float32)
    v[0] = float(e_pre)
    v[1] = float(e_torso)
    v[2] = float(e_lw)
    v[3] = float(e_rw)
    v[4] = float(gout["R_E"][i].item())
    v[5] = float(gout["R_S"][i].item())
    rp = rec_pack[i]
    v[6] = float(rp[9:18].norm().item())
    v[7] = float(rp[18:21].mean().item())
    root = asset.data.root_state_w[i]
    v[8] = float(root[0].item() - origin[0].item())
    v[9] = float(root[1].item() - origin[1].item())
    v[10] = float(root[2].item())
    v[11:15] = root[3:7].detach().cpu().numpy().astype(np.float32)
    v[15:18] = asset.data.root_lin_vel_w[i].detach().cpu().numpy().astype(np.float32)
    v[18:21] = asset.data.root_ang_vel_w[i].detach().cpu().numpy().astype(np.float32)
    jp = asset.data.joint_pos[i]
    jv = asset.data.joint_vel[i]
    act = am._action[i]
    v[21] = float(jp.mean().item())
    v[22] = float(jv.norm().item())
    v[23] = float(act.norm().item())
    contact = [0.0, 0.0]
    if cf is not None and ankle_ids and hasattr(cf.data, "net_forces_w"):
        nf = cf.data.net_forces_w[i]
        for k, bid in enumerate(ankle_ids[:2]):
            contact[k] = 1.0 if float(nf[bid].norm().item()) > 10.0 else 0.0
    v[24] = contact[0]
    v[25] = contact[1]
    v[26] = float(sum(contact))
    zn = z_nom[i].detach().cpu().numpy().astype(np.float32).reshape(-1)
    n_z = min(16, int(zn.size))
    v[27 : 27 + n_z] = zn[:n_z]
    return v


def _write_ucr2p_process(path: Path, snaps: list[dict]) -> None:
    n = len(snaps)
    h = int(getattr(args_cli, "ucr2p_hist", UCR2P_HIST) or UCR2P_HIST)
    proc = np.zeros((n, h, UCR2P_D), dtype=np.float32)
    rec = []
    z_nom = []
    clip, seed, t, task, lam, re, e0 = [], [], [], [], [], [], []
    for i, s in enumerate(snaps):
        p = s.get("process")
        if p is None:
            continue
        arr = np.asarray(p, dtype=np.float32)
        if arr.ndim == 2:
            hh = min(h, arr.shape[0])
            dd = min(UCR2P_D, arr.shape[1])
            proc[i, -hh:, :dd] = arr[-hh:, :dd]
        rec_obs = s.get("rec_obs")
        if rec_obs is not None:
            rec.append(np.asarray(rec_obs.detach().cpu() if hasattr(rec_obs, "detach") else rec_obs, dtype=np.float32))
        zn = s.get("z_nom")
        if zn is not None:
            z_nom.append(np.asarray(zn.detach().cpu() if hasattr(zn, "detach") else zn, dtype=np.float32))
        clip.append(str(s.get("clip", "")))
        seed.append(int(s.get("seed", 0)))
        t.append(int(s.get("t", -1)))
        task.append(str(s.get("task", args_cli.task_source)))
        lam.append(float(s.get("lam", args_cli.lambda_id)))
        re.append(float(s.get("re", np.nan)))
        e0.append(float(s.get("e0", np.nan)))
    rec_a = np.stack(rec, 0) if rec else np.zeros((0, 1), dtype=np.float32)
    zn_a = np.stack(z_nom, 0) if z_nom else np.zeros((0, 1), dtype=np.float32)
    np.savez_compressed(
        path,
        process=proc,
        rec_obs=rec_a,
        z_nom=zn_a,
        clip=np.asarray(clip),
        seed=np.asarray(seed, dtype=np.int32),
        t=np.asarray(t, dtype=np.int32),
        task=np.asarray(task),
        lam=np.asarray(lam, dtype=np.float32),
        re=np.asarray(re, dtype=np.float32),
        e0=np.asarray(e0, dtype=np.float32),
        hist=int(h),
        dim=int(UCR2P_D),
        dt=float(DT),
        causal=True,
        no_future_gt=True,
        n=int(n),
    )
    print(f"[ucr2p] wrote {path} n={n} hist={h} d={UCR2P_D}", flush=True)


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
    raw = env.unwrapped
    raw._t1_phys_snap = _apply_payload_com(env, None)
    raw._t13_warp_info = _apply_unified_intent_stress(env)
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


@torch.inference_mode()
def _rollout_probe_jr(
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
    pelvis_i,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parent or R-M3-burst clone. Returns (e_H, fail_any) for n_valid."""
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
    e_last = torch.zeros(n_envs, device=device, dtype=torch.float32)
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
        e_last = _visible_e(cmd, vis_i, vis_flags)
        failed = failed | _official_fail(cmd, asset, vis_i, n_envs, pelvis_i, gravity)
    return e_last[:n_valid].detach(), failed[:n_valid].detach()


def _ura0_feat(rec_pack_i, z_i, d_i, theta_deg, root_v, a_z, acc, w, roll, pitch, cf) -> np.ndarray:
    rp = rec_pack_i.detach().cpu().numpy().astype(np.float32)
    parts = [
        rp[0:9],
        rp[9:18],
        rp[18:21],
        root_v.detach().cpu().numpy().astype(np.float32),
        np.asarray([float(a_z), float(acc)], dtype=np.float32),
        w.detach().cpu().numpy().astype(np.float32),
        np.asarray([float(roll), float(pitch), float(cf)], dtype=np.float32),
        z_i.detach().cpu().numpy().astype(np.float32),
        d_i.detach().cpu().numpy().astype(np.float32),
        np.asarray([float(theta_deg) / 10.0], dtype=np.float32),
    ]
    return np.concatenate(parts, axis=0)


@torch.inference_mode()
def _ura0_clone_pairs(
    env,
    policy,
    runner,
    injector,
    snaps: list[dict],
    vis_i,
    vis_flags,
    adapter,
    mua,
    sda,
    bins: tuple[float, ...],
    pelvis_i,
    lam_fail: float,
    out_npz: Path,
) -> dict:
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    rows = []
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[ura0] clone batch {start}:{start + n_valid}/{len(snaps)}", flush=True)
        e_p, f_p = _rollout_probe_jr(
            env, policy, runner, injector, batch, n_valid,
            d_raw=None, theta_deg=0.0, burst=0, horizon=HORIZON_PAIR,
            vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i,
        )
        rec = torch.stack([s["rec_obs"] for s in batch], dim=0).to(device=device, dtype=torch.float32)
        d_pad = torch.zeros(n_envs, 16, device=device, dtype=torch.float32)
        th_pad = torch.full((n_envs,), 5.0, device=device, dtype=torch.float32)
        raw_a, logits = adapter((rec - mua) / sda)
        pred = logits.argmax(dim=-1)
        th_a = torch.tensor([float(bins[int(j)]) for j in pred.tolist()], device=device, dtype=torch.float32)
        d_pad[:n_valid] = raw_a
        th_pad[:n_valid] = th_a
        e_r, f_r = _rollout_probe_jr(
            env, policy, runner, injector, batch, n_valid,
            d_raw=d_pad, theta_deg=th_pad, burst=BURST_STEPS, horizon=HORIZON_PAIR,
            vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i,
        )
        for i in range(n_valid):
            ep = float(e_p[i].item())
            er = float(e_r[i].item())
            fp = bool(f_p[i].item())
            fr = bool(f_r[i].item())
            jp = ep + float(lam_fail) * float(fp)
            jr = er + float(lam_fail) * float(fr)
            s = batch[i]
            rows.append(
                {
                    "task": s["task"],
                    "lam": float(s["lam"]),
                    "clip": s["clip"],
                    "seed": int(s["seed"]),
                    "t": int(s["t"]),
                    "re": float(s["re"]),
                    "rs": float(s["rs"]),
                    "e0": float(s["e0"]),
                    "e_p": ep,
                    "e_r": er,
                    "fail_p": int(fp),
                    "fail_r": int(fr),
                    "j_p": jp,
                    "j_r": jr,
                    "i_rec": jr - jp,
                    "theta": float(th_a[i].item()),
                    "sr_ep": int(s.get("sr_ep", 1)),
                    "s_enabled": int(s.get("s_enabled", 0)),
                    "feat": s["feat"],
                    "hist": s["hist_feat"],
                }
            )
    if not rows:
        raise RuntimeError("URA-0: no clone rows")
    feat = np.stack([r["feat"] for r in rows], 0)
    hist = np.stack([r["hist"] for r in rows], 0)
    meta = {
        "n": len(rows),
        "lam_fail": float(lam_fail),
        "eps_i_rec": EPS_I_REC,
        "horizon_s": HORIZON_PAIR * DT,
        "burst_s": BURST_STEPS * DT,
        "feat_dim": int(feat.shape[1]),
        "hist_shape": list(hist.shape[1:]),
        "no_task_id_in_feat": True,
        "recovery": "frozen R-M3 shared direction + adaptive magnitude, 100ms then Parent",
        "sample": "uniform Parent states, not RE/RS triggers",
        "J": "masked_e_H + lam_fail * official_fail",
    }
    np.savez_compressed(
        out_npz,
        feat=feat.astype(np.float32),
        hist=hist.astype(np.float32),
        j_p=np.asarray([r["j_p"] for r in rows], dtype=np.float32),
        j_r=np.asarray([r["j_r"] for r in rows], dtype=np.float32),
        i_rec=np.asarray([r["i_rec"] for r in rows], dtype=np.float32),
        e_p=np.asarray([r["e_p"] for r in rows], dtype=np.float32),
        e_r=np.asarray([r["e_r"] for r in rows], dtype=np.float32),
        fail_p=np.asarray([r["fail_p"] for r in rows], dtype=np.int8),
        fail_r=np.asarray([r["fail_r"] for r in rows], dtype=np.int8),
        re=np.asarray([r["re"] for r in rows], dtype=np.float32),
        rs=np.asarray([r["rs"] for r in rows], dtype=np.float32),
        e0=np.asarray([r["e0"] for r in rows], dtype=np.float32),
        theta=np.asarray([r["theta"] for r in rows], dtype=np.float32),
        t=np.asarray([r["t"] for r in rows], dtype=np.int32),
        seed=np.asarray([r["seed"] for r in rows], dtype=np.int32),
        lam=np.asarray([r["lam"] for r in rows], dtype=np.float32),
        sr_ep=np.asarray([r["sr_ep"] for r in rows], dtype=np.int8),
        s_enabled=np.asarray([r["s_enabled"] for r in rows], dtype=np.int8),
        task=np.asarray([r["task"] for r in rows]),
        clip=np.asarray([r["clip"] for r in rows]),
        meta=np.array(json.dumps(meta)),
    )
    print(f"[ura0] wrote {out_npz} n={len(rows)}", flush=True)
    return meta


def _j_from_ef(e: torch.Tensor, fail: torch.Tensor, lam_fail: float) -> torch.Tensor:
    return e + float(lam_fail) * fail.to(dtype=e.dtype)


def _write_ucr0_empty(out_npz: Path, meta: dict) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(
        j_p=np.zeros(0, dtype=np.float32),
        j_r=np.zeros(0, dtype=np.float32),
        j_ora=np.zeros(0, dtype=np.float32),
        i_rm3=np.zeros(0, dtype=np.float32),
        i_ora=np.zeros(0, dtype=np.float32),
        e_p=np.zeros(0, dtype=np.float32),
        e_r=np.zeros(0, dtype=np.float32),
        e_ora=np.zeros(0, dtype=np.float32),
        fail_p=np.zeros(0, dtype=np.int8),
        fail_r=np.zeros(0, dtype=np.int8),
        fail_ora=np.zeros(0, dtype=np.int8),
        re=np.zeros(0, dtype=np.float32),
        rs=np.zeros(0, dtype=np.float32),
        e0=np.zeros(0, dtype=np.float32),
        theta=np.zeros(0, dtype=np.float32),
        theta_ora=np.zeros(0, dtype=np.float32),
        t=np.zeros(0, dtype=np.int32),
        seed=np.zeros(0, dtype=np.int32),
        lam=np.zeros(0, dtype=np.float32),
        sr_ep=np.zeros(0, dtype=np.int8),
        task=np.asarray([], dtype=object),
        clip=np.asarray([], dtype=object),
        rec_obs=np.zeros((0, 487), dtype=np.float32),
        z_nom=np.zeros((0, 16), dtype=np.float32),
        d_star=np.zeros((0, 16), dtype=np.float32),
        j_by_angle=np.zeros((0, len(UCR0_SWEEP_DEG)), dtype=np.float32),
        sweep_deg=np.asarray(UCR0_SWEEP_DEG, dtype=np.float32),
        meta=np.array(json.dumps(meta)),
    )
    np.savez_compressed(out_npz, **kw)
    if bool(getattr(args_cli, "ucr1_dump", False)):
        torch.save([], out_npz.with_name("snaps.pt"))


@torch.inference_mode()
def _ucr0_clone_pairs(
    env,
    policy,
    runner,
    injector,
    snaps: list[dict],
    vis_i,
    vis_flags,
    adapter,
    mua,
    sda,
    bins: tuple[float, ...],
    pelvis_i,
    lam_fail: float,
    out_npz: Path,
) -> dict:
    """RE-triggered clones: Parent vs frozen R-M3 vs local tangent oracle. No closed-loop."""
    meta = {
        "n": len(snaps),
        "lam_fail": float(lam_fail),
        "re_on": float(getattr(args_cli, "re_on", RE_ON_DEFAULT)),
        "persist": int(getattr(args_cli, "ucr0_persist", 3)),
        "horizon_s": HORIZON_PAIR * DT,
        "burst_s": BURST_STEPS * DT,
        "trigger": "R_E >= re_on persist N; RS not used",
        "no_task_id": True,
        "recovery": "frozen R-M3 adaptive 100ms then Parent",
        "oracle": "15-D tangent ±1° FD + magnitude sweep on d*",
        "J": "masked_e_H + lam_fail * official_fail",
    }
    if not snaps:
        _write_ucr0_empty(out_npz, meta)
        print(f"[ucr0] wrote empty {out_npz}", flush=True)
        return meta
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    tan_eps = math.tan(math.radians(UCR0_EPS_DEG))
    rows = []
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[ucr0] clone batch {start}:{start + n_valid}/{len(snaps)}", flush=True)
        e_p, f_p = _rollout_probe_jr(
            env, policy, runner, injector, batch, n_valid,
            d_raw=None, theta_deg=0.0, burst=0, horizon=HORIZON_PAIR,
            vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i,
        )
        j_p = _j_from_ef(e_p, f_p, lam_fail)
        rec = torch.stack([s["rec_obs"] for s in batch], dim=0).to(device=device, dtype=torch.float32)
        z0 = torch.stack([s["z_nom"] for s in batch], dim=0).to(device=device, dtype=torch.float32)
        d_pad = torch.zeros(n_envs, 16, device=device, dtype=torch.float32)
        th_pad = torch.full((n_envs,), 5.0, device=device, dtype=torch.float32)
        raw_a, logits = adapter((rec - mua) / sda)
        pred = logits.argmax(dim=-1)
        th_a = torch.tensor([float(bins[int(j)]) for j in pred.tolist()], device=device, dtype=torch.float32)
        d_pad[:n_valid] = raw_a
        th_pad[:n_valid] = th_a
        e_r, f_r = _rollout_probe_jr(
            env, policy, runner, injector, batch, n_valid,
            d_raw=d_pad, theta_deg=th_pad, burst=BURST_STEPS, horizon=HORIZON_PAIR,
            vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i,
        )
        j_r = _j_from_ef(e_r, f_r, lam_fail)

        b = tangent_basis(z0)
        j_plus, j_minus = [], []
        for k in range(15):
            d = torch.zeros(n_envs, 16, device=device, dtype=torch.float32)
            d[:n_valid] = b[:n_valid, :, k]
            jp_e, jp_f = _rollout_probe_jr(
                env, policy, runner, injector, batch, n_valid,
                d_raw=d, theta_deg=UCR0_EPS_DEG, burst=BURST_STEPS, horizon=HORIZON_PAIR,
                vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i,
            )
            jm_e, jm_f = _rollout_probe_jr(
                env, policy, runner, injector, batch, n_valid,
                d_raw=-d, theta_deg=UCR0_EPS_DEG, burst=BURST_STEPS, horizon=HORIZON_PAIR,
                vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i,
            )
            j_plus.append(_j_from_ef(jp_e, jp_f, lam_fail))
            j_minus.append(_j_from_ef(jm_e, jm_f, lam_fail))
            print(f"[ucr0]   fd {k + 1}/15", flush=True)
        jp_t = torch.stack(j_plus, dim=-1)
        jm_t = torch.stack(j_minus, dim=-1)
        g_i = (jp_t - jm_t) / (2.0 * tan_eps)
        g_t = torch.einsum("ni,ndi->nd", g_i, b[:n_valid])
        d_star = project_tangent(-g_t, z0[:n_valid])
        d_star_pad = torch.zeros(n_envs, 16, device=device, dtype=torch.float32)
        d_star_pad[:n_valid] = d_star
        i_fd = torch.cat([jp_t - j_p.unsqueeze(-1), jm_t - j_p.unsqueeze(-1)], dim=-1).min(dim=-1).values
        j_ora = j_p + i_fd
        e_ora = e_p.clone()
        f_ora = f_p.clone()
        th_ora = torch.full((n_valid,), UCR0_EPS_DEG, device=device, dtype=torch.float32)
        j_sweep = torch.zeros(n_valid, len(UCR0_SWEEP_DEG), device=device, dtype=torch.float32)
        for ti, th in enumerate(UCR0_SWEEP_DEG):
            js_e, js_f = _rollout_probe_jr(
                env, policy, runner, injector, batch, n_valid,
                d_raw=d_star_pad, theta_deg=th, burst=BURST_STEPS, horizon=HORIZON_PAIR,
                vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i,
            )
            js = _j_from_ef(js_e, js_f, lam_fail)
            j_sweep[:, ti] = js
            better = js < j_ora
            j_ora = torch.where(better, js, j_ora)
            e_ora = torch.where(better, js_e, e_ora)
            f_ora = torch.where(better, js_f, f_ora)
            th_ora = torch.where(better, torch.full_like(th_ora, float(th)), th_ora)
            print(f"[ucr0]   sweep {th}°", flush=True)

        rec_np = rec[:n_valid].detach().cpu().numpy().astype(np.float32)
        z_np = z0[:n_valid].detach().cpu().numpy().astype(np.float32)
        d_np = d_star[:n_valid].detach().cpu().numpy().astype(np.float32)
        jsw_np = j_sweep.detach().cpu().numpy().astype(np.float32)
        for i in range(n_valid):
            s = batch[i]
            jp = float(j_p[i].item())
            jr = float(j_r[i].item())
            jo = float(j_ora[i].item())
            rows.append(
                {
                    "task": s["task"],
                    "lam": float(s["lam"]),
                    "clip": s["clip"],
                    "seed": int(s["seed"]),
                    "t": int(s["t"]),
                    "re": float(s["re"]),
                    "rs": float(s["rs"]),
                    "e0": float(s["e0"]),
                    "e_p": float(e_p[i].item()),
                    "e_r": float(e_r[i].item()),
                    "e_ora": float(e_ora[i].item()),
                    "fail_p": int(bool(f_p[i].item())),
                    "fail_r": int(bool(f_r[i].item())),
                    "fail_ora": int(bool(f_ora[i].item())),
                    "j_p": jp,
                    "j_r": jr,
                    "j_ora": jo,
                    "i_rm3": jr - jp,
                    "i_ora": jo - jp,
                    "theta": float(th_a[i].item()),
                    "theta_ora": float(th_ora[i].item()),
                    "sr_ep": int(s.get("sr_ep", 1)),
                    "rec_obs": rec_np[i],
                    "z_nom": z_np[i],
                    "d_star": d_np[i],
                    "j_by_angle": jsw_np[i],
                    "snap": s,
                }
            )
    meta["n"] = len(rows)
    dump = bool(getattr(args_cli, "ucr1_dump", False))
    kw = dict(
        j_p=np.asarray([r["j_p"] for r in rows], dtype=np.float32),
        j_r=np.asarray([r["j_r"] for r in rows], dtype=np.float32),
        j_ora=np.asarray([r["j_ora"] for r in rows], dtype=np.float32),
        i_rm3=np.asarray([r["i_rm3"] for r in rows], dtype=np.float32),
        i_ora=np.asarray([r["i_ora"] for r in rows], dtype=np.float32),
        e_p=np.asarray([r["e_p"] for r in rows], dtype=np.float32),
        e_r=np.asarray([r["e_r"] for r in rows], dtype=np.float32),
        e_ora=np.asarray([r["e_ora"] for r in rows], dtype=np.float32),
        fail_p=np.asarray([r["fail_p"] for r in rows], dtype=np.int8),
        fail_r=np.asarray([r["fail_r"] for r in rows], dtype=np.int8),
        fail_ora=np.asarray([r["fail_ora"] for r in rows], dtype=np.int8),
        re=np.asarray([r["re"] for r in rows], dtype=np.float32),
        rs=np.asarray([r["rs"] for r in rows], dtype=np.float32),
        e0=np.asarray([r["e0"] for r in rows], dtype=np.float32),
        theta=np.asarray([r["theta"] for r in rows], dtype=np.float32),
        theta_ora=np.asarray([r["theta_ora"] for r in rows], dtype=np.float32),
        t=np.asarray([r["t"] for r in rows], dtype=np.int32),
        seed=np.asarray([r["seed"] for r in rows], dtype=np.int32),
        lam=np.asarray([r["lam"] for r in rows], dtype=np.float32),
        sr_ep=np.asarray([r["sr_ep"] for r in rows], dtype=np.int8),
        task=np.asarray([r["task"] for r in rows]),
        clip=np.asarray([r["clip"] for r in rows]),
        meta=np.array(json.dumps(meta)),
    )
    if dump and rows:
        kw["rec_obs"] = np.stack([r["rec_obs"] for r in rows], axis=0).astype(np.float32)
        kw["z_nom"] = np.stack([r["z_nom"] for r in rows], axis=0).astype(np.float32)
        kw["d_star"] = np.stack([r["d_star"] for r in rows], axis=0).astype(np.float32)
        kw["j_by_angle"] = np.stack([r["j_by_angle"] for r in rows], axis=0).astype(np.float32)
        kw["sweep_deg"] = np.asarray(UCR0_SWEEP_DEG, dtype=np.float32)
        meta["ucr1_dump"] = True
        meta["sweep_deg"] = list(UCR0_SWEEP_DEG)
        kw["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(out_npz, **kw)
    if dump:
        snaps_path = out_npz.with_name("snaps.pt")
        torch.save([r.get("snap") for r in rows], snaps_path)
        print(f"[ucr1] wrote {snaps_path} n={len(rows)}", flush=True)
    print(f"[ucr0] wrote {out_npz} n={len(rows)}", flush=True)
    return meta


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
        print(f"[t1] paired batch {start}:{start + n_valid}/{len(snaps)} {task}/{terrain}", flush=True)
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
        f"[t1] paired {task}/{terrain} n={len(rows)} "
        f"adapt P(I<0)={summary['all']['adaptive']['P_I_lt_0']} "
        f"med={summary['all']['adaptive']['median_cm']}",
        flush=True,
    )
    return summary


UFR_FEAT_NAMES = (
    "e_vis",
    "e_torso",
    "e_lw",
    "e_rw",
    "e_dot_vis",
    "vis_torso",
    "vis_lw",
    "vis_rw",
    "e9_0", "e9_1", "e9_2", "e9_3", "e9_4", "e9_5", "e9_6", "e9_7", "e9_8",
    "edot9_0", "edot9_1", "edot9_2", "edot9_3", "edot9_4",
    "edot9_5", "edot9_6", "edot9_7", "edot9_8",
    "root_vx",
    "root_vy",
    "root_vz",
    "a_root_z",
    "root_acc",
    "root_wx",
    "root_wy",
    "root_wz",
    "roll",
    "pitch",
    "wrist_cf",
    *[f"z{i}" for i in range(16)],
)


def _arr(xs, t: int, width: int = 1, dtype=np.float32) -> np.ndarray:
    if not xs:
        return np.zeros((t, width), dtype=dtype)
    a = np.asarray(xs, dtype=dtype)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[0] != t:
        out = np.zeros((t, a.shape[1]), dtype=dtype)
        n = min(t, a.shape[0])
        out[:n] = a[:n]
        return out
    return a


def _pack_ufr_seed(seed, use, paths, vis_flags, body_series, phys, ep_rows) -> list[dict]:
    out = []
    vis = np.asarray(vis_flags, dtype=np.float32)
    for i in range(use):
        bs = body_series[(seed, i)]
        ph = phys[(seed, i)]
        row = ep_rows[i] if i < len(ep_rows) else {}
        t = int(row.get("episode_length") or len(bs.get("torso") or []))
        if t <= 0:
            continue
        e_vis = _arr(bs.get("e_vis") or [], t)[:, 0]
        e_dot = np.zeros(t, dtype=np.float32)
        if t > 1:
            e_dot[1:] = (e_vis[1:] - e_vis[:-1]) / DT
        z = _arr(bs.get("z") or [], t, 16)
        e9 = _arr(bs.get("e9") or [], t, 9)
        edot9 = _arr(bs.get("edot9") or [], t, 9)
        vis_t = _arr(bs.get("vis3") or [], t, 3)
        if vis_t.sum() == 0:
            vis_t = np.broadcast_to(vis.reshape(1, 3), (t, 3)).copy()
        feat = np.concatenate(
            [
                e_vis.reshape(t, 1),
                _arr(bs.get("torso") or [], t),
                _arr(bs.get("lw") or [], t),
                _arr(bs.get("rw") or [], t),
                e_dot.reshape(t, 1),
                vis_t,
                e9,
                edot9,
                _arr(bs.get("root_vx") or [], t),
                _arr(bs.get("root_vy") or [], t),
                _arr(bs.get("root_vz") or bs.get("v_root_z") or [], t),
                _arr(bs.get("a_root_z") or [], t),
                _arr(ph.get("root_acc") or [], t),
                _arr(bs.get("root_wx") or [], t),
                _arr(bs.get("root_wy") or [], t),
                _arr(bs.get("root_wz") or [], t),
                _arr(ph.get("roll") or [], t),
                _arr(ph.get("pitch") or [], t),
                _arr(ph.get("wrist_cf") or [], t),
                z,
            ],
            axis=1,
        ).astype(np.float32)
        out.append(
            {
                "feat": feat,
                "re": _arr(bs.get("R_E") or [], t)[:, 0],
                "rs": _arr(bs.get("R_S") or [], t)[:, 0],
                "s_raw": _arr(bs.get("S_raw") or [], t)[:, 0],
                "e_vis": e_vis,
                "clip": str(row.get("clip") or (Path(paths[i]).name if i < len(paths) else f"env{i}")),
                "seed": int(seed),
                "env": int(i),
                "fail": int(row.get("fail") or 0),
                "fail_step": int(row.get("fail_step") if row.get("fail_step") is not None else -1),
                "fail_reason": str(row.get("fail_reason") or "none"),
                "sr_task": int(row.get("sr_task") if row.get("sr_task") is not None else int(not row.get("fail"))),
                "sr_5cm": float(row.get("sr_5cm") or 0.0),
                "vis": [float(vis[0]), float(vis[1]), float(vis[2])],
                "q": _arr(bs.get("q") or [], t, 29) if bs.get("q") else None,
                "dq": _arr(bs.get("dq") or [], t, 29) if bs.get("dq") else None,
                "q_cmd": _arr(bs.get("q_cmd") or [], t, 29) if bs.get("q_cmd") else None,
                "q_pre": _arr(bs.get("q_pre") or [], t, 29) if bs.get("q_pre") else None,
                "action": _arr(bs.get("action") or [], t, 29) if bs.get("action") else None,
                "root_pos": _arr(bs.get("root_pos") or [], t, 3) if bs.get("root_pos") else None,
                "root_quat": _arr(bs.get("root_quat") or [], t, 4) if bs.get("root_quat") else None,
            }
        )
    return out


def _write_ufr_npz(path: Path, episodes: list[dict], meta: dict) -> None:
    if not episodes:
        return
    feats, res, rss, sraw, evis, ts, eids = [], [], [], [], [], [], []
    clips, seeds, envs, fails, fsteps, reasons, srs, sr5s = [], [], [], [], [], [], [], []
    for ei, ep in enumerate(episodes):
        t = int(ep["feat"].shape[0])
        feats.append(ep["feat"])
        res.append(ep["re"])
        rss.append(ep["rs"])
        sraw.append(ep["s_raw"])
        evis.append(ep["e_vis"])
        ts.append(np.arange(t, dtype=np.int32))
        eids.append(np.full(t, ei, dtype=np.int32))
        clips.append(ep["clip"])
        seeds.append(int(ep["seed"]))
        envs.append(int(ep["env"]))
        fails.append(int(ep["fail"]))
        fsteps.append(int(ep["fail_step"]))
        reasons.append(ep["fail_reason"])
        srs.append(int(ep["sr_task"]))
        sr5s.append(float(ep["sr_5cm"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(
        feat=np.concatenate(feats, axis=0),
        re=np.concatenate(res, axis=0),
        rs=np.concatenate(rss, axis=0),
        s_raw=np.concatenate(sraw, axis=0),
        e_vis=np.concatenate(evis, axis=0),
        t=np.concatenate(ts, axis=0),
        ep_idx=np.concatenate(eids, axis=0),
        clip=np.asarray(clips),
        seed=np.asarray(seeds, dtype=np.int32),
        env=np.asarray(envs, dtype=np.int32),
        fail=np.asarray(fails, dtype=np.int32),
        fail_step=np.asarray(fsteps, dtype=np.int32),
        fail_reason=np.asarray(reasons),
        sr_task=np.asarray(srs, dtype=np.int32),
        sr_5cm=np.asarray(sr5s, dtype=np.float32),
        feat_names=np.asarray(UFR_FEAT_NAMES),
        meta=np.array(json.dumps(meta)),
    )
    if episodes[0].get("q") is not None:
        kw["q"] = np.concatenate([ep["q"] for ep in episodes], axis=0)
        kw["dq"] = np.concatenate([ep["dq"] for ep in episodes], axis=0)
        kw["q_cmd"] = np.concatenate([ep["q_cmd"] for ep in episodes], axis=0)
        kw["q_pre"] = np.concatenate([ep["q_pre"] for ep in episodes], axis=0)
        kw["action"] = np.concatenate([ep["action"] for ep in episodes], axis=0)
        kw["root_pos"] = np.concatenate([ep["root_pos"] for ep in episodes], axis=0)
        kw["root_quat"] = np.concatenate([ep["root_quat"] for ep in episodes], axis=0)
        kw["has_parent_cmd"] = np.array(True)
    np.savez_compressed(path, **kw)


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
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict] | None]:
    n_envs = int(env.unwrapped.num_envs)
    use, paths = _pin_clips(env, mask_name=mask_name)
    cmd = env.unwrapped.command_manager.get_term("motion")
    device = env.unwrapped.device
    asset = env.unwrapped.scene["robot"]
    am = env.unwrapped.action_manager
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
    fail_phase = [""] * use
    n_active_steps = [0] * use
    hold_start_t = getattr(cmd.motion_dir_loader, "_t13_hold_start", None)
    e9_prev = None
    apply_recovery = rec_kind == "shared_adapt"
    log_adapt = bool(log_adapter) and rec_kind in ("shared_adapt", "shadow")
    shadow_rows: list[dict] = []
    z_samples: list[dict] = []
    first_trig = [-1] * use
    ura_snaps: list[dict] = []
    ucr_snaps: list[dict] = []
    sample_ts: set[int] = set()
    if getattr(args_cli, "ura0_clone", False):
        sample_ts = {int(x) for x in str(args_cli.ura0_sample_ts).split(",") if x.strip()}
    hist_ring = [deque(maxlen=URA0_HIST) for _ in range(use)]
    ucr2p_on = bool(getattr(args_cli, "ucr2p_dump", False))
    ucr2p_h = int(getattr(args_cli, "ucr2p_hist", UCR2P_HIST) or UCR2P_HIST)
    proc_ring = [deque(maxlen=ucr2p_h) for _ in range(use)]
    re_streak = torch.zeros(use, dtype=torch.long, device=device)
    ucr_taken = torch.zeros(use, dtype=torch.bool, device=device)
    ucr0_on = bool(getattr(args_cli, "ucr0_clone", False)) or ucr2p_on
    persist_n = int(getattr(args_cli, "ucr0_persist", 3))
    re_on = float(getattr(args_cli, "re_on", RE_ON_DEFAULT))
    body_series = {
        (seed, e): {
            "torso": [], "lw": [], "rw": [], "S_raw": [], "R_E": [], "R_S": [], "R": [],
            "map_agg": [], "map_torso": [], "map_lw": [], "map_rw": [],
            "map_h01": [], "map_h02": [], "map_h04": [],
            "map_h01_torso": [], "map_h01_lw": [], "map_h01_rw": [],
            "map_h02_torso": [], "map_h02_lw": [], "map_h02_rw": [],
            "map_h04_torso": [], "map_h04_lw": [], "map_h04_rw": [],
            "coord_lr": [], "hand_geom": [], "v_root_z": [], "a_root_z": [],
            "active": [], "z": [], "e_vis": [],
            "root_vx": [], "root_vy": [], "root_vz": [],
            "root_wx": [], "root_wy": [], "root_wz": [],
            "e9": [], "edot9": [], "vis3": [],
            "q": [], "dq": [], "q_cmd": [], "q_pre": [], "action": [],
            "root_pos": [], "root_quat": [],
        }
        for e in range(use)
    }
    meta = _stress_meta()
    push_step = -1
    push_dv = 0.0
    push_axis = "lat"
    pushed = True
    push_info = {"applied": False, "command_dv": 0.0, "measured_dv_mean": 0.0}
    first_re = [-1] * use
    first_rs = [-1] * use
    phys = {
        (seed, e): {
            "arm_tau": [], "root_acc": [], "root_w": [], "roll": [], "pitch": [],
            "act_sat": [], "wrist_cf": [],
        }
        for e in range(use)
    }
    jnames = [str(x) for x in asset.data.joint_names]
    arm_idx = [i for i, n in enumerate(jnames) if any(k in n for k in (
        "shoulder", "elbow", "wrist"
    ))]
    prev_root_v = asset.data.root_lin_vel_w[:use].clone()
    prev_root_vz = asset.data.root_lin_vel_w[:use, 2].clone()
    cf = None
    try:
        cf = env.unwrapped.scene["contact_forces"]
    except Exception:
        cf = None
    wrist_body_ids = []
    if cf is not None:
        bnames = list(asset.data.body_names)
        for wn in WRIST_BODIES:
            if wn in bnames:
                wrist_body_ids.append(bnames.index(wn))
    bnames_all = list(asset.data.body_names)
    ankle_ids = [bnames_all.index(n) for n in _UCR2P_ANKLE if n in bnames_all]
    origins = env.unwrapped.scene.env_origins

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
        if (not pushed) and t == push_step:
            push_info = _apply_push(asset, use, push_dv, push_axis)
            pushed = True
        obs, _ = env.get_observations()
        obs = injector.patch_policy_obs(obs, env, "mapper")
        map_err = _mapper_body_errors(env, injector, vis_flags)
        e_pre = _visible_e(cmd, vis_i, vis_flags)
        e_torso, e_lw, e_rw = _body_e(cmd, vis_i)
        s_in = asset.data.root_lin_vel_w[:, 2].abs().to(dtype=e_pre.dtype)
        gout = gate.step(e_pre, s_in if bool(s_enabled) else torch.zeros_like(e_pre))
        active = gout["active"]
        z_nom, proprio = _z_and_proprio(policy, runner, obs)
        rec_pack, e9 = _recovery_pack(obs, z_nom, e9_prev)

        if (ucr0_on or ucr2p_on) and t >= 0:
            for i in range(use):
                if bool(failed[i]):
                    continue
                proc_ring[i].append(
                    _ucr2p_frame(
                        asset, am, origins[i], i,
                        float(e_pre[i].item()),
                        float(e_torso[i].item()),
                        float(e_lw[i].item()),
                        float(e_rw[i].item()),
                        rec_pack, gout, z_nom, cf, ankle_ids,
                    )
                )

        if ucr0_on and t >= WARMUP:
            on_e = gout["R_E"][:use] >= re_on
            re_streak = torch.where(on_e, re_streak + 1, torch.zeros_like(re_streak))
            take = (re_streak >= persist_n) & (~ucr_taken) & (~failed[:use])
            if bool(take.any()):
                hist_snap = _hist_snapshot(env)
                for i in range(use):
                    if not bool(take[i]):
                        continue
                    pos = cmd.robot_body_pos_w[i]
                    torso = pos[vis_i[0]]
                    d_lw = float((pos[vis_i[1]] - torso).norm().item())
                    d_rw = float((pos[vis_i[2]] - torso).norm().item())
                    wt_max = float(getattr(args_cli, "wrist_torso_max", 0.85))
                    if max(d_lw, d_rw) > wt_max:
                        ucr_taken[i] = True
                        continue
                    extra = {
                        "obs": obs[i].detach().cpu().clone(),
                        "rec_obs": rec_pack[i].detach().cpu().clone(),
                        "z_nom": z_nom[i].detach().cpu().clone(),
                        "task": task_source,
                        "lam": float(args_cli.lambda_id),
                        "clip": Path(paths[i]).name if i < len(paths) else f"env{i}",
                        "seed": int(seed),
                        "t": int(t),
                        "re": float(gout["R_E"][i].item()),
                        "rs": float(gout["R_S"][i].item()),
                        "e0": float(e_pre[i].item()),
                    }
                    if ucr2p_on:
                        hpad = np.zeros((ucr2p_h, UCR2P_D), dtype=np.float32)
                        arr = np.asarray(list(proc_ring[i]), dtype=np.float32)
                        if arr.size:
                            hpad[-arr.shape[0] :] = arr
                        extra["process"] = hpad
                    ucr_snaps.append(_capture_snap(env, i, hist_snap, extra))
                    ucr_taken[i] = True

        if sample_ts and (t in sample_ts) and adapter is not None:
            root_v_c = asset.data.root_lin_vel_w[:use]
            acc_c = (root_v_c - prev_root_v).norm(dim=-1) / DT
            a_z_c = (root_v_c[:, 2] - prev_root_vz) / DT
            roll_c, pitch_c, _yaw_c = euler_xyz_from_quat(asset.data.root_quat_w[:use])
            roll_c = torch.atan2(torch.sin(roll_c), torch.cos(roll_c))
            pitch_c = torch.atan2(torch.sin(pitch_c), torch.cos(pitch_c))
            w_c = asset.data.root_ang_vel_w[:use]
            cf_c = torch.zeros(use, device=device)
            if cf is not None and wrist_body_ids and hasattr(cf.data, "net_forces_w"):
                nf = cf.data.net_forces_w[:use]
                cf_c = nf[:, wrist_body_ids, :].norm(dim=-1).mean(dim=-1)
            raw_a, logits = adapter((rec_pack[:use] - mua) / sda)
            pred_th = logits.argmax(dim=-1)
            th_a = torch.tensor(
                [float(bins[int(j)]) for j in pred_th.tolist()], device=device, dtype=torch.float32
            )
            d_hat = project_tangent(raw_a, z_nom[:use])
            hist_snap = _hist_snapshot(env)
            for i in range(use):
                if bool(failed[i]):
                    continue
                feat = _ura0_feat(
                    rec_pack[i], z_nom[i], d_hat[i], float(th_a[i].item()),
                    root_v_c[i], float(a_z_c[i].item()), float(acc_c[i].item()),
                    w_c[i], float(roll_c[i].item()), float(pitch_c[i].item()), float(cf_c[i].item()),
                )
                h = np.zeros((URA0_HIST, 6), dtype=np.float32)
                arr = np.asarray(list(hist_ring[i]), dtype=np.float32)
                if arr.size:
                    h[-arr.shape[0] :] = arr
                extra = {
                    "obs": obs[i].detach().cpu().clone(),
                    "rec_obs": rec_pack[i].detach().cpu().clone(),
                    "z_nom": z_nom[i].detach().cpu().clone(),
                    "feat": feat,
                    "hist_feat": h,
                    "task": task_source,
                    "lam": float(args_cli.lambda_id),
                    "clip": Path(paths[i]).name if i < len(paths) else f"env{i}",
                    "seed": int(seed),
                    "t": int(t),
                    "re": float(gout["R_E"][i].item()),
                    "rs": float(gout["R_S"][i].item()),
                    "e0": float(e_pre[i].item()),
                    "s_enabled": int(bool(s_enabled)),
                }
                ura_snaps.append(_capture_snap(env, i, hist_snap, extra))

        for i in range(use):
            if bool(failed[i]):
                continue
            hist_ring[i].append(
                [
                    float(e_pre[i].item()),
                    float(rec_pack[i, 9:18].abs().mean().item()),
                    float(gout["R_E"][i].item()),
                    float(gout["R_S"][i].item()),
                    float(asset.data.root_lin_vel_w[i].norm().item()),
                    float(asset.data.root_ang_vel_w[i].norm().item()),
                ]
            )

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
        # Parent dump / shadow: never write recovery into the executed latent.
        if apply_recovery and bool(in_burst.any()):
            z_exec[in_burst] = apply_angle(z_nom[in_burst], held_d[in_burst], held_theta[in_burst])
        joints = _decode(policy, z_exec, proprio)
        q_pre_step = asset.data.joint_pos[:use].clone()
        env.step(joints)

        root_v = asset.data.root_lin_vel_w[:use]
        acc = (root_v - prev_root_v).norm(dim=-1) / DT
        a_z = (root_v[:, 2] - prev_root_vz) / DT
        prev_root_v = root_v.clone()
        prev_root_vz = root_v[:, 2].clone()
        roll, pitch, _yaw = euler_xyz_from_quat(asset.data.root_quat_w[:use])
        roll = torch.atan2(torch.sin(roll), torch.cos(roll))
        pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
        tau = asset.data.applied_torque[:use]
        arm_tau = tau[:, arm_idx].abs().mean(dim=-1) if arm_idx else torch.zeros(use, device=tau.device)
        wnorm = asset.data.root_ang_vel_w[:use].norm(dim=-1)
        act = getattr(env.unwrapped, "action_manager", None)
        sat = torch.zeros(use, device=tau.device)
        if act is not None and hasattr(act, "action"):
            sat = (act.action[:use].abs() > 0.95).float().mean(dim=-1)
        cf_w = torch.zeros(use, device=tau.device)
        if cf is not None and wrist_body_ids and hasattr(cf.data, "net_forces_w"):
            nf = cf.data.net_forces_w[:use]
            cf_w = nf[:, wrist_body_ids, :].norm(dim=-1).mean(dim=-1)

        e_post = _visible_e(cmd, vis_i, vis_flags)
        e_torso_p, e_lw_p, e_rw_p = _body_e(cmd, vis_i)
        hand_geom = torch.linalg.norm(
            (cmd.robot_body_pos_w[:use, vis_i[1]] - cmd.robot_body_pos_w[:use, vis_i[2]])
            - (cmd.body_pos_w[:use, vis_i[1]] - cmd.body_pos_w[:use, vis_i[2]]),
            dim=-1,
        )
        coord_lr = (e_lw_p[:use] - e_rw_p[:use]).abs()
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
            if hold_start_t is not None:
                mi = int(cmd.env_motion_indices[i].item())
                ts = int(cmd.time_steps[i].item())
                hs = int(hold_start_t[mi].item())
                if ts < hs:
                    n_active_steps[i] += 1
            e_post_series[(seed, i)].append(float(e_post[i].item()))
            body_series[(seed, i)]["e_vis"].append(float(e_post[i].item()))
            dump_extra = bool(getattr(args_cli, "ufr_dump", False) or getattr(args_cli, "ufr2_dump", False))
            if dump_extra:
                body_series[(seed, i)]["z"].append(z_nom[i].detach().cpu().numpy().astype(np.float32))
                body_series[(seed, i)]["e9"].append(e9[i].detach().cpu().numpy().astype(np.float32))
                body_series[(seed, i)]["edot9"].append(rec_pack[i, 9:18].detach().cpu().numpy().astype(np.float32))
                body_series[(seed, i)]["vis3"].append(rec_pack[i, 18:21].detach().cpu().numpy().astype(np.float32))
                body_series[(seed, i)]["root_vx"].append(float(root_v[i, 0].item()))
                body_series[(seed, i)]["root_vy"].append(float(root_v[i, 1].item()))
                body_series[(seed, i)]["root_vz"].append(float(root_v[i, 2].item()))
                rw = asset.data.root_ang_vel_w[i]
                body_series[(seed, i)]["root_wx"].append(float(rw[0].item()))
                body_series[(seed, i)]["root_wy"].append(float(rw[1].item()))
                body_series[(seed, i)]["root_wz"].append(float(rw[2].item()))
            if getattr(args_cli, "ufr2_dump", False):
                q_tgt = joints[i]
                if hasattr(asset.data, "joint_pos_target"):
                    q_tgt = asset.data.joint_pos_target[i]
                body_series[(seed, i)]["q"].append(asset.data.joint_pos[i].detach().cpu().numpy().astype(np.float32))
                body_series[(seed, i)]["dq"].append(asset.data.joint_vel[i].detach().cpu().numpy().astype(np.float32))
                body_series[(seed, i)]["q_cmd"].append(q_tgt.detach().cpu().numpy().astype(np.float32))
                body_series[(seed, i)]["q_pre"].append(q_pre_step[i].detach().cpu().numpy().astype(np.float32))
                body_series[(seed, i)]["action"].append(joints[i].detach().cpu().numpy().astype(np.float32))
                body_series[(seed, i)]["root_pos"].append(asset.data.root_pos_w[i].detach().cpu().numpy().astype(np.float32))
                body_series[(seed, i)]["root_quat"].append(asset.data.root_quat_w[i].detach().cpu().numpy().astype(np.float32))
            body_series[(seed, i)]["torso"].append(float(e_torso_p[i].item()))
            body_series[(seed, i)]["lw"].append(float(e_lw_p[i].item()))
            body_series[(seed, i)]["rw"].append(float(e_rw_p[i].item()))
            body_series[(seed, i)]["S_raw"].append(float(s_in[i].item()))
            body_series[(seed, i)]["R_E"].append(float(gout["R_E"][i].item()))
            body_series[(seed, i)]["R_S"].append(float(gout["R_S"][i].item()))
            body_series[(seed, i)]["R"].append(float(gout["R"][i].item()))
            body_series[(seed, i)]["coord_lr"].append(float(coord_lr[i].item()))
            body_series[(seed, i)]["hand_geom"].append(float(hand_geom[i].item()))
            body_series[(seed, i)]["v_root_z"].append(float(s_in[i].item()))
            body_series[(seed, i)]["a_root_z"].append(float(a_z[i].item()))
            body_series[(seed, i)]["active"].append(1.0 if bool(active[i]) else 0.0)
            if map_err is not None:
                body_series[(seed, i)]["map_agg"].append(float(map_err["agg"][i].mean().item()))
                body_series[(seed, i)]["map_torso"].append(float(map_err["torso"][i].mean().item()))
                body_series[(seed, i)]["map_lw"].append(float(map_err["lw"][i].mean().item()))
                body_series[(seed, i)]["map_rw"].append(float(map_err["rw"][i].mean().item()))
                body_series[(seed, i)]["map_h01"].append(float(map_err["h0.1_agg"][i].item()))
                body_series[(seed, i)]["map_h02"].append(float(map_err["h0.2_agg"][i].item()))
                body_series[(seed, i)]["map_h04"].append(float(map_err["h0.4_agg"][i].item()))
                body_series[(seed, i)]["map_h01_torso"].append(float(map_err["h0.1_torso"][i].item()))
                body_series[(seed, i)]["map_h01_lw"].append(float(map_err["h0.1_lw"][i].item()))
                body_series[(seed, i)]["map_h01_rw"].append(float(map_err["h0.1_rw"][i].item()))
                body_series[(seed, i)]["map_h02_torso"].append(float(map_err["h0.2_torso"][i].item()))
                body_series[(seed, i)]["map_h02_lw"].append(float(map_err["h0.2_lw"][i].item()))
                body_series[(seed, i)]["map_h02_rw"].append(float(map_err["h0.2_rw"][i].item()))
                body_series[(seed, i)]["map_h04_torso"].append(float(map_err["h0.4_torso"][i].item()))
                body_series[(seed, i)]["map_h04_lw"].append(float(map_err["h0.4_lw"][i].item()))
                body_series[(seed, i)]["map_h04_rw"].append(float(map_err["h0.4_rw"][i].item()))
            phys[(seed, i)]["arm_tau"].append(float(arm_tau[i].item()))
            phys[(seed, i)]["root_acc"].append(float(acc[i].item()))
            phys[(seed, i)]["root_w"].append(float(wnorm[i].item()))
            phys[(seed, i)]["roll"].append(float(roll[i].item()))
            phys[(seed, i)]["pitch"].append(float(pitch[i].item()))
            phys[(seed, i)]["act_sat"].append(float(sat[i].item()))
            phys[(seed, i)]["wrist_cf"].append(float(cf_w[i].item()))
            if first_re[i] < 0 and float(gout["R_E"][i].item()) >= 1.0 and t >= WARMUP:
                first_re[i] = int(t)
            if first_rs[i] < 0 and float(gout["R_S"][i].item()) >= 1.0 and t >= WARMUP:
                first_rs[i] = int(t)
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
                    mi = int(cmd.env_motion_indices[i].item())
                    ts = int(cmd.time_steps[i].item())
                    hs = int(hold_start_t[mi].item()) if hold_start_t is not None else 10**9
                    fail_phase[i] = "hold" if ts >= hs else "active"
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
                f"[t13] {rec_kind} {task_source} {terrain} s{seed} step {t+1}/{steps} "
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
                "poi": {
                    "visible": _poi_rates(e_vis.tolist()),
                    "torso": _poi_rates(bs["torso"] if vis_flags[0] else []),
                    "lw": _poi_rates(bs["lw"] if vis_flags[1] else []),
                    "rw": _poi_rates(bs["rw"] if vis_flags[2] else []),
                },
                "S_raw": _series_stats(bs["S_raw"]),
                "mapper": {
                    "agg": _series_stats(bs["map_agg"]),
                    "torso": _series_stats(bs["map_torso"]),
                    "lw": _series_stats(bs["map_lw"]),
                    "rw": _series_stats(bs["map_rw"]),
                    "h0.1": _series_stats(bs["map_h01"]),
                    "h0.2": _series_stats(bs["map_h02"]),
                    "h0.4": _series_stats(bs["map_h04"]),
                    "h0.1_torso": _series_stats(bs["map_h01_torso"]),
                    "h0.1_lw": _series_stats(bs["map_h01_lw"]),
                    "h0.1_rw": _series_stats(bs["map_h01_rw"]),
                    "h0.2_torso": _series_stats(bs["map_h02_torso"]),
                    "h0.2_lw": _series_stats(bs["map_h02_lw"]),
                    "h0.2_rw": _series_stats(bs["map_h02_rw"]),
                    "h0.4_torso": _series_stats(bs["map_h04_torso"]),
                    "h0.4_lw": _series_stats(bs["map_h04_lw"]),
                    "h0.4_rw": _series_stats(bs["map_h04_rw"]),
                },
                "coord_lr": _series_stats(bs["coord_lr"]),
                "hand_geom": _series_stats(bs["hand_geom"]),
                "v_root_z": _series_stats(bs["v_root_z"]),
                "a_root_z": _series_stats(bs["a_root_z"]),
                "trigger_duty": float(np.mean(bs["active"])) if bs["active"] else 0.0,
                "fail": int(fail),
                "fail_reason": fail_reason[i] or "none",
                "fail_phase": fail_phase[i] or ("none" if not fail else "unknown"),
                "n_active_steps": int(n_active_steps[i]),
                "active_motion_fraction": float(n_active_steps[i] / max(ep_len, 1)),
                "final_hold_fraction": float(1.0 - n_active_steps[i] / max(ep_len, 1)),
                "fail_taxonomy": tax,
                "fail_step": int(fail_step[i]),
                "episode_length": int(ep_len),
                "n_events": n_ev,
                "any_trigger": int(n_ev > 0),
                "first_trigger_t": int(t0),
                "trigger_before_fail": int(trig_before),
                "lead_time_s": lead,
                "task_completion": int(not fail),
                "first_RE_t": int(first_re[i]),
                "first_RS_t": int(first_rs[i]),
                "first_precursor": (
                    "RE" if first_re[i] >= 0 and (first_rs[i] < 0 or first_re[i] <= first_rs[i])
                    else ("RS" if first_rs[i] >= 0 else "neither")
                ),
                "R_E": _series_stats(bs["R_E"]),
                "R_S": _series_stats(bs["R_S"]),
                "max_RE": float(max(bs["R_E"]) if bs["R_E"] else float("nan")),
                "max_RS": float(max(bs["R_S"]) if bs["R_S"] else float("nan")),
                "stress": meta,
                "physics": {
                    "arm_torque_mean": _series_stats(phys[(seed, i)]["arm_tau"]),
                    "root_acc_mean": _series_stats(phys[(seed, i)]["root_acc"]),
                    "root_angvel": _series_stats(phys[(seed, i)]["root_w"]),
                    "roll": _series_stats(phys[(seed, i)]["roll"]),
                    "pitch": _series_stats(phys[(seed, i)]["pitch"]),
                    "action_sat": _series_stats(phys[(seed, i)]["act_sat"]),
                    "wrist_contact": _series_stats(phys[(seed, i)]["wrist_cf"]),
                    "peak_E": float(np.max(e_vis)) if e_vis.size else float("nan"),
                    "peak_roll": float(np.max(np.abs(phys[(seed, i)]["roll"]))) if phys[(seed, i)]["roll"] else float("nan"),
                    "peak_root_acc": float(np.max(phys[(seed, i)]["root_acc"])) if phys[(seed, i)]["root_acc"] else float("nan"),
                },
                "push": push_info,
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
    ufr = None
    if getattr(args_cli, "ufr_dump", False) or getattr(args_cli, "ufr2_dump", False):
        ufr = _pack_ufr_seed(seed, use, paths, vis_flags, body_series, phys, ep_rows)
    sr_map = {i: int(ep_rows[i].get("sr_task", 1)) for i in range(len(ep_rows))}
    for s in ura_snaps:
        s["sr_ep"] = sr_map.get(int(s.get("env_id", -1)), 1)
    for s in ucr_snaps:
        s["sr_ep"] = sr_map.get(int(s.get("env_id", -1)), 1)
    return events, ep_rows, shadow_rows, z_samples, ufr, ura_snaps, ucr_snaps


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
        "sr_2cm": _nanmean_key(all_eps, lambda r: r.get("sr_2cm")),
        "poi_err": _nanmean_key(all_eps, lambda r: r.get("e_kp_mean")),
        "poi": {
            "visible": {
                "mean": _nanmean_key(all_eps, lambda r: ((r.get("poi") or {}).get("visible") or {}).get("mean", r.get("e_kp_mean"))),
                "sr_5cm": _nanmean_key(all_eps, lambda r: ((r.get("poi") or {}).get("visible") or {}).get("sr_5cm", r.get("sr_5cm"))),
                "sr_2cm": _nanmean_key(all_eps, lambda r: ((r.get("poi") or {}).get("visible") or {}).get("sr_2cm", r.get("sr_2cm"))),
            },
            "torso": {
                "mean": _nanmean_key(all_eps, lambda r: ((r.get("poi") or {}).get("torso") or {}).get("mean", ((r.get("e_torso") or {}).get("mean")))),
                "sr_5cm": _nanmean_key(all_eps, lambda r: ((r.get("poi") or {}).get("torso") or {}).get("sr_5cm")),
                "sr_2cm": _nanmean_key(all_eps, lambda r: ((r.get("poi") or {}).get("torso") or {}).get("sr_2cm")),
            },
            "lw": {
                "mean": _nanmean_key(all_eps, lambda r: ((r.get("poi") or {}).get("lw") or {}).get("mean", ((r.get("e_lw") or {}).get("mean")))),
                "sr_5cm": _nanmean_key(all_eps, lambda r: ((r.get("poi") or {}).get("lw") or {}).get("sr_5cm")),
                "sr_2cm": _nanmean_key(all_eps, lambda r: ((r.get("poi") or {}).get("lw") or {}).get("sr_2cm")),
            },
            "rw": {
                "mean": _nanmean_key(all_eps, lambda r: ((r.get("poi") or {}).get("rw") or {}).get("mean", ((r.get("e_rw") or {}).get("mean")))),
                "sr_5cm": _nanmean_key(all_eps, lambda r: ((r.get("poi") or {}).get("rw") or {}).get("sr_5cm")),
                "sr_2cm": _nanmean_key(all_eps, lambda r: ((r.get("poi") or {}).get("rw") or {}).get("sr_2cm")),
            },
        },
        "fail_frac": float(np.mean([r["fail"] for r in all_eps])) if n else float("nan"),
        "fall_frac": float(np.mean([r.get("fail_reason") == "fall" for r in all_eps])) if n else float("nan"),
        "P_any_trigger": float(np.mean([r.get("any_trigger", 0) for r in all_eps])) if n else float("nan"),
        "events_per_episode": float(np.mean([r.get("n_events", 0) for r in all_eps])) if n else float("nan"),
        "trigger_before_failure_recall": recall,
        "n_failed": len(failed),
        "n_success": len(ok),
        "lead_time_s": _pct(np.array(leads, dtype=np.float64), qs=(10, 25, 50, 75, 90)),
        "fail_reasons": reasons,
        "n_fail_active": int(sum(1 for r in failed if r.get("fail_phase") == "active")),
        "n_fail_hold": int(sum(1 for r in failed if r.get("fail_phase") == "hold")),
        "mean_active_motion_fraction": float(np.mean([r.get("active_motion_fraction", np.nan) for r in all_eps])) if n else float("nan"),
        "mean_hold_fraction": float(np.mean([r.get("final_hold_fraction", np.nan) for r in all_eps])) if n else float("nan"),
        "taxonomy": {k: (v / n if n else 0.0) for k, v in tax.items()},
        "success_trigger_duty": float(np.mean([r.get("n_events", 0) for r in ok])) if ok else float("nan"),
        "fail_trigger_duty": float(np.mean([r.get("n_events", 0) for r in failed])) if failed else float("nan"),
        "false_positive_trigger_rate": float(np.mean([r.get("any_trigger", 0) for r in ok])) if ok else float("nan"),
        "e_kp": _pct(np.array([r["e_kp_mean"] for r in all_eps], dtype=np.float64) * 100.0),
        "success_max_RE": _pct(np.array([r.get("max_RE", np.nan) for r in ok], dtype=np.float64)),
        "fail_max_RE": _pct(np.array([r.get("max_RE", np.nan) for r in failed], dtype=np.float64)),
        "success_max_RS": _pct(np.array([r.get("max_RS", np.nan) for r in ok], dtype=np.float64)),
        "fail_max_RS": _pct(np.array([r.get("max_RS", np.nan) for r in failed], dtype=np.float64)),
        "mapper_agg": _pct(np.array([
            ((r.get("mapper") or {}).get("agg") or {}).get("mean", np.nan) for r in all_eps
        ], dtype=np.float64)),
        "mapper_h02": _pct(np.array([
            ((r.get("mapper") or {}).get("h0.2") or {}).get("mean", np.nan) for r in all_eps
        ], dtype=np.float64)),
        "exec_e": _pct(np.array([r.get("e_kp_mean", np.nan) for r in all_eps], dtype=np.float64)),
        "trigger_duty": float(np.mean([r.get("trigger_duty", 0.0) for r in all_eps])) if n else float("nan"),
        "success_trigger_rate": float(np.mean([r.get("any_trigger", 0) for r in ok])) if ok else float("nan"),
        "fail_trigger_rate": float(np.mean([r.get("any_trigger", 0) for r in failed])) if failed else float("nan"),
        "P_S_trigger_given_fail": float(np.mean([
            str(r.get("first_precursor")) == "RS" or (
                ((r.get("R_S") or {}).get("mean") or 0) >= 1.0
            ) for r in failed
        ])) if failed else float("nan"),
        "P_S_trigger_given_success": float(np.mean([
            r.get("first_precursor") == "RS" for r in ok
        ])) if ok else float("nan"),
        "peak_v_root_z": _pct(np.array([
            (r.get("v_root_z") or {}).get("p90", np.nan) for r in all_eps
        ], dtype=np.float64)),
        "peak_a_root_z": _pct(np.array([
            (r.get("a_root_z") or {}).get("p90", np.nan) for r in all_eps
        ], dtype=np.float64)),
        "physics": {
            "arm_torque_mean": _pct(np.array([
                (r.get("physics") or {}).get("arm_torque_mean", {}).get("mean", np.nan) for r in all_eps
            ], dtype=np.float64)),
            "root_acc_mean": _pct(np.array([
                (r.get("physics") or {}).get("root_acc_mean", {}).get("mean", np.nan) for r in all_eps
            ], dtype=np.float64)),
            "peak_roll": _pct(np.array([
                (r.get("physics") or {}).get("peak_roll", np.nan) for r in all_eps
            ], dtype=np.float64)),
            "peak_E": _pct(np.array([
                (r.get("physics") or {}).get("peak_E", np.nan) for r in all_eps
            ], dtype=np.float64)),
            "action_sat": _pct(np.array([
                (r.get("physics") or {}).get("action_sat", {}).get("mean", np.nan) for r in all_eps
            ], dtype=np.float64)),
            "wrist_contact": _pct(np.array([
                (r.get("physics") or {}).get("wrist_contact", {}).get("mean", np.nan) for r in all_eps
            ], dtype=np.float64)),
            "measured_push_dv": _pct(np.array([
                (r.get("push") or {}).get("measured_dv_mean", np.nan) for r in all_eps
            ], dtype=np.float64)),
        },
    }


@torch.inference_mode()
def _ucr1_heldout_cell(env_cfg, agent_cfg, resume_path: str, motion_dir: str, terrain: str, cell: Path) -> dict:
    """Restore test RE-trigger snaps and roll candidate adapters. Parent/old/oracle from clones.npz."""
    done_p = cell / "heldout.json"
    if done_p.exists():
        print(f"[ucr1-heldout] skip existing {done_p}", flush=True)
        return json.loads(done_p.read_text())
    snaps_path = Path(str(getattr(args_cli, "ucr1_snaps", "") or ""))
    adapters_path = Path(str(getattr(args_cli, "ucr1_adapters_json", "") or ""))
    clips_path = Path(str(getattr(args_cli, "ucr1_test_clips", "") or ""))
    clones_src = snaps_path.with_name("clones.npz") if snaps_path.suffix == ".pt" else cell / "clones.npz"
    if not snaps_path.is_file():
        payload = {"n": 0, "skipped": True, "reason": f"missing snaps {snaps_path}"}
        done_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    env, runner, policy, injector = _make_env(env_cfg, agent_cfg, resume_path, motion_dir, terrain)
    warp = getattr(env.unwrapped, "_t13_warp_info", {}) or {}
    if bool(warp.get("out_of_plausible_range")) or bool(warp.get("kinematic_impossible")):
        payload = {"n": 0, "skipped": True, "reason": "implausible_cell"}
        done_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    snaps_all = torch.load(snaps_path, map_location="cpu", weights_only=False)
    test_clips = set(json.loads(clips_path.read_text())) if clips_path.is_file() else None
    snaps = []
    for s in snaps_all or []:
        if test_clips is not None and str(s.get("clip")) not in test_clips:
            continue
        snaps.append(s)
    clones = np.load(clones_src, allow_pickle=True) if clones_src.is_file() else None
    key_to_row = {}
    if clones is not None and int(clones["j_p"].shape[0]) > 0:
        clips = np.asarray(clones["clip"]).astype(str)
        seeds = np.asarray(clones["seed"]).astype(int)
        ts = np.asarray(clones["t"]).astype(int)
        for i in range(len(clips)):
            key_to_row[(clips[i], int(seeds[i]), int(ts[i]))] = i
    kept = []
    parent_idx = []
    for s in snaps:
        k = (str(s.get("clip")), int(s.get("seed")), int(s.get("t")))
        if k in key_to_row:
            kept.append(s)
            parent_idx.append(key_to_row[k])
        elif clones is None:
            kept.append(s)
            parent_idx.append(-1)
    snaps = kept
    adapters = json.loads(adapters_path.read_text()) if adapters_path.is_file() else []
    cmd = env.unwrapped.command_manager.get_term("motion")
    body_names = list(cmd.cfg.body_names)
    vis_i = [body_names.index(n) for n in KP_VIS]
    mask_name = str(args_cli.mask_modes).split(",")[0].strip() or "torso"
    vis_flags = MASK_VIS.get(mask_name, (True, True, True))
    asset = env.unwrapped.scene["robot"]
    pelvis_i = list(asset.data.body_names).index("pelvis") if "pelvis" in asset.data.body_names else None
    lam_fail = float(args_cli.lam_fail)
    methods = {}
    if clones is not None and parent_idx and all(i >= 0 for i in parent_idx):
        ii = np.asarray(parent_idx, dtype=np.int64)
        methods["parent"] = {
            "j": np.asarray(clones["j_p"][ii], dtype=np.float32),
            "i": np.zeros(len(ii), dtype=np.float32),
        }
        methods["old_rm3"] = {
            "j": np.asarray(clones["j_r"][ii], dtype=np.float32),
            "i": np.asarray(clones["i_rm3"][ii], dtype=np.float32),
        }
        methods["oracle"] = {
            "j": np.asarray(clones["j_ora"][ii], dtype=np.float32),
            "i": np.asarray(clones["i_ora"][ii], dtype=np.float32),
        }
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    j_p = methods.get("parent", {}).get("j")
    for spec in adapters:
        name = str(spec["name"])
        adapter, mua, sda, bins = _load_rm2_adapter(str(spec["path"]), device)
        j_new = []
        for start in range(0, len(snaps), n_envs):
            batch = snaps[start : start + n_envs]
            n_valid = len(batch)
            print(f"[ucr1-heldout] {name} {start}:{start + n_valid}/{len(snaps)}", flush=True)
            rec = torch.stack([s["rec_obs"] for s in batch], dim=0).to(device=device, dtype=torch.float32)
            raw_a, logits = adapter((rec - mua) / sda)
            pred = logits.argmax(dim=-1)
            th_a = torch.tensor([float(bins[int(j)]) for j in pred.tolist()], device=device, dtype=torch.float32)
            d_pad = torch.zeros(n_envs, 16, device=device, dtype=torch.float32)
            th_pad = torch.full((n_envs,), 5.0, device=device, dtype=torch.float32)
            d_pad[:n_valid] = raw_a
            th_pad[:n_valid] = th_a
            e_n, f_n = _rollout_probe_jr(
                env, policy, runner, injector, batch, n_valid,
                d_raw=d_pad, theta_deg=th_pad, burst=BURST_STEPS, horizon=HORIZON_PAIR,
                vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i,
            )
            j_new.append(_j_from_ef(e_n, f_n, lam_fail).detach().cpu().numpy().astype(np.float32))
        j_arr = np.concatenate(j_new, axis=0) if j_new else np.zeros(0, dtype=np.float32)
        i_arr = (j_arr - j_p) if j_p is not None and j_p.shape[0] == j_arr.shape[0] else np.full_like(j_arr, np.nan)
        methods[name] = {"j": j_arr, "i": i_arr}
    def _stats(i):
        x = np.asarray(i, dtype=np.float64) * 100.0
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"n": 0}
        return {
            "n": int(x.size),
            "P_I_lt_0": float((x < 0).mean()),
            "P_I_lt_0.25cm": float((x < -0.25).mean()),
            "median_cm": float(np.median(x)),
            "mean_cm": float(x.mean()),
            "P_I_gt_0.5cm": float((x > 0.5).mean()),
            "P_I_gt_1.0cm": float((x > 1.0).mean()),
        }
    payload = {
        "n": len(snaps),
        "task": str(args_cli.task_source),
        "lambda_id": float(args_cli.lambda_id),
        "test_clips": sorted(test_clips) if test_clips is not None else None,
        "methods": {k: {**_stats(v["i"]), "j": v["j"].tolist(), "i_m": v["i"].tolist()} for k, v in methods.items()},
        "keys": [
            {"clip": str(s.get("clip")), "seed": int(s.get("seed")), "t": int(s.get("t"))}
            for s in snaps
        ],
    }
    done_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        cell / "heldout.npz",
        **{f"i_{k}": np.asarray(v["i"], dtype=np.float32) for k, v in methods.items()},
        **{f"j_{k}": np.asarray(v["j"], dtype=np.float32) for k, v in methods.items()},
        task=np.asarray([str(s.get("task", args_cli.task_source)) for s in snaps]),
        clip=np.asarray([str(s.get("clip")) for s in snaps]),
        seed=np.asarray([int(s.get("seed")) for s in snaps], dtype=np.int32),
        t=np.asarray([int(s.get("t")) for s in snaps], dtype=np.int32),
        lam=np.asarray([float(s.get("lam", args_cli.lambda_id)) for s in snaps], dtype=np.float32),
    )
    print(f"[ucr1-heldout] wrote {done_p} n={len(snaps)} adapters={len(adapters)}", flush=True)
    return payload


@torch.inference_mode()
def _ucr1d_probe_cell(env_cfg, agent_cfg, resume_path: str, motion_dir: str, terrain: str, cell: Path) -> dict:
    """kNN-assigned (d,θ) clone rolls + optional 15-D ±1° FD set. No new controller."""
    done_p = cell / "probe.json"
    if done_p.exists():
        print(f"[ucr1d] skip existing {done_p}", flush=True)
        return json.loads(done_p.read_text())
    snaps_path = Path(str(getattr(args_cli, "ucr1_snaps", "") or ""))
    assign_path = Path(str(getattr(args_cli, "ucr1d_assign", "") or ""))
    clips_path = Path(str(getattr(args_cli, "ucr1_test_clips", "") or ""))
    clones_src = snaps_path.with_name("clones.npz") if snaps_path.suffix == ".pt" else cell / "clones.npz"
    if not snaps_path.is_file() or not assign_path.is_file():
        payload = {"n": 0, "skipped": True, "reason": "missing snaps or assignments"}
        done_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    env, runner, policy, injector = _make_env(env_cfg, agent_cfg, resume_path, motion_dir, terrain)
    warp = getattr(env.unwrapped, "_t13_warp_info", {}) or {}
    if bool(warp.get("out_of_plausible_range")) or bool(warp.get("kinematic_impossible")):
        payload = {"n": 0, "skipped": True, "reason": "implausible_cell"}
        done_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    snaps_all = torch.load(snaps_path, map_location="cpu", weights_only=False)
    test_clips = set(json.loads(clips_path.read_text())) if clips_path.is_file() else None
    snaps = [s for s in (snaps_all or []) if test_clips is None or str(s.get("clip")) in test_clips]
    clones = np.load(clones_src, allow_pickle=True) if clones_src.is_file() else None
    key_to_row = {}
    if clones is not None and int(clones["j_p"].shape[0]) > 0:
        clips = np.asarray(clones["clip"]).astype(str)
        seeds = np.asarray(clones["seed"]).astype(int)
        ts = np.asarray(clones["t"]).astype(int)
        for i in range(len(clips)):
            key_to_row[(clips[i], int(seeds[i]), int(ts[i]))] = i
    asg = np.load(assign_path, allow_pickle=True)
    a_clip = np.asarray(asg["clip"]).astype(str)
    a_seed = np.asarray(asg["seed"]).astype(int)
    a_t = np.asarray(asg["t"]).astype(int)
    a_task = np.asarray(asg["task"]).astype(str)
    a_lam = np.asarray(asg["lam"]).astype(np.float32)
    cell_task = str(args_cli.task_source)
    cell_lam = float(args_cli.lambda_id)
    key_to_asg = {}
    for i in range(len(a_clip)):
        if a_task[i] != cell_task:
            continue
        if abs(float(a_lam[i]) - cell_lam) > 1e-6:
            continue
        key_to_asg[(a_clip[i], int(a_seed[i]), int(a_t[i]))] = i
    kept, parent_idx, asg_idx = [], [], []
    for s in snaps:
        k = (str(s.get("clip")), int(s.get("seed")), int(s.get("t")))
        if k not in key_to_asg:
            continue
        if k not in key_to_row:
            continue
        kept.append(s)
        parent_idx.append(key_to_row[k])
        asg_idx.append(key_to_asg[k])
    snaps = kept
    methods: dict[str, dict] = {}
    if clones is not None and parent_idx:
        ii = np.asarray(parent_idx, dtype=np.int64)
        methods["parent"] = {"j": np.asarray(clones["j_p"][ii], dtype=np.float32), "i": np.zeros(len(ii), dtype=np.float32)}
        methods["old_rm3"] = {"j": np.asarray(clones["j_r"][ii], dtype=np.float32), "i": np.asarray(clones["i_rm3"][ii], dtype=np.float32)}
        methods["oracle"] = {"j": np.asarray(clones["j_ora"][ii], dtype=np.float32), "i": np.asarray(clones["i_ora"][ii], dtype=np.float32)}
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    cmd = env.unwrapped.command_manager.get_term("motion")
    body_names = list(cmd.cfg.body_names)
    vis_i = [body_names.index(n) for n in KP_VIS]
    mask_name = str(args_cli.mask_modes).split(",")[0].strip() or "torso"
    vis_flags = MASK_VIS.get(mask_name, (True, True, True))
    asset = env.unwrapped.scene["robot"]
    pelvis_i = list(asset.data.body_names).index("pelvis") if "pelvis" in asset.data.body_names else None
    lam_fail = float(args_cli.lam_fail)
    j_p = methods.get("parent", {}).get("j")
    ai = np.asarray(asg_idx, dtype=np.int64)

    knn_specs = [
        ("knn_pooled", "d_pooled", "th_pooled"),
        ("knn_same", "d_same", "th_same"),
        ("knn_from_loco", "d_from_loco", "th_from_loco"),
        ("knn_from_stoop", "d_from_stoop", "th_from_stoop"),
        ("knn_from_reach", "d_from_reach", "th_from_reach"),
        ("knn_from_carry", "d_from_carry", "th_from_carry"),
        ("knn_p200", "d_p200", "th_p200"),
        ("knn_p500", "d_p500", "th_p500"),
        ("knn_p200_same", "d_p200_same", "th_p200_same"),
        ("knn_p500_same", "d_p500_same", "th_p500_same"),
        ("knn_p200_sum", "d_p200_sum", "th_p200_sum"),
        ("knn_p500_sum", "d_p500_sum", "th_p500_sum"),
    ]
    knn_specs = [(n, dk, tk) for n, dk, tk in knn_specs if dk in asg.files and tk in asg.files]

    def _roll_dth(name: str, d_np: np.ndarray, th_np: np.ndarray) -> None:
        j_chunks = []
        for start in range(0, len(snaps), n_envs):
            batch = snaps[start : start + n_envs]
            n_valid = len(batch)
            print(f"[ucr1d] {name} {start}:{start + n_valid}/{len(snaps)}", flush=True)
            d_pad = torch.zeros(n_envs, 16, device=device, dtype=torch.float32)
            th_pad = torch.full((n_envs,), 5.0, device=device, dtype=torch.float32)
            d_pad[:n_valid] = torch.as_tensor(d_np[start : start + n_valid], device=device, dtype=torch.float32)
            th_pad[:n_valid] = torch.as_tensor(th_np[start : start + n_valid], device=device, dtype=torch.float32)
            e_n, f_n = _rollout_probe_jr(
                env, policy, runner, injector, batch, n_valid,
                d_raw=d_pad, theta_deg=th_pad, burst=BURST_STEPS, horizon=HORIZON_PAIR,
                vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i,
            )
            j_chunks.append(_j_from_ef(e_n, f_n, lam_fail).detach().cpu().numpy().astype(np.float32))
        j_arr = np.concatenate(j_chunks, 0) if j_chunks else np.zeros(0, dtype=np.float32)
        i_arr = (j_arr - j_p) if j_p is not None and j_p.shape[0] == j_arr.shape[0] else np.full_like(j_arr, np.nan)
        methods[name] = {"j": j_arr, "i": i_arr}

    for name, dk, tk in knn_specs:
        _roll_dth(name, np.asarray(asg[dk][ai], dtype=np.float32), np.asarray(asg[tk][ai], dtype=np.float32))

    fd_i = None
    if bool(getattr(args_cli, "ucr1d_fd", False)) and snaps:
        # 15 tangent axes × ±1°. I relative to Parent.
        cols = []
        for k in range(15):
            for sign, tag in ((1.0, "p"), (-1.0, "m")):
                j_chunks = []
                for start in range(0, len(snaps), n_envs):
                    batch = snaps[start : start + n_envs]
                    n_valid = len(batch)
                    z0 = torch.stack([s["z_nom"] for s in batch], dim=0).to(device=device, dtype=torch.float32)
                    b = tangent_basis(z0)
                    d_pad = torch.zeros(n_envs, 16, device=device, dtype=torch.float32)
                    d_pad[:n_valid] = float(sign) * b[:n_valid, :, k]
                    print(f"[ucr1d] fd {k + 1}/15 {tag} {start}:{start + n_valid}/{len(snaps)}", flush=True)
                    e_n, f_n = _rollout_probe_jr(
                        env, policy, runner, injector, batch, n_valid,
                        d_raw=d_pad, theta_deg=UCR0_EPS_DEG, burst=BURST_STEPS, horizon=HORIZON_PAIR,
                        vis_i=vis_i, vis_flags=vis_flags, pelvis_i=pelvis_i,
                    )
                    j_chunks.append(_j_from_ef(e_n, f_n, lam_fail).detach().cpu().numpy().astype(np.float32))
                j_arr = np.concatenate(j_chunks, 0)
                i_arr = (j_arr - j_p) if j_p is not None else np.full_like(j_arr, np.nan)
                methods[f"fd_{k}_{tag}"] = {"j": j_arr, "i": i_arr}
                cols.append(i_arr)
        fd_i = np.stack(cols, axis=1) if cols else None
        if fd_i is not None:
            i_star = methods["oracle"]["i"] if "oracle" in methods else None
            i_fd_best = fd_i.min(axis=1)
            methods["fd_best"] = {"j": (j_p + i_fd_best) if j_p is not None else j_p, "i": i_fd_best}

    def _stats(i):
        x = np.asarray(i, dtype=np.float64) * 100.0
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"n": 0}
        return {
            "n": int(x.size),
            "P_I_lt_0": float((x < 0).mean()),
            "P_I_lt_0.25cm": float((x < -0.25).mean()),
            "median_cm": float(np.median(x)),
            "mean_cm": float(x.mean()),
            "P_I_gt_0.5cm": float((x > 0.5).mean()),
            "P_I_gt_1.0cm": float((x > 1.0).mean()),
        }

    amb = None
    if fd_i is not None and "oracle" in methods:
        i_star = methods["oracle"]["i"][:, None]
        eq_star = fd_i <= (i_star + 0.0025)
        i_best = fd_i.min(axis=1, keepdims=True)
        eq_best = fd_i <= (i_best + 0.0025)
        n_eq_star = eq_star.sum(axis=1)
        n_eq_best = eq_best.sum(axis=1)
        amb = {
            "n": int(fd_i.shape[0]),
            "n_dirs": int(fd_i.shape[1]),
            "mean_n_eq_vs_oracle_Istar": float(n_eq_star.mean()),
            "P_n_eq_ge_2_vs_oracle": float((n_eq_star >= 2).mean()),
            "mean_n_eq_vs_best_1deg": float(n_eq_best.mean()),
            "P_n_eq_ge_2_vs_best_1deg": float((n_eq_best >= 2).mean()),
            "P_any_1deg_beats_parent": float((fd_i.min(axis=1) < 0).mean()),
            "median_width_vs_oracle": float(np.median(n_eq_star)),
            "mean_best_1deg_I_cm": float(np.median(fd_i.min(axis=1)) * 100.0),
        }
        methods["fd_set"] = {"n_eq_vs_oracle": n_eq_star.astype(np.int16), "n_eq_vs_best1": n_eq_best.astype(np.int16)}

    payload = {
        "n": len(snaps),
        "task": cell_task,
        "lambda_id": cell_lam,
        "methods": {k: _stats(v["i"]) for k, v in methods.items() if k not in ("fd_set",) and not str(k).startswith("fd_")},
        "knn_methods": {k: _stats(methods[k]["i"]) for k, _, _ in knn_specs if k in methods},
        "baselines": {k: _stats(methods[k]["i"]) for k in ("parent", "old_rm3", "oracle") if k in methods},
        "label_set_1deg": amb,
        "keys": [{"clip": str(s.get("clip")), "seed": int(s.get("seed")), "t": int(s.get("t"))} for s in snaps],
    }
    done_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_kw = {}
    for k, v in methods.items():
        if "i" in v and v["i"] is not None:
            save_kw[f"i_{k}"] = np.asarray(v["i"], dtype=np.float32)
    if fd_i is not None:
        save_kw["i_fd"] = fd_i.astype(np.float32)
    np.savez_compressed(
        cell / "probe.npz",
        **save_kw,
        task=np.asarray([str(s.get("task", cell_task)) for s in snaps]),
        clip=np.asarray([str(s.get("clip")) for s in snaps]),
        seed=np.asarray([int(s.get("seed")) for s in snaps], dtype=np.int32),
        t=np.asarray([int(s.get("t")) for s in snaps], dtype=np.int32),
        lam=np.asarray([float(s.get("lam", cell_lam)) for s in snaps], dtype=np.float32),
    )
    print(f"[ucr1d] wrote {done_p} n={len(snaps)}", flush=True)
    return payload


def _run_terrain(env_cfg, agent_cfg, resume_path: str, motion_dir: str, terrain: str, out_root: Path) -> dict:
    cell = out_root / terrain
    cell.mkdir(parents=True, exist_ok=True)
    if bool(getattr(args_cli, "ucr1_heldout", False)):
        return _ucr1_heldout_cell(env_cfg, agent_cfg, resume_path, motion_dir, terrain, cell)
    if bool(getattr(args_cli, "ucr1d_probe", False)):
        return _ucr1d_probe_cell(env_cfg, agent_cfg, resume_path, motion_dir, terrain, cell)
    done_p = cell / "summary.json"
    clones_p = cell / "clones.npz"
    proc_p = cell / "process.npz"
    if bool(getattr(args_cli, "ucr2p_dump", False)) and proc_p.exists():
        print(f"[ucr2p] skip existing {proc_p}", flush=True)
        return json.loads(done_p.read_text()) if done_p.exists() else {"skipped": True, "reason": "process.npz"}
    if bool(getattr(args_cli, "ucr1_dump", False)) and clones_p.exists():
        print(f"[ucr1] skip existing {clones_p}", flush=True)
        return json.loads(done_p.read_text()) if done_p.exists() else {"skipped": True, "reason": "clones.npz"}
    if done_p.exists() and not bool(getattr(args_cli, "ucr1_dump", False)) and not bool(
        getattr(args_cli, "ucr2p_dump", False)
    ):
        print(f"[t1] skip existing {done_p}", flush=True)
        return json.loads(done_p.read_text())

    task = str(args_cli.task_source)
    rec_kind = str(args_cli.rec_kind)
    env, runner, policy, injector = _make_env(env_cfg, agent_cfg, resume_path, motion_dir, terrain)
    warp = getattr(env.unwrapped, "_t13_warp_info", {}) or {}
    if (
        bool(getattr(args_cli, "ucr1_dump", False)) or bool(getattr(args_cli, "ucr2p_dump", False))
    ) and (bool(warp.get("out_of_plausible_range")) or bool(warp.get("kinematic_impossible"))):
        reason = "out_of_plausible_range" if warp.get("out_of_plausible_range") else "kinematic_impossible"
        print(f"[ucr] skip implausible cell {task} λ={args_cli.lambda_id} ({reason})", flush=True)
        payload = {"skipped": True, "reason": reason, "HUMAN_INTENT_OUT_OF_PLAUSIBLE_RANGE": bool(warp.get("out_of_plausible_range")),
                   "KINEMATIC_IMPOSSIBLE": bool(warp.get("kinematic_impossible"))}
        if bool(getattr(args_cli, "ucr1_dump", False)):
            meta = {"n": 0, "skipped": True, "reason": reason, "warp": {k: warp.get(k) for k in (
                "out_of_plausible_range", "kinematic_impossible", "s_t", "s_x", "lambda_id")}}
            _write_ucr0_empty(clones_p, meta)
        if bool(getattr(args_cli, "ucr2p_dump", False)):
            np.savez_compressed(proc_p, n=0, skipped=True, reason=reason)
        done_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
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
    all_ufr: list[dict] = []
    all_ura: list[dict] = []
    all_ucr: list[dict] = []
    log_adapter = rec_kind in ("shared_adapt", "shadow")
    for seed in seeds:
        print(
            f"[t1] ===== kind={rec_kind} task={task} s={int(s_enabled)} "
            f"terrain={terrain} seed={seed} =====",
            flush=True,
        )
        torch.manual_seed(seed)
        np.random.seed(seed)
        events, ep_rows, shadow_rows, z_rows, ufr, ura_snaps, ucr_snaps = _rollout_cl(
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
        if ufr:
            all_ufr.extend(ufr)
        if ura_snaps:
            all_ura.extend(ura_snaps)
        if ucr_snaps:
            all_ucr.extend(ucr_snaps)
        print(
            f"[t1] seed={seed} events={len(events)} bursts={sum(e['n_bursts'] for e in events)} "
            f"shadow={len(shadow_rows)} z={len(z_rows)} ura_snaps={len(ura_snaps)} ucr_snaps={len(ucr_snaps)}",
            flush=True,
        )

    if getattr(args_cli, "ura0_clone", False) and all_ura:
        asset = env.unwrapped.scene["robot"]
        pelvis_i = list(asset.data.body_names).index("pelvis") if "pelvis" in asset.data.body_names else None
        _ura0_clone_pairs(
            env,
            policy,
            runner,
            injector,
            all_ura,
            vis_i,
            vis_flags,
            adapter,
            mua,
            sda,
            bins,
            pelvis_i,
            float(args_cli.lam_fail),
            cell / "clones.npz",
        )

    if getattr(args_cli, "ucr0_clone", False) and not bool(getattr(args_cli, "ucr2p_dump", False)):
        asset = env.unwrapped.scene["robot"]
        pelvis_i = list(asset.data.body_names).index("pelvis") if "pelvis" in asset.data.body_names else None
        cap = int(getattr(args_cli, "ucr0_max_snaps", 48))
        snaps = all_ucr[:cap]
        print(f"[ucr0] cloning n={len(snaps)} (collected={len(all_ucr)} cap={cap})", flush=True)
        _ucr0_clone_pairs(
            env,
            policy,
            runner,
            injector,
            snaps,
            vis_i,
            vis_flags,
            adapter,
            mua,
            sda,
            bins,
            pelvis_i,
            float(args_cli.lam_fail),
            cell / "clones.npz",
        )

    recov = _agg_events(all_events)
    mon = _ep_monitor(all_eps)
    warp = getattr(env.unwrapped, "_t13_warp_info", {})
    payload = {
        "step": "T1.3",
        "no_ppo": True,
        "success_predicate": {
            "source": "tracking_env_cfg.TerminationsCfg + fall_to_ground diagnostic",
            "fail": "fall (torso/pelvis z<0.4) OR bad_anchor_pos_z_only_terrain_rel(0.25) OR bad_anchor_ori(0.8)",
            "success": "episode reaches 400-step eval horizon without those failures",
            "not_used": ["SR@5", "SR@2", "POI", "obstacle_reach_succeeded", "ee_body_pos"],
            "sr_5cm_role": "secondary sparse-intent fidelity, never SR_task",
            "sr_2cm_role": "secondary sparse-intent fidelity, never SR_task",
            "poi_role": "visible-point tracking (torso/L-wrist/R-wrist); visualization only",
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
        "sr_2cm": mon["sr_2cm"],
        "poi_err": mon["poi_err"],
        "poi": mon.get("poi"),
        "fail_frac": mon["fail_frac"],
        "fall_frac": mon["fall_frac"],
        "task_completion": mon["sr_task"],
        "monitor": recov,
        "episode_monitor": mon,
        "episodes": all_eps,
        "events": all_events,
        "n_shadow_rows": len(all_shadow),
        "n_z_samples": len(all_z),
        "stress": _stress_meta(),
        "time_warp": warp,
        "intent_speed": float(args_cli.intent_speed),
        "intent_spatial": float(args_cli.intent_spatial),
        "lambda_id": float(args_cli.lambda_id),
        "clip_end_rule": "hold_final_valid_human_intent",
        "HUMAN_INTENT_OUT_OF_PLAUSIBLE_RANGE": bool((warp or {}).get("out_of_plausible_range")),
        "KINEMATIC_IMPOSSIBLE": bool((warp or {}).get("kinematic_impossible")),
        "temporal_stress_informative": bool((warp or {}).get("temporal_stress_informative", True)),
        "n_fail_active": mon.get("n_fail_active"),
        "n_fail_hold": mon.get("n_fail_hold"),
        "ucr2p_dump": bool(getattr(args_cli, "ucr2p_dump", False)),
        "n_ucr_snaps": len(all_ucr),
    }
    if bool(getattr(args_cli, "ucr2p_dump", False)):
        _write_ucr2p_process(cell / "process.npz", all_ucr)
        torch.save(all_ucr, cell / "snaps.pt")
    (cell / "events.json").write_text(json.dumps(_sanitize(all_events), indent=2), encoding="utf-8")
    (cell / "episodes.json").write_text(json.dumps(_sanitize(all_eps), indent=2), encoding="utf-8")
    (cell / "summary.json").write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    (cell / "human_kinematics.json").write_text(
        json.dumps(_sanitize((warp or {}).get("kinematics") or {}), indent=2), encoding="utf-8"
    )
    if (warp or {}).get("example"):
        (cell / "example_trajectory.json").write_text(
            json.dumps(_sanitize(warp.get("example")), indent=2), encoding="utf-8"
        )
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
    if all_ufr:
        _write_ufr_npz(
            cell / ("ufr2_steps.npz" if getattr(args_cli, "ufr2_dump", False) else "ufr_steps.npz"),
            all_ufr,
            {
                "task": task,
                "terrain": terrain,
                "mask": mask_name,
                "s_enabled": bool(s_enabled),
                "intent_speed": float(args_cli.intent_speed),
                "intent_spatial": float(args_cli.intent_spatial),
                "lambda_id": float(args_cli.lambda_id),
                "n_episodes": len(all_ufr),
                "feat_names": list(UFR_FEAT_NAMES),
                "no_task_id_in_feat": True,
                "shadow_only": rec_kind == "shadow",
                "ufr2": bool(getattr(args_cli, "ufr2_dump", False)),
                "HUMAN_INTENT_OUT_OF_PLAUSIBLE_RANGE": bool((warp or {}).get("out_of_plausible_range")),
                "KINEMATIC_IMPOSSIBLE": bool((warp or {}).get("kinematic_impossible")),
            },
        )
    print(
        f"[t13] {task}/{rec_kind}/{terrain} SR_task={mon['sr_task']:.3f} "
        f"SR@5={mon['sr_5cm']:.3f} SR@2={mon['sr_2cm']:.3f} "
        f"POI={mon['poi_err']:.3f} fail={mon['fail_frac']:.3f} "
        f"map={mon.get('mapper_agg')} exec={mon.get('exec_e')}",
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
            "sr_2cm": cell.get("sr_2cm"),
            "poi_err": cell.get("poi_err"),
            "poi": cell.get("poi"),
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
        "step": "T1",
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
        "sr_2cm_all": mon_all["sr_2cm"],
        "poi_err_all": mon_all["poi_err"],
        "poi_all": mon_all.get("poi"),
        "fail_frac_all": mon_all["fail_frac"],
        "fall_frac_all": mon_all["fall_frac"],
        "episode_monitor_all": mon_all,
        "recovery_all": recov_all,
        "by_terrain": by_terrain,
        "note": "T1 failure-stress. SR_task = no official fail termination. No PPO.",
    }
    (out_root / "pooled.json").write_text(json.dumps(_sanitize(pooled), indent=2), encoding="utf-8")
    print(
        f"[t1] POOLED {args_cli.task_source}/{args_cli.rec_kind} n_ep={len(all_eps)} "
        f"SR_task={mon_all['sr_task']:.3f} SR@5={mon_all['sr_5cm']:.3f} "
        f"SR@2={mon_all['sr_2cm']:.3f} POI={mon_all['poi_err']:.3f} "
        f"fail={mon_all['fail_frac']:.3f}",
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
    print(f"[t1] ckpt {resume_path}", flush=True)
    print(
        f"[t1] task={args_cli.task_source} kind={args_cli.rec_kind} "
        f"family={args_cli.stress_family} payload={args_cli.payload_kg} "
        f"com=({args_cli.com_x},{args_cli.com_y},{args_cli.com_z}) "
        f"ws=({args_cli.ws_fwd},{args_cli.ws_lat},{args_cli.ws_z}) "
        f"push_dv={args_cli.push_dv} NO PPO NO TRAIN",
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
        print(f"[t1] ===== {args_cli.task_source}/{args_cli.rec_kind}/{terrain} =====", flush=True)
        _run_terrain(env_cfg, agent_cfg, resume_path, motion_dir, terrain, out_dir)
        _write_pooled(out_dir, resume_path)
    print("[t1] done", flush=True)


if __name__ == "__main__":
    main()
    # Isaac Kit close() can spin for hours after a successful held-out write.
    if bool(getattr(args_cli, "ucr1_heldout", False)) or bool(getattr(args_cli, "ucr1_dump", False)) or bool(
        getattr(args_cli, "ucr1d_probe", False)
    ) or bool(getattr(args_cli, "ucr2p_dump", False)):
        os._exit(0)
    simulation_app.close()
