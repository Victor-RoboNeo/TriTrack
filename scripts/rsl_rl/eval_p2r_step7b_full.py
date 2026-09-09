#!/usr/bin/env python3
"""Step 7B-Full — 50 Hz process-state clones + paired I_b. No PPO. No 6V.

CL2 live trunk. Snapshot primarily burst 2/3/4 (Slope/Steps). Each clone
stores last 15 control frames (300 ms) of realized recovery trajectory:
E, e, ė, z_nom, d, T_rec, root vel, ankle contact. No terrain/depth.

Paired probe identical to 6S-R: Parent 0.5 s vs 5°/100 ms then Parent.
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
HORIZON = 25
THETA_DEG = 5.0
# 100 ms, ~250 ms (13 frames = 260 ms, same as Step-3 dE25), 500 ms
H_MARKS = (("I_100", 5), ("I_250", 13), ("I_500", 25))
TERRAINS = ("plane", "light_rough", "slope", "steps")
TERRAIN_LABEL = {
    "plane": "Flat",
    "light_rough": "Light",
    "slope": "Slope",
    "steps": "Steps",
}
KP_VIS = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
PARENT_CKPT = (
    "/data/home/chenxiangyu/robotics/Anybody/logs/rsl_rl/"
    "g1_flat_muse_kp_latent_rl/2026-08-26_00-38-10_tritrack_headhands_locomani_from35000/"
    "model_50000.pt"
)
MAPPER_B = "/data/home/chenxiangyu/victor/TriTrack/runs/mapper_b_intent72/mapper_best.pt"
CLIP_LOCO = "/data/home/chenxiangyu/robotics/Anybody/logs/tritrack/infer_clips/p1/loco"
STEP6S_CKPT = "/data/home/chenxiangyu/robotics/Anybody/results/p2r_step6s_full/model_best.pt"
OFFLINE_P = {"all": 0.78, "quality": 0.84, "exact": 0.91}
FAIL_ANCHOR_Z = 0.25
FAIL_ANCHOR_ORI = 0.8
BURST_GROUPS = ("2", "3", "4")
ROUGH_CAPS = {"2": 140, "3": 140, "4": 140}
FLAT_CAPS = {"2": 30, "3": 30, "4": 30}
TRAJ_LEN = 15  # 300 ms @ 50 Hz
TRAJ_DIM = 62  # E,RE,e9,edot,z,d,in_burst,T_rec_n,root_lin,root_ang,cL,cR
ANKLE_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _burst_group(b: int) -> str:
    b = int(b)
    if b in (2, 3, 4):
        return str(b)
    return "other"


def _caps_for(terrain: str) -> dict[str, int]:
    return dict(ROUGH_CAPS if terrain in ("slope", "steps") else FLAT_CAPS)


def project_tangent(d: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z = F.normalize(z, dim=-1, eps=1e-8)
    d = d - (d * z).sum(-1, keepdim=True) * z
    return F.normalize(d, dim=-1, eps=1e-8)


def apply_angle(z_nom: torch.Tensor, d_raw: torch.Tensor, theta_deg: float) -> torch.Tensor:
    d = project_tangent(d_raw, z_nom)
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

parser = argparse.ArgumentParser(description="P2-R Step 7B-Full 50Hz process-state clones.")
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--task", type=str, default="MUSE-Kp-LatentRL-Kp5-HeadHands-Locomani-G1-v0")
parser.add_argument("--motion", type=str, default=CLIP_LOCO)
parser.add_argument("--mask_modes", type=str, default="torso")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--out", type=str, default="results/p2r_step7b_full")
parser.add_argument("--start_frame", type=int, default=10)
parser.add_argument("--hint_iter", type=int, default=50000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51,52,53,54,55,56")
parser.add_argument("--terrain", type=str, default="plane", choices=TERRAINS)
parser.add_argument("--terrains", type=str, default="plane")
parser.add_argument("--keep_headhands_spec", action="store_true", default=True)
parser.add_argument("--mapper_path", type=str, default=MAPPER_B)
parser.add_argument("--mapper_b_path", type=str, default=MAPPER_B)
parser.add_argument("--mapper_a_path", type=str, default="")
parser.add_argument("--residual_alpha", type=float, default=1.0)
parser.add_argument("--step6s_ckpt", type=str, default=STEP6S_CKPT)
parser.add_argument("--theta_deg", type=float, default=THETA_DEG)
parser.add_argument("--burst_steps", type=int, default=BURST_STEPS)
parser.add_argument("--max_states", type=int, default=0, help="0 = use per-group caps")
parser.add_argument(
    "--dump_obs_only",
    action="store_true",
    help="Collect CL2 burst states and write rec_obs.npz; skip paired probe.",
)
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
        print("[7b-full] terrain=plane", flush=True)
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
    print(f"[7b-full] terrain={terrain} 1x1 generator", flush=True)


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
                print(f"[7b-full] pin mask {mask_name!r} idx={names.index(mask_name)}", flush=True)
        cmd._resample_command(ids)
        if hasattr(cmd, "resample_all_mask_modes"):
            cmd.resample_all_mask_modes()
        try:
            env.unwrapped.sim.forward()
        except Exception as exc:
            print(f"[7b-full] sim.forward skipped: {exc}", flush=True)
    return use, [str(p) for p in paths[:use]]


def _visible_e(cmd, vis_i, vis_flags) -> torch.Tensor:
    delta = cmd.robot_body_pos_w - cmd.body_pos_w
    parts = [torch.linalg.norm(delta[:, vis_i[k]], dim=-1) for k, flag in enumerate(vis_flags) if flag]
    stacked = torch.stack(parts, dim=-1)
    return stacked.mean(dim=-1)


def _ankle_contact(env, n: int) -> torch.Tensor:
    """Binary L/R ankle contact from existing ContactSensor. Zeros if missing."""
    uw = env.unwrapped
    device = uw.device
    out = torch.zeros(n, 2, device=device, dtype=torch.float32)
    if "contact_forces" not in uw.scene.keys():
        return out
    cs = uw.scene["contact_forces"]
    try:
        f = cs.data.net_forces_w  # [N, B, 3]
        names = list(cs.body_names)
        for k, name in enumerate(ANKLE_NAMES):
            if name in names:
                j = names.index(name)
                out[:, k] = (f[:n, j].norm(dim=-1) > 1.0).to(torch.float32)
    except Exception:
        pass
    return out


def _root_vel(env, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.unwrapped.scene["robot"]
    return robot.data.root_lin_vel_w[:n].detach(), robot.data.root_ang_vel_w[:n].detach()


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


def _load_6s_predictor(ckpt_path: str, device):
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from train_p2r_step6s import SupervisedRecoveryMLP, _project_unit

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SupervisedRecoveryMLP()
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mu = torch.as_tensor(blob["obs_mean"], device=device, dtype=torch.float32)
    sd = torch.as_tensor(blob["obs_std"], device=device, dtype=torch.float32)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    return model, mu, sd, _project_unit


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


def _restore_batch(env, snaps: list[dict]) -> int:
    n = len(snaps)
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


def _pad_snaps(snaps: list[dict], n_envs: int) -> tuple[list[dict], int]:
    if not snaps:
        return snaps, 0
    n = len(snaps)
    if n >= n_envs:
        return snaps[:n_envs], n_envs
    return list(snaps) + [snaps[-1]] * (n_envs - n), n


def _caps_full(counts: dict[str, int], caps: dict[str, int]) -> bool:
    return all(int(counts.get(g, 0)) >= int(caps.get(g, 0)) for g in BURST_GROUPS)


@torch.inference_mode()
def _collect_cl2_bursts(
    env,
    policy,
    runner,
    injector,
    rec_model,
    obs_mu,
    obs_sd,
    *,
    steps: int,
    terrain: str,
    seed: int,
    clip_paths: list[str],
    vis_i,
    vis_flags,
    theta_deg: float,
    burst_len: int,
    caps: dict[str, int],
    counts: dict[str, int],
) -> list[dict]:
    """CL2 live. 50 Hz event traj; clone burst 2/3/4 if quota open."""
    n_envs = int(env.unwrapped.num_envs)
    use = min(n_envs, len(clip_paths) if clip_paths else n_envs)
    device = env.unwrapped.device
    cmd = env.unwrapped.command_manager.get_term("motion")
    asset = env.unwrapped.scene["robot"]
    gravity = asset.data.GRAVITY_VEC_W
    if gravity.ndim == 1:
        gravity = gravity.unsqueeze(0).expand(cmd.num_envs, -1)
    gate = RecoveryRiskGate(s_enabled=False)
    active_prev = torch.zeros(n_envs, dtype=torch.bool, device=device)
    burst_left = torch.zeros(n_envs, dtype=torch.long, device=device)
    held_d = torch.zeros(n_envs, 16, dtype=torch.float32, device=device)
    failed = torch.zeros(n_envs, dtype=torch.bool, device=device)
    e9_prev = None
    events: list[dict] = []
    cur: list[dict | None] = [None] * use
    snaps: list[dict] = []
    traj_buf: list[list[np.ndarray]] = [[] for _ in range(use)]
    event_seq = 0
    for t in range(steps):
        if _caps_full(counts, caps):
            break
        obs, _ = env.get_observations()
        obs = injector.patch_policy_obs(obs, env, "mapper")
        e_pre = _visible_e(cmd, vis_i, vis_flags)
        gout = gate.step(e_pre, torch.zeros_like(e_pre))
        active = gout["active"]
        z_nom, proprio = _z_and_proprio(policy, runner, obs)
        rec_pack, e9 = _recovery_pack(obs, z_nom, e9_prev)
        rising = active & ~active_prev
        falling = (~active) & active_prev
        for i in range(use):
            if failed[i]:
                continue
            if bool(rising[i]) and t >= WARMUP:
                event_seq += 1
                ev = {
                    "released": False,
                    "failed": False,
                    "fail_after": False,
                    "n_bursts": 0,
                    "event_id": int(event_seq),
                    "t_event0": int(t),
                    "last_dE": 0.0,
                    "last_dR": 0.0,
                    "E_pre_burst": float(e_pre[i].item()),
                    "R_pre_burst": float(gout["R_E"][i].item()),
                }
                events.append(ev)
                cur[i] = ev
                traj_buf[i] = []
            if cur[i] is not None and (not cur[i]["released"]) and (not cur[i]["failed"]) and bool(falling[i]):
                cur[i]["released"] = True
        has_open = torch.zeros(n_envs, dtype=torch.bool, device=device)
        for i in range(use):
            ev = cur[i]
            if ev is not None and (not ev["released"]) and (not ev["failed"]):
                has_open[i] = True
        in_burst_now = burst_left > 0
        root_lin, root_ang = _root_vel(env, use)
        contact = _ankle_contact(env, use)
        e_dot = rec_pack[:, 9:18]
        vis = rec_pack[:, 18:21]
        for i in range(use):
            if not bool(has_open[i]):
                continue
            ev = cur[i]
            t_rec = (t - int(ev["t_event0"])) * DT
            t_rec_n = min(t_rec / 2.0, 1.0)
            frame = np.concatenate(
                [
                    np.array(
                        [float(e_pre[i].item()), float(gout["R_E"][i].item())],
                        dtype=np.float32,
                    ),
                    e9[i].detach().cpu().numpy().astype(np.float32),
                    e_dot[i].detach().cpu().numpy().astype(np.float32),
                    z_nom[i].detach().cpu().numpy().astype(np.float32),
                    held_d[i].detach().cpu().numpy().astype(np.float32),
                    np.array(
                        [
                            float(in_burst_now[i].item()),
                            t_rec_n,
                            *root_lin[i].detach().cpu().numpy().tolist(),
                            *root_ang[i].detach().cpu().numpy().tolist(),
                            float(contact[i, 0].item()),
                            float(contact[i, 1].item()),
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
            if int(frame.shape[0]) != TRAJ_DIM:
                raise RuntimeError(f"traj frame dim {frame.shape[0]} != {TRAJ_DIM}")
            traj_buf[i].append(frame)
            if len(traj_buf[i]) > TRAJ_LEN:
                traj_buf[i] = traj_buf[i][-TRAJ_LEN:]
        need = has_open & (burst_left == 0) & (~failed)
        idxs = need.nonzero(as_tuple=False).view(-1).tolist()
        if idxs:
            x = (rec_pack - obs_mu) / obs_sd
            raw = rec_model(x)
            nrm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            d_unit = raw / nrm
            hist = None
            for i in idxs:
                d_prev = held_d[i].detach().cpu().clone()
                cur[i]["n_bursts"] += 1
                bidx = int(cur[i]["n_bursts"])
                grp = _burst_group(bidx)
                if grp in caps and int(counts.get(grp, 0)) < int(caps.get(grp, 0)):
                    if hist is None:
                        hist = _hist_snapshot(env)
                    seq = traj_buf[i]
                    traj = np.zeros((TRAJ_LEN, TRAJ_DIM), dtype=np.float32)
                    nseq = min(len(seq), TRAJ_LEN)
                    if nseq:
                        traj[-nseq:] = np.stack(seq[-nseq:], axis=0)
                    extra = {
                        "obs": obs[i].detach().cpu().clone(),
                        "z_nom": z_nom[i].detach().cpu().clone(),
                        "rec_obs": rec_pack[i].detach().cpu().clone(),
                        "E0": float(e_pre[i].item()),
                        "R_E": float(gout["R_E"][i].item()),
                        "t0": int(t),
                        "terrain": terrain,
                        "seed": int(seed),
                        "clip": Path(clip_paths[i]).name if i < len(clip_paths) else f"env{i}",
                        "dt": 0,
                        "burst_index": bidx,
                        "burst_group": grp,
                        "event": cur[i],
                        "event_id": int(cur[i]["event_id"]),
                        "t_event0": int(cur[i]["t_event0"]),
                        "T_rec": (t - int(cur[i]["t_event0"])) * DT,
                        "d_prev": d_prev.numpy().astype(np.float32),
                        "d_now": d_unit[i].detach().cpu().numpy().astype(np.float32),
                        "dE_prev": float(cur[i].get("last_dE", 0.0)),
                        "dR_prev": float(cur[i].get("last_dR", 0.0)),
                        "traj": traj,
                        "traj_len": int(nseq),
                    }
                    snaps.append(_capture_snap(env, i, hist, extra))
                    counts[grp] = int(counts.get(grp, 0)) + 1
                cur[i]["E_pre_burst"] = float(e_pre[i].item())
                cur[i]["R_pre_burst"] = float(gout["R_E"][i].item())
                held_d[i] = d_unit[i]
                burst_left[i] = int(burst_len)
        in_burst = burst_left > 0
        z_exec = z_nom.clone()
        if bool(in_burst.any()):
            z_exec[in_burst] = apply_angle(z_nom[in_burst], held_d[in_burst], float(theta_deg))
        joints = _decode(policy, z_exec, proprio)
        env.step(joints)
        e_post = _visible_e(cmd, vis_i, vis_flags)
        offset = _ankle_mean_z_offset(cmd)[:use]
        adj_z = (cmd.anchor_pos_w[:use, -1] - cmd.robot_anchor_pos_w[:use, -1]) + offset
        fail_z = adj_z.abs() > FAIL_ANCHOR_Z
        mot_g = quat_rotate_inverse(cmd.anchor_quat_w, gravity)[:use]
        rob_g = quat_rotate_inverse(cmd.robot_anchor_quat_w, gravity)[:use]
        fail_ori = (mot_g[:, 2] - rob_g[:, 2]).abs() > FAIL_ANCHOR_ORI
        finishing = (burst_left == 1) & in_burst
        burst_left = torch.where(in_burst, burst_left - 1, burst_left)
        for i in range(use):
            if bool(finishing[i]) and cur[i] is not None:
                cur[i]["last_dE"] = float(e_post[i].item()) - float(cur[i]["E_pre_burst"])
                r_post = (float(e_post[i].item()) - Q50_E) / max(Q90_E - Q50_E, 1e-8)
                cur[i]["last_dR"] = r_post - float(cur[i]["R_pre_burst"])
            if (not bool(failed[i])) and t >= WARMUP:
                if bool(fail_z[i]) or bool(fail_ori[i]):
                    failed[i] = True
                    burst_left[i] = 0
                    if cur[i] is not None and (not cur[i]["released"]):
                        cur[i]["failed"] = True
                        cur[i]["fail_after"] = True
        e9_prev = e9.detach()
        active_prev = active.clone()
        if (t + 1) % 50 == 0:
            print(
                f"[7b-full] collect {terrain} s{seed} step {t+1}/{steps} "
                f"snaps={len(snaps)} caps={counts}",
                flush=True,
            )
    for s in snaps:
        ev = s.pop("event", {}) or {}
        s["released"] = bool(ev.get("released"))
        s["fail_after"] = bool(ev.get("fail_after"))
        s["n_bursts_event"] = int(ev.get("n_bursts", 0))
    print(
        f"[7b-full] collected {len(snaps)} burst-states terrain={terrain} seed={seed} counts={counts}",
        flush=True,
    )
    return snaps


@torch.inference_mode()
def _rollout_hist(
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
    """E after each of ``horizon`` steps. Shape [horizon, n_valid]. d_raw None = Parent."""
    n_envs = int(env.unwrapped.num_envs)
    padded, _ = _pad_snaps(snaps, n_envs)
    _restore_batch(env, padded)
    cmd = env.unwrapped.command_manager.get_term("motion")
    device = env.unwrapped.device
    obs_stack = torch.stack([s["obs"].to(device) for s in padded], dim=0)
    e_hist = []
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
        e_hist.append(_visible_e(cmd, vis_i, vis_flags)[:n_valid].detach())
    return torch.stack(e_hist, dim=0)


def _i_block(i_cm: np.ndarray) -> dict:
    if i_cm.size == 0:
        return {
            "n": 0,
            "P_I_lt_0": float("nan"),
            "P_I_lt_1cm": float("nan"),
            "P_I_lt_2cm": float("nan"),
            "median_cm": float("nan"),
            "mean_cm": float("nan"),
            "I_cm": _pct([]),
        }
    return {
        "n": int(i_cm.size),
        "P_I_lt_0": _rate(i_cm < 0.0),
        "P_I_lt_1cm": _rate(i_cm < -1.0),
        "P_I_lt_2cm": _rate(i_cm < -2.0),
        "median_cm": float(np.median(i_cm)),
        "mean_cm": float(i_cm.mean()),
        "I_cm": _pct(i_cm),
    }


def _summarize_rows(rows: list[dict], nest: bool = True) -> dict:
    if not rows:
        return {"n": 0}
    out: dict = {"n": len(rows)}
    for name, _h in H_MARKS:
        arr = np.array([r[name + "_cm"] for r in rows], dtype=np.float64)
        out[name] = _i_block(arr)
    if nest:
        by_g = {}
        for g in BURST_GROUPS:
            rs = [r for r in rows if str(r.get("burst_group", _burst_group(int(r.get("burst_index", 1))))) == g]
            if rs:
                by_g[g] = _summarize_rows(rs, nest=False)
        if by_g:
            out["by_burst_group"] = by_g
        terrains_present = sorted({r["terrain"] for r in rows})
        if len(terrains_present) > 1:
            out["by_terrain"] = {
                ter: _summarize_rows([r for r in rows if r["terrain"] == ter], nest=True)
                for ter in terrains_present
            }
    return out


def _verdict(metrics: dict) -> dict:
    by = metrics.get("by_burst_group") or {}
    p = {g: float((by.get(g) or {}).get("I_500", {}).get("P_I_lt_0", float("nan"))) for g in BURST_GROUPS}
    m = {g: float((by.get(g) or {}).get("I_500", {}).get("median_cm", float("nan"))) for g in BURST_GROUPS}
    return {
        "case": "7B-Full-collect",
        "next": "Event-level 5-fold CV on 50Hz traj. GO 7-CL iff regret stably < 0.671 cm.",
        "P_I500": p,
        "median_I500_cm": m,
    }


@torch.inference_mode()
def _eval_snaps(
    env,
    policy,
    runner,
    injector,
    snaps: list[dict],
    vis_i,
    vis_flags,
    rec_model,
    obs_mu,
    obs_sd,
    project_unit,
    theta_deg: float,
    burst: int,
) -> list[dict]:
    n_envs = int(env.unwrapped.num_envs)
    device = env.unwrapped.device
    rows: list[dict] = []
    for start in range(0, len(snaps), n_envs):
        batch = snaps[start : start + n_envs]
        n_valid = len(batch)
        print(f"[7b-full] probe batch {start}:{start + n_valid} / {len(snaps)}", flush=True)
        rec = torch.stack([s["rec_obs"] for s in batch], dim=0).to(device=device, dtype=torch.float32)
        z0 = torch.stack([s["z_nom"] for s in batch], dim=0).to(device=device, dtype=torch.float32)
        x = (rec - obs_mu) / obs_sd
        d_pred = project_unit(rec_model(x), z0)
        d_pad = torch.zeros(n_envs, 16, device=device, dtype=d_pred.dtype)
        d_pad[:n_valid] = d_pred
        e_p = _rollout_hist(
            env, policy, runner, injector, batch, n_valid,
            d_raw=None, theta_deg=0.0, burst=0, horizon=HORIZON,
            vis_i=vis_i, vis_flags=vis_flags,
        )
        e_r = _rollout_hist(
            env, policy, runner, injector, batch, n_valid,
            d_raw=d_pad, theta_deg=float(theta_deg), burst=int(burst), horizon=HORIZON,
            vis_i=vis_i, vis_flags=vis_flags,
        )
        for i, s in enumerate(batch):
            e0 = float(s["E0"])
            row = {
                "terrain": s["terrain"],
                "seed": int(s["seed"]),
                "clip": s["clip"],
                "t0": int(s["t0"]),
                "dt": 0,
                "E0_cm": e0 * 100.0,
                "R_E": float(s["R_E"]),
                "burst_index": int(s.get("burst_index", 1)),
                "burst_group": str(s.get("burst_group", _burst_group(int(s.get("burst_index", 1))))),
                "released": bool(s.get("released", False)),
                "fail_after": bool(s.get("fail_after", False)),
                "n_bursts_event": int(s.get("n_bursts_event", 0)),
                "event_id": int(s.get("event_id", 0)),
                "t_event0": int(s.get("t_event0", s.get("t0", 0))),
                "T_rec": float(s.get("T_rec", 0.0)),
                "dE_prev": float(s.get("dE_prev", 0.0)),
                "dR_prev": float(s.get("dR_prev", 0.0)),
            }
            for name, h in H_MARKS:
                ep = float(e_p[h - 1, i].item())
                er = float(e_r[h - 1, i].item())
                row[f"E_P_{name[2:]}_cm"] = ep * 100.0
                row[f"E_R_{name[2:]}_cm"] = er * 100.0
                row[name + "_cm"] = (er - ep) * 100.0
            row["dE_R_100_cm"] = (float(e_r[4, i].item()) - e0) * 100.0
            row["dE_P_100_cm"] = (float(e_p[4, i].item()) - e0) * 100.0
            row["dE_R_500_cm"] = (float(e_r[24, i].item()) - e0) * 100.0
            row["dE_P_500_cm"] = (float(e_p[24, i].item()) - e0) * 100.0
            rows.append(row)
        p500 = _rate(np.array([r["I_500_cm"] for r in rows]) < 0)
        print(
            f"[7b-full] running n={len(rows)} P(I_500<0)={p500:.3f} "
            f"median I_500={np.median([r['I_500_cm'] for r in rows]):.2f}cm",
            flush=True,
        )
    return rows


def _print_group_table(rows: list[dict], title: str) -> None:
    print(f"[7b-full] {title}", flush=True)
    print(
        f"{'group':<8} {'n':>5} {'P(I<0)':>8} {'P(I<-1cm)':>11} {'med_cm':>8} {'p25':>8} {'p75':>8}",
        flush=True,
    )
    by = {}
    for r in rows:
        by.setdefault(str(r.get("burst_group", "?")), []).append(r)
    for g in list(BURST_GROUPS) + [k for k in by if k not in BURST_GROUPS]:
        rs = by.get(g) or []
        if not rs:
            print(f"{g:<8} {0:5d}", flush=True)
            continue
        arr = np.array([r["I_500_cm"] for r in rs], dtype=np.float64)
        pct = _pct(arr)
        print(
            f"{g:<8} {len(rs):5d} {_rate(arr < 0):8.3f} {_rate(arr < -1.0):11.3f} "
            f"{float(np.median(arr)):8.2f} {pct['p25']:8.2f} {pct['p75']:8.2f}",
            flush=True,
        )


def _as_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _write_process_npz(path: Path, snaps: list[dict], labels: dict | None = None) -> None:
    kw = {
        "rec_obs": np.stack([_as_np(s["rec_obs"]) for s in snaps], axis=0).astype(np.float32),
        "traj": np.stack([np.asarray(s["traj"], dtype=np.float32) for s in snaps], axis=0),
        "traj_len": np.array([int(s["traj_len"]) for s in snaps], dtype=np.int32),
        "E0": np.array([float(s["E0"]) for s in snaps], dtype=np.float32),
        "R_E": np.array([float(s["R_E"]) for s in snaps], dtype=np.float32),
        "burst_index": np.array([int(s["burst_index"]) for s in snaps], dtype=np.int32),
        "t0": np.array([int(s["t0"]) for s in snaps], dtype=np.int32),
        "seed": np.array([int(s["seed"]) for s in snaps], dtype=np.int32),
        "event_id": np.array([int(s["event_id"]) for s in snaps], dtype=np.int32),
        "t_event0": np.array([int(s["t_event0"]) for s in snaps], dtype=np.int32),
        "T_rec": np.array([float(s["T_rec"]) for s in snaps], dtype=np.float32),
        "dE_prev": np.array([float(s["dE_prev"]) for s in snaps], dtype=np.float32),
        "dR_prev": np.array([float(s["dR_prev"]) for s in snaps], dtype=np.float32),
        "d_prev": np.stack([np.asarray(s["d_prev"], dtype=np.float32) for s in snaps], axis=0),
        "d_now": np.stack([np.asarray(s["d_now"], dtype=np.float32) for s in snaps], axis=0),
        "terrain": np.array([str(s["terrain"]) for s in snaps]),
        "clip": np.array([str(s["clip"]) for s in snaps]),
    }
    if labels:
        kw.update(labels)
    np.savez_compressed(path, **kw)


def _run_terrain(env_cfg, agent_cfg, resume_path: str, motion_dir: str, terrain: str, out_root: Path) -> dict:
    cell = out_root / terrain
    cell.mkdir(parents=True, exist_ok=True)
    done_p = cell / "summary.json"
    obs_p = cell / "rec_obs.npz"
    dump_only = bool(getattr(args_cli, "dump_obs_only", False))
    if dump_only and obs_p.exists():
        print(f"[7b-full] skip existing {obs_p}", flush=True)
        return {"terrain": terrain, "n": 0, "dump_obs_only": True}
    if (not dump_only) and done_p.exists():
        print(f"[7b-full] skip existing {done_p}", flush=True)
        return json.loads(done_p.read_text())

    env, runner, policy, injector = _make_env(env_cfg, agent_cfg, resume_path, motion_dir, terrain)
    rec_model, obs_mu, obs_sd, project_unit = _load_6s_predictor(args_cli.step6s_ckpt, env.device)
    cmd = env.unwrapped.command_manager.get_term("motion")
    body_names = list(cmd.cfg.body_names)
    vis_i = [body_names.index(n) for n in KP_VIS]
    vis_flags = (True, False, False)
    seeds = [int(s) for s in args_cli.seeds.split(",") if s.strip()] or [int(args_cli.seed)]
    caps = _caps_for(terrain)
    if int(args_cli.max_states) > 0:
        scale = int(args_cli.max_states) / max(1, sum(caps.values()))
        caps = {g: int(round(v * scale)) for g, v in caps.items()}
    counts = {g: 0 for g in BURST_GROUPS}
    all_snaps: list[dict] = []
    for seed in seeds:
        if _caps_full(counts, caps):
            break
        print(f"[7b-full] ===== collect {terrain} seed={seed} caps={caps} have={counts} =====", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        _use, paths = _pin_clips(env, mask_name="torso")
        snaps = _collect_cl2_bursts(
            env, policy, runner, injector, rec_model, obs_mu, obs_sd,
            steps=int(args_cli.steps),
            terrain=terrain,
            seed=seed,
            clip_paths=paths,
            vis_i=vis_i,
            vis_flags=vis_flags,
            theta_deg=float(args_cli.theta_deg),
            burst_len=int(args_cli.burst_steps),
            caps=caps,
            counts=counts,
        )
        all_snaps.extend(snaps)
    print(f"[7b-full] {terrain} n_states={len(all_snaps)} counts={counts}", flush=True)
    traj_p = cell / "traj.npz"
    if all_snaps:
        _write_process_npz(obs_p, all_snaps)
        _write_process_npz(traj_p, all_snaps)
        print(f"[7b-full] wrote {obs_p} and {traj_p} n={len(all_snaps)}", flush=True)
    if dump_only:
        try:
            env.close()
        except Exception:
            pass
        return {"terrain": terrain, "n": len(all_snaps), "dump_obs_only": True, "obs": str(obs_p)}
    rows = _eval_snaps(
        env, policy, runner, injector, all_snaps, vis_i, vis_flags,
        rec_model, obs_mu, obs_sd, project_unit,
        theta_deg=float(args_cli.theta_deg),
        burst=int(args_cli.burst_steps),
    )
    if all_snaps and rows:
        labels = {
            "I_100_cm": np.array([r["I_100_cm"] for r in rows], dtype=np.float32),
            "I_250_cm": np.array([r["I_250_cm"] for r in rows], dtype=np.float32),
            "I_500_cm": np.array([r["I_500_cm"] for r in rows], dtype=np.float32),
            "E_P_500_cm": np.array([r["E_P_500_cm"] for r in rows], dtype=np.float32),
            "E_R_500_cm": np.array([r["E_R_500_cm"] for r in rows], dtype=np.float32),
        }
        _write_process_npz(obs_p, all_snaps, labels)
        _write_process_npz(traj_p, all_snaps, labels)
    metrics = _summarize_rows(rows)
    payload = {
        "step": "7B-Full",
        "no_ppo": True,
        "dt": 0,
        "theta_deg": float(args_cli.theta_deg),
        "burst_steps": int(args_cli.burst_steps),
        "horizon_steps": HORIZON,
        "traj_len": TRAJ_LEN,
        "traj_dim": TRAJ_DIM,
        "ckpt": resume_path,
        "step6s_ckpt": args_cli.step6s_ckpt,
        "terrain": terrain,
        "seeds": seeds,
        "caps": caps,
        "n": len(rows),
        "n_by_group": {g: int(counts.get(g, 0)) for g in BURST_GROUPS},
        "metrics": metrics,
        "rows": rows,
        "note": "50Hz event traj + paired I_b on burst 2/3/4. No terrain/depth. No PPO.",
    }
    (cell / "rows.json").write_text(json.dumps(_sanitize(rows), indent=2), encoding="utf-8")
    (cell / "summary.json").write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    _print_group_table(rows, f"{terrain} I_500 by burst group")
    try:
        env.close()
    except Exception:
        pass
    return payload


def _write_pooled(out_root: Path, resume_path: str) -> dict | None:
    rows: list[dict] = []
    cells = {}
    for ter in TERRAINS:
        p = out_root / ter / "summary.json"
        if not p.exists():
            continue
        cell = json.loads(p.read_text())
        cells[ter] = cell
        rows.extend(cell.get("rows") or [])
    if not rows:
        return None
    metrics = _summarize_rows(rows)
    gates = _verdict(metrics)
    pooled = {
        "step": "7B-Full",
        "no_ppo": True,
        "dt": 0,
        "ckpt": resume_path,
        "step6s_ckpt": args_cli.step6s_ckpt,
        "terrains": sorted(cells),
        "n": len(rows),
        "n_by_terrain": {t: int(cells[t]["n"]) for t in cells},
        "metrics": metrics,
        "gates": gates,
        "note": "50Hz traj clones burst 2/3/4. I_b = E^R_{t+0.5}-E^P_{t+0.5}. No terrain/depth.",
    }
    (out_root / "pooled.json").write_text(json.dumps(_sanitize(pooled), indent=2), encoding="utf-8")
    _print_group_table(rows, "POOLED I_500 by burst group")
    for ter in ("slope", "steps", "plane", "light_rough"):
        rs = [r for r in rows if r.get("terrain") == ter]
        if rs:
            _print_group_table(rs, f"{ter} I_500 by burst group")
    print(
        f"[7b-full] POOLED n={len(rows)} case={gates['case']} "
        f"P_I500={gates['P_I500']} med={gates['median_I500_cm']} "
        f"next={gates['next']}",
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
    print(f"[7b-full] ckpt {resume_path}", flush=True)
    print("[7b-full] recovery-induced paired I_b. NO PPO. NO 6V. NO gate change.", flush=True)
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
        print(f"[7b-full] ===== {terrain} =====", flush=True)
        _run_terrain(env_cfg, agent_cfg, resume_path, motion_dir, terrain, out_dir)
        if not bool(getattr(args_cli, "dump_obs_only", False)):
            _write_pooled(out_dir, resume_path)
    print("[7b-full] done", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
