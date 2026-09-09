#!/usr/bin/env python3
"""Torso Height Buffer oracle — privileged diagnostic, Parent only, no R-M3.

Not part of the unified method. No terrain ID. Support height is estimated from
two-foot contact z, then torso-z is allowed a bounded upward slack:

    Δh_t = max(0, h_t - h_0)
    δz_T = β Δh_t ∈ [0, Δh_t]
    z_T^exec = z_T^human + δz_T
    wrists stay at original world targets (torso-buffer) or also +Δh (shift-all).

Question: how much of Steps `anchor_z` failure is "support rose, torso world-z glued"?
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

DT = 0.02
WARMUP = 10
H0_FRAMES = 15
CONTACT_N = 10.0
FAIL_ANCHOR_Z = 0.25
FAIL_ANCHOR_ORI = 0.8
FAIL_FALL_Z = 0.4
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
TERRAINS = ("plane", "light_rough", "slope", "steps")
TASK_SOURCES = ("loco", "stoop", "reach", "carry")
S_ENABLED_TASKS = ("stoop", "reach", "carry")
VARIANTS = (
    "original",
    "torso_b025",
    "torso_b050",
    "torso_b075",
    "torso_b100",
    "shift_all",
    "torso_b025_up",
)
ARM_TOKENS = ("shoulder", "elbow", "wrist")


def variant_spec(name: str) -> dict:
    if name == "original":
        return {"mode": "none", "beta": 0.0, "unsigned": False}
    if name == "shift_all":
        return {"mode": "shift_all", "beta": 1.0, "unsigned": False}
    if name == "torso_b025_up":
        return {"mode": "torso", "beta": 0.25, "unsigned": True}
    table = {
        "torso_b025": 0.25,
        "torso_b050": 0.50,
        "torso_b075": 0.75,
        "torso_b100": 1.00,
    }
    if name not in table:
        raise ValueError(name)
    return {"mode": "torso", "beta": float(table[name]), "unsigned": False}


def _pct(x, qs=(25, 50, 75)) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{q}": float("nan") for q in qs} | {"n": 0, "mean": float("nan")}
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} | {"n": int(x.size), "mean": float(x.mean())}


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

parser = argparse.ArgumentParser(description="Torso Height Buffer privileged oracle. Parent only. No R-M3.")
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--task", type=str, default="MUSE-Kp-LatentRL-Kp5-HeadHands-Locomani-G1-v0")
parser.add_argument("--motion", type=str, default=f"{CLIP_ROOT}/loco")
parser.add_argument("--mask_modes", type=str, default="torso")
parser.add_argument("--task_source", type=str, default="loco", choices=TASK_SOURCES)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out", type=str, default="results/thb_torso_height_buffer")
parser.add_argument("--start_frame", type=int, default=10)
parser.add_argument("--hint_iter", type=int, default=50000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--seeds", type=str, default="42,43,44")
parser.add_argument("--terrain", type=str, default="steps", choices=TERRAINS)
parser.add_argument("--terrains", type=str, default="steps")
parser.add_argument("--keep_headhands_spec", action="store_true", default=True)
parser.add_argument("--mapper_path", type=str, default=MAPPER_B)
parser.add_argument("--mapper_b_path", type=str, default=MAPPER_B)
parser.add_argument("--residual_alpha", type=float, default=1.0)
parser.add_argument("--s_enabled", action="store_true", default=False)
parser.add_argument("--no_s_enabled", action="store_true", default=False)
parser.add_argument("--variants", type=str, default=",".join(VARIANTS))
parser.add_argument("--thb_variant", type=str, default="")
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
        print("[thb] terrain=plane", flush=True)
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
    print(f"[thb] terrain={terrain} 1x1 generator", flush=True)


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
                print(f"[thb] pin mask {mask_name!r} idx={names.index(mask_name)}", flush=True)
        cmd._resample_command(ids)
        if hasattr(cmd, "resample_all_mask_modes"):
            cmd.resample_all_mask_modes()
        try:
            env.unwrapped.sim.forward()
        except Exception as exc:
            print(f"[thb] sim.forward skipped: {exc}", flush=True)
    return use, [str(p) for p in paths[:use]]


def _body_e(cmd, vis_i) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = cmd.robot_body_pos_w - cmd.body_pos_w
    return (
        torch.linalg.norm(delta[:, vis_i[0]], dim=-1),
        torch.linalg.norm(delta[:, vis_i[1]], dim=-1),
        torch.linalg.norm(delta[:, vis_i[2]], dim=-1),
    )


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


def _contact_ids(robot, cf, names: tuple[str, ...]) -> tuple[list[int], list[int]]:
    rnames = list(robot.data.body_names)
    rids = [rnames.index(n) for n in names if n in rnames]
    cids: list[int] = []
    if cf is None:
        return rids, cids
    cnames = None
    for attr in ("body_names",):
        obj = getattr(cf, attr, None)
        if obj:
            cnames = list(obj)
            break
    data = getattr(cf, "data", None)
    if cnames is None and data is not None:
        obj = getattr(data, "body_names", None)
        if obj:
            cnames = list(obj)
    if cnames:
        cids = [cnames.index(n) for n in names if n in cnames]
        return rids, cids
    nf = getattr(getattr(cf, "data", None), "net_forces_w", None)
    if nf is not None and nf.shape[1] == len(rnames):
        return rids, list(rids)
    return rids, []


def _install_thb(cmd) -> None:
    if bool(getattr(cmd, "_thb_installed", False)):
        return
    loader = cmd.motion_dir_loader
    orig = loader.gather
    bodies = list(cmd.cfg.body_names)
    torso_i = bodies.index("torso_link")
    lw = bodies.index("left_wrist_yaw_link")
    rw = bodies.index("right_wrist_yaw_link")
    n_env = int(cmd.num_envs)
    device = cmd.device
    cmd._thb_dz = torch.zeros(n_env, device=device)
    cmd._thb_dh = torch.zeros(n_env, device=device)
    cmd._thb_dh_up = torch.zeros(n_env, device=device)
    cmd._thb_h = torch.zeros(n_env, device=device)
    cmd._thb_h0 = torch.zeros(n_env, device=device)
    cmd._thb_mode = "none"
    cmd._thb_beta = 0.0
    cmd._thb_n_env = n_env
    cmd._thb_torso_i = torso_i
    cmd._thb_lw = lw
    cmd._thb_rw = rw

    def gather(name, motion_idx, frame_idx, out_device=None):
        out = orig(name, motion_idx, frame_idx, out_device=out_device)
        mode = getattr(cmd, "_thb_mode", "none")
        if name != "body_pos_w" or mode in ("none", "", None):
            return out
        dz = cmd._thb_dz
        q = int(out.shape[0])
        if q == n_env:
            d = dz
        elif n_env > 0 and q % n_env == 0:
            slots = q // n_env
            d = dz.view(n_env, 1).expand(n_env, slots).reshape(-1)
        else:
            return out
        out = out.clone()
        out[:, torso_i, 2] = out[:, torso_i, 2] + d
        if mode == "shift_all":
            out[:, lw, 2] = out[:, lw, 2] + d
            out[:, rw, 2] = out[:, rw, 2] + d
        return out

    loader.gather = gather
    cmd._thb_installed = True
    print(f"[thb] gather wrap installed torso={torso_i} lw={lw} rw={rw}", flush=True)


def _update_thb(env, cmd, t: int, use: int, ankle_r: list[int], cf, ankle_c: list[int]) -> None:
    robot = env.unwrapped.scene["robot"]
    n = int(cmd.num_envs)
    z = robot.data.body_pos_w[:, ankle_r, 2]
    contact = torch.ones_like(z, dtype=torch.bool)
    if cf is not None and ankle_c and hasattr(cf.data, "net_forces_w"):
        nf = cf.data.net_forces_w
        if nf.ndim >= 3 and nf.shape[1] > max(ankle_c):
            contact = nf[:, ankle_c, :].norm(dim=-1) > CONTACT_N
    n_c = contact.float().sum(dim=-1)
    h_raw = (z * contact.float()).sum(dim=-1) / n_c.clamp(min=1.0)
    none = n_c < 0.5
    last = getattr(cmd, "_thb_h", h_raw)
    h = torch.where(none, last, h_raw)
    cmd._thb_h = h
    buf = getattr(cmd, "_thb_h0_frames", None)
    if t == 0 or buf is None:
        cmd._thb_h0_frames = []
        buf = cmd._thb_h0_frames
    if WARMUP <= t < WARMUP + H0_FRAMES:
        buf.append(h.detach().clone())
    if t == WARMUP + H0_FRAMES - 1 and buf:
        stacked = torch.stack(buf, dim=0)
        cmd._thb_h0 = stacked.median(dim=0).values
        cmd._thb_h0_ready = True
        print(
            f"[thb] h0 median={float(cmd._thb_h0[:use].mean()):.4f} "
            f"n_contact={float(n_c[:use].mean()):.2f}",
            flush=True,
        )
    if not bool(getattr(cmd, "_thb_h0_ready", False)):
        dh_signed = torch.zeros_like(h)
    else:
        dh_signed = h - cmd._thb_h0
    # Signed support-relative slack. T0 `steps` is a pyramid (spawn on the
    # upper platform), so support often drops. max(0, Δh) is then a no-op.
    # β∈[0,1] still guarantees |δz| ≤ |h_t − h0| (the user's bound).
    cmd._thb_dh = dh_signed
    cmd._thb_dh_up = dh_signed.clamp(min=0.0)
    mode = getattr(cmd, "_thb_mode", "none")
    beta = float(getattr(cmd, "_thb_beta", 0.0))
    unsigned = bool(getattr(cmd, "_thb_unsigned", False))
    if mode in ("none", "", None):
        dz = torch.zeros_like(dh_signed)
    elif unsigned:
        dz = cmd._thb_dh_up * beta
    else:
        dz = dh_signed * beta
    cmd._thb_dz = dz


def _arm_indices(robot) -> torch.Tensor:
    names = list(robot.data.joint_names)
    idx = [i for i, n in enumerate(names) if any(tok in n for tok in ARM_TOKENS)]
    return torch.as_tensor(idx, device=robot.device, dtype=torch.long)


def _rollout(
    env,
    policy,
    runner,
    injector,
    steps: int,
    seed: int,
    terrain: str,
    vis_i,
    vis_flags,
    mask_name: str,
    task_source: str,
    variant: str,
    spec: dict,
) -> list[dict]:
    n_envs = int(env.unwrapped.num_envs)
    use, paths = _pin_clips(env, mask_name=mask_name)
    cmd = env.unwrapped.command_manager.get_term("motion")
    _install_thb(cmd)
    cmd._thb_mode = spec["mode"]
    cmd._thb_beta = float(spec["beta"])
    cmd._thb_unsigned = bool(spec.get("unsigned", False))
    cmd._thb_h0_ready = False
    cmd._thb_h0_frames = []
    cmd._thb_dz.zero_()
    cmd._thb_dh.zero_()
    device = env.unwrapped.device
    asset = env.unwrapped.scene["robot"]
    gravity = asset.data.GRAVITY_VEC_W
    if gravity.ndim == 1:
        gravity = gravity.unsqueeze(0).expand(cmd.num_envs, -1)

    failed = torch.zeros(n_envs, dtype=torch.bool, device=device)
    failed_orig = torch.zeros(n_envs, dtype=torch.bool, device=device)
    fail_step = [-1] * use
    fail_step_orig = [-1] * use
    fail_reason = [""] * use
    fail_reason_orig = [""] * use

    body_series = {
        (seed, e): {
            "torso_exec": [],
            "torso_orig": [],
            "lw_exec": [],
            "rw_exec": [],
            "lw_orig": [],
            "rw_orig": [],
            "dz": [],
            "dh": [],
            "dh_up": [],
            "h": [],
            "arm_sat": [],
            "act_sat": [],
            "adj_z_buf": [],
            "adj_z_orig": [],
        }
        for e in range(use)
    }

    try:
        cf = env.unwrapped.scene["contact_forces"]
    except Exception:
        cf = None
    ankle_r, ankle_c = _contact_ids(asset, cf, ANKLE_BODIES)
    arm_i = _arm_indices(asset)
    robot_names = list(asset.data.body_names)
    pelvis_i = robot_names.index("pelvis") if "pelvis" in robot_names else None
    am = env.unwrapped.action_manager
    lim = getattr(asset.data, "soft_joint_pos_limits", None)
    lw_i, rw_i, t_i = vis_i[1], vis_i[2], vis_i[0]

    for t in range(steps):
        _update_thb(env, cmd, t, use, ankle_r, cf, ankle_c)
        obs, _ = env.get_observations()
        obs = injector.patch_policy_obs(obs, env, "mapper")
        z_nom, proprio = _z_and_proprio(policy, runner, obs)
        joints = _decode(policy, z_nom, proprio)
        env.step(joints)

        dz = cmd._thb_dz[:use]
        dh = cmd._thb_dh[:use]
        e_torso, e_lw, e_rw = _body_e(cmd, vis_i)
        bp = cmd.body_pos_w[:use]
        rp = cmd.robot_body_pos_w[:use]
        torso_orig = bp[:, t_i].clone()
        torso_orig[:, 2] = torso_orig[:, 2] - dz
        lw_orig = bp[:, lw_i].clone()
        rw_orig = bp[:, rw_i].clone()
        if spec["mode"] == "shift_all":
            lw_orig[:, 2] = lw_orig[:, 2] - dz
            rw_orig[:, 2] = rw_orig[:, 2] - dz
        e_torso_orig = torch.linalg.norm(rp[:, t_i] - torso_orig, dim=-1)
        e_lw_orig = torch.linalg.norm(rp[:, lw_i] - lw_orig, dim=-1)
        e_rw_orig = torch.linalg.norm(rp[:, rw_i] - rw_orig, dim=-1)

        offset = _ankle_mean_z_offset(cmd)[:use]
        adj_z_buf = (cmd.anchor_pos_w[:use, -1] - cmd.robot_anchor_pos_w[:use, -1]) + offset
        adj_z_orig = (cmd.anchor_pos_w[:use, -1] - dz - cmd.robot_anchor_pos_w[:use, -1]) + offset
        fail_z = adj_z_buf.abs() > FAIL_ANCHOR_Z
        fail_z_orig = adj_z_orig.abs() > FAIL_ANCHOR_Z
        mot_g = quat_rotate_inverse(cmd.anchor_quat_w, gravity)[:use]
        rob_g = quat_rotate_inverse(cmd.robot_anchor_quat_w, gravity)[:use]
        fail_ori = (mot_g[:, 2] - rob_g[:, 2]).abs() > FAIL_ANCHOR_ORI
        torso_z = cmd.robot_body_pos_w[:use, vis_i[0], 2]
        fall = torso_z < FAIL_FALL_Z
        if pelvis_i is not None:
            fall = fall | (asset.data.body_pos_w[:use, pelvis_i, 2] < FAIL_FALL_Z)

        act = am._action[:use]
        act_sat = (act.abs() > 0.95).float().mean(dim=-1)
        arm_sat = torch.zeros(use, device=device)
        if lim is not None and arm_i.numel() > 0:
            jp = asset.data.joint_pos[:use][:, arm_i]
            lo = lim[:use][:, arm_i, 0]
            hi = lim[:use][:, arm_i, 1]
            arm_sat = ((jp <= lo + 0.02) | (jp >= hi - 0.02)).float().mean(dim=-1)

        for i in range(use):
            row = body_series[(seed, i)]
            row["torso_exec"].append(float(e_torso[i].item()))
            row["torso_orig"].append(float(e_torso_orig[i].item()))
            row["lw_exec"].append(float(e_lw[i].item()))
            row["rw_exec"].append(float(e_rw[i].item()))
            row["lw_orig"].append(float(e_lw_orig[i].item()))
            row["rw_orig"].append(float(e_rw_orig[i].item()))
            row["dz"].append(float(dz[i].item()))
            row["dh"].append(float(dh[i].item()))
            row["dh_up"].append(float(cmd._thb_dh_up[i].item()))
            row["h"].append(float(cmd._thb_h[i].item()))
            row["arm_sat"].append(float(arm_sat[i].item()))
            row["act_sat"].append(float(act_sat[i].item()))
            row["adj_z_buf"].append(float(adj_z_buf[i].item()))
            row["adj_z_orig"].append(float(adj_z_orig[i].item()))
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
            if (not bool(failed_orig[i])) and t >= WARMUP:
                if bool(fall[i]):
                    failed_orig[i] = True
                    fail_step_orig[i] = t
                    fail_reason_orig[i] = "fall"
                elif bool(fail_z_orig[i]):
                    failed_orig[i] = True
                    fail_step_orig[i] = t
                    fail_reason_orig[i] = "anchor_z"
                elif bool(fail_ori[i]):
                    failed_orig[i] = True
                    fail_step_orig[i] = t
                    fail_reason_orig[i] = "anchor_ori"

        if (t + 1) % 50 == 0:
            print(
                f"[thb] {variant} {task_source} {terrain} s{seed} {t+1}/{steps} "
                f"dz={float(dz.mean()):.3f} dh={float(dh.mean()):.3f} "
                f"e_lw={float(e_lw_orig.mean()):.3f} fail={int(failed[:use].sum())}",
                flush=True,
            )

    ep_rows = []
    for i in range(use):
        fail = fail_step[i] >= 0
        fail_o = fail_step_orig[i] >= 0
        bs = body_series[(seed, i)]
        dh_a = np.asarray(bs["dh"], dtype=np.float64)
        up_a = np.asarray(bs["dh_up"], dtype=np.float64)
        est = {"mae": float("nan"), "corr": float("nan"), "lag_frames": float("nan")}
        if dh_a.size and up_a.size and dh_a.size == up_a.size:
            est["mae"] = float(np.mean(np.abs(up_a - dh_a)))
            if dh_a.std() > 1e-8 and up_a.std() > 1e-8:
                est["corr"] = float(np.corrcoef(dh_a, up_a)[0, 1])
            d0 = dh_a - dh_a.mean()
            u0 = up_a - up_a.mean()
            if d0.size > 8:
                xc = np.correlate(u0, d0, mode="full")
                lag = int(np.argmax(xc) - (d0.size - 1))
                est["lag_frames"] = float(lag)
                est["lag_ms"] = float(lag * 20.0)
        ep_rows.append(
            {
                "clip": Path(paths[i]).name if i < len(paths) else f"env{i}",
                "seed": int(seed),
                "terrain": terrain,
                "variant": variant,
                "mode": spec["mode"],
                "beta": float(spec["beta"]),
                "sr_task": int(not fail),
                "sr_task_orig_constraint": int(not fail_o),
                "fail": int(fail),
                "fail_orig": int(fail_o),
                "fail_reason": fail_reason[i] or "none",
                "fail_reason_orig": fail_reason_orig[i] or "none",
                "fail_step": int(fail_step[i]),
                "fail_step_orig": int(fail_step_orig[i]),
                "e_torso_exec": _series_stats(bs["torso_exec"]),
                "e_torso_orig": _series_stats(bs["torso_orig"]),
                "e_lw_exec": _series_stats(bs["lw_exec"]),
                "e_rw_exec": _series_stats(bs["rw_exec"]),
                "e_lw_world_orig": _series_stats(bs["lw_orig"]),
                "e_rw_world_orig": _series_stats(bs["rw_orig"]),
                "dz": _series_stats(bs["dz"]),
                "dh": _series_stats(bs["dh"]),
                "dh_up": _series_stats(bs["dh_up"]),
                "h_support": _series_stats(bs["h"]),
                "arm_joint_sat": _series_stats(bs["arm_sat"]),
                "action_sat": _series_stats(bs["act_sat"]),
                "adj_z_buf": _series_stats(bs["adj_z_buf"]),
                "adj_z_orig": _series_stats(bs["adj_z_orig"]),
                "estimator_vs_signed": est,
            }
        )
    return ep_rows


def _ep_monitor(all_eps: list[dict]) -> dict:
    n = len(all_eps)
    if n == 0:
        return {"n_episodes": 0}
    reasons = {}
    reasons_orig = {}
    for r in all_eps:
        k = r.get("fail_reason") or "none"
        reasons[k] = reasons.get(k, 0) + 1
        k2 = r.get("fail_reason_orig") or "none"
        reasons_orig[k2] = reasons_orig.get(k2, 0) + 1

    def _mean_stat(key, inner="mean"):
        xs = []
        for r in all_eps:
            blk = r.get(key) or {}
            v = blk.get(inner)
            if v is not None and math.isfinite(float(v)):
                xs.append(float(v))
        return _pct(xs) if xs else _pct([])

    return {
        "n_episodes": n,
        "sr_task": float(np.mean([r.get("sr_task", 0) for r in all_eps])),
        "sr_task_orig_constraint": float(np.mean([r.get("sr_task_orig_constraint", 0) for r in all_eps])),
        "fail_frac": float(np.mean([r.get("fail", 0) for r in all_eps])),
        "fall_frac": float(np.mean([r.get("fail_reason") == "fall" for r in all_eps])),
        "anchor_z_frac": float(np.mean([r.get("fail_reason") == "anchor_z" for r in all_eps])),
        "anchor_ori_frac": float(np.mean([r.get("fail_reason") == "anchor_ori" for r in all_eps])),
        "anchor_z_frac_orig": float(np.mean([r.get("fail_reason_orig") == "anchor_z" for r in all_eps])),
        "fail_reasons": reasons,
        "fail_reasons_orig": reasons_orig,
        "e_torso_exec": _mean_stat("e_torso_exec"),
        "e_torso_orig": _mean_stat("e_torso_orig"),
        "e_lw_world_orig": _mean_stat("e_lw_world_orig"),
        "e_rw_world_orig": _mean_stat("e_rw_world_orig"),
        "e_lw_exec": _mean_stat("e_lw_exec"),
        "e_rw_exec": _mean_stat("e_rw_exec"),
        "dz": _mean_stat("dz"),
        "dh": _mean_stat("dh"),
        "dh_up": _mean_stat("dh_up"),
        "arm_joint_sat": _mean_stat("arm_joint_sat"),
        "action_sat": _mean_stat("action_sat"),
        "adj_z_buf": _mean_stat("adj_z_buf"),
        "adj_z_orig": _mean_stat("adj_z_orig"),
    }


def _run_variant(env, runner, policy, injector, terrain: str, variant: str, out_root: Path, resume_path: str) -> dict:
    spec = variant_spec(variant)
    cell = out_root / variant / terrain
    cell.mkdir(parents=True, exist_ok=True)
    done_p = cell / "summary.json"
    if done_p.exists():
        print(f"[thb] skip existing {done_p}", flush=True)
        return json.loads(done_p.read_text())

    task = str(args_cli.task_source)
    cmd = env.unwrapped.command_manager.get_term("motion")
    body_names = list(cmd.cfg.body_names)
    vis_i = [body_names.index(n) for n in KP_VIS]
    mask_name = str(args_cli.mask_modes).split(",")[0].strip() or ("torso" if task == "loco" else "vr")
    vis_flags = MASK_VIS.get(mask_name, (True, True, True))
    seeds = [int(s) for s in args_cli.seeds.split(",") if s.strip()] or [int(args_cli.seed)]
    all_eps: list[dict] = []
    for seed in seeds:
        print(
            f"[thb] ===== {variant} β={spec['beta']} task={task} terrain={terrain} seed={seed} =====",
            flush=True,
        )
        torch.manual_seed(seed)
        np.random.seed(seed)
        ep_rows = _rollout(
            env,
            policy,
            runner,
            injector,
            steps=int(args_cli.steps),
            seed=int(seed),
            terrain=terrain,
            vis_i=vis_i,
            vis_flags=vis_flags,
            mask_name=mask_name,
            task_source=task,
            variant=variant,
            spec=spec,
        )
        all_eps.extend(ep_rows)
        print(
            f"[thb] seed={seed} n={len(ep_rows)} "
            f"SR={float(np.mean([r['sr_task'] for r in ep_rows])):.3f} "
            f"SRorig={float(np.mean([r['sr_task_orig_constraint'] for r in ep_rows])):.3f}",
            flush=True,
        )

    mon = _ep_monitor(all_eps)
    payload = {
        "step": "THB",
        "privileged_oracle": True,
        "not_in_method": True,
        "no_rm3": True,
        "no_terrain_id": True,
        "support_height": "two-foot contact-region mean z (hold-last if none)",
        "p1_task": task,
        "mask": mask_name,
        "s_enabled": bool(args_cli.s_enabled),
        "variant": variant,
        "mode": spec["mode"],
        "beta": float(spec["beta"]),
        "unsigned": bool(spec.get("unsigned", False)),
        "not_in_method": True,
        "ckpt": resume_path,
        "mapper": args_cli.mapper_path,
        "terrain": terrain,
        "seeds": seeds,
        "steps": int(args_cli.steps),
        "n_envs": int(env.unwrapped.num_envs),
        "n_episodes": len(all_eps),
        "sr_task": mon.get("sr_task"),
        "sr_task_orig_constraint": mon.get("sr_task_orig_constraint"),
        "fail_frac": mon.get("fail_frac"),
        "fall_frac": mon.get("fall_frac"),
        "anchor_z_frac": mon.get("anchor_z_frac"),
        "episode_monitor": mon,
        "episodes": all_eps,
        "unsigned": bool(spec.get("unsigned", False)),
        "estimator_vs_signed": {
            "mae_mean": float(np.nanmean([r.get("estimator_vs_signed", {}).get("mae", np.nan) for r in all_eps])),
            "corr_mean": float(np.nanmean([r.get("estimator_vs_signed", {}).get("corr", np.nan) for r in all_eps])),
            "lag_ms_median": float(np.nanmedian([r.get("estimator_vs_signed", {}).get("lag_ms", np.nan) for r in all_eps])),
        },
        "note": (
            "Privileged diagnostic. Support slack is signed: δz=β(h_t-h0), |δz|≤|Δh|. "
            "T0 steps is a pyramid so support often drops; max(0,Δh) would be a no-op. "
            "SR_task uses buffered torso target. Wrist world-orig is vs unbuffered clip wrists. "
            "Official fail_z is already ankle-relative posture; this oracle still changes "
            "Mapper-B / command torso-z while locking wrist world targets."
        ),
    }
    (cell / "episodes.json").write_text(json.dumps(_sanitize(all_eps), indent=2), encoding="utf-8")
    (cell / "summary.json").write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    print(
        f"[thb] {task}/{variant}/{terrain} SR={mon.get('sr_task'):.3f} "
        f"SRorig={mon.get('sr_task_orig_constraint'):.3f} "
        f"anchor_z={mon.get('anchor_z_frac'):.3f} "
        f"e_lw={mon.get('e_lw_world_orig', {}).get('mean')} "
        f"dz={mon.get('dz', {}).get('mean')}",
        flush=True,
    )
    return payload


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
    print(f"[thb] ckpt {resume_path}", flush=True)
    _restore_adapter(agent_cfg, resume_path)
    if hasattr(agent_cfg.policy, "intent_recovery"):
        agent_cfg.policy.intent_recovery = False
    if hasattr(agent_cfg.policy, "terrain_scan_dim"):
        agent_cfg.policy.terrain_scan_dim = 0
    if hasattr(agent_cfg.policy, "adapter"):
        agent_cfg.policy.adapter = "residual"

    motion_dir = args_cli.motion
    terrains = [t.strip() for t in args_cli.terrains.split(",") if t.strip()] or [args_cli.terrain]
    names = [v.strip() for v in (args_cli.thb_variant or args_cli.variants).split(",") if v.strip()]
    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for terrain in terrains:
        env, runner, policy, injector = _make_env(env_cfg, agent_cfg, resume_path, motion_dir, terrain)
        for variant in names:
            print(f"[thb] ===== {args_cli.task_source}/{variant}/{terrain} =====", flush=True)
            _run_variant(env, runner, policy, injector, terrain, variant, out_dir, resume_path)
        try:
            env.close()
        except Exception:
            pass
    print("[thb] done", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
