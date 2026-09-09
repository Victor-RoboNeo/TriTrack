#!/usr/bin/env python3
"""Phase R-M Stoop — Bounded Recovery Authority. No PPO. No 6V. No Step 7.

Canonical: Mapper-B + model_50000 + frozen 6S 5° direction + Max-2.
Stoop gate: R = max(R_E, R_S), S = |v_root,z|, persist-3. Mask = vr.
No terrain/depth. After 2 bursts, Parent until R<0.6 persist-3.

Continuation learning is frozen (7A/7B). This script only measures whether
the Loco-validated finite budget transfers when failure can appear first as
physical instability rather than visible task error.
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
H_FRAMES = {"dE25": 13, "dE50": 25, "dE100": 50}
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
STEP6S_CKPT = "/data/home/chenxiangyu/robotics/Anybody/results/p2r_step6s_full/model_best.pt"
# P1 stoop SR@5 (merge_p2c). SR_recover is measured in the parent_es variant.
PARENT_SR5 = {"plane": 0.386, "light_rough": 0.410, "slope": 0.294, "steps": 0.272}
PARENT_SR_RECOVER: dict[str, float] = {}
OFFLINE_U5_QUALITY = 0.84
FAIL_ANCHOR_Z = 0.25
FAIL_ANCHOR_ORI = 0.8


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

parser = argparse.ArgumentParser(description="P2-R Phase R-M Stoop bounded recovery.")
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--task", type=str, default="MUSE-Kp-LatentRL-Kp5-HeadHands-Locomani-G1-v0")
parser.add_argument("--motion", type=str, default=CLIP_STOOP)
parser.add_argument("--mask_modes", type=str, default="vr")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out", type=str, default="results/p2r_rm_stoop")
parser.add_argument("--start_frame", type=int, default=10)
parser.add_argument("--hint_iter", type=int, default=50000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--seeds", type=str, default="42")
parser.add_argument("--terrain", type=str, default="plane", choices=TERRAINS)
parser.add_argument("--terrains", type=str, default="plane")
parser.add_argument("--keep_headhands_spec", action="store_true", default=True)
parser.add_argument("--mapper_path", type=str, default=MAPPER_B)
parser.add_argument("--mapper_b_path", type=str, default=MAPPER_B)
parser.add_argument("--mapper_a_path", type=str, default="")
parser.add_argument("--residual_alpha", type=float, default=1.0)
parser.add_argument("--step6s_ckpt", type=str, default=STEP6S_CKPT)
parser.add_argument("--mode", type=str, default="cl2", choices=("cl1", "cl2"))
parser.add_argument("--theta_deg", type=float, default=THETA_DEG)
parser.add_argument("--burst_steps", type=int, default=BURST_STEPS)
parser.add_argument(
    "--max_bursts",
    type=int,
    default=2,
    help="2 = Bounded Recovery Authority. 0 = unlimited (not canonical).",
)
parser.add_argument("--s_enabled", action="store_true", default=False, help="R = max(R_E, R_S).")
parser.add_argument("--no_s_enabled", action="store_true", default=False)
parser.add_argument(
    "--apply_recovery",
    action="store_true",
    default=True,
    help="If false, record the gate but never apply 5° (Parent baseline).",
)
parser.add_argument("--no_apply_recovery", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False
if bool(getattr(args_cli, "no_apply_recovery", False)):
    args_cli.apply_recovery = False
if bool(getattr(args_cli, "no_s_enabled", False)):
    args_cli.s_enabled = False
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import yaml  # noqa: E402
from isaaclab.utils.math import quat_error_magnitude, quat_rotate_inverse  # noqa: E402
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
        print("[rm-stoop] terrain=plane", flush=True)
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
    print(f"[rm-stoop] terrain={terrain} 1x1 generator", flush=True)


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
                print(f"[rm-stoop] pin mask {mask_name!r} idx={names.index(mask_name)}", flush=True)
        cmd._resample_command(ids)
        if hasattr(cmd, "resample_all_mask_modes"):
            cmd.resample_all_mask_modes()
        try:
            env.unwrapped.sim.forward()
        except Exception as exc:
            print(f"[rm-stoop] sim.forward skipped: {exc}", flush=True)
    return use, [str(p) for p in paths[:use]]


def _visible_e(cmd, vis_i, vis_flags) -> torch.Tensor:
    delta = cmd.robot_body_pos_w - cmd.body_pos_w
    parts = [torch.linalg.norm(delta[:, vis_i[k]], dim=-1) for k, flag in enumerate(vis_flags) if flag]
    stacked = torch.stack(parts, dim=-1)
    return stacked.mean(dim=-1)


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
        for name, h in H_FRAMES.items():
            t_h = t0 + h
            if t_h <= end and t_h < len(series):
                ev[name + "_complete"] = float(series[t_h] - e0)
                ev[name + "_last"] = float(series[t_h] - e0)
            else:
                ev[name + "_complete"] = None
                t_use = min(end, max(t0, end))
                t_use = min(t_use, len(series) - 1) if series else 0
                ev[name + "_last"] = float(series[t_use] - e0) if series else float("nan")
        ev["lived_1s"] = (t0 + H_FRAMES["dE100"]) <= end


def _agg_events(events: list[dict]) -> dict:
    n = len(events)
    n_rel = sum(1 for e in events if e.get("released"))
    n_fail = sum(1 for e in events if e.get("fail_after"))
    out = {
        "n_events": n,
        "n_release": n_rel,
        "n_fail_after": n_fail,
        "SR_recover": (n_rel / n) if n else float("nan"),
        "fail_after_trigger": (n_fail / n) if n else float("nan"),
    }
    trec = [e["Trec"] for e in events if e.get("Trec") is not None]
    auc = [e["auc"] for e in events]
    n_bursts = [e.get("n_bursts", 0) for e in events]
    out["T_rec"] = _pct(np.array(trec, dtype=np.float64))
    out["AUC_E"] = _pct(np.array(auc, dtype=np.float64))
    out["bursts_per_event"] = _pct(np.array(n_bursts, dtype=np.float64))
    out["active_duration_s"] = _pct(np.array([e["n_act"] * DT for e in events], dtype=np.float64))
    for name in H_FRAMES:
        c = [e[name + "_complete"] for e in events if e.get(name + "_complete") is not None]
        last = [e[name + "_last"] for e in events if e.get(name + "_last") is not None]
        out[name + "_complete_m"] = _pct(np.array(c, dtype=np.float64)) | {"n": len(c)}
        out[name + "_last_obs_m"] = _pct(np.array(last, dtype=np.float64)) | {"n": len(last)}
        out[name + "_complete_frac"] = (len(c) / n) if n else float("nan")
        med_cm = (out[name + "_complete_m"]["p50"] * 100.0) if c else float("nan")
        out[name + "_complete_cm_p50"] = med_cm
        out[name + "_coverage"] = out[name + "_complete_frac"]
    out["lived_1s_frac"] = (sum(1 for e in events if e.get("lived_1s")) / n) if n else float("nan")
    src = [str(e.get("trigger_channel") or "?") for e in events]
    out["trigger_channel"] = {
        k: _rate(np.array(src) == k) if n else float("nan")
        for k in ("E", "S", "both")
    }
    out["n_trigger_S"] = int(sum(s in ("S", "both") for s in src))
    out["n_trigger_E"] = int(sum(s in ("E", "both") for s in src))
    out["n_trigger_S_only"] = int(sum(s == "S" for s in src))

    burst_rows = []
    for e in events:
        for b in e.get("bursts", []):
            if b.get("dE_m") is None:
                continue
            burst_rows.append(b)
    first = [b for b in burst_rows if int(b.get("burst_index", 0)) == 1]
    dE1 = np.array([b["dE_m"] for b in first], dtype=np.float64) if first else np.array([], dtype=np.float64)
    out["first_burst"] = {
        "n": int(dE1.size),
        "U1": _rate(dE1 < 0.0) if dE1.size else float("nan"),
        "P_dE_lt_1cm": _rate(dE1 < -0.01) if dE1.size else float("nan"),
        "median_dE_cm": float(np.median(dE1) * 100.0) if dE1.size else float("nan"),
        "mean_dE_cm": float(dE1.mean() * 100.0) if dE1.size else float("nan"),
        "dE_cm": _pct(dE1 * 100.0) if dE1.size else _pct([]),
        "offline_U5_quality": OFFLINE_U5_QUALITY,
    }
    by_idx: dict[str, list[float]] = {"1": [], "2": [], "3": [], "ge4": []}
    for b in burst_rows:
        idx = int(b["burst_index"])
        key = str(idx) if idx in (1, 2, 3) else "ge4"
        by_idx[key].append(float(b["dE_m"]))
    out["by_burst_index"] = {}
    for k, vs in by_idx.items():
        arr = np.array(vs, dtype=np.float64)
        out["by_burst_index"][k] = {
            "n": int(arr.size),
            "P_dE_lt_0": _rate(arr < 0.0) if arr.size else float("nan"),
            "median_dE_cm": float(np.median(arr) * 100.0) if arr.size else float("nan"),
            "dE_cm": _pct(arr * 100.0) if arr.size else _pct([]),
        }
    return out


@torch.inference_mode()
def _rollout_cl(
    env,
    policy,
    runner,
    injector,
    rec_model,
    obs_mu,
    obs_sd,
    *,
    mode: str,
    steps: int,
    seed: int,
    terrain: str,
    theta_deg: float,
    burst_len: int,
    max_bursts: int,
    vis_i,
    vis_flags,
    clip_paths: list[str],
    s_enabled: bool,
    apply_recovery: bool,
    mask_name: str,
) -> tuple[list[dict], list[dict]]:
    n_envs = int(env.unwrapped.num_envs)
    use, paths = _pin_clips(env, mask_name=mask_name)
    if clip_paths:
        paths = clip_paths[:use]
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
    event_bursted = torch.zeros(n_envs, dtype=torch.bool, device=device)
    failed = torch.zeros(n_envs, dtype=torch.bool, device=device)
    fail_step = [-1] * use
    fail_reason = [""] * use
    e9_prev = None

    events: list[dict] = []
    cur: list[dict | None] = [None] * use
    open_bursts: list[dict | None] = [None] * use
    e_post_series: dict[tuple[int, int], list[float]] = {(seed, e): [] for e in range(use)}
    t_end_map: dict[tuple[int, int], int] = {(seed, e): steps - 1 for e in range(use)}

    cl2 = mode == "cl2"

    for t in range(steps):
        obs, _ = env.get_observations()
        obs = injector.patch_policy_obs(obs, env, "mapper")
        e_pre = _visible_e(cmd, vis_i, vis_flags)
        s_in = asset.data.root_lin_vel_w[:, 2].abs().to(dtype=e_pre.dtype)
        gout = gate.step(e_pre, s_in if bool(s_enabled) else torch.zeros_like(e_pre))
        active = gout["active"]
        z_nom, proprio = _z_and_proprio(policy, runner, obs)
        rec_pack, e9 = _recovery_pack(obs, z_nom, e9_prev)

        rising = active & ~active_prev
        falling = (~active) & active_prev

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
                }
                events.append(ev)
                cur[i] = ev
                event_bursted[i] = False
            if cur[i] is not None and (not cur[i]["released"]) and (not cur[i]["failed"]):
                if bool(active[i]):
                    cur[i]["n_act"] += 1
                    cur[i]["auc"] += float(e_pre[i].item()) * DT
                if bool(falling[i]):
                    cur[i]["released"] = True
                    cur[i]["t_rel"] = int(t)
                    event_bursted[i] = False

        has_open = torch.zeros(n_envs, dtype=torch.bool, device=device)
        for i in range(use):
            ev = cur[i]
            if ev is not None and (not ev["released"]) and (not ev["failed"]):
                has_open[i] = True
        # Predict only at burst boundaries. CL1: once per event. CL2: every 100 ms.
        # max_bursts>0: stop issuing recovery after N bursts; Parent runs; gate may still fall.
        need = has_open & (burst_left == 0) & (~failed)
        if not cl2:
            need = need & ~event_bursted
        if int(max_bursts) > 0:
            for i in range(use):
                if cur[i] is not None and int(cur[i]["n_bursts"]) >= int(max_bursts):
                    need[i] = False
        if not bool(apply_recovery):
            need[:] = False

        idxs = need.nonzero(as_tuple=False).view(-1).tolist()
        if idxs:
            x = (rec_pack - obs_mu) / obs_sd
            raw = rec_model(x)
            nrm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            d_unit = raw / nrm
            for i in idxs:
                held_d[i] = d_unit[i]
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
                    }
                    cur[i]["bursts"].append(b)
                    open_bursts[i] = b

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
            e_post_series[(seed, i)].append(float(e_post[i].item()))
            if bool(finishing[i]) and open_bursts[i] is not None:
                open_bursts[i]["e_after"] = float(e_post[i].item())
                open_bursts[i]["dE_m"] = float(e_post[i].item()) - float(open_bursts[i]["e_before"])
                open_bursts[i] = None
            if (not bool(failed[i])) and t >= WARMUP:
                if bool(fail_z[i]):
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
                f"[rm-stoop] {mode} {terrain} s{seed} step {t+1}/{steps} "
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
        ep_rows.append(
            {
                "clip": Path(paths[i]).name if i < len(paths) else f"env{i}",
                "seed": int(seed),
                "terrain": terrain,
                "sr_5cm": float((e_vis < 0.05).mean()) if e_vis.size else float("nan"),
                "sr_2cm": float((e_vis < 0.02).mean()) if e_vis.size else float("nan"),
                "e_kp_mean": float(e_vis.mean()) if e_vis.size else float("nan"),
                "fail": int(fail),
                "fail_reason": fail_reason[i] or "none",
                "episode_length": int(ep_len),
                "n_events": sum(1 for ev in events if ev["env"] == i),
            }
        )
    return events, ep_rows


def _run_terrain(env_cfg, agent_cfg, resume_path: str, motion_dir: str, terrain: str, out_root: Path) -> dict:
    cell = out_root / terrain
    cell.mkdir(parents=True, exist_ok=True)
    done_p = cell / "summary.json"
    if done_p.exists():
        print(f"[rm-stoop] skip existing {done_p}", flush=True)
        return json.loads(done_p.read_text())

    env, runner, policy, injector = _make_env(env_cfg, agent_cfg, resume_path, motion_dir, terrain)
    rec_model, obs_mu, obs_sd = _load_6s_predictor(args_cli.step6s_ckpt, env.device)
    cmd = env.unwrapped.command_manager.get_term("motion")
    body_names = list(cmd.cfg.body_names)
    vis_i = [body_names.index(n) for n in KP_VIS]
    mask_name = str(args_cli.mask_modes).split(",")[0].strip() or "vr"
    vis_flags = MASK_VIS.get(mask_name, (True, True, True))
    s_enabled = bool(args_cli.s_enabled)
    apply_recovery = bool(args_cli.apply_recovery)
    seeds = [int(s) for s in args_cli.seeds.split(",") if s.strip()] or [int(args_cli.seed)]
    mode = str(args_cli.mode)
    all_events: list[dict] = []
    all_eps: list[dict] = []
    for seed in seeds:
        print(
            f"[rm-stoop] ===== mode={mode} s={int(s_enabled)} apply={int(apply_recovery)} "
            f"terrain={terrain} seed={seed} =====",
            flush=True,
        )
        torch.manual_seed(seed)
        np.random.seed(seed)
        events, ep_rows = _rollout_cl(
            env,
            policy,
            runner,
            injector,
            rec_model,
            obs_mu,
            obs_sd,
            mode=mode,
            steps=int(args_cli.steps),
            seed=seed,
            terrain=terrain,
            theta_deg=float(args_cli.theta_deg),
            burst_len=int(args_cli.burst_steps),
            max_bursts=int(args_cli.max_bursts),
            vis_i=vis_i,
            vis_flags=vis_flags,
            clip_paths=[],
            s_enabled=s_enabled,
            apply_recovery=apply_recovery,
            mask_name=mask_name,
        )
        all_events.extend(events)
        all_eps.extend(ep_rows)
        n_trig = len(events)
        n_b = sum(e["n_bursts"] for e in events)
        print(f"[rm-stoop] seed={seed} events={n_trig} bursts={n_b} eps={len(ep_rows)}", flush=True)

    recov = _agg_events(all_events)
    sr5 = float(np.mean([r["sr_5cm"] for r in all_eps])) if all_eps else float("nan")
    fail_frac = float(np.mean([r["fail"] for r in all_eps])) if all_eps else float("nan")
    parent_sr5 = PARENT_SR5.get(terrain, float("nan"))
    parent_rec = PARENT_SR_RECOVER.get(terrain, float("nan"))
    payload = {
        "step": "R-M-stoop",
        "no_ppo": True,
        "p1_task": "stoop",
        "mask": str(args_cli.mask_modes),
        "s_enabled": bool(args_cli.s_enabled),
        "apply_recovery": bool(args_cli.apply_recovery),
        "binary_5deg": True,
        "no_continuous_alpha": True,
        "burst_steps": int(args_cli.burst_steps),
        "max_bursts": int(args_cli.max_bursts),
        "theta_deg": float(args_cli.theta_deg),
        "ckpt": resume_path,
        "mapper": args_cli.mapper_path,
        "step6s_ckpt": args_cli.step6s_ckpt,
        "terrain": terrain,
        "mode": mode,
        "seeds": seeds,
        "steps": int(args_cli.steps),
        "n_envs": int(env.unwrapped.num_envs),
        "n_episodes": len(all_eps),
        "sr_5cm": sr5,
        "parent_sr_5cm": parent_sr5,
        "delta_sr_5cm_pp": (sr5 - parent_sr5) * 100.0 if np.isfinite(sr5) else float("nan"),
        "fail_frac": fail_frac,
        "recovery": recov,
        "parent_SR_recover": parent_rec,
        "delta_SR_recover_pp": (recov["SR_recover"] - parent_rec) * 100.0
        if np.isfinite(recov["SR_recover"])
        else float("nan"),
        "episodes": all_eps,
        "events": all_events,
    }
    (cell / "events.json").write_text(json.dumps(_sanitize(all_events), indent=2), encoding="utf-8")
    (cell / "summary.json").write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    u1 = recov["first_burst"]["U1"]
    med = recov["first_burst"]["median_dE_cm"]
    print(
        f"[rm-stoop] {terrain} SR@5={sr5:.3f} (parent {parent_sr5:.3f}, Δ={payload['delta_sr_5cm_pp']:.1f}pp) "
        f"SRrec={recov['SR_recover']:.3f} (parent {parent_rec:.3f}) "
        f"U1={u1} medianΔE1={med}cm n_ev={recov['n_events']}",
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
            "sr_5cm": cell.get("sr_5cm"),
            "parent_sr_5cm": cell.get("parent_sr_5cm"),
            "delta_sr_5cm_pp": cell.get("delta_sr_5cm_pp"),
            "SR_recover": recov.get("SR_recover"),
            "parent_SR_recover": cell.get("parent_SR_recover"),
            "fail_after_trigger": recov.get("fail_after_trigger"),
            "first_burst": recov.get("first_burst"),
            "by_burst_index": recov.get("by_burst_index"),
            "n_events": recov.get("n_events"),
            "n_episodes": cell.get("n_episodes"),
            "trigger_channel": recov.get("trigger_channel"),
            "n_trigger_S_only": recov.get("n_trigger_S_only"),
            "AUC_E": recov.get("AUC_E"),
            "dE25_complete_cm_p50": recov.get("dE25_complete_cm_p50"),
            "dE25_coverage": recov.get("dE25_coverage"),
            "dE50_complete_cm_p50": recov.get("dE50_complete_cm_p50"),
            "dE50_coverage": recov.get("dE50_coverage"),
            "dE100_complete_cm_p50": recov.get("dE100_complete_cm_p50"),
            "dE100_coverage": recov.get("dE100_coverage"),
        }
    recov_all = _agg_events(all_events)
    sr5_all = float(np.mean([r["sr_5cm"] for r in all_eps])) if all_eps else float("nan")
    u1 = recov_all["first_burst"]["U1"]
    med = recov_all["first_burst"]["median_dE_cm"]
    pooled = {
        "step": "R-M-stoop",
        "no_ppo": True,
        "p1_task": "stoop",
        "s_enabled": bool(args_cli.s_enabled),
        "apply_recovery": bool(args_cli.apply_recovery),
        "ckpt": resume_path,
        "step6s_ckpt": args_cli.step6s_ckpt,
        "mode": args_cli.mode,
        "max_bursts": int(args_cli.max_bursts),
        "mask": str(args_cli.mask_modes),
        "terrains": sorted(cells),
        "n_episodes": len(all_eps),
        "n_events": recov_all["n_events"],
        "sr_5cm_all": sr5_all,
        "recovery_all": recov_all,
        "by_terrain": by_terrain,
        "gates": {
            "note": "Compare parent_es vs max2_es vs max2_e after all variants. No continuation learning.",
            "median_dE1_cm": med,
            "U1": u1,
            "trigger_channel": recov_all.get("trigger_channel"),
        },
        "note": "Bounded Recovery Authority on Stoop. Frozen 6S. No PPO.",
    }
    (out_root / "pooled.json").write_text(json.dumps(_sanitize(pooled), indent=2), encoding="utf-8")
    print(
        f"[rm-stoop] POOLED n_ep={len(all_eps)} n_ev={recov_all['n_events']} "
        f"SR@5={sr5_all:.3f} SRrec={recov_all.get('SR_recover')} "
        f"fail_after={recov_all.get('fail_after_trigger')} "
        f"trig={recov_all.get('trigger_channel')} U1={u1} medianΔE1={med}cm",
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
    print(f"[rm-stoop] ckpt {resume_path}", flush=True)
    print(
        f"[rm-stoop] mode={args_cli.mode} theta={args_cli.theta_deg} "
        f"burst={args_cli.burst_steps} max_bursts={args_cli.max_bursts} "
        f"s_enabled={args_cli.s_enabled} apply={args_cli.apply_recovery} "
        f"mask={args_cli.mask_modes} NO PPO",
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
        print(f"[rm-stoop] ===== {terrain} =====", flush=True)
        _run_terrain(env_cfg, agent_cfg, resume_path, motion_dir, terrain, out_dir)
        _write_pooled(out_dir, resume_path)
    print("[rm-stoop] done", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
