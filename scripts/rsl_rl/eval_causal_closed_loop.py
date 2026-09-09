"""Causal closed-loop eval: Oracle / Hold / Mapper on the same clips, seed, VR mask.

Does not train. Loads frozen ``model_50000`` (residual g_phi) and optional
``mapper_best.pt``. Mapper / Hold overwrite future KP slots using only K_<=t.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Causal closed-loop Oracle / Hold / Mapper eval.")
parser.add_argument("--num_envs", type=int, default=0, help="0 = one env per npz")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--motion", type=str, required=True, help="directory of npz clips")
parser.add_argument("--mask_modes", type=str, default="vr")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out", type=str, default="results/causal_closed_loop")
parser.add_argument("--start_frame", type=int, default=10)
parser.add_argument("--hint_iter", type=int, default=50000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--terrain", type=str, default="plane", choices=("plane", "light_rough", "slope", "steps"))
parser.add_argument("--seeds", type=str, default="", help="comma-separated seeds; empty = --seed only")
parser.add_argument("--task_name", type=str, default="", help="P1 task label: loco|reach|stoop|carry")
parser.add_argument("--keep_headhands_spec", action="store_true", help="keep 4-mode HeadHands spec and pin --mask_modes")
parser.add_argument(
    "--mapper_path",
    type=str,
    default="/data/home/chenxiangyu/victor/TriTrack/runs/mapper_b_intent72/mapper_best.pt",
)
parser.add_argument(
    "--modes",
    type=str,
    default="mapper",
    help="comma-separated: oracle,hold,mapper,mapper_a,mapper_b and optional _norest suffix",
)
parser.add_argument(
    "--mapper_a_path",
    type=str,
    default="",
    help="81-D Mapper-A checkpoint (robot KP in features). Overrides --mapper_path for mode mapper_a.",
)
parser.add_argument(
    "--mapper_b_path",
    type=str,
    default="",
    help="72-D Mapper-B checkpoint (intent history only).",
)
parser.add_argument(
    "--residual_alpha",
    type=float,
    default=1.0,
    help="g_φ scale. 0 = frozen Stage-2 only. Per-mode _norest suffix forces 0.",
)
parser.add_argument("--fail_anchor_z", type=float, default=0.25)
parser.add_argument("--fail_anchor_ori", type=float, default=0.8)
parser.add_argument("--fail_ee_z", type=float, default=0.25)
parser.add_argument(
    "--probe_clip_root",
    type=str,
    default="",
    help="If set with --probe_tasks/--probe_terrains, reuse this Kit and sweep cells.",
)
parser.add_argument("--probe_tasks", type=str, default="", help="comma: loco,reach,stoop,carry")
parser.add_argument("--probe_terrains", type=str, default="", help="comma: plane,light_rough,slope,steps")
parser.add_argument(
    "--probe_reach_mask",
    type=str,
    default="",
    help="Override reach mask for probe_tasks (e.g. head_right for 1-wrist).",
)
parser.add_argument(
    "--scan_mode",
    type=str,
    default="normal",
    choices=("normal", "zero", "shuffle"),
    help="P2-C height-scan ablation: true scan / H=0 / env i gets env i+1 scan.",
)
parser.add_argument(
    "--scan_dim",
    type=int,
    default=187,
    help="Trailing height-scan dim for --scan_mode. Ignored if obs is shorter.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import yaml
from isaaclab.utils.math import euler_xyz_from_quat, quat_conjugate, quat_error_magnitude, quat_mul, quat_rotate_inverse
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

import whole_body_tracking.tasks  # noqa: F401

from causal_future import CausalFutureInjector, DEFAULT_MAPPER
from whole_body_tracking.tasks.tracking.mdp.rewards import _ankle_mean_z_offset

KP_VIS = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
EE_FAIL_BODIES = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
CSV_FIELDS = [
    "mode",
    "terrain",
    "clip",
    "task",
    "seed",
    "mask",
    "n_steps",
    "sr_5cm",
    "sr_2cm",
    "e_kp_mean",
    "e_kp_p50",
    "e_kp_p90",
    "e_torso",
    "e_lw",
    "e_rw",
    "e_xy",
    "e_z",
    "e_torso_xy",
    "e_torso_z",
    "e_lw_xy",
    "e_lw_z",
    "e_rw_xy",
    "e_rw_z",
    "wrist_err_mean",
    "wrist_err_p90",
    "ori_err_rad",
    "ori_roll_rad",
    "ori_pitch_rad",
    "ori_yaw_rad",
    "fail",
    "fail_reason",
    "episode_length",
    "action_smoothness",
    "abs_dz",
    "mapper_err_0p1",
    "mapper_err_0p2",
    "mapper_err_0p3",
    "mapper_err_0p5",
    "residual_alpha",
]


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
        "terrain_gate_w_s",
        "terrain_gate_w_r",
        "terrain_gate_s0",
        "terrain_gate_tau",
        "terrain_gate_s_dead",
        "terrain_scan_clip",
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
        print("[causal] terrain=plane", flush=True)
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
            ),
        }
    elif terrain == "slope":
        subs = {
            "slope": HfPyramidSlopedTerrainCfg(
                proportion=1.0, slope_range=(0.087, 0.176), platform_width=2.0
            ),
        }
    elif terrain == "steps":
        subs = {
            "stairs": HfPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(0.03, 0.08),
                step_width=0.4,
                platform_width=2.0,
            ),
        }
    else:
        raise ValueError(f"unknown terrain {terrain!r}")
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
    print(f"[causal] terrain={terrain} 1x1 generator", flush=True)


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
                print(f"[causal] pin mask {mask_name!r} idx={names.index(mask_name)} among {names}", flush=True)
        cmd._resample_command(ids)
        if hasattr(cmd, "resample_all_mask_modes"):
            cmd.resample_all_mask_modes()
        try:
            env.unwrapped.sim.forward()
        except Exception as exc:
            print(f"[causal] sim.forward skipped: {exc}", flush=True)
    for i in range(use):
        print(f"  env{i} {Path(paths[i]).name}", flush=True)
    return use, paths[:use]


def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _rel_rpy(q_ref: torch.Tensor, q_robot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_err = quat_mul(quat_conjugate(q_ref), q_robot)
    roll, pitch, yaw = euler_xyz_from_quat(q_err)
    return _wrap_pi(roll), _wrap_pi(pitch), _wrap_pi(yaw)


def _percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(x.mean())


def _set_residual_alpha(policy_mod, alpha: float) -> None:
    rc = getattr(policy_mod, "residual_corrector", None)
    if rc is None:
        print(f"[causal] no residual_corrector; cannot set alpha={alpha}", flush=True)
        return
    rc.alpha = float(alpha)
    print(f"[causal] residual_alpha={rc.alpha}", flush=True)


def _future_base(mode: str) -> str:
    return mode[: -len("_norest")] if mode.endswith("_norest") else mode


def _intent_vis_flags(mask_name: str) -> tuple[bool, bool, bool]:
    n = (mask_name or "vr").lower()
    if n in ("torso", "kp5_torso"):
        return True, False, False
    if n in ("head_left",):
        return True, True, False
    if n in ("head_right",):
        return True, False, True
    return True, True, True


def _rollout(
    env,
    policy,
    injector: CausalFutureInjector,
    mode: str,
    steps: int,
    args,
    mask_name: str = "vr",
    policy_mod=None,
) -> list[dict]:
    cmd = env.unwrapped.command_manager.get_term("motion")
    use, paths = _pin_clips(env, mask_name=mask_name)
    body_names = list(cmd.cfg.body_names)
    vis_i = [body_names.index(n) for n in KP_VIS]
    vt, vl, vr = _intent_vis_flags(mask_name)
    vis_flags = (vt, vl, vr)
    ee_i = [body_names.index(n) for n in EE_FAIL_BODIES if n in body_names]
    torso_i = body_names.index("torso_link")
    ankle_l_i = body_names.index("left_ankle_roll_link") if "left_ankle_roll_link" in body_names else None
    ankle_r_i = body_names.index("right_ankle_roll_link") if "right_ankle_roll_link" in body_names else None
    asset = env.unwrapped.scene["robot"]
    gravity = asset.data.GRAVITY_VEC_W
    if gravity.ndim == 1:
        gravity = gravity.unsqueeze(0).expand(cmd.num_envs, -1)

    rec = {
        "e_torso": [[] for _ in range(use)],
        "e_lw": [[] for _ in range(use)],
        "e_rw": [[] for _ in range(use)],
        "e_torso_xy": [[] for _ in range(use)],
        "e_torso_z": [[] for _ in range(use)],
        "e_lw_xy": [[] for _ in range(use)],
        "e_lw_z": [[] for _ in range(use)],
        "e_rw_xy": [[] for _ in range(use)],
        "e_rw_z": [[] for _ in range(use)],
        "ori": [[] for _ in range(use)],
        "roll": [[] for _ in range(use)],
        "pitch": [[] for _ in range(use)],
        "yaw": [[] for _ in range(use)],
        "act_d": [[] for _ in range(use)],
        "dz": [[] for _ in range(use)],
        "h01": [[] for _ in range(use)],
        "h02": [[] for _ in range(use)],
        "h03": [[] for _ in range(use)],
        "h05": [[] for _ in range(use)],
        "ankle_l": [[] for _ in range(use)],
        "ankle_r": [[] for _ in range(use)],
        "root_z": [[] for _ in range(use)],
        "torso_xyz": [[] for _ in range(use)],
        "lw_xyz": [[] for _ in range(use)],
        "rw_xyz": [[] for _ in range(use)],
        "goal_torso": [[] for _ in range(use)],
        "goal_lw": [[] for _ in range(use)],
        "goal_rw": [[] for _ in range(use)],
        "fail_step": [None] * use,
        "fail_reason": [""] * use,
        "tang": [[] for _ in range(use)],
        "tup": [[] for _ in range(use)],
        "tsens": [[] for _ in range(use)],
    }
    prev_act = None
    prev_z = None
    scan_mode = str(getattr(args, "scan_mode", "normal") or "normal").strip().lower()
    scan_dim = int(getattr(args, "scan_dim", 187) or 0)
    scan_perm = None

    for t in range(steps):
        with torch.inference_mode():
            obs, _ = env.get_observations()
            obs = injector.patch_policy_obs(obs, env, mode)
            obs, scan_perm = _apply_scan_mode(obs, scan_mode, scan_dim, scan_perm)
            if (
                t == 50
                and scan_mode == "normal"
                and scan_dim > 0
                and int(obs.shape[-1]) >= scan_dim
            ):
                terrain_key = str(getattr(args, "terrain", "") or "")
                if terrain_key and terrain_key not in _OBS_T50:
                    _OBS_T50[terrain_key] = obs.detach().cpu()
            actions = policy(obs)
            if policy_mod is not None and hasattr(policy_mod, "last_terrain_stats"):
                st = policy_mod.last_terrain_stats()
                if st is not None:
                    for e in range(use):
                        rec["tang"][e].append(float(st["ang_deg"][e].item()))
                        rec["tup"][e].append(float(st["u_perp"][e].item()))
                        rec["tsens"][e].append(float(st["scan_sens"][e].item()))
            if mode != "oracle":
                ferr = injector.future_errors(env, injector.last_pred_abs)
            else:
                ferr = injector.future_errors(env, None)
                # Oracle: GT future in obs; report ~0 by comparing GT to itself after gather.
            env.step(actions)

        robot = cmd.robot_body_pos_w[:use]
        goal = cmd.body_pos_w[:use]
        delta = robot - goal
        e_torso = torch.linalg.norm(delta[:, vis_i[0]], dim=-1)
        e_lw = torch.linalg.norm(delta[:, vis_i[1]], dim=-1)
        e_rw = torch.linalg.norm(delta[:, vis_i[2]], dim=-1)
        xy = delta[:, vis_i, :2]
        zz = delta[:, vis_i, 2]
        e_torso_xy = torch.linalg.norm(xy[:, 0], dim=-1)
        e_lw_xy = torch.linalg.norm(xy[:, 1], dim=-1)
        e_rw_xy = torch.linalg.norm(xy[:, 2], dim=-1)

        ori = quat_error_magnitude(cmd.anchor_quat_w[:use], cmd.robot_anchor_quat_w[:use])
        roll, pitch, yaw = _rel_rpy(cmd.anchor_quat_w[:use], cmd.robot_anchor_quat_w[:use])

        offset = _ankle_mean_z_offset(cmd)[:use]
        adj_z = (cmd.anchor_pos_w[:use, -1] - cmd.robot_anchor_pos_w[:use, -1]) + offset
        fail_z = adj_z.abs() > args.fail_anchor_z
        mot_g = quat_rotate_inverse(cmd.anchor_quat_w, gravity)[:use]
        rob_g = quat_rotate_inverse(cmd.robot_anchor_quat_w, gravity)[:use]
        fail_ori = (mot_g[:, 2] - rob_g[:, 2]).abs() > args.fail_anchor_ori
        ee_err = (cmd.body_pos_w[:use, ee_i, 2] - cmd.robot_body_pos_w[:use, ee_i, 2]).abs()
        fail_ee = ee_err.any(dim=-1) if ee_err.numel() else torch.zeros(use, dtype=torch.bool, device=robot.device)

        z_torso = robot[:, torso_i, 2]
        if prev_act is None:
            d_act = torch.zeros(use, device=actions.device)
            d_z = torch.zeros(use, device=robot.device)
        else:
            d_act = torch.linalg.norm((actions[:use] - prev_act).float(), dim=-1)
            d_z = (z_torso - prev_z).abs()
        prev_act = actions[:use].clone()
        prev_z = z_torso.clone()

        cpu = lambda x: x.detach().cpu()  # noqa: E731
        e_torso_c, e_lw_c, e_rw_c = cpu(e_torso), cpu(e_lw), cpu(e_rw)
        h01 = cpu(ferr["h_0.1"][:use]) if "h_0.1" in ferr else torch.zeros(use)
        h02 = cpu(ferr["h_0.2"][:use])
        h03 = cpu(ferr["h_0.3"][:use])
        h05 = cpu(ferr["h_0.5"][:use])

        for e in range(use):
            rec["e_torso"][e].append(float(e_torso_c[e]))
            rec["e_lw"][e].append(float(e_lw_c[e]))
            rec["e_rw"][e].append(float(e_rw_c[e]))
            rec["e_torso_xy"][e].append(float(cpu(e_torso_xy)[e]))
            rec["e_torso_z"][e].append(float(cpu(zz[:, 0].abs())[e]))
            rec["e_lw_xy"][e].append(float(cpu(e_lw_xy)[e]))
            rec["e_lw_z"][e].append(float(cpu(zz[:, 1].abs())[e]))
            rec["e_rw_xy"][e].append(float(cpu(e_rw_xy)[e]))
            rec["e_rw_z"][e].append(float(cpu(zz[:, 2].abs())[e]))
            rec["ori"][e].append(float(cpu(ori)[e]))
            rec["roll"][e].append(float(cpu(roll.abs())[e]))
            rec["pitch"][e].append(float(cpu(pitch.abs())[e]))
            rec["yaw"][e].append(float(cpu(yaw.abs())[e]))
            rec["act_d"][e].append(float(cpu(d_act)[e]))
            rec["dz"][e].append(float(cpu(d_z)[e]))
            rec["h01"][e].append(float(h01[e]))
            rec["h02"][e].append(float(h02[e]))
            rec["h03"][e].append(float(h03[e]))
            rec["h05"][e].append(float(h05[e]))
            if ankle_l_i is not None:
                rec["ankle_l"][e].append([float(robot[e, ankle_l_i, d].detach().cpu()) for d in range(3)])
            if ankle_r_i is not None:
                rec["ankle_r"][e].append([float(robot[e, ankle_r_i, d].detach().cpu()) for d in range(3)])
            rec["root_z"][e].append(float(asset.data.root_pos_w[e, 2].detach().cpu()))
            rec["torso_xyz"][e].append([float(robot[e, vis_i[0], d].detach().cpu()) for d in range(3)])
            rec["lw_xyz"][e].append([float(robot[e, vis_i[1], d].detach().cpu()) for d in range(3)])
            rec["rw_xyz"][e].append([float(robot[e, vis_i[2], d].detach().cpu()) for d in range(3)])
            rec["goal_torso"][e].append([float(goal[e, vis_i[0], d].detach().cpu()) for d in range(3)])
            rec["goal_lw"][e].append([float(goal[e, vis_i[1], d].detach().cpu()) for d in range(3)])
            rec["goal_rw"][e].append([float(goal[e, vis_i[2], d].detach().cpu()) for d in range(3)])
            if rec["fail_step"][e] is None and t >= 10:
                if bool(fail_z[e]):
                    rec["fail_step"][e] = t
                    rec["fail_reason"][e] = "anchor_z"
                elif bool(fail_ori[e]):
                    rec["fail_step"][e] = t
                    rec["fail_reason"][e] = "anchor_ori"
                # ee_z vs clip is a false positive (relative vs world). Merge overlay
                # uses vis_err>0.25m or ori>0.8 after t>=10.

        if (t + 1) % 50 == 0:
            parts = []
            if vt:
                parts.append(e_torso_c)
            if vl:
                parts.append(e_lw_c)
            if vr:
                parts.append(e_rw_c)
            vis = sum(parts) / max(len(parts), 1)
            print(
                f"[causal] {mode} mask={mask_name} step {t+1}/{steps} "
                f"e_vis={['%.3f' % float(vis[i]) for i in range(use)]}",
                flush=True,
            )

    rows = []
    for e in range(use):
        et = np.asarray(rec["e_torso"][e], dtype=np.float64)
        el = np.asarray(rec["e_lw"][e], dtype=np.float64)
        er = np.asarray(rec["e_rw"][e], dtype=np.float64)
        parts = []
        if vis_flags[0]:
            parts.append(et)
        if vis_flags[1]:
            parts.append(el)
        if vis_flags[2]:
            parts.append(er)
        e_vis = np.mean(np.stack(parts, 0), 0) if parts else (et + el + er) / 3.0
        e_xy = (
            np.asarray(rec["e_torso_xy"][e]) + np.asarray(rec["e_lw_xy"][e]) + np.asarray(rec["e_rw_xy"][e])
        ) / 3.0
        e_z = (
            np.asarray(rec["e_torso_z"][e]) + np.asarray(rec["e_lw_z"][e]) + np.asarray(rec["e_rw_z"][e])
        ) / 3.0
        wparts = []
        if vis_flags[1]:
            wparts.append(el)
        if vis_flags[2]:
            wparts.append(er)
        wrist = np.mean(np.stack(wparts, 0), 0) if wparts else np.full_like(et, np.nan)
        fail = rec["fail_step"][e] is not None
        ep_len = (rec["fail_step"][e] + 1) if fail else steps
        rows.append(
            {
                "path": paths[e],
                "clip": Path(paths[e]).name,
                "series": {
                    "e_vis": e_vis,
                    "e_torso": et,
                    "e_lw": el,
                    "e_rw": er,
                    "wrist": wrist,
                    "ori": np.asarray(rec["ori"][e]),
                    "roll": np.asarray(rec["roll"][e]),
                    "pitch": np.asarray(rec["pitch"][e]),
                    "ankle_l": np.asarray(rec["ankle_l"][e], dtype=np.float64) if rec["ankle_l"][e] else np.zeros((0, 3)),
                    "ankle_r": np.asarray(rec["ankle_r"][e], dtype=np.float64) if rec["ankle_r"][e] else np.zeros((0, 3)),
                    "root_z": np.asarray(rec["root_z"][e], dtype=np.float64),
                    "torso_xyz": np.asarray(rec["torso_xyz"][e], dtype=np.float64),
                    "lw_xyz": np.asarray(rec["lw_xyz"][e], dtype=np.float64),
                    "rw_xyz": np.asarray(rec["rw_xyz"][e], dtype=np.float64),
                    "goal_torso": np.asarray(rec["goal_torso"][e], dtype=np.float64),
                    "goal_lw": np.asarray(rec["goal_lw"][e], dtype=np.float64),
                    "goal_rw": np.asarray(rec["goal_rw"][e], dtype=np.float64),
                },
                "sr_5cm": float((e_vis < 0.05).mean()),
                "sr_2cm": float((e_vis < 0.02).mean()),
                "e_kp_mean": float(e_vis.mean()),
                "e_kp_p50": _percentile(e_vis, 50),
                "e_kp_p90": _percentile(e_vis, 90),
                "e_torso": float(et.mean()),
                "e_lw": float(el.mean()),
                "e_rw": float(er.mean()),
                "e_xy": float(e_xy.mean()),
                "e_z": float(e_z.mean()),
                "e_torso_xy": float(np.mean(rec["e_torso_xy"][e])),
                "e_torso_z": float(np.mean(rec["e_torso_z"][e])),
                "e_lw_xy": float(np.mean(rec["e_lw_xy"][e])),
                "e_lw_z": float(np.mean(rec["e_lw_z"][e])),
                "e_rw_xy": float(np.mean(rec["e_rw_xy"][e])),
                "e_rw_z": float(np.mean(rec["e_rw_z"][e])),
                "wrist_err_mean": _nanmean(wrist),
                "wrist_err_p90": _percentile(wrist, 90),
                "ori_err_rad": float(np.mean(rec["ori"][e])),
                "ori_roll_rad": float(np.mean(rec["roll"][e])),
                "ori_pitch_rad": float(np.mean(rec["pitch"][e])),
                "ori_yaw_rad": float(np.mean(rec["yaw"][e])),
                "fail": int(fail),
                "fail_reason": rec["fail_reason"][e] or "none",
                "episode_length": int(ep_len),
                "action_smoothness": float(np.mean(rec["act_d"][e])),
                "abs_dz": float(np.mean(rec["dz"][e])),
                "mapper_err_0p1": float(np.mean(rec["h01"][e])),
                "mapper_err_0p2": float(np.mean(rec["h02"][e])),
                "mapper_err_0p3": float(np.mean(rec["h03"][e])),
                "mapper_err_0p5": float(np.mean(rec["h05"][e])),
                "residual_alpha": 0.0 if mode.endswith("_norest") else float(args.residual_alpha),
                "terrain_ang_deg": float(np.mean(rec["tang"][e])) if rec["tang"][e] else float("nan"),
                "terrain_u_perp": float(np.mean(rec["tup"][e])) if rec["tup"][e] else float("nan"),
                "terrain_scan_sens": float(np.mean(rec["tsens"][e])) if rec["tsens"][e] else float("nan"),
            }
        )
    return rows


def _write_csv(
    path: Path, rows: list[dict], mode: str, terrain: str, seed: int, mask: str, steps: int, task: str = ""
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "mode": mode,
                    "terrain": terrain,
                    "clip": r["clip"],
                    "task": task,
                    "seed": seed,
                    "mask": mask,
                    "n_steps": steps,
                    **{
                        k: r[k]
                        for k in CSV_FIELDS
                        if k not in ("mode", "terrain", "clip", "task", "seed", "mask", "n_steps")
                    },
                }
            )


def _mean_row(rows: list[dict]) -> dict:
    keys = [
        "sr_5cm",
        "sr_2cm",
        "e_kp_mean",
        "e_kp_p50",
        "e_kp_p90",
        "e_torso",
        "e_lw",
        "e_rw",
        "e_xy",
        "e_z",
        "wrist_err_mean",
        "wrist_err_p90",
        "ori_err_rad",
        "ori_roll_rad",
        "ori_pitch_rad",
        "fail",
        "episode_length",
        "action_smoothness",
        "abs_dz",
        "mapper_err_0p1",
        "mapper_err_0p2",
        "mapper_err_0p3",
        "mapper_err_0p5",
        "terrain_ang_deg",
        "terrain_u_perp",
        "terrain_scan_sens",
    ]
    out = {k: _nanmean([r[k] for r in rows if k in r]) for k in keys}
    reasons = {}
    for r in rows:
        fr = r.get("fail_reason")
        if not fr:
            continue
        reasons[fr] = reasons.get(fr, 0) + 1
    if reasons:
        out["fail_reasons"] = reasons
    out["n_clips"] = int(sum(r.get("n_clips", 1) for r in rows))
    return out


CANONICAL_PROBE_MASKS = {
    "loco": "torso",
    "reach": "head_left,head_right",
    "stoop": "vr",
    "carry": "vr",
}

# Raw policy obs at t=50, keyed by terrain. Used for scan-direction counterfactual.
_OBS_T50: dict[str, "torch.Tensor"] = {}


def _unit(x: "torch.Tensor", dim: int = -1) -> "torch.Tensor":
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-8)


def _angle_deg(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    c = (_unit(a) * _unit(b)).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(c) * (180.0 / math.pi)


def _apply_scan_mode(obs: "torch.Tensor", scan_mode: str, scan_dim: int, perm: "torch.Tensor | None"):
    """Zero or cross-env-shuffle the trailing height-scan block. Core obs unchanged."""
    mode = (scan_mode or "normal").strip().lower()
    if mode in ("", "normal") or int(scan_dim) <= 0 or int(obs.shape[-1]) < int(scan_dim):
        return obs, perm
    obs = obs.clone()
    if mode == "zero":
        obs[..., -int(scan_dim) :] = 0
        return obs, perm
    if mode == "shuffle":
        if perm is None:
            n = int(obs.shape[0])
            perm = (torch.arange(n, device=obs.device) + 1) % n
        obs[..., -int(scan_dim) :] = obs[..., -int(scan_dim) :][perm]
        return obs, perm
    return obs, perm


def _run_scan_counterfactual(runner, policy_mod, out_dir: Path) -> None:
    """Same Flat core (z_nom / K / proprio); swap Slope / Steps scans onto it."""
    need = ("plane", "slope", "steps")
    if not all(k in _OBS_T50 for k in need):
        return
    scan_dim = int(getattr(policy_mod, "terrain_scan_dim", 0) or 0)
    if scan_dim <= 0:
        return
    device = next(policy_mod.parameters()).device
    n = min(int(_OBS_T50["plane"].shape[0]), 8)
    core = _OBS_T50["plane"].to(device)[:n, :-scan_dim]
    scans = {k: _OBS_T50[k].to(device)[:n, -scan_dim:] for k in need}
    us: dict[str, torch.Tensor] = {}
    stats: dict[str, dict] = {}
    with torch.inference_mode():
        for name, scan in scans.items():
            obs = torch.cat([core, scan], dim=-1)
            obs_n = runner._normalize_student_obs(obs)
            policy_mod._encode_mean_latent(obs_n)
            cache = getattr(policy_mod, "_last_terrain", None)
            if cache is None:
                return
            us[name] = cache["u_perp"]
            st = policy_mod.last_terrain_stats()
            stats[name] = {
                "u_perp_mean": float(st["u_perp"].mean().cpu()),
                "ang_deg_mean": float(st["ang_deg"].mean().cpu()),
                "scan_sens_mean": float(st["scan_sens"].mean().cpu()),
            }
    payload = {
        "note": "Fixed Flat core obs at t=50; replace trailing scan only.",
        "n": n,
        "stats": stats,
        "angle_u_flat_slope_deg": float(_angle_deg(us["plane"], us["slope"]).mean().cpu()),
        "angle_u_flat_steps_deg": float(_angle_deg(us["plane"], us["steps"]).mean().cpu()),
        "angle_u_slope_steps_deg": float(_angle_deg(us["slope"], us["steps"]).mean().cpu()),
        "cos_u_flat_slope": float((_unit(us["plane"]) * _unit(us["slope"])).sum(-1).mean().cpu()),
        "cos_u_flat_steps": float((_unit(us["plane"]) * _unit(us["steps"])).sum(-1).mean().cpu()),
        "cos_u_slope_steps": float((_unit(us["slope"]) * _unit(us["steps"])).sum(-1).mean().cpu()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scan_counterfactual.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[causal] scan counterfactual {payload}", flush=True)
    print(f"[causal] wrote {path}", flush=True)


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


def _eval_one_cell(env_cfg, agent_cfg, resume_path: str, motion_dir: str, terrain: str, task_name: str, masks: list[str]) -> dict:
    n_files = len(list(Path(motion_dir).rglob("*.npz")))
    if n_files == 0:
        raise SystemExit(f"no npz in {motion_dir}")
    n_want = int(args_cli.num_envs) if args_cli.num_envs and int(args_cli.num_envs) > 0 else n_files
    env_cfg.scene.num_envs = min(n_want, n_files)
    env_cfg.seed = int(args_cli.seed)
    agent_cfg.seed = int(args_cli.seed)
    args_cli.terrain = terrain
    _apply_eval_motion(env_cfg, motion_dir, args_cli.start_frame)
    _apply_terrain(env_cfg, terrain)
    if not masks:
        masks = ["vr"]
    if args_cli.keep_headhands_spec:
        print(f"[causal] keep HeadHands spec; pin masks={masks}", flush=True)
    else:
        if len(masks) != 1:
            raise SystemExit("multiple --mask_modes require --keep_headhands_spec")
        from whole_body_tracking.tasks.tracking.config.g1.mask_modes import eval_single_mode_spec

        spec, probs = eval_single_mode_spec(masks[0])
        env_cfg.commands.motion.mask_mode_spec = spec
        env_cfg.commands.motion.mask_mode_probs = probs
        print(f"[causal] single-mode spec {masks[0]!r}", flush=True)
    _disable_eval_hooks(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    from whole_body_tracking.tasks.tracking.mdp.curriculums import attach_curriculum_rollout_hints

    attach_curriculum_rollout_hints(
        env, spe=int(agent_cfg.num_steps_per_env), learning_iteration=int(args_cli.hint_iter)
    )
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False, load_critic=True)
    if bool(getattr(runner.alg.policy, "intent_recovery", False)):
        runner.alg.policy._recovery_obs_nrm = [runner.obs_normalizer]
    policy = runner.get_inference_policy(device=env.device)
    policy_mod = runner.alg.policy
    mapper_paths = {}
    if args_cli.mapper_path:
        mapper_paths["mapper"] = args_cli.mapper_path
    if args_cli.mapper_a_path:
        mapper_paths["mapper_a"] = args_cli.mapper_a_path
    if args_cli.mapper_b_path:
        mapper_paths["mapper_b"] = args_cli.mapper_b_path
        mapper_paths.setdefault("mapper", args_cli.mapper_b_path)
    elif args_cli.mapper_path:
        mapper_paths.setdefault("mapper_b", args_cli.mapper_path)
    injector = CausalFutureInjector(mappers=mapper_paths, device=env.device)

    modes = [m.strip().lower() for m in args_cli.modes.split(",") if m.strip()]
    seeds = (
        [int(s) for s in args_cli.seeds.split(",") if s.strip()]
        if args_cli.seeds.strip()
        else [int(args_cli.seed)]
    )
    out = Path(args_cli.out)
    if task_name:
        out = out / task_name
    out = out / terrain
    out.mkdir(parents=True, exist_ok=True)
    (out / "curves").mkdir(exist_ok=True)

    summaries: dict = {}
    for mode in modes:
        alpha = 0.0 if mode.endswith("_norest") else float(args_cli.residual_alpha)
        _set_residual_alpha(policy_mod, alpha)
        summaries[mode] = {}
        for mask in masks:
            for seed in seeds:
                print(
                    f"[causal] ===== mode={mode} task={task_name or '-'} "
                    f"terrain={terrain} mask={mask} seed={seed} residual_alpha={alpha} "
                    f"scan={getattr(args_cli, 'scan_mode', 'normal')} =====",
                    flush=True,
                )
                torch.manual_seed(seed)
                np.random.seed(seed)
                rows = _rollout(
                    env, policy, injector, mode, int(args_cli.steps), args_cli, mask_name=mask, policy_mod=policy_mod
                )
                csv_name = f"{mode}_{mask}_s{seed}.csv"
                _write_csv(
                    out / csv_name,
                    rows,
                    mode,
                    terrain,
                    seed,
                    mask,
                    int(args_cli.steps),
                    task=task_name,
                )
                for r in rows:
                    np.savez_compressed(
                        out / "curves" / f"{mode}_{mask}_s{seed}_{Path(r['clip']).stem}.npz",
                        **r["series"],
                    )
                cell = _mean_row(rows)
                summaries[mode].setdefault(mask, {})[f"s{seed}"] = cell
                print(
                    f"[causal] {mode}/{mask}/s{seed} SR@5={cell['sr_5cm']:.3f} "
                    f"wrist={cell['wrist_err_mean']:.4f} P90w={cell['wrist_err_p90']:.4f} "
                    f"ori={cell['ori_err_rad']:.3f} fail={cell['fail']:.3f} "
                    f"ang={cell.get('terrain_ang_deg', float('nan')):.2f} "
                    f"u_perp={cell.get('terrain_u_perp', float('nan')):.4f} "
                    f"scan={getattr(args_cli, 'scan_mode', 'normal')}",
                    flush=True,
                )
            pooled_rows = []
            for seed in seeds:
                pooled_rows.append(summaries[mode][mask][f"s{seed}"])
            summaries[mode][mask]["pooled"] = _mean_row(pooled_rows)

    payload = {
        "ckpt": resume_path,
        "mapper": args_cli.mapper_path,
        "mapper_a": args_cli.mapper_a_path,
        "mapper_b": args_cli.mapper_b_path or args_cli.mapper_path,
        "residual_alpha_default": float(args_cli.residual_alpha),
        "terrain": terrain,
        "task": task_name,
        "masks": masks,
        "seeds": seeds,
        "steps": int(args_cli.steps),
        "n_envs": int(env_cfg.scene.num_envs),
        "scan_mode": str(getattr(args_cli, "scan_mode", "normal")),
        "modes": summaries,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    scan_mode = str(getattr(args_cli, "scan_mode", "normal") or "normal").strip().lower()
    if scan_mode == "normal" and task_name == "loco" and terrain == "steps":
        _run_scan_counterfactual(runner, policy_mod, Path(args_cli.out))
    env.close()
    print(f"[causal] wrote {out}", flush=True)
    return payload


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    direct = getattr(agent_cfg, "resume_checkpoint_path", None)
    if direct:
        resume_path = os.path.abspath(str(direct))
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[causal] ckpt {resume_path}", flush=True)
    _restore_adapter(agent_cfg, resume_path)

    probe_tasks = [t.strip() for t in args_cli.probe_tasks.split(",") if t.strip()]
    probe_terrains = [t.strip() for t in args_cli.probe_terrains.split(",") if t.strip()]
    if probe_tasks:
        clip_root = Path(args_cli.probe_clip_root or args_cli.motion)
        terrains = probe_terrains or [args_cli.terrain]
        print(f"[causal] probe tasks={probe_tasks} terrains={terrains} clip_root={clip_root}", flush=True)
        for task_name in probe_tasks:
            motion_dir = str(clip_root / task_name)
            masks_s = CANONICAL_PROBE_MASKS.get(task_name, args_cli.mask_modes)
            if task_name == "reach" and str(getattr(args_cli, "probe_reach_mask", "") or "").strip():
                masks_s = str(args_cli.probe_reach_mask).strip()
            masks = [m.strip() for m in masks_s.split(",") if m.strip()]
            for terrain in terrains:
                print(f"[causal] PROBE cell task={task_name} terrain={terrain} masks={masks}", flush=True)
                _eval_one_cell(env_cfg, agent_cfg, resume_path, motion_dir, terrain, task_name, masks)
        print("[causal] probe done", flush=True)
        return

    motion_dir = args_cli.motion
    n_files = len(list(Path(motion_dir).rglob("*.npz")))
    if n_files == 0:
        raise SystemExit(f"no npz in {motion_dir}")
    n_want = int(args_cli.num_envs) if args_cli.num_envs and int(args_cli.num_envs) > 0 else n_files
    env_cfg.scene.num_envs = min(n_want, n_files)
    env_cfg.seed = int(args_cli.seed)
    agent_cfg.seed = int(args_cli.seed)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    direct = getattr(agent_cfg, "resume_checkpoint_path", None)
    if direct:
        resume_path = os.path.abspath(str(direct))
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[causal] ckpt {resume_path}", flush=True)
    print(f"[causal] motion={motion_dir} n_npz={n_files} num_envs={env_cfg.scene.num_envs}", flush=True)
    _restore_adapter(agent_cfg, resume_path)
    _apply_eval_motion(env_cfg, motion_dir, args_cli.start_frame)
    _apply_terrain(env_cfg, args_cli.terrain)

    masks = [m.strip() for m in args_cli.mask_modes.split(",") if m.strip()]
    if not masks:
        masks = ["vr"]
    if args_cli.keep_headhands_spec:
        print(f"[causal] keep HeadHands spec; pin masks={masks}", flush=True)
    else:
        if len(masks) != 1:
            raise SystemExit("multiple --mask_modes require --keep_headhands_spec")
        from whole_body_tracking.tasks.tracking.config.g1.mask_modes import eval_single_mode_spec

        spec, probs = eval_single_mode_spec(masks[0])
        env_cfg.commands.motion.mask_mode_spec = spec
        env_cfg.commands.motion.mask_mode_probs = probs
        print(f"[causal] single-mode spec {masks[0]!r}", flush=True)
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

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    from whole_body_tracking.tasks.tracking.mdp.curriculums import attach_curriculum_rollout_hints

    attach_curriculum_rollout_hints(
        env, spe=int(agent_cfg.num_steps_per_env), learning_iteration=int(args_cli.hint_iter)
    )
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False, load_critic=True)
    if bool(getattr(runner.alg.policy, "intent_recovery", False)):
        runner.alg.policy._recovery_obs_nrm = [runner.obs_normalizer]
    policy = runner.get_inference_policy(device=env.device)
    policy_mod = runner.alg.policy
    mapper_paths = {}
    if args_cli.mapper_path:
        mapper_paths["mapper"] = args_cli.mapper_path
    if args_cli.mapper_a_path:
        mapper_paths["mapper_a"] = args_cli.mapper_a_path
    if args_cli.mapper_b_path:
        mapper_paths["mapper_b"] = args_cli.mapper_b_path
        mapper_paths.setdefault("mapper", args_cli.mapper_b_path)
    elif args_cli.mapper_path:
        mapper_paths.setdefault("mapper_b", args_cli.mapper_path)
    injector = CausalFutureInjector(mappers=mapper_paths, device=env.device)

    modes = [m.strip().lower() for m in args_cli.modes.split(",") if m.strip()]
    seeds = (
        [int(s) for s in args_cli.seeds.split(",") if s.strip()]
        if args_cli.seeds.strip()
        else [int(args_cli.seed)]
    )
    task_name = (args_cli.task_name or "").strip()
    out = Path(args_cli.out)
    if task_name:
        out = out / task_name
    out = out / args_cli.terrain
    out.mkdir(parents=True, exist_ok=True)
    (out / "curves").mkdir(exist_ok=True)

    summaries: dict = {}
    for mode in modes:
        alpha = 0.0 if mode.endswith("_norest") else float(args_cli.residual_alpha)
        _set_residual_alpha(policy_mod, alpha)
        summaries[mode] = {}
        for mask in masks:
            for seed in seeds:
                print(
                    f"[causal] ===== mode={mode} task={task_name or '-'} "
                    f"terrain={args_cli.terrain} mask={mask} seed={seed} residual_alpha={alpha} =====",
                    flush=True,
                )
                torch.manual_seed(seed)
                np.random.seed(seed)
                rows = _rollout(
                    env, policy, injector, mode, int(args_cli.steps), args_cli, mask_name=mask, policy_mod=policy_mod
                )
                csv_name = f"{mode}_{mask}_s{seed}.csv"
                _write_csv(
                    out / csv_name,
                    rows,
                    mode,
                    args_cli.terrain,
                    seed,
                    mask,
                    int(args_cli.steps),
                    task=task_name,
                )
                for r in rows:
                    np.savez_compressed(
                        out / "curves" / f"{mode}_{mask}_s{seed}_{Path(r['clip']).stem}.npz",
                        **r["series"],
                    )
                cell = _mean_row(rows)
                summaries[mode].setdefault(mask, {})[f"s{seed}"] = cell
                print(
                    f"[causal] {mode}/{mask}/s{seed} SR@5={cell['sr_5cm']:.3f} "
                    f"wrist={cell['wrist_err_mean']:.4f} P90w={cell['wrist_err_p90']:.4f} "
                    f"ori={cell['ori_err_rad']:.3f} fail={cell['fail']:.3f} "
                    f"ang={cell.get('terrain_ang_deg', float('nan')):.2f} "
                    f"u_perp={cell.get('terrain_u_perp', float('nan')):.4f} "
                    f"scan={getattr(args_cli, 'scan_mode', 'normal')}",
                    flush=True,
                )
            pooled_rows = []
            for seed in seeds:
                pooled_rows.append(summaries[mode][mask][f"s{seed}"])
            summaries[mode][mask]["pooled"] = _mean_row(pooled_rows)

    payload = {
        "ckpt": resume_path,
        "mapper": args_cli.mapper_path,
        "mapper_a": args_cli.mapper_a_path,
        "mapper_b": args_cli.mapper_b_path or args_cli.mapper_path,
        "residual_alpha_default": float(args_cli.residual_alpha),
        "terrain": args_cli.terrain,
        "task": task_name,
        "masks": masks,
        "seeds": seeds,
        "steps": int(args_cli.steps),
        "n_envs": int(env_cfg.scene.num_envs),
        "modes": summaries,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    env.close()
    print(f"[causal] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
