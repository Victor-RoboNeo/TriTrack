#!/usr/bin/env python3
"""SIRAC Phase-1 Isaac eval: Mapper-B + Stage-2 parent, with optional 7-D LBC transplant.

Baseline A: Stage-2 29-DoF actions (verified parent).
Baseline B: Stage-2 upper + extracted 7-D command + frozen HTD LBC lower/waist.
Baseline C: B with arms held at HTD zeros.

Does not modify eval_irr_r3_short.py or frozen P4 artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

ROOT = Path("/data/home/chenxiangyu/robotics/Anybody")
MAPPER_B = "/data/home/chenxiangyu/victor/TriTrack/runs/mapper_b_intent72/mapper_best.pt"
PARENT_CKPT = (
    ROOT
    / "logs/rsl_rl/g1_flat_muse_kp_latent_rl"
    / "2026-08-26_00-38-10_tritrack_headhands_locomani_from35000"
    / "model_50000.pt"
)
CLIP_LOCO = ROOT / "logs/tritrack/infer_clips/p1/loco"
KP_VIS = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
EE_FAIL = ("left_wrist_yaw_link", "right_wrist_yaw_link")
DT = 0.02

parser = argparse.ArgumentParser(description="SIRAC Phase-1 Isaac A/B/C eval.")
parser.add_argument("--baseline", type=str, required=True, choices=("A_stage2_direct", "B_realization_lbc", "C_static_arms_lbc"))
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--task", type=str, default="MUSE-Kp-LatentRL-Kp5-HeadHands-Locomani-G1-v0")
parser.add_argument("--motion", type=str, default=str(CLIP_LOCO))
parser.add_argument("--mask_modes", type=str, default="vr")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out", type=str, default=str(ROOT / "results/sirac_phase1/isaac"))
parser.add_argument("--start_frame", type=int, default=10)
parser.add_argument("--hint_iter", type=int, default=50000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--terrain", type=str, default="plane", choices=("plane", "light_rough", "slope", "steps"))
parser.add_argument("--keep_headhands_spec", action="store_true", default=True)
parser.add_argument("--mapper_path", type=str, default=MAPPER_B)
parser.add_argument("--mapper_b_path", type=str, default=MAPPER_B)
parser.add_argument("--residual_alpha", type=float, default=1.0)
parser.add_argument("--fail_anchor_z", type=float, default=0.25)
parser.add_argument("--fail_anchor_ori", type=float, default=0.8)
parser.add_argument("--fail_ee_z", type=float, default=0.25)
parser.add_argument("--jit", type=str, default="")
parser.add_argument("--task_name", type=str, default="loco")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False
sys.argv = [sys.argv[0]] + hydra_args

_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if _cvd:
    try:
        _phys = int(_cvd.split(",")[0].strip())
        sys.argv.append(f"--/renderer/activeGpu={_phys}")
    except ValueError:
        pass

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
sys.argv = [a for a in sys.argv if not a.startswith("--/renderer/activeGpu=")]

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_error_magnitude, quat_rotate_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import whole_body_tracking.tasks  # noqa: E402, F401
from causal_future import CausalFutureInjector  # noqa: E402
from whole_body_tracking.sirac.command_extract import extract_realization_command  # noqa: E402
from whole_body_tracking.sirac.lower_body_controller import (  # noqa: E402
    LowerBodyRealizationController,
    action_to_q_target,
)
from whole_body_tracking.sirac.mappings import arm_indices_in, lower_indices_in  # noqa: E402
from whole_body_tracking.sirac.metrics import EpisodeMetrics, write_csv, write_json  # noqa: E402
from whole_body_tracking.tasks.tracking.mdp.rewards import _ankle_mean_z_offset  # noqa: E402


def _restore_adapter(agent_cfg, resume_path: str) -> None:
    params = Path(resume_path).parent / "params" / "agent.yaml"
    if not params.exists():
        return
    cfg = yaml.safe_load(params.read_text()) or {}
    policy_cfg = cfg.get("policy") or {}
    for key, val in policy_cfg.items():
        if hasattr(agent_cfg.policy, key):
            setattr(agent_cfg.policy, key, val)


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
        return
    from isaaclab.terrains import (
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
    if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None and hasattr(env_cfg.curriculum, "terrain_levels"):
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


def _pin_clips(env, mask_name: str | None = None):
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
        if mask_name and hasattr(cmd, "set_eval_fixed_mask_mode_idx"):
            names = [str(x) for x in getattr(cmd, "_mode_names", ())]
            if mask_name in names:
                cmd.set_eval_fixed_mask_mode_idx(names.index(mask_name))
                print(f"[sirac] pin mask {mask_name!r}", flush=True)
        cmd._resample_command(ids)
        if hasattr(cmd, "resample_all_mask_modes"):
            cmd.resample_all_mask_modes()
        try:
            env.unwrapped.sim.forward()
        except Exception as exc:
            print(f"[sirac] sim.forward skipped: {exc}", flush=True)
    return use, paths[:use]


def _set_residual_alpha(policy_mod, alpha: float) -> None:
    rc = getattr(policy_mod, "residual_corrector", None)
    if rc is None:
        return
    rc.alpha = float(alpha)


def _action_scale_offset(env):
    term = env.unwrapped.action_manager.get_term("joint_pos")
    scale = term._scale
    offset = term._offset
    if scale.ndim == 1:
        scale = scale.unsqueeze(0)
    if offset.ndim == 1:
        offset = offset.unsqueeze(0)
    return scale, offset


def _mix_lbc_action(
    stage2_action: torch.Tensor,
    env,
    lbc: LowerBodyRealizationController,
    last_a15: np.ndarray,
    *,
    static_arms: bool,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Replace lower/waist of Stage-2 action with frozen HTD LBC targets."""
    asset = env.unwrapped.scene["robot"]
    cmd = env.unwrapped.command_manager.get_term("motion")
    names = list(asset.data.joint_names)
    li = lower_indices_in(names)
    ai = arm_indices_in(names)
    scale, offset = _action_scale_offset(env)
    n = int(stage2_action.shape[0])
    if scale.shape[0] == 1:
        scale = scale.expand(n, -1)
    if offset.shape[0] == 1:
        offset = offset.expand(n, -1)
    q_s2 = offset + stage2_action * scale
    body_names = list(cmd.cfg.body_names)
    ti = body_names.index("torso_link")
    # Nominal Stage-2 kinematics from the motion command; pelvis yaw from the live robot
    # (virtual-anchor heading). Command is robot-owned, not a human joystick.
    cmd7 = extract_realization_command(
        pelvis_quat_w=asset.data.root_quat_w[:n].detach().cpu().numpy(),
        pelvis_lin_vel_w=cmd.body_lin_vel_w[:n, ti].detach().cpu().numpy(),
        pelvis_ang_vel_w=cmd.body_ang_vel_w[:n, ti].detach().cpu().numpy(),
        torso_pos_w=cmd.body_pos_w[:n, ti].detach().cpu().numpy(),
        torso_quat_w=cmd.body_quat_w[:n, ti].detach().cpu().numpy(),
        clip=True,
        deploy_clip=True,
    )
    jp = asset.data.joint_pos[:n].detach().cpu().numpy()
    jv = asset.data.joint_vel[:n].detach().cpu().numpy()
    a15 = lbc(
        None,
        cmd7,
        ang_vel_b=asset.data.root_ang_vel_b[:n].detach().cpu().numpy(),
        root_quat_w=asset.data.root_quat_w[:n].detach().cpu().numpy(),
        joint_pos_lower=jp[:, li],
        joint_vel_lower=jv[:, li],
        last_action=last_a15,
    )
    q_lbc = action_to_q_target(a15)
    q = q_s2.detach().cpu().numpy()
    q[:, li] = q_lbc
    if static_arms:
        q[:, ai] = 0.0
    q_t = torch.as_tensor(q, device=stage2_action.device, dtype=stage2_action.dtype)
    sc = scale[:n].clamp_min(1e-6)
    mixed = (q_t - offset[:n]) / sc
    return mixed, np.asarray(a15, dtype=np.float32), np.asarray(cmd7, dtype=np.float64)


@torch.inference_mode()
def _rollout(env, policy, injector, lbc, baseline: str, steps: int, mask_name: str) -> list[EpisodeMetrics]:
    cmd = env.unwrapped.command_manager.get_term("motion")
    use, paths = _pin_clips(env, mask_name=mask_name)
    body_names = list(cmd.cfg.body_names)
    vis_i = [body_names.index(n) for n in KP_VIS]
    ee_i = [body_names.index(n) for n in EE_FAIL if n in body_names]
    asset = env.unwrapped.scene["robot"]
    gravity = asset.data.GRAVITY_VEC_W
    if gravity.ndim == 1:
        gravity = gravity.unsqueeze(0).expand(cmd.num_envs, -1)
    static_arms = baseline == "C_static_arms_lbc"
    use_lbc = baseline != "A_stage2_direct"
    if use_lbc and lbc.is_dummy:
        raise SystemExit("refusing Isaac B/C with DummyHoldPolicy; JIT must load")

    last_a15 = np.zeros((int(env.unwrapped.num_envs), 15), dtype=np.float32)
    e_head, e_lw, e_rw = [[] for _ in range(use)], [[] for _ in range(use)], [[] for _ in range(use)]
    e_head_r, e_lw_r, e_rw_r = [[] for _ in range(use)], [[] for _ in range(use)], [[] for _ in range(use)]
    act_d = [[] for _ in range(use)]
    cmd_track = [[] for _ in range(use)]
    fail_step = [None] * use
    fail_reason = [""] * use
    prev_act = None
    t0 = time.perf_counter()
    lat = []

    for t in range(steps):
        t1 = time.perf_counter()
        obs, _ = env.get_observations()
        obs = injector.patch_policy_obs(obs, env, "mapper")
        stage2 = policy(obs)
        if use_lbc:
            actions, last_a15, cmd7 = _mix_lbc_action(stage2, env, lbc, last_a15, static_arms=static_arms)
        else:
            actions = stage2
            cmd7 = None
        env.step(actions)
        lat.append((time.perf_counter() - t1) * 1e3)

        robot = cmd.robot_body_pos_w[:use]
        goal = cmd.body_pos_w[:use]
        delta = robot - goal
        et = torch.linalg.norm(delta[:, vis_i[0]], dim=-1)
        el = torch.linalg.norm(delta[:, vis_i[1]], dim=-1)
        er = torch.linalg.norm(delta[:, vis_i[2]], dim=-1)
        ori_t = quat_error_magnitude(cmd.body_quat_w[:use, vis_i[0]], cmd.robot_body_quat_w[:use, vis_i[0]])
        ori_l = quat_error_magnitude(cmd.body_quat_w[:use, vis_i[1]], cmd.robot_body_quat_w[:use, vis_i[1]])
        ori_r = quat_error_magnitude(cmd.body_quat_w[:use, vis_i[2]], cmd.robot_body_quat_w[:use, vis_i[2]])
        offset_z = _ankle_mean_z_offset(cmd)[:use]
        adj_z = (cmd.anchor_pos_w[:use, -1] - cmd.robot_anchor_pos_w[:use, -1]) + offset_z
        fail_z = adj_z.abs() > args_cli.fail_anchor_z
        mot_g = quat_rotate_inverse(cmd.anchor_quat_w, gravity)[:use]
        rob_g = quat_rotate_inverse(cmd.robot_anchor_quat_w, gravity)[:use]
        fail_ori = (mot_g[:, 2] - rob_g[:, 2]).abs() > args_cli.fail_anchor_ori
        ee_err = (cmd.body_pos_w[:use, ee_i, 2] - cmd.robot_body_pos_w[:use, ee_i, 2]).abs()
        fail_ee = ee_err.any(dim=-1) if ee_err.numel() else torch.zeros(use, dtype=torch.bool, device=robot.device)
        if prev_act is None:
            d_act = torch.zeros(use, device=actions.device)
        else:
            d_act = torch.linalg.norm((actions[:use] - prev_act).float(), dim=-1)
        prev_act = actions[:use].clone()

        measured = None
        if cmd7 is not None:
            ti = body_names.index("torso_link")
            measured = extract_realization_command(
                pelvis_quat_w=asset.data.root_quat_w[:use].detach().cpu().numpy(),
                pelvis_lin_vel_w=asset.data.root_lin_vel_w[:use].detach().cpu().numpy(),
                pelvis_ang_vel_w=asset.data.root_ang_vel_w[:use].detach().cpu().numpy(),
                torso_pos_w=cmd.robot_body_pos_w[:use, ti].detach().cpu().numpy(),
                torso_quat_w=cmd.robot_body_quat_w[:use, ti].detach().cpu().numpy(),
                clip=False,
            )

        for e in range(use):
            e_head[e].append(float(et[e]))
            e_lw[e].append(float(el[e]))
            e_rw[e].append(float(er[e]))
            e_head_r[e].append(float(ori_t[e]))
            e_lw_r[e].append(float(ori_l[e]))
            e_rw_r[e].append(float(ori_r[e]))
            act_d[e].append(float(d_act[e]))
            if measured is not None:
                cmd_track[e].append(float(np.linalg.norm(cmd7[e] - measured[e])))
            if fail_step[e] is None:
                if bool(fail_z[e]):
                    fail_step[e], fail_reason[e] = t, "anchor_z"
                elif bool(fail_ori[e]):
                    fail_step[e], fail_reason[e] = t, "anchor_ori"
                elif bool(fail_ee[e]):
                    fail_step[e], fail_reason[e] = t, "ee_z"

    runtime = time.perf_counter() - t0
    rows = []
    for e in range(use):
        vis = np.stack([e_head[e], e_lw[e], e_rw[e]], axis=0)
        vis_mean = vis.mean(axis=0)
        fell = fail_step[e] is not None
        d_intent = float(np.mean(vis_mean) + 0.2 * np.mean([e_head_r[e], e_lw_r[e], e_rw_r[e]]))
        rows.append(
            EpisodeMetrics(
                head_pos_err_m=float(np.mean(e_head[e])),
                head_rot_err_rad=float(np.mean(e_head_r[e])),
                left_hand_pos_err_m=float(np.mean(e_lw[e])),
                left_hand_rot_err_rad=float(np.mean(e_lw_r[e])),
                right_hand_pos_err_m=float(np.mean(e_rw[e])),
                right_hand_rot_err_rad=float(np.mean(e_rw_r[e])),
                success=float(not fell),
                fell=float(fell),
                sr_at_5cm=float(np.mean(vis_mean < 0.05)),
                action_smoothness=float(np.mean(act_d[e])),
                command_track_err=float(np.mean(cmd_track[e])) if cmd_track[e] else float("nan"),
                d_intent=d_intent,
                runtime_s=runtime,
                policy_latency_ms=float(np.mean(lat)),
                baseline=baseline,
                seed=int(args_cli.seed),
                n_steps=steps,
                notes=f"clip={Path(paths[e]).name};fail={fail_reason[e]};lbc={getattr(lbc, 'loaded_path', None)}",
            )
        )
    return rows


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    direct = getattr(agent_cfg, "resume_checkpoint_path", None)
    if direct:
        resume_path = os.path.abspath(str(direct))
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[sirac] baseline={args_cli.baseline} ckpt={resume_path}", flush=True)
    _restore_adapter(agent_cfg, resume_path)
    n_files = len(list(Path(args_cli.motion).rglob("*.npz")))
    n_want = int(args_cli.num_envs) if args_cli.num_envs > 0 else n_files
    env_cfg.scene.num_envs = min(n_want, max(n_files, 1))
    env_cfg.seed = int(args_cli.seed)
    agent_cfg.seed = int(args_cli.seed)
    _apply_eval_motion(env_cfg, args_cli.motion, args_cli.start_frame)
    _apply_terrain(env_cfg, args_cli.terrain)
    _disable_eval_hooks(env_cfg)
    masks = [m.strip() for m in args_cli.mask_modes.split(",") if m.strip()] or ["vr"]
    if not args_cli.keep_headhands_spec:
        from whole_body_tracking.tasks.tracking.config.g1.mask_modes import eval_single_mode_spec

        spec, probs = eval_single_mode_spec(masks[0])
        env_cfg.commands.motion.mask_mode_spec = spec
        env_cfg.commands.motion.mask_mode_probs = probs

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    from whole_body_tracking.tasks.tracking.mdp.curriculums import attach_curriculum_rollout_hints

    attach_curriculum_rollout_hints(env, spe=int(agent_cfg.num_steps_per_env), learning_iteration=int(args_cli.hint_iter))
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False, load_critic=True)
    policy = runner.get_inference_policy(device=env.device)
    policy_mod = runner.alg.policy
    _set_residual_alpha(policy_mod, float(args_cli.residual_alpha))
    injector = CausalFutureInjector(
        mappers={"mapper": args_cli.mapper_path, "mapper_b": args_cli.mapper_b_path or args_cli.mapper_path},
        device=env.device,
    )
    lbc = LowerBodyRealizationController(jit_path=args_cli.jit or None)
    print(f"[sirac] lbc dummy={lbc.is_dummy} path={lbc.loaded_path}", flush=True)

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))
    rows = _rollout(env, policy, injector, lbc, args_cli.baseline, int(args_cli.steps), masks[0])
    # D_intent_excess needs Baseline A as reference; filled by the aggregator later.
    out = Path(args_cli.out) / args_cli.task_name / args_cli.terrain / args_cli.baseline
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", rows)
    write_csv(out / "metrics.csv", rows)
    summary = {
        "baseline": args_cli.baseline,
        "terrain": args_cli.terrain,
        "task": args_cli.task_name,
        "seed": int(args_cli.seed),
        "n_eps": len(rows),
        "success": float(np.nanmean([r.success for r in rows])),
        "fell": float(np.nanmean([r.fell for r in rows])),
        "head_pos_err_m": float(np.nanmean([r.head_pos_err_m for r in rows])),
        "left_hand_pos_err_m": float(np.nanmean([r.left_hand_pos_err_m for r in rows])),
        "right_hand_pos_err_m": float(np.nanmean([r.right_hand_pos_err_m for r in rows])),
        "sr_at_5cm": float(np.nanmean([r.sr_at_5cm for r in rows])),
        "d_intent": float(np.nanmean([r.d_intent for r in rows])),
        "lbc": str(lbc.loaded_path),
        "dummy": bool(lbc.is_dummy),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
