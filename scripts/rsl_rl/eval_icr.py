#!/usr/bin/env python3
"""ICR M0–M3: same reference × unknown terrain. Frozen Stage-2 Parent.

No terrain ID / scan / experts in policy. M0 is Stage-2 only.
M1/M2/M3 load an InteractionResidual checkpoint if --icr_ckpt is set.
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
TERRAINS = ("plane", "slope", "slope_down", "light_rough", "steps", "slip")
TASK_SOURCES = ("loco", "stoop", "reach", "carry")
S_ENABLED_TASKS = ("stoop", "reach", "carry")
METHODS = ("m0", "m1", "m2", "m3")
METHOD_ICR = {
    "m1": {"encoder": "mlp", "history_len": 16, "tangent": False, "beta": 1.0, "r_max": 0.0875},
    "m2": {"encoder": "gru", "history_len": 16, "tangent": False, "beta": 1.0, "r_max": 0.0875},
    "m3": {"encoder": "gru", "history_len": 16, "tangent": True, "beta": 1.0, "r_max": 0.0875},
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
    return obj


from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="ICR: same reference, unknown terrain. Frozen Stage-2.")
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--task", type=str, default="MUSE-Kp-LatentRL-Kp5-HeadHands-Locomani-G1-v0")
parser.add_argument("--motion", type=str, default=f"{CLIP_ROOT}/loco")
parser.add_argument("--mask_modes", type=str, default="torso")
parser.add_argument("--task_source", type=str, default="loco", choices=TASK_SOURCES)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out", type=str, default="results/icr_interaction_recovery")
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
parser.add_argument("--method", type=str, default="m0", choices=METHODS)
parser.add_argument("--icr_ckpt", type=str, default="")
parser.add_argument("--dump_series", action="store_true", default=True)
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
from rsl_rl.modules.intent_recovery import extract_visible_task_error  # noqa: E402
from rsl_rl.modules.interaction_residual import InteractionResidual, pack_token  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import whole_body_tracking.tasks  # noqa: E402, F401
from causal_future import N_BODIES, SLOT_OFFSETS, CausalFutureInjector  # noqa: E402
from whole_body_tracking.tasks.tracking.mdp.rewards import _ankle_mean_z_offset  # noqa: E402


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
    """Terrain exists only in the simulator. Never written into policy obs."""
    mat = getattr(env_cfg.scene.terrain, "physics_material", None)
    if mat is not None:
        mat.static_friction = 1.0
        mat.dynamic_friction = 1.0
    if terrain == "plane":
        env_cfg.scene.terrain.terrain_type = "plane"
        print("[icr] terrain=plane friction=1.0", flush=True)
        return
    if terrain == "slip":
        env_cfg.scene.terrain.terrain_type = "plane"
        if mat is not None:
            mat.static_friction = 0.25
            mat.dynamic_friction = 0.20
        print("[icr] terrain=slip plane friction=0.25/0.20 (not in policy obs)", flush=True)
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
    elif terrain == "slope_down":
        subs = {
            "slope_inv": HfInvertedPyramidSlopedTerrainCfg(
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
    print(f"[icr] terrain={terrain} 1x1 generator (not in policy obs)", flush=True)


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
                print(f"[icr] pin mask {mask_name!r} idx={names.index(mask_name)}", flush=True)
        cmd._resample_command(ids)
        if hasattr(cmd, "resample_all_mask_modes"):
            cmd.resample_all_mask_modes()
        try:
            env.unwrapped.sim.forward()
        except Exception as exc:
            print(f"[icr] sim.forward skipped: {exc}", flush=True)
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


def _load_icr(path: str, device) -> InteractionResidual | None:
    if not path:
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(blob.get("cfg") or {})
    method_defaults = METHOD_ICR.get(str(args_cli.method), {})
    for k, v in method_defaults.items():
        cfg.setdefault(k, v)
    sd = blob.get("model")
    if sd is None:
        raw = blob.get("model_state_dict") or blob.get("state_dict") or blob
        if not isinstance(raw, dict):
            raise ValueError(f"cannot parse ICR ckpt {path}")
        prefix = "interaction_net."
        sd = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
        if not sd:
            raise ValueError(f"no interaction_net.* in {path}")
        if "encoder" not in cfg:
            cfg["encoder"] = "gru" if any(k.startswith("body.gru.") for k in sd) else "mlp"
    m = InteractionResidual(
        encoder=str(cfg.get("encoder", "mlp")),
        history_len=int(cfg.get("history_len", 16)),
        tangent=bool(cfg.get("tangent", True)),
        beta=float(cfg.get("beta", 1.0)),
        r_max=float(cfg.get("r_max", 0.0875)),
    )
    m.load_state_dict(sd, strict=False)
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    print(f"[icr] loaded adapter {path} {m.extra_repr()}", flush=True)
    return m


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


def _series_stats(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": float(a.mean()),
        "p50": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        "n": int(a.size),
    }


@torch.inference_mode()
def _rollout(env, policy, runner, injector, steps, seed, terrain, vis_i, vis_flags, mask_name, adapter):
    n_envs = int(env.unwrapped.num_envs)
    use, paths = _pin_clips(env, mask_name=mask_name)
    cmd = env.unwrapped.command_manager.get_term("motion")
    device = env.unwrapped.device
    asset = env.unwrapped.scene["robot"]
    am = env.unwrapped.action_manager
    gravity = asset.data.GRAVITY_VEC_W
    if gravity.ndim == 1:
        gravity = gravity.unsqueeze(0).expand(cmd.num_envs, -1)
    names = list(asset.data.body_names)
    pelvis_i = names.index("pelvis") if "pelvis" in names else None
    ankle_i = [names.index(n) for n in ANKLE_BODIES if n in names]
    try:
        cf = env.unwrapped.scene["contact_forces"]
    except Exception:
        cf = None
    if adapter is not None:
        adapter.reset()
    failed = torch.zeros(n_envs, dtype=torch.bool, device=device)
    fail_step = [-1] * use
    fail_reason = [""] * use
    e_prev = None
    e_vis_prev = None
    slot0 = int(SLOT_OFFSETS.index(0))
    series = {
        i: {
            "e": [], "de": [], "e_torso": [], "e_lw": [], "e_rw": [],
            "dz": [], "roll": [], "pitch": [], "contact": [], "z_nom": [],
        }
        for i in range(use)
    }
    for t in range(steps):
        obs, _ = env.get_observations()
        obs = injector.patch_policy_obs(obs, env, "mapper")
        z_nom, proprio, enc = _z_and_proprio(policy, runner, obs)
        bsz = int(z_nom.shape[0])
        z_exec = z_nom
        dz_n = z_nom.new_zeros(bsz)
        dz_bar_n = z_nom.new_zeros(bsz)
        e9 = z_nom.new_zeros(bsz, 9)
        de9 = z_nom.new_zeros(bsz, 9)
        if adapter is not None and str(args_cli.method) != "m0":
            nrm = getattr(runner, "obs_normalizer", None)
            enc_m = nrm.inverse(enc) if nrm is not None and hasattr(nrm, "inverse") else enc
            kp, kp_mask, _p = policy.muse.transformer_encoder.split_obs(enc_m)
            L = int(policy.muse.transformer_encoder.kp_lookahead_steps)
            n_b = int(policy.muse.transformer_encoder.kp_n_bodies)
            kp_lhn = kp.reshape(bsz, n_b, L, 3).transpose(1, 2)
            e9, _vis, _e_rms = extract_visible_task_error(kp_lhn, kp_mask, slot0)
            if e_prev is None:
                de9 = torch.zeros_like(e9)
            else:
                de9 = (e9 - e_prev) / DT
            e_prev = e9
            tok = pack_token(z_nom, e9, de9, proprio)
            out = adapter.apply(z_nom, tok)
            z_exec = out["z_exec"]
            dz_n = out["dz_norm"]
            dz_bar_n = out["dz_bar_norm"]
        joints = _decode(policy, z_exec, proprio)
        env.step(joints)

        e_torso, e_lw, e_rw = _body_e(cmd, vis_i)
        e_vis = _visible_e(cmd, vis_i, vis_flags)
        if e_vis_prev is None:
            e_vis_prev = e_vis.clone()
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
        roll, pitch, _yaw = euler_xyz_from_quat(asset.data.root_quat_w[:use])
        contact = torch.zeros(use, device=device)
        if cf is not None and ankle_i and hasattr(cf.data, "net_forces_w"):
            nf = cf.data.net_forces_w[:use]
            contact = (nf[:, ankle_i, :].norm(dim=-1) > CONTACT_N).float().sum(dim=-1)
        for i in range(use):
            if t >= WARMUP:
                series[i]["e"].append(float(e_vis[i].item()))
                series[i]["de"].append(float((e_vis[i] - (e_vis_prev[i] if e_vis_prev is not None else e_vis[i])).item()))
                series[i]["e_torso"].append(float(e_torso[i].item()))
                series[i]["e_lw"].append(float(e_lw[i].item()))
                series[i]["e_rw"].append(float(e_rw[i].item()))
                series[i]["dz"].append(float(dz_n[i].item()))
                series[i]["roll"].append(float(roll[i].item()))
                series[i]["pitch"].append(float(pitch[i].item()))
                series[i]["contact"].append(float(contact[i].item()))
                if t % 5 == 0:
                    series[i]["z_nom"].append(z_nom[i].detach().cpu().numpy().astype(np.float32))
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
        e_vis_prev = e_vis.clone()
        if (t + 1) % 50 == 0:
            print(
                f"[icr] {args_cli.method} {args_cli.task_source} {terrain} s{seed} {t+1}/{steps} "
                f"e={float(e_vis[:use].mean()):.3f} fail={int(failed[:use].sum())}",
                flush=True,
            )
    rows = []
    for i in range(use):
        fail = fail_step[i] >= 0
        e_a = np.asarray(series[i]["e"], dtype=np.float64)
        rec_t = None
        if fail and e_a.size:
            band = 0.13
            post = e_a  # already post-warmup
            below = np.where(post < band)[0]
            rec_t = int(below[0] * 20) if below.size else None
        rows.append(
            {
                "clip": Path(paths[i]).name if i < len(paths) else f"env{i}",
                "seed": int(seed),
                "terrain": terrain,
                "method": str(args_cli.method),
                "sr_task": int(not fail),
                "fail_reason": fail_reason[i] or "none",
                "fail_step": int(fail_step[i]),
                "e": _series_stats(series[i]["e"]),
                "e_torso": _series_stats(series[i]["e_torso"]),
                "e_lw": _series_stats(series[i]["e_lw"]),
                "e_rw": _series_stats(series[i]["e_rw"]),
                "dz": _series_stats(series[i]["dz"]),
                "recovery_ms_to_13cm": rec_t,
                "series": series[i] if bool(args_cli.dump_series) else None,
            }
        )
    return rows


def _run_terrain(env_cfg, agent_cfg, resume_path, motion_dir, terrain, out_root, adapter):
    cell = out_root / str(args_cli.method) / str(args_cli.task_source) / terrain
    cell.mkdir(parents=True, exist_ok=True)
    done_p = cell / "summary.json"
    if done_p.exists():
        print(f"[icr] skip {done_p}", flush=True)
        return json.loads(done_p.read_text())
    env, runner, policy, injector = _make_env(env_cfg, agent_cfg, resume_path, motion_dir, terrain)
    cmd = env.unwrapped.command_manager.get_term("motion")
    body_names = list(cmd.cfg.body_names)
    vis_i = [body_names.index(n) for n in KP_VIS]
    mask_name = str(args_cli.mask_modes).split(",")[0].strip() or "torso"
    vis_flags = MASK_VIS.get(mask_name, (True, True, True))
    seeds = [int(s) for s in args_cli.seeds.split(",") if s.strip()] or [int(args_cli.seed)]
    all_eps = []
    for seed in seeds:
        print(f"[icr] ===== {args_cli.method} {args_cli.task_source} {terrain} seed={seed} =====", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        rows = _rollout(
            env, policy, runner, injector, int(args_cli.steps), seed, terrain,
            vis_i, vis_flags, mask_name, adapter,
        )
        all_eps.extend(rows)
        print(f"[icr] seed={seed} SR={float(np.mean([r['sr_task'] for r in rows])):.3f}", flush=True)
    n = len(all_eps)
    reasons = {}
    for r in all_eps:
        k = r.get("fail_reason") or "none"
        reasons[k] = reasons.get(k, 0) + 1
    e_all = np.concatenate([np.asarray(r["series"]["e"], dtype=np.float64) for r in all_eps if r.get("series")]) if all_eps else np.zeros(0)
    payload = {
        "step": "ICR",
        "method": str(args_cli.method),
        "no_terrain_in_policy": True,
        "same_reference": True,
        "p1_task": str(args_cli.task_source),
        "terrain": terrain,
        "ckpt": resume_path,
        "n_episodes": n,
        "sr_task": float(np.mean([r["sr_task"] for r in all_eps])) if n else None,
        "fall_frac": float(reasons.get("fall", 0) / max(n, 1)),
        "anchor_z_frac": float(reasons.get("anchor_z", 0) / max(n, 1)),
        "fail_reasons": reasons,
        "e_mean": float(e_all.mean()) if e_all.size else None,
        "e_p95": float(np.percentile(e_all, 95)) if e_all.size else None,
        "e_max": float(e_all.max()) if e_all.size else None,
        "mean_dz": float(np.nanmean([r["dz"]["mean"] for r in all_eps])),
        "episodes": [{k: v for k, v in r.items() if k != "series"} for r in all_eps],
    }
    done_p.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    if bool(args_cli.dump_series):
        np.savez_compressed(
            cell / "series.npz",
            clip=np.asarray([r["clip"] for r in all_eps]),
            seed=np.asarray([r["seed"] for r in all_eps], dtype=np.int32),
            e=np.stack([np.asarray(r["series"]["e"], dtype=np.float32) for r in all_eps]),
            de=np.stack([np.asarray(r["series"]["de"], dtype=np.float32) for r in all_eps]),
            dz=np.stack([np.asarray(r["series"]["dz"], dtype=np.float32) for r in all_eps]),
            roll=np.stack([np.asarray(r["series"]["roll"], dtype=np.float32) for r in all_eps]),
            pitch=np.stack([np.asarray(r["series"]["pitch"], dtype=np.float32) for r in all_eps]),
            contact=np.stack([np.asarray(r["series"]["contact"], dtype=np.float32) for r in all_eps]),
        )
    print(f"[icr] wrote {done_p} SR={payload['sr_task']}", flush=True)
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
    print(f"[icr] ckpt {resume_path} method={args_cli.method}", flush=True)
    _restore_adapter(agent_cfg, resume_path)
    if hasattr(agent_cfg.policy, "intent_recovery"):
        agent_cfg.policy.intent_recovery = False
    if hasattr(agent_cfg.policy, "interaction_recovery"):
        agent_cfg.policy.interaction_recovery = False
    if hasattr(agent_cfg.policy, "terrain_scan_dim"):
        agent_cfg.policy.terrain_scan_dim = 0
    if hasattr(agent_cfg.policy, "adapter"):
        agent_cfg.policy.adapter = "residual"
    adapter = _load_icr(str(args_cli.icr_ckpt or ""), "cpu")
    terrains = [t.strip() for t in args_cli.terrains.split(",") if t.strip()] or [args_cli.terrain]
    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for terrain in terrains:
        if adapter is not None:
            adapter.reset()
            adapter.to("cuda:0")
        _run_terrain(env_cfg, agent_cfg, resume_path, args_cli.motion, terrain, out_dir, adapter)
    print("[icr] done", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
