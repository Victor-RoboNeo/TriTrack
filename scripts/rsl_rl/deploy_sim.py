"""Deployment-mode Isaac harness for the real-robot pipeline.

  Streaming 3-point (K_<=t only) → Mapper-B → frozen Stage-2 encoder
  → g_φ(model_50000) → frozen Stage-2 decoder → RobotBackend (Isaac).

Future slots never come from clip GT. `--source openxr` streams the same
JSON v1 UDP packets as Quest / mock_vr_sender (loopback), then Mapper-B.

Example:
  python scripts/rsl_rl/deploy_sim.py \\
      --policy /path/model_50000.pt \\
      --mapper /path/mapper_best.pt \\
      --source replay --terrain plane --suite test1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Causal deploy-sim: Mapper-B + model_50000 in Isaac.")
parser.add_argument("--task", type=str, default="MUSE-Kp-LatentRL-Kp5-HeadHands-Locomani-G1-v0")
parser.add_argument("--num_envs", type=int, default=0, help="0 = one env per selected clip")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--hint_iter", type=int, default=50000)
parser.add_argument("--start_frame", type=int, default=10)
parser.add_argument("--terrain", type=str, default="plane", choices=("plane", "light_rough", "slope", "steps"))
parser.add_argument("--source", type=str, default="replay", choices=("replay", "openxr"))
parser.add_argument("--suite", type=str, default="test1", help="test1 = 16 curated clips")
parser.add_argument("--clip_root", type=str, default="/data/home/chenxiangyu/robotics/Anybody/logs/tritrack/infer_clips/p1")
parser.add_argument(
    "--policy",
    type=str,
    default="/data/home/chenxiangyu/robotics/Anybody/logs/rsl_rl/g1_flat_muse_kp_latent_rl/2026-08-26_00-38-10_tritrack_headhands_locomani_from35000/model_50000.pt",
)
parser.add_argument("--stage2", type=str, default="", help="Unused: Stage-2 lives inside --policy.")
parser.add_argument(
    "--mapper",
    type=str,
    default="/data/home/chenxiangyu/victor/TriTrack/runs/mapper_b_intent72/mapper_best.pt",
)
parser.add_argument("--out", type=str, default="results/deploy_sim")
parser.add_argument("--latency_ms", type=float, default=0.0)
parser.add_argument("--hand_noise_cm", type=float, default=0.0)
parser.add_argument("--drop_prob", type=float, default=0.0)
parser.add_argument("--markers", action="store_true", help="Isaac VisualizationMarkers (env 0)")
parser.add_argument("--openxr_port", type=int, default=15151)
parser.add_argument("--openxr_canon", type=str, default="world", choices=("world", "headset"))
parser.add_argument(
    "--openxr_live_steps",
    type=int,
    default=400,
    help="After 16 npz-UDP clips, run in-process mock VR this many steps (0=skip).",
)
parser.add_argument(
    "--latency_shift",
    action="store_true",
    help="D2: shift Mapper-B futures by latency_steps so slots are robot-now relative.",
)
parser.add_argument("--z_min", type=float, default=0.40, help="Robot-centric fall: torso z below this (m).")
parser.add_argument("--ori_rad", type=float, default=1.2, help="Robot-centric fall: |roll| or |pitch| (rad).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

import whole_body_tracking.tasks  # noqa: F401
from causal_future import CausalFutureInjector, current_packet_world
from deploy_runtime import (
    ControllerPipeline,
    DeadManSwitch,
    DeployMarkers,
    IsaacBackend,
    IntentFrontend,
    MockVrThread,
    OpenXRIntentDriver,
    OpenXRLoopback,
    physical_fall,
    rel_rpy,
    vis_from_mask,
    write_deploy_html,
)
from whole_body_tracking.tasks.tracking.mdp.curriculums import attach_curriculum_rollout_hints
from whole_body_tracking.tasks.tracking.mdp.rewards import _ankle_mean_z_offset

CLIP_ROOT = Path("/data/home/chenxiangyu/robotics/Anybody/logs/tritrack/infer_clips/p1")

# Test 1: 16 clips the human will actually look at. Canonical HeadHands masks.
TEST1 = [
    ("loco", "torso", "00_bones_seed_g1_Neutral_walk_forward_002.npz"),
    ("loco", "torso", "01_bones_seed_g1_Loop_Forward_Walk_001.npz"),
    ("loco", "torso", "03_bones_seed_g1_Turn_Start_Jog_0000_001.npz"),
    ("loco", "torso", "06_bones_seed_g1_Sideway_Walk_Left_001.npz"),
    ("reach", "head_left", "00_bones_seed_g1_reaching_down_001.npz"),
    ("reach", "head_left", "06_bones_seed_g1_reaching_far_001.npz"),
    ("reach", "head_right", "03_bones_seed_g1_reaching_down_R_001.npz"),
    ("reach", "head_right", "08_bones_seed_g1_reaching_far_R_001.npz"),
    ("stoop", "vr", "00_bones_seed_g1_Neutral_stoop_down_001.npz"),
    ("stoop", "vr", "07_bones_seed_g1_squat_001.npz"),
    ("stoop", "vr", "04_bones_seed_g1_baby_full_diaper_pick_up_R_001.npz"),
    ("stoop", "vr", "09_bones_seed_g1_item_pick_up_crouch_walk_R_001.npz"),
    ("carry", "vr", "00_bones_seed_g1_lift_crate_start_001.npz"),
    ("carry", "vr", "01_bones_seed_g1_lift_crate_loop_001.npz"),
    ("carry", "vr", "04_bones_seed_g1_lift_crate_walk_ff_loop_180_R_001.npz"),
    ("carry", "vr", "05_bones_seed_g1_side_lift_360_R_001.npz"),
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
        print("[deploy] terrain=plane", flush=True)
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
    print(f"[deploy] terrain={terrain} 1x1 generator", flush=True)


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


def _stage_clips(suite: str, clip_root: Path, dest: Path) -> list[dict]:
    rows = TEST1 if suite == "test1" else TEST1
    dest.mkdir(parents=True, exist_ok=True)
    catalog = []
    for i, (task, mask, fname) in enumerate(rows):
        src = clip_root / task / fname
        if not src.is_file():
            raise SystemExit(f"missing clip {src}")
        dst = dest / f"{i:02d}_{task}_{mask}_{fname}"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        catalog.append({"task": task, "mask": mask, "name": fname, "path": str(src)})
    return catalog


def _pin_per_env_masks(env, masks: list[str]) -> None:
    cmd = env.unwrapped.command_manager.get_term("motion")
    names = [str(x) for x in getattr(cmd, "_mode_names", ())]
    idxs = []
    for m in masks:
        if m not in names:
            raise SystemExit(f"mask {m!r} not in HeadHands modes {names}")
        idxs.append(names.index(m))
    t = torch.tensor(idxs, device=cmd.device, dtype=torch.long)
    cmd._eval_fixed_mode_idx = None
    cmd._env_mode_idx[: len(t)] = t
    cmd._env_body_mask[: len(t)] = cmd._mode_body_visibility[t]
    print(f"[deploy] per-env masks {list(zip(range(len(masks)), masks, idxs))}", flush=True)


def _pin_clips(env, catalog: list[dict]) -> None:
    cmd = env.unwrapped.command_manager.get_term("motion")
    if hasattr(cmd, "p_mask"):
        cmd.p_mask = 0.0
    n = min(int(env.unwrapped.num_envs), len(catalog))
    with torch.inference_mode():
        env.reset()
        ids = torch.arange(n, device=cmd.device, dtype=torch.long)
        cmd.env_motion_indices[:n] = ids
        if hasattr(cmd, "env_motion_groups"):
            cmd.env_motion_groups[:n] = 0
        if hasattr(cmd, "_env_remap_version") and hasattr(cmd, "_remap_version"):
            cmd._env_remap_version[:n] = cmd._remap_version
        cmd._resample_command(ids)
        _pin_per_env_masks(env, [c["mask"] for c in catalog[:n]])
        try:
            env.unwrapped.sim.forward()
        except Exception as exc:
            print(f"[deploy] sim.forward skipped: {exc}", flush=True)
    for i in range(n):
        print(f"  env{i} {catalog[i]['task']}/{catalog[i]['mask']} {catalog[i]['name']}", flush=True)


def _pin_one(env, catalog: list[dict], idx: int) -> None:
    cmd = env.unwrapped.command_manager.get_term("motion")
    if hasattr(cmd, "p_mask"):
        cmd.p_mask = 0.0
    with torch.inference_mode():
        env.reset()
        ids = torch.zeros(1, device=cmd.device, dtype=torch.long)
        cmd.env_motion_indices[0] = int(idx)
        if hasattr(cmd, "env_motion_groups"):
            cmd.env_motion_groups[0] = 0
        if hasattr(cmd, "_env_remap_version") and hasattr(cmd, "_remap_version"):
            cmd._env_remap_version[0] = cmd._remap_version
        cmd._resample_command(ids)
        _pin_per_env_masks(env, [catalog[idx]["mask"]])
        try:
            env.unwrapped.sim.forward()
        except Exception as exc:
            print(f"[deploy] sim.forward skipped: {exc}", flush=True)
    print(
        f"[deploy] pin clip{idx} {catalog[idx]['task']}/{catalog[idx]['mask']} {catalog[idx]['name']}",
        flush=True,
    )


def _empty_rec() -> dict:
    return {
        "robot": [[[] for _ in range(3)]],
        "intent": [[[] for _ in range(3)]],
        "e_torso": [[]],
        "e_lw": [[]],
        "e_rw": [[]],
        "roll": [[]],
        "pitch": [[]],
        "dz": [[]],
        "hold": [[]],
        "h05": [[]],
        "fail_step": [None],
        "fail_reason": [""],
        "future": [None],
    }


def _append_step(rec, t, steps, vis_i, ankle_i, robot, goal, e_t, e_l, e_r, roll, pitch, dz, hold, h05, fail_z, fail_ori, future_w, markers):
    if ankle_i:
        feet = robot[:, ankle_i, :]
    else:
        feet = None
    markers.update(intent_w=goal[:, vis_i, :], robot_w=robot, future_w=future_w, foot_w=feet)
    e = 0
    for k, bi in enumerate(vis_i):
        rec["robot"][e][k].append([float(robot[e, bi, d].cpu()) for d in range(3)])
        rec["intent"][e][k].append([float(goal[e, bi, d].cpu()) for d in range(3)])
    rec["e_torso"][e].append(float(e_t[e].cpu()))
    rec["e_lw"][e].append(float(e_l[e].cpu()))
    rec["e_rw"][e].append(float(e_r[e].cpu()))
    rec["roll"][e].append(float(roll[e].abs().cpu()))
    rec["pitch"][e].append(float(pitch[e].abs().cpu()))
    rec["dz"][e].append(float(dz[e].cpu()))
    rec["hold"][e].append(1.0 if bool(hold[e]) else 0.0)
    rec["h05"][e].append(float(h05[e].cpu()))
    if rec["fail_step"][e] is None and t >= 10:
        if bool(fail_z[e]):
            rec["fail_step"][e] = t
            rec["fail_reason"][e] = "pelvis_z"
        elif bool(fail_ori[e]):
            rec["fail_step"][e] = t
            rec["fail_reason"][e] = "robot_ori"
    if t == steps - 1 and future_w is not None:
        rec["future"][e] = future_w[e].reshape(-1, 3).detach().cpu().tolist()


def _clip_row(rec, catalog_row: dict) -> dict:
    et = np.asarray(rec["e_torso"][0])
    el = np.asarray(rec["e_lw"][0])
    er = np.asarray(rec["e_rw"][0])
    mask = catalog_row["mask"]
    parts = [et]
    if mask not in ("torso",):
        if mask in ("vr", "head_left"):
            parts.append(el)
        if mask in ("vr", "head_right"):
            parts.append(er)
    e_vis = np.mean(np.stack(parts, 0), 0)
    fail = rec["fail_step"][0] is not None
    wrist = np.nanmean([el, er]) if mask != "torso" else float("nan")
    row = {
        "task": catalog_row["task"],
        "mask": mask,
        "name": catalog_row["name"],
        "sr5": float((e_vis < 0.05).mean()),
        "e_torso": float(et.mean()),
        "e_wrist": float(wrist) if np.isfinite(wrist) else None,
        "roll": float(np.mean(rec["roll"][0])),
        "pitch": float(np.mean(rec["pitch"][0])),
        "dz_mean": float(np.mean(rec["dz"][0])),
        "mapper_h05": float(np.mean(rec["h05"][0])),
        "fail": fail,
        "fail_reason": rec["fail_reason"][0],
        "hold_n": int(np.sum(rec["hold"][0])),
        "robot": rec["robot"][0],
        "intent": rec["intent"][0],
        "future": rec["future"][0],
        "e_torso_t": rec["e_torso"][0],
        "e_lw_t": rec["e_lw"][0],
        "e_rw_t": rec["e_rw"][0],
        "pitch_t": rec["pitch"][0],
        "dz_t": rec["dz"][0],
        "hold_t": rec["hold"][0],
    }
    print(
        f"[deploy] {catalog_row['task']:5s} {mask:10s} SR5={100*(e_vis<0.05).mean():5.1f}% "
        f"torso={et.mean():.3f} fail={rec['fail_reason'][0] or 'ok'}  {catalog_row['name']}",
        flush=True,
    )
    return row


def _run_openxr(
    args_cli,
    env,
    backend,
    pipe,
    injector,
    watchdog,
    markers,
    catalog,
    vis_i,
    ankle_i,
    policy_path,
    mapper_path,
    out,
):
    """16 clips as OpenXR JSON v1 UDP, then optional mock headset. One Kit, env.reset() between clips."""
    from tritrack.sources.replay_npz import NpzReplaySource

    cmd = env.unwrapped.command_manager.get_term("motion")
    steps = int(args_cli.steps)
    loopback = OpenXRLoopback(port=int(args_cli.openxr_port))
    driver = OpenXRIntentDriver(injector, loopback, canon=str(args_cli.openxr_canon))
    print(
        f"[deploy] OpenXR UDP 127.0.0.1:{loopback.port}  canon={driver.canon_mode}  "
        "obs[:300] from packets+Mapper-B; obs[300:750] Isaac proprio",
        flush=True,
    )
    clips_out = []
    try:
        for ci, row in enumerate(catalog):
            src = NpzReplaySource(row["path"])
            _pin_one(env, catalog, ci)
            watchdog.prev_intent = None
            watchdog.last_q = None
            vis = vis_from_mask(row["mask"], env.device, n_env=1)
            first = src.state_at(int(args_cli.start_frame))
            loopback.send(first)
            got = loopback.poll_after_send(timeout_s=1.0)
            if got is None:
                raise SystemExit(f"OpenXR UDP loopback silent on port {loopback.port}")
            pts0 = np.stack([p.pos for p in got.poses])
            driver.reset(pts0)
            pkt0 = torch.as_tensor(pts0, device=env.device, dtype=torch.float32).view(1, 3, 3)
            pipe.frontend.reset(pkt0, delay_steps=injector.latency_steps)
            rec = _empty_rec()
            misses = 0
            for t in range(steps):
                frame = int(cmd.time_steps[0].item())
                loopback.send(src.state_at(frame))
                intent = loopback.poll_after_send(timeout_s=0.05)
                if intent is None:
                    misses += 1
                    pts_w = driver.last_pts_w
                else:
                    pts_w = driver.ingest_state(intent)
                pkt = torch.as_tensor(pts_w, device=env.device, dtype=torch.float32).view(1, 3, 3)
                with torch.inference_mode():
                    obs, _ = backend.get_observations()
                    if driver.canon_mode == "headset":
                        patched = driver.overlay_headset(obs, env, vis, intent if intent is not None else driver.last_intent)
                        drop = torch.tensor([misses if intent is None else 0], device=obs.device, dtype=torch.long)
                        q, info = pipe.act_prepatched(
                            patched, env, drop_streak=drop, intent_cur_w=injector.last_cur_w
                        )
                    else:
                        q, info = pipe.act(obs, env, vis=vis, packet_w=pkt)
                    backend.send_joint_target(q)
                st = backend.get_base_state()
                robot = st["robot_body_pos_w"][:1]
                goal = st["intent_body_pos_w"][:1].clone()
                if injector.last_cur_w is not None:
                    cur = injector.last_cur_w
                    if cur.ndim == 3:
                        goal[:, vis_i] = cur[0, :3]
                delta = robot - goal
                e_t = torch.linalg.norm(delta[:, vis_i[0]], dim=-1)
                e_l = torch.linalg.norm(delta[:, vis_i[1]], dim=-1)
                e_r = torch.linalg.norm(delta[:, vis_i[2]], dim=-1)
                fail_z, fail_ori, _z, roll, pitch = physical_fall(
                    env, 1, z_min=float(args_cli.z_min), ori_rad=float(args_cli.ori_rad)
                )
                dz = info["delta_z"][:1].norm(dim=-1)
                hold = info["hold"][:1]
                try:
                    ferr = injector.future_errors(env, injector.last_pred_abs)
                    h05 = ferr["h_0.5"][:1]
                except Exception:
                    h05 = torch.zeros(1, device=env.device)
                _append_step(
                    rec, t, steps, vis_i, ankle_i, robot, goal, e_t, e_l, e_r,
                    roll, pitch, dz, hold, h05, fail_z, fail_ori, injector.last_pred_w, markers,
                )
                if (t + 1) % 50 == 0:
                    print(
                        f"[deploy] openxr clip{ci} step {t+1}/{steps} e_torso={float(e_t.mean()):.3f} "
                        f"||dZ||={float(dz.mean()):.3f} hold={int(hold.sum())} miss={misses}",
                        flush=True,
                    )
            row_out = _clip_row(rec, row)
            row_out["udp_packets"] = driver.packets
            row_out["udp_miss"] = misses
            clips_out.append(row_out)

        live_steps = int(args_cli.openxr_live_steps)
        if live_steps > 0:
            print(f"[deploy] OpenXR live mock VR  steps={live_steps} (no Quest on this box)", flush=True)
            _pin_one(env, catalog, 0)
            watchdog.prev_intent = None
            watchdog.last_q = None
            vis = vis_from_mask("vr", env.device, n_env=1)
            driver.canon_mode = "headset"
            driver.reset()
            mock = MockVrThread(loopback, rate_hz=50.0)
            mock.start()
            rec = _empty_rec()
            live_row = {
                "task": "live",
                "mask": "vr",
                "name": "mock_vr_sender",
            }
            import time as _time

            t_wait = _time.monotonic()
            first_intent = None
            while _time.monotonic() - t_wait < 2.0:
                first_intent = loopback.source.poll()
                if first_intent is not None:
                    break
                _time.sleep(0.01)
            if first_intent is None:
                print("[deploy] mock VR produced no packets; skip live", flush=True)
                mock.stop()
            else:
                driver.ingest_state(first_intent)
                misses = 0
                streak = 0
                for t in range(live_steps):
                    intent = loopback.source.poll()
                    if intent is None:
                        misses += 1
                        streak += 1
                    else:
                        streak = 0
                        driver.ingest_state(intent)
                    with torch.inference_mode():
                        obs, _ = backend.get_observations()
                        patched = driver.overlay_headset(obs, env, vis, intent if intent is not None else driver.last_intent)
                        drop = torch.tensor([streak], device=obs.device, dtype=torch.long)
                        q, info = pipe.act_prepatched(
                            patched, env, drop_streak=drop, intent_cur_w=injector.last_cur_w
                        )
                        backend.send_joint_target(q)
                    st = backend.get_base_state()
                    robot = st["robot_body_pos_w"][:1]
                    goal = st["intent_body_pos_w"][:1].clone()
                    if injector.last_cur_w is not None:
                        goal[:, vis_i] = injector.last_cur_w[0, :3]
                    delta = robot - goal
                    e_t = torch.linalg.norm(delta[:, vis_i[0]], dim=-1)
                    e_l = torch.linalg.norm(delta[:, vis_i[1]], dim=-1)
                    e_r = torch.linalg.norm(delta[:, vis_i[2]], dim=-1)
                    roll, pitch, _yaw = rel_rpy(st["anchor_quat_w"][:1], st["robot_anchor_quat_w"][:1])
                    offset = _ankle_mean_z_offset(cmd)[:1]
                    adj_z = (st["anchor_pos_w"][:1, -1] - st["robot_anchor_pos_w"][:1, -1]) + offset
                    fail_z = adj_z.abs() > 0.25
                    fail_ori = info["tilt"][:1].abs() > 1.2
                    dz = info["delta_z"][:1].norm(dim=-1)
                    hold = info["hold"][:1]
                    h05 = torch.zeros(1, device=env.device)
                    _append_step(
                        rec, t, live_steps, vis_i, ankle_i, robot, goal, e_t, e_l, e_r,
                        roll, pitch, dz, hold, h05, fail_z, fail_ori, injector.last_pred_w, markers,
                    )
                    if (t + 1) % 50 == 0:
                        print(
                            f"[deploy] openxr live step {t+1}/{live_steps} e_torso={float(e_t.mean()):.3f} "
                            f"hold={int(hold.sum())} miss={misses} sent={mock.n_sent}",
                            flush=True,
                        )
                mock.stop()
                row_out = _clip_row(rec, live_row)
                row_out["udp_packets"] = driver.packets
                row_out["udp_miss"] = misses
                clips_out.append(row_out)
    finally:
        loopback.close()
    return clips_out


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    policy_path = os.path.abspath(args_cli.policy)
    if getattr(agent_cfg, "resume_checkpoint_path", None):
        policy_path = os.path.abspath(str(agent_cfg.resume_checkpoint_path))
    mapper_path = os.path.abspath(args_cli.mapper)
    if not os.path.isfile(policy_path):
        raise SystemExit(f"policy not found: {policy_path}")
    if not os.path.isfile(mapper_path):
        raise SystemExit(f"mapper not found: {mapper_path}")
    if args_cli.stage2:
        print(
            f"[deploy] --stage2={args_cli.stage2} ignored: Stage-2 encoder/decoder "
            "are the frozen MUSE-Kp weights inside --policy (model_50000).",
            flush=True,
        )

    out = Path(args_cli.out) / args_cli.suite / args_cli.terrain
    out.mkdir(parents=True, exist_ok=True)
    clip_root = Path(args_cli.clip_root)
    catalog = _stage_clips(args_cli.suite, clip_root, out / "_clips")
    n = len(catalog)
    if args_cli.source == "openxr":
        n_envs = 1
        print("[deploy] source=openxr  num_envs=1 (sequential clips, UDP JSON v1)", flush=True)
    else:
        n_envs = int(args_cli.num_envs) if args_cli.num_envs and int(args_cli.num_envs) > 0 else n
    env_cfg.scene.num_envs = min(n_envs, n)
    env_cfg.seed = int(args_cli.seed)
    agent_cfg.seed = int(args_cli.seed)

    setattr(agent_cfg, "resume", True)
    setattr(agent_cfg, "resume_checkpoint_path", policy_path)
    _restore_adapter(agent_cfg, policy_path)
    if hasattr(agent_cfg.policy, "adapter"):
        agent_cfg.policy.adapter = "residual"
    _apply_eval_motion(env_cfg, str(out / "_clips"), args_cli.start_frame)
    _apply_terrain(env_cfg, args_cli.terrain)
    _disable_eval_hooks(env_cfg)
    print("[deploy] keep HeadHands 4-mode spec; per-env canonical masks", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    attach_curriculum_rollout_hints(env, spe=int(agent_cfg.num_steps_per_env), learning_iteration=int(args_cli.hint_iter))
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(policy_path, load_optimizer=False, load_critic=True)
    policy_mod = runner.alg.policy
    policy_mod.eval()

    injector = CausalFutureInjector(mappers={"mapper": mapper_path, "mapper_b": mapper_path}, device=env.device)
    injector.latency_steps = int(round(float(args_cli.latency_ms) / 20.0))
    injector.hand_noise_m = float(args_cli.hand_noise_cm) / 100.0
    injector.drop_prob = float(args_cli.drop_prob)
    injector.latency_shift_steps = injector.latency_steps if bool(args_cli.latency_shift) else 0
    print(
        f"[deploy] unified IntentPacket→obs[:300]  latency_steps={injector.latency_steps} "
        f"shift={injector.latency_shift_steps} hand_noise={injector.hand_noise_m:.3f}m "
        f"drop={injector.drop_prob:.2f}  (past+current+future from runtime buffer, no GT)",
        flush=True,
    )
    if injector.mapper is not None:
        print(f"[deploy] mapper in_dim={injector.mapper.in_dim} (72=intent-only Mapper-B)", flush=True)
    alpha = float(getattr(getattr(policy_mod, "residual_corrector", None), "alpha", float("nan")))
    print(f"[deploy] residual_alpha={alpha}  (Δz_eff = alpha * Δz_raw)", flush=True)

    frontend = IntentFrontend(injector, delay_steps=injector.latency_steps)
    backend = IsaacBackend(env)
    watchdog = DeadManSwitch()
    pipe = ControllerPipeline(
        policy_mod, injector, watchdog, obs_normalizer=runner.obs_normalizer, frontend=frontend
    )
    markers = DeployMarkers(env, enabled=bool(args_cli.markers))

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))
    cmd = env.unwrapped.command_manager.get_term("motion")
    body_names = list(cmd.cfg.body_names)
    vis_i = [body_names.index(n) for n in ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")]
    ankle_i = [body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link") if n in body_names]

    if args_cli.source == "openxr":
        clips_out = _run_openxr(
            args_cli=args_cli,
            env=env,
            backend=backend,
            pipe=pipe,
            injector=injector,
            watchdog=watchdog,
            markers=markers,
            catalog=catalog,
            vis_i=vis_i,
            ankle_i=ankle_i,
            policy_path=policy_path,
            mapper_path=mapper_path,
            out=out,
        )
        n_fail = sum(1 for c in clips_out if c["fail"])
        payload = {
            "meta": (
                f"source=openxr  terrain={args_cli.terrain}  policy={policy_path}  "
                f"mapper={mapper_path}  port={args_cli.openxr_port}  "
                f"canon={args_cli.openxr_canon}  future=Mapper-B (no GT)  "
                f"fall={n_fail}/{len(clips_out)}"
            ),
            "policy": policy_path,
            "mapper": mapper_path,
            "terrain": args_cli.terrain,
            "source": "openxr",
            "n_fail": n_fail,
            "n": len(clips_out),
            "clips": clips_out,
        }
        html_path = out / "index.html"
        write_deploy_html(html_path, payload)
        print(f"[deploy] wrote {html_path}  fall={n_fail}/{len(clips_out)} ({100*n_fail/max(len(clips_out),1):.1f}%)", flush=True)
        env.close()
        return

    _pin_clips(env, catalog)
    use = min(int(env.unwrapped.num_envs), len(catalog))
    vis = torch.stack([vis_from_mask(c["mask"], env.device, 1)[0] for c in catalog[:use]])
    pkt0 = current_packet_world(env)[:use]
    frontend.reset(pkt0, delay_steps=injector.latency_steps)
    print(
        f"[deploy] Test1R unified stack  n={use}  vis={[(c['task'], c['mask']) for c in catalog[:use]]}",
        flush=True,
    )

    rec = {
        "robot": [[[] for _ in range(3)] for _ in range(use)],
        "intent": [[[] for _ in range(3)] for _ in range(use)],
        "e_torso": [[] for _ in range(use)],
        "e_lw": [[] for _ in range(use)],
        "e_rw": [[] for _ in range(use)],
        "roll": [[] for _ in range(use)],
        "pitch": [[] for _ in range(use)],
        "dz": [[] for _ in range(use)],
        "dz_eff": [[] for _ in range(use)],
        "cos": [[] for _ in range(use)],
        "angle": [[] for _ in range(use)],
        "hold": [[] for _ in range(use)],
        "h05": [[] for _ in range(use)],
        "fail_step": [None] * use,
        "fail_reason": [""] * use,
        "future": [None] * use,
    }

    steps = int(args_cli.steps)
    for t in range(steps):
        with torch.inference_mode():
            obs, _ = backend.get_observations()
            q, info = pipe.act(obs, env, vis=vis)
            backend.send_joint_target(q)

        st = backend.get_base_state()
        robot = st["robot_body_pos_w"][:use]
        goal = st["intent_body_pos_w"][:use].clone()
        if injector.last_cur_w is not None:
            goal[:, vis_i] = injector.last_cur_w[:use]
        delta = robot - goal
        e_t = torch.linalg.norm(delta[:, vis_i[0]], dim=-1)
        e_l = torch.linalg.norm(delta[:, vis_i[1]], dim=-1)
        e_r = torch.linalg.norm(delta[:, vis_i[2]], dim=-1)
        fail_z, fail_ori, _z, roll, pitch = physical_fall(
            env, use, z_min=float(args_cli.z_min), ori_rad=float(args_cli.ori_rad)
        )
        dz = info["delta_z"][:use].norm(dim=-1)
        dz_eff = info.get("delta_z_eff", info["delta_z"])[:use].norm(dim=-1)
        cos = info.get("cos_base_exec")
        ang = info.get("angle_base_exec")
        cos = cos[:use] if cos is not None else torch.ones(use, device=obs.device)
        ang = ang[:use] if ang is not None else torch.zeros(use, device=obs.device)
        hold = info["hold"][:use]
        ferr = injector.future_errors(env, injector.last_pred_abs)
        h05 = ferr["h_0.5"][:use]
        future_w = injector.last_pred_w

        if ankle_i:
            feet = robot[:, ankle_i, :]
        else:
            feet = None
        markers.update(
            intent_w=goal[:, vis_i, :],
            robot_w=robot,
            future_w=future_w,
            foot_w=feet,
        )

        for e in range(use):
            for k, bi in enumerate(vis_i):
                rec["robot"][e][k].append([float(robot[e, bi, d].cpu()) for d in range(3)])
                rec["intent"][e][k].append([float(goal[e, bi, d].cpu()) for d in range(3)])
            rec["e_torso"][e].append(float(e_t[e].cpu()))
            rec["e_lw"][e].append(float(e_l[e].cpu()))
            rec["e_rw"][e].append(float(e_r[e].cpu()))
            rec["roll"][e].append(float(roll[e].abs().cpu()))
            rec["pitch"][e].append(float(pitch[e].abs().cpu()))
            rec["dz"][e].append(float(dz[e].cpu()))
            rec["dz_eff"][e].append(float(dz_eff[e].cpu()))
            rec["cos"][e].append(float(cos[e].cpu()))
            rec["angle"][e].append(float(ang[e].cpu()))
            rec["hold"][e].append(1.0 if bool(hold[e]) else 0.0)
            rec["h05"][e].append(float(h05[e].cpu()))
            if rec["fail_step"][e] is None and t >= 10:
                if bool(fail_z[e]):
                    rec["fail_step"][e] = t
                    rec["fail_reason"][e] = "pelvis_z"
                elif bool(fail_ori[e]):
                    rec["fail_step"][e] = t
                    rec["fail_reason"][e] = "robot_ori"
            if t == steps - 1 and future_w is not None:
                rec["future"][e] = future_w[e].reshape(-1, 3).detach().cpu().tolist()

        if (t + 1) % 50 == 0:
            print(
                f"[deploy] step {t+1}/{steps} e_torso={float(e_t.mean()):.3f} "
                f"||dZ_raw||={float(dz.mean()):.3f} ||dZ_eff||={float(dz_eff.mean()):.3f} "
                f"cos={float(cos.mean()):.3f} ang={float(ang.mean())*180/3.1416:.1f}deg "
                f"hold={int(hold.sum())}/{use}",
                flush=True,
            )

    clips_out = []
    for e in range(use):
        et = np.asarray(rec["e_torso"][e])
        el = np.asarray(rec["e_lw"][e])
        er = np.asarray(rec["e_rw"][e])
        mask = catalog[e]["mask"]
        parts = [et]
        if mask not in ("torso",):
            if mask in ("vr", "head_left"):
                parts.append(el)
            if mask in ("vr", "head_right"):
                parts.append(er)
        e_vis = np.mean(np.stack(parts, 0), 0)
        fail = rec["fail_step"][e] is not None
        wrist = np.nanmean([el, er]) if mask != "torso" else float("nan")
        clips_out.append(
            {
                "task": catalog[e]["task"],
                "mask": mask,
                "name": catalog[e]["name"],
                "sr5": float((e_vis < 0.05).mean()),
                "e_torso": float(et.mean()),
                "e_wrist": float(wrist) if np.isfinite(wrist) else None,
                "roll": float(np.mean(rec["roll"][e])),
                "pitch": float(np.mean(rec["pitch"][e])),
                "dz_mean": float(np.mean(rec["dz"][e])),
                "dz_eff": float(np.mean(rec["dz_eff"][e])) if rec.get("dz_eff") else None,
                "cos_base_exec": float(np.mean(rec["cos"][e])) if rec.get("cos") else None,
                "angle_deg": float(np.mean(rec["angle"][e]) * 180.0 / np.pi) if rec.get("angle") else None,
                "mapper_h05": float(np.mean(rec["h05"][e])),
                "fail": fail,
                "fail_reason": rec["fail_reason"][e],
                "hold_n": int(np.sum(rec["hold"][e])),
                "robot": rec["robot"][e],
                "intent": rec["intent"][e],
                "future": rec["future"][e],
                "e_torso_t": rec["e_torso"][e],
                "e_lw_t": rec["e_lw"][e],
                "e_rw_t": rec["e_rw"][e],
                "pitch_t": rec["pitch"][e],
                "dz_t": rec["dz"][e],
                "hold_t": rec["hold"][e],
            }
        )
        print(
            f"[deploy] {catalog[e]['task']:5s} {mask:10s} SR5={100*(e_vis<0.05).mean():5.1f}% "
            f"torso={et.mean():.3f} fail={rec['fail_reason'][e] or 'ok'}  "
            f"||dZ||={np.mean(rec['dz'][e]):.2f} cos={np.mean(rec['cos'][e]):.3f} "
            f"ang={np.mean(rec['angle'][e])*180/np.pi:.1f}deg  {catalog[e]['name']}",
            flush=True,
        )

    n_fail = sum(1 for c in clips_out if c["fail"])
    mean_cos = float(np.mean([c["cos_base_exec"] for c in clips_out if c.get("cos_base_exec") is not None]))
    mean_ang = float(np.mean([c["angle_deg"] for c in clips_out if c.get("angle_deg") is not None]))
    mean_dz = float(np.mean([c["dz_mean"] for c in clips_out]))
    mean_dze = float(np.mean([c["dz_eff"] for c in clips_out if c.get("dz_eff") is not None]))
    payload = {
        "meta": (
            f"source=replay-unified  terrain={args_cli.terrain}  "
            f"latency_ms={args_cli.latency_ms} shift={int(args_cli.latency_shift)}  "
            f"hand_noise_cm={args_cli.hand_noise_cm}  "
            f"||dZ_raw||={mean_dz:.2f} ||dZ_eff||={mean_dze:.2f} "
            f"cos={mean_cos:.3f} angle={mean_ang:.1f}deg  "
            f"physical_fall={n_fail}/{use}"
        ),
        "policy": policy_path,
        "mapper": mapper_path,
        "terrain": args_cli.terrain,
        "n_fail": n_fail,
        "n": use,
        "clips": clips_out,
    }
    html_path = out / "index.html"
    write_deploy_html(html_path, payload)
    print(f"[deploy] wrote {html_path}  fall={n_fail}/{use} ({100*n_fail/max(use,1):.1f}%)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
