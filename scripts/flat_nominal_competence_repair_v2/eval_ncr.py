#!/usr/bin/env python3
"""WORLD live-anchor eval for FLAT_NOMINAL_COMPETENCE_REPAIR_V2.

Canonical Chest+LH+RH. Deterministic mean action. No 18-D task error. Plane only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ANYBODY = Path("/data/home/chenxiangyu/robotics/Anybody")
HT_ROOT = Path("/data/home/chenxiangyu/humantracker_3pt_ood")
TRITRACK_ROOT = Path("/data/home/chenxiangyu/victor/TriTrack")
sys.path.insert(0, str(ANYBODY / "scripts"))
sys.path.insert(0, str(HT_ROOT))
sys.path.insert(0, str(TRITRACK_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

from flat_locomani.next5.constants import CLIP_LOCO, FPS, TASK  # noqa: E402

FAIL_Z_MIN = 0.40
FAIL_ORI_RAD = 1.2

ap = argparse.ArgumentParser()
ap.add_argument("--suite", required=True)
ap.add_argument("--policies", required=True, help="name:path,name:path")
ap.add_argument("--methods", default="known_preview")
ap.add_argument("--max-traj-per-family", type=int, default=64)
ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--out-parquet", required=True)
ap.add_argument("--dump-propagation", default="")
ap.add_argument("--dump-lower-body", default="")
ap.add_argument("--dump-obs", default="", help="npz of packed obs for drift (parent only)")
ap.add_argument("--families", default="", help="comma family prefixes; empty=all")
ap.add_argument("--invariance-check", action="store_true")
ap.add_argument("--mask", default="1,1,1", help="C,LH,RH active bits")
AppLauncher.add_app_launcher_args(ap)
args_cli, _ = ap.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import whole_body_tracking.tasks  # noqa: E402, F401
from tritrack.robot.isaaclab_bridge import IsaacLabG1Bridge  # noqa: E402
from whole_body_tracking.tasks.tracking.mdp.curriculums import attach_curriculum_rollout_hints  # noqa: E402
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner  # noqa: E402

from flat_locomani.eval_helpers import live_anchor_current_and_future, load_suite  # noqa: E402
from flat_locomani.next5.live_obs import LiveAnchorHistory, overlay_obs, pack_kp_mask, robot_kp_live_anchor  # noqa: E402
from flat_locomani.next5.tables import write_rows  # noqa: E402
from flat_locomani.live_anchor_v2.contract_audit import gym_pack_3pt, live_pack_3pt  # noqa: E402
from flat_locomani.live_anchor_v2.motion_convert import load_seed_and_index, world_to_motion_delta_seed, motion_to_world3  # noqa: E402

from flat_unified_sparse_intent_v1.metrics_active import episode_metrics_fusi  # noqa: E402
from flat_nominal_competence_repair_v2.canonicalize import (  # noqa: E402
    SYNTHETIC_TEST_CAL,
    canonical_3x3,
    canonicalize_sparse_input,
    head_from_canonical_chest,
)


def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _roll_pitch(quat_wxyz: np.ndarray, device: str) -> tuple[float, float]:
    q = torch.as_tensor(quat_wxyz, dtype=torch.float32, device=device).unsqueeze(0)
    roll, pitch, _ = euler_xyz_from_quat(q)
    return float(_wrap_pi(roll)[0]), float(_wrap_pi(pitch)[0])


def _apply_eval_motion(env_cfg, motion: str, start_frame: int) -> None:
    env_cfg.commands.motion.motion = motion
    if hasattr(env_cfg.commands.motion, "motion_groups"):
        env_cfg.commands.motion.motion_groups = None
    if hasattr(env_cfg.commands.motion, "motion_group_sampling_ratios"):
        env_cfg.commands.motion.motion_group_sampling_ratios = None
    if hasattr(env_cfg.commands.motion, "start_frame"):
        env_cfg.commands.motion.start_from_beginning = True
        env_cfg.commands.motion.start_frame = start_frame
    if hasattr(env_cfg.commands.motion, "random_init_frame"):
        env_cfg.commands.motion.random_init_frame = False
    if hasattr(env_cfg.commands.motion, "resample_motions_every_s"):
        env_cfg.commands.motion.resample_motions_every_s = 0.0
    zero = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    if hasattr(env_cfg.commands.motion, "pose_range"):
        env_cfg.commands.motion.pose_range = dict(zero)
    if hasattr(env_cfg.commands.motion, "velocity_range"):
        env_cfg.commands.motion.velocity_range = dict(zero)
    if hasattr(env_cfg.commands.motion, "joint_position_range"):
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    if hasattr(env_cfg.commands.motion, "motion_dataset_shard_across_gpus"):
        env_cfg.commands.motion.motion_dataset_shard_across_gpus = False
    if hasattr(env_cfg.commands.motion, "max_active_motions"):
        env_cfg.commands.motion.max_active_motions = None


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
    if hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = 1.0e6


def select_items(items, max_per_family: int, families: str = ""):
    allow = [x.strip() for x in families.split(",") if x.strip()]
    by_f: dict[str, list] = {}
    for w, r in items:
        fam = str(r.get("family", "unk"))
        if allow and not any(fam.startswith(a) for a in allow):
            continue
        by_f.setdefault(fam, []).append((w, r))
    out = []
    for fam in sorted(by_f):
        out.extend(by_f[fam][:max_per_family])
    return out


def parse_policies(s: str) -> list[tuple[str, Path]]:
    out = []
    for part in s.split(","):
        name, path = part.split(":", 1)
        out.append((name.strip(), Path(path.strip())))
    return out


def pack_one(obs, world, t, rs, hist, method):
    live_now, live_fut = live_anchor_current_and_future(
        world, t, rs.anchor_pos_w, rs.anchor_quat_w, method, h_ms=None
    )
    if not hist._buf:
        hist.reset(live_now)
    else:
        hist.push(live_now)
    robot_kp = robot_kp_live_anchor(rs.kp_body_pos_w, rs.anchor_pos_w, rs.anchor_quat_w)
    kp, mask = pack_kp_mask(live_now, live_fut, hist.past_abs(), robot_kp)
    return overlay_obs(obs, kp, mask)


def _split_encode(policy, obs_t: torch.Tensor):
    enc_obs, scan, obstacle_feat, obstacle_mask, rec_aux = policy._split_policy_obs(obs_t)
    mu, _log_sigma, proprio = policy.muse.transformer_encoder.encode(enc_obs)
    z_enc = policy.muse._maybe_normalize_latent(mu)
    g = None
    if getattr(policy, "residual_corrector", None) is not None:
        kp, kp_mask, _prop = policy.muse.transformer_encoder.split_obs(enc_obs)
        g = policy.residual_corrector(kp, kp_mask, proprio, z_enc.detach(), obstacle_feat, obstacle_mask)
        z_final = policy.muse._maybe_normalize_latent(z_enc.detach() + policy.residual_corrector.alpha * g)
    else:
        z_final = z_enc
        g = torch.zeros_like(z_enc)
    action = policy.muse._decode(z_final, proprio)
    return z_enc, g, z_final, action


def _run_invariance(policy_mod, template_obs, device, dtype) -> dict:
    rng = np.random.default_rng(2026)
    cal = SYNTHETIC_TEST_CAL
    diffs = {"z": [], "g": [], "a": []}
    dummy = np.asarray(template_obs, dtype=np.float64)
    from flat_locomani.next5.live_obs import overlay_obs, pack_kp_mask
    from flat_locomani.live_anchor import world_targets_to_live_anchor

    torso = np.array([0.0, 0.0, 0.78])
    quat = np.array([1.0, 0, 0, 0.0])
    robot = np.zeros((5, 3))
    robot[0] = [0, 0, 0.3]
    for _ in range(200):
        c = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), rng.uniform(0.68, 0.85)])
        lh = c + np.array([0.18, 0.22, -0.08])
        rh = c + np.array([0.18, -0.22, -0.08])
        cq = np.array([1.0, 0, 0, 0.0])
        chest = canonicalize_sparse_input("chest_hands", {"chest": {"pos": c, "quat_wxyz": cq}, "left_hand": lh, "right_hand": rh})
        h_pos, h_q = head_from_canonical_chest(c, cq, cal)
        head = canonicalize_sparse_input(
            "head_hands",
            {"head": {"pos": h_pos, "quat_wxyz": h_q}, "left_hand": lh, "right_hand": rh},
            calibration=cal,
            allow_synthetic=True,
        )
        w1, w2 = canonical_3x3(chest), canonical_3x3(head)
        live1 = world_targets_to_live_anchor(w1, torso, quat)
        live2 = world_targets_to_live_anchor(w2, torso, quat)
        fut = np.repeat(live1[None], 7, 0)
        past = np.repeat(live1[None], 7, 0)
        kp1, m1 = pack_kp_mask(live1, fut, past, robot)
        kp2, m2 = pack_kp_mask(live2, np.repeat(live2[None], 7, 0), np.repeat(live2[None], 7, 0), robot)
        o1 = overlay_obs(dummy, kp1, m1)
        o2 = overlay_obs(dummy, kp2, m2)
        with torch.no_grad():
            tm = torch.as_tensor(np.stack([o1, o2]), dtype=dtype, device=device)
            z, g, zf, a = _split_encode(policy_mod, tm)
        diffs["z"].append(float(torch.norm(z[0] - z[1]).item()))
        diffs["g"].append(float(torch.norm(g[0] - g[1]).item()))
        diffs["a"].append(float(torch.norm(a[0] - a[1]).item()))
    mx = {k: float(np.max(v)) for k, v in diffs.items()}
    mx["PASS"] = all(mx[k] < 1e-5 for k in ("z", "g", "a"))
    mx["n"] = 200
    return mx


def main() -> None:
    mask = np.array([float(x) for x in args_cli.mask.split(",")], dtype=np.float64)
    items = select_items(load_suite(Path(args_cli.suite)), args_cli.max_traj_per_family, args_cli.families)
    n_env = len(items)
    if n_env == 0:
        raise RuntimeError(f"empty suite {args_cli.suite}")
    worlds = [w for w, _ in items]
    recs = [r for _, r in items]
    clip_len = np.array([w.shape[0] for w in worlds], dtype=int)
    n_steps = int(clip_len.max())
    if args_cli.max_steps > 0:
        n_steps = min(n_steps, int(args_cli.max_steps))
        clip_len = np.minimum(clip_len, n_steps)
    policies = parse_policies(args_cli.policies)
    for name, path in policies:
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint {name}: {path}")
        print(f"[eval] policy={name} sha256={hashlib.sha256(path.read_bytes()).hexdigest()} path={path}", flush=True)

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=n_env, use_fabric=True)
    _apply_eval_motion(env_cfg, str(CLIP_LOCO), start_frame=10)
    env_cfg.scene.terrain.terrain_type = "plane"
    _disable_eval_hooks(env_cfg)
    gym_env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device
    agent_cfg.seed = int(args_cli.seed)
    attach_curriculum_rollout_hints(gym_env, spe=int(getattr(agent_cfg, "num_steps_per_env", 24) or 24), learning_iteration=50000)
    env = RslRlVecEnvWrapper(gym_env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir="/tmp/ncr_eval", device=agent_cfg.device)
    device = env.device
    bridges = [IsaacLabG1Bridge(env.unwrapped.scene["robot"], env_index=e) for e in range(n_env)]
    methods = [m.strip() for m in args_cli.methods.split(",") if m.strip()]
    dump_prop = Path(args_cli.dump_propagation) if args_cli.dump_propagation else None
    dump_lb = Path(args_cli.dump_lower_body) if args_cli.dump_lower_body else None
    seed0, idxs = (None, None)
    if dump_prop:
        seed0, idxs = load_seed_and_index()

    rows = []
    prop_rows = []
    for pol_name, ckpt in policies:
        runner.load(str(ckpt), load_optimizer=False, load_critic=True)
        policy_fn = runner.get_inference_policy(device=device)
        policy_mod = runner.alg.policy
        sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        for method in methods:
            try:
                obs, extras = env.reset()
            except Exception:
                obs = env.get_observations()
                if isinstance(obs, tuple):
                    obs, extras = obs
            offsets = []
            for e in range(n_env):
                rs = bridges[e].robot_state()
                offsets.append(rs.anchor_pos_w - worlds[e][0, 0])
            offsets = np.stack(offsets, axis=0)
            hists = [LiveAnchorHistory() for _ in range(n_env)]
            fallen = np.zeros(n_env, dtype=bool)
            fall_tick = np.full(n_env, -1, dtype=int)
            robot_log = [np.zeros((int(clip_len[e]), 3, 3), dtype=np.float64) for e in range(n_env)]
            target_log = [np.zeros((int(clip_len[e]), 3, 3), dtype=np.float64) for e in range(n_env)]
            surv_log = [np.zeros(int(clip_len[e]), dtype=bool) for e in range(n_env)]
            joint_log = [[] for _ in range(n_env)]
            obs_np = obs.detach().cpu().numpy()
            act_dim_ok = True
            obs_dump = []
            obs_traj = []
            if args_cli.invariance_check and pol_name == policies[0][0]:
                inv = _run_invariance(policy_mod, obs_np[0], device, obs.dtype)
                Path(args_cli.out_parquet).parent.mkdir(parents=True, exist_ok=True)
                Path(str(args_cli.out_parquet) + ".invariance.json").write_text(json.dumps(inv, indent=2) + "\n")
                print("[eval] invariance", inv, flush=True)
            for t in range(n_steps):
                patched = np.zeros_like(obs_np)
                for e in range(n_env):
                    rs = bridges[e].robot_state()
                    world_e = worlds[e] + offsets[e][None, None, :]
                    patched[e] = pack_one(obs_np[e], world_e, t, rs, hists[e], method)
                    if args_cli.dump_obs and pol_name in ("PARENT", "A") and len(obs_dump) < 10000:
                        fam = str(recs[e].get("family", ""))
                        if fam.startswith("F1") or fam.startswith("F2"):
                            obs_dump.append(patched[e].astype(np.float32))
                            obs_traj.append(str(recs[e].get("traj_id")))
                    if dump_prop and seed0 is not None and len(prop_rows) < 1200:
                        mot_m = world_to_motion_delta_seed(world_e, seed0, idxs)
                        world_m = motion_to_world3(mot_m, idxs)
                        kp_m, mask_m = gym_pack_3pt(world_m, t, rs.anchor_pos_w, rs.anchor_quat_w, rs.kp_body_pos_w)
                        kp_l, mask_l = live_pack_3pt(world_e, t, rs.anchor_pos_w, rs.anchor_quat_w, rs.kp_body_pos_w)
                        obs_m = overlay_obs(obs_np[e], kp_m, mask_m)
                        obs_l = overlay_obs(obs_np[e], kp_l, mask_l)
                        with torch.no_grad():
                            tm = torch.as_tensor(np.stack([obs_m, obs_l]), dtype=obs.dtype, device=device)
                            z_enc, g, z_f, act = _split_encode(policy_mod, tm)
                        d_obs = float(np.linalg.norm(np.nan_to_num(obs_m - obs_l)))
                        prop_rows.append(
                            {
                                "d_obs": d_obs,
                                "d_z": float(torch.norm(z_enc[0] - z_enc[1]).item()),
                                "d_g": float(torch.norm(g[0] - g[1]).item()),
                                "d_z_final": float(torch.norm(z_f[0] - z_f[1]).item()),
                                "d_action": float(torch.norm(act[0] - act[1]).item()),
                                "action_dim": int(act.shape[-1]),
                                "traj": recs[e].get("traj_id"),
                                "t": t,
                            }
                        )
                act = policy_fn(torch.as_tensor(patched, dtype=obs.dtype, device=device))
                if int(act.shape[-1]) != 29:
                    act_dim_ok = False
                obs, rew, dones, extras = env.step(act)
                obs_np = obs.detach().cpu().numpy()
                for e in range(n_env):
                    if fallen[e] or t >= clip_len[e]:
                        continue
                    rs = bridges[e].robot_state()
                    z = float(rs.anchor_pos_w[2])
                    roll, pitch = _roll_pitch(rs.anchor_quat_w, args_cli.device)
                    if z < FAIL_Z_MIN or abs(roll) > FAIL_ORI_RAD or abs(pitch) > FAIL_ORI_RAD:
                        fallen[e] = True
                        fall_tick[e] = t
                    robot_log[e][t] = rs.kp_body_pos_w[:3]
                    target_log[e][t] = worlds[e][t] + offsets[e]
                    surv_log[e][t] = not fallen[e]
                    if dump_lb:
                        jp = np.asarray(rs.joint_pos, dtype=np.float64)
                        joint_log[e].append(
                            {
                                "t": t,
                                "base_z": z,
                                "joint_pos": jp.tolist(),
                                "head_z_tgt": float(target_log[e][t, 0, 2]),
                                "head_z_act": float(robot_log[e][t, 0, 2]),
                                "lh_z_tgt": float(target_log[e][t, 1, 2]),
                                "rh_z_tgt": float(target_log[e][t, 2, 2]),
                            }
                        )
                if (t + 1) % 100 == 0:
                    print(f"[eval] {pol_name} {method} {t+1}/{n_steps} fallen={int(fallen.sum())}", flush=True)
            for e, rec in enumerate(recs):
                T = int(clip_len[e])
                mets = episode_metrics_fusi(
                    robot_log[e],
                    target_log[e],
                    survived=surv_log[e],
                    phases=rec.get("phases") or [],
                    target_height_drop_m=float(rec.get("target_height_drop_m") or 0.0),
                    h0_m=float(rec.get("h0_m") or 0.78),
                    hand_mode=str(rec.get("hand_mode") or ""),
                    mask=mask,
                    fps=FPS,
                )
                rows.append(
                    {
                        "policy": pol_name,
                        "checkpoint": str(ckpt),
                        "sha256": sha,
                        "method": method,
                        "family": rec.get("family"),
                        "campaign_family": rec.get("family"),
                        "traj_id": rec.get("traj_id"),
                        "fell": bool(fallen[e]),
                        "fall_tick": int(fall_tick[e]),
                        "n_steps_planned": T,
                        "action_dim_ok": act_dim_ok,
                        "terrain": "plane",
                        **{k: v for k, v in mets.items() if k != "phases"},
                    }
                )
            write_rows(Path(args_cli.out_parquet), rows)
            print(f"[eval] {pol_name} {method} n={len(rows)} fall={fallen.mean():.3f}", flush=True)
            if dump_lb:
                dump_lb.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dump_lb, allow_pickle=True, logs=np.array(joint_log, dtype=object))
            if args_cli.dump_obs and obs_dump:
                pobs = Path(args_cli.dump_obs)
                pobs.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    pobs,
                    obs=np.stack(obs_dump, axis=0),
                    traj_id=np.array(obs_traj, dtype=object),
                )
                print(f"[eval] dumped obs {len(obs_dump)} -> {pobs}", flush=True)

    out = Path(args_cli.out_parquet)
    write_rows(out, rows)
    if dump_prop:
        dump_prop.parent.mkdir(parents=True, exist_ok=True)
        dump_prop.write_text(json.dumps(prop_rows) + "\n")
        print(f"[eval] propagation samples {len(prop_rows)} -> {dump_prop}", flush=True)
    print(f"[eval] wrote {out} n={len(rows)}", flush=True)
    import os as _os

    _os._exit(0)


if __name__ == "__main__":
    import os
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
        os._exit(1)
