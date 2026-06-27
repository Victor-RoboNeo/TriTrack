"""Diagnose the ObstacleReach NaN: build num_envs=1 (like play), reset, and step while
printing robot-root finiteness + position. Isolates reset/env NaN (zero-action) from
policy-driven explosion. No checkpoint needed.

    CUDA_VISIBLE_DEVICES=7 python -u scripts/diag_obstacle_reach.py --motion outputs/obstacle_seed/phase0 --headless --steps 200
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="MUSE-Kp-LatentRL-Kp5-ObstacleReach-General-Tracking-Flat-G1-v0")
parser.add_argument("--motion", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--action", choices=["zero", "rand"], default="zero")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import gymnasium as gym
import torch

import whole_body_tracking.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def root_state(unwrapped):
    rb = unwrapped.scene["robot"].data
    return rb.root_pos_w.clone(), rb.root_quat_w.clone()


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.commands.motion.motion = args.motion
    if getattr(env_cfg, "curriculum", None) is not None and hasattr(env_cfg.curriculum, "keypoint_mask_mode"):
        env_cfg.curriculum.keypoint_mask_mode = None
    print(f"[diag] building num_envs={args.num_envs} action={args.action} motion={args.motion}")
    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    obs, _ = env.reset()
    cmd = unwrapped.command_manager.get_term("motion")
    sc = cmd.obstacle_scene
    p, q = root_state(unwrapped)
    print(f"[diag] post-reset root_pos={[round(v,4) for v in p[0].tolist()]} "
          f"root_quat={[round(v,4) for v in q[0].tolist()]} "
          f"target_rel={[round(v,4) for v in cmd._reach_target_rel[0].tolist()]} "
          f"obst_valid={int(sc.valid[0].sum())}")
    # finiteness of the post-reset obs groups
    om = unwrapped.observation_manager
    g = om.compute()
    for k, v in g.items():
        nf = int((~torch.isfinite(v)).sum())
        print(f"[diag]   obs[{k}] shape={tuple(v.shape)} nonfinite={nf}")

    adim = unwrapped.action_manager.total_action_dim
    dev = unwrapped.device
    first_bad = -1
    for i in range(args.steps):
        a = torch.zeros((args.num_envs, adim), device=dev) if args.action == "zero" \
            else torch.randn((args.num_envs, adim), device=dev) * 0.1
        obs, rew, term, trunc, info = env.step(a)
        p, q = root_state(unwrapped)
        rp_fin = bool(torch.isfinite(p).all())
        rq_fin = bool(torch.isfinite(q).all())
        rew_fin = bool(torch.isfinite(rew).all())
        if (not rp_fin or not rq_fin or not rew_fin) and first_bad < 0:
            first_bad = i
        if i < 5 or not rp_fin or not rq_fin or (i % 25 == 0):
            print(f"[diag] step {i:3d}: root_pos={[round(v,3) for v in p[0].tolist()]} "
                  f"pos_fin={rp_fin} quat_fin={rq_fin} rew_fin={rew_fin} "
                  f"term={int(term.sum())} trunc={int(trunc.sum())}")
        if first_bad >= 0 and i >= first_bad + 3:
            break

    print(f"[diag] FIRST NON-FINITE STEP: {first_bad}" if first_bad >= 0
          else f"[diag] all {args.steps} steps finite (action={args.action})")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
