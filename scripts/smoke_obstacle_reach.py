"""Smoke-test the ObstacleReach env: build it, step a few times, confirm the obstacle
scene + keep-out reward wire up. No policy/checkpoint needed (env-only).

    CUDA_VISIBLE_DEVICES=7 python scripts/smoke_obstacle_reach.py --motion /tmp/obs_pool_test/phase1 --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="MUSE-Kp-LatentRL-Kp5-ObstacleReach-General-Tracking-Flat-G1-v0")
parser.add_argument("--motion", required=True, help="phase clip pool dir (npz + .scene.json sidecars)")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=6)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import gymnasium as gym
import torch

import whole_body_tracking.tasks  # noqa: F401  (registers the gym tasks)
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.commands.motion.motion = args.motion
    # Env-only smoke: the mask curriculum needs runner-set rollout hints we don't have here.
    # Disable it (training via train.py sets the hints). Mask then stays at the sampled mix.
    if getattr(env_cfg, "curriculum", None) is not None and hasattr(env_cfg.curriculum, "keypoint_mask_mode"):
        env_cfg.curriculum.keypoint_mask_mode = None
    print(f"[smoke] building {args.task} num_envs={args.num_envs} motion={args.motion}")
    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    obs, _ = env.reset()
    cmd = unwrapped.command_manager.get_term("motion")
    print(f"[smoke] command class: {type(cmd).__name__}")
    sc = cmd.obstacle_scene
    print(f"[smoke] obstacle_scene: valid/env={sc.valid.sum(-1)[:8].tolist()}  "
          f"target[0]={[round(v,3) for v in sc.target[0].tolist()]}")

    # reward terms present?
    rm = unwrapped.reward_manager
    print(f"[smoke] reward terms: {list(rm.active_terms)}")

    # obs term ORDER per group — 'obstacle' MUST be last in policy (the actor-critic strips it).
    om = unwrapped.observation_manager
    for g in ("policy", "critic"):
        try:
            names = om.active_terms[g]
            dims = om.group_obs_term_dim[g]
            print(f"[smoke] {g} obs terms: {names}")
            print(f"[smoke] {g} obs dims : {dims}")
        except Exception as e:
            print(f"[smoke] {g} obs terms: <err {type(e).__name__}: {e}>")

    adim = unwrapped.action_manager.total_action_dim
    dev = unwrapped.device
    for i in range(args.steps):
        a = torch.zeros((args.num_envs, adim), device=dev)
        obs, rew, term, trunc, info = env.step(a)
        finite = bool(torch.isfinite(rew).all())
        print(f"[smoke] step {i}: rew_mean={float(rew.mean()):.4f} finite={finite} "
              f"term={int(term.sum())} trunc={int(trunc.sum())}")

    # keep-out term value distribution (penetration; should be >=0, finite)
    ko = rm.get_term_cfg("obstacle_keepout") if "obstacle_keepout" in rm.active_terms else None
    print(f"[smoke] obstacle_keepout term present: {ko is not None}")
    env.close()
    print("[smoke] SMOKE OK")


if __name__ == "__main__":
    main()
    app.close()
