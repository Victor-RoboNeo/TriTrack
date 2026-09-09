"""Isaac: forward dumped obs through multiple checkpoints for policy drift."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ANYBODY = Path("/data/home/chenxiangyu/robotics/Anybody")
HT_ROOT = Path("/data/home/chenxiangyu/humantracker_3pt_ood")
sys.path.insert(0, str(ANYBODY / "scripts"))
sys.path.insert(0, str(HT_ROOT))
sys.path.insert(0, "/data/home/chenxiangyu/victor/TriTrack")

from isaaclab.app import AppLauncher  # noqa: E402

from flat_locomani.next5.constants import CLIP_LOCO, TASK  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--obs-npz", required=True)
ap.add_argument("--policies", required=True)
ap.add_argument("--out-json", required=True)
ap.add_argument("--batch", type=int, default=256)
AppLauncher.add_app_launcher_args(ap)
args_cli, _ = ap.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
import whole_body_tracking.tasks  # noqa: E402, F401
from whole_body_tracking.tasks.tracking.mdp.curriculums import attach_curriculum_rollout_hints  # noqa: E402
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner  # noqa: E402
from flat_nominal_competence_repair_v2.eval_ncr import _apply_eval_motion, _disable_eval_hooks, _split_encode, parse_policies  # noqa: E402


def main():
    z = np.load(args_cli.obs_npz)
    obs_all = np.asarray(z["obs"], dtype=np.float32)
    n = int(obs_all.shape[0])
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1, use_fabric=True)
    _apply_eval_motion(env_cfg, str(CLIP_LOCO), start_frame=10)
    env_cfg.scene.terrain.terrain_type = "plane"
    _disable_eval_hooks(env_cfg)
    gym_env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device
    attach_curriculum_rollout_hints(gym_env, spe=24, learning_iteration=50000)
    env = RslRlVecEnvWrapper(gym_env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir="/tmp/ncr_drift", device=agent_cfg.device)
    device = env.device
    outs = {}
    policies = parse_policies(args_cli.policies)
    parent_z = parent_g = parent_a = None
    for name, ckpt in policies:
        runner.load(str(ckpt), load_optimizer=False, load_critic=True)
        pol = runner.alg.policy
        zs, gs, acts = [], [], []
        with torch.no_grad():
            for i in range(0, n, args_cli.batch):
                sl = obs_all[i : i + args_cli.batch]
                tm = torch.as_tensor(sl, dtype=torch.float32, device=device)
                z_enc, g, z_f, a = _split_encode(pol, tm)
                zs.append(z_f.detach().cpu().numpy())
                gs.append(g.detach().cpu().numpy())
                acts.append(a.detach().cpu().numpy())
        Z = np.concatenate(zs)
        G = np.concatenate(gs)
        A = np.concatenate(acts)
        outs[name] = {"z": Z, "g": G, "a": A}
        if name in ("PARENT", "A"):
            parent_z, parent_g, parent_a = Z, G, A
    summary = {"n": n}
    for name, pack in outs.items():
        if parent_z is None:
            continue
        dz = np.linalg.norm(pack["z"] - parent_z, axis=-1)
        dg = np.linalg.norm(pack["g"] - parent_g, axis=-1)
        da = np.linalg.norm(pack["a"] - parent_a, axis=-1)
        summary[name] = {
            "D_final_latent": {"mean": float(dz.mean()), "median": float(np.median(dz)), "p95": float(np.percentile(dz, 95)), "max": float(dz.max())},
            "D_gphi": {"mean": float(dg.mean()), "median": float(np.median(dg)), "p95": float(np.percentile(dg, 95)), "max": float(dg.max())},
            "D_action": {"mean": float(da.mean()), "median": float(np.median(da)), "p95": float(np.percentile(da, 95)), "max": float(da.max())},
        }
    Path(args_cli.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.out_json).write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(Path(args_cli.out_json).with_suffix(".npz"), **{f"{k}_{kk}": vv for k, p in outs.items() for kk, vv in p.items()})
    print(json.dumps(summary, indent=2))
    import os
    os._exit(0)


if __name__ == "__main__":
    import os, traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        os._exit(1)
