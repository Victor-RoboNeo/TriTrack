"""Diagnose g_φ terrain information + per-terrain PPO actor-gradient cosine.

Loads frozen ``model_50000`` on the P2 mix. Does not train. Does not resume P2-A/B.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="P2-C preflight: obs inventory + terrain gradient cosine.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--motion", type=str, required=True)
parser.add_argument("--out", type=str, default="results/p2c_diag/terrain_grad.json")
parser.add_argument("--min_samples", type=int, default=256)
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.mdp.curriculums import attach_curriculum_rollout_hints
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

TERRAIN_GROUPS = {
    "flat": ("flat", "plane"),
    "light": ("slightly_rough", "light_rough"),
    "slope": ("slope", "slope_inv"),
    "steps": ("stairs", "stairs_inv", "steps"),
}
GROUP_ORDER = ("flat", "light", "slope", "steps")
TERRAIN_TOKENS = (
    "height",
    "scan",
    "ray",
    "depth",
    "elev",
    "lidar",
    "camera",
    "map",
    "terrain",
)


def _restore_adapter(agent_cfg, resume_path: str) -> None:
    params = Path(resume_path).parent / "params" / "agent.yaml"
    if not params.exists():
        return
    cfg = yaml.safe_load(params.read_text()) or {}
    policy_cfg = cfg.get("policy") or {}
    for key in (
        "adapter",
        "residual_d_model",
        "residual_num_layers",
        "residual_nhead",
        "residual_ffn",
        "residual_last_layer_gain",
        "residual_alpha",
        "terrain_scan_dim",
        "terrain_r_max",
        "terrain_scan_zero",
    ):
        if key in policy_cfg and hasattr(agent_cfg.policy, key):
            setattr(agent_cfg.policy, key, policy_cfg[key])


def _replay_subterrain_type_map(gen_cfg):
    names = list(gen_cfg.sub_terrains.keys())
    proportions = np.array([float(gen_cfg.sub_terrains[k].proportion) for k in names], dtype=np.float64)
    proportions = proportions / proportions.sum()
    num_rows = int(gen_cfg.num_rows)
    num_cols = int(gen_cfg.num_cols)
    type_map = np.zeros((num_rows, num_cols), dtype=np.int64)
    if bool(gen_cfg.curriculum):
        cumsum = np.cumsum(proportions)
        for col in range(num_cols):
            sub_index = int(np.min(np.where(col / num_cols + 0.001 < cumsum)[0]))
            type_map[:, col] = sub_index
        return names, type_map
    if gen_cfg.seed is None:
        raise RuntimeError("terrain generator seed is required to replay random cell types")
    rng = np.random.default_rng(int(gen_cfg.seed))
    diff_lo, diff_hi = gen_cfg.difficulty_range
    for index in range(num_rows * num_cols):
        sub_row, sub_col = np.unravel_index(index, (num_rows, num_cols))
        sub_index = int(rng.choice(len(proportions), p=proportions))
        rng.uniform(float(diff_lo), float(diff_hi))
        type_map[sub_row, sub_col] = sub_index
    return names, type_map


def _env_group_ids(env) -> tuple[dict[str, torch.Tensor], list[str], dict[str, int]]:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    terrain = unwrapped.scene.terrain
    gen_cfg = getattr(terrain.cfg, "terrain_generator", None)
    names, type_map = _replay_subterrain_type_map(gen_cfg)
    rows = terrain.terrain_levels.detach().long().cpu().clamp(0, type_map.shape[0] - 1)
    cols = terrain.terrain_types.detach().long().cpu().clamp(0, type_map.shape[1] - 1)
    cell_type = torch.from_numpy(type_map[rows.numpy(), cols.numpy()])
    counts = {names[i]: int((cell_type == i).sum().item()) for i in range(len(names))}
    group_ids: dict[str, torch.Tensor] = {}
    for group, aliases in TERRAIN_GROUPS.items():
        idx = [i for i, n in enumerate(names) if n in aliases]
        mask = torch.zeros(cell_type.numel(), dtype=torch.bool)
        for i in idx:
            mask |= cell_type == i
        group_ids[group] = mask
    return group_ids, names, counts


def _obs_inventory(env) -> dict:
    unwrapped = env.unwrapped
    om = unwrapped.observation_manager
    inventory = {"policy": [], "critic": [], "teacher": []}
    for group in inventory:
        terms = []
        if hasattr(om, "active_terms"):
            names = list(om.active_terms.get(group, []) or [])
        else:
            names = []
        dims = None
        for attr in ("group_obs_term_dim", "_group_obs_term_dim"):
            raw = getattr(om, attr, None)
            if isinstance(raw, dict) and group in raw:
                dims = raw[group]
                break
        for i, name in enumerate(names):
            dim = None
            if dims is not None:
                try:
                    d = dims[i]
                    dim = int(np.prod(d)) if hasattr(d, "__len__") else int(d)
                except Exception:
                    dim = None
            terms.append(
                {
                    "name": str(name),
                    "dim": dim,
                    "looks_like_terrain": any(tok in str(name).lower() for tok in TERRAIN_TOKENS),
                }
            )
        inventory[group] = terms
    scene_keys = []
    try:
        scene_keys = sorted(str(k) for k in unwrapped.scene.keys())
    except Exception:
        scene_keys = []
    sensors = [k for k in scene_keys if any(tok in k.lower() for tok in TERRAIN_TOKENS)]
    return {
        "groups": inventory,
        "scene_keys": scene_keys,
        "scene_keys_looking_like_terrain": sensors,
        "policy_has_terrain_term": any(t["looks_like_terrain"] for t in inventory["policy"]),
        "critic_has_terrain_term": any(t["looks_like_terrain"] for t in inventory["critic"]),
    }


def _flatten_grad(grads, params) -> torch.Tensor:
    chunks = []
    for g, p in zip(grads, params):
        if g is None:
            chunks.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            chunks.append(g.reshape(-1))
    return torch.cat(chunks)


def _actor_grad(alg, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    storage = alg.storage
    obs = storage.observations.flatten(0, 1)[mask]
    actions = storage.actions.flatten(0, 1)[mask]
    adv = storage.advantages.flatten(0, 1)[mask]
    old_lp = storage.actions_log_prob.flatten(0, 1)[mask]
    n = int(mask.sum().item())
    if n == 0:
        raise RuntimeError("empty terrain mask")
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    alg.policy.train()
    alg.policy.act(obs)
    logp = alg.policy.get_actions_log_prob(actions)
    ratio = torch.exp(logp - torch.squeeze(old_lp))
    clip = float(alg.clip_param)
    surr = -torch.squeeze(adv) * ratio
    surr_c = -torch.squeeze(adv) * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    loss = torch.max(surr, surr_c).mean()
    params = [p for p in alg.policy.residual_corrector.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return _flatten_grad(grads, params).detach(), n


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = float(a.norm().item())
    nb = float(b.norm().item())
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(torch.dot(a, b).item() / (na * nb))


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    env_cfg.commands.motion.motion = args_cli.motion
    resume_path = os.path.abspath(str(args_cli.resume_student_checkpoint))
    _restore_adapter(agent_cfg, resume_path)
    if hasattr(agent_cfg.policy, "adapter"):
        agent_cfg.policy.adapter = "residual"
    agent_cfg.max_iterations = 1
    agent_cfg.save_interval = 10**9

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    attach_curriculum_rollout_hints(env, spe=int(agent_cfg.num_steps_per_env), learning_iteration=50000)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[diag] load {resume_path}", flush=True)
    runner.load(resume_path, load_optimizer=False, load_critic=True)
    policy = runner.alg.policy
    if getattr(policy, "residual_corrector", None) is None:
        raise RuntimeError("residual_corrector is missing; adapter is not residual")

    inventory = _obs_inventory(env)
    print("[diag] policy obs terms:", flush=True)
    for term in inventory["groups"]["policy"]:
        print(f"  {term['name']:32s} dim={term['dim']} terrain={term['looks_like_terrain']}", flush=True)
    print(
        f"[diag] policy_has_terrain_term={inventory['policy_has_terrain_term']} "
        f"scene_terrainish={inventory['scene_keys_looking_like_terrain']}",
        flush=True,
    )
    print(
        f"[diag] g_phi inputs = KP tokens + proprio history + frozen mu_hat "
        f"(obstacle_n={getattr(policy.residual_corrector, 'obstacle_n', 0)})",
        flush=True,
    )

    group_ids, names, counts = _env_group_ids(env)
    print(f"[diag] sub-terrains={names} counts={counts}", flush=True)
    for g in GROUP_ORDER:
        print(f"[diag] group {g}: {int(group_ids[g].sum().item())} envs", flush=True)

    payload = {
        "ckpt": resume_path,
        "task": args_cli.task,
        "num_envs": int(args_cli.num_envs),
        "obs": inventory,
        "g_phi": {
            "sees_forward_terrain": False,
            "reason": (
                "Actor obs are sparse KP lookahead + proprio/IMU/last-action. "
                "g_φ consumes the same split (kp, mask, proprio, mu_hat). "
                "No height scan / depth / elevation map in policy group or scene sensors."
            ),
            "obstacle_n": int(getattr(policy.residual_corrector, "obstacle_n", 0) or 0),
        },
        "terrain_names": names,
        "terrain_env_counts": counts,
        "n_gphi_params": int(sum(p.numel() for p in policy.residual_corrector.parameters())),
    }

    result: dict = {}

    def _diag_update(*_a, **_k):
        storage = runner.alg.storage
        t_max, n_env = storage.observations.shape[:2]
        env_mask = torch.arange(n_env, device=storage.observations.device)
        env_mask = env_mask.unsqueeze(0).expand(t_max, n_env).reshape(-1)
        vecs = {}
        ns = {}
        norms = {}
        for group in GROUP_ORDER:
            env_ok = group_ids[group].to(device=env_mask.device)
            mask = env_ok[env_mask]
            n = int(mask.sum().item())
            ns[group] = n
            if n < int(args_cli.min_samples):
                print(f"[diag] skip {group}: n={n} < min_samples", flush=True)
                continue
            vec, n_used = _actor_grad(runner.alg, mask)
            vecs[group] = vec
            ns[group] = n_used
            norms[group] = float(vec.norm().item())
            print(f"[diag] {group}: n={n_used} ||g||={norms[group]:.4f}", flush=True)
        labels = [g for g in GROUP_ORDER if g in vecs]
        matrix = {a: {b: None for b in GROUP_ORDER} for a in GROUP_ORDER}
        for a in labels:
            for b in labels:
                matrix[a][b] = round(_cosine(vecs[a], vecs[b]), 4)
        result["n_samples"] = ns
        result["grad_norm"] = norms
        result["cosine"] = matrix
        print("[diag] cosine matrix", json.dumps(matrix, indent=2), flush=True)
        return {"surrogate": 0.0, "value_function": 0.0, "entropy": 0.0}

    runner.alg.update = _diag_update
    runner.disable_logs = True
    runner.save_interval = 10**9
    print("[diag] collecting one PPO rollout", flush=True)
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
    payload.update(result)

    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[diag] wrote {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
