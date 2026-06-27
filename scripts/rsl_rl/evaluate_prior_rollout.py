# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Systematic prior-only rollouts (PULSE): parallel envs, fall-only MDP terminations + horizon.

Runs ``eval_iters`` outer iterations (one motion clip per iteration when given a directory),
and each iteration rolls out for up to ``rollout_length`` env steps (same role as
``play.py`` / ``--video_length``). Logs prior metrics to Weights & Biases (default project:
``evaluate_prior_rollout``) per iteration.

Rollout logic lives in ``MotionOnPolicyRunner.collect_prior_rollout_eval`` (same ``env.step`` /
observation normalization path as ``OnPolicyRunner.learn`` in eval mode). Fall statistics use
``termination_manager.get_term("fall")`` after each step, not a manual height check on post-reset
poses (which wrongly reported zero falls when fall termination resets the robot).

**Memory:** streaming scalars + small ``[N, A]`` CPU sums only. Like training, the simulation is
created **once**; each evaluation clip selects a motion index in ``MultiMotionCommand`` (no
per-clip ``gym.make`` / teardown).

**Progress:** each clip runs ``rollout_length`` steps × ``num_envs`` env-steps of sim work — not
instant at high ``num_envs``. Set ``--log_interval`` low to see step progress.

**Dataset loading:** multiple distinct clips require every sampled clip to be present in
``MultiMotionLoader`` (same as training: use ``motion_dataset_load_cap=None`` for the full
dataset under ``motion``, or a directory that contains only the clips you need).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# Training scripts often set this; keep wandb quiet but we still print rollout progress to stdout.
os.environ.setdefault("WANDB_SILENT", "true")

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

if WORLD_SIZE > 1:
    base = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"isaaclab_kit_{os.getuid()}")
    rank_dir = os.path.join(base, f"rank{RANK}")
    os.environ.setdefault("OMNI_USER_DIR", rank_dir)
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(rank_dir, "cache"))
    os.environ.setdefault("XDG_DATA_HOME", os.path.join(rank_dir, "data"))
    os.environ.setdefault("XDG_CONFIG_HOME", os.path.join(rank_dir, "config"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate prior rollout with fall-only terminations + wandb.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default=None, help="Gym task id (PULSE distillation).")
parser.add_argument("--motion", type=str, required=True, help="Motion .npz file or directory of clips.")
parser.add_argument(
    "--eval_iters",
    type=int,
    default=120,
    help="Number of evaluation iterations (motion clips). From a directory, samples or cycles clips.",
)
parser.add_argument(
    "--rollout_length",
    type=int,
    default=500,
    help="Max environment steps per iteration (same idea as play.py --video_length).",
)
parser.add_argument(
    "--prior_fall_min_height",
    type=float,
    default=0.35,
    help="World z (m) below which monitored bodies count as fallen.",
)
parser.add_argument(
    "--prior_fall_body_names",
    type=str,
    default="torso_link",
    help="Comma-separated robot body names for fall termination / metrics.",
)
parser.add_argument(
    "--disable_motion_group_sampling",
    action="store_true",
    default=False,
    help="Uniform motion assignment (disable group ratio sampling).",
)
parser.add_argument("--start_frame", type=int, default=10, help="Motion start frame (0-based).")
parser.add_argument(
    "--motion_sample_seed",
    type=int,
    default=None,
    help="When a directory has more clips than --eval_iters, seed for random subsampling.",
)
parser.add_argument(
    "--motion_dataset_load_cap",
    type=int,
    default=1,
    help="Cap loaded motions per env (default 1: all envs share the clip for that iteration).",
)
parser.add_argument(
    "--prior_metrics_json",
    type=str,
    default="",
    help="Optional path to write prior_rollout_metrics-style JSON (rank0 only).",
)
parser.add_argument(
    "--no_wandb",
    action="store_true",
    help="Disable wandb.init even if --logger wandb is set.",
)
parser.add_argument(
    "--log_interval",
    type=int,
    default=24,
    help="Print rollout progress every N env steps (0 = only per-clip summary). Default matches typical num_steps_per_env.",
)
parser.add_argument(
    "--prior_rollout_fixed_latent_std",
    type=float,
    default=None,
    help="If set, sample z with this fixed per-latent-dim std instead of the prior MLP's σ (μ unchanged).",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if getattr(args_cli, "log_project_name", None) is None:
    args_cli.log_project_name = "evaluate_prior_rollout"
if getattr(args_cli, "logger", None) is None:
    args_cli.logger = "wandb"

if WORLD_SIZE > 1:
    args_cli.distributed = True
if getattr(args_cli, "distributed", False):
    args_cli.device = f"cuda:{LOCAL_RANK}"
if getattr(args_cli, "distributed", False) and RANK != 0:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import random  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import whole_body_tracking.tasks  # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg  # noqa: E402
from rsl_rl.modules import LatentBottleneckPULSE  # noqa: E402
from whole_body_tracking.tasks.tracking import mdp  # noqa: E402
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner  # noqa: E402


def resolve_motion_paths(motion: str, eval_iters: int, seed: int | None) -> list[str]:
    """Build a list of length ``eval_iters`` motion roots (file, subfolder, or dataset dir).

    Mirrors ``run/rollout_prior_metrics.sh``: if ``motion`` is a directory with subfolders,
    each subfolder is a clip root; otherwise collect ``*.npz`` under the directory; a single
    file is repeated for ``eval_iters`` rollouts (e.g. to estimate variance).
    """
    p = Path(motion).expanduser().resolve()
    if p.is_file():
        return [str(p)] * int(eval_iters)
    if not p.is_dir():
        raise FileNotFoundError(f"Motion path not found: {motion}")
    subdirs = sorted([x for x in p.iterdir() if x.is_dir()])
    if subdirs:
        pool = [str(x) for x in subdirs]
    else:
        pool = sorted(str(x) for x in p.rglob("*.npz") if x.is_file())
        if not pool:
            pool = [str(p)]
    n = int(eval_iters)
    if len(pool) >= n:
        if seed is not None:
            rng = random.Random(int(seed))
            return rng.sample(pool, n)
        return pool[:n]
    return [pool[i % len(pool)] for i in range(n)]


def motion_dataset_root_for_loader(motion_arg: str) -> str:
    """``MultiMotionLoader`` expects a directory; if given a single ``.npz`` file, use its parent."""
    p = Path(motion_arg).expanduser().resolve()
    if p.is_file():
        return str(p.parent)
    return str(p)


def find_motion_index_in_loader(loader, requested: str) -> int:
    """Map a ``resolve_motion_paths`` entry to an index in ``MultiMotionLoader.motion_paths``."""
    req = Path(requested).expanduser().resolve()
    if not req.exists():
        raise FileNotFoundError(f"Motion path not found: {requested}")
    for i, lp in enumerate(loader.motion_paths):
        if Path(lp).resolve() == req:
            return i
    if req.is_dir():
        matches: list[tuple[int, Path]] = []
        for i, lp in enumerate(loader.motion_paths):
            lp_path = Path(lp).resolve()
            if not lp_path.is_file():
                continue
            try:
                lp_path.relative_to(req)
            except ValueError:
                continue
            matches.append((i, lp_path))
        if len(matches) == 1:
            return matches[0][0]
        if len(matches) > 1:
            matches.sort(key=lambda t: str(t[1]))
            return matches[0][0]
    raise RuntimeError(
        f"prior eval: could not map {requested!r} to a loaded motion. "
        "Ensure motion_dataset_load_cap loads this clip (multi-clip eval sets cap=None), "
        "or that paths match the dataset root passed to MultiMotionLoader."
    )


@configclass
class PriorRolloutFallOnlyTerminationsCfg:
    """Only fall-to-ground + time-out (horizon). No tracking / motion-end terminations."""

    fall = TerminationTermCfg(
        func=mdp.fall_to_ground,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]), "min_height": 0.35},
    )
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


def _logger_is_wandb(agent_cfg, train_cfg: dict) -> bool:
    """``agent_cfg.to_dict()`` may omit or stringify ``logger`` differently than ``agent_cfg``."""
    for lg in (getattr(agent_cfg, "logger", None), train_cfg.get("logger")):
        if lg is None:
            continue
        if str(lg).lower() == "wandb":
            return True
    return False


def _wandb_project_name(agent_cfg, args_cli) -> str:
    return (
        getattr(agent_cfg, "wandb_project", None)
        or getattr(args_cli, "log_project_name", None)
        or "evaluate_prior_rollout"
    )


def _wandb_scalar_dict(metrics: dict) -> dict:
    """W&B charts skip or flatten non-finite scalars; keep JSON-serializable floats."""
    out: dict = {}
    for k, v in metrics.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (float, int)):
            fv = float(v)
            if not math.isfinite(fv):
                continue
            out[k] = fv
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
    return out


def _merge_write_prior_json(out_json: str, payload: dict) -> None:
    motion_key = os.path.abspath(str(payload.get("motion", "unknown")))
    existing: dict = {"motions": {}}
    if os.path.isfile(out_json):
        try:
            with open(out_json, encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {"motions": {}}
    if "motions" not in existing:
        existing["motions"] = {}
    existing["motions"][motion_key] = payload
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    base_seed = int(agent_cfg.seed)
    rank = int(os.environ.get("RANK", "0"))
    seed_stride = int(os.environ.get("SEED_STRIDE", "1000"))
    env_seed = base_seed + rank * seed_stride
    env_cfg.seed = env_seed
    random.seed(env_seed)
    np.random.seed(env_seed)
    torch.manual_seed(env_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(env_seed)

    eval_iters = max(1, int(args_cli.eval_iters))
    rollout_length = max(1, int(args_cli.rollout_length))
    dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    env_cfg.episode_length_s = float(rollout_length) * dt

    motion_paths = resolve_motion_paths(
        args_cli.motion,
        eval_iters,
        getattr(args_cli, "motion_sample_seed", None),
    )

    fall_names = [n.strip() for n in str(args_cli.prior_fall_body_names).split(",") if n.strip()]
    term = PriorRolloutFallOnlyTerminationsCfg()
    term.fall.params["asset_cfg"] = SceneEntityCfg("robot", body_names=fall_names)
    term.fall.params["min_height"] = float(args_cli.prior_fall_min_height)
    env_cfg.terminations = term

    motion_root = motion_dataset_root_for_loader(args_cli.motion)
    env_cfg.commands.motion.motion = motion_root
    env_cfg.commands.motion.start_from_beginning = True
    env_cfg.commands.motion.start_frame = int(args_cli.start_frame)
    if args_cli.disable_motion_group_sampling and hasattr(env_cfg.commands.motion, "motion_group_sampling_ratios"):
        env_cfg.commands.motion.motion_group_sampling_ratios = None
    unique_motion_paths = list(dict.fromkeys(motion_paths))
    if len(unique_motion_paths) > 1:
        # Same as training: every evaluated clip must be resident in MultiMotionLoader.
        env_cfg.commands.motion.motion_dataset_load_cap = None
    else:
        cap = getattr(args_cli, "motion_dataset_load_cap", None)
        if cap is not None and int(cap) > 0:
            env_cfg.commands.motion.motion_dataset_load_cap = int(cap)
    env_cfg.commands.motion.motion_dataset_shard_across_gpus = False
    env_cfg.commands.motion.motion_dataset_log_wandb_summary = False

    if hasattr(env_cfg, "events"):
        env_cfg.events = None
    if hasattr(env_cfg, "observations"):
        for group_name in ("policy", "teacher", "critic", "ref_vel_estimator"):
            if hasattr(env_cfg.observations, group_name):
                g = getattr(env_cfg.observations, group_name)
                if hasattr(g, "enable_corruption"):
                    g.enable_corruption = False

    if hasattr(env_cfg, "commands"):
        motion_cfg = getattr(env_cfg.commands, "motion", None)
        if motion_cfg is not None:
            z = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
            if hasattr(motion_cfg, "pose_range"):
                motion_cfg.pose_range = dict(z)
            if hasattr(motion_cfg, "velocity_range"):
                motion_cfg.velocity_range = dict(z)
            if hasattr(motion_cfg, "joint_position_range"):
                motion_cfg.joint_position_range = (0.0, 0.0)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = os.path.join(
        log_root_path,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        f"prior_eval_{agent_cfg.run_name or 'no_run_name'}",
    )
    os.makedirs(log_dir, exist_ok=True)

    if int(os.environ.get("RANK", "0")) == 0:
        dump_yaml(os.path.join(log_dir, "params_env.yaml"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params_agent.pkl"), agent_cfg)

    train_cfg = agent_cfg.to_dict()
    if _logger_is_wandb(agent_cfg, train_cfg) and getattr(agent_cfg, "wandb_project", None):
        train_cfg["wandb_project"] = agent_cfg.wandb_project
    elif _logger_is_wandb(agent_cfg, train_cfg):
        wp = _wandb_project_name(agent_cfg, args_cli)
        train_cfg["wandb_project"] = wp

    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    use_wandb = (
        _logger_is_wandb(agent_cfg, train_cfg)
        and int(os.environ.get("RANK", "0")) == 0
        and not getattr(args_cli, "no_wandb", False)
        and os.environ.get("WANDB_DISABLED", "").lower() not in ("1", "true")
        and os.environ.get("WANDB_MODE", "").lower() != "disabled"
    )

    if use_wandb:
        try:
            import wandb
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Install wandb for logging, or pass --no_wandb.") from e

        project = _wandb_project_name(agent_cfg, args_cli)
        wname = agent_cfg.run_name or os.path.basename(log_dir)
        wandb.init(
            project=project,
            entity=os.environ.get("WANDB_USERNAME") or None,
            name=wname,
            config={
                "task": args_cli.task,
                "motion_root": motion_root,
                "motion_arg": os.path.abspath(args_cli.motion),
                "checkpoint": resume_path,
                "eval_iters": eval_iters,
                "rollout_length": rollout_length,
                "num_envs": int(env_cfg.scene.num_envs),
                "motion_sample_seed": getattr(args_cli, "motion_sample_seed", None),
                "prior_fall_min_height": float(args_cli.prior_fall_min_height),
                "prior_fall_body_names": fall_names,
            },
        )
        if RANK == 0:
            wm = os.environ.get("WANDB_MODE", "").lower()
            if wm == "offline":
                print(
                    "[Prior eval] WANDB_MODE=offline — metrics are written under ./wandb/; "
                    "run `wandb sync` or open the run folder to upload to the web UI.",
                    flush=True,
                )
            if wandb.run is not None:
                url = getattr(wandb.run, "url", None) or ""
                if url:
                    print(f"[Prior eval] W&B run URL: {url}", flush=True)
                print(
                    "[Prior eval] Charts: workspace → this run → search metrics starting with "
                    "`prior_eval/` (or add panels from the metric sidebar).",
                    flush=True,
                )

    fall_rates: list[float] = []

    log_interval = max(0, int(getattr(args_cli, "log_interval", 24)))

    if RANK == 0:
        print(f"[Prior eval] Creating env once (motion_root={motion_root})...", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    runner = MotionOnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False, load_critic=False)
    if not isinstance(runner.alg.policy, LatentBottleneckPULSE):
        raise RuntimeError("evaluate_prior_rollout requires LatentBottleneckPULSE (PULSE checkpoint).")

    num_envs = int(env.num_envs)

    for it, motion_path in enumerate(motion_paths):
        motion_cmd = env.unwrapped.command_manager.get_term("motion")
        motion_idx = find_motion_index_in_loader(motion_cmd.motion_dir_loader, motion_path)

        metrics, first_fall_step = runner.collect_prior_rollout_eval(
            rollout_length,
            log_interval=log_interval,
            clip_idx=it,
            clip_total=eval_iters,
            motion_path=motion_path,
            fall_term_name="fall",
            prior_eval_motion_idx=motion_idx,
            fixed_prior_latent_std=getattr(args_cli, "prior_rollout_fixed_latent_std", None),
        )
        metrics["prior_eval/iter"] = float(it)
        fall_rates.append(float(metrics["prior_eval/fall_rate"]))

        if use_wandb:
            import wandb

            wandb.log(_wandb_scalar_dict(metrics), step=it, commit=True)

        if int(os.environ.get("RANK", "0")) == 0:
            tag = "[Prior eval]"
            corr = metrics["prior_eval/sensitivity_corr_delta_p_delta_a"]
            print(
                f"{tag} iter {it + 1}/{eval_iters} motion={motion_path} "
                f"envs={num_envs} steps={int(metrics['prior_eval/steps_recorded'])} "
                f"fall_rate={metrics['prior_eval/fall_rate']:.3f} "
                f"entropy_mean={metrics['prior_eval/prior_latent_entropy_mean']:.4f} "
                f"prior_sigma_mean={metrics['prior_eval/prior_predicted_sigma_mean']:.5f} "
                f"prior_sigma_std={metrics['prior_eval/prior_predicted_sigma_std']:.5f}"
            )

            payload = {
                "task": args_cli.task,
                "motion": os.path.abspath(motion_path),
                "checkpoint": resume_path,
                "prior_rollout_fixed_latent_std": getattr(args_cli, "prior_rollout_fixed_latent_std", None),
                "eval_iter": it,
                "eval_iters": eval_iters,
                "rollout_length": rollout_length,
                "num_envs": num_envs,
                "terminations": "fall_to_ground + time_out",
                "prior_fall": {
                    "min_height_m": float(args_cli.prior_fall_min_height),
                    "body_names": fall_names,
                    "fall_rate": metrics["prior_eval/fall_rate"],
                    "per_env_first_fall_step": first_fall_step.detach().cpu().tolist(),
                },
                "prior_latent_entropy": {
                    "mean": metrics["prior_eval/prior_latent_entropy_mean"],
                    "std": metrics["prior_eval/prior_latent_entropy_std"],
                },
                "prior_predicted_sigma": {
                    "mean": metrics["prior_eval/prior_predicted_sigma_mean"],
                    "std": metrics["prior_eval/prior_predicted_sigma_std"],
                    "note": "σ = exp(log σ) from prior MLP; mean/std over all per-dim σ pooled across envs/steps.",
                },
                "actions": {
                    "temporal_std_mean": metrics["prior_eval/action_temporal_std_mean"],
                    "mean_abs_per_dim": metrics["prior_eval/action_mean_abs"],
                    "delta_l2_mean": metrics["prior_eval/delta_action_l2_mean"],
                    "delta_l2_std": metrics["prior_eval/delta_action_l2_std"],
                },
                "proprio": {"delta_l2_mean": metrics["prior_eval/delta_proprio_l2_mean"]},
                "sensitivity": {
                    "delta_action_over_delta_proprio": metrics["prior_eval/sensitivity_delta_a_over_delta_p"],
                    "corr_delta_proprio_delta_action": (None if not math.isfinite(corr) else corr),
                },
                "wandb_metrics": {k.replace("prior_eval/", ""): v for k, v in metrics.items()},
            }
            out_json = str(args_cli.prior_metrics_json).strip()
            if out_json:
                _merge_write_prior_json(out_json, payload)
            else:
                default_json = os.path.join(log_dir, "prior_eval_metrics.json")
                _merge_write_prior_json(default_json, payload)

    if use_wandb:
        import wandb

        if fall_rates:
            wandb.log(
                {
                    "prior_eval/aggregate/fall_rate_mean": float(sum(fall_rates) / len(fall_rates)),
                    "prior_eval/aggregate/eval_iters": float(eval_iters),
                },
                step=eval_iters,
                commit=True,
            )
        wandb.finish()

    if int(os.environ.get("RANK", "0")) == 0 and motion_paths:
        tag = "[Prior eval]"
        mj = str(args_cli.prior_metrics_json).strip()
        if mj:
            print(f"{tag} Appended per-clip metrics to {mj}")
        else:
            print(f"{tag} Wrote per-clip metrics under {log_dir}/prior_eval_metrics.json")

    if env is not None:
        env.close()


if __name__ == "__main__":
    main()  # type: ignore[misc]
    simulation_app.close()
