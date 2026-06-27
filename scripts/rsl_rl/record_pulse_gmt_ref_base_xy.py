# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record world-frame anchor XY trajectories: reference motion, PULSE student, optional GMT teacher.

By default runs two rollouts (student then teacher). Pass ``--no_teacher`` to only roll
the student and plot reference + PULSE (red/black). Saves ``.npz`` + matplotlib figure
(blue=GMT when included, red=PULSE, black=reference).

Example::

    ./isaaclab.sh -p scripts/rsl_rl/record_pulse_gmt_ref_base_xy.py \\
      --task General-Tracking-Flat-G1-Wo-State-Estimation-v0-World-Coordinate-Reward \\
      --motion /path/to/motion_or_dir \\
      --load_run 2026-04-01_14-43-13_GMT_sonic_data_filtered_data_small \\
      --checkpoint model_17000.pt \\
      --headless \\
      --motion_idx 0 --num_steps 500 --frame_stride 5
"""

from __future__ import annotations

import argparse
import os
import sys

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

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # isort: skip  # noqa: E402

parser = argparse.ArgumentParser(
    description="Record reference / PULSE / GMT anchor XY trajectories (world frame)."
)
parser.add_argument("--task", type=str, default=None, help="Hydra task name.")
parser.add_argument("--motion", type=str, required=True, help="Motion .npz file or directory.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs (trajectory uses env 0).")
parser.add_argument(
    "--num_steps",
    type=int,
    default=None,
    help="Env steps per rollout (default: clip until motion end minus start frame).",
)
parser.add_argument(
    "--motion_idx",
    type=int,
    default=0,
    help="Clip index for MultiMotionCommand (ignored for single-file MotionCommand).",
)
parser.add_argument(
    "--start_frame",
    type=int,
    default=0,
    help="Start frame when forcing motion from beginning (matches play.py).",
)
parser.add_argument(
    "--disable_motion_group_sampling",
    action="store_true",
    default=False,
    help="Uniform motion sampling when using a motion directory.",
)
parser.add_argument(
    "--frame_stride",
    type=int,
    default=5,
    help="Plot every Nth recorded step (markers + connecting lines).",
)
parser.add_argument(
    "--out_npz",
    type=str,
    default=None,
    help="Output .npz path (default: next to checkpoint: base_xy_gmt_pulse_ref.npz).",
)
parser.add_argument(
    "--out_fig",
    type=str,
    default=None,
    help="Output figure path .png or .pdf (default: same basename as npz with .png).",
)
parser.add_argument(
    "--no_teacher",
    action="store_true",
    default=False,
    help="Skip GMT teacher rollout; plot and npz contain only reference + PULSE student.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

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
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import whole_body_tracking.tasks  # noqa: F401,E402
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand, MultiMotionCommand  # noqa: E402
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner  # noqa: E402

from rsl_rl.modules import LatentBottleneckPULSE  # noqa: E402


def _max_safe_steps(cmd: MotionCommand | MultiMotionCommand) -> int:
    start = int(cmd.time_steps[0].item())
    if isinstance(cmd, MultiMotionCommand):
        idx = int(cmd.env_motion_indices[0].item())
        total = int(cmd.motion_lengths[idx].item())
    else:
        total = int(cmd.motion.time_step_total)
    return max(0, total - 1 - start)


def _apply_eval_env_style_overrides(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    *,
    motion: str,
    start_frame: int,
    disable_motion_group_sampling: bool,
) -> None:
    env_cfg.commands.motion.motion = motion
    if disable_motion_group_sampling and hasattr(env_cfg.commands.motion, "motion_group_sampling_ratios"):
        env_cfg.commands.motion.motion_group_sampling_ratios = None

    if hasattr(env_cfg.commands.motion, "start_from_beginning"):
        env_cfg.commands.motion.start_from_beginning = True
    if hasattr(env_cfg.commands.motion, "start_frame"):
        env_cfg.commands.motion.start_frame = int(start_frame)

    if hasattr(env_cfg, "commands"):
        motion_cfg = getattr(env_cfg.commands, "motion", None)
        if motion_cfg is not None:
            zero_ranges = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
            if hasattr(motion_cfg, "pose_range"):
                motion_cfg.pose_range = dict(zero_ranges)
            if hasattr(motion_cfg, "velocity_range"):
                motion_cfg.velocity_range = dict(zero_ranges)
            if hasattr(motion_cfg, "joint_position_range"):
                motion_cfg.joint_position_range = (0.0, 0.0)

    if hasattr(env_cfg, "observations"):
        for group_name in ("policy", "teacher", "critic", "ref_vel_estimator"):
            if hasattr(env_cfg.observations, group_name):
                group_cfg = getattr(env_cfg.observations, group_name)
                if hasattr(group_cfg, "enable_corruption"):
                    group_cfg.enable_corruption = False

    if hasattr(env_cfg, "events"):
        env_cfg.events = None

    # Avoid auto-reset mid-rollout (would desync student vs teacher comparisons).
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations = None


def _normalize_obs_after_step(
    runner: MotionOnPolicyRunner,
    obs: torch.Tensor,
    obs_dict: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Match ``OnPolicyRunner.learn`` observation extraction and normalization."""
    device = runner.device
    if runner.policy_obs_type is not None and runner.policy_obs_type in obs_dict:
        obs = obs_dict[runner.policy_obs_type]
    privileged_obs = obs_dict.get(runner.privileged_obs_type, obs)
    teacher_obs = obs_dict.get(runner.teacher_obs_type)
    obs = obs.to(device)
    privileged_obs = privileged_obs.to(device)
    if teacher_obs is not None:
        teacher_obs = teacher_obs.to(device)
    else:
        teacher_obs = privileged_obs
    runner._assert_anybody_latent_proprio_alignment(obs, teacher_obs)

    ref_vel = obs_dict.get(runner.ref_vel_estimator_obs_type)
    if ref_vel is not None:
        ref_vel = ref_vel.to(device)

    obs = runner._normalize_student_obs(obs)
    privileged_obs = runner.privileged_obs_normalizer(privileged_obs)
    teacher_obs = runner.teacher_obs_normalizer(teacher_obs)

    return obs, teacher_obs, ref_vel


def _student_actions(
    runner: MotionOnPolicyRunner,
    student_obs_norm: torch.Tensor,
    ref_vel_estimator_obs: torch.Tensor | None,
) -> torch.Tensor:
    use_velocity_estimator = (
        hasattr(runner.alg, "ref_vel_estimator")
        and runner.alg.ref_vel_estimator is not None
        and hasattr(runner.alg, "use_estimate_ref_vel")
        and runner.alg.use_estimate_ref_vel
    )
    if use_velocity_estimator:
        if ref_vel_estimator_obs is None:
            raise RuntimeError("use_estimate_ref_vel=True but ref_vel_estimator observations missing.")
        est = runner.alg.ref_vel_estimator(ref_vel_estimator_obs) * 1.0
        aug = torch.cat([student_obs_norm, est], dim=-1)
        return runner.alg.policy.act_inference(aug)
    return runner.alg.policy.act_inference(student_obs_norm)


def _rollout_xy(
    runner: MotionOnPolicyRunner,
    *,
    mode: str,
    num_steps: int,
    motion_idx: int | None,
    env_device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    assert mode in ("student", "teacher")
    policy = runner.alg.policy
    assert isinstance(policy, LatentBottleneckPULSE)

    ref_list: list[np.ndarray] = []
    rob_list: list[np.ndarray] = []

    with torch.inference_mode():
        runner.env.reset()
        if motion_idx is not None:
            runner._apply_prior_eval_motion_index(int(motion_idx))

        obs, extras = runner.env.get_observations()
        obs_dict = extras.get("observations", {})

        for _ in range(num_steps):
            student_obs_norm, teacher_obs_norm, ref_vel_obs = _normalize_obs_after_step(
                runner, obs, obs_dict
            )
            if mode == "student":
                actions = _student_actions(runner, student_obs_norm, ref_vel_obs)
            else:
                actions = policy.evaluate(teacher_obs_norm)

            obs, _rewards, dones, infos = runner.env.step(actions.to(env_device))
            obs_dict = infos.get("observations", {})

            cmd = runner.env.unwrapped.command_manager.get_term("motion")
            ref_xy = cmd.anchor_pos_w[0, :2].detach().float().cpu().numpy()
            rob_xy = cmd.robot_anchor_pos_w[0, :2].detach().float().cpu().numpy()
            ref_list.append(ref_xy)
            rob_list.append(rob_xy)

            if hasattr(policy, "reset"):
                policy.reset(dones)

    return np.stack(ref_list, axis=0), np.stack(rob_list, axis=0)


def _save_figure(
    out_path: str,
    ref_xy: np.ndarray,
    student_xy: np.ndarray,
    teacher_xy: np.ndarray | None,
    frame_stride: int,
    *,
    plot_teacher: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stride = max(1, int(frame_stride))
    idx = np.arange(0, ref_xy.shape[0], stride)
    ref_p = ref_xy[idx]
    stu_p = student_xy[idx]

    fig, ax = plt.subplots(figsize=(7, 7))
    if plot_teacher and teacher_xy is not None:
        tea_p = teacher_xy[idx]
        ax.plot(
            tea_p[:, 0],
            tea_p[:, 1],
            color="blue",
            linestyle="-",
            marker="o",
            markersize=4,
            label="GMT teacher",
        )
    ax.plot(
        stu_p[:, 0],
        stu_p[:, 1],
        color="red",
        linestyle="-",
        marker="o",
        markersize=4,
        label="PULSE student",
    )
    ax.plot(
        ref_p[:, 0],
        ref_p[:, 1],
        color="black",
        linestyle="-",
        marker="o",
        markersize=4,
        label="Reference motion",
    )

    # Square view in data space: same meter span on x and y (centered on trajectories).
    point_blocks: list[np.ndarray] = [ref_p, stu_p]
    if plot_teacher and teacher_xy is not None:
        point_blocks.append(tea_p)
    xy = np.vstack(point_blocks)
    xmin, ymin = float(xy[:, 0].min()), float(xy[:, 1].min())
    xmax, ymax = float(xy[:, 0].max()), float(xy[:, 1].max())
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    span_data = max(xmax - xmin, ymax - ymin, 1e-6)
    pad = 0.05 * span_data
    half = 0.5 * span_data + pad
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    ax.set_aspect("equal", adjustable="box")
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    env_cfg.scene.num_envs = max(1, int(args_cli.num_envs))

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

    _apply_eval_env_style_overrides(
        env_cfg,
        motion=args_cli.motion,
        start_frame=int(args_cli.start_frame),
        disable_motion_group_sampling=bool(args_cli.disable_motion_group_sampling),
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    out_npz = args_cli.out_npz
    if out_npz is None:
        out_npz = os.path.join(os.path.dirname(resume_path), "base_xy_gmt_pulse_ref.npz")
    out_fig = args_cli.out_fig
    if out_fig is None:
        out_fig = os.path.splitext(out_npz)[0] + ".png"

    train_cfg = agent_cfg.to_dict()
    train_cfg["logger"] = "tensorboard"
    log_dir = os.path.join(os.path.dirname(resume_path), "base_xy_record")
    os.makedirs(log_dir, exist_ok=True)

    runner = MotionOnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False, load_critic=False)
    runner.eval_mode()

    policy = runner.alg.policy
    if not isinstance(policy, LatentBottleneckPULSE):
        raise RuntimeError(
            f"This script only supports LatentBottleneckPULSE; got {type(policy).__name__}."
        )
    if not bool(args_cli.no_teacher) and not getattr(policy, "loaded_teacher", False):
        raise RuntimeError("Checkpoint has no loaded GMT teacher (policy.loaded_teacher is False).")
    policy.deterministic_latent = True

    motion_term = runner.env.unwrapped.command_manager.get_term("motion")
    motion_idx_arg: int | None = None
    if isinstance(motion_term, MultiMotionCommand):
        motion_idx_arg = int(args_cli.motion_idx)
    elif isinstance(motion_term, MotionCommand):
        motion_idx_arg = None
    else:
        raise RuntimeError(f"Unsupported motion command type: {type(motion_term).__name__}")

    with torch.inference_mode():
        runner.env.reset()
        if motion_idx_arg is not None:
            runner._apply_prior_eval_motion_index(motion_idx_arg)

    cmd = runner.env.unwrapped.command_manager.get_term("motion")
    max_safe = _max_safe_steps(cmd)
    num_steps = args_cli.num_steps
    if num_steps is None:
        num_steps = max_safe
    else:
        num_steps = int(num_steps)
    if num_steps > max_safe:
        print(
            f"[record_base_xy] Capping num_steps {num_steps} -> {max_safe} "
            f"(motion length / start frame)."
        )
        num_steps = max_safe
    if num_steps <= 0:
        raise RuntimeError("No steps to record (num_steps<=0). Check motion length and start_frame.")

    env_device = runner.env.device

    ref_xy_s, student_xy = _rollout_xy(
        runner,
        mode="student",
        num_steps=num_steps,
        motion_idx=motion_idx_arg,
        env_device=env_device,
    )
    ref_xy = ref_xy_s
    teacher_xy: np.ndarray | None = None
    if not bool(args_cli.no_teacher):
        ref_xy_t, teacher_xy = _rollout_xy(
            runner,
            mode="teacher",
            num_steps=num_steps,
            motion_idx=motion_idx_arg,
            env_device=env_device,
        )
        if not np.allclose(ref_xy_s, ref_xy_t, rtol=1e-5, atol=1e-4):
            print(
                "[record_base_xy] WARNING: reference XY differs between student and teacher rollouts; "
                "using the student rollout reference in the npz."
            )

    out_npz_dir = os.path.dirname(os.path.abspath(out_npz))
    if out_npz_dir:
        os.makedirs(out_npz_dir, exist_ok=True)
    save_kw: dict = {
        "ref_xy": ref_xy,
        "student_xy": student_xy,
        "include_teacher": np.asarray(not bool(args_cli.no_teacher)),
        "frame_stride": int(args_cli.frame_stride),
        "motion_idx": int(motion_idx_arg if motion_idx_arg is not None else -1),
        "start_frame": int(args_cli.start_frame),
        "num_steps": int(num_steps),
        "checkpoint": np.asarray(resume_path, dtype=str),
    }
    if teacher_xy is not None:
        save_kw["teacher_xy"] = teacher_xy
    np.savez_compressed(out_npz, **save_kw)
    fig_dir = os.path.dirname(os.path.abspath(out_fig))
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
    _save_figure(
        out_fig,
        ref_xy,
        student_xy,
        teacher_xy,
        int(args_cli.frame_stride),
        plot_teacher=not bool(args_cli.no_teacher),
    )
    print(f"[record_base_xy] Wrote {out_npz}")
    print(f"[record_base_xy] Wrote {out_fig}")

    env.close()


if __name__ == "__main__":
    main()  # type: ignore[misc]
    simulation_app.close()