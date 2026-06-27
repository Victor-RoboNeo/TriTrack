# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained RSL-RL policy checkpoint on a motion directory without policy updates.

Uses the same environment and runner stack as training, but calls
`OnPolicyRunner.learn(..., eval_mode=True)` so the runner rolls out and logs
metrics while skipping `compute_returns()` and `alg.update()`.

Works for standard RL (GMT / actor-critic), distillation, PULSE, etc.,
as long as the Hydra ``--task`` matches the checkpoint's training configuration.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime

# Replicate distributed init pattern from scripts/rsl_rl/train.py
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


parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint without policy updates.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion", type=str, required=True, help="Path to the motion file or motion directory.")
parser.add_argument(
    "--eval_iters",
    type=int,
    default=5,
    help="How many runner iterations to execute (each iteration runs num_steps_per_env steps per env).",
)
parser.add_argument(
    "--pulse_deterministic_latent",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="For LatentBottleneckPULSE: use encoder mean μ (no z sampling) during rollout. "
    "Recommended for test/eval.",
)
# VR-Tracking (same flags as train.py so eval matches training env / normalizer)
parser.add_argument(
    "--prior_checkpoint",
    type=str,
    default=None,
    help="Path to a PULSE-style .pt containing prior.* and student_core.decoder.* "
    "(required for VR-Tracking-Flat-G1-v0; for VR-Tracking-Joint-Flat-G1-v0 policy init only, not obs normalizer).",
)
parser.add_argument(
    "--latent_rl_checkpoint",
    type=str,
    default=None,
    help="[VR-Tracking-Joint-Flat-G1-v0] If set, split obs normalizer from this latent VR PPO .pt; "
    "otherwise from the eval resume checkpoint.",
)
parser.add_argument("--vr_residual_scale", type=float, default=None, help="Override VR residual_scale.")
parser.add_argument("--vr_latent_dim", type=int, default=None, help="Override VR latent_dim.")
parser.add_argument(
    "--vr_proprio_history_length",
    type=int,
    default=None,
    help="Override env actions.joint_pos.proprio_history_length.",
)
parser.add_argument(
    "--vr_mask_mode_probs",
    type=str,
    default=None,
    help="Comma-separated mask-mode sampling probabilities (renormalized if needed).",
)
parser.add_argument(
    "--vr_compact_goal_obs",
    type=str,
    default=None,
    help="true/false: compact vs full keypoint goal observation layout.",
)
parser.add_argument(
    "--vr_use_full_goal_obs_with_distill_pretrain",
    type=str,
    default="true",
    help="When true and distillation warmstart flags are set, force full goal obs unless "
    "--vr_compact_goal_obs is set.",
)
parser.add_argument(
    "--warmstart_from_masked_partial_kp_tracker",
    action="store_true",
    default=False,
    help="Must match training if used with --warmstart_lock_full_normalizer (obs normalizer path).",
)
parser.add_argument("--warmstart_checkpoint", type=str, default=None, help="Warmstart / normalizer-lock checkpoint.")

# Co-train modality + mask overrides for KP-tracker testing (see run/test/eval_cotrain/).
parser.add_argument(
    "--modality",
    type=str,
    default=None,
    choices=("kp", "jc"),
    help="For MUSE co-train policies: force the action source to one modality. "
    "Sets ``policy.pilot_kp_fraction`` to 1.0 (kp) or 0.0 (jc) after checkpoint load.",
)
parser.add_argument(
    "--p_mask",
    type=float,
    default=None,
    help="For ``jc`` modality: per-step probability that the JC goal is masked (Bernoulli). "
    "0.0 = always-visible goal (recommended for clean tracking eval).",
)
parser.add_argument(
    "--mask_modes",
    type=str,
    default=None,
    help="Name of a single visibility mode to pin every env to "
    "(e.g. ``pelvis_only``, ``end_effector``, ``left_end_effector``). Overrides "
    "``mask_mode_spec``/``mask_mode_probs`` on the motion command. With ``--modality=kp`` "
    "(cotrain) or standalone (single-modality KP tasks like MUSE-Kp-Distill, no --modality).",
)
parser.add_argument(
    "--warmstart_lock_full_normalizer",
    action="store_true",
    default=False,
    help="Load full obs normalizer from --warmstart_checkpoint (match training).",
)

# Reuse RSL-RL CLI args so we get --logger, --seed, etc.
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if getattr(args_cli, "warmstart_from_masked_partial_kp_tracker", False):
    ws = getattr(args_cli, "warmstart_checkpoint", None)
    if ws in (None, ""):
        raise ValueError(
            "--warmstart_from_masked_partial_kp_tracker requires --warmstart_checkpoint=/path/to.pt"
        )

# Match train.py distributed device selection
if WORLD_SIZE > 1:
    args_cli.distributed = True
if getattr(args_cli, "distributed", False):
    args_cli.device = f"cuda:{LOCAL_RANK}"
if getattr(args_cli, "distributed", False) and RANK != 0:
    # Avoid multi-rank wandb/tensorboard noise if any logger tries to init.
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim app
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
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import whole_body_tracking.tasks  # noqa: F401,E402
from whole_body_tracking.tasks.tracking.mdp.curriculums import (  # noqa: E402
    attach_curriculum_rollout_hints,
)
from whole_body_tracking.utils.my_on_policy_runner import (  # noqa: E402
    MotionOnPolicyRunner as OnPolicyRunner,
)

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg  # noqa: E402
from rsl_rl.modules import LatentBottleneckPULSE  # noqa: E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    # Override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # Match train.py: ensure sim/device are set for each distributed rank
    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # Basic seeding (for deterministic evaluation-ish behavior)
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

    env_cfg.commands.motion.motion = args_cli.motion

    # Co-train modality / mask override: validate flag combinations and apply env_cfg overrides
    # *before* ``gym.make`` so the command term is built with the single-mode spec.
    cotrain_modality = getattr(args_cli, "modality", None)
    cotrain_mask_modes = getattr(args_cli, "mask_modes", None)
    cotrain_p_mask = getattr(args_cli, "p_mask", None)
    # Standalone single-mode eval: ``--mask_modes`` WITHOUT ``--modality``. Single-modality
    # KP tasks (e.g. MUSE-Kp-Distill) have no ``pilot_kp_fraction``, so the cotrain
    # ``--modality=kp`` path would raise — mirror play.py's ``elif mask_modes`` branch:
    # pin the env to one mask mode + disable the curriculum, skip the policy-side pilot patch.
    single_mode_eval = cotrain_modality is None and cotrain_mask_modes is not None

    def _apply_single_mode_spec(mode_name: str) -> None:
        from whole_body_tracking.tasks.tracking.config.g1.mask_modes import eval_single_mode_spec

        spec, probs = eval_single_mode_spec(mode_name)
        if not hasattr(env_cfg.commands.motion, "mask_mode_spec"):
            raise RuntimeError(
                "--mask_modes requires a partial-masked command term "
                "(env_cfg.commands.motion must have mask_mode_spec)."
            )
        env_cfg.commands.motion.mask_mode_spec = spec
        env_cfg.commands.motion.mask_mode_probs = probs
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[evaluate_policy] KP mask pinned: mask_modes={mode_name!r} → single-mode spec {spec}.")

    def _disable_eval_curricula() -> None:
        # Curricula require runner-side anchors not set during ``eval_mode=True`` rollouts,
        # and would fight the manual mask/p_mask overrides.
        if hasattr(env_cfg, "curriculum"):
            for _term_name in ("keypoint_mask_mode", "goal_mask_p"):
                if hasattr(env_cfg.curriculum, _term_name) and getattr(env_cfg.curriculum, _term_name) is not None:
                    setattr(env_cfg.curriculum, _term_name, None)
                    if int(os.environ.get("RANK", "0")) == 0:
                        print(
                            f"[evaluate_policy] Disabled curriculum term: {_term_name} "
                            "(eval has manual override)."
                        )

    if cotrain_modality is not None:
        if cotrain_modality == "jc":
            if cotrain_mask_modes is not None:
                raise ValueError("--mask_modes is only meaningful for --modality=kp.")
        else:  # kp
            if cotrain_p_mask is not None:
                raise ValueError("--p_mask is only meaningful for --modality=jc.")
            if cotrain_mask_modes is None:
                cotrain_mask_modes = "pelvis_only"  # default per Tier-1/2/4 testing convention
            _apply_single_mode_spec(cotrain_mask_modes)
        _disable_eval_curricula()
    elif single_mode_eval:
        if cotrain_p_mask is not None:
            raise ValueError("--p_mask is only meaningful for --modality=jc.")
        _apply_single_mode_spec(cotrain_mask_modes)
        _disable_eval_curricula()
    elif cotrain_p_mask is not None and "VR-Tracking" not in str(getattr(args_cli, "task", "") or ""):
        # Non-cotrain task (e.g. the JC MUSE-Transformer teacher) with an explicit
        # --p_mask: kill the goal-mask curriculum so it can't push p_mask off the
        # requested value at the checkpoint iter (the curriculum ramps p_mask
        # 0->0.5 by iter ~2000). p_mask itself is pinned post-env-creation below.
        _disable_eval_curricula()
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[evaluate_policy] Non-cotrain --p_mask={float(cotrain_p_mask):.2f}: "
                "goal-mask curriculum disabled; p_mask pinned after env creation."
            )

    task_name = str(getattr(args_cli, "task", "") or "")
    if "VR-Tracking" in task_name:
        if getattr(args_cli, "prior_checkpoint", None) in (None, ""):
            raise ValueError(
                f"Task {task_name!r} requires --prior_checkpoint=/path/to/pulse_student.pt "
                "(must contain prior.* and student_core.decoder.* for latent env; joint env uses it for obs norm)."
            )
        if "Joint" not in task_name:
            jp = env_cfg.actions.joint_pos
            jp.prior_checkpoint = os.path.abspath(str(args_cli.prior_checkpoint))
            if getattr(args_cli, "vr_residual_scale", None) is not None:
                jp.residual_scale = float(args_cli.vr_residual_scale)
            if getattr(args_cli, "vr_latent_dim", None) is not None:
                jp.latent_dim = int(args_cli.vr_latent_dim)
            if getattr(args_cli, "vr_proprio_history_length", None) is not None:
                jp.proprio_history_length = int(args_cli.vr_proprio_history_length)
        if getattr(args_cli, "vr_mask_mode_probs", None) not in (None, ""):
            parts = [float(x.strip()) for x in str(args_cli.vr_mask_mode_probs).split(",") if x.strip()]
            if len(parts) == 0:
                raise ValueError("--vr_mask_mode_probs must be a non-empty comma-separated list.")
            env_cfg.commands.motion.mask_mode_probs = tuple(parts)
        if getattr(args_cli, "vr_compact_goal_obs", None) not in (None, ""):
            compact_raw = str(args_cli.vr_compact_goal_obs).strip().lower()
            if compact_raw in ("1", "true", "t", "yes", "y", "on"):
                env_cfg.commands.motion.compact_goal_observation = True
            elif compact_raw in ("0", "false", "f", "no", "n", "off"):
                env_cfg.commands.motion.compact_goal_observation = False
            else:
                raise ValueError("--vr_compact_goal_obs must be a boolean string (true/false).")
        else:
            auto_full_raw = str(getattr(args_cli, "vr_use_full_goal_obs_with_distill_pretrain", "true")).strip().lower()
            if auto_full_raw in ("1", "true", "t", "yes", "y", "on"):
                auto_full_goal_obs = True
            elif auto_full_raw in ("0", "false", "f", "no", "n", "off"):
                auto_full_goal_obs = False
            else:
                raise ValueError(
                    "--vr_use_full_goal_obs_with_distill_pretrain must be a boolean string (true/false)."
                )
            using_distill_pretrain = bool(
                getattr(args_cli, "warmstart_from_masked_partial_kp_tracker", False)
                or getattr(args_cli, "load_pretrain", None) not in (None, "")
            )
            if auto_full_goal_obs and using_distill_pretrain:
                env_cfg.commands.motion.compact_goal_observation = False
                if int(os.environ.get("RANK", "0")) == 0:
                    print(
                        "[INFO] VR goal obs auto-set to full (non-compact) for distillation pretrain compatibility."
                    )
        if int(os.environ.get("RANK", "0")) == 0:
            if "Joint" not in task_name:
                print(
                    f"[INFO] VR-Tracking env overrides: prior_checkpoint={jp.prior_checkpoint!r}, "
                    f"latent_dim={getattr(jp, 'latent_dim', None)}, residual_scale={getattr(jp, 'residual_scale', None)}, "
                    f"mask_mode_probs={getattr(env_cfg.commands.motion, 'mask_mode_probs', None)}, "
                    f"compact_goal_observation={getattr(env_cfg.commands.motion, 'compact_goal_observation', None)}"
                )
            else:
                print(
                    f"[INFO] VR-Tracking-Joint eval: prior_checkpoint={os.path.abspath(str(args_cli.prior_checkpoint))!r}, "
                    f"mask_mode_probs={getattr(env_cfg.commands.motion, 'mask_mode_probs', None)}, "
                    f"compact_goal_observation={getattr(env_cfg.commands.motion, 'compact_goal_observation', None)}"
                )

    # Create logging directory for evaluation
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    eval_tag = f"eval_{agent_cfg.run_name or 'no_run_name'}"
    if cotrain_modality is not None:
        if cotrain_modality == "kp":
            eval_tag += f"__kp__{cotrain_mask_modes}"
        else:
            p_mask_value = float(cotrain_p_mask) if cotrain_p_mask is not None else 0.0
            eval_tag += f"__jc__pmask{p_mask_value:.2f}"
    elif single_mode_eval:
        eval_tag += f"__kp__{cotrain_mask_modes}"
    log_dir = os.path.join(
        log_root_path,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        eval_tag,
    )

    os.makedirs(log_dir, exist_ok=True)

    if int(os.environ.get("RANK", "0")) == 0:
        dump_yaml(os.path.join(log_dir, "params_env.yaml"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params_agent.pkl"), agent_cfg)

    # Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    vr_joint_proprio_dim_eval: int | None = None
    if "VR-Tracking-Joint" in task_name:
        _ce = env.unwrapped
        _jpe = _ce.action_manager.get_term("joint_pos")
        vr_joint_proprio_dim_eval = getattr(_jpe, "_proprio_dim", None)
        if not isinstance(vr_joint_proprio_dim_eval, int) or vr_joint_proprio_dim_eval <= 0:
            raise RuntimeError("VR-Tracking-Joint eval: missing ``_proprio_dim`` on joint_pos action term.")

    # Curriculum rollout hints: mirror train.py so iteration-gated curricula
    # (``phase_until_learning_iterations``, e.g. the MUSE-Kp p_see schedule) work.
    # RslRlVecEnvWrapper resets in __init__, which fires curriculum_manager.compute()
    # *before* runner.learn() sets the anchors — without this the curriculum raises.
    # Anchor at the checkpoint's iteration (parsed from the model_<iter>.pt filename) so
    # the curriculum evaluates at its FINAL phase from the very first reset (e.g. KP6
    # p_see=0.4 deployment-sparse), not phase 1 (all keypoints visible).
    _spe = int(getattr(agent_cfg, "num_steps_per_env", 0) or 0)
    if _spe > 0:
        _ckpt_name = str(
            getattr(args_cli, "checkpoint", None) or getattr(agent_cfg, "load_checkpoint", "") or ""
        )
        _digits = re.findall(r"(\d+)", os.path.basename(_ckpt_name))
        _anchor_iter = int(_digits[-1]) if _digits else 0
        attach_curriculum_rollout_hints(env, spe=_spe, learning_iteration=_anchor_iter)
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[evaluate_policy] curriculum rollout hints attached "
                f"(spe={_spe}, anchor_iter={_anchor_iter} from {_ckpt_name!r})."
            )

    env = RslRlVecEnvWrapper(env)

    resume_path_eval = getattr(agent_cfg, "resume_checkpoint_path", None)
    if resume_path_eval is None:
        resume_path_eval = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path_eval = os.path.abspath(str(resume_path_eval))

    # Build train config (mirror train.py so VR obs normalizer matches checkpoint training)
    train_cfg = agent_cfg.to_dict()
    if vr_joint_proprio_dim_eval is not None:
        train_cfg.setdefault("policy", {})
        train_cfg["policy"]["proprio_dim"] = int(vr_joint_proprio_dim_eval)
    if getattr(args_cli, "critic_warmup_itrs", None) is not None and isinstance(train_cfg.get("algorithm"), dict):
        train_cfg["algorithm"]["critic_warmup_itrs"] = max(0, int(args_cli.critic_warmup_itrs))
    if "VR-Tracking-Joint" in task_name:
        _lc_ev = getattr(args_cli, "latent_rl_checkpoint", None)
        if _lc_ev not in (None, ""):
            train_cfg["obs_normalizer_checkpoint_path"] = os.path.abspath(str(_lc_ev))
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    "[INFO] VR-Tracking-Joint eval: obs_normalizer_checkpoint_path from latent_rl_checkpoint "
                    f"{train_cfg['obs_normalizer_checkpoint_path']!r}"
                )
        else:
            train_cfg["obs_normalizer_checkpoint_path"] = resume_path_eval
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    "[INFO] VR-Tracking-Joint eval: obs_normalizer_checkpoint_path from eval resume "
                    f"{train_cfg['obs_normalizer_checkpoint_path']!r}"
                )
    elif "VR-Tracking" in task_name and getattr(args_cli, "prior_checkpoint", None) not in (None, ""):
        train_cfg["obs_normalizer_checkpoint_path"] = os.path.abspath(str(args_cli.prior_checkpoint))
    if (
        "VR-Tracking" in task_name
        and getattr(args_cli, "warmstart_from_masked_partial_kp_tracker", False)
        and getattr(args_cli, "warmstart_lock_full_normalizer", False)
    ):
        ws_norm_path = os.path.abspath(str(args_cli.warmstart_checkpoint))
        train_cfg["obs_normalizer_checkpoint_path"] = ws_norm_path
        train_cfg["enable_rl_split_obs_normalizer"] = False
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "[INFO] Warmstart normalizer lock (eval): "
                f"obs normalizer from {ws_norm_path!r}, split normalizer disabled."
            )

    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # Load checkpoint (path resolved above for joint obs norm)
    runner.load(resume_path_eval, load_optimizer=False, load_critic=False)

    # PULSE: optional explicit encoder-mean rollout (no latent sampling)
    policy = runner.alg.policy
    if args_cli.pulse_deterministic_latent and isinstance(policy, LatentBottleneckPULSE):
        policy.deterministic_latent = True
        if int(os.environ.get("RANK", "0")) == 0:
            print("[evaluate_policy] PULSE policy: deterministic_latent=True (encoder mean μ, no z sampling).")

    # Co-train: force single modality + apply runtime mask/goal overrides.
    if cotrain_modality is not None:
        if not hasattr(policy, "pilot_kp_fraction"):
            raise RuntimeError(
                "--modality requires a MUSE co-train policy with ``pilot_kp_fraction`` "
                f"(got {type(policy).__name__})."
            )
        policy.pilot_kp_fraction = 1.0 if cotrain_modality == "kp" else 0.0
        # Resolve the motion command term and apply per-modality overrides.
        env_u = env.unwrapped
        motion_term = (
            env_u.command_manager.get_term("motion") if hasattr(env_u, "command_manager") else None
        )
        if cotrain_modality == "kp":
            if motion_term is not None and hasattr(motion_term, "set_eval_fixed_mask_mode_idx"):
                motion_term.set_eval_fixed_mask_mode_idx(0)  # single-mode spec → index 0
                if hasattr(motion_term, "resample_all_mask_modes"):
                    motion_term.resample_all_mask_modes()
            # Defensive: KP modality should never see a goal-masked JC encoder; force p_mask=0.
            if motion_term is not None and hasattr(motion_term, "p_mask"):
                motion_term.p_mask = 0.0
        else:  # jc
            p_mask_value = float(cotrain_p_mask) if cotrain_p_mask is not None else 0.0
            if motion_term is not None and hasattr(motion_term, "p_mask"):
                motion_term.p_mask = p_mask_value
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[evaluate_policy] Co-train modality forced: {cotrain_modality!r} "
                f"(pilot_kp_fraction={policy.pilot_kp_fraction}, "
                f"p_mask={getattr(motion_term, 'p_mask', None)})."
            )
    elif cotrain_p_mask is not None:
        # Non-cotrain --p_mask (e.g. MUSE-Transformer teacher eval with every
        # timestep visible -> --p_mask 0.0). The goal-mask curriculum was already
        # disabled above; pin p_mask directly on the motion command.
        env_u = env.unwrapped
        motion_term = (
            env_u.command_manager.get_term("motion") if hasattr(env_u, "command_manager") else None
        )
        if motion_term is not None and hasattr(motion_term, "p_mask"):
            motion_term.p_mask = float(cotrain_p_mask)
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    f"[evaluate_policy] Non-cotrain p_mask pinned: motion.p_mask = {motion_term.p_mask:.2f} "
                    f"({'all timesteps visible' if motion_term.p_mask <= 0.0 else 'goal masked w.p. p_mask'})."
                )
        elif int(os.environ.get("RANK", "0")) == 0:
            print(
                "[evaluate_policy] WARNING: --p_mask given but motion command has no ``p_mask`` "
                f"attribute (got {type(motion_term).__name__ if motion_term is not None else None}); ignored."
            )

    # Per-clip metric logger (Phase 1): attach to the motion command, dump after rollouts.
    # Meaningful when --modality is set (cotrain testing) OR for standalone single-mode KP
    # eval (--mask_modes without --modality). For other eval flows the existing per-iter
    # wandb / tensorboard logging is sufficient.
    per_clip_logger = None
    if cotrain_modality is not None or single_mode_eval:
        from whole_body_tracking.utils.per_clip_eval_logger import attach_to_motion_command

        env_u = env.unwrapped
        motion_term = (
            env_u.command_manager.get_term("motion") if hasattr(env_u, "command_manager") else None
        )
        # MultiMotionCommand keeps the per-rank clip list on its loader
        # (``cmd.motion_dir_loader.motion_paths``), not directly on the command;
        # ``env_motion_indices`` indexes into that same list. PerClipEvalLogger reads
        # ``cmd.motion_paths`` — alias the loader's list onto the command so the logger
        # works for MultiMotionCommand subclasses (MUSE-Kp / cotrain) without touching
        # the shared per_clip_eval_logger / commands modules.
        if motion_term is not None and not getattr(motion_term, "motion_paths", None):
            _ldr = getattr(motion_term, "motion_dir_loader", None)
            _lpaths = list(getattr(_ldr, "motion_paths", []) or []) if _ldr is not None else []
            if _lpaths:
                motion_term.motion_paths = _lpaths
        n_paths = len(getattr(motion_term, "motion_paths", []) or [])
        if motion_term is not None and n_paths > 0:
            per_clip_logger = attach_to_motion_command(motion_term)
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    f"[evaluate_policy] PerClipEvalLogger attached "
                    f"(num_motions={n_paths})."
                )
        elif int(os.environ.get("RANK", "0")) == 0:
            print(
                "[evaluate_policy] WARNING: PerClipEvalLogger NOT attached "
                f"(motion_term={type(motion_term).__name__ if motion_term is not None else None}, "
                f"has_command_manager={hasattr(env_u, 'command_manager')}, "
                f"n_motion_paths={n_paths}). Per-clip CSV/summary will be absent; "
                "success_rate still comes from tfevents."
            )

    # Run evaluation rollouts without any updates
    runner.learn(num_learning_iterations=int(args_cli.eval_iters), init_at_random_ep_len=False, eval_mode=True)

    if per_clip_logger is not None:
        if cotrain_modality == "kp" or single_mode_eval:
            tag = f"kp__{cotrain_mask_modes}"
        elif cotrain_modality == "jc":
            p_mask_value = float(cotrain_p_mask) if cotrain_p_mask is not None else 0.0
            tag = f"jc__pmask{p_mask_value:.2f}"
        else:
            tag = ""
        rank = int(os.environ.get("RANK", "0"))
        per_clip_logger.dump(out_dir=log_dir, tag=f"{tag}__rank{rank}" if WORLD_SIZE > 1 else tag)

    env.close()


if __name__ == "__main__":
    # Hydra-decorated entrypoint: pyright/basedpyright cannot infer injected arguments.
    main()  # type: ignore[misc]
    simulation_app.close()
