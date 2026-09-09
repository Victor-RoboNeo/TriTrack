from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    arg_group.add_argument(
        "--resume_student_checkpoint",
        type=str,
        default=None,
        help="Simplified resume: provide full path to student checkpoint (e.g., /path/to/model_10000.pt). "
        "This automatically sets --resume=True and parses --load_run and --checkpoint from the path.",
    )
    arg_group.add_argument(
        "--load_pretrain",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint (.pt) to initialize the policy weights (no optimizer state). "
        "Intended for initializing PPO from an offline/online distillation run.",
    )
    # -- teacher load arguments (for MOSAIC distillation)
    arg_group.add_argument("--load_teacher_run", type=str, default=None, help="Name of the teacher run folder (for MOSAIC).")
    arg_group.add_argument("--teacher_checkpoint", type=str, default=None, help="Teacher checkpoint file (for MOSAIC).")
    arg_group.add_argument(
        "--encoder_decoder_warmstart",
        type=str,
        default=None,
        help="Optional MUSE-Transformer .pt to warmstart encoder + decoder weights AFTER the teacher "
        "checkpoint loads (model_state_dict only; obs normalizer untouched). Used by MUSE-Kp.",
    )
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    arg_group.add_argument(
        "--wandb_path", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    arg_group.add_argument("--distributed", action="store_true", default=False, help="Enable torchrun DDP.")
    arg_group.add_argument(
        "--warmstart_itrs",
        type=int,
        default=0,
        help="Number of iterations to use warmstart (teacher actions) during PULSE/2b rollout. Zero disables warmstart.",
    )
    arg_group.add_argument(
        "--teacher_decay_iters",
        type=int,
        default=None,
        help="[2b only] Number of iters for teacher→student decay. Larger = softer curriculum.",
    )
    arg_group.add_argument(
        "--teacher_decay_schedule",
        type=str,
        default=None,
        choices=["linear", "sqrt", "log", "exp"],
        help="[2b only] Decay shape: sqrt/log = rapid at first, then levels off.",
    )
    arg_group.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="[2b/PULSE/distillation] Override algorithm learning rate (e.g. 3e-4 for gentler updates).",
    )
    arg_group.add_argument(
        "--critic_warmup_itrs",
        type=int,
        default=None,
        help="[PPO] Number of initial learning iterations to update critic only (actor frozen).",
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # handle simplified resume_student_checkpoint parameter (CLI override)
    if hasattr(args_cli, "resume_student_checkpoint") and args_cli.resume_student_checkpoint is not None:
        import os

        checkpoint_path = os.path.abspath(args_cli.resume_student_checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Student checkpoint not found: {checkpoint_path}")

        # parse run directory and checkpoint filename
        # Expected path format: logs/rsl_rl/{experiment_name}/{run_name}/model_*.pt
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_file = os.path.basename(checkpoint_path)
        run_name = os.path.basename(checkpoint_dir)

        # automatically set resume parameters
        args_cli.resume = True
        args_cli.load_run = run_name
        args_cli.checkpoint = checkpoint_file
        # train.py must load this file directly: get_checkpoint_path() uses the *current*
        # experiment_name, so init-from-another-run (e.g. prior-only after full PULSE) would break.
        setattr(agent_cfg, "resume_checkpoint_path", checkpoint_path)

        print(f"[CLI] Simplified resume from CLI: {checkpoint_path}")
        print(f"[CLI]   - load_run: {run_name}")
        print(f"[CLI]   - checkpoint: {checkpoint_file}")

    # handle student_checkpoint_path from config (if not overridden by CLI)
    elif hasattr(agent_cfg, "student_checkpoint_path") and agent_cfg.student_checkpoint_path is not None:
        import os

        checkpoint_path = os.path.abspath(agent_cfg.student_checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Student checkpoint not found in config: {checkpoint_path}")

        # parse run directory and checkpoint filename
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_file = os.path.basename(checkpoint_path)
        run_name = os.path.basename(checkpoint_dir)

        # automatically set resume parameters
        agent_cfg.resume = True
        agent_cfg.load_run = run_name
        agent_cfg.load_checkpoint = checkpoint_file
        setattr(agent_cfg, "resume_checkpoint_path", checkpoint_path)

        print(f"[Config] Resuming from student_checkpoint_path: {checkpoint_path}")
        print(f"[Config]   - load_run: {run_name}")
        print(f"[Config]   - checkpoint: {checkpoint_file}")

    # handle load_pretrain: initialize from a pretrained checkpoint without resuming optimizer/state.
    # This is useful for initializing PPO from a distillation run.
    if hasattr(args_cli, "load_pretrain") and args_cli.load_pretrain is not None:
        import os

        pretrain_path = os.path.abspath(args_cli.load_pretrain)
        if not os.path.exists(pretrain_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrain_path}")

        # Store on agent_cfg so scripts can pass it through to the runner.
        # We deliberately do NOT flip agent_cfg.resume here; the training script
        # should call runner.load(pretrain_path, load_optimizer=False, load_critic=True)
        # before starting PPO.
        agent_cfg.pretrained_checkpoint_path = pretrain_path  # type: ignore[attr-defined]
        print(f"[CLI] Will initialize policy from pretrained checkpoint: {pretrain_path}")

    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if getattr(args_cli, "experiment_name", None):
        agent_cfg.experiment_name = args_cli.experiment_name
    if hasattr(args_cli, "warmstart_itrs"):
        agent_cfg.warmstart_itrs = args_cli.warmstart_itrs
    if hasattr(args_cli, "teacher_decay_iters") and args_cli.teacher_decay_iters is not None:
        if hasattr(agent_cfg, "algorithm") and hasattr(agent_cfg.algorithm, "teacher_decay_iters"):
            agent_cfg.algorithm.teacher_decay_iters = args_cli.teacher_decay_iters
            print(f"[CLI] Set teacher_decay_iters={args_cli.teacher_decay_iters}")
    if hasattr(args_cli, "teacher_decay_schedule") and args_cli.teacher_decay_schedule is not None:
        if hasattr(agent_cfg, "algorithm") and hasattr(agent_cfg.algorithm, "teacher_decay_schedule"):
            agent_cfg.algorithm.teacher_decay_schedule = args_cli.teacher_decay_schedule
            print(f"[CLI] Set teacher_decay_schedule={args_cli.teacher_decay_schedule}")
    if hasattr(args_cli, "learning_rate") and args_cli.learning_rate is not None:
        if hasattr(agent_cfg, "algorithm") and hasattr(agent_cfg.algorithm, "learning_rate"):
            agent_cfg.algorithm.learning_rate = args_cli.learning_rate
            print(f"[CLI] Set learning_rate={args_cli.learning_rate}")
        # Prior-only PULSE: Adam is only over the prior; mirror CLI LR into prior_fit_learning_rate.
        if hasattr(agent_cfg, "algorithm") and hasattr(agent_cfg.algorithm, "prior_fit_learning_rate"):
            agent_cfg.algorithm.prior_fit_learning_rate = args_cli.learning_rate
            print(f"[CLI] Set prior_fit_learning_rate={args_cli.learning_rate}")
    if hasattr(args_cli, "critic_warmup_itrs") and args_cli.critic_warmup_itrs is not None:
        if hasattr(agent_cfg, "algorithm"):
            warm_iters = max(0, int(args_cli.critic_warmup_itrs))
            setattr(agent_cfg.algorithm, "critic_warmup_itrs", warm_iters)
            print(f"[CLI] Set critic_warmup_itrs={warm_iters} (from --critic_warmup_itrs)")
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    # override teacher checkpoint path for MOSAIC (if provided)
    if hasattr(args_cli, "load_teacher_run") and args_cli.load_teacher_run is not None:
        if hasattr(agent_cfg, "algorithm") and hasattr(agent_cfg.algorithm, "teacher_checkpoint_path"):
            # construct teacher checkpoint path from load_teacher_run and teacher_checkpoint
            import os
            teacher_checkpoint_file = args_cli.teacher_checkpoint if args_cli.teacher_checkpoint is not None else "model_18000.pt"
            log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
            teacher_checkpoint_path = os.path.join(log_root_path, args_cli.load_teacher_run, teacher_checkpoint_file)
            teacher_checkpoint_path = os.path.abspath(teacher_checkpoint_path)
            agent_cfg.algorithm.teacher_checkpoint_path = teacher_checkpoint_path
            print(f"[CLI] Overriding teacher checkpoint path: {teacher_checkpoint_path}")
    elif hasattr(args_cli, "teacher_checkpoint") and args_cli.teacher_checkpoint is not None:
        # direct path override (absolute or relative)
        if hasattr(agent_cfg, "algorithm") and hasattr(agent_cfg.algorithm, "teacher_checkpoint_path"):
            import os
            teacher_checkpoint_path = os.path.abspath(args_cli.teacher_checkpoint)
            agent_cfg.algorithm.teacher_checkpoint_path = teacher_checkpoint_path
            print(f"[CLI] Overriding teacher checkpoint path: {teacher_checkpoint_path}")

    # Optional encoder/decoder warmstart (MUSE-Kp): applied AFTER teacher checkpoint, model_state_dict
    # only (no normalizer touched). Set via --encoder_decoder_warmstart.
    if hasattr(args_cli, "encoder_decoder_warmstart") and args_cli.encoder_decoder_warmstart is not None:
        if hasattr(agent_cfg, "algorithm") and hasattr(
            agent_cfg.algorithm, "encoder_decoder_warmstart_checkpoint_path"
        ):
            import os
            warmstart_path = os.path.abspath(args_cli.encoder_decoder_warmstart)
            agent_cfg.algorithm.encoder_decoder_warmstart_checkpoint_path = warmstart_path
            print(f"[CLI] Overriding encoder/decoder warmstart checkpoint path: {warmstart_path}")

    return agent_cfg
