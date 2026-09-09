#!/usr/bin/env python3
"""Clean P1R training: 50% F1 / 50% F2, LR=1e-6, frozen normalizer, plane, mask=[1,1,1]."""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

ANYBODY = Path("/data/home/chenxiangyu/robotics/Anybody")
sys.path.insert(0, str(ANYBODY / "scripts"))
sys.path.insert(0, "/data/home/chenxiangyu/humantracker_3pt_ood")
sys.path.insert(0, "/data/home/chenxiangyu/victor/TriTrack")

from flat_nominal_competence_repair_v2.constants import (  # noqa: E402
    ACTOR_LR,
    PARENT_A,
    PARENT_ITER,
    P1_MASK_HYDRA,
    PLANE_HYDRA,
    PRIOR_A,
    SEED,
    TASK,
    experiment_name,
)


def install_mix_hook() -> None:
    import builtins

    if getattr(install_mix_hook, "_installed", False):
        return
    prev = builtins.__import__

    def _patch():
        mod = sys.modules.get("isaaclab.envs.manager_based_env")
        cls = getattr(mod, "ManagerBasedEnv", None) if mod is not None else None
        if cls is None or getattr(cls, "_ncr_mix_patched", False):
            return
        orig = cls.__init__

        def _init(self, cfg, *args, **kwargs):
            mix = os.environ.get("FLAT_NCR_MIX")
            if mix:
                motion = getattr(getattr(cfg, "commands", None), "motion", None)
                if motion is not None:
                    motion.motion_groups = {"static": ["/static/"], "reach": ["/reach/"]}
                    motion.motion_group_sampling_ratios = {"static": 0.5, "reach": 0.5}
            orig(self, cfg, *args, **kwargs)

        cls.__init__ = _init
        cls._ncr_mix_patched = True

    def hooked(name, globals=None, locals=None, fromlist=(), level=0):
        mod = prev(name, globals, locals, fromlist, level)
        if name.startswith("isaaclab.envs"):
            _patch()
        return mod

    builtins.__import__ = hooked
    install_mix_hook._installed = True
    _patch()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="R5")
    ap.add_argument("--experiment-name", default="")
    ap.add_argument("--run-name", default=f"seed{SEED}")
    ap.add_argument("--max-iterations", type=int, default=100)
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", default=str(PARENT_A))
    ap.add_argument("--motion", required=True)
    ap.add_argument("--learning-rate", type=float, default=ACTOR_LR)
    ap.add_argument("--print-only", action="store_true")
    args, unknown = ap.parse_known_args()
    name = args.experiment_name or experiment_name(args.stage)
    argv = [
        str(ANYBODY / "scripts/rsl_rl/train.py"),
        "--headless",
        f"--device={args.device}",
        f"--task={TASK}",
        f"--motion={args.motion}",
        f"--resume_student_checkpoint={args.resume}",
        f"--experiment_name={name}",
        f"--run_name={args.run_name}",
        f"--seed={args.seed}",
        f"--learning_rate={args.learning_rate}",
        f"--max_iterations={args.max_iterations}",
        f"--num_envs={args.num_envs}",
        *list(PRIOR_A),
        *list(PLANE_HYDRA),
        *list(P1_MASK_HYDRA),
        "agent.save_interval=100",
        "agent.freeze_normalizer_on_resume=true",
        "agent.empirical_normalization=true",
        "env.curriculum.motion_group_ratio=null",
    ] + unknown
    print("[train_cell]", " ".join(argv), flush=True)
    if args.print_only:
        return
    os.environ["FLAT_LOCOMANI_LAV2_LIVE"] = "1"
    os.environ["FLAT_LOCOMANI_LAV2_ADDED_ITERS"] = str(args.max_iterations)
    os.environ["FLAT_LOCOMANI_LAV2_PARENT_ITER"] = str(PARENT_ITER)
    os.environ["FLAT_NCR_MIX"] = "1"
    os.environ.pop("FLAT_LOCOMANI_C0", None)
    os.environ.pop("FLAT_LOCOMANI_LAV3_CELL", None)
    from flat_locomani.live_anchor_v2.live_train_wrapper import install_import_hook

    install_import_hook()
    install_mix_hook()
    sys.argv = argv
    os.chdir(str(ANYBODY))
    sys.path.insert(0, str(ANYBODY / "scripts" / "rsl_rl"))
    runpy.run_path(argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
