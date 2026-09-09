#!/usr/bin/env python3
"""Clean continuation training: live-anchor pack, NO 18-D task error, plane only."""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

ANYBODY = Path("/data/home/chenxiangyu/robotics/Anybody")
HT_ROOT = Path("/data/home/chenxiangyu/humantracker_3pt_ood")
sys.path.insert(0, str(ANYBODY / "scripts"))
sys.path.insert(0, str(HT_ROOT))
sys.path.insert(0, "/data/home/chenxiangyu/victor/TriTrack")

from flat_unified_sparse_intent_v1.constants import (  # noqa: E402
    ACTOR_LR,
    ANYBODY as ANYBODY_C,
    CKPT_TRAIN_A,
    EXTRA_ITERS,
    NUM_ENVS,
    P1_MASK_HYDRA,
    PARENT_ITER,
    PLANE_HYDRA,
    PRIOR_A,
    SEED,
    SONIC_TRAIN,
    TASK,
    RESULTS,
    experiment_name,
)


def install_noop_hook(out_json: Path) -> None:
    import builtins

    if getattr(install_noop_hook, "_installed", False):
        return
    prev = builtins.__import__

    def _patch():
        mod = sys.modules.get("rsl_rl.runners.on_policy_runner")
        if mod is None or not hasattr(mod, "OnPolicyRunner"):
            return
        cls = mod.OnPolicyRunner
        if getattr(cls, "_fusi_noop_patched", False):
            return
        orig = cls.learn

        def learn(self, *args, **kwargs):
            import torch

            before = {n: p.detach().cpu().clone() for n, p in self.alg.policy.named_parameters()}
            out = orig(self, *args, **kwargs)
            max_abs = 0.0
            for n, p in self.alg.policy.named_parameters():
                d = float((p.detach().cpu() - before[n]).abs().max().item())
                if d > max_abs:
                    max_abs = d
            payload = {
                "max_abs_parameter_delta": max_abs,
                "PASS": max_abs == 0.0,
                "n_params": len(before),
            }
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(payload, indent=2) + "\n")
            print("[noop]", payload, flush=True)
            if max_abs != 0.0:
                raise RuntimeError(f"NOOP identity failed max_abs_parameter_delta={max_abs}")
            return out

        cls.learn = learn
        cls._fusi_noop_patched = True

    def hooked(name, globals=None, locals=None, fromlist=(), level=0):
        mod = prev(name, globals, locals, fromlist, level)
        if name.startswith("rsl_rl"):
            _patch()
        return mod

    builtins.__import__ = hooked
    install_noop_hook._installed = True
    _patch()


def build_argv(args) -> list[str]:
    name = args.experiment_name or experiment_name(args.stage)
    argv = [
        str(ANYBODY_C / "scripts/rsl_rl/train.py"),
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
        "agent.save_interval=250",
    ]
    return argv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="P1")
    ap.add_argument("--experiment-name", default="")
    ap.add_argument("--run-name", default=f"seed{SEED}")
    ap.add_argument("--max-iterations", type=int, default=EXTRA_ITERS)
    ap.add_argument("--num-envs", type=int, default=NUM_ENVS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", default=str(CKPT_TRAIN_A))
    ap.add_argument("--motion", default=str(SONIC_TRAIN))
    ap.add_argument("--learning-rate", type=float, default=ACTOR_LR)
    ap.add_argument("--noop", action="store_true")
    ap.add_argument("--print-only", action="store_true")
    args, unknown = ap.parse_known_args()
    argv = build_argv(args)
    argv += unknown
    print("[train_cell]", " ".join(argv), flush=True)
    if args.print_only:
        return
    os.environ["FLAT_LOCOMANI_LAV2_LIVE"] = "1"
    os.environ["FLAT_LOCOMANI_LAV2_ADDED_ITERS"] = str(args.max_iterations)
    os.environ["FLAT_LOCOMANI_LAV2_PARENT_ITER"] = str(PARENT_ITER)
    os.environ.pop("FLAT_LOCOMANI_C0", None)
    os.environ.pop("FLAT_LOCOMANI_LAV3_CELL", None)
    from flat_locomani.live_anchor_v2.live_train_wrapper import install_import_hook

    install_import_hook()
    if args.noop:
        install_noop_hook(RESULTS / "02_p1_3pt_competence" / "noop_identity.json")
    sys.argv = argv
    os.chdir(str(ANYBODY_C))
    sys.path.insert(0, str(ANYBODY_C / "scripts" / "rsl_rl"))
    runpy.run_path(argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
