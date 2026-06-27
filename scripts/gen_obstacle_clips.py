"""Generate a pool of "reach a fixed point" clips + obstacle sidecars for one phase.

Each clip is a standing seed with the RIGHT WRIST pinned to a sampled reach point P
(a ``Constant`` primitive); the other KP5 bodies are held standing (``stay_bodies``).
This is the wrist-writing synthetic pipeline minus the moving letters, so the masked-KP
obs / reward / terminations work unchanged. Alongside each ``<name>.npz`` we write a
``<name>.scene.json`` (same schema as the scene-editor library) carrying the obstacle
boxes + target, which the ObstacleReach command loads for keep-out + obstacle obs.

The pool *is* the randomization: ``--num-clips`` scenes sampled from the per-phase ranges
(:data:`obstacle_reach.PHASE_SPECS`) ≈ continuous. No Isaac needed (numpy only).

    python scripts/gen_obstacle_clips.py --phase 1 --num-clips 256
    for p in 0 1 2 4; do python scripts/gen_obstacle_clips.py --phase $p; done
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("WHOLE_BODY_TRACKING_NO_TASKS", "1")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
WBT_SRC = REPO / "source/whole_body_tracking"
sys.path.insert(0, str(WBT_SRC))

from whole_body_tracking.synth import SyntheticMotionBuilder       # noqa: E402
from whole_body_tracking.synth.primitives import Constant          # noqa: E402


def _load_by_path(name: str, rel: str):
    """Import a module straight from its file, bypassing the heavy mdp package __init__."""
    spec = importlib.util.spec_from_file_location(name, str(WBT_SRC / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_orh = _load_by_path("obstacle_reach", "whole_body_tracking/tasks/tracking/mdp/obstacle_reach.py")

# mask_modes.COTRAIN_KP5_BODIES (stable; hardcoded to avoid importing the config package).
KP5_BODIES = [
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]
WRIST_BODY = "right_wrist_yaw_link"
assert WRIST_BODY in KP5_BODIES, f"{WRIST_BODY} not in KP5 bodies {KP5_BODIES}"
STAY_BODIES = tuple(b for b in KP5_BODIES if b != WRIST_BODY)

DEFAULT_SEED = REPO / "scene_editor/assets/g1_standing_seed.npz"


def _freeze_to_constant(npz_path: Path) -> None:
    """Make a fully-finite CONSTANT-pose clip from a builder npz.

    The synth builder fills only the named bodies (primitive + ``stay_bodies``) for frames
    ``t>=1`` and leaves every *other* body NaN by design (incl. the pelvis/root, body 0).
    That is fine when the consumer always resets at frame 0, but ObstacleReach inherits
    ``random_init_frame`` and would reset the ROOT at a NaN frame -> NaN robot. For the
    standing seed we want a degenerate constant clip anyway, so tile frame 0 across ALL
    frames (all 30 bodies finite) and zero the velocities. The right wrist is overridden to
    the per-reset target P by the command, so the seed's wrist value here is irrelevant.
    """
    d = dict(np.load(npz_path))
    for k in ("body_pos_w", "body_quat_w", "joint_pos"):
        d[k][:] = d[k][0:1]                       # tile frame 0 (already finite for all bodies)
    for k in ("body_lin_vel_w", "body_ang_vel_w", "joint_vel"):
        d[k][:] = 0.0                             # constant pose -> zero velocity
    assert np.isfinite(d["body_pos_w"]).all(), "freeze left NaNs in body_pos_w"
    np.savez(npz_path, **d)


def _scene_sidecar(scene, i: int) -> dict:
    """ObstacleScene[i] -> scene-editor JSON dict (full-extent sizes)."""
    obstacles = []
    valid = scene.valid[i]
    for k in range(scene.centers.shape[1]):
        if not bool(valid[k]):
            continue
        obstacles.append({
            "id": f"obs_{k}",
            "type": "box",
            "pos": [round(float(v), 5) for v in scene.centers[i, k].tolist()],
            "size": [round(float(2.0 * v), 5) for v in scene.half[i, k].tolist()],
            "quat": [round(float(v), 6) for v in scene.quat[i, k].tolist()],
        })
    return {"obstacles": obstacles, "target": [round(float(v), 5) for v in scene.target[i].tolist()]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate obstacle-reach clip pool for a phase.")
    ap.add_argument("--phase", type=int, required=True, choices=[0, 1, 2, 4])
    ap.add_argument("--num-clips", type=int, default=256)
    ap.add_argument("--duration", type=float, default=8.0, help="clip length (s); must exceed the episode")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--seed-clip", type=str, default=str(DEFAULT_SEED))
    ap.add_argument("--out-root", type=str, default=str(REPO / "outputs/obstacle_pool"))
    ap.add_argument("--rng", type=int, default=0)
    args = ap.parse_args()

    import torch
    gen = torch.Generator().manual_seed(args.rng)
    scene = _orh.sample_phase(args.phase, args.num_clips, "cpu", gen=gen)

    out_dir = Path(args.out_root) / f"phase{args.phase}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gen] phase {args.phase}: {args.num_clips} clips -> {out_dir}")
    print(f"[gen] wrist={WRIST_BODY} stay={STAY_BODIES}")

    for i in range(args.num_clips):
        P = tuple(float(v) for v in scene.target[i].tolist())
        builder = SyntheticMotionBuilder(
            seed_clip_path=args.seed_clip,
            body_primitives={WRIST_BODY: Constant(pos=P)},
            stay_bodies=STAY_BODIES,
            duration_s=args.duration,
            fps=args.fps,
            face_world_yaw=0.0,             # robot faces +x
            anchor_torso_to_seed=False,     # use the ABSOLUTE point P (don't re-anchor to seed wrist)
            anchor_yaw_to_seed=True,
        )
        name = f"reach_{args.phase}_{i:04d}"
        npz_path = out_dir / f"{name}.npz"
        builder.write(npz_path)
        # Standing seed must be FULLY finite at every frame (root reset can land on any
        # frame); the builder leaves non-KP bodies NaN at t>=1. Freeze to a constant clip.
        _freeze_to_constant(npz_path)
        (out_dir / f"{name}.scene.json").write_text(json.dumps(_scene_sidecar(scene, i)))
        if (i + 1) % 64 == 0 or i == args.num_clips - 1:
            print(f"  [{i + 1}/{args.num_clips}] {name}  P={tuple(round(v,3) for v in P)}")

    print(f"[gen] done: {args.num_clips} clips + sidecars in {out_dir}")


if __name__ == "__main__":
    main()
