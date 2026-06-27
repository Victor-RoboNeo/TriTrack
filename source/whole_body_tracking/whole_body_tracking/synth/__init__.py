"""Synthetic torso-only trajectory generator for testing partial-keypoint trackers.

The pipeline produces standard-format motion ``.npz`` files (identical to SONIC clips)
in which **only the torso body has a non-NaN trajectory after frame 0**. Frame 0 is
borrowed wholesale from a real "standing" seed clip so reset/articulation-root state
is valid; frames 1..T-1 set the torso to the synthetic trajectory and leave every
other body NaN. Combined with ``--mask_modes kp*_torso`` at play time, this exercises
a partial-KP tracker on out-of-distribution torso commands (squat, walk, …) while the
existing masked-obs pipeline cleanly hides the NaN reference for non-visible bodies.

The output drops straight into ``play.py --motion <synth.npz>`` and into any motion-
dir-based PPO/distillation training (same npz schema as SONIC). No env-cfg changes
are required: ``--start_frame 0`` and the default ``--video`` termination-disable
behaviour cover the rest.

Submodules
----------
- :mod:`.primitives`     — composable torso trajectories (Constant, LinearTrans, …).
- :mod:`.motion_writer`  — :class:`SyntheticMotionBuilder` assembles the npz.
- :mod:`.recipes`        — named, declarative trajectory recipes.
- :mod:`.cli`            — command-line generator entry point (offline).

Top-level scripts
-----------------
- ``scripts/synth_cli.py``               — offline CLI wrapper (sets
  ``WHOLE_BODY_TRACKING_NO_TASKS=1`` so the package import doesn't pull Isaac).
- ``scripts/synth_dump_body_names.py``   — one-time Isaac launcher that caches the
  robot's body-name order to ``synth/cache/<robot>_body_names.json``. Must run BEFORE
  the package import (mirrors ``scripts/csv_to_npz.py``'s structure).
"""

from .primitives import (
    Compose,
    Constant,
    LinearTrans,
    Sequence,
    SquatDown,
    TorsoPrimitive,
    TorsoState,
    VerticalSine,
)
from .motion_writer import SyntheticMotionBuilder
from .recipes import Recipe, RECIPES, get_recipe, list_recipes

__all__ = [
    "Compose",
    "Constant",
    "LinearTrans",
    "Sequence",
    "SquatDown",
    "TorsoPrimitive",
    "TorsoState",
    "VerticalSine",
    "SyntheticMotionBuilder",
    "Recipe",
    "RECIPES",
    "get_recipe",
    "list_recipes",
]
