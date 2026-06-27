"""Named synthetic-motion recipes.

A :class:`Recipe` is a declarative bundle of (name, duration, fps, factory). The
factory is called with no args and returns a :class:`.primitives.TorsoPrimitive`,
keeping recipes serializable-by-name (no closures over mutable state) while
remaining trivial to add.

The v1 set covers the user's two stated probes:

- **Squat** — vertical sine at ~standing torso height, three amplitudes × two freqs.
- **Walk ±x / ±y** — constant-velocity translation at four speeds in four directions.

All recipes default to ``fps=50`` (matches SONIC) and yaw=0 (face +x at reset).
Z-trajectories are expressed about the seed clip's frame-0 torso height (the
recentered torso starts at the seed's torso z); horizontal trajectories start at
(0, 0) — same as the recentered seed frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

from .primitives import LinearTrans, SquatDown, TorsoPrimitive

if TYPE_CHECKING:
    from .motion_writer import SyntheticMotionBuilder


# Approximate G1 standing torso z (used as the centerline for vertical sines). Doesn't
# have to be exact — the seed clip's frame-0 torso z anchors the actual reset state;
# the primitive sets the *commanded* trajectory and the policy must follow.
_STANDING_TORSO_Z = 1.07


@dataclass(frozen=True)
class Recipe:
    """A named recipe that produces one synthetic motion clip.

    Two flavours:

    - **Legacy single-primitive** (torso-only recipes): set ``primitive_factory``;
      the CLI wraps it as a torso-only builder.
    - **Multi-body** (wrist-writing recipes): set ``builder_factory`` instead;
      the CLI passes the chosen seed clip path + robot key and the factory
      returns a fully-configured :class:`SyntheticMotionBuilder`. This route is
      needed when the recipe wants to place trajectories relative to the seed's
      frame-0 pose (e.g. "wrist writes letters in a plane in front of the torso").
    """

    name: str
    duration_s: float
    fps: int
    description: str = ""
    primitive_factory: Optional[Callable[[], TorsoPrimitive]] = None
    builder_factory: Optional[
        Callable[[str, str, float, int], "SyntheticMotionBuilder"]
    ] = None

    def __post_init__(self) -> None:
        if (self.primitive_factory is None) == (self.builder_factory is None):
            raise ValueError(
                f"Recipe {self.name!r} must set exactly one of primitive_factory / builder_factory."
            )

    def make_primitive(self) -> TorsoPrimitive:
        if self.primitive_factory is None:
            raise RuntimeError(
                f"Recipe {self.name!r} uses builder_factory (multi-body); call make_builder instead."
            )
        return self.primitive_factory()

    def make_builder(self, seed_clip_path: str, robot: str = "g1") -> "SyntheticMotionBuilder":
        """Return a fully-configured SyntheticMotionBuilder.

        For builder-factory recipes, the factory is invoked with the (seed_clip_path,
        robot, duration_s, fps) so it can load the seed and place letter planes
        relative to the seed's post-rotation torso position.

        For legacy primitive-factory recipes, we wrap the primitive into a
        torso-only builder with default anchoring (same as the CLI's legacy path).
        """
        from .motion_writer import SyntheticMotionBuilder

        if self.builder_factory is not None:
            return self.builder_factory(seed_clip_path, robot, self.duration_s, self.fps)
        return SyntheticMotionBuilder(
            seed_clip_path=seed_clip_path,
            primitive=self.make_primitive(),
            duration_s=self.duration_s,
            fps=self.fps,
            robot=robot,
        )


# ------------------------------------------------------------------ Squats
#
# Squats are intentionally **one-sided** (:class:`SquatDown`): the trajectory starts at
# standing height and drops down by ``depth_m``, never rising above the start. Real
# bipedal squats look like this; a symmetric sine around standing would push the torso
# above the standing height (physically unreachable without a hop) for half the cycle.


def _squat(depth_m: float, freq_hz: float) -> Callable[[], TorsoPrimitive]:
    def _factory() -> TorsoPrimitive:
        return SquatDown(
            top_z=_STANDING_TORSO_Z,
            depth_m=depth_m,
            freq_hz=freq_hz,
            xy=(0.0, 0.0),
            yaw_rad=0.0,
        )

    return _factory


# ------------------------------------------------------------------ Walks
#
# Direction = world-frame angle CCW from +x (math convention). With the motion
# writer's default ``anchor_yaw_to_seed=True``, the robot's *facing* is locked to
# the seed clip's frame-0 torso quat (typically near-+x for ``walk_forward`` seed
# clips); only the translation vector changes. So:
#   - 000°  → forward (along seed-facing)
#   - 045°  → forward-left diagonal
#   - 090°  → strafe left
#   - 180°  → walk backward
#   - 270°  → strafe right
#   - etc.
# 8 directions × 6 speeds = 48 walk recipes. Speed is *ground* speed; the
# direction vector is unit-normalized by ``LinearTrans`` so naming the speed
# axis stays semantically clean.


_WALK_ANGLES_DEG = (0, 45, 90, 135, 180, 225, 270, 315)
_WALK_SPEEDS_MS = (0.1, 0.3, 0.6, 1.0, 1.5, 2.0)


def _walk(direction: tuple[float, float, float], speed_ms: float) -> Callable[[], TorsoPrimitive]:
    def _factory() -> TorsoPrimitive:
        return LinearTrans(
            start_pos=(0.0, 0.0, _STANDING_TORSO_Z),
            direction=direction,
            speed_ms=speed_ms,
            yaw_rad=0.0,  # ignored when anchor_yaw_to_seed=True (the default).
        )

    return _factory


def _angle_to_dir(deg: int) -> tuple[float, float, float]:
    """Unit vector in the XY plane at ``deg`` CCW from +x. Z = 0."""
    rad = math.radians(deg)
    return (math.cos(rad), math.sin(rad), 0.0)


# ------------------------------------------------------------------ Registry


# ------------------------------------------------------------------ Wrist writing
#
# The wrist-writing recipes use the *multi-body* path of ``SyntheticMotionBuilder``:
#   - ``face_world_yaw=0``: robot is rotated to face +x so a frontal camera at
#     (~+x, 0, h) sees it head-on (play.py sets ``viewer.eye`` accordingly under
#     ``--synth_eval``).
#   - ``body_primitives = {right_wrist_yaw_link: Word("MUSE", ...)}``: only the
#     right wrist traces letters in a chest-high YZ plane in front of the torso.
#   - ``stay_bodies = [torso_link, pelvis, left_*, *_ankle]``: every other tracked
#     body is pinned to its seed standing pose. With ``--mask_modes kp5_full``
#     the policy sees a steady-standing target on every body except the right
#     wrist — so the demo isn't dominated by a wobbly torso.
#   - ``letter_trail_points``: passed through to the npz so play.py can spawn
#     static red dots showing the letter outline (independent of which frame is
#     playing).


# Plane geometry, picked once for all wrist-writing recipes. Letters live in a
# vertical plane in front of the robot:
#   - plane normal: +x world (away from robot, toward the frontal camera)
#   - plane "x" axis (letters' horizontal): +y world (camera-right when looking back
#     at robot from +x, so words read left → right as expected)
#   - plane "y" axis (letters' vertical): +z world
_WRIST_PLANE_FORWARD_OFFSET_M = 0.40   # how far in front of the torso the plane sits
_WRIST_PLANE_VERTICAL_OFFSET_M = 0.10  # how far above the torso the word baseline sits


def build_wrist_writing_builder(
    *,
    seed_clip_path: str,
    text: str,
    letter_size_m: float,
    letter_spacing_m: float,
    stroke_speed_ms: float,
    duration_s: float,
    body_name: str = "right_wrist_yaw_link",
    resample_step_m: float = 0.03,
    robot: str = "g1",
    fps: int = 50,
    plane_forward_offset_m: float = _WRIST_PLANE_FORWARD_OFFSET_M,
    plane_vertical_offset_m: float = _WRIST_PLANE_VERTICAL_OFFSET_M,
    plane_y_offset_m: float = 0.0,
) -> "SyntheticMotionBuilder":
    """Standalone constructor for a multi-body wrist-writing builder.

    Loads the seed clip's frame-0 torso position, places the writing plane in
    front of it, compiles ``text`` into a Polyline3D + trail-point cloud, and
    returns a fully-configured :class:`SyntheticMotionBuilder`. Used by both:

    - the registered recipes in :func:`_build_registry` (via the closure-based
      ``_word_builder_factory``);
    - the ad-hoc ``python scripts/synth_cli.py write ...`` subcommand, which
      lets you draw an arbitrary string at any size without registering a recipe.
    """
    import numpy as np

    from .letters import write_word
    from .motion_writer import (
        SyntheticMotionBuilder,
        _rotate_xy,
        _yaw_from_quat_wxyz,
        load_body_meta,
    )

    meta = load_body_meta(robot)
    # Replicate the writer's frame-0 transform stack so we can resolve the
    # post-rotation torso and wrist positions used as the writing plane anchor.
    seed = np.load(seed_clip_path)
    seed_body_pos = seed["body_pos_w"][0].copy().astype(np.float64)
    seed_body_quat = seed["body_quat_w"][0].copy().astype(np.float64)
    pelvis_xy = seed_body_pos[meta.pelvis_index, :2].copy()
    seed_body_pos[:, :2] -= pelvis_xy  # recenter
    seed_torso_yaw = _yaw_from_quat_wxyz(seed_body_quat[meta.torso_index])
    face_world_yaw = 0.0
    delta_yaw = face_world_yaw - seed_torso_yaw
    if abs(delta_yaw) > 1e-6:
        seed_body_pos = _rotate_xy(seed_body_pos, delta_yaw)
    torso_post = seed_body_pos[meta.torso_index]  # (3,)
    body_idx = meta.body_names.index(body_name)
    wrist_natural = seed_body_pos[body_idx]  # (3,)  post-rotation wrist xyz

    # Letter plane: the FIRST letter's lower-left corner is placed close to the
    # wrist's natural standing pose so the wrist barely has to move to begin
    # writing (no long lift-in). The word then extends in +y (camera-right /
    # robot's left), which means for long words the robot still has to step
    # sideways to reach later letters — but the *start* is right next to where
    # the arm naturally rests.
    #   - plane_origin_x  : in front of the wrist (slightly forward)
    #   - plane_origin_y  : the wrist's natural y (so first letter starts there)
    #   - plane_origin_z  : chest height (above torso), so writing is in view
    plane_origin = (
        float(wrist_natural[0]) + plane_forward_offset_m,
        float(wrist_natural[1]) + plane_y_offset_m,
        float(torso_post[2]) + plane_vertical_offset_m,
    )
    plane_x_axis = (0.0, 1.0, 0.0)  # letter +x = world +y (camera's right when viewing from +x)
    plane_y_axis = (0.0, 0.0, 1.0)  # letter +y = world +z (up)
    word_width_m = max(0.0, len(text) * letter_size_m + max(0, len(text) - 1) * letter_spacing_m)

    polyline, trail = write_word(
        text,
        letter_size_m=letter_size_m,
        letter_spacing_m=letter_spacing_m,
        stroke_speed_ms=stroke_speed_ms,
        plane_origin=plane_origin,
        plane_x_axis=plane_x_axis,
        plane_y_axis=plane_y_axis,
        resample_step_m=resample_step_m,
    )

    # Prepend a "lift-in" waypoint at the wrist's frame-0 (post-rotation)
    # position so the wrist smoothly travels from its natural standing pose
    # to the first letter point. With the plane anchored at wrist_natural y
    # (above), this lift-in segment is now very short (just the small forward
    # + vertical offset to the writing plane) instead of the previous ~30 cm
    # off to the side. The trail points (red dots) are unchanged.
    from .primitives import Polyline3D

    lifted_waypoints = (tuple(wrist_natural.astype(np.float64)), *polyline.waypoints)
    polyline = Polyline3D(waypoints=lifted_waypoints, speed_ms=polyline.speed_ms)
    # Stay bodies = every tracked KP5 body except the one drawing letters.
    # Hardcoded inline (vs importing from the tracking config) to keep this
    # module Isaac-free for offline npz generation.
    kp5 = (
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    stay_bodies = tuple(b for b in kp5 if b != body_name)

    return SyntheticMotionBuilder(
        seed_clip_path=seed_clip_path,
        primitive=None,
        duration_s=duration_s,
        fps=fps,
        robot=robot,
        anchor_torso_to_seed=False,  # the Word polyline is already in absolute world coords
        anchor_yaw_to_seed=True,     # keep the (post-rotation) seed quat for the wrist
        body_primitives={body_name: polyline},
        stay_bodies=stay_bodies,
        face_world_yaw=face_world_yaw,
        letter_trail_points=trail,
        letter_trail_body=body_name,
    )


def _word_builder_factory(
    text: str,
    *,
    letter_size_m: float,
    letter_spacing_m: float,
    stroke_speed_ms: float,
    body_name: str = "right_wrist_yaw_link",
    resample_step_m: float = 0.03,
) -> Callable[[str, str, float, int], "SyntheticMotionBuilder"]:
    """Closure form used by the registered recipes.

    Defers to :func:`build_wrist_writing_builder`; the closure is what
    ``Recipe.builder_factory`` expects (called with ``(seed, robot, dur, fps)``).
    """

    def _factory(seed_clip_path: str, robot: str, duration_s: float, fps: int) -> "SyntheticMotionBuilder":
        return build_wrist_writing_builder(
            seed_clip_path=seed_clip_path,
            text=text,
            letter_size_m=letter_size_m,
            letter_spacing_m=letter_spacing_m,
            stroke_speed_ms=stroke_speed_ms,
            duration_s=duration_s,
            body_name=body_name,
            resample_step_m=resample_step_m,
            robot=robot,
            fps=fps,
        )

    return _factory


def _build_registry() -> dict[str, Recipe]:
    out: dict[str, Recipe] = {}

    # Squats — depth-from-standing ∈ {0.10, 0.15, 0.20} m × freq ∈ {0.5, 1.0} Hz.
    # Name tag = depth in cm (e.g. ``squat_15cm_1p0hz`` drops 15cm at 1Hz).
    for depth in (0.10, 0.15, 0.20):
        for freq in (0.5, 1.0):
            depth_tag = f"{int(round(depth * 100)):02d}cm"  # 10cm, 15cm, 20cm
            freq_tag = f"{freq:.1f}hz".replace(".", "p")  # 0.5hz -> 0p5hz
            name = f"squat_{depth_tag}_{freq_tag}"
            # Three full cycles minimum: duration = max(4 s, 3 / freq).
            dur = max(4.0, 3.0 / freq)
            out[name] = Recipe(
                name=name,
                duration_s=dur,
                fps=50,
                primitive_factory=_squat(depth, freq),
                description=(
                    f"Torso squat-down: depth={depth:.2f} m below standing, "
                    f"freq={freq:.1f} Hz, top z={_STANDING_TORSO_Z:.2f} m, "
                    f"duration={dur:.1f} s."
                ),
            )

    # Walks — 8 directions × 6 speeds.
    # 6s × max 2 m/s = 12m max excursion. The camera follows the env origin so the
    # robot can drift this far without leaving frame (verified for 6m diagonals).
    walk_duration_s = 6.0
    for deg in _WALK_ANGLES_DEG:
        dir_vec = _angle_to_dir(deg)
        for speed in _WALK_SPEEDS_MS:
            speed_tag = f"{speed:.1f}ms".replace(".", "p")  # 0.3 -> 0p3ms
            name = f"walk_{deg:03d}deg_{speed_tag}"
            out[name] = Recipe(
                name=name,
                duration_s=walk_duration_s,
                fps=50,
                primitive_factory=_walk(dir_vec, speed),
                description=(
                    f"Torso linear translation: world-frame angle={deg}° CCW from +x, "
                    f"unit dir={dir_vec}, speed={speed:.1f} m/s, start z={_STANDING_TORSO_Z:.2f} m, "
                    f"duration={walk_duration_s:.1f} s. Facing locked to seed (anchor_yaw_to_seed)."
                ),
            )

    # ------------------------------------------------------------------ Wrist writing
    # The right wrist traces a string of letters in a vertical plane in front of the
    # robot. Two play-time mask strategies are useful with these recipes:
    #   - ``--mask_modes kp5_full``: every tracked body has a target (other bodies
    #     pinned to seed standing pose). Cleanest visual demo at small letter
    #     sizes — the body stays still while the wrist draws.
    #   - ``--mask_modes right_wrist_only``: only the right wrist target is visible
    #     to the policy. The policy is FREE to move the torso, ankles, etc. — and at
    #     large letter sizes (≥ 0.30 m) it has to in order to reach. This is the
    #     "does the policy locomote to draw?" probe.

    # Default small-letter variants (one per word, body-pinned demo).
    _WRIST_WORDS = (
        ("MUSE", 8.0, "right"),
        ("HI",   3.0, "right"),
        ("MUSE", 8.0, "left"),  # mirror-friendly variant if you want both wrists.
    )
    for text, duration_s, side in _WRIST_WORDS:
        body = "right_wrist_yaw_link" if side == "right" else "left_wrist_yaw_link"
        name = f"write_{text}_{'rwrist' if side == 'right' else 'lwrist'}_chest"
        out[name] = Recipe(
            name=name,
            duration_s=duration_s,
            fps=50,
            description=(
                f"{side.capitalize()} wrist writes {text!r} (small, 12 cm). KP5_full mask suggested; "
                f"torso + ankles + non-writing wrist pinned to seed standing pose. "
                f"face_world_yaw=0 (robot faces +x, frontal camera)."
            ),
            builder_factory=_word_builder_factory(
                text,
                letter_size_m=0.12,
                letter_spacing_m=0.04,
                stroke_speed_ms=0.30,
                body_name=body,
                resample_step_m=0.025,
            ),
        )

    # Right-wrist MUSE size sweep — exercise locomotion. Letter spacing scales
    # proportionally (1/3 of letter size). Stroke speed scales so total drawing
    # time stays in the 8–12 s range. Suggested play-time mask:
    # ``--mask_modes right_wrist_only`` so the policy isn't forced to keep the
    # torso steady; it can step/lean/twist to reach the bigger letters.
    _WRIST_SIZE_SWEEP = (
        # (size_m, duration_s, speed_ms)
        (0.12,  8.0, 0.30),
        (0.20, 10.0, 0.45),
        (0.30, 12.0, 0.65),
        (0.45, 14.0, 0.90),
    )
    for size, dur, speed in _WRIST_SIZE_SWEEP:
        size_cm = int(round(size * 100))
        spacing = round(size / 3.0, 3)
        name = f"write_MUSE_rwrist_size{size_cm:02d}cm"
        out[name] = Recipe(
            name=name,
            duration_s=dur,
            fps=50,
            description=(
                f"Right wrist writes 'MUSE' at letter_size={size:.2f} m (spacing={spacing:.2f} m), "
                f"stroke_speed={speed:.2f} m/s, duration={dur:.1f} s. Suggested mask: "
                f"right_wrist_only (policy free to locomote to reach the larger letters)."
            ),
            builder_factory=_word_builder_factory(
                "MUSE",
                letter_size_m=size,
                letter_spacing_m=spacing,
                stroke_speed_ms=speed,
                body_name="right_wrist_yaw_link",
                # Scale the resample step proportionally so the trail-dot density per
                # cm of stroke stays constant across the sweep (≈25 samples per letter).
                resample_step_m=max(0.015, size * 0.16),
            ),
        )

    return out


RECIPES: dict[str, Recipe] = _build_registry()


def list_recipes() -> list[str]:
    """Sorted list of recipe names."""
    return sorted(RECIPES.keys())


def get_recipe(name: str) -> Recipe:
    """Lookup with a clear KeyError message."""
    if name not in RECIPES:
        raise KeyError(
            f"Unknown recipe {name!r}. Known recipes ({len(RECIPES)}): {list_recipes()}"
        )
    return RECIPES[name]
