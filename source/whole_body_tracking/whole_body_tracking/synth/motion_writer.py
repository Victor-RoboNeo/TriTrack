"""Assemble a synthetic torso-only motion ``.npz``.

Pipeline
--------
1. Load a "seed" SONIC clip (a standing/walking real motion).
2. Copy its frame 0 wholesale (joint_pos, joint_vel, body_*_w for ALL 30 bodies).
3. **Recenter** so the seed frame-0 pelvis is at world (0, 0): subtract the pelvis
   XY offset from every body's XY across the seed frame. This guarantees the robot
   spawns at the env origin (XY-centered in the camera) at reset, no extra play.py
   flags needed.
4. Build the torso trajectory from a :class:`.primitives.TorsoPrimitive` and write
   it into ``body_pos_w[1:T, torso_idx]`` (likewise quat/lin_vel/ang_vel).
5. Fill every other body's data at frames 1..T-1 with **NaN**. The masked-obs
   pipeline already turns NaN at masked bodies into the NaN sentinel in the
   observation; combined with ``--mask_modes kp{5,6}_torso`` at play time, the
   policy sees only the torso reference.
6. ``joint_pos`` for frames 1..T-1 = frame-0 joint pose (placeholder; not read by
   the reward/observation path since reset only fires at frame 0). ``joint_vel`` = 0.

Output schema matches the existing SONIC npz exactly: keys ``fps``, ``joint_pos``,
``joint_vel``, ``body_pos_w``, ``body_quat_w``, ``body_lin_vel_w``, ``body_ang_vel_w``.

The writer is **deliberately offline**: no Isaac dependency. It reads the cached
body-name list from :mod:`.dump_body_names` to find the torso index.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .primitives import TorsoPrimitive, quat_from_yaw


def _yaw_from_quat_wxyz(q: np.ndarray) -> float:
    """Extract yaw (rotation about world +Z) from a WXYZ quaternion.

    Uses the standard yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)) formula. Stable
    for any unit quaternion; exact for pure-yaw rotations.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two WXYZ quaternions: returns a * b (apply b first then a)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _rotate_xy(points_xyz: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Rotate points about world +Z by ``yaw_rad`` (in-place safe: returns a copy)."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    out = points_xyz.astype(np.float64).copy()
    x = out[..., 0].copy()
    y = out[..., 1].copy()
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out


# Canonical npz keys (every key must be present in the seed and produced in the output).
_REQUIRED_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass
class BodyMeta:
    """Cached body-name metadata loaded from ``synth/cache/<robot>_body_names.json``."""

    robot: str
    body_names: list[str]
    joint_names: list[str]
    pelvis_index: int
    torso_index: int

    @property
    def num_bodies(self) -> int:
        return len(self.body_names)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


def load_body_meta(robot: str = "g1") -> BodyMeta:
    """Load cached body-name metadata. Raises with a useful hint if missing."""
    cache = Path(__file__).resolve().parent / "cache" / f"{robot}_body_names.json"
    if not cache.exists():
        raise FileNotFoundError(
            f"Body-name cache not found at {cache}. Generate it once with:\n"
            f"    python scripts/synth_dump_body_names.py --robot {robot}\n"
            "This launches Isaac briefly to dump robot.body_names; the synth writer is "
            "then fully offline."
        )
    with cache.open() as fh:
        d = json.load(fh)
    return BodyMeta(
        robot=d["robot"],
        body_names=d["body_names"],
        joint_names=d["joint_names"],
        pelvis_index=d["pelvis_index"],
        torso_index=d["torso_index"],
    )


@dataclass
class SyntheticMotionBuilder:
    """Builds a synthetic torso-only motion npz from a seed clip + a torso primitive.

    Parameters
    ----------
    seed_clip_path:
        Absolute path to a SONIC ``.npz`` whose frame 0 is a stable standing pose
        (e.g. anything under ``test/loco/walk_forward/``).
    primitive:
        :class:`TorsoPrimitive` describing the torso trajectory in the **recentered**
        frame (seed-frame pelvis at origin).
    duration_s:
        Total trajectory duration in seconds.
    fps:
        Output frame rate. Defaults to the seed clip's fps.
    robot:
        Robot platform key for the body-name cache. Defaults to ``g1``.
    anchor_torso_to_seed:
        When True (default), translates the primitive's trajectory so that
        ``primitive.at(0)`` exactly matches the seed clip's frame-0 torso position
        (after pelvis-XY recentering). This eliminates the otherwise-jarring
        frame-0 → frame-1 jump between the seed's actual torso pose and the
        recipe's hardcoded centerline. Recipe authors can write trajectories in
        nominal "standing torso z" coordinates and have them anchor naturally
        per-seed. Velocities and quaternions are translation-invariant.
    anchor_yaw_to_seed:
        When True (default), overrides the primitive's quaternion across all
        frames with the seed clip's frame-0 torso quaternion (and zeroes angular
        velocity). Eliminates the yaw discontinuity between frame 0 (which keeps
        the seed quat) and frames 1..T-1 (which would otherwise use the
        primitive's yaw, typically identity). The policy sees a *constant* facing
        direction, with translation direction supplied by the primitive — exactly
        what walks-in-arbitrary-directions probes want. Set False if you add a
        primitive that drives yaw explicitly (none of the current primitives do).
    """

    seed_clip_path: str | Path
    primitive: Optional[TorsoPrimitive] = None
    duration_s: float = 0.0
    fps: Optional[int] = None
    robot: str = "g1"
    anchor_torso_to_seed: bool = True
    anchor_yaw_to_seed: bool = True
    # ---- Multi-body extensions (used by the wrist-writing pipeline) -----------------
    # ``body_primitives``: maps body NAME → primitive. Supersedes ``primitive`` when
    # set (which then becomes optional). Each body's trajectory is anchored to ITS
    # OWN frame-0 seed pose (analogous to ``anchor_torso_to_seed`` but per-body).
    body_primitives: Optional[dict[str, TorsoPrimitive]] = None
    # ``stay_bodies``: list of body NAMES that get a constant trajectory pinned to
    # their frame-0 seed pose. Use this for bodies that the mask covers but you
    # don't want to move (e.g. torso/ankles steady while the right wrist writes
    # letters). Bodies that appear in neither dict nor list remain NaN at t≥1,
    # matching the existing torso-only behaviour.
    stay_bodies: tuple[str, ...] = ()
    # ``face_world_yaw``: rotates the entire frame-0 setup so the seed's torso
    # forward direction aligns with this world-frame yaw (radians, CCW from +x).
    # The rotation is applied to every body's position (about the recentered
    # origin) and every body's quaternion; frame-0 linear velocities are rotated
    # too (angular velocity preserved). Use 0.0 to force "robot faces +x" so a
    # frontal camera at (+x, 0, h) sees the robot head-on.
    face_world_yaw: Optional[float] = None
    # ``letter_trail_points``: optional ``(N, 3)`` point cloud (world frame) stashed in
    # the npz under ``_synth_letter_trail_xyz``. ``play.py --synth_eval`` reads it
    # to spawn N **static** sphere markers (red dots fixed in air) showing the
    # target letter shape. Independent from the body's trajectory — used purely
    # for visualisation. Already expressed in WORLD coordinates (apply
    # ``face_world_yaw`` rotation yourself before passing it in, since the
    # writer's rotation happens to body data, not to standalone trail points).
    letter_trail_points: Optional[np.ndarray] = None
    # ``letter_trail_body``: name of the body whose moving goal marker should be
    # HIDDEN at play time when the static trail is rendered (so we see only the
    # static red dots + the moving green robot wrist). Typically the same body
    # whose primitive draws the letters (e.g. ``right_wrist_yaw_link``).
    letter_trail_body: Optional[str] = None

    def __post_init__(self) -> None:
        self.seed_clip_path = Path(self.seed_clip_path)
        if not self.seed_clip_path.is_file():
            raise FileNotFoundError(f"Seed clip not found: {self.seed_clip_path}")
        if self.duration_s <= 0:
            raise ValueError(f"duration_s must be > 0; got {self.duration_s}")
        if self.primitive is None and not self.body_primitives:
            raise ValueError(
                "SyntheticMotionBuilder needs either ``primitive`` (legacy torso-only) or "
                "``body_primitives`` (multi-body)."
            )

    # ------------------------------------------------------------------ helpers

    def _load_seed_frame0(self) -> dict[str, np.ndarray]:
        data = np.load(self.seed_clip_path)
        missing = [k for k in _REQUIRED_KEYS if k not in data.files]
        if missing:
            raise ValueError(
                f"Seed clip {self.seed_clip_path} missing required keys: {missing}. "
                f"Got: {list(data.files)}"
            )
        seed_fps = int(np.asarray(data["fps"]).item())
        return {
            "fps": seed_fps,
            "joint_pos": data["joint_pos"][0].copy(),       # (J,)
            "joint_vel": data["joint_vel"][0].copy(),       # (J,)
            "body_pos_w": data["body_pos_w"][0].copy(),     # (B, 3)
            "body_quat_w": data["body_quat_w"][0].copy(),   # (B, 4)
            "body_lin_vel_w": data["body_lin_vel_w"][0].copy(),  # (B, 3)
            "body_ang_vel_w": data["body_ang_vel_w"][0].copy(),  # (B, 3)
        }

    def _recenter_inplace(self, seed: dict[str, np.ndarray], meta: BodyMeta) -> tuple[float, float]:
        """Translate every body so the seed pelvis XY is at (0, 0). Z untouched.

        Returns the (dx, dy) offset that was *subtracted* (for logging / reproducibility).
        """
        pelvis_xy = seed["body_pos_w"][meta.pelvis_index, :2].copy()
        seed["body_pos_w"][:, :2] -= pelvis_xy
        # Velocities/orientations are translation-invariant — no adjustment needed.
        return float(pelvis_xy[0]), float(pelvis_xy[1])

    # ------------------------------------------------------------------ build

    def build(self) -> dict[str, np.ndarray]:
        """Build the synthetic motion arrays. Does not write to disk.

        Three code paths share the same output schema:

        - **Legacy torso-only** (``primitive`` set, ``body_primitives is None``):
          writes the primitive into the torso slot; every other body is NaN at t≥1.
        - **Multi-body** (``body_primitives`` set): writes each named body's primitive
          into its slot. Bodies in ``stay_bodies`` get a Constant-at-seed-frame-0
          trajectory. Unlisted bodies remain NaN at t≥1.
        - **Both** behaviours combine cleanly with ``face_world_yaw``, which rotates
          the entire frame-0 setup so the seed's torso forward aligns with the given
          world yaw. Used by the wrist-writing pipeline to put the robot at +x
          facing for the frontal camera.
        """
        meta = load_body_meta(self.robot)
        seed = self._load_seed_frame0()

        # Validate seed body axis matches the cached body count.
        if seed["body_pos_w"].shape[0] != meta.num_bodies:
            raise ValueError(
                f"Seed clip has {seed['body_pos_w'].shape[0]} bodies but cached body_names has "
                f"{meta.num_bodies}. Re-run dump_body_names if the URDF changed; otherwise check "
                f"that the seed clip was generated with the same robot."
            )

        out_fps = int(self.fps) if self.fps is not None else seed["fps"]
        if out_fps <= 0:
            raise ValueError(f"fps must be > 0; got {out_fps}")

        dx, dy = self._recenter_inplace(seed, meta)

        # ---- Frame-0 yaw alignment (applied AFTER recentering, BEFORE primitives) ---
        # When ``face_world_yaw`` is set we rotate every frame-0 body around the
        # recentered origin (pelvis XY = 0) so the seed's torso forward direction
        # ends up aligned with the requested world yaw. The wrist-writing pipeline
        # uses this to make the robot face +x (yaw=0), which makes the frontal
        # camera position predictable and lets recipes write letters in the YZ
        # plane in front of the robot without per-seed math.
        applied_yaw_delta = 0.0
        if self.face_world_yaw is not None:
            seed_torso_yaw = _yaw_from_quat_wxyz(seed["body_quat_w"][meta.torso_index])
            applied_yaw_delta = float(self.face_world_yaw) - seed_torso_yaw
            if abs(applied_yaw_delta) > 1e-6:
                seed["body_pos_w"] = _rotate_xy(seed["body_pos_w"], applied_yaw_delta)
                seed["body_lin_vel_w"] = _rotate_xy(seed["body_lin_vel_w"], applied_yaw_delta)
                # Rotate quaternions: new_q = R_yaw * old_q (pre-multiply with the
                # delta-yaw quat so the rotation is applied in the world frame).
                delta_q = quat_from_yaw(applied_yaw_delta)
                rotated_quats = np.zeros_like(seed["body_quat_w"], dtype=np.float64)
                for i in range(meta.num_bodies):
                    rotated_quats[i] = _quat_mul_wxyz(delta_q, seed["body_quat_w"][i].astype(np.float64))
                seed["body_quat_w"] = rotated_quats.astype(np.float32)
                # Angular velocity around +Z is invariant under a yaw rotation; XY
                # angular vel rotates like a planar vector. For the static reset
                # frame these are typically tiny — rotate them properly for
                # correctness.
                seed["body_ang_vel_w"] = _rotate_xy(seed["body_ang_vel_w"], applied_yaw_delta)

        # ---- Determine which bodies get which trajectories --------------------------
        # Legacy path: ``primitive`` set, ``body_primitives is None`` → torso only.
        if self.body_primitives is None:
            body_prims: dict[str, TorsoPrimitive] = {"torso_link": self.primitive}
        else:
            body_prims = dict(self.body_primitives)

        for name in body_prims:
            if name not in meta.body_names:
                raise KeyError(
                    f"body_primitives key {name!r} not in robot.body_names ({len(meta.body_names)} entries). "
                    f"Available: {meta.body_names}"
                )
        stay = list(self.stay_bodies)
        for name in stay:
            if name not in meta.body_names:
                raise KeyError(f"stay_bodies entry {name!r} not in robot.body_names.")
            if name in body_prims:
                raise ValueError(
                    f"Body {name!r} appears in both ``body_primitives`` and ``stay_bodies``; pick one."
                )

        # ---- Sample each body's primitive (anchor to its own frame-0 seed pose) -----
        sampled_trajectories: dict[str, dict[str, np.ndarray]] = {}
        anchor_offsets: dict[str, np.ndarray] = {}
        for name, prim in body_prims.items():
            pos_traj, quat_traj, lin_traj, ang_traj = prim.sample(out_fps, self.duration_s)
            body_idx = meta.body_names.index(name)
            offset = np.zeros(3, dtype=np.float64)
            if self.anchor_torso_to_seed:
                seed_pos = seed["body_pos_w"][body_idx].astype(np.float64)
                offset = seed_pos - pos_traj[0]
                pos_traj = pos_traj + offset[None, :]
            if self.anchor_yaw_to_seed:
                seed_q = seed["body_quat_w"][body_idx].astype(np.float64)
                quat_traj = np.broadcast_to(seed_q, quat_traj.shape).astype(np.float64).copy()
                ang_traj = np.zeros_like(ang_traj)
            sampled_trajectories[name] = {
                "pos": pos_traj,
                "quat": quat_traj,
                "lin": lin_traj,
                "ang": ang_traj,
            }
            anchor_offsets[name] = offset

        # T derived from the first sampled trajectory (all primitives are sampled at
        # the same fps/duration so lengths agree).
        any_traj = next(iter(sampled_trajectories.values()))
        T = any_traj["pos"].shape[0]
        if T < 2:
            raise ValueError(
                f"Sampled trajectory too short (T={T}); need at least 2 frames "
                f"(duration_s={self.duration_s}, fps={out_fps})."
            )

        B = meta.num_bodies
        J = seed["joint_pos"].shape[0]

        out = {
            "fps": np.array([out_fps], dtype=np.int64),
            "joint_pos": np.zeros((T, J), dtype=np.float32),
            "joint_vel": np.zeros((T, J), dtype=np.float32),
            "body_pos_w": np.full((T, B, 3), np.nan, dtype=np.float32),
            "body_quat_w": np.full((T, B, 4), np.nan, dtype=np.float32),
            "body_lin_vel_w": np.full((T, B, 3), np.nan, dtype=np.float32),
            "body_ang_vel_w": np.full((T, B, 3), np.nan, dtype=np.float32),
        }

        # Frame 0: copy seed frame 0 wholesale (recentered, possibly yaw-rotated).
        out["joint_pos"][0] = seed["joint_pos"]
        out["joint_vel"][0] = seed["joint_vel"]
        out["body_pos_w"][0] = seed["body_pos_w"]
        out["body_quat_w"][0] = seed["body_quat_w"]
        out["body_lin_vel_w"][0] = seed["body_lin_vel_w"]
        out["body_ang_vel_w"][0] = seed["body_ang_vel_w"]

        # Frames 1..T-1: joint_pos repeats frame-0 pose (placeholder; not consumed by
        # the reward/obs path), joint_vel = 0.
        out["joint_pos"][1:] = seed["joint_pos"][None, :]
        # joint_vel already zero-initialized.

        # Write each body's primitive trajectory.
        for name, traj in sampled_trajectories.items():
            bi = meta.body_names.index(name)
            out["body_pos_w"][1:, bi, :] = traj["pos"][1:].astype(np.float32)
            out["body_quat_w"][1:, bi, :] = traj["quat"][1:].astype(np.float32)
            out["body_lin_vel_w"][1:, bi, :] = traj["lin"][1:].astype(np.float32)
            out["body_ang_vel_w"][1:, bi, :] = traj["ang"][1:].astype(np.float32)

        # Stay bodies: broadcast frame-0 (post-rotation) pose across all frames.
        for name in stay:
            bi = meta.body_names.index(name)
            out["body_pos_w"][1:, bi, :] = seed["body_pos_w"][bi].astype(np.float32)[None, :]
            out["body_quat_w"][1:, bi, :] = seed["body_quat_w"][bi].astype(np.float32)[None, :]
            # Zero velocities for stay bodies (they're commanded to stand still).
            out["body_lin_vel_w"][1:, bi, :] = 0.0
            out["body_ang_vel_w"][1:, bi, :] = 0.0

        # Sanity assertions: every named body is non-NaN; unlisted bodies are NaN.
        named = set(body_prims) | set(stay)
        for name in named:
            bi = meta.body_names.index(name)
            assert not np.isnan(out["body_pos_w"][1:, bi]).any(), f"named body {name!r} has NaN at t>=1"
        unnamed_mask = np.array(
            [name not in named for name in meta.body_names], dtype=bool
        )
        assert np.isnan(out["body_pos_w"][1:, unnamed_mask]).all(), (
            "unnamed bodies must be NaN at t>=1; got finite values somewhere"
        )

        # Stash the recenter offset as a non-required key so it's discoverable later
        # without polluting the schema (np.savez accepts arbitrary kwargs; downstream
        # readers ignore unknown keys).
        out["_synth_recenter_offset_xy"] = np.array([dx, dy], dtype=np.float64)
        out["_synth_torso_index"] = np.array([meta.torso_index], dtype=np.int64)
        out["_synth_pelvis_index"] = np.array([meta.pelvis_index], dtype=np.int64)
        out["_synth_duration_s"] = np.array([float(self.duration_s)], dtype=np.float64)
        out["_synth_seed_clip"] = np.array([str(self.seed_clip_path)])
        out["_synth_face_world_yaw"] = np.array(
            [float(self.face_world_yaw) if self.face_world_yaw is not None else float("nan")],
            dtype=np.float64,
        )
        out["_synth_applied_yaw_delta"] = np.array([applied_yaw_delta], dtype=np.float64)
        # Names of the bodies driven by an explicit primitive (vs ``stay_bodies``).
        # play.py uses this to know which body to render the static letter-trail
        # markers for, and which body's moving goal marker to hide.
        out["_synth_primitive_bodies"] = np.array(list(body_prims.keys()))
        out["_synth_stay_bodies"] = np.array(list(stay))
        # Legacy field kept for backwards-compat with the torso-only video pipeline.
        if "torso_link" in body_prims:
            out["_synth_torso_anchor_offset_xyz"] = anchor_offsets["torso_link"].astype(np.float64)
        # Optional letter-trail point cloud + body name (consumed by play.py to
        # render static red dots and hide the moving goal marker for the wrist).
        if self.letter_trail_points is not None:
            arr = np.asarray(self.letter_trail_points, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"letter_trail_points must have shape (N, 3); got {arr.shape}")
            out["_synth_letter_trail_xyz"] = arr
        if self.letter_trail_body is not None:
            if self.letter_trail_body not in meta.body_names:
                raise KeyError(
                    f"letter_trail_body {self.letter_trail_body!r} not in robot.body_names."
                )
            out["_synth_letter_trail_body"] = np.array([self.letter_trail_body])
        return out

    def write(self, out_path: str | Path) -> Path:
        """Build and save the npz. Returns the written path."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self.build()
        # Use savez (not savez_compressed) to match SONIC clips' on-disk layout.
        np.savez(out_path, **arrays)
        return out_path
