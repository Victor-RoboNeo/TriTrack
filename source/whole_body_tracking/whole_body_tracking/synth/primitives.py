"""Composable torso trajectory primitives.

Every primitive maps **time-in-seconds → world-frame torso state**: position,
quaternion (WXYZ, Isaac Lab convention), linear velocity, angular velocity. Velocities
are computed analytically per-primitive (not finite-differenced) so composed
trajectories don't accumulate boundary artifacts.

The torso state is interpreted in the **recentered** seed-clip frame: the seed clip's
frame-0 pelvis XY is the origin (see :class:`.motion_writer.SyntheticMotionBuilder`).
A "stay" primitive at the recentered torso position therefore corresponds to the
robot standing still at the env origin.

Composition
-----------
- :class:`Compose` — additive overlay (e.g. translating *and* squatting at once).
  Sums positions and velocities; uses the FIRST primitive's quaternion/angular vel.
- :class:`Sequence` — temporal concatenation, with each segment expressed in its
  own local time. Positions of later segments are offset so the trajectory is
  continuous across boundaries (no teleports).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# ---- Quaternion helpers (WXYZ, Isaac Lab convention) -----------------------------


def quat_identity() -> np.ndarray:
    """Identity quaternion in WXYZ order."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def quat_from_yaw(yaw_rad: float) -> np.ndarray:
    """Pure-yaw quaternion (rotation about world +Z) in WXYZ order."""
    c, s = math.cos(0.5 * yaw_rad), math.sin(0.5 * yaw_rad)
    return np.array([c, 0.0, 0.0, s], dtype=np.float64)


# ---- State container --------------------------------------------------------------


@dataclass(frozen=True)
class TorsoState:
    """Torso state at a single instant. All vectors in **world frame**."""

    pos: np.ndarray  # shape (3,)
    quat_wxyz: np.ndarray  # shape (4,)
    lin_vel: np.ndarray  # shape (3,)
    ang_vel: np.ndarray  # shape (3,)

    def __post_init__(self) -> None:
        assert self.pos.shape == (3,), self.pos.shape
        assert self.quat_wxyz.shape == (4,), self.quat_wxyz.shape
        assert self.lin_vel.shape == (3,), self.lin_vel.shape
        assert self.ang_vel.shape == (3,), self.ang_vel.shape


# ---- Base class -------------------------------------------------------------------


class TorsoPrimitive(ABC):
    """Abstract primitive: ``t -> TorsoState``. ``t`` is in seconds since segment start."""

    @abstractmethod
    def at(self, t: float) -> TorsoState:
        """Evaluate the primitive at time ``t`` (seconds, ≥ 0)."""
        raise NotImplementedError

    def sample(self, fps: int, duration_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample ``ceil(duration_s * fps) + 1`` evenly-spaced frames including t=0.

        Returns (pos, quat_wxyz, lin_vel, ang_vel) with leading axis ``T``.
        """
        num_frames = int(math.ceil(duration_s * fps)) + 1
        ts = np.arange(num_frames, dtype=np.float64) / float(fps)
        pos = np.zeros((num_frames, 3), dtype=np.float64)
        quat = np.zeros((num_frames, 4), dtype=np.float64)
        lin = np.zeros((num_frames, 3), dtype=np.float64)
        ang = np.zeros((num_frames, 3), dtype=np.float64)
        for i, t in enumerate(ts):
            s = self.at(float(t))
            pos[i] = s.pos
            quat[i] = s.quat_wxyz
            lin[i] = s.lin_vel
            ang[i] = s.ang_vel
        return pos, quat, lin, ang


# ---- Concrete primitives ----------------------------------------------------------


@dataclass(frozen=True)
class Constant(TorsoPrimitive):
    """Stay put at a fixed pose (no velocity, identity rotation by default)."""

    pos: tuple[float, float, float]
    yaw_rad: float = 0.0

    def at(self, t: float) -> TorsoState:
        return TorsoState(
            pos=np.array(self.pos, dtype=np.float64),
            quat_wxyz=quat_from_yaw(self.yaw_rad),
            lin_vel=np.zeros(3, dtype=np.float64),
            ang_vel=np.zeros(3, dtype=np.float64),
        )


@dataclass(frozen=True)
class LinearTrans(TorsoPrimitive):
    """Constant-velocity translation from ``start_pos`` along ``direction`` at ``speed_ms``.

    ``direction`` is normalized internally. The yaw is fixed (no auto-facing) — set it
    explicitly via ``yaw_rad`` if you want the robot to face the direction of travel.
    """

    start_pos: tuple[float, float, float]
    direction: tuple[float, float, float]
    speed_ms: float
    yaw_rad: float = 0.0

    def at(self, t: float) -> TorsoState:
        d = np.array(self.direction, dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            raise ValueError(f"LinearTrans direction must be non-zero; got {self.direction!r}")
        d = d / n
        vel = d * float(self.speed_ms)
        pos = np.array(self.start_pos, dtype=np.float64) + vel * float(t)
        return TorsoState(
            pos=pos,
            quat_wxyz=quat_from_yaw(self.yaw_rad),
            lin_vel=vel,
            ang_vel=np.zeros(3, dtype=np.float64),
        )


@dataclass(frozen=True)
class VerticalSine(TorsoPrimitive):
    """Sinusoidal vertical oscillation about ``mean_z`` with amplitude/freq.

    z(t)   = mean_z + amp * sin(2π f t + phase)
    vz(t)  = amp * 2π f * cos(2π f t + phase)

    XY stays at ``xy``; yaw is fixed.
    """

    mean_z: float
    amp_m: float
    freq_hz: float
    xy: tuple[float, float] = (0.0, 0.0)
    phase_rad: float = 0.0
    yaw_rad: float = 0.0

    def at(self, t: float) -> TorsoState:
        omega = 2.0 * math.pi * float(self.freq_hz)
        z = self.mean_z + self.amp_m * math.sin(omega * t + self.phase_rad)
        vz = self.amp_m * omega * math.cos(omega * t + self.phase_rad)
        return TorsoState(
            pos=np.array([self.xy[0], self.xy[1], z], dtype=np.float64),
            quat_wxyz=quat_from_yaw(self.yaw_rad),
            lin_vel=np.array([0.0, 0.0, vz], dtype=np.float64),
            ang_vel=np.zeros(3, dtype=np.float64),
        )


@dataclass(frozen=True)
class SquatDown(TorsoPrimitive):
    """Cosine-shaped drop-from-standing: starts at ``top_z``, drops ``depth_m``, returns.

    Unlike :class:`VerticalSine`, this primitive is **one-sided** — the trajectory never
    goes above ``top_z``, which matches what a real squat looks like (the torso can't
    rise above the standing height without a hop). When paired with the motion writer's
    seed-anchoring (default on), ``top_z`` is shifted to the seed clip's frame-0 torso
    pose, so the robot starts at its actual standing z and drops by ``depth_m``::

        z(t)  = top_z - 0.5 * depth_m * (1 - cos(2π f t))
              = top_z                          at t = 0, T, 2T, …
              = top_z - depth_m                at t = T/2, 3T/2, …
        vz(t) = -0.5 * depth_m * 2π f * sin(2π f t)
              = 0                              at t = 0 (starts at rest at the top)

    where T = 1/freq_hz. XY stays at ``xy``; yaw is fixed.
    """

    top_z: float
    depth_m: float
    freq_hz: float
    xy: tuple[float, float] = (0.0, 0.0)
    yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.depth_m < 0:
            raise ValueError(f"SquatDown.depth_m must be >= 0; got {self.depth_m}")
        if self.freq_hz <= 0:
            raise ValueError(f"SquatDown.freq_hz must be > 0; got {self.freq_hz}")

    def at(self, t: float) -> TorsoState:
        omega = 2.0 * math.pi * float(self.freq_hz)
        z = self.top_z - 0.5 * self.depth_m * (1.0 - math.cos(omega * t))
        vz = -0.5 * self.depth_m * omega * math.sin(omega * t)
        return TorsoState(
            pos=np.array([self.xy[0], self.xy[1], z], dtype=np.float64),
            quat_wxyz=quat_from_yaw(self.yaw_rad),
            lin_vel=np.array([0.0, 0.0, vz], dtype=np.float64),
            ang_vel=np.zeros(3, dtype=np.float64),
        )


@dataclass(frozen=True)
class Polyline3D(TorsoPrimitive):
    """Constant-speed traversal of a 3D polyline (arc-length parameterised).

    The polyline visits ``waypoints`` in order at constant ground speed ``speed_ms``.
    Velocity is the tangent vector of the current segment, scaled to ``speed_ms``.
    After the end of the polyline (when traversed arc length ≥ total length), the
    primitive holds the last waypoint with zero velocity.

    Notes
    -----
    - Quaternion is fixed identity; angular velocity is zero. Pair with
      ``anchor_yaw_to_seed=True`` in the motion writer if you want the seed's
      original facing preserved during writing.
    - At sharp corners the velocity is discontinuous (changes direction abruptly).
      That's the expected behaviour for a constant-speed polyline; if you want
      smoothed corners, sample more waypoints along a Bezier curve before passing
      them in.
    """

    waypoints: tuple[tuple[float, float, float], ...]
    speed_ms: float

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise ValueError(f"Polyline3D needs ≥2 waypoints; got {len(self.waypoints)}.")
        if self.speed_ms <= 0:
            raise ValueError(f"Polyline3D speed_ms must be > 0; got {self.speed_ms}.")

    def _arc_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return (waypoints, segment_diffs, cumulative_arc_length, total_arc_length)."""
        wp = np.array(self.waypoints, dtype=np.float64)
        diffs = wp[1:] - wp[:-1]
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        return wp, diffs, cum, float(cum[-1])

    def at(self, t: float) -> TorsoState:
        wp, diffs, cum, total = self._arc_table()
        s = float(self.speed_ms) * float(t)
        if total <= 0.0 or s >= total:
            return TorsoState(
                pos=wp[-1].copy(),
                quat_wxyz=quat_identity(),
                lin_vel=np.zeros(3, dtype=np.float64),
                ang_vel=np.zeros(3, dtype=np.float64),
            )
        seg_idx = int(np.searchsorted(cum, s, side="right") - 1)
        seg_idx = max(0, min(seg_idx, len(diffs) - 1))
        seg_len = float(cum[seg_idx + 1] - cum[seg_idx])
        u = (s - cum[seg_idx]) / seg_len if seg_len > 1e-12 else 0.0
        pos = wp[seg_idx] + u * diffs[seg_idx]
        # Velocity magnitude = speed_ms; direction = unit tangent of current segment.
        vel = (diffs[seg_idx] / max(seg_len, 1e-12)) * float(self.speed_ms)
        return TorsoState(
            pos=pos,
            quat_wxyz=quat_identity(),
            lin_vel=vel,
            ang_vel=np.zeros(3, dtype=np.float64),
        )


@dataclass(frozen=True)
class Compose(TorsoPrimitive):
    """Additive overlay of primitives (positions + linear velocities summed).

    Useful for e.g. "walk forward while squatting": stack ``LinearTrans`` + ``VerticalSine``.
    The first primitive's quaternion and angular velocity are used (the overlay
    contributes only translational additions).
    """

    primitives: tuple[TorsoPrimitive, ...]

    def __post_init__(self) -> None:
        if not self.primitives:
            raise ValueError("Compose requires at least one primitive.")

    def at(self, t: float) -> TorsoState:
        states = [p.at(t) for p in self.primitives]
        pos = sum((s.pos for s in states), start=np.zeros(3, dtype=np.float64))
        lin = sum((s.lin_vel for s in states), start=np.zeros(3, dtype=np.float64))
        return TorsoState(
            pos=pos,
            quat_wxyz=states[0].quat_wxyz,
            lin_vel=lin,
            ang_vel=states[0].ang_vel,
        )


@dataclass(frozen=True)
class Sequence(TorsoPrimitive):
    """Temporal concatenation. Segment ``i`` runs for ``durations[i]`` seconds.

    Each segment is evaluated in its own local time (t' = t - sum(prev durations)).
    The position of segment ``i`` is offset by the *end* position of segment ``i-1``
    minus segment ``i``'s *start* position, so the composed trajectory is continuous
    across segment boundaries. Quaternion/velocity discontinuities are NOT smoothed
    — callers should ensure segment endpoints are compatible (e.g. matching yaws).
    """

    segments: tuple[TorsoPrimitive, ...]
    durations_s: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.segments) != len(self.durations_s):
            raise ValueError(
                f"Sequence: segments ({len(self.segments)}) and durations ({len(self.durations_s)}) "
                "must have the same length."
            )
        if any(d <= 0 for d in self.durations_s):
            raise ValueError(f"Sequence: all durations must be > 0; got {self.durations_s!r}.")

    def _segment_index(self, t: float) -> tuple[int, float, np.ndarray]:
        """Return (segment_idx, local_t, cumulative_position_offset)."""
        elapsed = 0.0
        offset = np.zeros(3, dtype=np.float64)
        for i, d in enumerate(self.durations_s):
            if t <= elapsed + d or i == len(self.durations_s) - 1:
                return i, max(0.0, t - elapsed), offset
            end_state = self.segments[i].at(d)
            next_start_state = self.segments[i + 1].at(0.0)
            offset = offset + (end_state.pos - next_start_state.pos)
            elapsed += d
        raise AssertionError("unreachable")

    def at(self, t: float) -> TorsoState:
        idx, local_t, offset = self._segment_index(t)
        s = self.segments[idx].at(local_t)
        return TorsoState(
            pos=s.pos + offset,
            quat_wxyz=s.quat_wxyz,
            lin_vel=s.lin_vel,
            ang_vel=s.ang_vel,
        )
