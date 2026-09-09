"""Morphology-invariant analytic 3-point adapter. No joints, no SMPL."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import CALIB_S, OUT_HZ


def _finite(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def heading_axes(l0: np.ndarray, r0: np.ndarray) -> np.ndarray:
    """Columns = (forward, left, up) in world. Isaac Z-up."""
    up = np.array([0.0, 0.0, 1.0])
    left = l0 - r0
    n = np.linalg.norm(left)
    if n < 1e-6:
        left = np.array([0.0, 1.0, 0.0])
    else:
        left = left / n
    left = left - up * float(left @ up)
    ln = np.linalg.norm(left)
    left = left / ln if ln > 1e-8 else np.array([0.0, 1.0, 0.0])
    fwd = np.cross(left, up)
    fn = np.linalg.norm(fwd)
    if fn < 1e-8:
        fwd = np.array([1.0, 0.0, 0.0])
        left = np.cross(up, fwd)
    else:
        fwd = fwd / fn
    left = np.cross(up, fwd)
    return np.stack([fwd, left, up], axis=1)  # 3x3, columns


def world_to_canon(p: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    rel = _finite(p) - origin.reshape(1, 3)
    return rel @ axes  # because axes columns are world basis of canon axes


def canon_to_world(p: np.ndarray, origin: np.ndarray, axes: np.ndarray) -> np.ndarray:
    return origin.reshape(1, 3) + _finite(p) @ axes.T


@dataclass
class Calib:
    origin: np.ndarray  # [3] world
    axes: np.ndarray  # [3,3] columns fwd,left,up
    h0: np.ndarray
    l0: np.ndarray
    r0: np.ndarray
    s: float
    rest_l: np.ndarray  # (L0-H0) in canon
    rest_r: np.ndarray


def calibrate(hlr: np.ndarray, fps: float, calib_s: float = CALIB_S) -> Calib:
    """hlr [T,3,3] Head,Left,Right world (Isaac)."""
    hlr = _finite(hlr)
    n = max(int(round(calib_s * fps)), 1)
    n = min(n, hlr.shape[0])
    win = hlr[:n]
    h0 = np.median(win[:, 0], axis=0)
    l0 = np.median(win[:, 1], axis=0)
    r0 = np.median(win[:, 2], axis=0)
    axes = heading_axes(l0, r0)
    spans = 0.5 * (np.linalg.norm(win[:, 1] - win[:, 0], axis=-1) + np.linalg.norm(win[:, 2] - win[:, 0], axis=-1))
    s = float(np.median(spans))
    s = max(s, 1e-4)
    rest_l = world_to_canon((l0 - h0).reshape(1, 3), np.zeros(3), axes)[0]
    rest_r = world_to_canon((r0 - h0).reshape(1, 3), np.zeros(3), axes)[0]
    return Calib(origin=h0, axes=axes, h0=h0, l0=l0, r0=r0, s=s, rest_l=rest_l, rest_r=rest_r)


def to_u(hlr: np.ndarray, calib: Calib) -> np.ndarray:
    """Dimensionless [T,9] = [ΔH, ΔL_rel, ΔR_rel] / s in heading frame."""
    h = world_to_canon(hlr[:, 0], calib.origin, calib.axes)
    l = world_to_canon(hlr[:, 1], calib.origin, calib.axes)
    r = world_to_canon(hlr[:, 2], calib.origin, calib.axes)
    dh = h / calib.s
    dl = ((l - h) - calib.rest_l) / calib.s
    dr = ((r - h) - calib.rest_r) / calib.s
    return np.concatenate([dh, dl, dr], axis=-1)


def from_u(u: np.ndarray, calib: Calib, s_out: float | None = None) -> np.ndarray:
    """u [T,9] -> world [T,3,3] using this calib's frame. s_out defaults to calib.s."""
    s = float(calib.s if s_out is None else s_out)
    u = _finite(u).reshape(-1, 9)
    dh, dl, dr = u[:, 0:3], u[:, 3:6], u[:, 6:9]
    h = dh * s
    l = h + calib.rest_l * (s / calib.s) * calib.s / calib.s
    # rest stored in canon meters relative to calib.s human. For robot, rest is robot rest in meters.
    # Caller should pass a robot Calib with robot rest_l/rest_r and s_out=s_G1.
    l = h + calib.rest_l + dl * s
    r = h + calib.rest_r + dr * s
    pts = np.stack([h, l, r], axis=1)
    out = np.empty_like(pts)
    for i in range(3):
        out[:, i] = canon_to_world(pts[:, i], calib.origin, calib.axes)
    return out


def analytic_map(u_h: np.ndarray, robot: Calib, s_g1: float) -> np.ndarray:
    """Human u [T,9] -> robot world [T,3,3] with fixed robot scale s_g1 and robot rest/frame."""
    u = _finite(u_h).reshape(-1, 9)
    dh, dl, dr = u[:, 0:3], u[:, 3:6], u[:, 6:9]
    t = dh * s_g1
    wl = t + robot.rest_l + dl * s_g1
    wr = t + robot.rest_r + dr * s_g1
    pts = np.stack([t, wl, wr], axis=1)
    out = np.empty_like(pts)
    for i in range(3):
        out[:, i] = canon_to_world(pts[:, i], robot.origin, robot.axes)
    return out


def naive_affine_map(hlr: np.ndarray, h_cal: Calib, r_cal: Calib, s_g1: float) -> np.ndarray:
    """Fixed-scale affine: each point independently (p-p0)*s_g1/s_h, then robot frame."""
    scale = s_g1 / max(h_cal.s, 1e-4)
    rel = hlr - np.stack([h_cal.h0, h_cal.l0, h_cal.r0])[None]
    # into human canon then out robot
    canon = np.empty_like(rel)
    for i in range(3):
        canon[:, i] = world_to_canon(hlr[:, i], h_cal.origin, h_cal.axes)
    c0 = np.stack(
        [
            world_to_canon(h_cal.h0[None], h_cal.origin, h_cal.axes)[0],
            world_to_canon(h_cal.l0[None], h_cal.origin, h_cal.axes)[0],
            world_to_canon(h_cal.r0[None], h_cal.origin, h_cal.axes)[0],
        ]
    )
    mapped = (canon - c0[None]) * scale
    r0 = np.stack(
        [
            np.zeros(3),
            r_cal.rest_l,
            r_cal.rest_r,
        ]
    )
    pts = mapped + r0[None]
    out = np.empty_like(pts)
    for i in range(3):
        out[:, i] = canon_to_world(pts[:, i], r_cal.origin, r_cal.axes)
    return out


def resample_traj(p: np.ndarray, fps_in: float, fps_out: float = OUT_HZ) -> np.ndarray:
    """p [T,...] linear resample to fps_out, duration preserved."""
    t_in = p.shape[0]
    if t_in < 2:
        return p.copy()
    dur = (t_in - 1) / max(fps_in, 1e-8)
    t_out = max(int(round(dur * fps_out)) + 1, 2)
    x_in = np.linspace(0.0, dur, t_in)
    x_out = np.linspace(0.0, dur, t_out)
    flat = p.reshape(t_in, -1)
    out = np.stack([np.interp(x_out, x_in, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return out.reshape((t_out,) + p.shape[1:])


def align_pair(h: np.ndarray, fps_h: float, g: np.ndarray, fps_g: float, dur_tol: float = 0.08):
    """Resample both to 50Hz on shared duration = min. Returns (h50, g50, info) or raises."""
    dur_h = (h.shape[0] - 1) / max(fps_h, 1e-8)
    dur_g = (g.shape[0] - 1) / max(fps_g, 1e-8)
    rel = abs(dur_h - dur_g) / max(max(dur_h, dur_g), 1e-8)
    if rel > dur_tol:
        raise ValueError(f"duration mismatch {dur_h:.3f} vs {dur_g:.3f} rel={rel:.3f}")
    dur = min(dur_h, dur_g)
    t = max(int(round(dur * OUT_HZ)) + 1, 2)
    x = np.linspace(0.0, dur, t)

    def rs(p, fps):
        n = p.shape[0]
        xin = np.linspace(0.0, (n - 1) / max(fps, 1e-8), n)
        flat = p.reshape(n, -1)
        out = np.stack([np.interp(x, xin, flat[:, j]) for j in range(flat.shape[1])], axis=1)
        return out.reshape((t,) + p.shape[1:])

    return rs(h, fps_h), rs(g, fps_g), {"dur_h": dur_h, "dur_g": dur_g, "dur": dur, "rel": rel, "T": t, "fps": OUT_HZ}


def pairwise_d(p: np.ndarray) -> np.ndarray:
    """p [T,3,3] -> [T,3] distances TL, TR, LR."""
    t, l, r = p[:, 0], p[:, 1], p[:, 2]
    return np.stack(
        [np.linalg.norm(l - t, axis=-1), np.linalg.norm(r - t, axis=-1), np.linalg.norm(l - r, axis=-1)],
        axis=-1,
    )
