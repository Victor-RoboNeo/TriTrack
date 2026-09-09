"""Campaign metrics. Legacy SR_3PT_5CM is never renamed or replaced."""
from __future__ import annotations

import numpy as np

from flat_locomani.metrics import episode_metrics, sr_3pt_5cm


def sr_active_5cm(
    robot_xyz: np.ndarray,
    target_xyz: np.ndarray,
    mask: np.ndarray,
    survived: np.ndarray | None = None,
    planned_T: int | None = None,
) -> dict:
    """Timestep-level: all ACTIVE points simultaneously within 5 cm.

    mask: [3] or [T,3] with 1=active. Inactive points are excluded.
    Remaining planned ticks after a fall count as failure (same as legacy SR).
    """
    r = np.asarray(robot_xyz, dtype=np.float64)
    t = np.asarray(target_xyz, dtype=np.float64)
    if r.ndim == 2:
        r = r[None, ...]
        t = t[None, ...]
    n = min(r.shape[0], t.shape[0])
    r = r[:n]
    t = t[:n]
    m = np.asarray(mask, dtype=np.float64)
    if m.ndim == 1:
        m = np.broadcast_to(m.reshape(1, 3), (n, 3))
    else:
        m = m[:n]
    err = np.linalg.norm(r - t, axis=-1)  # [n,3]
    active = m > 0.5
    # If no active point at a tick, treat as success (vacuous).
    per_t_ok = np.ones(n, dtype=bool)
    any_active = active.any(axis=-1)
    worst = np.where(active, err, -np.inf)
    per_t_ok[any_active] = worst[any_active].max(axis=-1) <= 0.05
    per_t_ok[~any_active] = True
    out = {
        "sr_active_5cm_executed_prefix": float(per_t_ok.mean()) if n else float("nan"),
    }
    T = int(planned_T if planned_T is not None else n)
    full = np.zeros(T, dtype=bool)
    if survived is None:
        full[:n] = per_t_ok
        out["sr_active_5cm_strict_full_planned"] = float(full.mean()) if T else float("nan")
        return out
    surv = np.asarray(survived, dtype=bool).reshape(-1)[:n]
    full[:n] = per_t_ok & surv
    out["sr_active_5cm_strict_full_planned"] = float(full.mean()) if T else float("nan")
    return out


def point_error_stats(robot_xyz: np.ndarray, target_xyz: np.ndarray, mask: np.ndarray, survived: np.ndarray) -> dict:
    r = np.asarray(robot_xyz, dtype=np.float64)
    t = np.asarray(target_xyz, dtype=np.float64)
    n = min(len(r), len(t), len(survived))
    r, t = r[:n], t[:n]
    alive = np.asarray(survived, dtype=bool).reshape(-1)[:n]
    m = np.asarray(mask, dtype=np.float64)
    if m.ndim == 1:
        m = np.broadcast_to(m.reshape(1, 3), (n, 3))
    names = ("H", "LH", "RH")
    err = np.linalg.norm(r - t, axis=-1)
    zerr = np.abs(r[..., 2] - t[..., 2])
    out: dict = {}
    for i, name in enumerate(names):
        sel = alive & (m[:, i] > 0.5)
        e = err[sel, i] if sel.any() else np.array([])
        z = zerr[sel, i] if sel.any() else np.array([])
        out[f"{name}_rmse_xyz"] = float(np.sqrt(np.mean(e**2))) if e.size else float("nan")
        out[f"{name}_rmse_z"] = float(np.sqrt(np.mean(z**2))) if z.size else float("nan")
        out[f"{name}_p50"] = float(np.percentile(e, 50)) if e.size else float("nan")
        out[f"{name}_p95"] = float(np.percentile(e, 95)) if e.size else float("nan")
    e_all = err[alive]
    if e_all.size:
        out["p50_position_error"] = float(np.percentile(e_all, 50))
        out["p95_position_error"] = float(np.percentile(e_all, 95))
    else:
        out["p50_position_error"] = float("nan")
        out["p95_position_error"] = float("nan")
    return out


def episode_metrics_fusi(
    robot_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    survived: np.ndarray,
    phases: list[dict],
    target_height_drop_m: float,
    h0_m: float,
    hand_mode: str,
    mask: np.ndarray | None = None,
    fps: float = 50.0,
) -> dict:
    mask_arr = np.ones(3, dtype=np.float64) if mask is None else np.asarray(mask, dtype=np.float64).reshape(-1)[:3]
    base = episode_metrics(
        robot_xyz,
        target_xyz,
        survived=survived,
        phases=phases,
        target_height_drop_m=target_height_drop_m,
        h0_m=h0_m,
        hand_mode=hand_mode,
        fps=fps,
    )
    T = int(np.asarray(target_xyz).shape[0])
    active = sr_active_5cm(robot_xyz, target_xyz, mask_arr, survived=survived, planned_T=T)
    stats = point_error_stats(robot_xyz, target_xyz, mask_arr, survived)
    fall_free = bool(np.asarray(survived).reshape(-1)[:T].all()) if T else False
    return {
        **base,
        **active,
        **stats,
        "mask_H": float(mask_arr[0]),
        "mask_LH": float(mask_arr[1]),
        "mask_RH": float(mask_arr[2]),
        "fall_free_completion": float(fall_free),
        "episode_completion": float(np.asarray(survived).reshape(-1)[:T].mean()) if T else float("nan"),
    }


def mean_finite(vals) -> float:
    x = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.mean(x)) if x else float("nan")
