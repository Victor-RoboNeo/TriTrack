"""SIRAC Phase-1 metrics. Machine-readable JSON/CSV writers.

Do not hide head/hand tracking degradation behind locomotion success.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .frames import quat_conjugate, quat_mul, quat_to_rpy_wxyz, wrap_to_pi


def geodesic_rotation_error(q_pred: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Angle (rad) of q_ref^{-1} * q_pred."""
    rel = quat_mul(quat_conjugate(q_ref), q_pred)
    w = np.clip(np.abs(rel[..., 0]), 0.0, 1.0)
    return 2.0 * np.arccos(w)


def pos_error(p_pred: np.ndarray, p_ref: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(p_pred) - np.asarray(p_ref), axis=-1)


@dataclass
class EpisodeMetrics:
    head_pos_err_m: float = float("nan")
    head_rot_err_rad: float = float("nan")
    left_hand_pos_err_m: float = float("nan")
    left_hand_rot_err_rad: float = float("nan")
    right_hand_pos_err_m: float = float("nan")
    right_hand_rot_err_rad: float = float("nan")
    success: float = float("nan")
    fell: float = float("nan")
    sr_at_5cm: float = float("nan")
    foot_slip_m: float = float("nan")
    contact_impact_n: float = float("nan")
    command_track_err: float = float("nan")
    actuator_sat_rate: float = float("nan")
    action_smoothness: float = float("nan")
    com_stability: float = float("nan")
    d_intent: float = float("nan")
    d_intent_excess: float = float("nan")
    runtime_s: float = float("nan")
    policy_latency_ms: float = float("nan")
    baseline: str = ""
    seed: int = -1
    n_steps: int = 0
    notes: str = ""


def intent_deviation(
    head_pos, head_quat, left_pos, left_quat, right_pos, right_quat, ref: dict
) -> float:
    """Scalar D_intent: mean pos (m) + 0.2 * mean rot (rad) over three bodies."""
    e_p = np.mean(
        [
            pos_error(head_pos, ref["head_pos"]),
            pos_error(left_pos, ref["left_pos"]),
            pos_error(right_pos, ref["right_pos"]),
        ]
    )
    e_r = np.mean(
        [
            geodesic_rotation_error(head_quat, ref["head_quat"]),
            geodesic_rotation_error(left_quat, ref["left_quat"]),
            geodesic_rotation_error(right_quat, ref["right_quat"]),
        ]
    )
    return float(e_p + 0.2 * e_r)


def action_smoothness(actions: np.ndarray, dt: float = 0.02) -> float:
    a = np.asarray(actions, dtype=np.float64)
    if a.shape[0] < 2:
        return 0.0
    da = np.diff(a, axis=0) / dt
    return float(np.mean(np.linalg.norm(da, axis=-1)))


def command_tracking_error(cmd: np.ndarray, measured: np.ndarray) -> float:
    d = np.asarray(cmd) - np.asarray(measured)
    # unwrap yaw-like last three? they are already wrap_to_pi in extractor
    d[..., 4:] = wrap_to_pi(d[..., 4:])
    return float(np.mean(np.linalg.norm(d, axis=-1)))


def write_json(path: Path, rows: list[EpisodeMetrics] | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, dict):
        payload = rows
    else:
        payload = [asdict(r) for r in rows]
    path.write_text(json.dumps(payload, indent=2))


def write_csv(path: Path, rows: list[EpisodeMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
