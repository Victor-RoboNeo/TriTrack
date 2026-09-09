"""SE(3) canonicalization: Head+Hands / Chest+Hands → Chest+LH+RH.

Policy never sees input_mode. Canonical slots:
  C  = torso_link (H_LEGACY_ALIAS)
  LH = left_wrist_yaw_link
  RH = right_wrist_yaw_link

Head→Chest uses an explicit T_H_C. Real-human calibration is NOT validated;
a synthetic T_H_C exists only for unit tests. Training/eval use canonical chest.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tritrack.intent.se3_utils import quat_conjugate, quat_multiply, quat_rotate, quat_rotate_inverse


IDENT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _as_pose(p) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(p, dict):
        pos = np.asarray(p["pos"] if "pos" in p else p["position"], dtype=np.float64).reshape(3)
        quat = np.asarray(p.get("quat_wxyz") or p.get("quat") or IDENT_QUAT, dtype=np.float64).reshape(4)
    elif isinstance(p, (tuple, list)) and len(p) == 2:
        pos = np.asarray(p[0], dtype=np.float64).reshape(3)
        quat = np.asarray(p[1], dtype=np.float64).reshape(4)
    else:
        arr = np.asarray(p, dtype=np.float64)
        if arr.shape == (3,):
            pos, quat = arr, IDENT_QUAT.copy()
        elif arr.shape == (7,):
            pos, quat = arr[:3], arr[3:]
        else:
            raise ValueError(f"unrecognized pose shape {arr.shape}")
    n = float(np.linalg.norm(quat))
    quat = quat / max(n, 1e-12)
    return pos, quat


def compose_se3(pos_a, quat_a, pos_b, quat_b) -> tuple[np.ndarray, np.ndarray]:
    """T_W_C = T_W_H @ T_H_C. pos/quat are WORLD←frame and child-in-parent."""
    qa = np.asarray(quat_a, dtype=np.float64)
    qb = np.asarray(quat_b, dtype=np.float64)
    pa = np.asarray(pos_a, dtype=np.float64)
    pb = np.asarray(pos_b, dtype=np.float64)
    quat = quat_multiply(qa, qb)
    pos = pa + quat_rotate(qa, pb)
    n = float(np.linalg.norm(quat))
    return pos, quat / max(n, 1e-12)


def invert_se3(pos, quat) -> tuple[np.ndarray, np.ndarray]:
    qinv = quat_conjugate(np.asarray(quat, dtype=np.float64))
    pinv = -quat_rotate(qinv, np.asarray(pos, dtype=np.float64))
    return pinv, qinv


def geodesic(q1: np.ndarray, q2: np.ndarray) -> float:
    a = q1 / (np.linalg.norm(q1) + 1e-12)
    b = q2 / (np.linalg.norm(q2) + 1e-12)
    d = min(1.0, abs(float(np.dot(a, b))))
    return float(2.0 * np.arccos(d))


@dataclass
class Calibration:
    """T_H_C: transform of canonical chest expressed in the human head frame."""

    translation_xyz: np.ndarray
    quaternion_wxyz: np.ndarray
    status: str
    synthetic_test_only: bool
    source: str

    @property
    def T_H_C(self) -> tuple[np.ndarray, np.ndarray]:
        return self.translation_xyz, self.quaternion_wxyz


class CalibrationRequired(RuntimeError):
    pass


def load_calibration(path: Path | None = None) -> Calibration | None:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "results/flat_nominal_competence_repair_v2/configs/head_to_chest_calibration.yaml"
        alt = Path("/data/home/chenxiangyu/robotics/Anybody/results/flat_nominal_competence_repair_v2/configs/head_to_chest_calibration.yaml")
        path = path if path.exists() else alt
    if path is None or not Path(path).exists():
        return None
    text = Path(path).read_text()
    # tiny yaml subset
    kv: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        kv[k.strip()] = v.strip().strip('"').strip("'")
    trans = kv.get("translation_xyz", "[0, 0, 0]")
    nums = [float(x) for x in trans.replace("[", "").replace("]", "").split(",") if x.strip()]
    quat = kv.get("quaternion_wxyz", "[1, 0, 0, 0]")
    qn = [float(x) for x in quat.replace("[", "").replace("]", "").split(",") if x.strip()]
    syn = kv.get("synthetic_test_only", "true").lower() in ("true", "1", "yes")
    return Calibration(
        translation_xyz=np.array(nums, dtype=np.float64),
        quaternion_wxyz=np.array(qn, dtype=np.float64),
        status=kv.get("status", "NOT_VALIDATED"),
        synthetic_test_only=syn,
        source=str(path),
    )


SYNTHETIC_TEST_CAL = Calibration(
    translation_xyz=np.array([0.0, 0.0, -0.22], dtype=np.float64),
    quaternion_wxyz=IDENT_QUAT.copy(),
    status="SYNTHETIC_TEST_ONLY",
    synthetic_test_only=True,
    source="unit-test T_H_C; NOT a real-human calibration",
)


def canonicalize_sparse_input(
    input_mode: str,
    external_targets: dict,
    calibration: Calibration | None = None,
    *,
    allow_synthetic: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Return canonical {chest,left_hand,right_hand} with pos/quat_wxyz.

    input_mode in {head_hands, chest_hands}. Never written into policy obs.
    """
    mode = str(input_mode)
    if mode not in ("head_hands", "chest_hands"):
        raise ValueError(mode)
    lh_pos, lh_q = _as_pose(external_targets["left_hand"] if "left_hand" in external_targets else external_targets["LH"])
    rh_pos, rh_q = _as_pose(external_targets["right_hand"] if "right_hand" in external_targets else external_targets["RH"])
    if mode == "chest_hands":
        c_pos, c_q = _as_pose(external_targets["chest"] if "chest" in external_targets else external_targets["C"])
        return {
            "chest": {"pos": c_pos, "quat_wxyz": c_q},
            "left_hand": {"pos": lh_pos, "quat_wxyz": lh_q},
            "right_hand": {"pos": rh_pos, "quat_wxyz": rh_q},
        }
    h_pos, h_q = _as_pose(external_targets["head"] if "head" in external_targets else external_targets["H"])
    cal = calibration
    if cal is None:
        cal = load_calibration()
    if cal is None or (cal.synthetic_test_only and not allow_synthetic):
        raise CalibrationRequired(
            "HEAD_MODE requires a validated T_H_C. Real-human calibration is NOT VALIDATED. "
            "Pass allow_synthetic=True only for unit tests, or run scripts/calibrate_head_to_chest.py."
        )
    t_hc, q_hc = cal.T_H_C
    c_pos, c_q = compose_se3(h_pos, h_q, t_hc, q_hc)
    return {
        "chest": {"pos": c_pos, "quat_wxyz": c_q},
        "left_hand": {"pos": lh_pos, "quat_wxyz": lh_q},
        "right_hand": {"pos": rh_pos, "quat_wxyz": rh_q},
    }


def canonical_3x3(cano: dict) -> np.ndarray:
    return np.stack(
        [cano["chest"]["pos"], cano["left_hand"]["pos"], cano["right_hand"]["pos"]],
        axis=0,
    )


def head_from_canonical_chest(chest_pos, chest_quat, cal: Calibration) -> tuple[np.ndarray, np.ndarray]:
    """T_W_H = T_W_C @ inverse(T_H_C) for synthetic roundtrip tests."""
    t_inv, q_inv = invert_se3(*cal.T_H_C)
    return compose_se3(chest_pos, chest_quat, t_inv, q_inv)
