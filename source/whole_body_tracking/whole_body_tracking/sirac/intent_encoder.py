"""Phase 1B optional sparse-intent encoder (compact MLP / TCN).

Predicts upcoming *dynamic effects of the upper body* in the pelvis-yaw
frame. Does not replace the existing AnyBody head/hand tracker.

Student inputs allowed at inference:
    proprioceptive history, sparse head/hand intent, realization command,
    previous action.

Forbidden at inference: terrain identity, terrain class routing.

The encoder is untrained in Phase 1A. ``IntentConditionedLBC`` falls back to
the frozen HTD student and ignores ``z_intent`` until a checkpoint is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# User-requested horizons; AnyBody log-spaced slots include 0.20 s and 0.40 s
# at 50 Hz (offsets 10 and 20 in SLOT_OFFSETS).
INTENT_HORIZONS_S = (0.0, 0.2, 0.4)
BODIES = ("head", "left_hand", "right_hand")
# Per body, per horizon: pos(3) + rot6d(6) + lin_vel(3) = 12. 3 bodies * 3 horizons = 108.
PER_BODY_DIM = 12
INTENT_RAW_DIM = PER_BODY_DIM * 3 * 3
DEFAULT_Z_DIM = 16


def rot6d_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Zhou et al. rotation-6D from wxyz quaternion (first two columns of R)."""
    from .frames import quat_apply

    q = np.asarray(q, dtype=np.float64)
    x = quat_apply(q, np.broadcast_to(np.array([1.0, 0.0, 0.0]), q[..., :3].shape))
    y = quat_apply(q, np.broadcast_to(np.array([0.0, 1.0, 0.0]), q[..., :3].shape))
    return np.concatenate([x, y], axis=-1)


def pack_sparse_intent(
    head_pos_yaw: np.ndarray,
    head_quat_yaw: np.ndarray,
    left_pos_yaw: np.ndarray,
    left_quat_yaw: np.ndarray,
    right_pos_yaw: np.ndarray,
    right_quat_yaw: np.ndarray,
    head_lin_vel_yaw: np.ndarray,
    left_lin_vel_yaw: np.ndarray,
    right_lin_vel_yaw: np.ndarray,
) -> np.ndarray:
    """Pack one horizon of head/hand relative pose+vel → 36-D.

    Shapes: each pos/vel (..., 3), each quat (..., 4). Leading dims shared.
    """
    chunks = []
    for pos, quat, vel in (
        (head_pos_yaw, head_quat_yaw, head_lin_vel_yaw),
        (left_pos_yaw, left_quat_yaw, left_lin_vel_yaw),
        (right_pos_yaw, right_quat_yaw, right_lin_vel_yaw),
    ):
        chunks.extend([pos, rot6d_from_quat_wxyz(quat), vel])
    return np.concatenate(chunks, axis=-1)


class SparseIntentEncoder:
    """Compact MLP. Untrained weights are identity-scale noise; do not deploy as-is."""

    def __init__(self, in_dim: int = INTENT_RAW_DIM, z_dim: int = DEFAULT_Z_DIM, hidden: int = 128, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.02, size=(in_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = rng.normal(0.0, 0.02, size=(hidden, z_dim)).astype(np.float32)
        self.b2 = np.zeros(z_dim, dtype=np.float32)
        self.z_dim = z_dim
        self.trained = False

    def __call__(self, head_hand_trajectory: np.ndarray) -> np.ndarray:
        x = np.asarray(head_hand_trajectory, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        h = np.tanh(x @ self.w1 + self.b1)
        z = h @ self.w2 + self.b2
        return z if head_hand_trajectory.ndim == 2 else z[0]


@dataclass
class IntentConditionedLBC:
    """Phase 1B wrapper. Until trained, ignores z_intent and calls the frozen LBC."""

    frozen_lbc: object
    encoder: SparseIntentEncoder | None = None
    use_intent: bool = False
    proprio_history_s: float = 0.32  # 200-400 ms request; 16 frames @ 50 Hz

    def __call__(self, proprio_history, realization_command, z_intent=None, **kwargs):
        if self.use_intent and z_intent is not None:
            raise RuntimeError(
                "Intent-conditioned LBC has no trained student yet. "
                "Keep use_intent=False for Phase 1A frozen transplant."
            )
        return self.frozen_lbc(proprio_history, realization_command, **kwargs)
