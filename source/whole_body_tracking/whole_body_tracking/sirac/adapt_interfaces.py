"""ADAPT-ready interfaces, disabled for Phase 1.

Navigator output ``xi_t`` is 3-5 dimensional and maps to a *command residual*
``delta_c = B @ xi``, never to the 15-D joint action.

Phase 1: ``xi = 0``, ``delta_c = 0``. Do not train this module yet.

The PredictiveIntentGovernor is identity: it does not reject or clip adapted
head/hand trajectories. A future version will enforce an intent tube around
the nominal Stage-2 rollout.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NAVIGATOR_DIM = 3  # 3-5 allowed; Phase 1 uses 3 zeros
COMMAND_DIM = 7


def default_basis(dim_xi: int = NAVIGATOR_DIM) -> np.ndarray:
    """Identity embedding of xi into the first ``dim_xi`` command channels.

    Default maps residual steering onto (vx, vy, yaw_rate). Torso pose is
    left to the analytic extractor unless a later experiment changes B.
    """
    b = np.zeros((COMMAND_DIM, dim_xi), dtype=np.float64)
    for i in range(min(dim_xi, COMMAND_DIM)):
        b[i, i] = 1.0
    return b


@dataclass
class RealizationResidualNavigator:
    """Phase-1 stub: always returns xi = 0."""

    dim_xi: int = NAVIGATOR_DIM
    basis: np.ndarray = field(default_factory=default_basis)
    enabled: bool = False

    def __call__(
        self,
        proprio_history: np.ndarray | None,
        sparse_intent: np.ndarray | None,
        nominal_realization_command: np.ndarray,
        interaction_residual: np.ndarray | None = None,
    ) -> np.ndarray:
        cmd = np.asarray(nominal_realization_command, dtype=np.float64)
        batch = cmd.shape[0] if cmd.ndim == 2 else 1
        if not self.enabled:
            return np.zeros((batch, self.dim_xi), dtype=np.float64)
        raise RuntimeError("Phase 1 forbids a trained navigator; keep enabled=False")

    def command_residual(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        if xi.ndim == 1:
            return self.basis @ xi
        return xi @ self.basis.T


@dataclass
class PredictiveIntentGovernor:
    """Identity governor. Future: reject adapted realizations that leave the intent tube."""

    epsilon_pos: float = 0.05
    epsilon_rot: float = 0.15
    enabled: bool = False

    def filter(
        self,
        adapted_command: np.ndarray,
        nominal_command: np.ndarray,
        future_head_hand_dev: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self.enabled:
            return np.asarray(adapted_command, dtype=np.float64)
        raise RuntimeError("Phase 1 forbids an active governor; keep enabled=False")


COUNTERFACTUAL_BRANCHES = (
    "nominal_realization",
    "unguarded_adapted_realization",
    "guarded_adapted_realization",
    "actual_robot_response",
)


@dataclass
class CounterfactualLogger:
    """Logging hooks for four future counterfactual branches. Phase 1 records nominal + actual only."""

    records: dict = field(default_factory=lambda: {k: [] for k in COUNTERFACTUAL_BRANCHES})

    def log(self, branch: str, payload: dict) -> None:
        if branch not in self.records:
            raise KeyError(branch)
        self.records[branch].append(payload)

    def reset(self) -> None:
        self.records = {k: [] for k in COUNTERFACTUAL_BRANCHES}
