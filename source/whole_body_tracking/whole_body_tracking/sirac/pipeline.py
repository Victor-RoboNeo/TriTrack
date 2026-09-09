"""Phase-1A pipeline: sparse intent → Stage-2 nominal → 7-D command → frozen LBC.

Upper body remains on the existing AnyBody sparse-intent / Stage-2 tracker.
Lower body and waist are replaced only in Baselines B and C.

This module is Isaac-free. The Isaac glue lives in ``scripts/sirac``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .adapt_interfaces import (
    CounterfactualLogger,
    PredictiveIntentGovernor,
    RealizationResidualNavigator,
)
from .command_extract import extract_realization_command
from .lower_body_controller import LowerBodyRealizationController, action_to_q_target
from .mappings import ARM_NAMES, G1_JOINT_NAMES_29, arm_indices_in, lower_indices_in


class Baseline(str, Enum):
    A_STAGE2_DIRECT = "A_stage2_direct"
    B_REALIZATION_LBC = "B_realization_lbc"
    C_STATIC_ARMS_LBC = "C_static_arms_lbc"


@dataclass
class SplitWholeBodyAction:
    """29-D joint position *targets* (rad), AnyBody/URDF name order."""

    q_target_29: np.ndarray
    lower_action_15: np.ndarray | None = None
    command_7: np.ndarray | None = None
    xi: np.ndarray | None = None


def merge_upper_from_stage2_lower_from_lbc(
    stage2_q_target_29: np.ndarray,
    lbc_q_target_15: np.ndarray,
    joint_names: tuple[str, ...] | list[str] = G1_JOINT_NAMES_29,
    *,
    static_arms: bool = False,
    static_arm_pose: np.ndarray | None = None,
) -> np.ndarray:
    """Replace lower/waist of a Stage-2 29-D target with frozen-LBC targets.

    Arm joints stay Stage-2 (Baseline B) or a static default (Baseline C).
    Mapping is *by name*, never by assumed Isaac index.
    """
    q = np.asarray(stage2_q_target_29, dtype=np.float64).copy()
    li = lower_indices_in(joint_names)
    q[..., li] = np.asarray(lbc_q_target_15, dtype=np.float64)
    if static_arms:
        ai = arm_indices_in(joint_names)
        if static_arm_pose is None:
            static_arm_pose = np.zeros(len(ARM_NAMES), dtype=np.float64)
        q[..., ai] = static_arm_pose
    return q


class SiracPhase1Controller:
    """Unified controller object used by all three baselines.

    Baseline A: pass-through Stage-2 29-D action (LBC unused).
    Baseline B/C: extract c_t from nominal kinematics, run frozen LBC, merge.
    Navigator and governor stay identity (xi = 0).
    """

    def __init__(
        self,
        baseline: Baseline = Baseline.B_REALIZATION_LBC,
        lbc: LowerBodyRealizationController | None = None,
        navigator: RealizationResidualNavigator | None = None,
        governor: PredictiveIntentGovernor | None = None,
        logger: CounterfactualLogger | None = None,
        joint_names: tuple[str, ...] = G1_JOINT_NAMES_29,
    ):
        self.baseline = baseline
        self.lbc = lbc or LowerBodyRealizationController()
        self.navigator = navigator or RealizationResidualNavigator()
        self.governor = governor or PredictiveIntentGovernor()
        self.logger = logger or CounterfactualLogger()
        self.joint_names = joint_names
        self.last_action_15 = np.zeros((1, 15), dtype=np.float32)

    def reset(self, batch: int = 1) -> None:
        self.lbc.reset(batch)
        self.last_action_15 = np.zeros((batch, 15), dtype=np.float32)
        self.logger.reset()

    def step(
        self,
        *,
        stage2_q_target_29: np.ndarray,
        pelvis_quat_w: np.ndarray,
        pelvis_lin_vel_w: np.ndarray,
        pelvis_ang_vel_w: np.ndarray,
        torso_pos_w: np.ndarray,
        torso_quat_w: np.ndarray,
        ang_vel_b: np.ndarray,
        joint_pos_29: np.ndarray,
        joint_vel_29: np.ndarray,
        interaction_residual: np.ndarray | None = None,
        sparse_intent: np.ndarray | None = None,
    ) -> SplitWholeBodyAction:
        if self.baseline is Baseline.A_STAGE2_DIRECT:
            out = SplitWholeBodyAction(q_target_29=np.asarray(stage2_q_target_29, dtype=np.float64))
            self.logger.log("nominal_realization", {"q": out.q_target_29.copy()})
            self.logger.log("actual_robot_response", {"q": out.q_target_29.copy()})
            return out

        cmd0 = extract_realization_command(
            pelvis_quat_w=pelvis_quat_w,
            pelvis_lin_vel_w=pelvis_lin_vel_w,
            pelvis_ang_vel_w=pelvis_ang_vel_w,
            torso_pos_w=torso_pos_w,
            torso_quat_w=torso_quat_w,
        )
        if cmd0.ndim == 1:
            cmd0 = cmd0[None, :]
        xi = self.navigator(None, sparse_intent, cmd0, interaction_residual)
        delta = self.navigator.command_residual(xi)
        adapted = cmd0 + delta
        cmd = self.governor.filter(adapted, cmd0)
        self.logger.log("nominal_realization", {"c": cmd0.copy()})
        self.logger.log("unguarded_adapted_realization", {"c": adapted.copy()})
        self.logger.log("guarded_adapted_realization", {"c": cmd.copy()})

        li = lower_indices_in(self.joint_names)
        jp = np.asarray(joint_pos_29, dtype=np.float64)
        jv = np.asarray(joint_vel_29, dtype=np.float64)
        if jp.ndim == 1:
            jp = jp[None, :]
            jv = jv[None, :]
        action15 = self.lbc(
            None,
            cmd,
            ang_vel_b=np.asarray(ang_vel_b, dtype=np.float64).reshape(cmd.shape[0], 3),
            root_quat_w=np.asarray(pelvis_quat_w, dtype=np.float64).reshape(cmd.shape[0], 4),
            joint_pos_lower=jp[:, li],
            joint_vel_lower=jv[:, li],
            last_action=np.broadcast_to(self.last_action_15, (cmd.shape[0], 15)).copy(),
        )
        self.last_action_15 = np.asarray(action15, dtype=np.float32)
        q_lbc = action_to_q_target(action15)
        q29 = merge_upper_from_stage2_lower_from_lbc(
            np.asarray(stage2_q_target_29, dtype=np.float64),
            q_lbc[-1] if np.asarray(stage2_q_target_29).ndim == 1 else q_lbc,
            self.joint_names,
            static_arms=(self.baseline is Baseline.C_STATIC_ARMS_LBC),
        )
        self.logger.log("actual_robot_response", {"q": np.asarray(q29).copy(), "a15": action15.copy()})
        return SplitWholeBodyAction(q_target_29=q29, lower_action_15=action15, command_7=cmd, xi=xi)
