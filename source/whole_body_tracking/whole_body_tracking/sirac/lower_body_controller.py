"""Frozen HTD lower-body realization controller.

Conceptual interface (Phase 1A):

    lower_action = controller(proprio_history, realization_command)

``realization_command`` is robot-owned. It is not a joystick or extra human
channel. Human-owned intent remains head / left-hand / right-hand SE(3).

Student observation (verified from ``deploy/deploy_student_htd.py``):

    [ang_vel(3), gravity(3), command(7), joint_pos(15), joint_vel(15), last_action(15)]
    history_length = 2  →  116-D input

Action:

    q_target = q_default_htd + clip(action, ±100) * 0.25

Joint positions in the observation are offsets from *HTD* defaults, not
AnyBody Beyond-Mimic defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .frames import projected_gravity_b
from .mappings import (
    HTD_ACTION_SCALE,
    HTD_DEFAULT_LOWER,
    HTD_HISTORY,
    HTD_NUM_ACTIONS,
    HTD_NUM_OBS,
    WAIST_PITCH_LIMITS,
    clip_command,
)

# Search order for the public G1-flat student JIT. None of these are required
# for unit tests; DummyHoldPolicy is the CI fallback.
DEFAULT_JIT_CANDIDATES = (
    Path("/data/home/chenxiangyu/robotics/IsaacLab-Decoupled-WBC/example/student_checkpoints/student_policy_jit.pt"),
    Path("/data/home/chenxiangyu/robotics/IsaacLab-Decoupled-WBC/deploy/policy/g1_student/student_policy_jit.pt"),
    Path("/data/home/chenxiangyu/robotics/humanoid-touch-dream/htd_wbc/isaaclab_decoupled_wbc/example/student_checkpoints/student_policy_jit.pt"),
    Path("/data/home/chenxiangyu/robotics/Anybody/vendor/htd/student_policy_jit.pt"),
)


class DummyHoldPolicy:
    """Deterministic placeholder: outputs zeros → hold HTD default pose.

    Used when the JIT is absent. Smoke tests must not treat this as evidence
    that the realization bottleneck works.
    """

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        batch = obs.shape[0] if obs.ndim == 2 else 1
        return np.zeros((batch, HTD_NUM_ACTIONS), dtype=np.float32)


def find_student_jit(extra: str | Path | None = None) -> Path | None:
    paths = []
    if extra is not None:
        paths.append(Path(extra))
    paths.extend(DEFAULT_JIT_CANDIDATES)
    for p in paths:
        if p.is_file():
            return p
    return None


def load_student_policy(jit_path: str | Path | None = None) -> tuple[Any, Path | None]:
    path = find_student_jit(jit_path)
    if path is None:
        return DummyHoldPolicy(), None
    try:
        import torch
    except ImportError:
        return DummyHoldPolicy(), path
    try:
        policy = torch.jit.load(str(path), map_location="cpu")
        policy.eval()
        return policy, path
    except Exception:
        return DummyHoldPolicy(), path


def build_student_obs58(
    *,
    ang_vel_b: np.ndarray,
    projected_gravity: np.ndarray,
    command7: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    last_action: np.ndarray,
    default_joint_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble one 58-D student observation (no feet contact)."""
    default_joint_pos = HTD_DEFAULT_LOWER if default_joint_pos is None else np.asarray(default_joint_pos)
    q_err = np.asarray(joint_pos, dtype=np.float64) - default_joint_pos
    pieces = [
        np.asarray(ang_vel_b, dtype=np.float64),
        np.asarray(projected_gravity, dtype=np.float64),
        np.asarray(command7, dtype=np.float64),
        q_err,
        np.asarray(joint_vel, dtype=np.float64),
        np.asarray(last_action, dtype=np.float64),
    ]
    obs = np.concatenate(pieces, axis=-1)
    if obs.shape[-1] != HTD_NUM_OBS:
        raise ValueError(f"student obs dim {obs.shape[-1]} != {HTD_NUM_OBS}")
    return obs.astype(np.float32)


def action_to_q_target(action: np.ndarray, default: np.ndarray | None = None, *, clip_waist_pitch: bool = True) -> np.ndarray:
    default = HTD_DEFAULT_LOWER if default is None else np.asarray(default)
    a = np.clip(np.asarray(action, dtype=np.float64), -100.0, 100.0)
    q = default + a * HTD_ACTION_SCALE
    if clip_waist_pitch:
        lo, hi = WAIST_PITCH_LIMITS
        q = np.array(q, copy=True)
        q[..., 14] = np.clip(q[..., 14], lo, hi)
    return q


@dataclass
class LowerBodyRealizationController:
    """Frozen wrapper around the HTD G1-flat student.

    ``__call__(proprio_history, realization_command) -> lower_action``
    """

    history_length: int = HTD_HISTORY
    jit_path: str | Path | None = None
    clip_command_deploy: bool = True
    policy: Any = field(init=False)
    loaded_path: Path | None = field(init=False, default=None)
    _hist: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.policy, self.loaded_path = load_student_policy(self.jit_path)
        self.reset()

    @property
    def is_dummy(self) -> bool:
        return isinstance(self.policy, DummyHoldPolicy)

    def reset(self, batch: int = 1) -> None:
        self._hist = np.zeros((batch, self.history_length, HTD_NUM_OBS), dtype=np.float32)

    def push_and_infer(self, obs58: np.ndarray) -> np.ndarray:
        obs58 = np.asarray(obs58, dtype=np.float32)
        if obs58.ndim == 1:
            obs58 = obs58[None, :]
        b = obs58.shape[0]
        if self._hist is None or self._hist.shape[0] != b:
            self.reset(b)
        self._hist = np.concatenate([self._hist[:, 1:], obs58[:, None, :]], axis=1)
        flat = self._hist.reshape(b, -1)
        return self._forward(flat)

    def _forward(self, flat_hist: np.ndarray) -> np.ndarray:
        if self.is_dummy:
            return self.policy(flat_hist)
        import torch

        with torch.no_grad():
            x = torch.from_numpy(np.asarray(flat_hist, dtype=np.float32))
            y = self.policy(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            return np.asarray(y.detach().cpu().numpy(), dtype=np.float32)

    def __call__(
        self,
        proprio_history: np.ndarray | None,
        realization_command: np.ndarray,
        *,
        ang_vel_b: np.ndarray | None = None,
        root_quat_w: np.ndarray | None = None,
        joint_pos_lower: np.ndarray | None = None,
        joint_vel_lower: np.ndarray | None = None,
        last_action: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return 15-D HTD actions.

        Two calling conventions:

        1. ``proprio_history`` already a (B, H, 58) or (B, H*58) buffer that
           includes the *current* command in each frame — used as-is.
        2. Pieces: ang_vel_b, root_quat_w, joint_pos/vel lower, last_action,
           plus ``realization_command``. History is maintained internally.
        """
        cmd = clip_command(np.asarray(realization_command, dtype=np.float64), deploy=self.clip_command_deploy)
        if cmd.ndim == 1:
            cmd = cmd[None, :]
        if proprio_history is not None:
            h = np.asarray(proprio_history, dtype=np.float32)
            if h.ndim == 3:
                h = h.reshape(h.shape[0], -1)
            elif h.ndim == 1:
                h = h[None, :]
            return self._forward(h)
        assert ang_vel_b is not None and root_quat_w is not None
        assert joint_pos_lower is not None and joint_vel_lower is not None
        if last_action is None:
            last_action = np.zeros((cmd.shape[0], HTD_NUM_ACTIONS), dtype=np.float32)
        grav = projected_gravity_b(root_quat_w)
        obs = build_student_obs58(
            ang_vel_b=ang_vel_b,
            projected_gravity=grav,
            command7=cmd,
            joint_pos=joint_pos_lower,
            joint_vel=joint_vel_lower,
            last_action=last_action,
        )
        return self.push_and_infer(obs)
