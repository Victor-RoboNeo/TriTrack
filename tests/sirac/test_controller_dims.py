"""Checkpoint-loading and batched-execution tests (Isaac-free)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
import sirac_isaacfree  # noqa: E402

sirac_isaacfree.install()
sys.path.insert(0, str(ROOT / "source" / "whole_body_tracking"))

from whole_body_tracking.sirac.intent_encoder import INTENT_RAW_DIM, SparseIntentEncoder, pack_sparse_intent
from whole_body_tracking.sirac.lower_body_controller import (
    DummyHoldPolicy,
    LowerBodyRealizationController,
    find_student_jit,
    load_student_policy,
)
from whole_body_tracking.sirac.mappings import HTD_HISTORY, HTD_NUM_ACTIONS, HTD_NUM_OBS


def test_missing_jit_is_dummy():
    policy, path = load_student_policy("/no/such/student_policy_jit.pt")
    assert isinstance(policy, DummyHoldPolicy)
    assert path is None or not Path("/no/such/student_policy_jit.pt").is_file()
    assert find_student_jit("/no/such/file.pt") is None


def test_history_flat_dim():
    ctl = LowerBodyRealizationController()
    ctl.reset(8)
    obs = np.zeros((8, HTD_NUM_OBS), dtype=np.float32)
    a1 = ctl.push_and_infer(obs)
    a2 = ctl.push_and_infer(obs)
    assert a1.shape == (8, HTD_NUM_ACTIONS)
    assert a2.shape == (8, HTD_NUM_ACTIONS)
    assert ctl._hist.shape == (8, HTD_HISTORY, HTD_NUM_OBS)


def test_intent_pack_dim():
    b = 3
    pos = np.zeros((b, 3))
    quat = np.tile(np.array([1.0, 0, 0, 0]), (b, 1))
    vel = np.zeros((b, 3))
    x = pack_sparse_intent(pos, quat, pos, quat, pos, quat, vel, vel, vel)
    assert x.shape == (b, INTENT_RAW_DIM // 3)  # one horizon = 36
    enc = SparseIntentEncoder()
    traj = np.zeros((b, INTENT_RAW_DIM), dtype=np.float32)
    z = enc(traj)
    assert z.shape == (b, enc.z_dim)
    assert enc.trained is False
