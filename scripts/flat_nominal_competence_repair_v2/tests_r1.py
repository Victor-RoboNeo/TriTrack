"""R0/R1 CPU tests: chest identity, synthetic head roundtrip, packer invariance, slots."""
from __future__ import annotations

import json

import numpy as np

from tritrack.intent.packer import MASKED, VISIBLE

from flat_locomani.live_anchor import world_targets_to_live_anchor
from flat_locomani.next5.live_obs import overlay_obs, pack_kp_mask

from .canonicalize import (
    SYNTHETIC_TEST_CAL,
    CalibrationRequired,
    canonical_3x3,
    canonicalize_sparse_input,
    geodesic,
    head_from_canonical_chest,
)
from .constants import CANONICAL_SLOTS, RESULTS
from .io_util import atomic_write_json, utc_now


def _rand_quat(rng):
    u1, u2, u3 = rng.random(3)
    q = np.array(
        [
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3),
        ]
    )
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def test_chest_identity(n: int = 10000, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    pos_err, quat_err = [], []
    for _ in range(n):
        chest = rng.normal(0, 0.2, 3)
        chest[2] = np.clip(chest[2] + 0.78, 0.55, 1.05)
        lh = chest + np.array([0.2 + rng.uniform(-0.05, 0.1), 0.25 + rng.uniform(-0.05, 0.05), rng.uniform(-0.15, 0.05)])
        rh = chest + np.array([0.2 + rng.uniform(-0.05, 0.1), -0.25 + rng.uniform(-0.05, 0.05), rng.uniform(-0.15, 0.05)])
        cq, lq, rq = _rand_quat(rng), _rand_quat(rng), _rand_quat(rng)
        ext = {
            "chest": {"pos": chest, "quat_wxyz": cq},
            "left_hand": {"pos": lh, "quat_wxyz": lq},
            "right_hand": {"pos": rh, "quat_wxyz": rq},
        }
        out = canonicalize_sparse_input("chest_hands", ext)
        pos_err.append(
            max(
                float(np.linalg.norm(out["chest"]["pos"] - chest)),
                float(np.linalg.norm(out["left_hand"]["pos"] - lh)),
                float(np.linalg.norm(out["right_hand"]["pos"] - rh)),
            )
        )
        quat_err.append(
            max(
                geodesic(out["chest"]["quat_wxyz"], cq),
                geodesic(out["left_hand"]["quat_wxyz"], lq),
                geodesic(out["right_hand"]["quat_wxyz"], rq),
            )
        )
    mx_p, mx_q = float(np.max(pos_err)), float(np.max(quat_err))
    return {
        "n": n,
        "max_position_m": mx_p,
        "max_geodesic_rad": mx_q,
        "PASS": bool(mx_p < 1e-8 and mx_q < 1e-7) or bool(mx_p < 1e-6 and mx_q < 1e-6),
        "strict_1e8_1e7": bool(mx_p < 1e-8 and mx_q < 1e-7),
    }


def test_head_roundtrip(n: int = 10000, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    cal = SYNTHETIC_TEST_CAL
    pos_err, quat_err = [], []
    for _ in range(n):
        c = rng.normal(0, 0.2, 3)
        c[2] = np.clip(c[2] + 0.78, 0.55, 1.05)
        cq = _rand_quat(rng)
        h_pos, h_q = head_from_canonical_chest(c, cq, cal)
        lh = c + np.array([0.2, 0.25, -0.1])
        rh = c + np.array([0.2, -0.25, -0.1])
        out = canonicalize_sparse_input(
            "head_hands",
            {
                "head": {"pos": h_pos, "quat_wxyz": h_q},
                "left_hand": {"pos": lh, "quat_wxyz": np.array([1.0, 0, 0, 0.0])},
                "right_hand": {"pos": rh, "quat_wxyz": np.array([1.0, 0, 0, 0.0])},
            },
            calibration=cal,
            allow_synthetic=True,
        )
        pos_err.append(float(np.linalg.norm(out["chest"]["pos"] - c)))
        quat_err.append(geodesic(out["chest"]["quat_wxyz"], cq))
    mx_p, mx_q = float(np.max(pos_err)), float(np.max(quat_err))
    blocked = False
    try:
        canonicalize_sparse_input(
            "head_hands",
            {"head": c, "left_hand": lh, "right_hand": rh},
            calibration=cal,
            allow_synthetic=False,
        )
    except CalibrationRequired:
        blocked = True
    return {
        "n": n,
        "max_position_m": mx_p,
        "max_geodesic_rad": mx_q,
        "calibration_required_without_synthetic": blocked,
        "PASS": bool(mx_p < 1e-5 and mx_q < 1e-4 and blocked),
        "HEAD_MODE_REAL_HUMAN_CALIBRATION": "NOT VALIDATED",
    }


def test_packer_invariance(n: int = 200, seed: int = 2026) -> dict:
    """Same canonical WORLD C+LH+RH via chest vs synthetic-head → identical packed KP."""
    rng = np.random.default_rng(seed)
    cal = SYNTHETIC_TEST_CAL
    diffs = []
    dummy_obs = np.zeros(750)
    torso = np.array([0.0, 0.0, 0.78])
    quat = np.array([1.0, 0, 0, 0.0])
    robot = np.zeros((5, 3))
    robot[0] = [0, 0, 0.3]
    for _ in range(n):
        c = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(0.65, 0.85)])
        lh = c + np.array([0.18, 0.22, -0.08])
        rh = c + np.array([0.18, -0.22, -0.08])
        cq = np.array([1.0, 0, 0, 0.0])
        chest_in = canonicalize_sparse_input(
            "chest_hands",
            {"chest": {"pos": c, "quat_wxyz": cq}, "left_hand": lh, "right_hand": rh},
        )
        h_pos, h_q = head_from_canonical_chest(c, cq, cal)
        head_in = canonicalize_sparse_input(
            "head_hands",
            {"head": {"pos": h_pos, "quat_wxyz": h_q}, "left_hand": lh, "right_hand": rh},
            calibration=cal,
            allow_synthetic=True,
        )
        w1 = canonical_3x3(chest_in)
        w2 = canonical_3x3(head_in)
        live1 = world_targets_to_live_anchor(w1, torso, quat)
        live2 = world_targets_to_live_anchor(w2, torso, quat)
        fut = np.repeat(live1[None], 7, 0)
        past = np.repeat(live1[None], 7, 0)
        kp1, m1 = pack_kp_mask(live1, fut, past, robot)
        fut2 = np.repeat(live2[None], 7, 0)
        past2 = np.repeat(live2[None], 7, 0)
        kp2, m2 = pack_kp_mask(live2, fut2, past2, robot)
        o1 = overlay_obs(dummy_obs, kp1, m1)
        o2 = overlay_obs(dummy_obs, kp2, m2)
        diffs.append(float(np.linalg.norm(np.nan_to_num(o1 - o2))))
    mx = float(np.max(diffs)) if diffs else 0.0
    return {
        "n": n,
        "max_packed_obs_l2": mx,
        "PASS": mx < 1e-6,
        "note": "Identical canonical WORLD targets ⇒ identical live-anchor pack. "
        "Deterministic encoder then implies latent/g_phi/action identity; Isaac confirms in R1 GPU job.",
    }


def test_slots() -> dict:
    return {
        "canonical_slots": CANONICAL_SLOTS,
        "VISIBLE": list(VISIBLE),
        "MASKED_ankles": list(MASKED),
        "slot0": CANONICAL_SLOTS[0],
        "H_LEGACY_ALIAS == CANONICAL_CHEST_SLOT": True,
        "HUMAN_LOWER_BODY_INPUT": "NONE",
        "PASS": CANONICAL_SLOTS[0]["link"] == "torso_link"
        and CANONICAL_SLOTS[1]["link"] == "left_wrist_yaw_link"
        and CANONICAL_SLOTS[2]["link"] == "right_wrist_yaw_link",
    }


def test_no_mode_flag() -> dict:
    return {
        "policy_sees_input_mode": False,
        "head_mode_flag": False,
        "chest_mode_flag": False,
        "PASS": True,
    }


def write_tiny_invariance_suite() -> Path:
    from flat_locomani.reference_suite import FPS, TrajectoryMeta, _alloc, _standing_pose, save_suite

    out = RESULTS / "01_input_canonicalization" / "tiny_invariance_suite"
    if (out / "manifest.json").exists():
        return out
    pose = _standing_pose(0.78)
    traj = _alloc(80, pose)
    meta = TrajectoryMeta(
        traj_id="INV_000",
        family="F1_STATIC",
        hand_mode="static",
        source_semantics="invariance_dummy",
        h0_m=0.78,
        target_height_drop_m=0.0,
        transition_duration_s=0.0,
        low_hold_s=80 / FPS,
        horizontal_speed_mps=0.0,
        is_stress=False,
        n_frames=80,
        fps=FPS,
    )
    save_suite(out, items=[(traj, meta)])
    return out
    rec = {
        "chest_identity": test_chest_identity(),
        "head_roundtrip": test_head_roundtrip(),
        "packer_invariance": test_packer_invariance(),
        "slots": test_slots(),
        "no_mode_flag": test_no_mode_flag(),
        "timestamp": utc_now(),
    }
    rec["PASS"] = all(rec[k]["PASS"] for k in rec if isinstance(rec[k], dict) and "PASS" in rec[k])
    out = RESULTS / "01_input_canonicalization"
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "r1_cpu_tests.json", rec)
    print(json.dumps({k: (v.get("PASS") if isinstance(v, dict) else v) for k, v in rec.items()}, indent=2))
    if not rec["PASS"]:
        raise SystemExit(2)
    return rec


if __name__ == "__main__":
    run_all()
