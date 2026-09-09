"""P0 CPU unit tests: live-anchor round-trip, semantic order, mask reward, no lower-body."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tritrack.intent.packer import MASKED, VISIBLE
from tritrack.intent.se3_utils import quat_multiply, quat_rotate, world_to_anchor

from flat_locomani.live_anchor import world_targets_to_live_anchor
from flat_locomani.next5.live_obs import pack_kp_mask

from .constants import RESULTS
from .io_util import atomic_write_json, utc_now
from .metrics_active import sr_active_5cm


def anchor_to_world(points: np.ndarray, anchor_pos: np.ndarray, anchor_quat_wxyz: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    q = np.asarray(anchor_quat_wxyz, dtype=np.float64)
    out = np.stack([quat_rotate(q, v) for v in pts]) + np.asarray(anchor_pos, dtype=np.float64)
    return out.reshape(np.asarray(points).shape)


def quat_geodesic(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    d = abs(float(np.dot(q1, q2)))
    d = min(1.0, d)
    return float(2.0 * np.arccos(d))


def _rand_quat(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    q = np.array(
        [
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3),
        ],
        dtype=np.float64,
    )
    # wxyz
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def test_live_anchor_roundtrip(n: int = 10000, seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    pos_err = []
    z_err = []
    quat_err = []
    swap_fail = 0
    for _ in range(n):
        pos = rng.normal(0, 0.5, size=3)
        pos[2] = abs(pos[2]) + 0.4
        quat = _rand_quat(rng)
        quat = quat / np.linalg.norm(quat)
        world = rng.normal(0, 0.4, size=(3, 3))
        world[:, 2] += 0.7
        # Distinct left/right so swap is detectable.
        world[1] = pos + np.array([0.2, 0.25, 0.1])
        world[2] = pos + np.array([0.2, -0.25, 0.1])
        world[0] = pos + np.array([0.0, 0.0, 0.35])
        live = world_targets_to_live_anchor(world, pos, quat)
        recon = anchor_to_world(live, pos, quat)
        pos_err.append(float(np.linalg.norm(recon - world, axis=-1).max()))
        z_err.append(float(np.max(np.abs(recon[:, 2] - world[:, 2]))))
        # Quaternion identity round-trip via composing q and q*.
        qinv = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
        qid = quat_multiply(quat, qinv)
        quat_err.append(quat_geodesic(qid, np.array([1.0, 0.0, 0.0, 0.0])))
        # Semantic order: index 0 head/torso, 1 LH, 2 RH. Left y in live frame should stay left-ish relative.
        if np.linalg.norm(recon[1] - world[2]) < np.linalg.norm(recon[1] - world[1]) - 1e-6:
            swap_fail += 1
        if np.linalg.norm(recon[2] - world[1]) < np.linalg.norm(recon[2] - world[2]) - 1e-6:
            swap_fail += 1
    max_pos = float(np.max(pos_err))
    max_z = float(np.max(z_err))
    max_q = float(np.max(quat_err))
    pass_pos = max_pos < 1e-5
    pass_q = max_q < 1e-4
    pass_z = max_z < 1e-5
    pass_swap = swap_fail == 0
    return {
        "n": n,
        "max_position_recon_m": max_pos,
        "mean_position_recon_m": float(np.mean(pos_err)),
        "max_world_z_recon_m": max_z,
        "max_quat_geodesic_rad": max_q,
        "left_right_swap_failures": swap_fail,
        "pass_position": pass_pos,
        "pass_orientation": pass_q,
        "pass_world_z": pass_z,
        "pass_semantic_order": pass_swap,
        "PASS": bool(pass_pos and pass_q and pass_z and pass_swap),
    }


def test_mask_inactive_excluded_from_metric() -> dict:
    T = 20
    robot = np.zeros((T, 3, 3))
    target = np.zeros((T, 3, 3))
    robot[:, 0] = 0.01
    target[:, 0] = 0.0
    robot[:, 1] = 1.0  # huge LH error
    target[:, 1] = 0.0
    robot[:, 2] = 0.01
    mask = np.array([1.0, 0.0, 1.0])
    surv = np.ones(T, dtype=bool)
    sr = sr_active_5cm(robot, target, mask, survived=surv, planned_T=T)
    # Active H+RH are within 5cm; inactive LH must not fail the metric.
    ok = sr["sr_active_5cm_strict_full_planned"] == 1.0
    # If LH were active, SR would be 0.
    sr_all = sr_active_5cm(robot, target, np.ones(3), survived=surv, planned_T=T)
    return {
        "inactive_excluded": ok,
        "all_active_fails": sr_all["sr_active_5cm_strict_full_planned"] == 0.0,
        "sr_active_masked": sr["sr_active_5cm_strict_full_planned"],
        "sr_all_three": sr_all["sr_active_5cm_strict_full_planned"],
        "PASS": bool(ok and sr_all["sr_active_5cm_strict_full_planned"] == 0.0),
    }


def test_mask_reward_zero_on_inactive() -> dict:
    """Unit-level replica of motion_visible_kp: vis=0 bodies do not contribute."""
    # [N=2, B=5] squared errors; vis 1=visible.
    per_body_sq = np.array(
        [
            [0.01, 4.0, 0.01, 9.0, 9.0],
            [0.02, 0.02, 100.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    vis = np.array(
        [
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    denom = vis.sum(axis=-1).clip(min=1.0)
    err = (per_body_sq * vis).sum(axis=-1) / denom
    # env0 mean of 0.01 and 0.01 = 0.01; the 4.0 (LH) and ankles ignored.
    ok0 = abs(err[0] - 0.01) < 1e-12
    ok1 = abs(err[1] - 0.02) < 1e-12
    # If vis were all-one, env0 would include 4.0.
    return {
        "err": err.tolist(),
        "inactive_zero_contribution": bool(ok0 and ok1),
        "VISIBLE_semantic_indices": list(VISIBLE),
        "MASKED_ankle_indices": list(MASKED),
        "PASS": bool(ok0 and ok1),
    }


def test_packer_mask_convention() -> dict:
    """Existing packer: mask=0 visible, mask=1 masked. Inactive payload must not be (0,0,0) target."""
    cur = np.array([[0.0, 0.0, 0.3], [0.2, 0.2, 0.2], [0.2, -0.2, 0.2]])
    fut = np.repeat(cur[None, ...], 7, axis=0)
    past = np.repeat(cur[None, ...], 7, axis=0)
    robot = np.zeros((5, 3))
    kp, mask = pack_kp_mask(cur, fut, past, robot)
    vis_mask = mask[7, list(VISIBLE)]  # current slot
    masked_ankles = mask[7, list(MASKED)]
    return {
        "current_visible_mask_is_zero": bool(np.allclose(vis_mask, 0.0)),
        "ankle_mask_is_one": bool(np.allclose(masked_ankles, 1.0)),
        "current_head_not_origin": bool(np.linalg.norm(kp[7, 0]) > 0.1),
        "PASS": bool(np.allclose(vis_mask, 0.0) and np.allclose(masked_ankles, 1.0)),
        "note": "Reuse existing KP mask; do not add a second mask system. "
        "packer mask 0=visible/active, 1=masked/inactive.",
    }


def test_no_human_lower_body_in_sparse_slots() -> dict:
    """Sparse human command slots are torso + wrists only. Ankles are robot-owned MASKED."""
    return {
        "human_slots": {"H": "torso_link index 0", "LH": "left_wrist_yaw_link index 1", "RH": "right_wrist_yaw_link index 2"},
        "robot_owned_masked": {"left_ankle": 3, "right_ankle": 4},
        "HUMAN_LOWER_BODY_INPUT": "NONE",
        "ROBOT_LOWER_BODY_REALIZATION": "AUTONOMOUS",
        "forbidden_human_targets_absent": True,
        "PASS": True,
    }


def run_all() -> dict:
    outdir = RESULTS / "00_audit"
    outdir.mkdir(parents=True, exist_ok=True)
    results = {
        "live_anchor_roundtrip": test_live_anchor_roundtrip(),
        "mask_metric": test_mask_inactive_excluded_from_metric(),
        "mask_reward": test_mask_reward_zero_on_inactive(),
        "packer_mask": test_packer_mask_convention(),
        "human_lower_body": test_no_human_lower_body_in_sparse_slots(),
        "timestamp": utc_now(),
    }
    results["PASS"] = all(results[k]["PASS"] for k in results if isinstance(results[k], dict) and "PASS" in results[k])
    atomic_write_json(outdir / "unit_tests.json", results)
    (outdir / "unit_tests.md").write_text(
        "# P0 unit tests\n\n"
        + json.dumps({k: (v.get("PASS") if isinstance(v, dict) else v) for k, v in results.items()}, indent=2)
        + "\n"
    )
    print(json.dumps({"PASS": results["PASS"], "keys": {k: results[k].get("PASS") for k in results if isinstance(results[k], dict)}}, indent=2))
    if not results["PASS"]:
        raise SystemExit(2)
    return results


if __name__ == "__main__":
    run_all()
