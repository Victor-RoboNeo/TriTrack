"""Unit tests for SIRAC mappings, frames, command extract, controller dims.

No Isaac Sim / Kit required. Run:

    python -m pytest tests/sirac -q
    # or
    python tests/sirac/run_all.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
import sirac_isaacfree  # noqa: E402

sirac_isaacfree.install()
sys.path.insert(0, str(ROOT / "source" / "whole_body_tracking"))

from whole_body_tracking.sirac.adapt_interfaces import (  # noqa: E402
    PredictiveIntentGovernor,
    RealizationResidualNavigator,
)
from whole_body_tracking.sirac.command_extract import (  # noqa: E402
    extract_from_rollout,
    extract_realization_command,
    world_to_anchor_yaw_pose,
)
from whole_body_tracking.sirac.frames import (  # noqa: E402
    finite_difference,
    projected_gravity_b,
    quat_mul,
    torso_in_pelvis_yaw_rpy,
    wrap_to_pi,
    yaw_frame_lin_vel_xy,
)
from whole_body_tracking.sirac.lower_body_controller import (  # noqa: E402
    LowerBodyRealizationController,
    action_to_q_target,
    build_student_obs58,
)
from whole_body_tracking.sirac.mappings import (  # noqa: E402
    ARM_NAMES,
    COMMAND_NAMES,
    G1_JOINT_NAMES_29,
    HTD_ACTION_SCALE,
    HTD_COMMAND_RANGE,
    HTD_DEFAULT_LOWER,
    HTD_NUM_ACTIONS,
    HTD_NUM_OBS,
    LOWER_WAIST_NAMES,
    arm_indices_in,
    clip_command,
    lower_indices_in,
)
from whole_body_tracking.sirac.pipeline import (  # noqa: E402
    Baseline,
    SiracPhase1Controller,
    merge_upper_from_stage2_lower_from_lbc,
)

# Identity by design: mappings.G1_JOINT_NAMES_29 is the HTD target_joint_order.
# The registry/URDF tests below re-read source files so this cannot silently drift.


def test_anybody_registry_matches_htd_order():
    registry = ROOT / "source/whole_body_tracking/whole_body_tracking/robots/robot_registry.py"
    text = registry.read_text()
    # Parse the g1 list by locating the first 29 quoted joint names after '"g1"'.
    import re

    g1_block = text.split('"g1"', 1)[1].split("h1_2", 1)[0]
    names = re.findall(r'"([a-z0-9_]+_joint)"', g1_block)
    assert names[:29] == list(G1_JOINT_NAMES_29)


def test_urdf_revolute_order_matches():
    urdf = ROOT / "source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf"
    if not urdf.is_file():
        print("skip test_urdf_revolute_order_matches: URDF not present")
        return
    import re

    names = re.findall(r'<joint name="([^"]+)" type="revolute"', urdf.read_text())
    assert names[:29] == list(G1_JOINT_NAMES_29), names[:29]


def test_name_based_remap_survives_permutation():
    perm = list(G1_JOINT_NAMES_29[::-1])
    li = lower_indices_in(perm)
    ai = arm_indices_in(perm)
    assert [perm[i] for i in li] == list(LOWER_WAIST_NAMES)
    assert [perm[i] for i in ai] == list(ARM_NAMES)


def test_command_clip_and_order():
    assert COMMAND_NAMES == (
        "lin_vel_x",
        "lin_vel_y",
        "ang_vel_z",
        "height",
        "body_roll",
        "body_pitch",
        "body_yaw",
    )
    c = np.array([10.0, -10.0, 10.0, 2.0, 2.0, -2.0, 3.0])
    out = clip_command(c, deploy=True)
    lo = [HTD_COMMAND_RANGE[n][0] for n in COMMAND_NAMES]
    hi = [HTD_COMMAND_RANGE[n][1] for n in COMMAND_NAMES]
    np.testing.assert_allclose(out, np.clip(c, lo, hi))


def test_yaw_frame_velocity_identity_when_yaw_zero():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    v = np.array([0.3, -0.1, 0.0])
    np.testing.assert_allclose(yaw_frame_lin_vel_xy(q, v), [0.3, -0.1], atol=1e-6)


def test_yaw_frame_velocity_rotates_with_heading():
    yaw = math.pi / 2
    q = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    v_w = np.array([0.4, 0.0, 0.0])  # world +x
    v_yaw = yaw_frame_lin_vel_xy(q, v_w)
    # Robot facing +y; world +x is -y in yaw frame.
    np.testing.assert_allclose(v_yaw, [0.0, -0.4], atol=1e-6)


def test_finite_difference():
    t = np.linspace(0.0, 1.0, 51)
    x = np.stack([0.2 * t, np.zeros_like(t), np.zeros_like(t)], axis=-1)
    v = finite_difference(x, 0.02)
    np.testing.assert_allclose(v[1:, 0], 0.2, atol=1e-6)
    assert v[0, 0] == 0.0


def test_relative_rpy_identity():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    rpy = torso_in_pelvis_yaw_rpy(q, q)
    np.testing.assert_allclose(rpy, [0, 0, 0], atol=1e-6)


def test_relative_yaw_separated_from_heading():
    yaw = 0.7
    q_pelvis = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    # torso same heading as pelvis → relative yaw ~ 0
    rpy = torso_in_pelvis_yaw_rpy(q_pelvis, q_pelvis)
    np.testing.assert_allclose(rpy[2], 0.0, atol=1e-6)
    # extra torso yaw of +0.2 about world z, composed after pelvis yaw
    extra = 0.2
    q_extra = np.array([math.cos(extra / 2), 0.0, 0.0, math.sin(extra / 2)])
    q_torso = quat_mul(q_pelvis, q_extra)
    rpy = torso_in_pelvis_yaw_rpy(q_torso, q_pelvis)
    np.testing.assert_allclose(rpy[2], extra, atol=1e-5)


def test_height_is_torso_z_not_pelvis():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    c = extract_realization_command(
        pelvis_quat_w=q,
        pelvis_lin_vel_w=np.zeros(3),
        pelvis_ang_vel_w=np.zeros(3),
        torso_pos_w=np.array([0.0, 0.0, 0.71]),
        torso_quat_w=q,
        clip=False,
    )
    assert c.shape == (7,)
    np.testing.assert_allclose(c[3], 0.71)


def test_extract_rollout_and_clip():
    t = 20
    dt = 0.02
    pelvis_pos = np.zeros((t, 3))
    pelvis_pos[:, 0] = np.linspace(0.0, 0.4, t)  # 1 m/s would exceed clip; this is 1 m / 0.38s ≈ 1.05 → clip
    pelvis_pos[:, 0] = 0.2 * np.arange(t) * dt  # 0.2 m/s
    pelvis_quat = np.zeros((t, 4))
    pelvis_quat[:, 0] = 1.0
    torso_pos = pelvis_pos.copy()
    torso_pos[:, 2] = 0.72
    torso_quat = pelvis_quat.copy()
    c = extract_from_rollout(pelvis_pos, pelvis_quat, torso_pos, torso_quat, dt)
    assert c.shape == (t, 7)
    np.testing.assert_allclose(c[1:, 0], 0.2, atol=1e-5)
    np.testing.assert_allclose(c[:, 3], 0.72, atol=1e-8)
    # over-height clips
    torso_pos[:, 2] = 1.5
    c2 = extract_from_rollout(pelvis_pos, pelvis_quat, torso_pos, torso_quat, dt, clip=True)
    assert np.all(c2[:, 3] == HTD_COMMAND_RANGE["height"][1])


def test_obs58_and_action_scale():
    obs = build_student_obs58(
        ang_vel_b=np.zeros(3),
        projected_gravity=np.array([0, 0, -1.0]),
        command7=np.zeros(7),
        joint_pos=HTD_DEFAULT_LOWER,
        joint_vel=np.zeros(15),
        last_action=np.zeros(15),
    )
    assert obs.shape == (HTD_NUM_OBS,)
    np.testing.assert_allclose(obs[13:28], 0.0)  # q - q_default
    q = action_to_q_target(np.ones(15) * 0.4)
    np.testing.assert_allclose(q[:14], HTD_DEFAULT_LOWER[:14] + 0.4 * HTD_ACTION_SCALE)
    # waist pitch clipped
    q_big = action_to_q_target(np.ones(15) * 10.0)
    assert q_big[14] <= 0.60 + 1e-9


def test_controller_dummy_batched():
    ctl = LowerBodyRealizationController()
    assert ctl.is_dummy
    ctl.reset(4)
    a = ctl(
        None,
        np.zeros((4, 7)),
        ang_vel_b=np.zeros((4, 3)),
        root_quat_w=np.tile(np.array([1.0, 0, 0, 0]), (4, 1)),
        joint_pos_lower=np.tile(HTD_DEFAULT_LOWER, (4, 1)),
        joint_vel_lower=np.zeros((4, 15)),
        last_action=np.zeros((4, 15)),
    )
    assert a.shape == (4, HTD_NUM_ACTIONS)
    np.testing.assert_allclose(a, 0.0)


def test_navigator_and_governor_are_identity():
    nav = RealizationResidualNavigator()
    gov = PredictiveIntentGovernor()
    cmd = np.array([[0.1, 0.0, 0.0, 0.72, 0, 0, 0]])
    xi = nav(None, None, cmd, None)
    np.testing.assert_allclose(xi, 0.0)
    np.testing.assert_allclose(nav.command_residual(xi), 0.0)
    np.testing.assert_allclose(gov.filter(cmd, cmd), cmd)


def test_pipeline_baseline_a_passthrough():
    ctl = SiracPhase1Controller(baseline=Baseline.A_STAGE2_DIRECT)
    q = np.linspace(-0.2, 0.2, 29)
    out = ctl.step(
        stage2_q_target_29=q,
        pelvis_quat_w=np.array([1.0, 0, 0, 0]),
        pelvis_lin_vel_w=np.zeros(3),
        pelvis_ang_vel_w=np.zeros(3),
        torso_pos_w=np.array([0, 0, 0.72]),
        torso_quat_w=np.array([1.0, 0, 0, 0]),
        ang_vel_b=np.zeros(3),
        joint_pos_29=np.zeros(29),
        joint_vel_29=np.zeros(29),
    )
    np.testing.assert_allclose(out.q_target_29, q)


def test_pipeline_baseline_b_replaces_lower_keeps_arms():
    ctl = SiracPhase1Controller(baseline=Baseline.B_REALIZATION_LBC)
    q = np.arange(29, dtype=np.float64) * 0.01
    out = ctl.step(
        stage2_q_target_29=q,
        pelvis_quat_w=np.array([1.0, 0, 0, 0]),
        pelvis_lin_vel_w=np.zeros(3),
        pelvis_ang_vel_w=np.zeros(3),
        torso_pos_w=np.array([0, 0, 0.72]),
        torso_quat_w=np.array([1.0, 0, 0, 0]),
        ang_vel_b=np.zeros(3),
        joint_pos_29=np.concatenate([HTD_DEFAULT_LOWER, np.zeros(14)]),
        joint_vel_29=np.zeros(29),
    )
    # dummy LBC holds HTD default on lower/waist
    np.testing.assert_allclose(out.q_target_29[:15], HTD_DEFAULT_LOWER)
    np.testing.assert_allclose(out.q_target_29[15:], q[15:])
    assert out.command_7.shape[-1] == 7


def test_pipeline_baseline_c_zeros_arms():
    ctl = SiracPhase1Controller(baseline=Baseline.C_STATIC_ARMS_LBC)
    q = np.ones(29)
    out = ctl.step(
        stage2_q_target_29=q,
        pelvis_quat_w=np.array([1.0, 0, 0, 0]),
        pelvis_lin_vel_w=np.zeros(3),
        pelvis_ang_vel_w=np.zeros(3),
        torso_pos_w=np.array([0, 0, 0.72]),
        torso_quat_w=np.array([1.0, 0, 0, 0]),
        ang_vel_b=np.zeros(3),
        joint_pos_29=np.concatenate([HTD_DEFAULT_LOWER, np.ones(14)]),
        joint_vel_29=np.zeros(29),
    )
    np.testing.assert_allclose(out.q_target_29[15:], 0.0)


def test_merge_name_based_not_index_based():
    perm = list(G1_JOINT_NAMES_29[::-1])
    stage = np.ones(29)
    lbc = HTD_DEFAULT_LOWER
    q = merge_upper_from_stage2_lower_from_lbc(stage, lbc, perm)
    li = lower_indices_in(perm)
    np.testing.assert_allclose(q[li], HTD_DEFAULT_LOWER)


def test_projected_gravity_upright():
    g = projected_gravity_b(np.array([1.0, 0, 0, 0]))
    np.testing.assert_allclose(g, [0, 0, -1], atol=1e-6)


def test_wrap_to_pi():
    np.testing.assert_allclose(wrap_to_pi(np.array([math.pi + 0.1])), [-math.pi + 0.1], atol=1e-6)


def test_anchor_yaw_relative_position():
    yaw = math.pi / 2
    q_pelvis = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    pelvis_pos = np.array([1.0, 2.0, 0.0])
    # world +x offset from pelvis; robot facing +y → yaw-frame -y
    pos_w = pelvis_pos + np.array([0.3, 0.0, 0.0])
    pos_yaw, _ = world_to_anchor_yaw_pose(pos_w, q_pelvis, q_pelvis, pelvis_pos)
    np.testing.assert_allclose(pos_yaw, [0.0, -0.3, 0.0], atol=1e-6)
