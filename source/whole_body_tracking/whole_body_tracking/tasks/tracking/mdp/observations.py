from __future__ import annotations

import torch
import warnings
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_rotate_inverse

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_NONFINITE_OBS_WARNED_TERMS: set[str] = set()


def _ensure_finite_obs(tensor: torch.Tensor, term_name: str) -> torch.Tensor:
    """Observation-manager safety guard: replace non-finite values with 0."""
    if torch.isfinite(tensor).all():
        return tensor

    bad = ~torch.isfinite(tensor)
    bad_count = int(bad.sum().item())
    if term_name not in _NONFINITE_OBS_WARNED_TERMS:
        _NONFINITE_OBS_WARNED_TERMS.add(term_name)
        warnings.warn(
            f"[ObservationManager] Non-finite values detected in observation term '{term_name}' "
            f"(count={bad_count}). Replacing NaN/Inf with 0.",
            stacklevel=2,
        )
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return _ensure_finite_obs(mat[..., :2].reshape(mat.shape[0], -1), "robot_anchor_ori_w")


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return _ensure_finite_obs(command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1), "robot_anchor_lin_vel_w")


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return _ensure_finite_obs(command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1), "robot_anchor_ang_vel_w")


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return _ensure_finite_obs(pos_b.view(env.num_envs, -1), "robot_body_pos_b")


# ---- Absolute world-frame critic obs (privileged; latent-RL asymmetric critic) -----------------
# These deliberately expose ABSOLUTE world-frame state (env-origin removed so all
# envs share one coordinate semantics) for ALL bodies, UNMASKED. They are for the
# value function only — never deployable. Matching the critic's frame to the
# reward's frame (world-frame visible-POI pos/vel) makes value regression easy.


def ref_body_pos_w_abs(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference body positions, world frame, env-local (env-origin removed), ALL bodies, unmasked."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    p = command.body_pos_w - env.scene.env_origins[:, None, :]  # [N, B, 3]
    return _ensure_finite_obs(p.reshape(env.num_envs, -1), "ref_body_pos_w_abs")


def robot_body_pos_w_abs(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Actual robot body positions, world frame, env-local (env-origin removed), ALL bodies."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    p = command.robot_body_pos_w - env.scene.env_origins[:, None, :]  # [N, B, 3]
    return _ensure_finite_obs(p.reshape(env.num_envs, -1), "robot_body_pos_w_abs")


def ref_body_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference body linear velocities, world frame (origin-invariant), ALL bodies, unmasked."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return _ensure_finite_obs(
        command.body_lin_vel_w.reshape(env.num_envs, -1), "ref_body_lin_vel_w"
    )


def robot_body_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Actual robot body linear velocities, world frame (origin-invariant), ALL bodies."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return _ensure_finite_obs(
        command.robot_body_lin_vel_w.reshape(env.num_envs, -1), "robot_body_lin_vel_w"
    )


def robot_anchor_lin_vel_w_abs(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot anchor (root) linear velocity, world frame. Uses the command's
    SPLIT ``robot_anchor_lin_vel_w`` property (the legacy combined
    ``robot_anchor_vel_w``-based :func:`robot_anchor_lin_vel_w` does not exist on
    ``PartialMaskedMultiMotionCommand``)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return _ensure_finite_obs(
        command.robot_anchor_lin_vel_w.view(env.num_envs, -1), "robot_anchor_lin_vel_w_abs"
    )


def robot_anchor_ang_vel_w_abs(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot anchor (root) angular velocity, world frame (split command property)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return _ensure_finite_obs(
        command.robot_anchor_ang_vel_w.view(env.num_envs, -1), "robot_anchor_ang_vel_w_abs"
    )


def ref_body_pos_robot_anchor_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference motion body positions in the robot-anchor frame (same layout as ``robot_body_pos_b``).

    Use this for imitation / tracking policy inputs so the student sees **motion targets**, not
    the simulated robot's forward-kinematics keypoints.

    Intentionally **not** passed through :func:`_ensure_finite_obs`: for partial keypoint masking,
    :class:`~whole_body_tracking.tasks.tracking.mdp.commands.PartialMaskedMultiMotionCommand`
    writes NaN for hidden bodies so the policy and :class:`~rsl_rl.modules.normalizer.EmpiricalNormalization`
    can ignore those dims in statistics while still receiving NaN on those dims.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.ref_body_pos_robot_anchor_b()


def ref_body_lin_vel_robot_anchor_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference motion body linear velocities in the robot-anchor frame (same layout as ``ref_body_pos_robot_anchor_b``).

    Same NaN semantics as :func:`ref_body_pos_robot_anchor_b` under partial masking.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.ref_body_lin_vel_robot_anchor_b()


def ref_body_pos_robot_anchor_b_lookahead(
    env: ManagerBasedEnv,
    command_name: str,
    lookahead_steps: int = 5,
) -> torch.Tensor:
    """Reference body keypoints over a future lookahead window, all in the CURRENT robot-anchor frame.

    For k = 0..H-1, fetches motion body positions at ``time_steps + k`` (clamped per-clip by the
    motion loader) and expresses them relative to the robot's CURRENT anchor pos+quat — a single
    anchor across the entire window, so all H slots are frame-consistent.

    Spatial mask from :class:`~whole_body_tracking.tasks.tracking.mdp.commands.PartialMaskedMultiMotionCommand`
    (``_env_body_mask``: 1.0 visible / 0.0 masked) is applied uniformly across the H frames by
    writing NaN at masked bodies. The encoder must ``nan_to_num(0)`` before any linear projection
    and ``key_padding_mask`` the matching tokens.

    Multi-motion only: relies on ``env_motion_indices`` + ``motion_dir_loader.gather``.

    Returns ``[N, H * num_bodies * 3]``.
    """
    command = env.command_manager.get_term(command_name)
    H = int(lookahead_steps)
    n_bodies = len(command.cfg.body_names)
    N = env.num_envs

    # Future frame indices per env. compute_global_indices clamps to clip end; no leak into next clip.
    offsets = torch.arange(H, device=command.device, dtype=torch.long)
    frame_indices_2d = command.time_steps[:, None] + offsets[None, :]  # [N, H]
    motion_indices_2d = command.env_motion_indices[:, None].expand(N, H)  # [N, H]

    body_pos_w_flat = command.motion_dir_loader.gather(
        "body_pos_w",
        motion_indices_2d.reshape(-1),
        frame_indices_2d.reshape(-1),
        out_device=command.device,
    )  # [N*H, n_bodies, 3] — raw motion frame, no env origin yet
    body_pos_w = body_pos_w_flat.view(N, H, n_bodies, 3) + env.scene.env_origins[:, None, None, :]

    # Single CURRENT anchor for the whole window.
    anchor_pos_w = command.robot_anchor_pos_w  # [N, 3]
    anchor_quat_w = command.robot_anchor_quat_w  # [N, 4]

    rel_pos_w = body_pos_w - anchor_pos_w[:, None, None, :]  # [N, H, n_bodies, 3]
    flat_count = N * H * n_bodies
    rel_pos_w_flat = rel_pos_w.reshape(flat_count, 3)
    anchor_quat_flat = anchor_quat_w[:, None, None, :].expand(N, H, n_bodies, 4).reshape(flat_count, 4)
    pos_b_flat = quat_rotate_inverse(anchor_quat_flat, rel_pos_w_flat)
    pos_b = pos_b_flat.view(N, H, n_bodies, 3)

    mask = getattr(command, "_env_body_mask", None)
    if mask is not None:
        vis = (mask > 0.5).view(N, 1, n_bodies, 1)
        nan = torch.full_like(pos_b, float("nan"))
        pos_b = torch.where(vis, pos_b, nan)

    # Intentionally NOT _ensure_finite_obs: NaN at masked bodies is the contract.
    return pos_b.reshape(N, H * n_bodies * 3)


def partial_kp_mask_lookahead(
    env: ManagerBasedEnv,
    command_name: str,
    lookahead_steps: int = 5,
) -> torch.Tensor:
    """Per-(frame, body) mask flag for the KP encoder, broadcast across H lookahead frames.

    Layout matches ``ref_body_pos_robot_anchor_b_lookahead``: row-major ``[H, n_bodies]`` flattened
    to ``[N, H * n_bodies]``. The same per-env body mask sampled by
    :class:`~whole_body_tracking.tasks.tracking.mdp.commands.PartialMaskedMultiMotionCommand` is
    repeated across all H frames (within a single obs call, lookahead is a snapshot of "what mask
    applies right now to the upcoming window").

    Convention: 1.0 = masked, 0.0 = visible — matches ``goal_mask_history`` so the encoder builds
    its ``key_padding_mask`` the same way (threshold > 0).
    """
    command = env.command_manager.get_term(command_name)
    H = int(lookahead_steps)
    n_bodies = len(command.cfg.body_names)
    mask = getattr(command, "_env_body_mask", None)
    if mask is None:
        return torch.zeros(env.num_envs, H * n_bodies, device=env.device)
    masked = (mask < 0.5).float()  # [N, n_bodies], 1.0 where masked
    return masked[:, None, :].expand(-1, H, -1).reshape(env.num_envs, -1)


def ref_body_pos_robot_anchor_b_logspaced(
    env: ManagerBasedEnv,
    command_name: str,
    slot_offsets: tuple[int, ...] = (0, 1, 2, 3, 5, 8, 13, 20, 25),
) -> torch.Tensor:
    """Per-body KP packing over arbitrary slot offsets (mixed history/current/future).

    Generalizes :func:`ref_body_pos_robot_anchor_b_delta_lookahead` to a sparse, log-spaced layout
    that mixes history (negative offsets), current (offset 0), and future (positive offsets) slots.
    Per-body packing along the slot axis, in the order given by ``slot_offsets``:
      - offset == 0: absolute ref_pos_t in anchor frame (3D).
      - offset != 0: delta = ref_pos_{t+offset} − robot_body_pos_t in anchor frame (3D).
                     History uses negative offsets (reference body's commanded position in the past
                     expressed relative to the robot's current body position).

    Edge handling: ``compute_global_indices`` clamps both ends per-clip → off-clip slots hold the
    nearest available frame (matches the "pad with closest" semantics).

    Masking: per-body visibility from
    :class:`~whole_body_tracking.tasks.tracking.mdp.commands.PartialMaskedMultiMotionCommand` is
    broadcast across all L slots (NaN at masked bodies). Encoder must ``nan_to_num(0)`` before any
    linear projection and ``key_padding_mask`` the matching tokens.

    Returns ``[N, L * n_bodies * 3]`` row-major ``[L, n_bodies, 3]`` along (slot, body, xyz) so the
    same split-and-transpose used by the existing lookahead obs packs slot-i as a per-body 3D entry.
    """
    command = env.command_manager.get_term(command_name)
    n_bodies = len(command.cfg.body_names)
    N = env.num_envs
    L = len(slot_offsets)

    offsets_t = torch.as_tensor(slot_offsets, device=command.device, dtype=torch.long)  # [L]
    frame_indices_2d = command.time_steps[:, None] + offsets_t[None, :]  # [N, L]
    motion_indices_2d = command.env_motion_indices[:, None].expand(N, L)  # [N, L]

    body_pos_w_flat = command.motion_dir_loader.gather(
        "body_pos_w",
        motion_indices_2d.reshape(-1),
        frame_indices_2d.reshape(-1),
        out_device=command.device,
    )  # [N*L, n_bodies, 3]
    ref_pos_local = body_pos_w_flat.view(N, L, n_bodies, 3)
    ref_pos_w = ref_pos_local + env.scene.env_origins[:, None, None, :]

    anchor_pos_w = command.robot_anchor_pos_w  # [N, 3]
    anchor_quat_w = command.robot_anchor_quat_w  # [N, 4]

    # Ref positions into anchor frame.
    rel_ref_w = ref_pos_w - anchor_pos_w[:, None, None, :]
    flat_count = N * L * n_bodies
    rel_ref_flat = rel_ref_w.reshape(flat_count, 3)
    quat_flat = anchor_quat_w[:, None, None, :].expand(N, L, n_bodies, 4).reshape(flat_count, 4)
    ref_pos_b = quat_rotate_inverse(quat_flat, rel_ref_flat).view(N, L, n_bodies, 3)

    # Robot's CURRENT body pos into anchor frame.
    robot_body_pos_w = command.robot_body_pos_w  # [N, n_bodies, 3]
    rel_robot_w = robot_body_pos_w - anchor_pos_w[:, None, :]
    rel_robot_flat = rel_robot_w.reshape(N * n_bodies, 3)
    quat_robot_flat = anchor_quat_w[:, None, :].expand(N, n_bodies, 4).reshape(N * n_bodies, 4)
    robot_pos_b = quat_rotate_inverse(quat_robot_flat, rel_robot_flat).view(N, n_bodies, 3)

    # Per-slot: abs if offset==0, else delta from robot's CURRENT body pos.
    # ``is_abs_per_slot`` broadcasts across body/xyz axes.
    is_abs_per_slot = (offsets_t == 0).view(1, L, 1, 1)
    packed = torch.where(is_abs_per_slot, ref_pos_b, ref_pos_b - robot_pos_b.unsqueeze(1))

    mask = getattr(command, "_env_body_mask", None)
    if mask is not None:
        vis = (mask > 0.5).view(N, 1, n_bodies, 1)
        nan = torch.full_like(packed, float("nan"))
        packed = torch.where(vis, packed, nan)

    return packed.reshape(N, L * n_bodies * 3)


def ref_single_body_pos_robot_anchor_b_logspaced(
    env: ManagerBasedEnv,
    command_name: str,
    body_name: str,
    slot_offsets: tuple[int, ...] = (0, 1, 2, 3, 5, 8, 13, 20, 25),
) -> torch.Tensor:
    """Privileged, UNMASKED, **single-body** KP lookahead in the robot-anchor frame.

    Identical packing to :func:`ref_body_pos_robot_anchor_b_logspaced` (offset 0 → absolute
    ref pos in anchor frame; offset != 0 → delta from the robot's current body pos), but for
    ONE named body and with NO visibility masking. For a privileged critic that needs a
    specific body's (e.g. the writing wrist's) past/current/future target trajectory with no
    NaN / constant-zero waste — every returned dim is informative. Returns ``[N, L*3]``.
    """
    command = env.command_manager.get_term(command_name)
    body_idx = command.cfg.body_names.index(body_name)
    n_bodies = len(command.cfg.body_names)
    N = env.num_envs
    L = len(slot_offsets)

    offsets_t = torch.as_tensor(slot_offsets, device=command.device, dtype=torch.long)  # [L]
    frame_indices_2d = command.time_steps[:, None] + offsets_t[None, :]  # [N, L]
    motion_indices_2d = command.env_motion_indices[:, None].expand(N, L)  # [N, L]

    body_pos_w_flat = command.motion_dir_loader.gather(
        "body_pos_w",
        motion_indices_2d.reshape(-1),
        frame_indices_2d.reshape(-1),
        out_device=command.device,
    )  # [N*L, n_bodies, 3]
    ref_pos_local = body_pos_w_flat.view(N, L, n_bodies, 3)[:, :, body_idx, :]  # [N, L, 3]
    ref_pos_w = ref_pos_local + env.scene.env_origins[:, None, :]

    anchor_pos_w = command.robot_anchor_pos_w  # [N, 3]
    anchor_quat_w = command.robot_anchor_quat_w  # [N, 4]

    rel_ref_w = ref_pos_w - anchor_pos_w[:, None, :]  # [N, L, 3]
    rel_ref_flat = rel_ref_w.reshape(N * L, 3)
    quat_flat = anchor_quat_w[:, None, :].expand(N, L, 4).reshape(N * L, 4)
    ref_pos_b = quat_rotate_inverse(quat_flat, rel_ref_flat).view(N, L, 3)

    robot_body_pos_w = command.robot_body_pos_w[:, body_idx, :]  # [N, 3]
    rel_robot_w = robot_body_pos_w - anchor_pos_w  # [N, 3]
    robot_pos_b = quat_rotate_inverse(anchor_quat_w, rel_robot_w)  # [N, 3]

    is_abs_per_slot = (offsets_t == 0).view(1, L, 1)
    packed = torch.where(is_abs_per_slot, ref_pos_b, ref_pos_b - robot_pos_b[:, None, :])
    return _ensure_finite_obs(packed.reshape(N, L * 3), "ref_single_body_pos_robot_anchor_b_logspaced")


def partial_kp_mask_logspaced(
    env: ManagerBasedEnv,
    command_name: str,
    num_slots: int = 12,
) -> torch.Tensor:
    """Per-(slot, body) mask flag, broadcast across L slots — matches the layout of
    :func:`ref_body_pos_robot_anchor_b_logspaced`. ``num_slots`` MUST equal
    ``len(slot_offsets)`` of the paired position obs term; the encoder's per-body collapse
    (``mask.reshape(L, N)[..., 0, :]``) reads the same bit from any slot.
    """
    command = env.command_manager.get_term(command_name)
    L = int(num_slots)
    n_bodies = len(command.cfg.body_names)
    mask = getattr(command, "_env_body_mask", None)
    if mask is None:
        return torch.zeros(env.num_envs, L * n_bodies, device=env.device)
    masked = (mask < 0.5).float()
    return masked[:, None, :].expand(-1, L, -1).reshape(env.num_envs, -1)


def ref_body_pos_robot_anchor_b_delta_lookahead(
    env: ManagerBasedEnv,
    command_name: str,
    lookahead_steps: int = 10,
) -> torch.Tensor:
    """Per-body KP packing combining a current-frame absolute slot with future delta slots,
    all in the CURRENT robot-anchor frame.

    For each body i, the returned packing along the slot axis is:
      - slot 0 (3D): absolute reference position at the CURRENT step ``t`` (ref_pos_{t} in anchor frame).
      - slots 1..H (3D each): delta = ref_pos_{t+k} - robot_body_pos_{t} in anchor frame, k = 1..H.

    Total per body = (1 + H) * 3 dims; output shape ``[N, (1+H) * n_bodies * 3]`` in row-major
    ``[L=1+H, n_bodies, 3]`` so the same split-and-transpose used by the lookahead-only obs (i.e.
    ``reshape(L, N, 3).transpose(-3, -2).reshape(N, L*3)``) packs slot 0 as the per-body abs and
    slots 1..H as per-body deltas — no encoder-side code change.

    Masking: per-body visibility from :class:`PartialMaskedMultiMotionCommand` is broadcast across
    ALL L slots and 3 dims (NaN at masked bodies). Encoder must ``nan_to_num(0)`` before any linear
    projection and ``key_padding_mask`` the matching tokens.

    Multi-motion only: relies on ``env_motion_indices`` + ``motion_dir_loader.gather``.
    """
    command = env.command_manager.get_term(command_name)
    H = int(lookahead_steps)
    n_bodies = len(command.cfg.body_names)
    N = env.num_envs
    L = 1 + H  # slot 0 = abs at t, slots 1..H = deltas at t+1..t+H

    offsets = torch.arange(L, device=command.device, dtype=torch.long)  # [0..H]
    frame_indices_2d = command.time_steps[:, None] + offsets[None, :]  # [N, L]
    motion_indices_2d = command.env_motion_indices[:, None].expand(N, L)  # [N, L]

    body_pos_w_flat = command.motion_dir_loader.gather(
        "body_pos_w",
        motion_indices_2d.reshape(-1),
        frame_indices_2d.reshape(-1),
        out_device=command.device,
    )  # [N*L, n_bodies, 3]
    ref_pos_w = body_pos_w_flat.view(N, L, n_bodies, 3) + env.scene.env_origins[:, None, None, :]

    anchor_pos_w = command.robot_anchor_pos_w  # [N, 3]
    anchor_quat_w = command.robot_anchor_quat_w  # [N, 4]

    # Ref positions into anchor frame: subtract anchor pos in world, then rotate.
    rel_ref_w = ref_pos_w - anchor_pos_w[:, None, None, :]
    flat_count = N * L * n_bodies
    rel_ref_flat = rel_ref_w.reshape(flat_count, 3)
    quat_flat = anchor_quat_w[:, None, None, :].expand(N, L, n_bodies, 4).reshape(flat_count, 4)
    ref_pos_b = quat_rotate_inverse(quat_flat, rel_ref_flat).view(N, L, n_bodies, 3)

    # Robot's CURRENT body pos into anchor frame: [N, n_bodies, 3].
    robot_body_pos_w = command.robot_body_pos_w  # [N, n_bodies, 3]
    rel_robot_w = robot_body_pos_w - anchor_pos_w[:, None, :]
    rel_robot_flat = rel_robot_w.reshape(N * n_bodies, 3)
    quat_robot_flat = anchor_quat_w[:, None, :].expand(N, n_bodies, 4).reshape(N * n_bodies, 4)
    robot_pos_b = quat_rotate_inverse(quat_robot_flat, rel_robot_flat).view(N, n_bodies, 3)

    # Slot 0 = absolute ref at t; slots 1..H = ref_{t+k} − robot_pos_{t}.
    packed = ref_pos_b.clone()
    packed[:, 1:, :, :] = packed[:, 1:, :, :] - robot_pos_b.unsqueeze(1)

    mask = getattr(command, "_env_body_mask", None)
    if mask is not None:
        vis = (mask > 0.5).view(N, 1, n_bodies, 1)
        nan = torch.full_like(packed, float("nan"))
        packed = torch.where(vis, packed, nan)

    return packed.reshape(N, L * n_bodies * 3)


def ref_body_pos_robot_anchor_b_window(
    env: ManagerBasedEnv,
    command_name: str,
    lookback_steps: int = 5,
    lookahead_steps: int = 5,
) -> torch.Tensor:
    """Reference body keypoints over a (lookback + lookahead) window, all in the CURRENT robot-anchor frame.

    Same single-anchor convention as :func:`ref_body_pos_robot_anchor_b_lookahead` (one rotation
    across the whole window so the L=K+H slots are frame-consistent), but offsets span
    ``[-K, ..., -1, 0, 1, ..., H-1]``. The slot at offset 0 is the current reference.

    Boundary handling: :meth:`MultiMotionLoader.compute_global_indices` clamps frame indices on
    BOTH sides — out-of-range frames hold-first/hold-last (the reference is "frozen" at the edge).
    Without this clamp, negative indices would silently leak into the previous motion's tail.

    Mask: per-body visibility from ``PartialMaskedMultiMotionCommand`` is broadcast across all L
    slots (NaN at masked bodies). Encoder must ``nan_to_num(0)`` before any linear projection.

    Returns ``[N, (K+H) * num_bodies * 3]`` row-major along (window, body, xyz).
    """
    command = env.command_manager.get_term(command_name)
    K = int(lookback_steps)
    H = int(lookahead_steps)
    L = K + H
    n_bodies = len(command.cfg.body_names)
    N = env.num_envs

    offsets = torch.arange(-K, H, device=command.device, dtype=torch.long)  # [-K..H-1]
    frame_indices_2d = command.time_steps[:, None] + offsets[None, :]  # [N, L]
    motion_indices_2d = command.env_motion_indices[:, None].expand(N, L)  # [N, L]

    body_pos_w_flat = command.motion_dir_loader.gather(
        "body_pos_w",
        motion_indices_2d.reshape(-1),
        frame_indices_2d.reshape(-1),
        out_device=command.device,
    )  # [N*L, n_bodies, 3]
    body_pos_w = body_pos_w_flat.view(N, L, n_bodies, 3) + env.scene.env_origins[:, None, None, :]

    anchor_pos_w = command.robot_anchor_pos_w  # [N, 3]
    anchor_quat_w = command.robot_anchor_quat_w  # [N, 4]

    rel_pos_w = body_pos_w - anchor_pos_w[:, None, None, :]
    flat_count = N * L * n_bodies
    rel_pos_w_flat = rel_pos_w.reshape(flat_count, 3)
    anchor_quat_flat = anchor_quat_w[:, None, None, :].expand(N, L, n_bodies, 4).reshape(flat_count, 4)
    pos_b_flat = quat_rotate_inverse(anchor_quat_flat, rel_pos_w_flat)
    pos_b = pos_b_flat.view(N, L, n_bodies, 3)

    mask = getattr(command, "_env_body_mask", None)
    if mask is not None:
        vis = (mask > 0.5).view(N, 1, n_bodies, 1)
        nan = torch.full_like(pos_b, float("nan"))
        pos_b = torch.where(vis, pos_b, nan)

    return pos_b.reshape(N, L * n_bodies * 3)


def partial_kp_mask_window(
    env: ManagerBasedEnv,
    command_name: str,
    lookback_steps: int = 5,
    lookahead_steps: int = 5,
) -> torch.Tensor:
    """Per-(slot, body) mask flag for the windowed KP encoder, broadcast across L=K+H slots.

    Layout matches :func:`ref_body_pos_robot_anchor_b_window` (row-major ``[L, n_bodies]``
    flattened to ``[N, L * n_bodies]``). Same convention as :func:`partial_kp_mask_lookahead`:
    1.0 = masked, 0.0 = visible.
    """
    command = env.command_manager.get_term(command_name)
    L = int(lookback_steps) + int(lookahead_steps)
    n_bodies = len(command.cfg.body_names)
    mask = getattr(command, "_env_body_mask", None)
    if mask is None:
        return torch.zeros(env.num_envs, L * n_bodies, device=env.device)
    masked = (mask < 0.5).float()  # [N, n_bodies], 1.0 where masked
    return masked[:, None, :].expand(-1, L, -1).reshape(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return _ensure_finite_obs(mat[..., :2].reshape(mat.shape[0], -1), "robot_body_ori_b")


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return _ensure_finite_obs(pos.view(env.num_envs, -1), "motion_anchor_pos_b")


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return _ensure_finite_obs(mat[..., :2].reshape(mat.shape[0], -1), "motion_anchor_ori_b")


def delta_command(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Delta-command obs term for MUSE: ``[ref_joint_pos - current_joint_pos, ref_joint_vel - current_joint_vel]``.

    Same shape ``[N, 2J]`` as ``generated_commands``. When the command term has ``p_mask > 0`` and
    the per-env ``goal_mask`` is True for an env, that env's row is zeroed (the masked / "no goal"
    semantic, in-distribution because zero-delta means "already at target"). Masking is applied
    sync-free via ``torch.where``.
    """
    command = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    delta_pos = command.joint_pos - robot.data.joint_pos
    delta_vel = command.joint_vel - robot.data.joint_vel
    out = torch.cat([delta_pos, delta_vel], dim=-1)
    goal_mask = getattr(command, "goal_mask", None)
    if goal_mask is not None:
        out = torch.where(goal_mask.unsqueeze(-1), torch.zeros_like(out), out)
    return _ensure_finite_obs(out, "delta_command")


def motion_anchor_ori_b_maskable(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """``motion_anchor_ori_b`` with optional per-env masking to identity (6-D).

    For envs where the command's ``goal_mask`` is True, the row is replaced with the 6-D identity
    rotation ``[1, 0, 0, 1, 0, 0]`` (row-major reshape of the first two columns of I). Masking is
    applied sync-free via ``torch.where``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    out = mat[..., :2].reshape(mat.shape[0], -1)
    goal_mask = getattr(command, "goal_mask", None)
    if goal_mask is not None:
        identity_row = torch.zeros(6, device=out.device, dtype=out.dtype)
        identity_row[0] = 1.0
        identity_row[3] = 1.0
        out = torch.where(goal_mask.unsqueeze(-1), identity_row.expand_as(out), out)
    return _ensure_finite_obs(out, "motion_anchor_ori_b_maskable")


def delta_command_real(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Delta-command for SAGEII encoder: real delta everywhere, never masked.

    Same shape ``[N, 2J]`` as :func:`delta_command`, but ignores ``command.goal_mask`` so the
    encoder always sees the actual command. SAGEII routes masking on the *decoder* side via the
    z-token's ``key_padding_mask`` instead of zeroing the encoder input — this keeps the encoder's
    consecutive-μ smoothness regularization meaningful (no curriculum-induced jumps in μ when the
    mask flips).
    """
    command = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    delta_pos = command.joint_pos - robot.data.joint_pos
    delta_vel = command.joint_vel - robot.data.joint_vel
    out = torch.cat([delta_pos, delta_vel], dim=-1)
    return _ensure_finite_obs(out, "delta_command_real")


def goal_mask_history(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Per-step goal-mask flag as a ``[N, 1]`` float, history-rolled by Isaac to ``[N, history]``.

    Returns 1.0 where ``command.goal_mask`` is True (this frame is masked), else 0.0. The
    transformer encoder reads this (post-normalization, threshold > 0) to build its
    ``key_padding_mask`` for goal tokens.
    """
    command = env.command_manager.get_term(command_name)
    goal_mask = getattr(command, "goal_mask", None)
    if goal_mask is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return goal_mask.float().unsqueeze(-1)


def command_self_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Self-targeting command: replaces ``generated_commands`` with current absolute joint state.

    Returns ``concat([joint_pos, joint_vel], dim=-1)`` of the actuated articulation, matching the
    ``[ref_joint_pos, ref_joint_vel]`` layout of ``generated_commands``. Used for the
    encoder-as-proprio rollout (see ``--encoder_as_proprio`` in play.py): feeding the encoder a
    "target = current state" signal so the delta is approximately zero with respect to the encoder's
    absolute-target training distribution.
    """
    del command_name
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    return _ensure_finite_obs(torch.cat([joint_pos, joint_vel], dim=-1), "command_self_target")


def motion_anchor_ori_b_identity(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Constant identity rotation in the same 6-D representation as ``motion_anchor_ori_b``.

    The 6-D rep takes ``mat[..., :2]`` of the 3x3 rotation matrix (shape ``[3, 2]``) and reshapes
    row-major to length 6. Layout is therefore
    ``[mat[0,0], mat[0,1], mat[1,0], mat[1,1], mat[2,0], mat[2,1]]``; identity has 1s at indices 0
    and 3. Used by the encoder-as-proprio rollout to remove anchor-orientation goal information.
    """
    del command_name
    out = torch.zeros(env.num_envs, 6, device=env.device)
    out[:, 0] = 1.0
    out[:, 3] = 1.0
    return out


def ref_base_lin_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference base linear velocity in the robot's base frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    # Get reference anchor linear velocity in world frame
    ref_lin_vel_w = command.anchor_lin_vel_w

    # Transform to robot's base frame using inverse quaternion rotation
    ref_lin_vel_b = quat_rotate_inverse(command.anchor_quat_w, ref_lin_vel_w)

    return _ensure_finite_obs(ref_lin_vel_b.view(env.num_envs, -1), "ref_base_lin_vel_b")


def ref_projected_gravity(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference projected gravity in the reference motion's base frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    # World frame gravity vector [0, 0, -1]
    gravity_w = torch.zeros(env.num_envs, 3, device=env.device)
    gravity_w[:, 2] = -1.0

    # Transform to reference motion's base frame using inverse quaternion rotation
    ref_gravity_b = quat_rotate_inverse(command.anchor_quat_w, gravity_w)

    return _ensure_finite_obs(ref_gravity_b.view(env.num_envs, -1), "ref_projected_gravity")

def body_pos_relative_w(
    env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return _ensure_finite_obs(
        command.body_pos_relative_w.reshape(command.body_pos_relative_w.size(0), -1),
        "body_pos_relative_w",
    )

def body_quat_relative_w(
    env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return _ensure_finite_obs(
        command.body_quat_relative_w.reshape(command.body_quat_relative_w.size(0), -1),
        "body_quat_relative_w",
    )

def selected_keypoints_pos_w_heading(
    env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return _ensure_finite_obs(
        command.selected_keypoints_pos_w_heading.reshape(command.selected_keypoints_pos_w_heading.size(0), -1),
        "selected_keypoints_pos_w_heading",
    )