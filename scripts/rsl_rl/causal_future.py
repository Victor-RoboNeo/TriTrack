"""Causal future-slot injection for KP5 policy obs.

The env's ``ref_body_pos_robot_anchor_b_logspaced`` packs ground-truth future
slots from the replay clip. That is Oracle. Hold / Mapper must overwrite those
slots using only intent at offsets ``<= 0`` plus the robot's current KP.

Layout (byte-match Stage-2 / HeadHands actor obs):
    obs[..., 0:225]  kp  row-major [L=15, N=5, 3]
    obs[..., 225:300] mask [L, N]
    obs[..., 300:750] proprio

Slot offsets: (-25,-20,-15,-10,-6,-3,-1,0,1,3,6,10,15,20,25).
Visible bodies 0..2 = torso, left wrist, right wrist. Ankles stay NaN/masked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from isaaclab.utils.math import quat_rotate, quat_rotate_inverse

_TRITRACK = Path("/data/home/chenxiangyu/victor/TriTrack")
if str(_TRITRACK) not in sys.path:
    sys.path.insert(0, str(_TRITRACK))

from tritrack.intent.mapper import IN_DIM_INTENT, FutureIntentMapper  # noqa: E402

SLOT_OFFSETS = (-25, -20, -15, -10, -6, -3, -1, 0, 1, 3, 6, 10, 15, 20, 25)
N_BODIES = 5
N_VISIBLE = 3
VISIBLE = (0, 1, 2)
NONPOS_OFFSETS = tuple(o for o in SLOT_OFFSETS if o <= 0)  # 8, causal
FUTURE_OFFSETS = tuple(o for o in SLOT_OFFSETS if o > 0)  # 7
FUTURE_SLOT_IDX = tuple(i for i, o in enumerate(SLOT_OFFSETS) if o > 0)
NONPOS_SLOT_IDX = tuple(i for i, o in enumerate(SLOT_OFFSETS) if o <= 0)

# Flat obs indices of the 63 future-slot entries for the 3 visible bodies,
# matching tritrack.training.dataset.FUTURE_OBS_IDX / train_mapper.student_obs.
FUTURE_OBS_IDX = torch.tensor(
    [(li * N_BODIES + bi) * 3 + d for li in FUTURE_SLOT_IDX for bi in VISIBLE for d in range(3)],
    dtype=torch.long,
)

# User-requested mapper horizons (seconds) @ 50 Hz control.
HORIZON_S = (0.1, 0.2, 0.3, 0.5)
DT = 0.02

DEFAULT_MAPPER_A = "/data/home/chenxiangyu/victor/TriTrack/runs/mapper_full/mapper_best.pt"
DEFAULT_MAPPER = "/data/home/chenxiangyu/victor/TriTrack/runs/mapper_b_intent72/mapper_best.pt"  # canonical 72-D
DEFAULT_MAPPER_B = DEFAULT_MAPPER


def _motion(env):
    return env.unwrapped.command_manager.get_term("motion")


def gather_ref_world(env, offsets: tuple[int, ...], time_steps: torch.Tensor | None = None) -> torch.Tensor:
    """Clip body positions at ``t+offset`` in the world frame. ``[N, L, n_bodies, 3]``.

    ``time_steps`` overrides the command clock (used for simulated HMD latency).
    """
    command = _motion(env)
    n_bodies = len(command.cfg.body_names)
    n_env = int(env.unwrapped.num_envs)
    n_slots = len(offsets)
    offsets_t = torch.as_tensor(offsets, device=command.device, dtype=torch.long)
    ts = command.time_steps if time_steps is None else time_steps
    frame_2d = ts[:, None] + offsets_t[None, :]
    motion_2d = command.env_motion_indices[:, None].expand(n_env, n_slots)
    body_pos_w_flat = command.motion_dir_loader.gather(
        "body_pos_w",
        motion_2d.reshape(-1),
        frame_2d.reshape(-1),
        out_device=command.device,
    )
    return body_pos_w_flat.view(n_env, n_slots, n_bodies, 3) + env.unwrapped.scene.env_origins[:, None, None, :]


def _points_to_frame(points_w: torch.Tensor, origin: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """``points_w [N, ..., 3]`` → origin/quat frame. ``origin [N,3]``, ``quat [N,4]`` wxyz."""
    n_env = origin.shape[0]
    extra = points_w.shape[1:-1]
    rel = points_w - origin.view(n_env, *([1] * len(extra)), 3)
    n_pts = int(rel.numel() // (n_env * 3))
    q = quat[:, None, :].expand(n_env, n_pts, 4).reshape(-1, 4)
    return quat_rotate_inverse(q, rel.reshape(-1, 3)).view_as(points_w)


def _points_from_frame(points_f: torch.Tensor, origin: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    n_env = origin.shape[0]
    extra = points_f.shape[1:-1]
    n_pts = int(points_f.numel() // (n_env * 3))
    q = quat[:, None, :].expand(n_env, n_pts, 4).reshape(-1, 4)
    return quat_rotate(q, points_f.reshape(-1, 3)).view_as(points_f) + origin.view(n_env, *([1] * len(extra)), 3)


def gather_ref_abs_anchor(env, offsets: tuple[int, ...]) -> torch.Tensor:
    """Clip body positions at ``t+offset`` in the CURRENT **robot**-anchor frame. ``[N, L, n_bodies, 3]``."""
    command = _motion(env)
    return _points_to_frame(gather_ref_world(env, offsets), command.robot_anchor_pos_w, command.robot_anchor_quat_w)


def robot_kp_abs_anchor(env) -> torch.Tensor:
    """Robot's CURRENT KP5 positions in the robot-anchor frame. ``[N, n_bodies, 3]``."""
    command = _motion(env)
    n_bodies = len(command.cfg.body_names)
    n_env = int(env.unwrapped.num_envs)
    rel = command.robot_body_pos_w - command.robot_anchor_pos_w[:, None, :]
    quat = command.robot_anchor_quat_w[:, None, :].expand(n_env, n_bodies, 4).reshape(n_env * n_bodies, 4)
    return quat_rotate_inverse(quat, rel.reshape(n_env * n_bodies, 3)).view(n_env, n_bodies, 3)


def current_packet_world(env) -> torch.Tensor:
    """Clip's CURRENT visible 3-point in world. ``[N, 3, 3]``. No past, no future."""
    return gather_ref_world(env, (0,))[:, 0, :N_VISIBLE, :]


class CausalIntentBuffer:
    """Per-env ring of current-only world packets. History is sampled at ``NONPOS_OFFSETS``.

    Single causal memory for NPZ replay and OpenXR/UDP. A packet is always
    ``(torso, left, right)`` at one timestamp — never a GT future slot.
    """

    def __init__(self, delay_steps: int = 0):
        self.max_history = -min(NONPOS_OFFSETS)
        self.delay_steps = int(delay_steps)
        self._w: torch.Tensor | None = None
        self._delay: torch.Tensor | None = None

    def reset(self, pts_w: torch.Tensor, delay_steps: int | None = None) -> None:
        pts = pts_w.reshape(-1, N_VISIBLE, 3).contiguous()
        n = int(pts.shape[0])
        if delay_steps is not None:
            self.delay_steps = int(delay_steps)
        self._w = pts[:, None].expand(n, self.max_history + 1, N_VISIBLE, 3).clone()
        if self.delay_steps > 0:
            self._delay = pts[:, None].expand(n, self.delay_steps, N_VISIBLE, 3).clone()
        else:
            self._delay = None

    def push(self, pts_w: torch.Tensor) -> torch.Tensor:
        pts = pts_w.reshape(-1, N_VISIBLE, 3).contiguous()
        if self._w is None or int(self._w.shape[0]) != int(pts.shape[0]):
            self.reset(pts)
            return pts
        released = pts
        if self.delay_steps > 0 and self._delay is not None:
            released = self._delay[:, 0].clone()
            self._delay = torch.cat([self._delay[:, 1:], pts[:, None]], dim=1)
        self._w = torch.cat([self._w[:, 1:], released[:, None]], dim=1)
        return released

    def hist(self) -> torch.Tensor:
        """``[N, 8, 3, 3]`` at ``NONPOS_OFFSETS`` (oldest … current)."""
        if self._w is None:
            raise RuntimeError("CausalIntentBuffer.reset() before hist()")
        h = int(self._w.shape[1])
        idx = [h - 1 if o >= 0 else max(h - 1 + int(o), 0) for o in NONPOS_OFFSETS]
        return self._w[:, idx]


def resample_future_slots(pred: torch.Tensor, src_offs: tuple[int, ...], query_offs: torch.Tensor) -> torch.Tensor:
    """Linear resample of Mapper-B futures. ``pred [N,7,3,3]``, ``query_offs [7]`` in ticks."""
    src = torch.as_tensor(src_offs, device=pred.device, dtype=pred.dtype)
    q = query_offs.to(device=pred.device, dtype=pred.dtype)
    out = pred[:, -1].unsqueeze(1).expand(-1, int(q.numel()), -1, -1).clone()
    for i in range(int(q.numel())):
        t = float(q[i].item())
        if t <= float(src[0]):
            out[:, i] = pred[:, 0]
            continue
        if t >= float(src[-1]):
            span = max(float(src[-1] - src[-2]), 1e-6)
            w = (t - float(src[-1])) / span
            out[:, i] = pred[:, -1] + w * (pred[:, -1] - pred[:, -2])
            continue
        hi = int((src >= t).nonzero(as_tuple=False)[0].item())
        lo = max(hi - 1, 0)
        span = max(float(src[hi] - src[lo]), 1e-6)
        w = (t - float(src[lo])) / span
        out[:, i] = (1.0 - w) * pred[:, lo] + w * pred[:, hi]
    return out


def _lerp_horizon(pred_abs: torch.Tensor, gt_abs: torch.Tensor, horizon_s: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly interpolate 7 future slots to ``horizon_s``. Tensors ``[N, 7, 3, 3]``."""
    steps = horizon_s / DT
    offs = FUTURE_OFFSETS
    if steps <= offs[0]:
        return pred_abs[:, 0], gt_abs[:, 0]
    if steps >= offs[-1]:
        return pred_abs[:, -1], gt_abs[:, -1]
    hi = next(i for i, o in enumerate(offs) if o >= steps)
    lo = max(hi - 1, 0)
    span = max(float(offs[hi] - offs[lo]), 1e-6)
    w = max(0.0, min(1.0, (steps - offs[lo]) / span))
    pred = (1.0 - w) * pred_abs[:, lo] + w * pred_abs[:, hi]
    gt = (1.0 - w) * gt_abs[:, lo] + w * gt_abs[:, hi]
    return pred, gt


def intent_visible_from_obs(obs: torch.Tensor) -> torch.Tensor:
    """``[N, 3]`` bool: torso / L wrist / R wrist visible in the current mask (slot 0)."""
    mask = obs[:, 225:300].reshape(-1, 15, N_BODIES)
    return mask[:, NONPOS_SLOT_IDX[-1], :N_VISIBLE] < 0.5


class CausalFutureInjector:
    """Overwrite future KP slots in a 750-D policy obs. Mapper never sees t+>0 GT."""

    def __init__(
        self,
        mapper_path: str | None = DEFAULT_MAPPER,
        device: str | torch.device = "cpu",
        mappers: dict[str, str] | None = None,
    ):
        self.device = torch.device(device)
        self.mappers: dict[str, FutureIntentMapper] = {}
        self.paths: dict[str, str] = {}
        self.val_cos: dict[str, float] = {}
        paths = dict(mappers or {})
        if mapper_path and "mapper" not in paths:
            paths.setdefault("mapper", mapper_path)
        for name, path in paths.items():
            if not path:
                continue
            mapper, extra = FutureIntentMapper.from_checkpoint(path, device=self.device)
            self.mappers[name] = mapper
            self.paths[name] = str(path)
            self.val_cos[name] = float(extra.get("val_cos", float("nan"))) if extra else float("nan")
            print(
                f"[causal] loaded {name} {path} in_dim={mapper.in_dim} "
                f"val_cos={self.val_cos[name]:.4f} future_obs_idx={int(FUTURE_OBS_IDX.numel())}",
                flush=True,
            )
        self.mapper = self.mappers.get("mapper") or next(iter(self.mappers.values()), None)
        self.mapper_path = self.paths.get("mapper", mapper_path or "")
        self.last_pred_abs: torch.Tensor | None = None  # [N, 7, 3, 3] last mapper/hold future
        self.last_gt_abs: torch.Tensor | None = None
        self.last_pred_w: torch.Tensor | None = None  # [N, 7, 3, 3] mapper future in world
        self.last_cur_w: torch.Tensor | None = None  # [N, 3, 3] current intent world
        self.latency_steps: int = 0
        self.hand_noise_m: float = 0.0
        self.drop_prob: float = 0.0
        self.latency_shift_steps: int = 0
        self._held_hist_w: torch.Tensor | None = None
        self._drop_streak: torch.Tensor | None = None
        self.last_dropped: torch.Tensor | None = None

    @torch.no_grad()
    def mapper_features(self, env, vis: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Causal mapper features in the CURRENT **intent** (clip torso) frame.

        Masked intent bodies (1/2-point) are zeroed so Mapper-B never sees clip GT
        for a wrist the human did not command.

        Returns ``(feats81, intent72, cur, robot_vis_in_intent)``.
        """
        command = _motion(env)
        n_env = int(env.unwrapped.num_envs)
        ts = command.time_steps
        lag = int(getattr(self, "latency_steps", 0) or 0)
        if lag > 0:
            ts = (ts - lag).clamp(min=0)
        hist_w = gather_ref_world(env, NONPOS_OFFSETS, time_steps=ts)[:, :, :N_VISIBLE, :]
        dropped = torch.zeros(n_env, dtype=torch.bool, device=hist_w.device)
        p_drop = float(getattr(self, "drop_prob", 0.0) or 0.0)
        if p_drop > 0.0:
            dropped = torch.rand(n_env, device=hist_w.device) < p_drop
            if self._held_hist_w is not None and bool(dropped.any()):
                hist_w = hist_w.clone()
                hist_w[dropped] = self._held_hist_w[dropped]
        self._held_hist_w = hist_w.clone()
        if self._drop_streak is None or int(self._drop_streak.numel()) != n_env:
            self._drop_streak = torch.zeros(n_env, dtype=torch.long, device=hist_w.device)
        self._drop_streak = torch.where(dropped, self._drop_streak + 1, torch.zeros_like(self._drop_streak))
        self.last_dropped = dropped
        self.last_cur_w = hist_w[:, -1].clone()
        hist = _points_to_frame(hist_w, command.anchor_pos_w, command.anchor_quat_w)
        robot_w = command.robot_body_pos_w[:, :N_VISIBLE, :]
        robot_vis = _points_to_frame(robot_w, command.anchor_pos_w, command.anchor_quat_w)
        hist = torch.nan_to_num(hist, 0.0)
        robot_vis = torch.nan_to_num(robot_vis, 0.0)
        noise_m = float(getattr(self, "hand_noise_m", 0.0) or 0.0)
        if noise_m > 0.0:
            # Wrist channels only (bodies 1, 2). Torso stays clean.
            eps = torch.randn_like(hist[:, :, 1:3, :]) * noise_m
            hist[:, :, 1:3, :] = hist[:, :, 1:3, :] + eps
        if vis is not None:
            v = vis.to(dtype=hist.dtype).view(n_env, 1, N_VISIBLE, 1)
            hist = hist * v
            robot_vis = robot_vis * vis.to(dtype=robot_vis.dtype).view(n_env, N_VISIBLE, 1)
        robot_vis = robot_vis.reshape(n_env, 9)
        cur = hist[:, -1].reshape(n_env, 9)
        intent72 = hist.reshape(n_env, 72)
        feats81 = torch.cat([intent72, robot_vis], dim=-1)
        return feats81, intent72, cur, robot_vis

    def _mapper_for(self, mode: str) -> FutureIntentMapper:
        if mode in self.mappers:
            return self.mappers[mode]
        if mode in ("mapper", "mapper_a", "mapper_b") and "mapper" in self.mappers:
            return self.mappers["mapper"]
        alias = {"mapper_a": "mapper", "mapper_b": "mapper"}
        key = alias.get(mode, mode)
        if key in self.mappers:
            return self.mappers[key]
        raise RuntimeError(f"no mapper loaded for mode={mode!r} have={list(self.mappers)}")

    @torch.no_grad()
    def predict_future_robot_anchor(
        self, env, mode: str = "mapper", vis: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mapper future converted into the robot-anchor frame. ``([N,7,3,3], robot_vis_ra [N,9])``."""
        command = _motion(env)
        feats81, intent72, cur, _robot_intent = self.mapper_features(env, vis=vis)
        mapper = self._mapper_for(mode)
        feats = intent72 if mapper.in_dim == IN_DIM_INTENT else feats81
        pred_intent = mapper(feats.float(), cur.float()).reshape(-1, 7, N_VISIBLE, 3)
        pred_w = _points_from_frame(pred_intent, command.anchor_pos_w, command.anchor_quat_w)
        self.last_pred_w = pred_w
        pred_ra = _points_to_frame(pred_w, command.robot_anchor_pos_w, command.robot_anchor_quat_w)
        robot_ra = robot_kp_abs_anchor(env)[:, :N_VISIBLE, :].reshape(pred_ra.shape[0], 9)
        if vis is not None:
            pred_ra = pred_ra * vis.to(dtype=pred_ra.dtype).view(-1, 1, N_VISIBLE, 1)
        if not hasattr(self, "_dbg_n"):
            self._dbg_n = {}
        n = self._dbg_n.get(mode, 0)
        if n < 5:
            self._dbg_n[mode] = n + 1
            res = (pred_intent.reshape(-1, 7, 9) - cur.reshape(-1, 1, 9)).abs()
            n_vis = float(vis.float().mean()) if vis is not None else 1.0
            print(
                f"[causal-dbg] {mode} in_dim={mapper.in_dim} t={n+1} vis_frac={n_vis:.2f} "
                f"feat_max={float(feats.abs().max()):.3f} cur_max={float(cur.abs().max()):.3f} "
                f"res_mean={float(res.mean()):.4f} res_max={float(res.max()):.4f}",
                flush=True,
            )
        return pred_ra, robot_ra

    @torch.no_grad()
    def patch_policy_obs(self, obs: torch.Tensor, env, mode: str) -> torch.Tensor:
        """Return a cloned obs with future slots set by ``mode``.

        Only currently-visible intent bodies (torso / L wrist / R wrist) are
        written. Masked bodies keep the env's NaN + mask=1 — 1/2-point eval
        never injects hallucinated wrist commands.
        """
        mode = mode.lower()
        if mode.endswith("_norest"):
            mode = mode[: -len("_norest")]
        vis = intent_visible_from_obs(obs)
        if mode == "oracle":
            self.last_pred_abs = None
            self.last_gt_abs = None
            return obs
        if obs.shape[-1] < 225:
            raise RuntimeError(f"policy obs dim {obs.shape[-1]} < 225; cannot patch KP block")

        n_env = obs.shape[0]
        robot = robot_kp_abs_anchor(env)
        robot_vis = robot[:, :N_VISIBLE, :]
        cur = obs[:, :225].reshape(n_env, 15, N_BODIES, 3)[:, NONPOS_SLOT_IDX[-1], :N_VISIBLE, :]
        cur = torch.nan_to_num(cur, 0.0)

        if mode == "hold":
            pred_abs = cur[:, None, :, :].expand(-1, 7, -1, -1).contiguous()
        elif mode in ("mapper", "mapper_a", "mapper_b") or mode in self.mappers:
            pred_abs, robot_vis_flat = self.predict_future_robot_anchor(env, mode=mode, vis=vis)
            robot_vis = robot_vis_flat.reshape(n_env, N_VISIBLE, 3)
        else:
            raise ValueError(f"unknown future mode {mode!r}")

        self.last_pred_abs = pred_abs
        deltas = pred_abs - robot_vis.reshape(n_env, 1, N_VISIBLE, 3)
        out = obs.clone()
        kp = out[:, :225].reshape(n_env, 15, N_BODIES, 3)
        for k, li in enumerate(FUTURE_SLOT_IDX):
            for bi in range(N_VISIBLE):
                sel = vis[:, bi]
                if bool(sel.any()):
                    kp[sel, li, bi] = deltas[sel, k, bi].to(dtype=kp.dtype)
        out[:, :225] = kp.reshape(n_env, 225)
        return out

    @torch.no_grad()
    def patch_from_world_hist(
        self,
        obs: torch.Tensor,
        env,
        hist_w: torch.Tensor,
        vis: torch.Tensor | None = None,
        mode: str = "mapper",
    ) -> torch.Tensor:
        """Overwrite KP from a causal WORLD-frame 3-point stream (OpenXR / UDP).

        ``hist_w`` is ``[N, 8, 3, 3]`` at ``NONPOS_OFFSETS`` (torso + wrists).
        Mapper-B still runs in the clip/intent torso frame (same as replay).
        Past + current + future visible slots are all rewritten — clip GT never
        enters the actor. Ankles and masked bodies stay NaN / mask=1.
        """
        command = _motion(env)
        n_env = obs.shape[0]
        if vis is None:
            vis = torch.ones(n_env, N_VISIBLE, dtype=torch.bool, device=obs.device)
        vis_f = vis.to(dtype=hist_w.dtype).view(n_env, 1, N_VISIBLE, 1)
        hist_intent = _points_to_frame(hist_w, command.anchor_pos_w, command.anchor_quat_w)
        hist_intent = torch.nan_to_num(hist_intent, 0.0) * vis_f
        cur = hist_intent[:, -1].reshape(n_env, 9)
        intent72 = hist_intent.reshape(n_env, 72)
        mapper = self._mapper_for(mode)
        robot_vis_intent = _points_to_frame(
            command.robot_body_pos_w[:, :N_VISIBLE, :],
            command.anchor_pos_w,
            command.anchor_quat_w,
        )
        robot_vis_intent = robot_vis_intent * vis.to(dtype=robot_vis_intent.dtype).view(n_env, N_VISIBLE, 1)
        feats81 = torch.cat([intent72, robot_vis_intent.reshape(n_env, 9)], dim=-1)
        feats = intent72 if mapper.in_dim == IN_DIM_INTENT else feats81
        pred_intent = mapper(feats.float(), cur.float()).reshape(-1, 7, N_VISIBLE, 3)
        shift = int(getattr(self, "latency_shift_steps", 0) or 0)
        if shift > 0:
            query = torch.as_tensor(FUTURE_OFFSETS, device=pred_intent.device, dtype=pred_intent.dtype) + float(shift)
            pred_intent = resample_future_slots(pred_intent, FUTURE_OFFSETS, query)
        pred_w = _points_from_frame(pred_intent, command.anchor_pos_w, command.anchor_quat_w)
        self.last_pred_w = pred_w
        self.last_cur_w = hist_w[:, -1].clone()
        pred_ra = _points_to_frame(pred_w, command.robot_anchor_pos_w, command.robot_anchor_quat_w)
        if vis is not None:
            pred_ra = pred_ra * vis.to(dtype=pred_ra.dtype).view(n_env, 1, N_VISIBLE, 1)
        hist_ra = _points_to_frame(hist_w, command.robot_anchor_pos_w, command.robot_anchor_quat_w)
        robot_vis = robot_kp_abs_anchor(env)[:, :N_VISIBLE, :]
        self.last_pred_abs = pred_ra
        deltas = pred_ra - robot_vis.reshape(n_env, 1, N_VISIBLE, 3)

        out = obs.clone()
        kp = out[:, :225].reshape(n_env, 15, N_BODIES, 3)
        mask = out[:, 225:300].reshape(n_env, 15, N_BODIES)
        nan = torch.full((3,), float("nan"), device=obs.device, dtype=kp.dtype)
        kp[:] = nan
        mask[:] = 1.0
        for k, li in enumerate(NONPOS_SLOT_IDX):
            off = SLOT_OFFSETS[li]
            tgt = hist_ra[:, k]
            for bi in range(N_VISIBLE):
                sel = vis[:, bi]
                if not bool(sel.any()):
                    continue
                if off == 0:
                    kp[sel, li, bi] = tgt[sel, bi].to(dtype=kp.dtype)
                else:
                    kp[sel, li, bi] = (tgt[:, bi] - robot_vis[:, bi])[sel].to(dtype=kp.dtype)
                mask[sel, li, bi] = 0.0
        for k, li in enumerate(FUTURE_SLOT_IDX):
            for bi in range(N_VISIBLE):
                sel = vis[:, bi]
                if bool(sel.any()):
                    kp[sel, li, bi] = deltas[sel, k, bi].to(dtype=kp.dtype)
                    mask[sel, li, bi] = 0.0
        out[:, :225] = kp.reshape(n_env, 225).clone()
        out[:, 225:300] = mask.reshape(n_env, 75).clone()
        return out

    @torch.no_grad()
    def future_errors(self, env, pred_abs: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Post-hoc GT future error. Does not feed the policy.

        Returns per-env dict of horizon L2 (mean over 3 visible bodies).
        """
        pred = pred_abs if pred_abs is not None else self.last_pred_abs
        gt = gather_ref_abs_anchor(env, FUTURE_OFFSETS)[:, :, :N_VISIBLE, :]
        self.last_gt_abs = gt
        if pred is None:
            pred = gt
        err_slots = torch.linalg.norm(pred - gt, dim=-1).mean(dim=-1)  # [N, 7]
        out = {f"slot_{o}": err_slots[:, i] for i, o in enumerate(FUTURE_OFFSETS)}
        for h in HORIZON_S:
            p, g = _lerp_horizon(pred, gt, h)
            out[f"h_{h:.1f}"] = torch.linalg.norm(p - g, dim=-1).mean(dim=-1)
        return out
