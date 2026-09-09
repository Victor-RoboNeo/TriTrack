"""Isaac deployment runtime: same controller graph as future Unitree G1.

Streaming 3-point (past+current only) → Mapper-B → Stage-2 encoder → g_φ
→ Stage-2 decoder → RobotBackend.

Future KP slots never come from clip GT.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_conjugate, quat_error_magnitude, quat_mul, quat_rotate_inverse

from causal_future import (
    FUTURE_OFFSETS,
    NONPOS_OFFSETS,
    N_VISIBLE,
    CausalFutureInjector,
    CausalIntentBuffer,
    current_packet_world,
    _motion,
    robot_kp_abs_anchor,
)


class RobotBackend(ABC):
    @abstractmethod
    def get_observations(self):
        ...

    @abstractmethod
    def send_joint_target(self, q: torch.Tensor):
        ...

    @abstractmethod
    def get_base_state(self) -> dict[str, torch.Tensor]:
        ...


class IsaacBackend(RobotBackend):
    """Isaac PhysX G1. Swap this for UnitreeG1Backend on hardware — nothing upstream changes."""

    def __init__(self, env):
        self.env = env

    def get_observations(self):
        return self.env.get_observations()

    def send_joint_target(self, q: torch.Tensor):
        return self.env.step(q)

    def get_base_state(self) -> dict[str, torch.Tensor]:
        cmd = _motion(self.env)
        asset = self.env.unwrapped.scene["robot"]
        return {
            "root_pos_w": asset.data.root_pos_w,
            "root_quat_w": asset.data.root_quat_w,
            "robot_body_pos_w": cmd.robot_body_pos_w,
            "intent_body_pos_w": cmd.body_pos_w,
            "anchor_pos_w": cmd.anchor_pos_w,
            "anchor_quat_w": cmd.anchor_quat_w,
            "robot_anchor_pos_w": cmd.robot_anchor_pos_w,
            "robot_anchor_quat_w": cmd.robot_anchor_quat_w,
        }


class UnitreeG1Backend(RobotBackend):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("UnitreeG1Backend is the hardware swap; Isaac-only until Gate A–E pass.")


class DeadManSwitch:
    """Hold / damp / stop. Triggered in Isaac first, reused on G1."""

    def __init__(
        self,
        drop_hold_steps: int = 5,
        jump_m: float = 0.25,
        tilt_rad: float = 1.2,
    ):
        self.drop_hold_steps = drop_hold_steps
        self.jump_m = jump_m
        self.tilt_rad = tilt_rad
        self.prev_intent: torch.Tensor | None = None
        self.last_q: torch.Tensor | None = None
        self.hold = False
        self.reason = ""

    def check(
        self,
        *,
        action: torch.Tensor,
        drop_streak: torch.Tensor | None,
        intent_cur_w: torch.Tensor | None,
        tilt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        n = int(action.shape[0])
        hold = torch.zeros(n, dtype=torch.bool, device=action.device)
        reasons = [""] * n
        if bool(torch.isnan(action).any() | torch.isinf(action).any()):
            bad = torch.isnan(action).any(dim=-1) | torch.isinf(action).any(dim=-1)
            hold |= bad
            for i in range(n):
                if bool(bad[i]):
                    reasons[i] = "nan_action"
        if drop_streak is not None:
            late = drop_streak >= self.drop_hold_steps
            hold |= late
            for i in range(n):
                if bool(late[i]) and not reasons[i]:
                    reasons[i] = "packet_gap"
        if intent_cur_w is not None:
            cur = intent_cur_w[:, 0]  # torso
            if self.prev_intent is not None and self.prev_intent.shape == cur.shape:
                jump = torch.linalg.norm(cur - self.prev_intent, dim=-1) > self.jump_m
                hold |= jump
                for i in range(n):
                    if bool(jump[i]) and not reasons[i]:
                        reasons[i] = "intent_jump"
            self.prev_intent = cur.clone()
        tilt_bad = tilt.abs() > self.tilt_rad
        hold |= tilt_bad
        for i in range(n):
            if bool(tilt_bad[i]) and not reasons[i]:
                reasons[i] = "torso_tilt"
        if self.last_q is None:
            self.last_q = action.detach().clone()
        out = action.clone()
        if bool(hold.any()):
            out[hold] = self.last_q[hold]
        live = ~hold
        if bool(live.any()):
            self.last_q[live] = action[live].detach()
        self.hold = bool(hold.any())
        return out, hold, reasons


def wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def rel_rpy(q_ref: torch.Tensor, q_robot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_err = quat_mul(quat_conjugate(q_ref), q_robot)
    roll, pitch, yaw = euler_xyz_from_quat(q_err)
    return wrap_pi(roll), wrap_pi(pitch), wrap_pi(yaw)


class DeployMarkers:
    """Red current intent, green robot KP, orange Mapper-B future (env 0)."""

    def __init__(self, env, enabled: bool = True):
        self.enabled = enabled
        self._viz = None
        if not enabled:
            return
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/DeployPipeline",
                markers={
                    "intent": sim_utils.SphereCfg(
                        radius=0.035,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.15, 0.1)),
                    ),
                    "robot": sim_utils.SphereCfg(
                        radius=0.028,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.85, 0.25)),
                    ),
                    "future": sim_utils.SphereCfg(
                        radius=0.018,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.15)),
                    ),
                    "foot": sim_utils.SphereCfg(
                        radius=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.7, 1.0)),
                    ),
                },
            )
            self._viz = VisualizationMarkers(cfg)
            self._viz.set_visibility(True)
            env.unwrapped._deploy_pipeline_viz = self._viz
        except Exception as exc:
            print(f"[deploy] markers disabled: {exc}", flush=True)
            self.enabled = False

    def update(self, intent_w, robot_w, future_w, foot_w):
        if not self.enabled or self._viz is None:
            return
        pts = []
        proto = []
        def _add(xyz, name):
            p = xyz.reshape(-1, 3)
            pts.append(p)
            proto.extend([name] * int(p.shape[0]))
        _add(intent_w[:1], "intent")
        _add(robot_w[:1, :N_VISIBLE], "robot")
        if future_w is not None:
            _add(future_w[:1], "future")
        if foot_w is not None:
            _add(foot_w[:1], "foot")
        xyz = torch.cat(pts, dim=0)
        quat = torch.zeros((xyz.shape[0], 4), device=xyz.device, dtype=xyz.dtype)
        quat[:, 0] = 1.0
        marker_indices = torch.tensor(
            [{"intent": 0, "robot": 1, "future": 2, "foot": 3}[n] for n in proto],
            device=xyz.device,
            dtype=torch.long,
        )
        try:
            self._viz.visualize(translations=xyz, orientations=quat, marker_indices=marker_indices)
        except TypeError:
            self._viz.visualize(translations=xyz, orientations=quat)


class ControllerPipeline:
    """The graph that must stay identical when swapping IsaacBackend → UnitreeG1Backend.

    Intent packet (NPZ or UDP) → CausalIntentBuffer → Mapper-B → obs[:300].
    Isaac proprio (obs[300:750]) is never rebuilt here.
    """

    def __init__(
        self,
        policy,
        injector: CausalFutureInjector,
        watchdog: DeadManSwitch,
        obs_normalizer=None,
        frontend: "IntentFrontend | None" = None,
    ):
        self.policy = policy
        self.injector = injector
        self.watchdog = watchdog
        self.obs_normalizer = obs_normalizer
        self.frontend = frontend

    @torch.no_grad()
    def act(self, obs, env, vis=None, packet_w=None) -> tuple[torch.Tensor, dict]:
        if self.frontend is not None:
            if packet_w is None:
                packet_w = current_packet_world(env)
            self.frontend.ingest(packet_w, hand_noise_m=float(getattr(self.injector, "hand_noise_m", 0.0) or 0.0))
            patched = self.frontend.build(obs, env, vis)
        else:
            patched = self.injector.patch_policy_obs(obs, env, "mapper")
        return self._infer(patched, env)

    @torch.no_grad()
    def act_prepatched(self, patched, env, drop_streak=None, intent_cur_w=None) -> tuple[torch.Tensor, dict]:
        return self._infer(patched, env, drop_streak=drop_streak, intent_cur_w=intent_cur_w)

    @torch.no_grad()
    def _infer(self, patched, env, drop_streak=None, intent_cur_w=None) -> tuple[torch.Tensor, dict]:
        if self.obs_normalizer is not None:
            patched = self.obs_normalizer(patched)
        info = self.policy.inspect_inference(patched)
        cmd = _motion(env)
        n = int(cmd.num_envs)
        ident = torch.zeros((n, 4), device=cmd.robot_anchor_quat_w.device, dtype=cmd.robot_anchor_quat_w.dtype)
        ident[:, 0] = 1.0
        roll, pitch, _yaw = rel_rpy(ident, cmd.robot_anchor_quat_w)
        tilt = torch.maximum(roll.abs(), pitch.abs())
        q, hold, reasons = self.watchdog.check(
            action=info["action"],
            drop_streak=getattr(self.injector, "_drop_streak", None) if drop_streak is None else drop_streak,
            intent_cur_w=self.injector.last_cur_w if intent_cur_w is None else intent_cur_w,
            tilt=tilt,
        )
        info["action"] = q
        info["hold"] = hold
        info["hold_reason"] = reasons
        info["tilt"] = tilt
        info["robot_roll"] = roll
        info["robot_pitch"] = pitch
        return q, info


class IntentFrontend:
    """NPZ and UDP both end here. Only the source of ``packet_w`` differs."""

    def __init__(self, injector: CausalFutureInjector, delay_steps: int = 0):
        self.injector = injector
        self.buffer = CausalIntentBuffer(delay_steps=delay_steps)

    def reset(self, packet_w: torch.Tensor, delay_steps: int | None = None) -> None:
        self.buffer.reset(packet_w, delay_steps=delay_steps)

    def ingest(self, packet_w: torch.Tensor, hand_noise_m: float = 0.0) -> torch.Tensor:
        pts = packet_w.reshape(-1, N_VISIBLE, 3)
        if hand_noise_m > 0.0:
            pts = pts.clone()
            pts[:, 1:3, :] = pts[:, 1:3, :] + torch.randn_like(pts[:, 1:3, :]) * float(hand_noise_m)
        return self.buffer.push(pts)

    def build(self, obs: torch.Tensor, env, vis: torch.Tensor | None) -> torch.Tensor:
        hist = self.buffer.hist()
        return self.injector.patch_from_world_hist(obs, env, hist, vis=vis, mode="mapper")


def physical_fall(env, use: int, z_min: float = 0.40, ori_rad: float = 1.2):
    """Robot-centric fall. Never compares against clip GT torso."""
    cmd = _motion(env)
    z = cmd.robot_anchor_pos_w[:use, 2]
    n = int(use)
    ident = torch.zeros((n, 4), device=z.device, dtype=cmd.robot_anchor_quat_w.dtype)
    ident[:, 0] = 1.0
    roll, pitch, _yaw = rel_rpy(ident, cmd.robot_anchor_quat_w[:use])
    fail_z = z < float(z_min)
    fail_ori = (roll.abs() > float(ori_rad)) | (pitch.abs() > float(ori_rad))
    return fail_z, fail_ori, z, roll, pitch


MASK_VIS = {
    "torso": (True, False, False),
    "head_left": (True, True, False),
    "head_right": (True, False, True),
    "vr": (True, True, True),
}


def vis_from_mask(name: str, device, n_env: int = 1) -> torch.Tensor:
    flags = MASK_VIS.get(name, MASK_VIS["vr"])
    return torch.tensor([flags], device=device, dtype=torch.bool).expand(n_env, -1).contiguous()


class WorldIntentRing:
    """Causal 3-point world poses at Mapper-B history offsets (50 Hz ticks)."""

    def __init__(self):
        self.max_history = -min(NONPOS_OFFSETS)
        self._w: list[np.ndarray] = []

    def reset(self, pts: np.ndarray) -> None:
        t = np.asarray(pts, dtype=np.float64).reshape(3, 3)
        self._w = [t.copy() for _ in range(self.max_history + 1)]

    def push(self, pts: np.ndarray) -> None:
        self._w.append(np.asarray(pts, dtype=np.float64).reshape(3, 3).copy())
        if len(self._w) > self.max_history + 1:
            self._w = self._w[-(self.max_history + 1) :]

    def hist_world(self) -> np.ndarray:
        out = []
        for o in NONPOS_OFFSETS:
            if o >= 0:
                out.append(self._w[-1])
            else:
                idx = len(self._w) - 1 + o
                out.append(self._w[max(idx, 0)])
        return np.stack(out, axis=0)


class OpenXRLoopback:
    """JSON v1 UDP: same packets as Quest / mock_vr_sender / Unity client."""

    def __init__(self, port: int = 15151):
        import sys
        from pathlib import Path as _P

        root = _P("/data/home/chenxiangyu/victor/TriTrack")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from tritrack.sources.openxr_udp import OpenXRUDPSource  # noqa: WPS433

        self.port = int(port)
        self.source = OpenXRUDPSource(host="127.0.0.1", port=self.port, timeout_s=0.15)
        self._sock = __import__("socket").socket(
            __import__("socket").AF_INET, __import__("socket").SOCK_DGRAM
        )
        self._encode = None

    def encode(self, state) -> bytes:
        if self._encode is None:
            from tritrack.sources.openxr_udp import encode_packet  # noqa: WPS433

            self._encode = encode_packet
        return self._encode(state)

    def send(self, state) -> None:
        self._sock.sendto(self.encode(state), ("127.0.0.1", self.port))

    def poll_after_send(self, timeout_s: float = 0.25):
        import time

        t0 = time.monotonic()
        latest = None
        while time.monotonic() - t0 < timeout_s:
            got = self.source.poll()
            if got is not None:
                latest = got
                break
            time.sleep(0.0004)
        return latest

    def close(self) -> None:
        try:
            self.source.close()
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass


class MockVrThread:
    """In-process stand-in for scripts/mock_vr_sender.py (same JSON v1)."""

    def __init__(self, loopback: OpenXRLoopback, rate_hz: float = 50.0):
        import threading

        self.loopback = loopback
        self.rate_hz = float(rate_hz)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.n_sent = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        import math
        import time

        from tritrack.intent.state import IntentState, SE3

        ident = np.array([1.0, 0.0, 0.0, 0.0])
        t0 = time.time()
        dt = 1.0 / max(self.rate_hz, 1.0)
        while not self._stop.is_set():
            t = time.time() - t0
            head = SE3([0.05 * math.sin(0.5 * t), 0.0, 1.60 + 0.02 * math.sin(1.1 * t)], ident)
            lh = SE3(
                [0.25 + 0.10 * math.sin(1.5 * t), 0.30, 1.05 + 0.10 * math.cos(1.5 * t)], ident
            )
            rh = SE3(
                [
                    0.25 + 0.10 * math.sin(1.5 * t + math.pi),
                    -0.30,
                    1.05 + 0.10 * math.cos(1.5 * t + math.pi),
                ],
                ident,
            )
            self.loopback.send(IntentState(time.time(), head, lh, rh))
            self.n_sent += 1
            self._stop.wait(dt)


class OpenXRIntentDriver:
    """UDP JSON v1 → Mapper-B KP overlay. Isaac proprio (obs[300:750]) is untouched."""

    def __init__(self, injector: CausalFutureInjector, loopback: OpenXRLoopback, canon: str = "world"):
        self.injector = injector
        self.loopback = loopback
        self.canon_mode = canon
        self.ring = WorldIntentRing()
        self._canon = None
        self._packer = None
        self.last_intent = None
        self.last_pts_w: np.ndarray | None = None
        self.packets = 0
        self.misses = 0

    def reset(self, first_pts_w: np.ndarray | None = None) -> None:
        self.ring = WorldIntentRing()
        if first_pts_w is not None:
            self.ring.reset(first_pts_w)
            self.last_pts_w = np.asarray(first_pts_w, dtype=np.float64).reshape(3, 3)
        else:
            self.last_pts_w = None
        self.packets = 0
        self.misses = 0
        if self.canon_mode == "headset":
            from tritrack.intent.canonicalizer import IntentCanonicalizer
            from tritrack.intent.packer import SparseKpPacker

            self._canon = IntentCanonicalizer()
            self._packer = SparseKpPacker()

    def ingest_state(self, intent) -> np.ndarray:
        pts = np.stack([p.pos for p in intent.poses])
        self.last_intent = intent
        self.last_pts_w = pts
        self.packets += 1
        return pts

    def overlay_world(self, obs, env, vis: torch.Tensor, pts_w: np.ndarray) -> torch.Tensor:
        if len(self.ring._w) == 0:
            self.ring.reset(pts_w)
        else:
            self.ring.push(pts_w)
        hist = torch.as_tensor(self.ring.hist_world(), device=obs.device, dtype=obs.dtype)
        hist = hist.unsqueeze(0).expand(obs.shape[0], -1, -1, -1).contiguous()
        noise = float(getattr(self.injector, "hand_noise_m", 0.0) or 0.0)
        if noise > 0.0:
            hist = hist.clone()
            hist[:, :, 1:3, :] = hist[:, :, 1:3, :] + torch.randn_like(hist[:, :, 1:3, :]) * noise
        lag = int(getattr(self.injector, "latency_steps", 0) or 0)
        if lag > 0 and len(self.ring._w) > 1:
            delayed = self.ring._w[max(0, len(self.ring._w) - 1 - lag)]
            hist = hist.clone()
            hist[:, -1] = torch.as_tensor(delayed, device=obs.device, dtype=obs.dtype)
        return self.injector.patch_from_world_hist(obs, env, hist, vis=vis, mode="mapper")

    def overlay_headset(self, obs, env, vis: torch.Tensor, intent) -> torch.Tensor:
        from tritrack.intent.packer import SLOT_OFFSETS, VISIBLE

        from causal_future import _points_from_frame

        cmd = _motion(env)
        if intent is not None:
            if self._canon._zero is None:
                self._canon.calibrate(intent)
                tgt = self._canon(intent)
                self._packer.reset(tgt)
            else:
                tgt = self._canon(intent)
                self._packer.push(tgt)
        robot_kp = robot_kp_abs_anchor(env)[0].detach().cpu().numpy()
        kp, mask = self._packer.build(robot_kp)
        mapper = self.injector.mapper
        feats, cur = self._packer.mapper_features(robot_kp, in_dim=int(mapper.in_dim))
        vis_np = vis[0].detach().cpu().numpy().astype(bool)
        hist = feats[:72].reshape(8, 3, 3).copy()
        hist[:, ~vis_np, :] = 0.0
        feats = feats.copy()
        feats[:72] = hist.reshape(-1)
        cur = cur.copy().reshape(3, 3)
        cur[~vis_np] = 0.0
        with torch.no_grad():
            pred = mapper(
                torch.as_tensor(feats, dtype=torch.float32, device=obs.device).unsqueeze(0),
                torch.as_tensor(cur.reshape(-1), dtype=torch.float32, device=obs.device).unsqueeze(0),
            )
        pred = pred.squeeze(0).cpu().numpy().reshape(7, 3, 3)
        fi = 0
        for li, off in enumerate(SLOT_OFFSETS):
            if off <= 0:
                continue
            for vi, bi in enumerate(VISIBLE):
                if vis_np[vi]:
                    kp[li, bi] = pred[fi, vi] - robot_kp[bi]
            fi += 1
        for bi in range(3):
            if not vis_np[bi]:
                kp[:, bi] = np.nan
                mask[:, bi] = 1.0
        out = obs.clone()
        out[:, :225] = torch.as_tensor(kp.reshape(-1), device=obs.device, dtype=obs.dtype)
        out[:, 225:300] = torch.as_tensor(mask.reshape(-1), device=obs.device, dtype=obs.dtype)
        tgt_ra = torch.as_tensor(self._packer._targets[-1], device=obs.device, dtype=obs.dtype).view(1, 3, 3)
        self.injector.last_cur_w = _points_from_frame(
            tgt_ra, cmd.robot_anchor_pos_w[:1], cmd.robot_anchor_quat_w[:1]
        )
        pred_ra = torch.as_tensor(pred, device=obs.device, dtype=obs.dtype).view(1, 7, 3, 3)
        self.injector.last_pred_abs = pred_ra
        self.injector.last_pred_w = _points_from_frame(
            pred_ra, cmd.robot_anchor_pos_w[:1], cmd.robot_anchor_quat_w[:1]
        )
        return out


def write_deploy_html(out: Path, payload: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    html = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Deploy sim · model_50000</title>
<style>
  :root { --bg:#141414; --panel:#1c1c1c; --line:#2a2a2a; --text:#e8e8e8; --muted:#8a8a8a; --red:#e85d4c; --green:#3dba7a; --orange:#e09a3d; }
  * { box-sizing:border-box; }
  html,body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif; }
  body { padding:20px 24px 48px; }
  h1 { font-size:20px; margin:0 0 4px; }
  .muted { color:var(--muted); }
  select { background:#222; color:var(--text); border:1px solid var(--line); padding:6px 8px; }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:8px; margin:14px 0; }
  .stat { background:var(--panel); border:1px solid var(--line); padding:10px 12px; }
  .stat .v { font:600 20px/1.1 ui-monospace,monospace; }
  .stat .k { color:var(--muted); font-size:12px; margin-top:4px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .chart { background:var(--panel); border:1px solid var(--line); padding:10px 12px; }
  .chart h3 { margin:0 0 6px; font-size:13px; }
  canvas { width:100%; height:260px; display:block; }
  .leg { font-size:12px; color:var(--muted); margin-top:6px; }
  table { width:100%; border-collapse:collapse; margin-top:16px; font-size:13px; }
  th,td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; }
  td.num { font-family:ui-monospace,monospace; text-align:right; }
  .fail { color:var(--red); }
  .ok { color:var(--green); }
</style></head><body>
<h1>Deploy pipeline · causal Mapper-B · model_50000</h1>
<div class="muted" id="meta"></div>
<p><label>Clip </label><select id="sel"></select></p>
<div class="stats" id="stats"></div>
<div class="grid">
  <div class="chart"><h3>Top (X–Y) · red intent · green robot · orange Mapper-B future</h3><canvas id="xy"></canvas></div>
  <div class="chart"><h3>Side (X–Z)</h3><canvas id="xz"></canvas></div>
  <div class="chart"><h3>Tracking error (m)</h3><canvas id="err"></canvas></div>
  <div class="chart"><h3>Safety / residual</h3><canvas id="hud"></canvas></div>
</div>
<div class="leg">Solid = robot · dashed = current intent (causal stream, no GT future) · orange dots = Mapper-B 0.1–0.5 s at last frame</div>
<h3>All clips</h3>
<table id="tab"></table>
<script>
const D = __DATA__;
const fmt = (x,d=3) => (x==null||Number.isNaN(x)) ? "—" : Number(x).toFixed(d);
const pct = (x) => (x==null||Number.isNaN(x)) ? "—" : (100*x).toFixed(1)+"%";
document.getElementById("meta").textContent = D.meta;
const sel = document.getElementById("sel");
D.clips.forEach((c,i)=>{ const o=document.createElement("option"); o.value=i; o.textContent=`${c.task} / ${c.mask} / ${c.name}`; sel.appendChild(o); });
function drawTrails(canvas, clip, a, b) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width = canvas.clientWidth * devicePixelRatio;
  const H = canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.clearRect(0,0,W,H);
  const pad = 28*devicePixelRatio;
  const pts = [...clip.robot.flat(), ...clip.intent.flat(), ...(clip.future||[]).flat()];
  let mnA=Infinity,mxA=-Infinity,mnB=Infinity,mxB=-Infinity;
  for (const p of pts) { mnA=Math.min(mnA,p[a]); mxA=Math.max(mxA,p[a]); mnB=Math.min(mnB,p[b]); mxB=Math.max(mxB,p[b]); }
  const span = Math.max(mxA-mnA, mxB-mnB, 0.25);
  const cx=(mnA+mxA)/2, cy=(mnB+mxB)/2;
  const X = v => pad + (v-(cx-span/2))/span*(W-2*pad);
  const Y = v => H-pad - (v-(cy-span/2))/span*(H-2*pad);
  ctx.strokeStyle="#2a2a2a"; ctx.strokeRect(pad,pad,W-2*pad,H-2*pad);
  const colors=["#e85d4c","#3dba7a","#d4a017"];
  for (let k=0;k<3;k++) {
    const rob=clip.robot[k], intent=clip.intent[k];
    if (!rob) continue;
    ctx.strokeStyle=colors[k]; ctx.lineWidth=1.7*devicePixelRatio; ctx.setLineDash([]);
    ctx.beginPath(); rob.forEach((p,i)=>{ const x=X(p[a]),y=Y(p[b]); i?ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke();
    if (intent) {
      ctx.setLineDash([4*devicePixelRatio,4*devicePixelRatio]); ctx.beginPath();
      intent.forEach((p,i)=>{ const x=X(p[a]),y=Y(p[b]); i?ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke();
    }
    const r=rob.at(-1); ctx.setLineDash([]); ctx.fillStyle="#3dba7a";
    ctx.beginPath(); ctx.arc(X(r[a]),Y(r[b]),3.4*devicePixelRatio,0,7); ctx.fill();
    if (intent) { const g=intent.at(-1); ctx.strokeStyle="#e85d4c"; ctx.beginPath(); ctx.arc(X(g[a]),Y(g[b]),6*devicePixelRatio,0,7); ctx.stroke(); }
  }
  if (clip.future) {
    clip.future.forEach((p,i) => {
      const t = i / Math.max(clip.future.length-1,1);
      ctx.fillStyle = `rgba(224,154,61,${0.35+0.65*t})`;
      ctx.beginPath(); ctx.arc(X(p[a]),Y(p[b]),2.6*devicePixelRatio,0,7); ctx.fill();
    });
  }
}
function drawSeries(canvas, series, yMaxHint) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width = canvas.clientWidth * devicePixelRatio;
  const H = canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.clearRect(0,0,W,H);
  const pad = {l:42*devicePixelRatio,r:10*devicePixelRatio,t:8*devicePixelRatio,b:20*devicePixelRatio};
  let ymax = yMaxHint || 0.01;
  for (const s of series) for (const y of s.ys) if (Number.isFinite(y)) ymax = Math.max(ymax, y);
  ymax *= 1.12;
  const n = series[0].ys.length;
  const X = i => pad.l + i/Math.max(n-1,1) * (W-pad.l-pad.r);
  const Y = v => pad.t + (1-v/ymax) * (H-pad.t-pad.b);
  ctx.strokeStyle="#2a2a2a"; ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.lineTo(W-pad.r,H-pad.b); ctx.stroke();
  for (const s of series) {
    ctx.strokeStyle=s.color; ctx.lineWidth=1.5*devicePixelRatio; ctx.beginPath();
    s.ys.forEach((y,i)=>{ const x=X(i), yy=Y(y); i?ctx.lineTo(x,yy):ctx.moveTo(x,yy); });
    ctx.stroke();
  }
}
function show(i) {
  const c = D.clips[i];
  const cells = [
    ["SR@5cm", pct(c.sr5)], ["Fall", c.fail ? "YES" : "no"],
    ["Torso err", fmt(c.e_torso)+" m"], ["Wrist err", fmt(c.e_wrist)+" m"],
    ["Roll/pitch", fmt(c.roll,2)+" / "+fmt(c.pitch,2)],
    ["||Δz|| mean", fmt(c.dz_mean,3)], ["Hold steps", c.hold_n],
    ["Mapper 0.5s", fmt(c.mapper_h05,3)+" m"],
  ];
  document.getElementById("stats").innerHTML = cells.map(([k,v])=>`<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
  drawTrails(document.getElementById("xy"), c, 0, 1);
  drawTrails(document.getElementById("xz"), c, 0, 2);
  drawSeries(document.getElementById("err"), [
    {ys:c.e_torso_t, color:"#e85d4c"}, {ys:c.e_lw_t, color:"#3dba7a"}, {ys:c.e_rw_t, color:"#d4a017"},
  ], 0.15);
  drawSeries(document.getElementById("hud"), [
    {ys:c.pitch_t, color:"#e85d4c"}, {ys:c.dz_t, color:"#6ea8fe"}, {ys:c.hold_t, color:"#e09a3d"},
  ], 1.0);
}
sel.onchange = () => show(+sel.value);
const rows = D.clips.map(c => `<tr>
  <td>${c.task}</td><td>${c.mask}</td><td>${c.name}</td>
  <td class="num">${pct(c.sr5)}</td>
  <td class="num">${fmt(c.e_torso)}</td>
  <td class="num">${fmt(c.e_wrist)}</td>
  <td class="num">${fmt(c.dz_mean,3)}</td>
  <td class="${c.fail?"fail":"ok"}">${c.fail ? (c.fail_reason||"fail") : "ok"}</td>
</tr>`).join("");
document.getElementById("tab").innerHTML = `<tr><th>Task</th><th>Mask</th><th>Clip</th><th>SR@5</th><th>Torso</th><th>Wrist</th><th>||Δz||</th><th>Gate</th></tr>`+rows;
show(0);
</script></body></html>
"""
    out.write_text(html.replace("__DATA__", json.dumps(payload)), encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
