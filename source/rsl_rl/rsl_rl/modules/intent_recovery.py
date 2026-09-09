# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""P2-R intent-recovery: causal gate + zero-init tangent residual ``r_η``.

Not a terrain residual. Not an inner controller. ``v_E`` is an observation for
``r_η`` (via ``ė``) and is **not** an authority trigger.

Gate (runtime, causal, ``≤ t`` only):

    R_E = (E − Q50_E) / (Q90_E − Q50_E)
    R_S = (S − Q50_S) / (Q90_S − Q50_S)
    R   = max(R_E, R_S)

``E`` is RMS of *visible* intent keypoints (torso / L wrist / R wrist).
``S`` is ``|v_root,z|`` (stability). The 0.5 s self-recover label from Step 3
is offline analysis only — it never enters this module.

Trigger: ``R ≥ 1`` for ``persist_on`` frames (default 3 = 60 ms) → ``active=1``.
Release: ``R < R_off`` for ``persist_off`` frames → ``active=0``.
Authority is continuous:

    α = active · clip((R − R_off) / (R_full − R_off), 0, 1)

so the first ON frame already has ``α > 0`` (because ``R_off < 1``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.modules.terrain_residual import apply_tangent_correction


# Paper freeze (Loco Flat+Light healthy envelope, metres / m/s).
Q50_E = 0.046
Q90_E = 0.130
Q50_S = 0.041
Q90_S = 0.241

THETA_MAX_DEG = 5.0
DEFAULT_R_MAX = math.tan(math.radians(THETA_MAX_DEG))
DEFAULT_R_OFF = 0.6
DEFAULT_R_FULL = 2.0
DEFAULT_PERSIST_ON = 3
DEFAULT_PERSIST_OFF = 3
DT = 0.02
# R1a-2a: clip(R_E,t − R_E,t+1). Terminal transitions are zeroed, not clipped.
DEFAULT_PROGRESS_CLIP = 1.0

# Visible intent: torso, left wrist, right wrist. Ankles never enter E / r_η obs.
N_TASK_BODIES = 3
E_DIM = N_TASK_BODIES * 3  # 9


def risk_scores(
    e_m: torch.Tensor,
    s: torch.Tensor,
    q50_e: float = Q50_E,
    q90_e: float = Q90_E,
    q50_s: float = Q50_S,
    q90_s: float = Q90_S,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stateless ``(R, R_E, R_S)``. ``e_m`` and ``s`` are ``[B]``."""
    den_e = max(float(q90_e) - float(q50_e), 1e-6)
    den_s = max(float(q90_s) - float(q50_s), 1e-6)
    r_e = (e_m - float(q50_e)) / den_e
    r_s = (s - float(q50_s)) / den_s
    r = torch.maximum(r_e, r_s)
    return r, r_e, r_s


def extract_visible_task_error(
    kp_lhn: torch.Tensor,
    kp_mask: torch.Tensor,
    slot0_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Visible intent error from encoder obs.

    ``kp_lhn`` must be in **metres** (denormalized). Offset-0 torso in the
    robot-anchor frame is the torso tracking error. Wrists at offset 0 are
    commanded pose, not tracking error — they are zeroed unless visible; R1a
    is torso-only so only the torso term enters ``E``.

    ``E`` is the mean L2 of visible bodies, matching
    ``error_body_pos_w_visible`` (Step 3). Encoder mask: **1 = invisible**.

    Returns ``(e [B, 9], vis [B, 3], E [B])``.
    """
    slot = int(slot0_idx)
    e_b = kp_lhn[..., slot, :N_TASK_BODIES, :]  # [B, 3, 3]
    vis = (kp_mask[..., :N_TASK_BODIES] < 0.5).to(dtype=e_b.dtype)
    e_b = torch.nan_to_num(e_b, nan=0.0) * vis.unsqueeze(-1)
    e = e_b.reshape(*e_b.shape[:-2], E_DIM)
    per = e_b.norm(dim=-1)
    nvis = vis.sum(dim=-1).clamp_min(1e-8)
    e_mean = (per * vis).sum(dim=-1) / nvis
    e_mean = torch.where(vis.sum(dim=-1) > 0, e_mean, torch.zeros_like(e_mean))
    return e, vis, e_mean


def recovery_progress_reward(
    active_t: torch.Tensor,
    r_e_t: torch.Tensor,
    r_e_next: torch.Tensor,
    done: torch.Tensor,
    clip_p: float = DEFAULT_PROGRESS_CLIP,
) -> torch.Tensor:
    """Active-only clipped ``R_E`` progress. **Terminal transitions are 0.**

    Isaac vectorized envs reset on ``done``, so the observation at ``t+1`` is the
    new episode (``R_E`` often near healthy). Using that next-state would credit
    a fall with ``R_{E,t}-R_{E,reset} ≫ 0``. Canonical fall penalty handles
    failure; progress does not.

    ``r_prog = 1[active_t ∧ ¬done] clip(R_{E,t} − R_{E,t+1}, −c_p, c_p)``.
    """
    active = active_t.reshape(-1).to(dtype=torch.bool)
    done_b = done.reshape(-1).to(dtype=torch.bool)
    live = active & ~done_b
    delta = (r_e_t.reshape(-1) - r_e_next.reshape(-1)).clamp(-float(clip_p), float(clip_p))
    return torch.where(live, delta, torch.zeros_like(delta))


def recovery_progress_reset_leak(
    active_t: torch.Tensor,
    r_e_t: torch.Tensor,
    r_e_next: torch.Tensor,
    done: torch.Tensor,
    clip_p: float = DEFAULT_PROGRESS_CLIP,
) -> torch.Tensor:
    """Counterfactual: progress we *would* have paid on ``done ∧ active`` if we
    used post-reset ``R_E``. Must not enter the PPO reward. Logger-only."""
    active = active_t.reshape(-1).to(dtype=torch.bool)
    done_b = done.reshape(-1).to(dtype=torch.bool)
    leak = active & done_b
    delta = (r_e_t.reshape(-1) - r_e_next.reshape(-1)).clamp(-float(clip_p), float(clip_p))
    return torch.where(leak, delta, torch.zeros_like(delta))


class RecoveryRiskGate(nn.Module):
    """Hysteresis gate. No learned parameters. Future labels never enter."""

    def __init__(
        self,
        q50_e: float = Q50_E,
        q90_e: float = Q90_E,
        q50_s: float = Q50_S,
        q90_s: float = Q90_S,
        r_off: float = DEFAULT_R_OFF,
        r_full: float = DEFAULT_R_FULL,
        persist_on: int = DEFAULT_PERSIST_ON,
        persist_off: int = DEFAULT_PERSIST_OFF,
        s_enabled: bool = False,
    ):
        super().__init__()
        if not (0.0 < float(r_off) < 1.0):
            raise ValueError(f"R_off must be in (0, 1), got {r_off}")
        if float(r_full) <= float(r_off):
            raise ValueError(f"R_full must exceed R_off, got {r_full} <= {r_off}")
        self.q50_e = float(q50_e)
        self.q90_e = float(q90_e)
        self.q50_s = float(q50_s)
        self.q90_s = float(q90_s)
        self.r_off = float(r_off)
        self.r_full = float(r_full)
        self.persist_on = int(persist_on)
        self.persist_off = int(persist_off)
        self.s_enabled = bool(s_enabled)
        self._active: torch.Tensor | None = None
        self._on_cnt: torch.Tensor | None = None
        self._off_cnt: torch.Tensor | None = None

    def extra_repr(self) -> str:
        return (
            f"Q_E=({self.q50_e:.3f},{self.q90_e:.3f}) Q_S=({self.q50_s:.3f},{self.q90_s:.3f}) "
            f"R_off={self.r_off} R_full={self.r_full} persist={self.persist_on}/{self.persist_off} "
            f"S={'on' if self.s_enabled else 'off'}"
        )

    def _ensure(self, n: int, device, dtype) -> None:
        if self._active is not None and int(self._active.shape[0]) == int(n) and self._active.device == device:
            return
        self._active = torch.zeros(n, dtype=torch.bool, device=device)
        self._on_cnt = torch.zeros(n, dtype=torch.long, device=device)
        self._off_cnt = torch.zeros(n, dtype=torch.long, device=device)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Clear hysteresis. ``dones`` True → that env starts a new episode."""
        if self._active is None:
            return
        if dones is None:
            self._active.zero_()
            self._on_cnt.zero_()
            self._off_cnt.zero_()
            return
        d = dones.reshape(-1).to(device=self._active.device, dtype=torch.bool)
        if int(d.shape[0]) != int(self._active.shape[0]):
            self._active.zero_()
            self._on_cnt.zero_()
            self._off_cnt.zero_()
            return
        self._active[d] = False
        self._on_cnt[d] = 0
        self._off_cnt[d] = 0

    def scores(self, e_m: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r, r_e, r_s = risk_scores(e_m, s, self.q50_e, self.q90_e, self.q50_s, self.q90_s)
        if not self.s_enabled:
            return r_e, r_e, r_s
        return r, r_e, r_s

    def authority_from_r(self, r: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        span = max(self.r_full - self.r_off, 1e-6)
        return active.to(dtype=r.dtype) * ((r - self.r_off) / span).clamp(0.0, 1.0)

    def step(
        self,
        e_m: torch.Tensor,
        s: torch.Tensor,
        *,
        mutate: bool = True,
    ) -> dict[str, torch.Tensor]:
        """One causal step. Uses only current ``E,S`` plus per-env hysteresis.

        ``mutate=False``: do not update counters (PPO minibatch / mismatched B).
        Stateless fallback then uses ``active := R ≥ 1`` (no persist) so a
        shuffled batch cannot corrupt rollout hysteresis.
        """
        e_m = e_m.reshape(-1)
        s = s.reshape(-1).to(dtype=e_m.dtype, device=e_m.device)
        r, r_e, r_s = self.scores(e_m, s)
        n = int(e_m.shape[0])
        if mutate:
            self._ensure(n, e_m.device, e_m.dtype)
            above = r >= 1.0
            below = r < self.r_off
            on_cnt = torch.where(above, self._on_cnt + 1, torch.zeros_like(self._on_cnt))
            turn_on = (~self._active) & (on_cnt >= self.persist_on)
            off_cnt = torch.where(self._active & below, self._off_cnt + 1, torch.zeros_like(self._off_cnt))
            turn_off = self._active & (off_cnt >= self.persist_off)
            active = (self._active | turn_on) & ~turn_off
            self._on_cnt = on_cnt
            self._off_cnt = off_cnt
            self._active = active
        else:
            if self._active is not None and int(self._active.shape[0]) == n:
                active = self._active
            else:
                active = r >= 1.0
        alpha = self.authority_from_r(r, active)
        return {
            "alpha": alpha,
            "active": active.to(dtype=torch.bool),
            "R": r,
            "R_E": r_e,
            "R_S": r_s,
            "E": e_m,
            "S": s,
        }


class RecoveryResidualMLP(nn.Module):
    """``r_η(e, ė, M, o_prop, sg[z_nom]) → R^{16}``. Last Linear is zero-init."""

    def __init__(
        self,
        proprio_dim: int,
        latent_dim: int = 16,
        hidden: tuple[int, int] = (256, 128),
    ):
        super().__init__()
        self.e_dim = E_DIM
        self.mask_dim = N_TASK_BODIES
        self.proprio_dim = int(proprio_dim)
        self.latent_dim = int(latent_dim)
        din = self.e_dim + self.e_dim + self.mask_dim + self.proprio_dim + self.latent_dim
        h1, h2 = int(hidden[0]), int(hidden[1])
        self.net = nn.Sequential(
            nn.Linear(din, h1),
            nn.ELU(),
            nn.Linear(h1, h2),
            nn.ELU(),
            nn.Linear(h2, self.latent_dim),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.in_dim = din

    def pack(
        self,
        e: torch.Tensor,
        e_dot: torch.Tensor,
        vis: torch.Tensor,
        proprio: torch.Tensor,
        z_nom: torch.Tensor,
    ) -> torch.Tensor:
        prop = proprio.reshape(proprio.shape[0], -1)
        return torch.cat([e, e_dot, vis, prop, z_nom], dim=-1)

    def forward(
        self,
        e: torch.Tensor,
        e_dot: torch.Tensor,
        vis: torch.Tensor,
        proprio: torch.Tensor,
        z_nom: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(self.pack(e, e_dot, vis, proprio, z_nom))


def apply_recovery_correction(
    z_nom: torch.Tensor,
    r_raw: torch.Tensor,
    alpha: torch.Tensor,
    r_max: float = DEFAULT_R_MAX,
) -> dict[str, torch.Tensor]:
    """Tangent project, cap at ``tan 5°``, scale by ``α``, re-normalize."""
    z = F.normalize(z_nom, dim=-1, eps=1e-8)
    r_par = (r_raw * z).sum(dim=-1, keepdim=True) * z
    r_perp = r_raw - r_par
    z_exec, r_bar = apply_tangent_correction(z_nom, r_raw, r_max=float(r_max), alpha=alpha)
    n_raw = r_perp.norm(dim=-1)
    n_bar = r_bar.norm(dim=-1)
    rho = n_raw / max(float(r_max), 1e-8)
    cos = (z * z_exec).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos)
    return {
        "z_exec": z_exec,
        "r_raw": r_raw,
        "r_perp": r_perp,
        "r_bar": r_bar,
        "r_raw_norm": r_raw.norm(dim=-1),
        "r_perp_norm": n_raw,
        "r_bar_norm": n_bar,
        "rho_r": rho,
        "theta": theta,
        "alpha": alpha.reshape(-1).to(dtype=z.dtype),
    }


class RecoveryRolloutTracker:
    """Per-env Detect→Recover→Release stats. No future labels, no terrain identity.

    Horizons are 0.25 / 0.5 / 1.0 s at 50 Hz. Flushed once per PPO iteration.
    """

    GROUPS = ("flat", "light", "slope", "steps")
    H25 = 13
    H50 = 25
    H100 = 50

    def __init__(self, dt: float = DT):
        self.dt = float(dt)
        self._n = 0
        self._acc: dict[str, list[float]] | None = None

    def _ensure(self, n: int, device) -> None:
        if self._n == n and getattr(self, "active_prev", None) is not None:
            if self.active_prev.device == device:
                return
        self._n = int(n)
        z = torch.zeros(n, device=device)
        b = torch.zeros(n, dtype=torch.bool, device=device)
        lng = torch.zeros(n, dtype=torch.long, device=device)
        self.active_prev = b.clone()
        self.in_event = b.clone()
        self.released = b.clone()
        self.got25 = b.clone()
        self.got50 = b.clone()
        self.got100 = b.clone()
        self.age = lng.clone()
        self.n_act = lng.clone()
        self.e0 = z.clone()
        self.auc = z.clone()
        self.ep_e_sum = z.clone()
        self.ep_n = lng.clone()
        self.ep_trig = lng.clone()
        self._zero_acc()

    def _zero_acc(self) -> None:
        acc: dict[str, list[float]] = {
            "dE25": [0.0, 0.0],
            "dE50": [0.0, 0.0],
            "dE100": [0.0, 0.0],
            "Trec": [0.0, 0.0],
            "aucE": [0.0, 0.0],
            "SRrec_ok": [0.0],
            "SRrec_n": [0.0],
            "pcap_n": [0.0],
            "pcap_d": [0.0],
            "rho": [0.0, 0.0],
            "theta": [0.0, 0.0],
        }
        for g in self.GROUPS:
            acc[f"duty_{g}"] = [0.0, 0.0]
            acc[f"Tact_{g}"] = [0.0, 0.0]
            acc[f"trig_{g}"] = [0.0, 0.0]
            acc[f"sr5_{g}"] = [0.0, 0.0]
            acc[f"fail_{g}"] = [0.0, 0.0]
            acc[f"alpha_{g}"] = [0.0, 0.0]
            acc[f"rel_{g}"] = [0.0, 0.0]
        self._acc = acc

    def _add(self, key: str, value: torch.Tensor | float, count: torch.Tensor | float | None = None) -> None:
        acc = self._acc
        if acc is None:
            return
        if torch.is_tensor(value):
            v = float(value.detach().float().sum().item())
        else:
            v = float(value)
        acc[key][0] += v
        if count is None:
            return
        if torch.is_tensor(count):
            c = float(count.detach().float().sum().item())
        else:
            c = float(count)
        if len(acc[key]) > 1:
            acc[key][1] += c

    def step(
        self,
        stats: dict[str, torch.Tensor],
        gid: torch.Tensor | None,
        dones: torch.Tensor,
        time_outs: torch.Tensor | None,
    ) -> None:
        e = stats["E"].reshape(-1)
        n = int(e.shape[0])
        self._ensure(n, e.device)
        active = stats["active"].reshape(-1).to(dtype=torch.bool)
        alpha = stats["alpha"].reshape(-1).to(dtype=e.dtype)
        rho = stats["rho_r"].reshape(-1).to(dtype=e.dtype)
        theta = stats.get("theta_deg")
        if theta is None:
            theta = stats.get("theta", e.new_zeros(n))
            theta = theta.reshape(-1).to(dtype=e.dtype) * (180.0 / math.pi)
        else:
            theta = theta.reshape(-1).to(dtype=e.dtype)
        g = gid.reshape(-1).to(device=e.device, dtype=torch.long) if gid is not None else e.new_zeros(n, dtype=torch.long)
        g = g.clamp(0, 3)
        done = dones.reshape(-1).to(device=e.device, dtype=torch.bool)
        tout = (
            time_outs.reshape(-1).to(device=e.device, dtype=torch.bool)
            if time_outs is not None
            else torch.zeros_like(done)
        )
        fail = done & ~tout

        rise = active & ~self.active_prev
        fall = (~active) & self.active_prev

        if bool(rise.any()):
            self.in_event[rise] = True
            self.released[rise] = False
            self.got25[rise] = False
            self.got50[rise] = False
            self.got100[rise] = False
            self.age[rise] = 0
            self.n_act[rise] = 0
            self.e0[rise] = e[rise]
            self.auc[rise] = 0
            self.ep_trig[rise] = self.ep_trig[rise] + 1

        live = self.in_event
        self.age = torch.where(live, self.age + 1, self.age)
        self.n_act = torch.where(active, self.n_act + 1, self.n_act)
        self.auc = torch.where(active, self.auc + e * self.dt, self.auc)

        hit25 = live & ~self.got25 & (self.age == self.H25)
        hit50 = live & ~self.got50 & (self.age == self.H50)
        hit100 = live & ~self.got100 & (self.age == self.H100)
        if bool(hit25.any()):
            self._add("dE25", (e - self.e0)[hit25], hit25)
            self.got25[hit25] = True
        if bool(hit50.any()):
            self._add("dE50", (e - self.e0)[hit50], hit50)
            self.got50[hit50] = True
        if bool(hit100.any()):
            self._add("dE100", (e - self.e0)[hit100], hit100)
            self.got100[hit100] = True

        if bool(fall.any()):
            rel = fall & self.in_event & ~self.released
            if bool(rel.any()):
                self._add("Trec", self.n_act[rel].to(dtype=e.dtype) * self.dt, rel)
                self._add("aucE", self.auc[rel], rel)
                self._acc["SRrec_ok"][0] += float(rel.sum().item())
                self._acc["SRrec_n"][0] += float(rel.sum().item())
                for gi, name in enumerate(self.GROUPS):
                    m = rel & (g == gi)
                    if bool(m.any()):
                        self._add(f"Tact_{name}", self.n_act[m].to(dtype=e.dtype) * self.dt, m)
                        self._add(f"rel_{name}", m.to(dtype=e.dtype).sum(), m.to(dtype=e.dtype).sum())
                self.released[rel] = True
            close = fall & self.got100
            self.in_event[close] = False

        # fail while an event was opened and never released
        fail_open = fail & self.in_event & ~self.released
        if bool(fail_open.any()):
            self._acc["SRrec_n"][0] += float(fail_open.sum().item())
            self._add("aucE", self.auc[fail_open], fail_open)
            for gi, name in enumerate(self.GROUPS):
                m = fail_open & (g == gi)
                if bool(m.any()):
                    self._add(f"Tact_{name}", self.n_act[m].to(dtype=e.dtype) * self.dt, m)
                    self._add(f"rel_{name}", 0.0, float(m.sum().item()))

        act_f = active.to(dtype=e.dtype)
        self._add("pcap_n", (rho > 1.0).to(dtype=e.dtype) * act_f)
        self._add("pcap_d", act_f)
        if bool(active.any()):
            self._add("rho", rho[active], active)
            self._add("theta", theta[active], active)

        self.ep_e_sum = self.ep_e_sum + e
        self.ep_n = self.ep_n + 1
        for gi, name in enumerate(self.GROUPS):
            m = g == gi
            if not bool(m.any()):
                continue
            self._add(f"duty_{name}", act_f[m], m.to(dtype=e.dtype))
            self._add(f"alpha_{name}", alpha[m], m.to(dtype=e.dtype))
            self._add(f"sr5_{name}", (e[m] < 0.05).to(dtype=e.dtype), m.to(dtype=e.dtype))

        if bool(done.any()):
            for gi, name in enumerate(self.GROUPS):
                m = done & (g == gi)
                if not bool(m.any()):
                    continue
                self._add(f"fail_{name}", fail[m].to(dtype=e.dtype), m.to(dtype=e.dtype))
                self._add(f"trig_{name}", self.ep_trig[m].to(dtype=e.dtype), m.to(dtype=e.dtype))
            self.active_prev[done] = False
            self.in_event[done] = False
            self.released[done] = False
            self.got25[done] = False
            self.got50[done] = False
            self.got100[done] = False
            self.age[done] = 0
            self.n_act[done] = 0
            self.e0[done] = 0
            self.auc[done] = 0
            self.ep_e_sum[done] = 0
            self.ep_n[done] = 0
            self.ep_trig[done] = 0

        self.active_prev = active.clone()

    def flush(self) -> dict[str, float]:
        acc = self._acc
        if not acc:
            return {}
        out: dict[str, float] = {}

        def mean_pair(key: str, default: float = float("nan")) -> float:
            s, c = acc[key]
            return (s / c) if c > 0 else default

        out["rec_dE25"] = mean_pair("dE25")
        out["rec_dE50"] = mean_pair("dE50")
        out["rec_dE100"] = mean_pair("dE100")
        out["rec_Trec"] = mean_pair("Trec")
        out["rec_aucE"] = mean_pair("aucE")
        n_sr = acc["SRrec_n"][0]
        out["rec_SRrec"] = (acc["SRrec_ok"][0] / n_sr) if n_sr > 0 else float("nan")
        out["rec_fail_trig"] = (1.0 - out["rec_SRrec"]) if n_sr > 0 else float("nan")
        d = acc["pcap_d"][0]
        out["rec_pcap"] = (acc["pcap_n"][0] / d) if d > 0 else float("nan")
        out["rec_rho"] = mean_pair("rho")
        out["rec_theta"] = mean_pair("theta")
        for g in self.GROUPS:
            out[f"rec_duty_{g}"] = mean_pair(f"duty_{g}")
            out[f"rec_Tact_{g}"] = mean_pair(f"Tact_{g}")
            out[f"rec_trig_{g}"] = mean_pair(f"trig_{g}")
            out[f"rec_sr5_{g}"] = mean_pair(f"sr5_{g}")
            out[f"rec_fail_{g}"] = mean_pair(f"fail_{g}")
            out[f"rec_alpha_{g}"] = mean_pair(f"alpha_{g}")
            out[f"rec_rel_{g}"] = mean_pair(f"rel_{g}")
        self._zero_acc()
        return out

