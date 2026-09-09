# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Latent-space RL actor-critic for finetuning a distilled MUSE-Kp policy.

Design (see ``docs/latent_rl_finetune_plan.md``)
------------------------------------------------
PPO in this repo is action-agnostic: ``PPO.act`` stores whatever ``policy.act``
returns and recomputes ``get_actions_log_prob`` against the policy's
distribution. So latent-space RL = **the RL action IS the latent**; the frozen
decoder is a deterministic transform applied between ``alg.act`` and
``env.step`` (in the runner, gated on ``training_type == "latent_rl"``).

This module wraps a :class:`LatentBottleneckMUSEKp` (``self.muse``) to reuse its
KP-token encoder, MUSE decoder, unit-norm latent geometry, and — crucially — its
slot-aware ``load_state_dict`` so a distilled MUSE-Kp checkpoint warmstarts
cleanly. On top it adds:

- a state-independent Gaussian over the **latent** (decision D1): mean =
  ``normalize(mu_encoder)``; ``log_std`` is a free ``nn.Parameter`` of size
  ``latent_dim``. At the deterministic mean ``act_inference`` reproduces the
  distilled policy exactly (safe start).
- a fresh MLP **critic** on the critic obs the runner passes (M1: symmetric on
  the student obs; asymmetric/privileged critic is a later enhancement).
- ``decode_for_env``: renormalize the sampled latent and run the frozen decoder
  to produce the joint action the env consumes.

Adapters (decision D6 build order): only ``"full_ft"`` is implemented in M1
(encoder trainable, decoder frozen). ``"lora"`` / ``"residual"`` are wired as
explicit ``NotImplementedError`` until M3 / M4.
"""

from __future__ import annotations

from typing import Any, Sequence

import copy
import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.latent_bottleneck_muse_kp import LatentBottleneckMUSEKp
from rsl_rl.modules.lora import apply_lora_to_encoder
from rsl_rl.utils import resolve_nn_activation


_VALID_ADAPTERS = ("full_ft", "lora", "residual")

# MUSE-Kp constructor kwargs we forward from the (flattened) policy cfg. Anything
# else in **kwargs is ignored with a notice (mirrors LatentBottleneckMUSEKp).
_MUSE_KWARG_KEYS = (
    "proprio_dim",
    "history_length",
    "proprio_term_sizes",
    "kp_n_bodies",
    "kp_lookahead_steps",
    "kp_layout",
    "latent_dim",
    "d_model",
    "nhead",
    "num_layers",
    "ffn_dim",
    "decoder_hidden_dims",
    "teacher_hidden_dims",
    "activation",
    "init_noise_std",
    "latent_predict_std_min",
    "latent_predict_std_max",
    "fixed_encoder_std",
    "encoder_dropout",
    "decoder_one_step_proprio",
    "latent_normalize",
    "deterministic_encoder",
    "aux_predictor_enabled",
)

# adapter -> freeze_mode forced on the inner MUSE-Kp module.
#   full_ft  : encoder trains, decoder frozen      -> "decoder_only"
#   lora     : base frozen, LoRA adapters train    -> "decoder_only" + LoRA (M3)
#   residual : encoder + decoder frozen            -> "decoder_only" + g_phi (M4)
_ADAPTER_FREEZE_MODE = {
    "full_ft": "decoder_only",
    "lora": "decoder_only",
    "residual": "decoder_only",
}


class LatentRLActorCritic(nn.Module):
    """ActorCritic-contract policy whose action is the MUSE-Kp latent.

    Constructor signature matches the runner's call
    ``policy_class(num_actor_obs, num_critic_obs, num_actions, **policy_cfg)``.
    """

    is_recurrent = False
    is_encoding = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        num_teacher_obs: int = 815,
        adapter: str = "full_ft",
        init_latent_std: float = 0.1,
        prior_anchor_coef: float = 0.0,
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        critic_activation: str = "elu",
        # --- LoRA adapter (M3) ---
        lora_rank: int = 8,
        lora_alpha: float | None = None,
        lora_targets: Sequence[str] = ("attn_qkv", "attn_out", "mu_head"),
        # --- obstacle obs (option C) ---
        # The policy obs has obstacle_feat_dim dims appended LAST (obstacle_n boxes ×
        # per-box). They are stripped before the frozen encoder and fed to the residual
        # corrector as per-box tokens. 0 (default) -> no obstacle (writing/general unchanged).
        obstacle_feat_dim: int = 0,
        obstacle_n: int = 0,
        # --- P2-C terrain residual (height scan LAST on policy obs) ---
        terrain_scan_dim: int = 0,
        terrain_r_max: float = 0.1763,
        terrain_scan_zero: bool = False,
        terrain_gate: bool = True,
        terrain_gate_w_s: float = 1.0,
        terrain_gate_w_r: float = 2.5,
        terrain_gate_s0: float = 0.06,
        terrain_gate_tau: float = 0.025,
        terrain_gate_s_dead: float = 0.02,
        terrain_scan_clip: float = 0.5,
        # --- P2-R intent recovery (no terrain scan; last Linear zero-init) ---
        intent_recovery: bool = False,
        intent_recovery_r_max: float = 0.0875,
        intent_recovery_r_off: float = 0.6,
        intent_recovery_r_full: float = 2.0,
        intent_recovery_persist_on: int = 3,
        intent_recovery_persist_off: int = 3,
        intent_recovery_q50_e: float = 0.046,
        intent_recovery_q90_e: float = 0.130,
        intent_recovery_q50_s: float = 0.041,
        intent_recovery_q90_s: float = 0.241,
        intent_recovery_aux_dim: int = 0,
        intent_recovery_s_enabled: bool = False,
        interaction_recovery: bool = False,
        interaction_encoder: str = "mlp",
        interaction_history_len: int = 16,
        interaction_tangent: bool = True,
        interaction_beta: float = 1.0,
        interaction_r_max: float = 0.0875,
        **kwargs: Any,
    ):
        super().__init__()
        # Prior anchor (decision D3): >0 enables a reward-side penalty toward the
        # distilled latent (set/consumed by LatentPPO). The frozen reference
        # encoder is snapshotted lazily on first prior_anchor_cos() call — by
        # then the runner has applied the distilled warmstart. Kept OUT of the
        # nn.Module graph (object.__setattr__) so it is not a checkpoint param.
        self.prior_anchor_coef = float(prior_anchor_coef)
        object.__setattr__(self, "_ref_encoder", None)

        if adapter not in _VALID_ADAPTERS:
            raise ValueError(f"adapter must be one of {_VALID_ADAPTERS}, got {adapter!r}")
        self.adapter = str(adapter)

        # LoRA config (consumed lazily after warmstart — see _ensure_lora_applied).
        self._lora_rank = int(lora_rank)
        self._lora_alpha = float(lora_alpha) if lora_alpha is not None else float(lora_rank)
        self._lora_targets = tuple(lora_targets)
        self._lora_applied = False

        # ---- Inner MUSE-Kp (encoder + decoder + frozen teacher, warmstart-load) ----
        muse_kwargs = {k: kwargs[k] for k in _MUSE_KWARG_KEYS if k in kwargs}
        # residual_* arrive via the policy cfg and ARE consumed (kwargs.get in the
        # residual block below) — exclude them from the "ignored" notice so the
        # log isn't misleading. (Kept as kwargs rather than explicit signature
        # params to avoid churning the signature shared with the LoRA adapter.)
        _consumed_via_kwargs = {
            "residual_d_model", "residual_num_layers", "residual_nhead",
            "residual_ffn", "residual_last_layer_gain", "residual_alpha",
            "latent_std_min", "latent_std_max",
            "terrain_scan_dim", "terrain_r_max", "terrain_scan_zero",
            "terrain_gate", "terrain_gate_w_s", "terrain_gate_w_r",
            "terrain_gate_s0", "terrain_gate_tau", "terrain_gate_s_dead",
            "terrain_scan_clip",
            "intent_recovery", "intent_recovery_r_max", "intent_recovery_r_off",
            "intent_recovery_r_full", "intent_recovery_persist_on",
            "intent_recovery_persist_off", "intent_recovery_q50_e",
            "intent_recovery_q90_e", "intent_recovery_q50_s",
            "intent_recovery_q90_s", "intent_recovery_aux_dim",
            "intent_recovery_s_enabled",
            "interaction_recovery", "interaction_encoder", "interaction_history_len",
            "interaction_tangent", "interaction_beta", "interaction_r_max",
            "freeze_mode",
        }
        extra = sorted(set(kwargs) - set(_MUSE_KWARG_KEYS) - _consumed_via_kwargs)
        if extra:
            print(f"[LatentRLActorCritic] ignoring unexpected kwargs: {extra}")
        # Force the latent-RL geometry contract regardless of cfg drift:
        #   deterministic_encoder=True  -> z := mu at the mean (we add our own
        #     Gaussian on the unit-norm latent; the inner sampler is unused).
        #   latent_normalize=True       -> decoder sees the unit-norm latent it
        #     was distilled on.
        #   freeze_mode per adapter     -> decoder (and teacher) frozen for full_ft.
        muse_kwargs["deterministic_encoder"] = True
        muse_kwargs["latent_normalize"] = True
        muse_kwargs["freeze_mode"] = _ADAPTER_FREEZE_MODE[self.adapter]
        # Trailing extras (option C obstacle, then P2-C height scan LAST). The frozen
        # encoder must see only its own [kp|mask|proprio] width.
        self.obstacle_feat_dim = int(obstacle_feat_dim)
        self.obstacle_n = int(obstacle_n)
        self.terrain_scan_dim = int(terrain_scan_dim)
        self.terrain_r_max = float(terrain_r_max)
        self.terrain_scan_zero = bool(terrain_scan_zero)
        self.terrain_gate_enabled = bool(terrain_gate)
        self.terrain_gate_w_s = float(terrain_gate_w_s)
        self.terrain_gate_w_r = float(terrain_gate_w_r)
        self.terrain_gate_s0 = float(terrain_gate_s0)
        self.terrain_gate_tau = float(terrain_gate_tau)
        self.terrain_gate_s_dead = float(terrain_gate_s_dead)
        self.terrain_scan_clip = float(terrain_scan_clip)
        self.terrain_gate: nn.Module | None = None
        self.intent_recovery = bool(intent_recovery)
        self.intent_recovery_r_max = float(intent_recovery_r_max)
        self.intent_recovery_aux_dim = int(intent_recovery_aux_dim)
        self.intent_recovery_net: nn.Module | None = None
        self.intent_recovery_gate: nn.Module | None = None
        self.interaction_recovery = bool(interaction_recovery)
        self.interaction_net: nn.Module | None = None
        self._recovery_force_alpha: float | None = None
        self._recovery_alpha_override: torch.Tensor | None = None
        self._prev_e: torch.Tensor | None = None
        self._last_recovery: dict | None = None
        # List, not nn.Module: assigning the runner normalizer as an attribute
        # would register it in state_dict and break eval load.
        self._recovery_obs_nrm: list = []
        self._enc_obs_dim = (
            int(num_actor_obs)
            - self.obstacle_feat_dim
            - self.terrain_scan_dim
            - self.intent_recovery_aux_dim
        )
        if self._enc_obs_dim <= 0:
            raise ValueError(
                f"encoder obs dim non-positive: num_actor_obs={num_actor_obs} "
                f"obstacle={self.obstacle_feat_dim} scan={self.terrain_scan_dim} "
                f"recovery_aux={self.intent_recovery_aux_dim}"
            )
        self._obstacle_per_box = (self.obstacle_feat_dim // self.obstacle_n) if self.obstacle_n > 0 else 0
        self.terrain_residual: nn.Module | None = None
        self.muse = LatentBottleneckMUSEKp(
            num_student_obs=self._enc_obs_dim,
            num_teacher_obs=int(num_teacher_obs),
            num_actions=int(num_actions),
            **muse_kwargs,
        )
        self.latent_dim = int(self.muse.transformer_encoder.latent_dim)
        self.num_actions = int(num_actions)

        # The inner MUSE-Kp carries its own distillation-era action-std Parameter
        # (``muse.std``) that the latent-RL policy never uses (we have our own
        # latent Gaussian). Freeze it so it doesn't masquerade as trainable / sit
        # in the optimizer. Adapter-agnostic.
        if isinstance(getattr(self.muse, "std", None), nn.Parameter):
            self.muse.std.requires_grad_(False)

        # ---- Residual adapter (M4 / decision D2) ----
        # g_phi sees the full (normalized) encoder obs + the encoder's unit-norm
        # latent and outputs Δz; the corrected mean is normalize(μ̂ + α·Δz).
        # Encoder is frozen here too (decoder already frozen via freeze_mode), so
        # ONLY g_phi (+ latent_log_std + critic) train. Small-gain last layer ⇒
        # Δz≈0 at init ⇒ step-0 == distilled policy (safe start; the D3 prior
        # anchor is unnecessary for residual — it starts at the prior for free).
        # Residual hyperparams arrive via the policy cfg (kwargs); defaults are
        # used when absent. Local import keeps the shared import block untouched.
        self.residual_corrector: nn.Module | None = None
        if self.adapter == "residual":
            from rsl_rl.modules.latent_residual import build_residual_corrector

            for p in self.muse.transformer_encoder.parameters():
                p.requires_grad_(False)
            self.muse.transformer_encoder.eval()
            enc = self.muse.transformer_encoder
            # Shallow per-body-token transformer g_φ (handles KP masking
            # structurally via key_padding_mask; ~12× fewer params than the old
            # MLP at the default size). Default d_model=64/1 layer (~45k); a
            # larger combo d_model=96/num_layers=2/nhead=4/ffn=256 (~191k) is
            # available via the residual_* policy-cfg knobs.
            self.residual_corrector = build_residual_corrector(
                kp_n_bodies=int(enc.kp_n_bodies),
                kp_feat_dim=int(enc.kp_lookahead_steps) * 3,
                proprio_feat_dim=int(enc.per_frame_proprio_dim),
                proprio_history_length=int(enc.proprio_history_length),
                latent_dim=self.latent_dim,
                # Obstacle tokens: one per box; per-box feature = per_box − 1 (drop the
                # trailing validity flag, which becomes the token's key_padding mask).
                obstacle_n=self.obstacle_n,
                obstacle_feat_dim=(self._obstacle_per_box - 1) if self.obstacle_n > 0 else 0,
                d_model=int(kwargs.get("residual_d_model", 64)),
                num_layers=int(kwargs.get("residual_num_layers", 1)),
                nhead=int(kwargs.get("residual_nhead", 4)),
                ffn=int(kwargs.get("residual_ffn", 128)),
                last_layer_gain=float(kwargs.get("residual_last_layer_gain", 0.01)),
                alpha=float(kwargs.get("residual_alpha", 1.0)),
            )

        # P2-D: scan-only h_η + analytic α(H) with α(0)=0. Last Linear zero-init
        # so π(0) = g_{φ,50000}. g_φ and latent_log_std frozen; only h_η + critic train.
        if self.terrain_scan_dim > 0:
            from rsl_rl.modules.terrain_residual import TerrainResidualMLP, TerrainSeverityGate

            self.terrain_residual = TerrainResidualMLP(
                scan_dim=self.terrain_scan_dim, latent_dim=self.latent_dim
            )
            if self.terrain_gate_enabled:
                self.terrain_gate = TerrainSeverityGate(
                    scan_dim=self.terrain_scan_dim,
                    w_s=self.terrain_gate_w_s,
                    w_r=self.terrain_gate_w_r,
                    s0=self.terrain_gate_s0,
                    tau=self.terrain_gate_tau,
                    s_dead=self.terrain_gate_s_dead,
                    clip_abs=self.terrain_scan_clip,
                )
            if self.residual_corrector is not None:
                for p in self.residual_corrector.parameters():
                    p.requires_grad_(False)
                self.residual_corrector.eval()
            last = self.terrain_residual.head[-1]
            print(
                f"[LatentRLActorCritic] P2-D gated residual scan_dim={self.terrain_scan_dim} "
                f"r_max={self.terrain_r_max:.4f} gate={self.terrain_gate_enabled} "
                f"last||W||={float(last.weight.abs().max()):.1e} (g_phi frozen, h_eta=h(H) only)"
            )

        # ---- State-independent Gaussian over the latent (decision D1) ----
        init_latent_std = max(float(init_latent_std), 1.0e-6)
        self.latent_log_std = nn.Parameter(
            torch.log(init_latent_std * torch.ones(self.latent_dim))
        )
        if self.terrain_scan_dim > 0:
            self.latent_log_std.requires_grad_(False)

        # P2-R: freeze the entire nominal stack (encoder, g_φ, decoder, log_std).
        # Only r_η + critic train. Last Linear of r_η is zero so r≡0 at init.
        if self.intent_recovery:
            from rsl_rl.modules.intent_recovery import RecoveryResidualMLP, RecoveryRiskGate

            if self.residual_corrector is not None:
                for p in self.residual_corrector.parameters():
                    p.requires_grad_(False)
                self.residual_corrector.eval()
            self.latent_log_std.requires_grad_(False)
            enc = self.muse.transformer_encoder
            prop_dim = int(enc.per_frame_proprio_dim) * int(enc.proprio_history_length)
            self.intent_recovery_net = RecoveryResidualMLP(
                proprio_dim=prop_dim, latent_dim=self.latent_dim
            )
            self.intent_recovery_gate = RecoveryRiskGate(
                q50_e=float(intent_recovery_q50_e),
                q90_e=float(intent_recovery_q90_e),
                q50_s=float(intent_recovery_q50_s),
                q90_s=float(intent_recovery_q90_s),
                r_off=float(intent_recovery_r_off),
                r_full=float(intent_recovery_r_full),
                persist_on=int(intent_recovery_persist_on),
                persist_off=int(intent_recovery_persist_off),
                s_enabled=bool(intent_recovery_s_enabled),
            )
            last = self.intent_recovery_net.net[-1]
            print(
                f"[LatentRLActorCritic] P2-R intent recovery in={self.intent_recovery_net.in_dim} "
                f"r_max={self.intent_recovery_r_max:.4f} ({math.degrees(math.atan(self.intent_recovery_r_max)):.1f}°) "
                f"gate={self.intent_recovery_gate.extra_repr()} "
                f"last||W||={float(last.weight.abs().max()):.1e} "
                f"(g_phi frozen, r_eta=0 at init, no terrain scan)"
            )

        # ICR: small shared residual. Frozen Stage-2 / g_φ. No terrain, no gate.
        if self.interaction_recovery:
            from rsl_rl.modules.interaction_residual import InteractionResidual

            if self.intent_recovery:
                raise ValueError("interaction_recovery and intent_recovery cannot both be True")
            if self.residual_corrector is not None:
                for p in self.residual_corrector.parameters():
                    p.requires_grad_(False)
                self.residual_corrector.eval()
            self.latent_log_std.requires_grad_(False)
            self.interaction_net = InteractionResidual(
                encoder=str(interaction_encoder),
                history_len=int(interaction_history_len),
                tangent=bool(interaction_tangent),
                beta=float(interaction_beta),
                r_max=float(interaction_r_max),
            )
            print(
                f"[LatentRLActorCritic] ICR {self.interaction_net.extra_repr()} "
                f"(g_phi frozen, Δz=0 at init, no terrain obs, no gate)"
            )

        # 0 disables. Used to stop unbounded log_std blow-ups in long latent-RL runs.
        self.latent_std_min = float(kwargs.get("latent_std_min", 0.0) or 0.0)
        self.latent_std_max = float(kwargs.get("latent_std_max", 0.0) or 0.0)

        # ---- Fresh critic MLP (M1: on whatever critic obs the runner passes) ----
        act_cls = resolve_nn_activation(critic_activation)
        critic_layers: list[nn.Module] = []
        prev = int(num_critic_obs)
        for h in critic_hidden_dims:
            critic_layers.append(nn.Linear(prev, int(h)))
            critic_layers.append(act_cls)
            prev = int(h)
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers)
        self._last_terrain = None

        self.distribution: Normal | None = None
        # LatentBottleneckMUSEKp.__init__ (constructed above) reassigns
        # ``Normal.set_default_validate_args = False`` (clobbers the classmethod
        # with a bool), so a plain call here would raise. Guard it.
        if callable(getattr(Normal, "set_default_validate_args", None)):
            Normal.set_default_validate_args(False)

        print(
            f"[LatentRLActorCritic] adapter={self.adapter!r} latent_dim={self.latent_dim} "
            f"num_actor_obs={num_actor_obs} num_critic_obs={num_critic_obs} "
            f"num_actions={num_actions} init_latent_std={init_latent_std} "
            f"prior_anchor_coef={self.prior_anchor_coef} "
            f"inner_freeze_mode={self.muse.freeze_mode!r}"
            + (
                f" terrain_scan_dim={self.terrain_scan_dim} r_max={self.terrain_r_max:.3f}"
                if self.terrain_scan_dim > 0
                else ""
            )
            + (
                f" lora(r={self._lora_rank}, alpha={self._lora_alpha}, "
                f"targets={list(self._lora_targets)})"
                if self.adapter == "lora"
                else ""
            )
        )

    # ----- LoRA (M3) ----------------------------------------------------------------------------

    def _ensure_lora_applied(self) -> None:
        """Inject LoRA parametrizations on the encoder, once.

        Deferred until AFTER the distilled warmstart is loaded: the inner
        MUSE-Kp loader matches checkpoint keys by name, and parametrization
        renames ``weight`` -> ``parametrizations.weight.original``. Called from
        :meth:`load_state_dict` (both the raw-warmstart and resume paths) and,
        as a safety net for the no-warmstart case, at the top of
        :meth:`_encode_mean_latent`. Idempotent.
        """
        if self.adapter != "lora" or self._lora_applied:
            return
        n = apply_lora_to_encoder(
            self.muse.transformer_encoder,
            targets=self._lora_targets,
            r=self._lora_rank,
            alpha=self._lora_alpha,
        )
        self._lora_applied = True
        print(
            f"[LatentRLActorCritic] LoRA applied: {n} parametrized weights "
            f"(r={self._lora_rank}, alpha={self._lora_alpha}, "
            f"targets={list(self._lora_targets)}); encoder base frozen."
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen nominal stack must stay eval even when PPO sets policy.train().
        if self.residual_corrector is not None and (
            self.terrain_scan_dim > 0 or self.intent_recovery or self.interaction_recovery
        ):
            self.residual_corrector.eval()
        if self.adapter == "residual":
            self.muse.transformer_encoder.eval()
        return self

    # ----- Latent helpers -----------------------------------------------------------------------

    def _split_policy_obs(self, obs: torch.Tensor):
        """Split into (enc_obs, scan, obstacle_feat, obstacle_mask).

        Layout: ``[encoder_core | optional obstacle | height_scan LAST]``.
        ``scan`` is zeros when the env does not append the 187-D block (scan-zero /
        old 750-D eval). Obstacle extras follow the existing option-C contract.
        """
        rest = obs
        scan = None
        rec_aux = None
        scan_dim = int(self.terrain_scan_dim)
        aux_dim = int(self.intent_recovery_aux_dim)
        if scan_dim > 0:
            core_plus_obs = self._enc_obs_dim + self.obstacle_feat_dim
            if int(obs.shape[-1]) == core_plus_obs:
                scan = obs.new_zeros(*obs.shape[:-1], scan_dim)
            elif int(obs.shape[-1]) >= scan_dim:
                scan = obs[..., -scan_dim:]
                rest = obs[..., :-scan_dim]
            else:
                scan = obs.new_zeros(*obs.shape[:-1], scan_dim)
            if self.terrain_scan_zero:
                scan = torch.zeros_like(scan)
        if aux_dim > 0:
            if int(rest.shape[-1]) >= aux_dim:
                rec_aux = rest[..., -aux_dim:]
                rest = rest[..., :-aux_dim]
            else:
                rec_aux = rest.new_zeros(*rest.shape[:-1], aux_dim)
        if self.obstacle_feat_dim <= 0:
            return rest, scan, None, None, rec_aux
        enc = rest[..., : self._enc_obs_dim]
        raw = rest[..., self._enc_obs_dim :].reshape(
            *rest.shape[:-1], self.obstacle_n, self._obstacle_per_box
        )
        feat = raw[..., : self._obstacle_per_box - 1]
        mask = raw[..., self._obstacle_per_box - 1] < 0.5
        return enc, scan, feat, mask, rec_aux

    def _nominal_mean_latent(self, enc_obs, obstacle_feat, obstacle_mask):
        """Frozen Stage-2 + frozen g_φ → unit-norm ``z_nom``."""
        mu, _log_sigma, proprio_per_frame = self.muse.transformer_encoder.encode(enc_obs)
        mu_hat = self.muse._maybe_normalize_latent(mu)
        if getattr(self, "residual_corrector", None) is not None:
            kp, kp_mask, _prop = self.muse.transformer_encoder.split_obs(enc_obs)
            base = mu_hat.detach()
            delta = self.residual_corrector(
                kp, kp_mask, proprio_per_frame, base, obstacle_feat, obstacle_mask
            )
            mu_hat = self.muse._maybe_normalize_latent(
                base + self.residual_corrector.alpha * delta
            )
        return mu_hat, proprio_per_frame

    def _apply_terrain_residual(self, z_nom: torch.Tensor, scan: torch.Tensor | None):
        """``h_η(H)`` + tangent cap + analytic α(H). Grad through ``h_η`` only."""
        from rsl_rl.modules.terrain_residual import apply_tangent_correction

        if self.terrain_residual is None or int(self.terrain_scan_dim) <= 0:
            z = self.muse._maybe_normalize_latent(z_nom)
            zeros = torch.zeros_like(z)
            return z, zeros, zeros, zeros[..., 0]
        if scan is None:
            scan = z_nom.new_zeros(z_nom.shape[0], self.terrain_scan_dim)
        z_det = z_nom.detach()
        u = self.terrain_residual(scan)
        if self.terrain_gate is not None:
            alpha, _s, _ss, _sr = self.terrain_gate(scan)
        else:
            alpha = scan.new_ones(scan.shape[0])
        z_exec, u_bar = apply_tangent_correction(z_det, u, r_max=self.terrain_r_max, alpha=alpha)
        return z_exec, u_bar, u, alpha

    def _slot0_idx(self) -> int:
        offsets = tuple(int(x) for x in self.muse.kp_slot_offsets)
        return int(offsets.index(0))

    def _denorm_enc_obs(self, enc_obs: torch.Tensor) -> torch.Tensor:
        """Invert empirical normalization so gate ``E`` is metres, not z-scores.

        The frozen encoder still sees the normalized ``enc_obs``. Only ``r_η``'s
        ``e, ė`` and the risk gate read this inverse. A missing or dim-mismatched
        normalizer is a hard error in recovery mode: silent fallback would
        re-open the always-on gate (``E≈1.2`` vs Q90 ``0.13 m``).
        """
        if not (self.intent_recovery or self.interaction_recovery):
            return enc_obs
        nrm = self._recovery_obs_nrm[0] if self._recovery_obs_nrm else None
        if nrm is None or not hasattr(nrm, "inverse"):
            return enc_obs
        mean = getattr(nrm, "_mean", None)
        if mean is not None and int(mean.reshape(-1).shape[0]) != int(enc_obs.shape[-1]):
            raise RuntimeError(
                f"obs_normalizer dim {int(mean.reshape(-1).shape[0])} != enc_obs {int(enc_obs.shape[-1])}"
            )
        return nrm.inverse(enc_obs)

    def _stability_from_aux(self, rec_aux: torch.Tensor | None, batch: int, like: torch.Tensor) -> torch.Tensor:
        """``S = |v_root,z|`` from optional trailing aux; else 0 (R1a loco: R = R_E)."""
        if rec_aux is None or int(self.intent_recovery_aux_dim) <= 0:
            return like.new_zeros(batch)
        return rec_aux[..., 0].reshape(batch).abs()

    def _apply_intent_recovery(
        self,
        z_nom: torch.Tensor,
        enc_obs: torch.Tensor,
        proprio_per_frame: torch.Tensor,
        rec_aux: torch.Tensor | None,
        mutate_gate: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """Gated tangent residual. ``sg(z_nom)`` into ``r_η``; encoder/g_φ stay frozen."""
        from rsl_rl.modules.intent_recovery import (
            DT,
            apply_recovery_correction,
            extract_visible_task_error,
        )

        if self.intent_recovery_net is None or self.intent_recovery_gate is None:
            z = self.muse._maybe_normalize_latent(z_nom)
            empty = {
                "z_exec": z,
                "r_raw": torch.zeros_like(z),
                "r_perp": torch.zeros_like(z),
                "r_bar": torch.zeros_like(z),
                "r_raw_norm": z.new_zeros(z.shape[0]),
                "r_perp_norm": z.new_zeros(z.shape[0]),
                "r_bar_norm": z.new_zeros(z.shape[0]),
                "rho_r": z.new_zeros(z.shape[0]),
                "theta": z.new_zeros(z.shape[0]),
                "alpha": z.new_zeros(z.shape[0]),
                "E": z.new_zeros(z.shape[0]),
                "S": z.new_zeros(z.shape[0]),
                "R": z.new_zeros(z.shape[0]),
                "R_E": z.new_zeros(z.shape[0]),
                "R_S": z.new_zeros(z.shape[0]),
                "active": z.new_zeros(z.shape[0]),
                "e": z.new_zeros(z.shape[0], 9),
                "vis": z.new_zeros(z.shape[0], 3),
            }
            return z, empty

        enc_m = self._denorm_enc_obs(enc_obs)
        kp, kp_mask, _prop = self.muse.transformer_encoder.split_obs(enc_m)
        # kp is [B, N, L*3]; restore [B, L, N, 3] via the layout used in split_obs.
        enc = self.muse.transformer_encoder
        bsz = int(z_nom.shape[0])
        L, n_b = int(enc.kp_lookahead_steps), int(enc.kp_n_bodies)
        kp_lhn = kp.reshape(bsz, n_b, L, 3).transpose(1, 2)
        e, vis, e_rms = extract_visible_task_error(kp_lhn, kp_mask, self._slot0_idx())
        if self._prev_e is None or int(self._prev_e.shape[0]) != bsz:
            e_dot = torch.zeros_like(e)
        else:
            e_dot = (e - self._prev_e) / DT
        e_dot = e_dot * vis.repeat_interleave(3, dim=-1)
        if mutate_gate:
            self._prev_e = e.detach()

        s = self._stability_from_aux(rec_aux, bsz, e_rms)
        if self._recovery_force_alpha is not None:
            alpha = e_rms.new_full((bsz,), float(self._recovery_force_alpha))
            gate_out = self.intent_recovery_gate.scores(e_rms, s)
            r, r_e, r_s = gate_out
            active = alpha > 0
        elif self._recovery_alpha_override is not None:
            alpha = self._recovery_alpha_override.reshape(-1).to(dtype=e_rms.dtype, device=e_rms.device)
            r, r_e, r_s = self.intent_recovery_gate.scores(e_rms, s)
            active = alpha > 0
        else:
            gate_n = (
                0
                if self.intent_recovery_gate._active is None
                else int(self.intent_recovery_gate._active.shape[0])
            )
            use_state = mutate_gate and gate_n in (0, bsz)
            gout = self.intent_recovery_gate.step(e_rms, s, mutate=use_state)
            alpha, r, r_e, r_s = gout["alpha"], gout["R"], gout["R_E"], gout["R_S"]
            active = gout["active"]

        r_raw = self.intent_recovery_net(
            e, e_dot, vis, proprio_per_frame.reshape(bsz, -1), z_nom.detach()
        )
        out = apply_recovery_correction(z_nom, r_raw, alpha, r_max=self.intent_recovery_r_max)
        out.update(
            {
                "E": e_rms,
                "S": s,
                "R": r,
                "R_E": r_e,
                "R_S": r_s,
                "active": active.to(dtype=e_rms.dtype),
                "e": e,
                "vis": vis,
                "e_dot": e_dot,
            }
        )
        return out["z_exec"], out

    def _apply_interaction_residual(
        self,
        z_nom: torch.Tensor,
        enc_obs: torch.Tensor,
        proprio_per_frame: torch.Tensor,
        mutate_hist: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """Shared latent residual. No terrain features. Last Linear is zero at init."""
        from rsl_rl.modules.intent_recovery import DT, extract_visible_task_error
        from rsl_rl.modules.interaction_residual import pack_token

        z = self.muse._maybe_normalize_latent(z_nom)
        if self.interaction_net is None:
            empty = {
                "z_exec": z,
                "dz": torch.zeros_like(z),
                "dz_bar": torch.zeros_like(z),
                "r_raw": torch.zeros_like(z),
                "r_bar": torch.zeros_like(z),
                "r_raw_norm": z.new_zeros(z.shape[0]),
                "r_bar_norm": z.new_zeros(z.shape[0]),
                "alpha": z.new_ones(z.shape[0]),
                "E": z.new_zeros(z.shape[0]),
                "R_E": z.new_zeros(z.shape[0]),
                "R": z.new_zeros(z.shape[0]),
                "active": z.new_ones(z.shape[0]),
                "e": z.new_zeros(z.shape[0], 9),
            }
            return z, empty
        enc_m = self._denorm_enc_obs(enc_obs)
        kp, kp_mask, _prop = self.muse.transformer_encoder.split_obs(enc_m)
        enc = self.muse.transformer_encoder
        bsz = int(z_nom.shape[0])
        L, n_b = int(enc.kp_lookahead_steps), int(enc.kp_n_bodies)
        kp_lhn = kp.reshape(bsz, n_b, L, 3).transpose(1, 2)
        e, vis, e_rms = extract_visible_task_error(kp_lhn, kp_mask, self._slot0_idx())
        if self._prev_e is None or int(self._prev_e.shape[0]) != bsz:
            e_dot = torch.zeros_like(e)
        else:
            e_dot = (e - self._prev_e) / DT
        e_dot = e_dot * vis.repeat_interleave(3, dim=-1)
        if mutate_hist:
            self._prev_e = e.detach()
        tok = pack_token(z.detach(), e, e_dot, proprio_per_frame)
        if not mutate_hist and self.interaction_net.encoder_name == "gru":
            dz = self.interaction_net.body(self.interaction_net._hist)
            z_n = torch.nn.functional.normalize(z, dim=-1, eps=1e-8)
            if self.interaction_net.tangent:
                from rsl_rl.modules.terrain_residual import apply_tangent_correction

                z_exec, dz_bar = apply_tangent_correction(
                    z_n, dz, r_max=self.interaction_net.r_max,
                    alpha=z_n.new_full((z_n.shape[0],), self.interaction_net.beta),
                )
            else:
                z_exec = torch.nn.functional.normalize(
                    z_n + self.interaction_net.beta * dz, dim=-1, eps=1e-8
                )
                dz_bar = dz
            out = {
                "z_exec": z_exec,
                "dz": dz,
                "dz_bar": dz_bar,
                "dz_norm": dz.norm(dim=-1),
                "dz_bar_norm": dz_bar.norm(dim=-1),
            }
        else:
            out = self.interaction_net.apply(z, tok)
        rec = {
            "z_exec": out["z_exec"],
            "r_raw": out["dz"],
            "r_bar": out["dz_bar"],
            "r_raw_norm": out["dz_norm"],
            "r_bar_norm": out["dz_bar_norm"],
            "alpha": z.new_full((bsz,), float(self.interaction_net.beta)),
            "E": e_rms,
            "R_E": e_rms,
            "R": e_rms,
            "R_S": z.new_zeros(bsz),
            "S": z.new_zeros(bsz),
            "active": z.new_ones(bsz),
            "e": e,
            "vis": vis,
            "theta": z.new_zeros(bsz),
            "rho_r": z.new_zeros(bsz),
            "r_perp_norm": out["dz_bar_norm"],
        }
        return out["z_exec"], rec

    def _encode_mean_latent(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (unit-norm mean latent ``mu_hat`` [B, latent_dim], proprio_per_frame)."""
        if self.adapter == "lora":
            self._ensure_lora_applied()
        enc_obs, scan, obstacle_feat, obstacle_mask, rec_aux = self._split_policy_obs(obs)
        z_nom, proprio_per_frame = self._nominal_mean_latent(
            enc_obs, obstacle_feat, obstacle_mask
        )
        z_mid, u_bar, u, t_alpha = self._apply_terrain_residual(z_nom, scan)
        gate_n = (
            0
            if self.intent_recovery_gate is None or self.intent_recovery_gate._active is None
            else int(self.intent_recovery_gate._active.shape[0])
        )
        bsz = int(z_nom.shape[0])
        mutate = (
            self._recovery_force_alpha is None
            and self._recovery_alpha_override is None
            and gate_n in (0, bsz)
        )
        z_exec, rec = self._apply_intent_recovery(
            z_mid, enc_obs, proprio_per_frame, rec_aux, mutate_gate=mutate
        )
        if self.interaction_net is not None:
            z_exec, rec = self._apply_interaction_residual(
                z_exec if self.intent_recovery else z_mid,
                enc_obs,
                proprio_per_frame,
                mutate_hist=mutate,
            )
        rec_alpha = rec.get("alpha")
        rec_bar = rec.get("r_bar", u)
        self._last_terrain = {
            "u_perp": (rec_bar if torch.is_tensor(rec_bar) else u_bar).detach(),
            "z_nom": z_nom.detach(),
            "z_exec": z_exec.detach(),
            "u": rec.get("r_raw", u).detach() if torch.is_tensor(rec.get("r_raw", u)) else u.detach(),
            "alpha": rec_alpha.detach() if torch.is_tensor(rec_alpha) else t_alpha,
            "scan": None if scan is None else scan.detach(),
        }
        self._last_recovery = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in rec.items()}
        self._last_recovery["z_nom"] = z_nom.detach()
        return z_exec, proprio_per_frame

    def terrain_aux_terms(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``L_zero = ‖h_η(0)‖²``, ``L_calm = (1-α)‖ū‖²``. Grad through ``h_η``."""
        if self.terrain_residual is None or int(self.terrain_scan_dim) <= 0:
            return None
        _enc, scan, _feat, _mask, _aux = self._split_policy_obs(obs)
        if scan is None:
            return None
        u0 = self.terrain_residual(torch.zeros_like(scan))
        l_zero = u0.pow(2).sum(dim=-1).mean()
        z_nom, _ = self._nominal_mean_latent(_enc, _feat, _mask)
        _z_exec, u_bar, _u, alpha = self._apply_terrain_residual(z_nom, scan)
        a = alpha.reshape(-1).to(dtype=u_bar.dtype)
        l_calm = ((1.0 - a) * u_bar.pow(2).sum(dim=-1)).mean()
        return l_zero, l_calm

    def residual_deltas(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Student vs frozen parent ``g_φ`` outputs ``(Δz, Δz0)``. Encoder is frozen."""
        parent = getattr(self, "parent_residual_corrector", None)
        if self.residual_corrector is None or parent is None:
            raise RuntimeError("residual_deltas requires residual adapter + parent_residual_corrector")
        enc_obs, _scan, obstacle_feat, obstacle_mask, _aux = self._split_policy_obs(obs)
        mu, _log_sigma, proprio_per_frame = self.muse.transformer_encoder.encode(enc_obs)
        base = self.muse._maybe_normalize_latent(mu).detach()
        kp, kp_mask, _prop = self.muse.transformer_encoder.split_obs(enc_obs)
        delta = self.residual_corrector(kp, kp_mask, proprio_per_frame, base, obstacle_feat, obstacle_mask)
        with torch.no_grad():
            delta0 = parent(kp, kp_mask, proprio_per_frame, base, obstacle_feat, obstacle_mask)
        return delta, delta0

    def exec_and_parent_latents(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unit-sphere ``(z_exec, z_parent)``. Grad flows only through student ``g_φ``."""
        parent = getattr(self, "parent_residual_corrector", None)
        if self.residual_corrector is None or parent is None:
            raise RuntimeError("exec_and_parent_latents needs residual adapter + frozen parent")
        enc_obs, scan, obstacle_feat, obstacle_mask, rec_aux = self._split_policy_obs(obs)
        mu, _log_sigma, proprio_per_frame = self.muse.transformer_encoder.encode(enc_obs)
        z_base = self.muse._maybe_normalize_latent(mu).detach()
        kp, kp_mask, _prop = self.muse.transformer_encoder.split_obs(enc_obs)
        alpha = float(self.residual_corrector.alpha)
        delta = self.residual_corrector(kp, kp_mask, proprio_per_frame, z_base, obstacle_feat, obstacle_mask)
        z_nom = self.muse._maybe_normalize_latent(z_base + alpha * delta)
        z_mid, _u_bar, _u, _alpha = self._apply_terrain_residual(z_nom, scan)
        z_exec, _rec = self._apply_intent_recovery(
            z_mid, enc_obs, proprio_per_frame, rec_aux, mutate_gate=False
        )
        with torch.no_grad():
            delta0 = parent(kp, kp_mask, proprio_per_frame, z_base, obstacle_feat, obstacle_mask)
            z_parent = self.muse._maybe_normalize_latent(z_base + alpha * delta0)
        return z_exec, z_parent

    @torch.no_grad()
    def last_terrain_stats(self) -> dict[str, torch.Tensor] | None:
        """Cheap per-env stats from the most recent ``act`` encode (no encoder replay)."""
        cache = getattr(self, "_last_terrain", None)
        if cache is None:
            return None
        z_nom = cache["z_nom"]
        z_exec = cache["z_exec"]
        u_perp = cache["u_perp"]
        u = cache["u"]
        alpha = cache.get("alpha")
        if alpha is None:
            alpha = z_nom.new_ones(z_nom.shape[0])
        elif not torch.is_tensor(alpha):
            alpha = z_nom.new_full((z_nom.shape[0],), float(alpha))
        alpha = alpha.reshape(-1).to(dtype=z_nom.dtype)
        cos = (z_nom * z_exec).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        ang = torch.acos(cos) * (180.0 / math.pi)
        r_max = float(self.intent_recovery_r_max if self.intent_recovery else self.terrain_r_max)
        theta_avail = torch.atan(alpha * r_max) * (180.0 / math.pi)
        # α≈0 (dead zone / true-flat) makes θ_avail~0 and acos noise (~0.08°)
        # would explode ρ. Only score saturation when the gate actually granted budget.
        rho = torch.where(
            theta_avail > 0.3,
            ang / theta_avail.clamp_min(1e-3),
            torch.zeros_like(ang),
        )
        dz = (alpha * u_perp.norm(dim=-1))
        sens = torch.zeros(z_nom.shape[0], device=z_nom.device, dtype=z_nom.dtype)
        scan = cache.get("scan")
        if self.terrain_residual is not None and scan is not None:
            u0 = self.terrain_residual(torch.zeros_like(scan))
            sens = (u - u0).norm(dim=-1)
        return {
            "u_perp": u_perp.norm(dim=-1),
            "ang_deg": ang,
            "scan_sens": sens,
            "alpha": alpha,
            "rho_auth": rho,
            "dz": dz,
            "sat": (rho > 0.95).to(dtype=z_nom.dtype),
        }

    @torch.no_grad()
    def last_recovery_stats(self) -> dict[str, torch.Tensor] | None:
        """Per-env recovery logs from the most recent encode (no future labels)."""
        cache = getattr(self, "_last_recovery", None)
        if cache is None or not (self.intent_recovery or self.interaction_recovery):
            return None
        z_nom = cache.get("z_nom")
        z_exec = cache["z_exec"]
        if z_nom is None:
            z_nom = z_exec
        keys = (
            "E", "S", "R", "R_E", "R_S", "alpha", "active",
            "r_raw_norm", "r_perp_norm", "r_bar_norm", "rho_r", "theta",
        )
        out = {}
        for k in keys:
            v = cache.get(k)
            if v is None:
                continue
            out[k] = v.reshape(-1).to(dtype=z_exec.dtype) if torch.is_tensor(v) else z_exec.new_full((z_exec.shape[0],), float(v))
        out["theta_deg"] = out["theta"] * (180.0 / math.pi) if "theta" in out else z_exec.new_zeros(z_exec.shape[0])
        return out

    @torch.no_grad()
    def recovery_R_E_from_obs(self, obs: torch.Tensor, *, obs_is_normalized: bool) -> torch.Tensor:
        """Stateless ``R_E`` from actor obs. Does **not** step the hysteresis gate."""
        from rsl_rl.modules.intent_recovery import extract_visible_task_error

        if self.intent_recovery_gate is None and self.interaction_net is None:
            return obs.new_zeros(obs.shape[0])
        enc, _scan, _feat, _mask, rec_aux = self._split_policy_obs(obs)
        enc_m = self._denorm_enc_obs(enc) if obs_is_normalized else enc
        kp, kp_mask, _prop = self.muse.transformer_encoder.split_obs(enc_m)
        bsz = int(enc_m.shape[0])
        enc = self.muse.transformer_encoder
        L, n_b = int(enc.kp_lookahead_steps), int(enc.kp_n_bodies)
        kp_lhn = kp.reshape(bsz, n_b, L, 3).transpose(1, 2)
        _e, _vis, e_rms = extract_visible_task_error(kp_lhn, kp_mask, self._slot0_idx())
        if self.intent_recovery_gate is None:
            return e_rms.reshape(-1)
        s = self._stability_from_aux(rec_aux, bsz, e_rms)
        _r, r_e, _r_s = self.intent_recovery_gate.scores(e_rms, s)
        return r_e.reshape(-1)

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean, _proprio = self._encode_mean_latent(observations)
        if self.latent_std_min > 0.0 or self.latent_std_max > 0.0:
            lo = math.log(self.latent_std_min) if self.latent_std_min > 0.0 else None
            hi = math.log(self.latent_std_max) if self.latent_std_max > 0.0 else None
            if lo is not None and hi is not None:
                self.latent_log_std.data.clamp_(lo, hi)
            elif lo is not None:
                self.latent_log_std.data.clamp_(min=lo)
            else:
                self.latent_log_std.data.clamp_(max=hi)
        std = torch.exp(self.latent_log_std).expand_as(mean).clamp_min(1.0e-6)
        self.distribution = Normal(mean, std)

    # ----- ActorCritic contract -----------------------------------------------------------------

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample a latent. This is the PPO action (stored + learned on)."""
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Deterministic deploy path: mean latent -> frozen decoder -> joint action.

        Returns the joint action directly (so play.py / eval step the env with it,
        unchanged). At the mean this reproduces the distilled MUSE-Kp policy.
        """
        mean, proprio = self._encode_mean_latent(observations)
        z = self.muse._maybe_normalize_latent(mean)
        return self.muse._decode(z, proprio)

    @torch.no_grad()
    def inspect_inference(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        """Deploy diagnostics: ``z_base``, residual ``Δz``, ``z_exec``, 29-D joints."""
        enc_obs, scan, obstacle_feat, obstacle_mask, rec_aux = self._split_policy_obs(observations)
        z_nom, proprio_per_frame = self._nominal_mean_latent(
            enc_obs, obstacle_feat, obstacle_mask
        )
        z_base = z_nom
        delta = torch.zeros_like(z_nom)
        z_mid, u_perp_t, u_t, gate_t = self._apply_terrain_residual(z_nom, scan)
        z_exec, rec = self._apply_intent_recovery(
            z_mid, enc_obs, proprio_per_frame, rec_aux, mutate_gate=False
        )
        if self.interaction_net is not None:
            z_exec, rec = self._apply_interaction_residual(
                z_mid, enc_obs, proprio_per_frame, mutate_hist=False
            )
        u_perp = rec.get("r_bar", u_perp_t)
        u = rec.get("r_raw", u_t)
        gate_a = rec.get("alpha", gate_t)
        action = self.muse._decode(z_exec, proprio_per_frame)
        cos = (z_nom * z_exec).sum(dim=-1) / (
            z_nom.norm(dim=-1).clamp_min(1e-8) * z_exec.norm(dim=-1).clamp_min(1e-8)
        )
        angle = torch.acos(cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        out = {
            "action": action,
            "z_base": z_base,
            "z_nom": z_nom,
            "delta_z": rec.get("r_raw", delta),
            "delta_z_eff": u_perp,
            "u": u,
            "u_perp": u_perp,
            "z_exec": z_exec,
            "cos_base_exec": cos,
            "angle_base_exec": angle,
            "residual_alpha": gate_a if torch.is_tensor(gate_a) else torch.full(cos.shape, float(gate_a), device=cos.device, dtype=cos.dtype),
            "terrain_alpha": gate_t if torch.is_tensor(gate_t) else torch.full(cos.shape, float(gate_t), device=cos.device, dtype=cos.dtype),
        }
        for k in ("E", "S", "R", "R_E", "R_S", "rho_r", "vis", "e"):
            if k in rec:
                out[k] = rec[k]
        return out

    def decode_for_env(self, latent: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """Frozen decoder: renormalize the sampled latent and decode to a joint action.

        Called by the runner between ``alg.act`` (which returned/stored the latent)
        and ``env.step``. Deterministic, no grad (decoder is frozen; PPO already
        detached the stored action).
        """
        with torch.no_grad():
            enc_obs, _scan, _f, _m, _aux = self._split_policy_obs(observations)
            _kp, _kp_mask, proprio_per_frame = self.muse.transformer_encoder.split_obs(enc_obs)
            z = self.muse._maybe_normalize_latent(latent)
            return self.muse._decode(z, proprio_per_frame)

    @property
    def prior_anchor_enabled(self) -> bool:
        return self.prior_anchor_coef > 0.0

    def prior_anchor_cos(self, observations: torch.Tensor) -> torch.Tensor:
        """Per-env cosine similarity between the CURRENT mean latent and the
        frozen distilled-reference mean latent (decision D3 prior anchor).

        Lazily snapshots the reference encoder on first call — by then the runner
        has applied the distilled warmstart, and (because the actor is frozen
        during critic warmup and PPO.update runs only after a full rollout) the
        encoder is still exactly the distilled one at that first rollout step.

        All no-grad: this feeds a reward-side penalty in
        :meth:`LatentPPO.process_env_step`, NOT a policy loss term — equivalent
        in effect to a "stay near the distilled latent" regularizer but without
        forking PPO's monolithic ``update``.

        Caveat: on ``--resume`` the snapshot captures the resumed (already
        finetuned) encoder, not the original distilled one. The anchored
        comparison runs (decision D3) start from the distilled ckpt, not a
        resume, so the reference is the intended distilled latent there.
        """
        with torch.no_grad():
            if self._ref_encoder is None:
                ref = copy.deepcopy(self.muse.transformer_encoder)
                for p in ref.parameters():
                    p.requires_grad_(False)
                ref.eval()
                object.__setattr__(self, "_ref_encoder", ref)
            enc_obs, _scan, _f, _m, _aux = self._split_policy_obs(observations)
            mu_cur, _, _ = self.muse.transformer_encoder.encode(enc_obs)
            mu_ref, _, _ = self._ref_encoder.encode(enc_obs)
            mu_cur = self.muse._maybe_normalize_latent(mu_cur)
            mu_ref = self.muse._maybe_normalize_latent(mu_ref)
            return nn.functional.cosine_similarity(mu_cur, mu_ref, dim=-1)  # [N]

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    def reset(self, dones=None) -> None:
        if self.intent_recovery_gate is not None:
            self.intent_recovery_gate.reset(dones)
        if self.interaction_net is not None:
            self.interaction_net.reset(dones)
        if dones is None:
            self._prev_e = None
            return
        if self._prev_e is None:
            return
        d = dones.reshape(-1).to(device=self._prev_e.device, dtype=torch.bool)
        if int(d.shape[0]) == int(self._prev_e.shape[0]):
            self._prev_e[d] = 0

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None) -> None:
        return

    # ----- Checkpoint loading -------------------------------------------------------------------

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Two accepted shapes:

        1. **Raw MUSE / MUSE-Kp checkpoint** (keys like ``transformer_encoder.*`` /
           ``decoder.*`` / ``teacher.*``, no ``muse.`` prefix): the runner's
           ``encoder_decoder_warmstart`` path. Forward to the inner module's
           loader (handles MUSE-T / MUSE-Kp-resume / PHC + slot-aware kp_proj
           remap). Returns ``False`` (not a latent-RL resume).
        2. **Latent-RL resume** (keys prefixed ``muse.`` / ``critic.`` /
           ``latent_log_std``): standard ``nn.Module`` load. Returns ``True``.
        """
        keys = list(state_dict.keys())
        has_own_prefix = any(
            k.startswith("muse.") or k.startswith("critic.") or k == "latent_log_std"
            for k in keys
        )
        looks_like_raw_muse = (not has_own_prefix) and any(
            k.startswith("transformer_encoder.")
            or k.startswith("decoder.")
            or k.startswith("teacher.")
            or ("actor." in k)
            for k in keys
        )
        if looks_like_raw_muse:
            # Distilled warmstart: load into the PLAIN encoder (names match),
            # THEN inject LoRA on top of the distilled base (parametrization
            # would otherwise rename the weight keys and break this load).
            self.muse.load_state_dict(state_dict, strict=False)
            self._ensure_lora_applied()
            return False

        # Latent-RL resume: the saved ckpt already carries parametrization keys
        # (muse...parametrizations.*), so LoRA must be injected BEFORE loading
        # so the keys exist on this module.
        self._ensure_lora_applied()
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("intent_recovery_obs_normalizer.")
        }
        own_keys = set(self.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        missing = own_keys - ckpt_keys
        def _ok_new(k: str) -> bool:
            return k.startswith("terrain_residual.") or k.startswith("intent_recovery") or k.startswith("interaction_net.")
        if missing and all(_ok_new(k) for k in missing):
            super().load_state_dict(state_dict, strict=False)
            prefixes = sorted({k.split(".")[0] for k in missing})
            print(
                f"[LatentRLActorCritic] loaded nominal ckpt; "
                f"zero-init {len(missing)} new tensors ({prefixes})"
            )
            return True
        super().load_state_dict(state_dict, strict=strict)
        return True
