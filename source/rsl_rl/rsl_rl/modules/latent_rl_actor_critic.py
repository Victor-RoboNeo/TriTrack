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
        # Obstacle obs (option C): the policy obs has the obstacle block appended LAST. The
        # frozen encoder must see only its own [kp|mask|proprio] width, so build it on the
        # stripped dim and slice the obstacle off at encode time (-> residual corrector).
        self.obstacle_feat_dim = int(obstacle_feat_dim)
        self.obstacle_n = int(obstacle_n)
        self._enc_obs_dim = int(num_actor_obs) - self.obstacle_feat_dim
        self._obstacle_per_box = (self.obstacle_feat_dim // self.obstacle_n) if self.obstacle_n > 0 else 0
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

        # ---- State-independent Gaussian over the latent (decision D1) ----
        init_latent_std = max(float(init_latent_std), 1.0e-6)
        self.latent_log_std = nn.Parameter(
            torch.log(init_latent_std * torch.ones(self.latent_dim))
        )

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

    # ----- Latent helpers -----------------------------------------------------------------------

    def _split_policy_obs(self, obs: torch.Tensor):
        """Split the policy obs into (enc_obs, obstacle_feat [B,K,F], obstacle_mask [B,K]).

        Option C: the obstacle block is the LAST ``obstacle_feat_dim`` dims, so the frozen
        encoder only ever sees ``enc_obs``. obstacle_* are None when obstacle obs is disabled.
        Per box = [center_b(3), half(3), valid(1)]; the trailing valid flag becomes the
        token key_padding mask (True = invalid/empty box -> excluded by g_φ)."""
        if self.obstacle_feat_dim <= 0:
            return obs, None, None
        enc = obs[..., : self._enc_obs_dim]
        raw = obs[..., self._enc_obs_dim:].reshape(*obs.shape[:-1], self.obstacle_n, self._obstacle_per_box)
        feat = raw[..., : self._obstacle_per_box - 1]
        mask = raw[..., self._obstacle_per_box - 1] < 0.5
        return enc, feat, mask

    def _encode_mean_latent(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (unit-norm mean latent ``mu_hat`` [B, latent_dim], proprio_per_frame)."""
        # Safety net for the lora + no-warmstart path (idempotent; load_state_dict
        # handles the warmstart/resume paths). NOTE for the parallel M4 (residual)
        # work: keep this guard line if this method is rewritten.
        if self.adapter == "lora":
            self._ensure_lora_applied()
        # Option C: strip the obstacle block so the frozen encoder sees its exact dim.
        enc_obs, obstacle_feat, obstacle_mask = self._split_policy_obs(obs)
        mu, _log_sigma, proprio_per_frame = self.muse.transformer_encoder.encode(enc_obs)
        mu_hat = self.muse._maybe_normalize_latent(mu)  # unit-norm; decoder geometry
        if getattr(self, "residual_corrector", None) is not None:
            # Residual adapter (M4 / D2): shallow per-body-token transformer g_φ
            # corrects the frozen encoder's latent. It consumes the SAME split
            # tensors the encoder sees (split_obs is parse-only, no weights;
            # masked KP excluded structurally via key_padding_mask inside g_φ)
            # plus μ̂ AND the obstacle tokens (option C). μ̂ is constant w.r.t.
            # trainable params (encoder frozen) -> detach; grad flows only through
            # g_φ. One re-normalize keeps the unit-norm decoder geometry.
            kp, kp_mask, _prop = self.muse.transformer_encoder.split_obs(enc_obs)
            base = mu_hat.detach()
            delta = self.residual_corrector(kp, kp_mask, proprio_per_frame, base, obstacle_feat, obstacle_mask)
            mu_hat = self.muse._maybe_normalize_latent(
                base + self.residual_corrector.alpha * delta
            )
        return mu_hat, proprio_per_frame

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean, _proprio = self._encode_mean_latent(observations)
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

    def decode_for_env(self, latent: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """Frozen decoder: renormalize the sampled latent and decode to a joint action.

        Called by the runner between ``alg.act`` (which returned/stored the latent)
        and ``env.step``. Deterministic, no grad (decoder is frozen; PPO already
        detached the stored action).
        """
        with torch.no_grad():
            enc_obs, _f, _m = self._split_policy_obs(observations)  # option C: drop obstacle block
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
            mu_cur, _, _ = self.muse.transformer_encoder.encode(observations)
            mu_ref, _, _ = self._ref_encoder.encode(observations)
            mu_cur = self.muse._maybe_normalize_latent(mu_cur)
            mu_ref = self.muse._maybe_normalize_latent(mu_ref)
            return nn.functional.cosine_similarity(mu_cur, mu_ref, dim=-1)  # [N]

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    def reset(self, dones=None) -> None:
        return

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
        super().load_state_dict(state_dict, strict=strict)
        return True
