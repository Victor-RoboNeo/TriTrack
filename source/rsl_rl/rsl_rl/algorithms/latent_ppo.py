# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Latent-space PPO for MUSE-Kp RL finetuning.

PPO in this repo is action-agnostic (see :class:`rsl_rl.algorithms.ppo.PPO`):
ratio / advantage / log-prob / entropy are computed against whatever
``policy.act`` returns and whatever distribution the policy exposes. With
:class:`rsl_rl.modules.LatentRLActorCritic` the action IS the latent, so
**plain PPO is already latent-space RL** — no algorithm changes are needed.

The only reason this subclass exists is to carry a distinct ``class_name`` so
the runner can:
  * resolve ``training_type = "latent_rl"`` (-> size rollout storage to
    ``latent_dim`` and decode latent->joint before ``env.step``), and
  * keep stock ``PPO`` untouched.

This is also where the optional **prior anchor** (decision D3: full_ft run with
KL/penalty toward the distilled latent) will land in M1b — added as one extra
loss term in :meth:`update`. M1 ships the unanchored variant: a pure
pass-through so the latent-PPO loop can be validated first (decision D6).
"""

from __future__ import annotations

from rsl_rl.algorithms.ppo import PPO


class LatentPPO(PPO):
    """PPO over the latent action space (frozen decoder = motor prior).

    M1: behaviourally identical to :class:`PPO`. M1b will override
    :meth:`update` to add the prior-anchor term.
    """

    def __init__(
        self,
        policy,
        *,
        encoder_decoder_warmstart_checkpoint_path: str | None = None,
        teacher_checkpoint_path: str | None = None,
        prior_anchor_coef: float = 0.0,
        **kwargs,
    ):
        # These are consumed by the *runner* (the latent_rl warmstart block reads
        # ``alg_cfg["encoder_decoder_warmstart_checkpoint_path"]``; the
        # distillation gate reads ``teacher_checkpoint_path``) or by this class
        # (``prior_anchor_coef``). They arrive via ``**alg_cfg`` but are NOT
        # ``PPO.__init__`` params — capture so stock PPO doesn't choke.
        self.encoder_decoder_warmstart_checkpoint_path = encoder_decoder_warmstart_checkpoint_path
        self.teacher_checkpoint_path = teacher_checkpoint_path
        self.prior_anchor_coef = float(prior_anchor_coef)
        self._opt_synced_late_params = False
        super().__init__(policy, **kwargs)

    def _ensure_optimizer_covers_policy(self) -> None:
        """Rebuild the optimizer to cover trainable params created AFTER
        ``PPO.__init__`` built it.

        Critical for the LoRA adapter: LoRA injection is *deferred* until the
        distilled warmstart is loaded (``LatentRLActorCritic._ensure_lora_applied``
        runs inside ``policy.load_state_dict``), but the runner builds the
        algorithm — hence ``optim.Adam(policy.parameters())`` in
        ``PPO.__init__`` — *before* that warmstart. So the LoRA ``A``/``B``
        tensors are orphaned: they get gradients but ``optimizer.step()`` never
        touches them ⇒ the encoder mean is frozen ⇒ ratio≡1 ⇒ surrogate≡0
        forever (observed: `…_latentrl_lora` surrogate 0 past 300 steps).

        Runs once, at the first ``update()`` (after warmstart + LoRA injection,
        before any actor step). No-op for full_ft/residual (their trainable
        params exist at ``__init__``, already covered) — only LoRA triggers a
        rebuild. Fresh Adam state is correct here: these params have taken zero
        steps. Mirrors ``PPO.__init__`` (passes ``policy.parameters()``; frozen
        params just never receive grad).
        """
        if self._opt_synced_late_params:
            return
        self._opt_synced_late_params = True
        covered = {id(p) for grp in self.optimizer.param_groups for p in grp["params"]}
        missing = [
            p for p in self.policy.parameters() if p.requires_grad and id(p) not in covered
        ]
        if missing:
            import torch.optim as optim

            self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
            print(
                f"[LatentPPO] optimizer rebuilt to include {len(missing)} late-created "
                f"trainable params (deferred LoRA adapters); without this the encoder "
                f"would never train (surrogate≡0)."
            )

    def update(self):
        # Must precede the first optimizer use in PPO.update().
        self._ensure_optimizer_covers_policy()
        return super().update()

    def process_env_step(self, rewards, dones, infos):
        """Decision D3 prior anchor (reward-side operationalization).

        When ``prior_anchor_coef > 0`` (the *anchored* full_ft run), subtract
        ``coef * (1 - cos(mu, mu_distilled))`` from the per-step reward so the
        encoder is pulled toward the distilled latent — the safe-start
        regularizer that makes the anchored-vs-unanchored comparison honest
        (full_ft would otherwise drift immediately, unlike LoRA/residual which
        start at the prior for free). ``coef == 0`` (default) ⇒ identical to
        stock PPO (the unanchored M1 path), so this override is a no-op there.

        Reward-side (not a policy-loss term) is a deliberate, equivalent, far
        more maintainable choice than forking PPO's monolithic ``update`` —
        see :meth:`LatentRLActorCritic.prior_anchor_cos`. ``self.transition``
        was just populated by ``PPO.act`` (incl. ``.observations``), so the
        anchor is evaluated on exactly the obs the policy acted on this step.
        """
        if self.prior_anchor_coef > 0.0:
            cos = self.policy.prior_anchor_cos(self.transition.observations)  # [N]
            penalty = self.prior_anchor_coef * (1.0 - cos)
            rewards = rewards - penalty.reshape(rewards.shape).to(rewards.device, rewards.dtype)
        return super().process_env_step(rewards, dones, infos)
