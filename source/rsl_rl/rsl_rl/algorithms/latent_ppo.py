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

import torch

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
        parent_residual_anchor_coef: float = 0.0,
        elastic_latent_coef: float = 0.0,
        elastic_damp_coef: float = 0.05,
        elastic_theta_easy_deg: float = 10.0,
        elastic_theta_hard_deg: float = 25.0,
        elastic_k_easy: float = 1.0,
        elastic_k_hard: float = 0.25,
        terrain_zero_coef: float = 0.0,
        terrain_calm_coef: float = 0.0,
        recovery_progress_coef: float = 0.0,
        recovery_progress_clip: float = 1.0,
        recovery_release_coef: float = 0.0,
        interaction_dz_coef: float = 0.0,
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
        self.parent_residual_anchor_coef = float(parent_residual_anchor_coef)
        self.elastic_latent_coef = float(elastic_latent_coef)
        self.elastic_damp_coef = float(elastic_damp_coef)
        self.elastic_theta_easy_deg = float(elastic_theta_easy_deg)
        self.elastic_theta_hard_deg = float(elastic_theta_hard_deg)
        self.elastic_k_easy = float(elastic_k_easy)
        self.elastic_k_hard = float(elastic_k_hard)
        self.terrain_zero_coef = float(terrain_zero_coef)
        self.terrain_calm_coef = float(terrain_calm_coef)
        self.recovery_progress_coef = float(recovery_progress_coef)
        self.recovery_progress_clip = float(recovery_progress_clip)
        self.recovery_release_coef = float(recovery_release_coef)
        self.interaction_dz_coef = float(interaction_dz_coef)
        if self.recovery_release_coef != 0.0:
            raise ValueError(
                "R1a-2a is progress-only; recovery_release_coef must be 0 "
                f"(got {self.recovery_release_coef})"
            )
        self.easy_env_mask = None
        self.terrain_group_id = None
        self._anchor_logs: dict = {}
        self._terrain_logs: dict = {}
        self._recovery_logs: dict = {}
        self._tr_cnt = None
        self._tr_sum = None
        self._rec_n = 0
        self._rec_sum: dict[str, float] | None = None
        self._rec_tracker = None
        self._prog_acc: dict[str, float] | None = None
        self._prog_missing_obs = False
        self._opt_synced_late_params = False
        self._live_prev_disp = None
        self._damp_valid = None
        self._prev_disp_buf = None
        self._damp_valid_buf = None
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

    @staticmethod
    def _grad_norm(grads) -> float:
        sq = 0.0
        for g in grads:
            if g is None:
                continue
            sq += float(g.detach().float().pow(2).sum().item())
        return sq ** 0.5

    def _augment_actor_loss(self, loss, obs_batch, surrogate_loss):
        if float(self.elastic_latent_coef) > 0.0:
            return self._augment_elastic(loss, obs_batch, surrogate_loss)
        if float(self.terrain_zero_coef) > 0.0 or float(self.terrain_calm_coef) > 0.0:
            return self._augment_terrain_aux(loss, obs_batch, surrogate_loss)
        coef = float(self.parent_residual_anchor_coef)
        parent = getattr(self.policy, "parent_residual_corrector", None)
        if coef <= 0.0 or parent is None or self.easy_env_mask is None:
            return loss
        easy_env = self.easy_env_mask
        if not bool(easy_env.any()):
            return loss
        batch_idx = getattr(self.storage, "_batch_idx", None)
        if batch_idx is None:
            return loss
        env_ids = batch_idx % int(self.storage.num_envs)
        easy = easy_env[env_ids]
        hard = ~easy
        delta, delta0 = self.policy.residual_deltas(obs_batch)
        per = (delta - delta0).pow(2).sum(dim=-1)
        n_easy = easy.float().sum().clamp_min(1.0)
        anchor = (per * easy.float()).sum() / n_easy
        drift_easy = (delta - delta0).norm(dim=-1)
        logs = {
            "parent_anchor": float(anchor.detach().item()),
            "delta_z_drift_easy": float(drift_easy[easy].mean().detach().item()) if bool(easy.any()) else 0.0,
            "delta_z_drift_hard": float(drift_easy[hard].mean().detach().item()) if bool(hard.any()) else 0.0,
        }
        std = torch.exp(self.policy.latent_log_std).detach().clamp_min(1e-6)
        kl_el = 0.5 * ((delta - delta0) / std).pow(2).sum(dim=-1)
        logs["kl_to_parent_easy"] = float((kl_el * easy.float()).sum().detach().item() / float(n_easy))
        if not getattr(self, "_logged_anchor_grads", False):
            params = [p for p in self.policy.residual_corrector.parameters() if p.requires_grad]
            try:
                g_ppo = torch.autograd.grad(surrogate_loss, params, retain_graph=True, allow_unused=True)
                g_anc = torch.autograd.grad(anchor, params, retain_graph=True, allow_unused=True)
                logs["grad_norm_ppo"] = self._grad_norm(g_ppo)
                logs["grad_norm_anchor"] = coef * self._grad_norm(g_anc)
            except RuntimeError:
                pass
            self._logged_anchor_grads = True
            self._anchor_logs = logs
        else:
            kept = {k: self._anchor_logs[k] for k in ("grad_norm_ppo", "grad_norm_anchor") if k in self._anchor_logs}
            logs.update(kept)
            self._anchor_logs = logs
        return loss + coef * anchor

    def _augment_elastic(self, loss, obs_batch, surrogate_loss):
        parent = getattr(self.policy, "parent_residual_corrector", None)
        if parent is None or self.easy_env_mask is None:
            return loss
        batch_idx = getattr(self.storage, "_batch_idx", None)
        if batch_idx is None:
            return loss
        env_ids = batch_idx % int(self.storage.num_envs)
        easy = self.easy_env_mask[env_ids]
        hard = ~easy
        z_exec, z_parent = self.policy.exec_and_parent_latents(obs_batch)
        z_exec = torch.nn.functional.normalize(z_exec, dim=-1)
        z_parent = torch.nn.functional.normalize(z_parent.detach(), dim=-1)
        cos = (z_exec * z_parent).sum(-1).clamp(-0.9999, 0.9999)
        theta = torch.acos(cos)
        hard_f = hard.float()
        theta_free = torch.deg2rad(
            torch.as_tensor(self.elastic_theta_easy_deg, device=z_exec.device, dtype=z_exec.dtype)
        ) * (1.0 - hard_f) + torch.deg2rad(
            torch.as_tensor(self.elastic_theta_hard_deg, device=z_exec.device, dtype=z_exec.dtype)
        ) * hard_f
        k = (1.0 - hard_f) * float(self.elastic_k_easy) + hard_f * float(self.elastic_k_hard)
        extension = torch.relu(theta - theta_free)
        spring = 0.5 * k * extension.square()
        disp = z_exec - z_parent
        damp = torch.zeros_like(spring)
        if self._prev_disp_buf is not None:
            prev = self._prev_disp_buf.reshape(-1, disp.shape[-1])[batch_idx].to(disp.device, disp.dtype)
            valid = self._damp_valid_buf.reshape(-1)[batch_idx].to(disp.device)
            damp = (disp - prev.detach()).square().sum(-1) * valid.float()
        spring_mean = spring.mean()
        damp_mean = damp.mean()
        elastic = spring_mean + float(self.elastic_damp_coef) * damp_mean
        coef = float(self.elastic_latent_coef)
        rad2deg = 180.0 / 3.141592653589793
        logs = {
            "spring_angle_easy": float(theta[easy].mean().detach().item() * rad2deg) if bool(easy.any()) else 0.0,
            "spring_angle_hard": float(theta[hard].mean().detach().item() * rad2deg) if bool(hard.any()) else 0.0,
            "spring_ext_easy": float(extension[easy].mean().detach().item() * rad2deg) if bool(easy.any()) else 0.0,
            "spring_ext_hard": float(extension[hard].mean().detach().item() * rad2deg) if bool(hard.any()) else 0.0,
            "spring_energy": float(spring_mean.detach().item()),
            "damping_energy": float(damp_mean.detach().item()),
        }
        if not getattr(self, "_logged_anchor_grads", False):
            params = [p for p in self.policy.residual_corrector.parameters() if p.requires_grad]
            try:
                g_ppo = torch.autograd.grad(surrogate_loss, params, retain_graph=True, allow_unused=True)
                g_el = torch.autograd.grad(elastic, params, retain_graph=True, allow_unused=True)
                logs["grad_norm_ppo"] = self._grad_norm(g_ppo)
                logs["grad_norm_elastic"] = coef * self._grad_norm(g_el)
            except RuntimeError:
                pass
            self._logged_anchor_grads = True
            self._anchor_logs = logs
        else:
            kept = {k: self._anchor_logs[k] for k in ("grad_norm_ppo", "grad_norm_elastic") if k in self._anchor_logs}
            logs.update(kept)
            self._anchor_logs = logs
        return loss + coef * elastic

    def _augment_terrain_aux(self, loss, obs_batch, surrogate_loss):
        """P2-D: ``L_zero=‖h_η(0)‖²`` + ``L_calm=(1-α)‖ū‖²``. Log gradient norms only."""
        if getattr(self.policy, "terrain_residual", None) is None:
            return loss
        terms = self.policy.terrain_aux_terms(obs_batch)
        if terms is None:
            return loss
        l_zero, l_calm = terms
        lz = float(self.terrain_zero_coef) * l_zero
        lc = float(self.terrain_calm_coef) * l_calm
        logs = {
            "terrain_l_zero": float(l_zero.detach().item()),
            "terrain_l_calm": float(l_calm.detach().item()),
        }
        if not getattr(self, "_logged_anchor_grads", False):
            params = [p for p in self.policy.terrain_residual.parameters() if p.requires_grad]
            try:
                g_ppo = torch.autograd.grad(surrogate_loss, params, retain_graph=True, allow_unused=True)
                g_z = torch.autograd.grad(l_zero, params, retain_graph=True, allow_unused=True)
                g_c = torch.autograd.grad(l_calm, params, retain_graph=True, allow_unused=True)
                logs["grad_norm_ppo"] = self._grad_norm(g_ppo)
                logs["grad_norm_zero"] = float(self.terrain_zero_coef) * self._grad_norm(g_z)
                logs["grad_norm_calm"] = float(self.terrain_calm_coef) * self._grad_norm(g_c)
            except RuntimeError:
                pass
            self._logged_anchor_grads = True
            self._anchor_logs = logs
        else:
            kept = {
                k: self._anchor_logs[k]
                for k in ("grad_norm_ppo", "grad_norm_zero", "grad_norm_calm")
                if k in self._anchor_logs
            }
            logs.update(kept)
            self._anchor_logs = logs
        return loss + lz + lc

    _TERRAIN_NAMES = ("flat", "light", "slope", "steps")

    def _accum_terrain_step(self) -> None:
        gid = self.terrain_group_id
        if gid is None or getattr(self.policy, "terrain_residual", None) is None:
            return
        stats = self.policy.last_terrain_stats()
        if stats is None:
            return
        if self._tr_cnt is None:
            self._tr_cnt = {n: 0 for n in self._TERRAIN_NAMES}
            self._tr_sum = {
                f"{k}_{n}": 0.0
                for n in self._TERRAIN_NAMES
                for k in ("u", "ang", "sens", "alpha", "rho", "dz", "sat")
            }
        gid = gid.to(device=stats["u_perp"].device)
        for i, name in enumerate(self._TERRAIN_NAMES):
            m = gid == i
            n = int(m.sum().item())
            if n == 0:
                continue
            self._tr_cnt[name] += n
            self._tr_sum[f"u_{name}"] += float(stats["u_perp"][m].sum().item())
            self._tr_sum[f"ang_{name}"] += float(stats["ang_deg"][m].sum().item())
            self._tr_sum[f"sens_{name}"] += float(stats["scan_sens"][m].sum().item())
            if "alpha" in stats:
                self._tr_sum[f"alpha_{name}"] += float(stats["alpha"][m].sum().item())
                self._tr_sum[f"rho_{name}"] += float(stats["rho_auth"][m].sum().item())
                self._tr_sum[f"dz_{name}"] += float(stats["dz"][m].sum().item())
                self._tr_sum[f"sat_{name}"] += float(stats["sat"][m].sum().item())

    def _flush_terrain_logs(self) -> None:
        if not self._tr_cnt:
            return
        logs = {}
        for name in self._TERRAIN_NAMES:
            n = int(self._tr_cnt[name])
            den = float(max(n, 1))
            logs[f"terrain_u_{name}"] = self._tr_sum[f"u_{name}"] / den
            logs[f"terrain_ang_{name}"] = self._tr_sum[f"ang_{name}"] / den
            logs[f"terrain_sens_{name}"] = self._tr_sum[f"sens_{name}"] / den
            logs[f"terrain_alpha_{name}"] = self._tr_sum[f"alpha_{name}"] / den
            logs[f"terrain_rho_{name}"] = self._tr_sum[f"rho_{name}"] / den
            logs[f"terrain_dz_{name}"] = self._tr_sum[f"dz_{name}"] / den
            logs[f"terrain_sat_{name}"] = self._tr_sum[f"sat_{name}"] / den
        self._terrain_logs = logs
        self._tr_cnt = None
        self._tr_sum = None

    _REC_KEYS = ("E", "R_E", "R_S", "R", "alpha", "r_raw_norm", "r_perp_norm", "r_bar_norm", "rho_r", "theta_deg")

    def _accum_recovery_step(self, dones=None, infos=None) -> None:
        if getattr(self.policy, "intent_recovery", False) is not True and getattr(
            self.policy, "interaction_recovery", False
        ) is not True:
            return
        stats = self.policy.last_recovery_stats()
        if stats is None:
            return
        if self._rec_sum is None:
            self._rec_sum = {k: 0.0 for k in self._REC_KEYS}
            self._rec_n = 0
        n = int(stats["alpha"].shape[0])
        self._rec_n += n
        for k in self._REC_KEYS:
            if k not in stats:
                continue
            self._rec_sum[k] += float(stats[k].sum().item())
        if self._rec_tracker is None:
            from rsl_rl.modules.intent_recovery import RecoveryRolloutTracker

            self._rec_tracker = RecoveryRolloutTracker()
        tout = None
        if infos is not None and "time_outs" in infos:
            tout = infos["time_outs"]
        if dones is None:
            return
        self._rec_tracker.step(stats, self.terrain_group_id, dones, tout)

    def _flush_recovery_logs(self) -> None:
        logs: dict[str, float] = {}
        if self._rec_n and self._rec_sum:
            den = float(self._rec_n)
            logs.update({f"recovery_{k}": self._rec_sum[k] / den for k in self._REC_KEYS})
        if self._rec_tracker is not None:
            logs.update(self._rec_tracker.flush())
        if self._prog_acc:
            a = self._prog_acc
            n = max(float(a["n"]), 1.0)
            n_act = max(float(a["n_active"]), 1.0)
            logs["recovery_r_can"] = a["r_can"] / n
            logs["recovery_r_prog"] = a["r_prog"] / n
            logs["recovery_r_prog_active"] = a["r_prog_active"] / n_act
            logs["recovery_r_prog_ratio"] = a["ratio_num"] / max(a["ratio_den"], 1e-8)
            logs["recovery_prog_done_zero"] = a["done_prog"] / max(float(a["n_done"]), 1.0)
            logs["recovery_prog_reset_leak"] = a["leak"] / max(float(a["n_done_active"]), 1.0)
        self._recovery_logs = {k: v for k, v in logs.items() if v == v}  # drop NaN
        self._rec_sum = None
        self._rec_n = 0
        self._prog_acc = None

    def update(self):
        # Must precede the first optimizer use in PPO.update().
        self._ensure_optimizer_covers_policy()
        self._logged_anchor_grads = False
        self._anchor_logs = {}
        self._flush_terrain_logs()
        self._flush_recovery_logs()
        loss_dict = super().update()
        if self._anchor_logs:
            loss_dict.update(self._anchor_logs)
        if self._terrain_logs:
            loss_dict.update(self._terrain_logs)
            self._terrain_logs = {}
        if self._recovery_logs:
            loss_dict.update(self._recovery_logs)
            self._recovery_logs = {}
        return loss_dict

    def _apply_recovery_progress(self, rewards, dones, infos):
        """Add ``λ_p · 1[active_t] clip(ΔR_E, −c, c)``. ``done ⇒ 0`` (no reset leak)."""
        rec_on = bool(getattr(self.policy, "intent_recovery", False)) or bool(
            getattr(self.policy, "interaction_recovery", False)
        )
        if self.recovery_progress_coef == 0.0 or not rec_on:
            return rewards
        from rsl_rl.modules.intent_recovery import (
            recovery_progress_reset_leak,
            recovery_progress_reward,
        )

        stats = self.policy.last_recovery_stats()
        if stats is None or "R_E" not in stats or "active" not in stats:
            return rewards
        obs_dict = infos.get("observations") if infos is not None else None
        next_obs = None
        if isinstance(obs_dict, dict):
            next_obs = obs_dict.get("policy")
        if next_obs is None:
            if not self._prog_missing_obs:
                print("[P2-R R1a-2] infos['observations']['policy'] missing; progress reward skipped")
                self._prog_missing_obs = True
            return rewards
        next_obs = next_obs.to(device=stats["R_E"].device, dtype=stats["R_E"].dtype)
        r_next = self.policy.recovery_R_E_from_obs(next_obs, obs_is_normalized=False)
        done = dones.reshape(-1).to(device=stats["R_E"].device)
        r_prog = recovery_progress_reward(
            stats["active"],
            stats["R_E"],
            r_next,
            done,
            clip_p=self.recovery_progress_clip,
        )
        leak = recovery_progress_reset_leak(
            stats["active"],
            stats["R_E"],
            r_next,
            done,
            clip_p=self.recovery_progress_clip,
        )
        lam = float(self.recovery_progress_coef)
        bonus = (lam * r_prog).to(device=rewards.device, dtype=rewards.dtype)
        r_can = rewards.reshape(-1)
        bonus_flat = bonus.reshape(-1)
        rewards = rewards + bonus.reshape(rewards.shape)

        active = (stats["active"].reshape(-1) > 0).to(device=bonus_flat.device)
        done_b = done.to(device=bonus_flat.device, dtype=torch.bool)
        r_prog_d = r_prog.to(device=bonus_flat.device)
        leak_d = leak.to(device=bonus_flat.device)
        if self._prog_acc is None:
            self._prog_acc = {
                "r_can": 0.0,
                "r_prog": 0.0,
                "r_prog_active": 0.0,
                "ratio_num": 0.0,
                "ratio_den": 0.0,
                "done_prog": 0.0,
                "leak": 0.0,
                "n": 0.0,
                "n_active": 0.0,
                "n_done": 0.0,
                "n_done_active": 0.0,
            }
        a = self._prog_acc
        a["n"] += float(r_can.numel())
        a["r_can"] += float(r_can.sum().item())
        a["r_prog"] += float(bonus_flat.sum().item())
        n_act = int(active.sum().item())
        a["n_active"] += float(n_act)
        if n_act:
            a["r_prog_active"] += float(bonus_flat[active].sum().item())
            live = active & ~done_b
            if bool(live.any()):
                num = bonus_flat[live].abs()
                den = r_can[live].abs() + 1e-6
                a["ratio_num"] += float(num.sum().item())
                a["ratio_den"] += float(den.sum().item())
        n_done = int(done_b.sum().item())
        a["n_done"] += float(n_done)
        if n_done:
            a["done_prog"] += float(r_prog_d[done_b].abs().sum().item())
        n_da = int((active & done_b).sum().item())
        a["n_done_active"] += float(n_da)
        if n_da:
            a["leak"] += float(leak_d[active & done_b].sum().item())
        return rewards

    def process_env_step(self, rewards, dones, infos):
        self._accum_terrain_step()
        self._accum_recovery_step(dones=dones, infos=infos)
        rewards = self._apply_recovery_progress(rewards, dones, infos)
        lam_z = float(getattr(self, "interaction_dz_coef", 0.0) or 0.0)
        if lam_z > 0.0 and bool(getattr(self.policy, "interaction_recovery", False)):
            stats = self.policy.last_recovery_stats()
            if stats is not None and "r_bar_norm" in stats:
                pen = lam_z * stats["r_bar_norm"].square()
                rewards = rewards - pen.reshape(rewards.shape).to(rewards.device, rewards.dtype)
        if self.elastic_latent_coef > 0.0 and getattr(self.policy, "parent_residual_corrector", None) is not None:
            obs = self.transition.observations
            n = int(obs.shape[0])
            with torch.no_grad():
                z_exec, z_parent = self.policy.exec_and_parent_latents(obs)
                disp = z_exec - z_parent
            if self._live_prev_disp is None or int(self._live_prev_disp.shape[0]) != n:
                dim = int(disp.shape[-1])
                t_max = int(self.storage.num_transitions_per_env)
                self._live_prev_disp = torch.zeros(n, dim, device=disp.device, dtype=disp.dtype)
                self._damp_valid = torch.zeros(n, dtype=torch.bool, device=disp.device)
                self._prev_disp_buf = torch.zeros(t_max, n, dim, device=disp.device, dtype=disp.dtype)
                self._damp_valid_buf = torch.zeros(t_max, n, dtype=torch.bool, device=disp.device)
            step = int(self.storage.step)
            if step < self._prev_disp_buf.shape[0]:
                self._prev_disp_buf[step].copy_(self._live_prev_disp)
                self._damp_valid_buf[step].copy_(self._damp_valid)
            done = dones.reshape(-1).to(dtype=torch.bool)
            self._live_prev_disp = disp
            self._damp_valid.fill_(True)
            self._damp_valid[done] = False
            self._live_prev_disp[done] = 0
        if self.prior_anchor_coef > 0.0:
            cos = self.policy.prior_anchor_cos(self.transition.observations)  # [N]
            penalty = self.prior_anchor_coef * (1.0 - cos)
            rewards = rewards - penalty.reshape(rewards.shape).to(rewards.device, rewards.dtype)
        return super().process_env_step(rewards, dones, infos)
