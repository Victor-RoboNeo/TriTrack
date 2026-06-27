"""MUSE co-train distillation: dual-modality (joint-cmd + KP) BC + smoothness + alignment.

Forward contract:
- ``policy.act(obs)`` does the per-env 50/50 piloting (both encoders forward; partition selects
  which action drives ``env.step()``). Stored in the rollout for env state propagation only.
- ``policy.forward_for_update(obs)`` re-forwards both encoders on the full obs batch and returns
  ``{action_jc, action_kp, mu_jc, mu_kp, log_sigma_jc, log_sigma_kp}``.

Loss (per transition, summed across all envs):
    L = MSE(action_jc, a_teacher) + MSE(action_kp, a_teacher)            # 2× behavior
      + λ_reg_jc · (1 − cos(μ_jc^t, μ_jc^{t-1})) + λ_reg_kp · (1 − cos(μ_kp^t, μ_kp^{t-1}))   # smoothness
      + λ_align · (1 − cos(sg(μ_jc^t), μ_kp^t))                          # cross-modal alignment

Alignment is **asymmetric**: ``μ_jc`` is stop-gradient'd so KP chases the privileged JC latent as
a fixed gold target. JC never gets dragged toward the (masked) KP latent — it trains only via its
own behavior loss + smoothness. Smoothness weights are split per modality
(``weight_regularization_jc`` vs ``weight_regularization_kp``) so the privileged JC encoder stays
regularized while the KP encoder is free to adapt.
All terms operate on the FULL batch every step. Smoothness uses the same masked-mean reduction
as MuseDistillation; alignment is a plain mean over the batch (no done-masking — same-step pairs).

Optimizer / gradient-accumulation / multi-GPU bookkeeping is identical to MuseDistillation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms.muse_distillation import (
    MuseDistillation,
    _muse_cosine_smoothness_regularization,
    _muse_temporal_mu_regularization,
)


def _cross_modal_cosine_alignment(
    mu_a: torch.Tensor, mu_b: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """L_align = mean(1 − cos(μ_a, μ_b)). Range [0, 2]; 0 ⇒ perfect agreement.

    No done-masking needed — both encoders observe the same step, so every (μ_a, μ_b) pair is a
    valid same-step pair regardless of episode boundaries.
    """
    cos_sim = nn.functional.cosine_similarity(mu_a, mu_b, dim=-1, eps=eps)  # [B]
    return (1.0 - cos_sim).mean()


class MuseCoTrainDistillation(MuseDistillation):
    """MuseDistillation extended to co-train two modalities with a cross-modal alignment loss.

    Same optimizer / storage / multi-GPU contract as MuseDistillation; only :meth:`update` is
    overridden because the loss and forward contract are different. ``__init__`` adds the
    ``weight_align`` knob and otherwise forwards to the parent.
    """

    def __init__(
        self,
        policy,
        *,
        weight_align: float = 0.1,
        weight_regularization_jc: float = 0.1,
        weight_regularization_kp: float = 0.01,
        **kwargs,
    ):
        # Don't forward ``weight_regularization`` to parent — cotrain's update path uses the
        # split per-modality weights below and never reads ``self.weight_regularization``.
        kwargs.pop("weight_regularization", None)
        super().__init__(policy, **kwargs)
        self.weight_align = float(weight_align)
        self.weight_regularization_jc = float(weight_regularization_jc)
        self.weight_regularization_kp = float(weight_regularization_kp)
        # Per-pilot rollout counters, flushed by the runner at log() time. Eagerly allocated
        # here (outside inference_mode) so in-place ops at flush() don't trip the
        # PyTorch "inplace update to inference tensor" guard. Layout: 0=JC pilot, 1=KP pilot.
        #
        # Semantics: ``success_rate`` mirrors Episode/mdp_termination_success_rate — of the
        # episodes that ENDED this iteration under a given pilot, what fraction ended cleanly
        # (timeout) vs. fell (reset_terminated=True). NOT a step-wise alive-rate, which would
        # trivially approach 1 as episode length grows.
        #
        # Pilot attribution at the termination step: whichever pilot was driving when the
        # env terminated. Episodes are mixed-pilot under the per-step partition, so this is
        # the cleanest causal attribution available without per-episode pilot sampling.
        try:
            _device = next(policy.parameters()).device
        except StopIteration:
            _device = torch.device("cpu")
        self._pilot_episode_end_counts: torch.Tensor = torch.zeros(2, device=_device, dtype=torch.long)
        self._pilot_failure_counts: torch.Tensor = torch.zeros(2, device=_device, dtype=torch.long)
        # ``env_unwrapped`` is plumbed in by ``MotionOnPolicyRunner.__init__`` so we can read
        # ``reset_terminated`` (per-env, non-timeout terminations) at process_env_step time.
        # ``dones`` alone collapses timeouts and falls; ``reset_terminated`` keeps them separate.
        self._env_unwrapped = None

    def process_env_step(self, rewards, dones, infos) -> None:
        """Parent transition handling + per-pilot episode-end / failure accumulation."""
        is_kp = getattr(self.policy, "_last_pilot_kp_mask", None)
        env_u = self._env_unwrapped
        if is_kp is not None and env_u is not None and hasattr(env_u, "reset_terminated"):
            device = self._pilot_episode_end_counts.device
            done_mask = dones.bool().view(-1).to(device=device)
            if done_mask.any():
                pilot_idx = is_kp.to(device=device, dtype=torch.long)  # [N], 0=jc, 1=kp
                fail_mask = env_u.reset_terminated.bool().view(-1).to(device=device)
                self._pilot_episode_end_counts.index_add_(0, pilot_idx, done_mask.to(dtype=torch.long))
                self._pilot_failure_counts.index_add_(0, pilot_idx, (done_mask & fail_mask).to(dtype=torch.long))
        super().process_env_step(rewards, dones, infos)

    def flush_pilot_counts(self) -> dict[str, float]:
        """Return per-pilot success rate + episode-end / failure totals, then reset.

        Called from runner.log(). Pilots whose envs did not finish any episode this iteration
        are omitted (no defined success rate)."""
        ends = self._pilot_episode_end_counts.detach().cpu()
        fails = self._pilot_failure_counts.detach().cpu()
        out: dict[str, float] = {}
        for pi, pilot in enumerate(("jc", "kp")):
            e = int(ends[pi].item())
            f = int(fails[pi].item())
            if e == 0:
                continue
            out[f"pilot/{pilot}/episode_ends"] = float(e)
            out[f"pilot/{pilot}/failures"] = float(f)
            out[f"pilot/{pilot}/success_rate"] = 1.0 - (float(f) / float(e))
        self._pilot_episode_end_counts.zero_()
        self._pilot_failure_counts.zero_()
        return out

    def update(self, current_iter: int | None = None) -> dict:
        del current_iter
        self.num_updates += 1

        # Per-modality BC + smoothness running sums.
        mean_behavior_jc = 0.0
        mean_behavior_kp = 0.0
        mean_reg_jc = 0.0
        mean_reg_kp = 0.0
        mean_reg_jc_contribution = 0.0
        mean_reg_kp_contribution = 0.0
        mean_align = 0.0
        mean_align_contribution = 0.0

        # v1 mask-stratified latent telemetry. Per-sample 1−cos(μ_kp, sg μ_jc) summed into
        # KP-visible-fraction bins [heavy <1/3, mid, light ≥2/3] (Job-A reachability), plus
        # heavy-mask μ_kp distance to the batch JC-latent mean direction (Job-B: is the
        # decoder-freeze-safe region holding?).
        _bin_sum = [0.0, 0.0, 0.0]
        _bin_cnt = [0, 0, 0]
        _heavy_jcmean_sum = 0.0
        _heavy_jcmean_cnt = 0

        # μ-scale diagnostics (per modality).
        mu_norm_jc_sum = 0.0
        mu_norm_kp_sum = 0.0
        mu_stats_count = 0

        cnt = 0
        total_loss_accumulator: torch.Tensor | None = None

        for _epoch in range(self.num_learning_epochs):
            prev_mu_jc: torch.Tensor | None = None
            prev_mu_kp: torch.Tensor | None = None
            prev_dones: torch.Tensor | None = None
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()

            for _t, (obs, _, _, privileged_actions, dones) in enumerate(
                self.storage.generator()
            ):
                fwd = self.policy.forward_for_update(obs, sample_z=True)
                action_jc = fwd["action_jc"]
                action_kp = fwd["action_kp"]
                mu_jc = fwd["mu_jc"]
                mu_kp = fwd["mu_kp"]

                # 1. Behavior loss (both modalities, same teacher target).
                behavior_jc = self.loss_fn(action_jc, privileged_actions)
                behavior_kp = self.loss_fn(action_kp, privileged_actions)
                mean_behavior_jc += behavior_jc.item()
                mean_behavior_kp += behavior_kp.item()
                total_loss = behavior_jc + behavior_kp

                # μ-scale logging.
                with torch.no_grad():
                    mu_jc_flat = mu_jc.detach().reshape(-1, mu_jc.shape[-1])
                    mu_kp_flat = mu_kp.detach().reshape(-1, mu_kp.shape[-1])
                    if mu_jc_flat.shape[0] > 1:
                        mu_norm_jc_sum += float(mu_jc_flat.norm(dim=-1).mean().item())
                        mu_norm_kp_sum += float(mu_kp_flat.norm(dim=-1).mean().item())
                        mu_stats_count += 1

                # 2. Per-modality temporal smoothness (cosine or l2 — gated by smoothness_type).
                smoothness_fn = (
                    _muse_cosine_smoothness_regularization
                    if self.smoothness_type == "cosine"
                    else _muse_temporal_mu_regularization
                )
                reg_jc_loss = torch.zeros((), device=obs.device, dtype=behavior_jc.dtype)
                reg_kp_loss = torch.zeros((), device=obs.device, dtype=behavior_jc.dtype)
                reg_valid = False
                if prev_mu_jc is not None and prev_mu_kp is not None and prev_dones is not None:
                    reg_jc_loss, valid_jc = smoothness_fn(mu_jc, prev_mu_jc, prev_dones, behavior_jc)
                    reg_kp_loss, valid_kp = smoothness_fn(mu_kp, prev_mu_kp, prev_dones, behavior_jc)
                    reg_valid = valid_jc and valid_kp
                reg_jc_contribution_value = 0.0
                reg_kp_contribution_value = 0.0
                if self.use_adaptive_regularization:
                    if reg_valid:
                        s = self.log_var_reg.clamp(
                            self.regularization_log_var_min, self.regularization_log_var_max
                        )
                        eff_weight = torch.exp(-s)
                        # One Kendall scalar applied to the SUM of the two reg terms — keeps the
                        # adaptive equilibrium semantics intact (one effective weight, one s).
                        # When adaptive is on, the JC/KP split weights are ignored.
                        adaptive_term = eff_weight * (reg_jc_loss + reg_kp_loss) + self.regularization_alpha * s
                        total_loss = total_loss + adaptive_term
                        reg_jc_contribution_value = float((eff_weight * reg_jc_loss).item())
                        reg_kp_contribution_value = float((eff_weight * reg_kp_loss).item())
                else:
                    if self.weight_regularization_jc > 0:
                        total_loss = total_loss + self.weight_regularization_jc * reg_jc_loss
                        reg_jc_contribution_value = float((self.weight_regularization_jc * reg_jc_loss).item())
                    if self.weight_regularization_kp > 0:
                        total_loss = total_loss + self.weight_regularization_kp * reg_kp_loss
                        reg_kp_contribution_value = float((self.weight_regularization_kp * reg_kp_loss).item())
                mean_reg_jc += reg_jc_loss.item()
                mean_reg_kp += reg_kp_loss.item()
                mean_reg_jc_contribution += reg_jc_contribution_value
                mean_reg_kp_contribution += reg_kp_contribution_value

                # 3. Cross-modal alignment — ASYMMETRIC: detach μ_jc so KP chases the privileged
                # JC latent as a fixed gold target (JC is not pulled toward masked KP). Always
                # computed; weighted in only when w > 0.
                align_loss = _cross_modal_cosine_alignment(mu_jc.detach(), mu_kp)
                align_contribution_value = 0.0
                if self.weight_align > 0:
                    total_loss = total_loss + self.weight_align * align_loss
                    align_contribution_value = float((self.weight_align * align_loss).item())
                mean_align += float(align_loss.item())
                mean_align_contribution += align_contribution_value

                # v1 telemetry: per-sample 1−cos(μ_kp, sg μ_jc) stratified by KP visible
                # fraction (Job-A reachability), + heavy-mask μ_kp distance to the batch
                # JC-latent mean direction (Job-B: frozen-decoder-safe region holding?).
                with torch.no_grad():
                    vf = fwd.get("kp_visible_frac", None)
                    if vf is not None:
                        vf = vf.reshape(-1)
                        mk = mu_kp.detach().reshape(-1, mu_kp.shape[-1])
                        mj = mu_jc.detach().reshape(-1, mu_jc.shape[-1])
                        per = 1.0 - nn.functional.cosine_similarity(mk, mj, dim=-1, eps=1e-6)
                        masks = (
                            vf < (1.0 / 3.0),
                            (vf >= (1.0 / 3.0)) & (vf < (2.0 / 3.0)),
                            vf >= (2.0 / 3.0),
                        )
                        for bi, m in enumerate(masks):
                            c = int(m.sum().item())
                            if c > 0:
                                _bin_sum[bi] += float(per[m].sum().item())
                                _bin_cnt[bi] += c
                        heavy = masks[0]
                        if int(heavy.sum().item()) > 0:
                            jc_mean_dir = nn.functional.normalize(
                                mj.mean(dim=0, keepdim=True), p=2.0, dim=-1, eps=1e-8
                            )
                            d = 1.0 - nn.functional.cosine_similarity(
                                mk[heavy], jc_mean_dir, dim=-1, eps=1e-6
                            )
                            _heavy_jcmean_sum += float(d.sum().item())
                            _heavy_jcmean_cnt += int(heavy.sum().item())

                if total_loss_accumulator is None:
                    total_loss_accumulator = total_loss
                else:
                    total_loss_accumulator = total_loss_accumulator + total_loss
                cnt += 1

                # Gradient step.
                just_optimized = False
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    total_loss_accumulator.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    total_loss_accumulator = None
                    just_optimized = True

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))
                if just_optimized:
                    prev_mu_jc = None
                    prev_mu_kp = None
                    prev_dones = None
                else:
                    prev_mu_jc = mu_jc.detach()
                    prev_mu_kp = mu_kp.detach()
                    prev_dones = dones.detach()

        if total_loss_accumulator is not None:
            self.optimizer.zero_grad()
            total_loss_accumulator.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.policy.detach_hidden_states()

        if cnt > 0:
            mean_behavior_jc /= cnt
            mean_behavior_kp /= cnt
            mean_reg_jc /= cnt
            mean_reg_kp /= cnt
            mean_reg_jc_contribution /= cnt
            mean_reg_kp_contribution /= cnt
            mean_align /= cnt
            mean_align_contribution /= cnt

        if mu_stats_count > 0:
            mu_norm_jc_mean = mu_norm_jc_sum / mu_stats_count
            mu_norm_kp_mean = mu_norm_kp_sum / mu_stats_count
        else:
            mu_norm_jc_mean = 0.0
            mu_norm_kp_mean = 0.0

        if self.use_adaptive_regularization:
            log_var_value = float(self.log_var_reg.item())
            reg_effective_weight = float(
                torch.exp(
                    -self.log_var_reg.clamp(
                        self.regularization_log_var_min, self.regularization_log_var_max
                    )
                ).item()
            )
            reg_effective_weight_jc = reg_effective_weight
            reg_effective_weight_kp = reg_effective_weight
        else:
            log_var_value = 0.0
            reg_effective_weight = float(self.weight_regularization_jc)  # legacy key (JC by convention)
            reg_effective_weight_jc = float(self.weight_regularization_jc)
            reg_effective_weight_kp = float(self.weight_regularization_kp)

        # Cross-modal contribution diagnostics. ``align_to_behavior_ratio`` answers "how loud is
        # the alignment term compared to one BC head" — useful for tuning λ_align.
        mean_behavior_total = mean_behavior_jc + mean_behavior_kp
        align_to_behavior_ratio = (
            mean_align_contribution / mean_behavior_total
            if mean_behavior_total > 1e-12 else 0.0
        )

        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        return {
            # Behavior (kept under "behavior" key for runner-side rolling-mean compatibility).
            "behavior": (mean_behavior_jc + mean_behavior_kp) / 2.0,
            "behavior_jc": mean_behavior_jc,
            "behavior_kp": mean_behavior_kp,
            # Smoothness.
            "regularization": (mean_reg_jc + mean_reg_kp) / 2.0,
            "reg_jc": mean_reg_jc,
            "reg_kp": mean_reg_kp,
            "reg_contribution": mean_reg_jc_contribution + mean_reg_kp_contribution,
            "reg_jc_contribution": mean_reg_jc_contribution,
            "reg_kp_contribution": mean_reg_kp_contribution,
            "reg_log_var": log_var_value,
            "reg_effective_weight": reg_effective_weight,
            "reg_effective_weight_jc": reg_effective_weight_jc,
            "reg_effective_weight_kp": reg_effective_weight_kp,
            # Cross-modal alignment.
            "align": mean_align,
            "align_contribution": mean_align_contribution,
            "align_weight": self.weight_align,
            "align_to_behavior_ratio": align_to_behavior_ratio,
            # μ scale.
            "mu_norm_jc_mean": mu_norm_jc_mean,
            "mu_norm_kp_mean": mu_norm_kp_mean,
            # v1 mask-stratified latent telemetry. align_kp_jc_* = mean 1−cos(μ_kp, sg μ_jc)
            # per KP-visible-fraction bin (Job-A: heavy lagging light is expected/benign alone;
            # actionable only if it co-moves with heavy-mask wrist degradation in eval).
            # kp_heavy_to_jc_mean_dist = heavy-mask μ_kp distance to the batch JC-latent mean
            # direction (Job-B decoder-freeze safety: must stay bounded; divergence ⇒ stop-and-fix).
            # Wrist ee_body_pos stratified by mask level is an eval/play concern (indomain video
            # suite), NOT a training-loop metric.
            "align_kp_jc_heavy": (_bin_sum[0] / _bin_cnt[0]) if _bin_cnt[0] > 0 else 0.0,
            "align_kp_jc_mid": (_bin_sum[1] / _bin_cnt[1]) if _bin_cnt[1] > 0 else 0.0,
            "align_kp_jc_light": (_bin_sum[2] / _bin_cnt[2]) if _bin_cnt[2] > 0 else 0.0,
            "align_bin_frac_heavy": _bin_cnt[0] / max(_bin_cnt[0] + _bin_cnt[1] + _bin_cnt[2], 1),
            "align_bin_frac_mid": _bin_cnt[1] / max(_bin_cnt[0] + _bin_cnt[1] + _bin_cnt[2], 1),
            "align_bin_frac_light": _bin_cnt[2] / max(_bin_cnt[0] + _bin_cnt[1] + _bin_cnt[2], 1),
            "kp_heavy_to_jc_mean_dist": (
                _heavy_jcmean_sum / _heavy_jcmean_cnt if _heavy_jcmean_cnt > 0 else 0.0
            ),
        }
