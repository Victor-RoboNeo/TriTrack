"""MUSE-Kp latent-space distillation against a frozen JC MUSE-Transformer teacher.

Unlike :class:`MuseKpDistillation` (action-space BC vs the frozen MLP teacher, encoder + decoder
co-tuned), this distils the KP-token student in **latent space** against a near-perfect
joint-command (JC) MUSE-Transformer:

  - Teacher (frozen): JC MUSE-T encoder → μ_jc, fed the full unmasked JC obs (privileged).
  - Student (trainable): KP encoder → μ_kp, fed masked/partial KP obs.
  - Decoder (frozen, shared, loaded from the JC ckpt): action = decoder([μ, proprio]).

Because the decoder is the *same frozen weights* for both, matching μ_kp → μ_jc makes the
student's action match the teacher's at convergence. The JC teacher always sees full info, so
the KP student learns to infer the right latent from partial keypoints.

Loss (per env per step):
    L = weight_latent   · (1 - cos(z_kp, sg(z_jc)))                  # latent alignment (cosine)
      + weight_behavior · mse(decoder(z_kp), sg(action_jc))          # action anchor (frozen dec)

The latent is matched on **unit-norm z** (2026-05-18: the JC MUSE-T teacher runs the unit-norm
cosine recipe ``latent_normalize=True``, so the frozen decoder consumes z = L2-normalize(μ);
the policy's ``encode_latent`` / ``get_teacher_targets`` return the normalized z so the loss
lives on the unit sphere — the same geometry the decoder sees. MSE on unit vectors ≡ 2−2·cos).

Training schedule:
  - Teacher-pilot warmup: the JC teacher drives the env for the first
    ``teacher_pilot_warmup_iters`` iterations (good on-distribution states while the fresh
    KP front-end catches up), then a hard switch to student-pilot. The supervision target is
    the JC teacher throughout (the loss recomputes the student from stored obs).
  - Freeze warmup: hold the policy in ``warmup_freeze_mode`` (default
    ``decoder_plus_shared_encoder`` — only kp_proj/body_id_emb train) for
    ``warmup_freeze_iters`` iters, then ``post_warmup_freeze_mode`` (default ``decoder_only``
    — decoder stays frozen forever, the full KP encoder trains) + rebuild the optimizer.

Structurally mirrors :class:`AnyBodyLatentDistillation` (no PULSE prior / residual here; the
encoder is warmstarted from the JC backbone instead).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.storage import RolloutStorage


class MuseKpLatentDistillation:
    """Latent-space KP distillation vs a frozen JC MUSE-Transformer, frozen shared decoder."""

    def __init__(
        self,
        policy,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        loss_type: str = "mse",
        latent_loss_type: str = "cosine",
        weight_latent: float = 1.0,
        weight_behavior: float = 0.05,
        teacher_pilot_warmup_iters: int = 0,
        warmup_freeze_iters: int = 0,
        warmup_freeze_mode: str = "decoder_plus_shared_encoder",
        post_warmup_freeze_mode: str = "decoder_only",
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "MuseKpLatentDistillation got unexpected arguments (ignored): "
                + str(list(kwargs.keys()))
            )
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None
        self.policy = policy
        self.policy.to(self.device)
        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None

        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.weight_latent = weight_latent
        self.weight_behavior = weight_behavior
        self.teacher_pilot_warmup_iters = int(teacher_pilot_warmup_iters)

        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported: mse, huber")

        # Latent term metric. Under unit-norm latents, MSE = (2-2cos)/latent_dim saturates
        # near alignment (vanishing gradient exactly where a frozen decoder needs fidelity),
        # so "cosine" (1 - cos(z_kp, z_jc)) is the default — the unit-sphere-natural metric,
        # the same one the JC teacher trains with, scale ~[0,2] with a non-vanishing gradient.
        self.latent_loss_type = str(latent_loss_type)
        if self.latent_loss_type not in ("cosine", "mse", "huber"):
            raise ValueError(
                f"Unknown latent_loss_type: {self.latent_loss_type}. Supported: cosine, mse, huber"
            )

        # --- Freeze-mode warmup curriculum --------------------------------------------------------
        # Pin the policy into ``warmup_freeze_mode`` BEFORE building the optimizer so the optimizer
        # locks the right (smaller) param set; transition to ``post_warmup_freeze_mode`` at iter
        # ``warmup_freeze_iters`` and rebuild the optimizer over the new param set.
        self.warmup_freeze_iters = int(warmup_freeze_iters)
        self.warmup_freeze_mode = str(warmup_freeze_mode)
        self.post_warmup_freeze_mode = str(post_warmup_freeze_mode)
        if self.warmup_freeze_iters > 0 and hasattr(policy, "set_freeze_mode"):
            if getattr(policy, "freeze_mode", None) != self.warmup_freeze_mode:
                policy.set_freeze_mode(self.warmup_freeze_mode)
                print(
                    f"[MuseKpLatentDistillation] Freeze warmup active for "
                    f"{self.warmup_freeze_iters} iters: freeze_mode -> {self.warmup_freeze_mode!r} "
                    f"(post -> {self.post_warmup_freeze_mode!r})"
                )
        self._warmup_done = self.warmup_freeze_iters <= 0

        self.num_updates = 0
        self._build_optimizer()

        if self.teacher_pilot_warmup_iters > 0:
            print(
                f"[MuseKpLatentDistillation] Teacher-pilot warmup: JC teacher drives the env for "
                f"the first {self.teacher_pilot_warmup_iters} iters, then student pilots."
            )

    def _build_optimizer(self) -> None:
        # Only ``policy.student`` params train; the JC teacher encoder + shared decoder are frozen
        # by the policy (requires_grad False) and never enter this group.
        self.optimizer = optim.Adam(self.policy.student.parameters(), lr=self.learning_rate)

    def _rebuild_optimizer(self) -> None:
        """Rebuild the optimizer over the post-warmup param set (fresh Adam moments)."""
        self.policy.set_freeze_mode(self.post_warmup_freeze_mode)
        self.optimizer = optim.Adam(self.policy.student.parameters(), lr=self.learning_rate)
        print(
            f"[MuseKpLatentDistillation] Freeze warmup complete: freeze_mode -> "
            f"{self.post_warmup_freeze_mode!r}; optimizer rebuilt."
        )

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        student_obs_shape: list,
        teacher_obs_shape: list,
        actions_shape: list,
        latent_dim: int | None = None,
    ) -> None:
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
            teacher_latent_shape=[latent_dim] if latent_dim is not None else None,
        )

    def act(
        self,
        student_obs: torch.Tensor,
        teacher_obs: torch.Tensor,
    ) -> torch.Tensor:
        teacher_mu, teacher_actions = self.policy.get_teacher_targets(teacher_obs)
        self.transition.privileged_actions = teacher_actions
        self.transition.teacher_latent = teacher_mu
        student_actions = self.policy.act(student_obs).detach()
        # Teacher-pilot warmup: drive the env with the JC teacher early so the visited state
        # distribution is on-distribution while the fresh KP front-end catches up. The supervision
        # target (privileged_actions / teacher_latent) is the JC teacher regardless of who drives.
        if self.num_updates < self.teacher_pilot_warmup_iters:
            executed = teacher_actions.detach()
        else:
            executed = student_actions
        self.transition.actions = executed
        self.transition.observations = student_obs
        self.transition.privileged_observations = teacher_obs
        return self.transition.actions

    def process_env_step(
        self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict
    ) -> None:
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self, current_iter: int | None = None) -> dict:
        self.num_updates += 1

        # Freeze-warmup transition (before this iteration's optimization).
        if (
            not self._warmup_done
            and current_iter is not None
            and current_iter >= self.warmup_freeze_iters
        ):
            self._rebuild_optimizer()
            self._warmup_done = True

        mean_behavior_loss = 0.0
        mean_latent_loss = 0.0
        cnt_behavior = 0
        cnt_latent = 0
        step_counter = 0
        total_loss_accum: torch.Tensor | None = None

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for batch in self.storage.generator():
                obs = batch[0]
                privileged_actions = batch[3]
                dones = batch[4]
                teacher_latent = batch[5] if len(batch) > 5 else None

                loss = torch.zeros((), device=self.device)

                # Latent alignment on unit-norm z (JC recipe is latent_normalize=True → the
                # frozen decoder consumes z=norm(μ); encode_latent/get_teacher_targets both
                # return the normalized z, so this matches on the unit sphere = decoder geometry).
                if teacher_latent is not None and self.weight_latent > 0:
                    z_kp, _, _ = self.policy.encode_latent(obs)
                    if self.latent_loss_type == "cosine":
                        # 1 - cos on the unit sphere (z_kp, z_jc already L2-normed). Scale
                        # ~[0,2], non-vanishing gradient near alignment (unlike MSE/dim).
                        latent_loss = (
                            1.0
                            - nn.functional.cosine_similarity(
                                z_kp, teacher_latent, dim=-1, eps=1e-8
                            )
                        ).mean()
                    else:
                        latent_loss = self.loss_fn(z_kp, teacher_latent)
                    loss = loss + self.weight_latent * latent_loss
                    mean_latent_loss += latent_loss.item()
                    cnt_latent += 1

                # Behavior anchor through the frozen shared decoder — keeps the latent from
                # drifting into action-irrelevant directions the frozen decoder ignores.
                if self.weight_behavior > 0.0:
                    actions = self.policy.act_inference(obs)
                    behavior_loss = self.loss_fn(actions, privileged_actions)
                    loss = loss + self.weight_behavior * behavior_loss
                    mean_behavior_loss += behavior_loss.item()
                    cnt_behavior += 1

                if total_loss_accum is None:
                    total_loss_accum = loss
                else:
                    total_loss_accum = total_loss_accum + loss

                step_counter += 1
                if step_counter % self.gradient_length == 0 and total_loss_accum is not None:
                    self.optimizer.zero_grad()
                    total_loss_accum.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    total_loss_accum = None

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        # Backward any leftover accumulated gradients.
        if total_loss_accum is not None:
            self.optimizer.zero_grad()
            total_loss_accum.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.policy.detach_hidden_states()

        if cnt_behavior > 0:
            mean_behavior_loss /= cnt_behavior
        if cnt_latent > 0:
            mean_latent_loss /= cnt_latent
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        return {"behavior": mean_behavior_loss, "latent": mean_latent_loss}

    def broadcast_parameters(self) -> None:
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        grads = [
            param.grad.view(-1)
            for param in self.policy.parameters()
            if param.grad is not None
        ]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(
                    all_grads[offset : offset + numel].view_as(param.grad.data)
                )
                offset += numel
