"""AnyBody latent distillation with residual latent supervision.

Student predicts residual latent over frozen PULSE prior mean:
mu_student = mu_prior + delta_mu(obs).
Prior and decoder are frozen; optimizer only updates residual encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.storage import RolloutStorage


class AnyBodyLatentDistillation:
    """Residual-latent distillation with optional behavior regularization."""

    def __init__(
        self,
        policy,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        loss_type: str = "mse",
        weight_latent: float = 0.1,
        weight_behavior: float = 0.0,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
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

        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported: mse, huber")

        self.num_updates = 0

        # Optimizer: stage 1 only encoder; stage 2+ full student
        self._build_optimizer()

    def _build_optimizer(self) -> None:
        params = self.policy.student.parameters()
        self.optimizer = optim.Adam(params, lr=self.learning_rate)

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
        self.transition.actions = self.policy.act(student_obs).detach()
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

    def update(self, current_iter: int) -> dict:
        self.num_updates += 1

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

                # Optional behavior (action) loss as a weak regularizer.
                if self.weight_behavior > 0.0:
                    actions = self.policy.act_inference(obs)
                    behavior_loss = self.loss_fn(actions, privileged_actions)
                    mean_behavior_loss += behavior_loss.item()
                    cnt_behavior += 1
                    loss = self.weight_behavior * behavior_loss
                else:
                    loss = torch.zeros((), device=self.device)

                if teacher_latent is not None and self.weight_latent > 0:
                    mu, _, _ = self.policy.encode_residual_latent(obs)
                    latent_loss = self.loss_fn(mu, teacher_latent)
                    loss = loss + self.weight_latent * latent_loss
                    mean_latent_loss += latent_loss.item()
                    cnt_latent += 1

                # Accumulate loss for gradient step.
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

        # Backward any leftover accumulated gradients (when steps % gradient_length != 0).
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
