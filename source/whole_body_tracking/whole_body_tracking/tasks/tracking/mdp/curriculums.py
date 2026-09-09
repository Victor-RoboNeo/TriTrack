"""Curriculum terms for tracking MDPs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def attach_curriculum_rollout_hints(env: object, *, spe: int, learning_iteration: int = 0) -> None:
    """Set env fields needed by ``phase_until_learning_iterations`` before the first ``reset()`` / curriculum compute.

    :class:`RslRlVecEnvWrapper` resets the env in ``__init__`` before :meth:`OnPolicyRunner.learn` runs, so the
    runner cannot be the only place that sets these attributes. Training calls this from ``train.py`` using
    ``agent_cfg.num_steps_per_env``.
    """
    e: object | None = env
    seen: set[int] = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if hasattr(e, "common_step_counter") and hasattr(e, "command_manager"):
            setattr(e, "curriculum_env_steps_per_learning_iteration", int(spe))
            setattr(e, "curriculum_counter_anchor", int(getattr(e, "common_step_counter", 0)))
            setattr(e, "curriculum_learning_iteration_anchor", int(learning_iteration))
            return
        unwrapped = getattr(e, "unwrapped", None)
        e = unwrapped if unwrapped is not None and unwrapped is not e else None


def _virtual_env_step_for_curriculum(env: "ManagerBasedRLEnv") -> int:
    """Virtual step index aligned with global PPO iteration (see :meth:`OnPolicyRunner.learn`).

    On ``learn()`` entry the runner sets on the unwrapped env:

    - ``curriculum_env_steps_per_learning_iteration`` — ``num_steps_per_env``;
    - ``curriculum_counter_anchor`` — :attr:`common_step_counter` at entry;
    - ``curriculum_learning_iteration_anchor`` — runner's ``current_learning_iteration`` at entry.

    ``virtual_step = anchor_iter * spe + (common_step_counter - anchor_counter)`` increases by ``spe`` per
    learning iteration and stays consistent across resume and chunked ``learn()`` calls.
    """
    spe = int(getattr(env, "curriculum_env_steps_per_learning_iteration", 0))
    anchor_c = int(getattr(env, "curriculum_counter_anchor", 0))
    anchor_i = int(getattr(env, "curriculum_learning_iteration_anchor", 0))
    counter = int(getattr(env, "common_step_counter", 0))
    return anchor_i * spe + (counter - anchor_c)


def keypoint_mask_mode_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    mode_probs_phases: tuple[tuple[float, ...], ...] | None = None,
    phase_until_steps: tuple[int | None, ...] | None = None,
    phase_until_learning_iterations: tuple[int | None, ...] | None = None,
    mask_phases: tuple[dict[str, Any], ...] | None = None,
    bernoulli_p_start_per_phase: tuple[float, ...] | None = None,
    bernoulli_p_end_per_phase: tuple[float, ...] | None = None,
) -> dict[str, float] | None:
    """Update partial keypoint mask sampling probabilities by phase.

    Isaac Lab calls curriculum terms from :meth:`CurriculumManager.compute`, which runs inside
    :meth:`ManagerBasedRLEnv._reset_idx` when envs reset. Phase uses
    :func:`_virtual_env_step_for_curriculum` when iteration-based bounds are configured (see runner).

    Args:
        env: Manager-based RL environment.
        env_ids: Unused (API compatibility).
        command_name: Command term name (typically ``\"motion\"``).
        mode_probs_phases: Mode probability tuple per phase (same length as ``mask_mode_spec``).
        phase_until_steps: Upper bounds on :attr:`common_step_counter` (exclusive); ``None`` = no limit.
            For play or custom schedules without the runner anchors. Mutually exclusive with
            ``phase_until_learning_iterations``.
        phase_until_learning_iterations: Upper bounds in **global learning-iteration** space (same as the
            runner's ``it``). Internally converted to the same virtual step axis as
            :func:`_virtual_env_step_for_curriculum` (``bound * num_steps_per_env`` vs. virtual step).
            Requires ``curriculum_env_steps_per_learning_iteration`` on the env before the first curriculum
            compute (``OnPolicyRunner.learn`` and/or :func:`attach_curriculum_rollout_hints`).
        mask_phases: Optional per-phase dicts. Each dict should contain:
            - ``mode_probs``: probability tuple for each mask mode
            - ``p_start``: Bernoulli keep-probability at the phase start
            - ``p_end``: Bernoulli keep-probability at the phase end
            When provided, it overrides ``mode_probs_phases`` and the Bernoulli `*_per_phase` args.
        bernoulli_p_start_per_phase: Optional keep-probability start values (per phase) for the special
            stochastic `bernoulli` mask mode. Must have the same length as ``mode_probs_phases``.
            If provided together with ``bernoulli_p_end_per_phase``, the curriculum updates `p_keep` only on
            phases where the motion term actually samples Bernoulli (phase-gated by mode probability > 0).
        bernoulli_p_end_per_phase: Optional keep-probability end values (per phase) for the special
            stochastic `bernoulli` mask mode. Must have the same length as ``mode_probs_phases``.
    """
    del env_ids
    if (phase_until_steps is None) == (phase_until_learning_iterations is None):
        raise ValueError(
            "Specify exactly one of phase_until_steps or phase_until_learning_iterations "
            f"(got steps={phase_until_steps!r}, iters={phase_until_learning_iterations!r})."
        )
    motion = env.command_manager.get_term(command_name)
    if not getattr(motion, "uses_command_manager_mask_sampling", False):
        return None

    # Resolve effective per-phase configuration.
    mode_probs_phases_eff: tuple[tuple[float, ...], ...] | None = mode_probs_phases
    bernoulli_p_start_eff: tuple[float, ...] | None = bernoulli_p_start_per_phase
    bernoulli_p_end_eff: tuple[float, ...] | None = bernoulli_p_end_per_phase

    if mask_phases is not None:
        if mode_probs_phases_eff is not None:
            # Keep behavior deterministic: dict phases override explicit tuples.
            pass
        mode_probs_phases_eff = tuple(tuple(phase["mode_probs"]) for phase in mask_phases)
        bernoulli_p_start_eff = tuple(float(phase["p_start"]) for phase in mask_phases)
        bernoulli_p_end_eff = tuple(float(phase["p_end"]) for phase in mask_phases)

    if mode_probs_phases_eff is None:
        raise ValueError(
            "keypoint_mask_mode_curriculum requires either `mask_phases` or `mode_probs_phases`."
        )

    if phase_until_learning_iterations is not None:
        spe = int(getattr(env, "curriculum_env_steps_per_learning_iteration", 0))
        if spe <= 0:
            raise ValueError(
                "phase_until_learning_iterations requires the runner to set "
                "env.curriculum_env_steps_per_learning_iteration = num_steps_per_env "
                "(and curriculum anchors) before rollouts — see OnPolicyRunner.learn() or "
                "attach_curriculum_rollout_hints() from train.py."
            )
        bounds: tuple[int | None, ...] = tuple(
            None if b is None else int(b) * spe for b in phase_until_learning_iterations
        )
        step = _virtual_env_step_for_curriculum(env)
    else:
        assert phase_until_steps is not None
        bounds = phase_until_steps
        step = int(getattr(env, "common_step_counter", 0))

    if len(bounds) != len(mode_probs_phases_eff):
        raise ValueError(
            "phase span length mismatch: len(phase_until_*) must equal number of phases "
            f"(got len(bounds)={len(bounds)} vs len(mode_probs_phases)={len(mode_probs_phases_eff)})."
        )
    # Determine active phase index (same bounds logic used for mode sampling).
    active = 0
    prev_bound = 0
    for i, bound in enumerate(bounds):
        active = i
        if bound is None:
            break
        if step < int(bound):
            break
        prev_bound = int(bound)

    if active >= len(mode_probs_phases_eff):
        active = len(mode_probs_phases_eff) - 1

    probs = mode_probs_phases_eff[active]

    # Optional: update Bernoulli keep-probability within the active phase span.
    p_keep: float | None = None
    if (
        bernoulli_p_start_eff is not None
        and bernoulli_p_end_eff is not None
        and hasattr(motion, "set_bernoulli_keep_prob")
    ):
        if len(bernoulli_p_start_eff) != len(mode_probs_phases_eff):
            raise ValueError(
                "bernoulli_p_start_per_phase length must match mode_probs_phases length "
                f"(got {len(bernoulli_p_start_eff)} vs {len(mode_probs_phases_eff)})."
            )
        if len(bernoulli_p_end_eff) != len(mode_probs_phases_eff):
            raise ValueError(
                "bernoulli_p_end_per_phase length must match mode_probs_phases length "
                f"(got {len(bernoulli_p_end_eff)} vs {len(mode_probs_phases_eff)})."
            )

        # Find Bernoulli index in the motion term's mode order for phase-gating.
        bernoulli_idx: int | None = None
        mode_names = getattr(motion, "_mode_names", None)
        if mode_names is not None:
            for mi, mn in enumerate(mode_names):
                if str(mn).lower() in ("bernoulli", "bernouli"):
                    bernoulli_idx = mi
                    break

        # Gate update to phases where Bernoulli mode is sampled (mode prob > 0).
        if bernoulli_idx is not None and bernoulli_idx < len(probs) and probs[bernoulli_idx] > 0.0:
            p0 = float(bernoulli_p_start_eff[active])
            p1 = float(bernoulli_p_end_eff[active])

            bound_end = bounds[active] if active < len(bounds) else None
            # For the last open-ended phase (bound=None), use t=0 (hold at p_start).
            if bound_end is None:
                t = 0.0
            else:
                denom = float(bound_end - prev_bound)
                if denom <= 0.0:
                    t = 0.0
                else:
                    t = float(step - prev_bound) / denom
                    t = max(0.0, min(1.0, t))

            p_keep = p0 + t * (p1 - p0)
            p_keep = max(0.0, min(1.0, p_keep))

    # IMPORTANT: apply Bernoulli p_keep BEFORE resampling mask modes.
    # set_mask_mode_probs_tuple triggers an immediate resample, so if we set p_keep after,
    # Bernoulli env masks can be generated using the previous p_keep value.
    if p_keep is not None:
        motion.set_bernoulli_keep_prob(p_keep)

    # Now update semantic mode sampling probabilities (this resamples env mask modes immediately).
    motion.set_mask_mode_probs_tuple(tuple(probs))

    if p_keep is None:
        return {"mask_curriculum_phase": float(active)}
    return {"mask_curriculum_phase": float(active), "bernoulli_keep_prob": float(p_keep)}


def goal_mask_probability_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_name: str,
    p_start: float = 0.0,
    p_end: float = 0.5,
    ramp_start_iter: int = 0,
    ramp_end_iter: int = 5000,
    schedule: str = "linear",
) -> dict[str, float]:
    """Schedule the MUSE goal-block mask probability ``p_mask`` on the motion command.

    Reads the current global learning iteration via :func:`_virtual_env_step_for_curriculum` and
    interpolates between ``p_start`` (at ``ramp_start_iter`` and earlier) and ``p_end`` (at
    ``ramp_end_iter`` and later). ``schedule`` is ``"linear"`` (default) or ``"cosine"``.
    """
    del env_ids  # unused; we set a global attribute on the command term.
    spe = int(getattr(env, "curriculum_env_steps_per_learning_iteration", 0))
    if spe <= 0:
        # Curriculum not yet anchored (env reset before runner.learn()): leave p_mask at p_start.
        current_iter = 0
    else:
        current_iter = _virtual_env_step_for_curriculum(env) // spe

    if ramp_end_iter <= ramp_start_iter:
        progress = 1.0 if current_iter >= ramp_end_iter else 0.0
    elif current_iter <= ramp_start_iter:
        progress = 0.0
    elif current_iter >= ramp_end_iter:
        progress = 1.0
    else:
        progress = (current_iter - ramp_start_iter) / (ramp_end_iter - ramp_start_iter)

    if schedule == "cosine":
        import math
        progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    elif schedule != "linear":
        raise ValueError(f"goal_mask_probability_curriculum: unknown schedule {schedule!r}.")

    p_mask = float(p_start) + progress * (float(p_end) - float(p_start))
    p_mask = max(0.0, min(1.0, p_mask))
    motion = env.command_manager.get_term(command_name)
    motion.p_mask = p_mask
    return {"goal_mask_p": p_mask, "goal_mask_iter": float(current_iter)}


def motion_group_ratio_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_name: str = "motion",
    phase_until_learning_iterations: tuple[int | None, ...] = (3000, 8000, None),
    ratio_phases: tuple[dict[str, float], ...] = (
        {"loco": 0.70, "stoop": 0.10, "default": 0.20},
        {"loco": 0.45, "stoop": 0.20, "default": 0.35},
        {"loco": 0.40, "stoop": 0.25, "default": 0.35},
    ),
) -> dict[str, float]:
    """Schedule ``motion_group_sampling_ratios`` by PPO iteration.

    Phase 0 emphasizes locomotion so the frozen decoder's walk prior is reused; later
    phases mix in stoop (height change) and high-dynamic default clips (kick/throw/…).
    Ratios are read on the next motion resample.
    """
    del env_ids
    if len(phase_until_learning_iterations) != len(ratio_phases):
        raise ValueError(
            "motion_group_ratio_curriculum: phase_until_learning_iterations and "
            f"ratio_phases must have the same length (got {len(phase_until_learning_iterations)} vs "
            f"{len(ratio_phases)})."
        )
    spe = int(getattr(env, "curriculum_env_steps_per_learning_iteration", 0))
    if spe <= 0:
        current_iter = 0
    else:
        current_iter = _virtual_env_step_for_curriculum(env) // spe

    active = 0
    for i, bound in enumerate(phase_until_learning_iterations):
        active = i
        if bound is None or current_iter < int(bound):
            break

    ratios = {str(k): float(v) for k, v in ratio_phases[active].items()}
    total = sum(ratios.values())
    if total <= 0.0:
        raise ValueError(f"motion_group_ratio_curriculum: phase {active} ratios sum to {total}.")
    if abs(total - 1.0) > 1e-5:
        ratios = {k: v / total for k, v in ratios.items()}

    motion = env.command_manager.get_term(command_name)
    motion.cfg.motion_group_sampling_ratios = ratios
    out = {f"motion_group_{k}": float(v) for k, v in ratios.items()}
    out["motion_group_phase"] = float(active)
    out["motion_group_iter"] = float(current_iter)
    return out


def terrain_levels_tracking(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    move_up_frac: float = 0.60,
    move_down_frac: float = 0.25,
):
    """Promote/demote generator terrain level from episode length (tracking analogue of velocity terrain curriculum).

    Long episodes (clip nearly finished, few early terms) move up; very short episodes move down.
    No height scan is added to the actor.
    """
    terrain = env.scene.terrain
    if not hasattr(terrain, "update_env_origins") or getattr(terrain, "terrain_origins", None) is None:
        return None
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if ids.numel() == 0:
        return torch.mean(terrain.terrain_levels.float())
    length = env.episode_length_buf[ids].float()
    max_len = float(env.max_episode_length)
    move_up = length > float(move_up_frac) * max_len
    move_down = (length < float(move_down_frac) * max_len) & ~move_up
    terrain.update_env_origins(ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
