from __future__ import annotations

import os
from typing import cast
import torch


def _build_full_obs_norm_state_from_split(
    goal_state: dict | None, proprio_state: dict | None
) -> dict | None:
    """Compose a full obs normalizer state from split goal/proprio states."""
    if not isinstance(goal_state, dict) or not isinstance(proprio_state, dict):
        return None

    goal_mean, goal_var, goal_std = (
        goal_state.get("_mean"),
        goal_state.get("_var"),
        goal_state.get("_std"),
    )
    proprio_mean, proprio_var, proprio_std = (
        proprio_state.get("_mean"),
        proprio_state.get("_var"),
        proprio_state.get("_std"),
    )
    if not all(
        isinstance(t, torch.Tensor)
        for t in (goal_mean, goal_var, goal_std, proprio_mean, proprio_var, proprio_std)
    ):
        return None
    goal_mean_t = cast(torch.Tensor, goal_mean)
    goal_var_t = cast(torch.Tensor, goal_var)
    goal_std_t = cast(torch.Tensor, goal_std)
    proprio_mean_t = cast(torch.Tensor, proprio_mean)
    proprio_var_t = cast(torch.Tensor, proprio_var)
    proprio_std_t = cast(torch.Tensor, proprio_std)

    full_count = goal_state.get("count")
    if not isinstance(full_count, torch.Tensor):
        full_count = proprio_state.get("count")
    if not isinstance(full_count, torch.Tensor):
        # EmpiricalNormalization expects a scalar count; default to 0 if absent.
        full_count = torch.tensor(0, dtype=goal_mean_t.dtype, device=goal_mean_t.device)

    return {
        "_mean": torch.cat([goal_mean_t, proprio_mean_t], dim=-1),
        "_var": torch.cat([goal_var_t, proprio_var_t], dim=-1),
        "_std": torch.cat([goal_std_t, proprio_std_t], dim=-1),
        "count": full_count,
    }


def resolve_obs_norm_checkpoint_state(
    cfg: dict, device: str, empirical_normalization: bool
) -> tuple[str | None, dict | None]:
    """Load observation normalizer state from optional checkpoint path in cfg."""
    obs_norm_ckpt_path = cfg.get("obs_normalizer_checkpoint_path")
    if not empirical_normalization or obs_norm_ckpt_path in (None, ""):
        return None, None
    if not os.path.isabs(obs_norm_ckpt_path):
        obs_norm_ckpt_path = os.path.abspath(obs_norm_ckpt_path)
    if not os.path.isfile(obs_norm_ckpt_path):
        raise FileNotFoundError(f"obs_normalizer_checkpoint_path not found: {obs_norm_ckpt_path}")
    obs_norm_ckpt = torch.load(obs_norm_ckpt_path, map_location=device, weights_only=False)
    obs_norm_state = obs_norm_ckpt.get("obs_norm_state_dict")
    if not isinstance(obs_norm_state, dict):
        obs_norm_state = _build_full_obs_norm_state_from_split(
            obs_norm_ckpt.get("student_goal_obs_norm_state_dict"),
            obs_norm_ckpt.get("student_proprio_obs_norm_state_dict"),
        )
    if not isinstance(obs_norm_state, dict):
        raise KeyError(
            f"Checkpoint {obs_norm_ckpt_path} does not contain 'obs_norm_state_dict' "
            "or reconstructable split normalizer states."
        )
    return obs_norm_ckpt_path, obs_norm_state


def try_build_rl_split_normalizers_from_checkpoint(
    *,
    training_type: str,
    empirical_normalization: bool,
    obs_norm_state: dict | None,
    cfg: dict,
    env,
    num_obs: int,
    device: str,
) -> tuple[torch.nn.Module | None, torch.nn.Module | None, int | None, int | None]:
    """Build command/proprio split normalizers for RL and load+freeze proprio tail from checkpoint."""
    enable_rl_split = bool(cfg.get("enable_rl_split_obs_normalizer", False))
    if (not enable_rl_split) or training_type != "rl" or (not empirical_normalization) or obs_norm_state is None:
        return None, None, None, None

    proprio_dim = cfg.get("obs_normalizer_proprio_dim", None)
    if not isinstance(proprio_dim, int):
        env_unwrap = env.unwrapped if hasattr(env, "unwrapped") else env
        proprio_dim = None
        try:
            if hasattr(env_unwrap, "action_manager"):
                action_term = env_unwrap.action_manager.get_term("joint_pos")
                proprio_dim = getattr(action_term, "_proprio_dim", None)
        except Exception:
            proprio_dim = None
    if not isinstance(proprio_dim, int) or not (0 < proprio_dim < num_obs):
        return None, None, None, None

    goal_dim = num_obs - proprio_dim
    # Local import avoids circular import during package initialization
    # (rsl_rl.modules -> actor_critic -> rsl_rl.utils).
    from rsl_rl.modules.normalizer import EmpiricalNormalization

    goal_norm = EmpiricalNormalization(shape=[goal_dim], until=1.0e8).to(device)
    proprio_norm = EmpiricalNormalization(shape=[proprio_dim], until=1.0e8).to(device)

    ckpt_mean = obs_norm_state.get("_mean")
    ckpt_var = obs_norm_state.get("_var")
    ckpt_std = obs_norm_state.get("_std")
    if not all(isinstance(t, torch.Tensor) for t in (ckpt_mean, ckpt_var, ckpt_std)):
        raise KeyError("obs_norm_state_dict misses _mean/_var/_std tensors.")
    if ckpt_mean.shape[-1] < proprio_dim:
        raise ValueError(
            f"Checkpoint obs normalizer dim {ckpt_mean.shape[-1]} is smaller than proprio_dim {proprio_dim}."
        )
    with torch.no_grad():
        proprio_norm._mean.copy_(ckpt_mean[..., -proprio_dim:])
        proprio_norm._var.copy_(ckpt_var[..., -proprio_dim:])
        proprio_norm._std.copy_(ckpt_std[..., -proprio_dim:])
        ckpt_count = obs_norm_state.get("count")
        if isinstance(ckpt_count, torch.Tensor):
            proprio_norm.count.copy_(ckpt_count)
    proprio_norm.eval()
    if hasattr(proprio_norm, "until"):
        proprio_norm.until = getattr(proprio_norm, "count", 0)

    return goal_norm, proprio_norm, goal_dim, proprio_dim


def load_and_freeze_full_obs_normalizer(obs_normalizer, obs_norm_state: dict | None) -> bool:
    """Fallback: load entire obs normalizer from checkpoint and freeze it.

    If the live obs normalizer is WIDER than the checkpoint's (extra obs dims appended at the
    TAIL — e.g. the obstacle-reach task appends an obstacle block LAST to the policy obs), the
    extra tail dims are padded with fresh stats (mean 0 / var 1 / std 1) so the strict load
    succeeds and the appended dims pass through ~raw (they feed the residual corrector /
    critic, not the frozen encoder). Equal-or-narrower checkpoints load unchanged.
    """
    if obs_norm_state is None or isinstance(obs_normalizer, torch.nn.Identity):
        return False
    mean = obs_norm_state.get("_mean") if isinstance(obs_norm_state, dict) else None
    if isinstance(mean, torch.Tensor) and hasattr(obs_normalizer, "_mean"):
        tgt = int(obs_normalizer._mean.shape[-1])
        src = int(mean.shape[-1])
        if tgt > src:
            pad = tgt - src
            obs_norm_state = dict(obs_norm_state)
            for key, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
                t = obs_norm_state.get(key)
                if isinstance(t, torch.Tensor):
                    pad_t = torch.full((*t.shape[:-1], pad), fill, dtype=t.dtype, device=t.device)
                    obs_norm_state[key] = torch.cat([t, pad_t], dim=-1)
    obs_normalizer.load_state_dict(obs_norm_state)
    obs_normalizer.eval()
    if hasattr(obs_normalizer, "until"):
        obs_normalizer.until = getattr(obs_normalizer, "count", 0)
    return True


def save_normalizer_states(
    *,
    saved_dict: dict,
    empirical_normalization: bool,
    use_split_normalizers: bool,
    obs_normalizer,
    privileged_obs_normalizer,
    student_goal_obs_normalizer,
    student_proprio_obs_normalizer,
    training_type: str,
    teacher_obs_normalizer,
) -> None:
    """Save normalizer states into checkpoint dict."""
    if not empirical_normalization:
        return
    if use_split_normalizers:
        saved_dict["student_goal_obs_norm_state_dict"] = student_goal_obs_normalizer.state_dict()
        saved_dict["student_proprio_obs_norm_state_dict"] = student_proprio_obs_normalizer.state_dict()
    else:
        saved_dict["obs_norm_state_dict"] = obs_normalizer.state_dict()
    saved_dict["privileged_obs_norm_state_dict"] = privileged_obs_normalizer.state_dict()
    if training_type == "mosaic" and not isinstance(teacher_obs_normalizer, torch.nn.Identity):
        saved_dict["teacher_obs_norm_state_dict"] = teacher_obs_normalizer.state_dict()


def load_normalizer_states_on_resume(
    *,
    loaded_dict: dict,
    load_critic: bool,
    resumed_training: bool,
    training_type: str,
    is_residual_policy: bool,
    use_split_normalizers: bool,
    obs_normalizer,
    privileged_obs_normalizer,
    teacher_obs_normalizer,
    student_goal_obs_normalizer,
    student_proprio_obs_normalizer,
    anybody_latent_proprio_dim: int | None,
    alg,
) -> None:
    """Load normalizer states from checkpoint in runner.load()."""
    if not resumed_training:
        if load_critic:
            privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        return

    if not is_residual_policy:
        if use_split_normalizers:
            if "student_goal_obs_norm_state_dict" in loaded_dict:
                student_goal_obs_normalizer.load_state_dict(loaded_dict["student_goal_obs_norm_state_dict"])
            if "student_proprio_obs_norm_state_dict" in loaded_dict:
                student_proprio_obs_normalizer.load_state_dict(loaded_dict["student_proprio_obs_norm_state_dict"])
            elif "obs_norm_state_dict" in loaded_dict:
                old_mean = loaded_dict["obs_norm_state_dict"].get("_mean")
                old_var = loaded_dict["obs_norm_state_dict"].get("_var")
                old_std = loaded_dict["obs_norm_state_dict"].get("_std")
                proprio_dim = anybody_latent_proprio_dim
                if (
                    isinstance(old_mean, torch.Tensor)
                    and isinstance(old_var, torch.Tensor)
                    and isinstance(old_std, torch.Tensor)
                    and isinstance(proprio_dim, int)
                    and old_mean.shape[-1] >= proprio_dim
                ):
                    with torch.no_grad():
                        student_proprio_obs_normalizer._mean.copy_(old_mean[..., -proprio_dim:])
                        student_proprio_obs_normalizer._var.copy_(old_var[..., -proprio_dim:])
                        student_proprio_obs_normalizer._std.copy_(old_std[..., -proprio_dim:])
            if hasattr(student_proprio_obs_normalizer, "until"):
                student_proprio_obs_normalizer.until = getattr(student_proprio_obs_normalizer, "count", 0)
            student_proprio_obs_normalizer.eval()
        else:
            obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])

    if training_type == "mosaic":
        load_privileged_normalizer = load_critic
        if hasattr(alg, "teacher_critic_checkpoint_path") and alg.teacher_critic_checkpoint_path is not None:
            if hasattr(alg, "teacher_critic_frozen") and alg.teacher_critic_frozen:
                load_privileged_normalizer = False
                print("[Runner] Keeping privileged_obs_normalizer from teacher_critic_checkpoint (frozen).")
        if load_privileged_normalizer:
            if "privileged_obs_norm_state_dict" in loaded_dict:
                privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
                print("[Runner] Loaded privileged_obs_normalizer from checkpoint.")
            else:
                print("[Runner] WARNING: No privileged_obs_norm_state_dict in checkpoint!")
        if "teacher_obs_norm_state_dict" in loaded_dict:
            teacher_obs_normalizer.load_state_dict(loaded_dict["teacher_obs_norm_state_dict"])
            print("[Runner] Loaded teacher_obs_normalizer from checkpoint.")
    else:
        if load_critic:
            ckpt_priv = loaded_dict.get("privileged_obs_norm_state_dict", None)
            if ckpt_priv is None:
                pass
            else:
                ckpt_mean = ckpt_priv.get("_mean", None)
                cur_mean = getattr(privileged_obs_normalizer, "_mean", None)
                if (
                    isinstance(ckpt_mean, torch.Tensor)
                    and isinstance(cur_mean, torch.Tensor)
                    and tuple(ckpt_mean.shape) != tuple(cur_mean.shape)
                ):
                    # Shape mismatch happens when the teacher was swapped between runs (e.g. the
                    # teacher's obs schema changed). The privileged_obs_normalizer is aliased to
                    # the teacher's obs dim for distillation, and the new teacher's normalizer
                    # was already loaded from its own checkpoint. Keeping that fresh one is the
                    # correct thing to do — skip the stale resume-side overwrite.
                    print(
                        f"[Runner] WARNING: privileged_obs_norm_state_dict shape mismatch "
                        f"(ckpt {tuple(ckpt_mean.shape)} vs current {tuple(cur_mean.shape)}); "
                        f"keeping the freshly-loaded teacher normalizer and skipping resume overwrite."
                    )
                else:
                    privileged_obs_normalizer.load_state_dict(ckpt_priv)


def freeze_normalizers_on_resume(
    *,
    empirical_normalization: bool,
    freeze_normalizer: bool,
    use_split_normalizers: bool,
    obs_normalizer,
    privileged_obs_normalizer,
    student_goal_obs_normalizer,
    student_proprio_obs_normalizer,
) -> None:
    """Freeze normalizers after resume if requested by config."""
    if not (freeze_normalizer and empirical_normalization):
        return
    if use_split_normalizers:
        student_goal_obs_normalizer.eval()
        if hasattr(student_goal_obs_normalizer, "until"):
            student_goal_obs_normalizer.until = student_goal_obs_normalizer.count
        print(f"[Runner] Froze student_goal_obs_normalizer (count={student_goal_obs_normalizer.count})")
        student_proprio_obs_normalizer.eval()
        if hasattr(student_proprio_obs_normalizer, "until"):
            student_proprio_obs_normalizer.until = student_proprio_obs_normalizer.count
        print(f"[Runner] Froze student_proprio_obs_normalizer (count={student_proprio_obs_normalizer.count})")
    else:
        obs_normalizer.eval()
        if hasattr(obs_normalizer, "until"):
            obs_normalizer.until = obs_normalizer.count
        print(f"[Runner] Froze obs_normalizer (count={obs_normalizer.count})")
    privileged_obs_normalizer.eval()
    if hasattr(privileged_obs_normalizer, "until"):
        privileged_obs_normalizer.until = privileged_obs_normalizer.count
    print(f"[Runner] Froze privileged_obs_normalizer (count={privileged_obs_normalizer.count})")
