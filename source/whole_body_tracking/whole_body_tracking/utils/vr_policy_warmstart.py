"""Load compatible weights into a PPO ``ActorCritic`` from a masked partial-KP student or another PPO ckpt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCritic


def _linear_child_indices(seq: nn.Sequential) -> list[int]:
    return [i for i, m in enumerate(seq.children()) if isinstance(m, torch.nn.Linear)]


def _linear_children(seq: nn.Sequential) -> list[nn.Linear]:
    return [m for m in seq.children() if isinstance(m, nn.Linear)]


def _state_dict_linear_child_indices(state_dict: dict[str, Any], prefix: str) -> list[int]:
    p = prefix if prefix.endswith(".") else prefix + "."
    found: set[int] = set()
    for k in state_dict:
        if not k.startswith(p) or not k.endswith(".weight"):
            continue
        rest = k[len(p) :]
        head = rest.split(".", 1)[0]
        if head.isdigit():
            found.add(int(head))
    return sorted(found)


def _is_residual_student_checkpoint(sd: dict[str, Any]) -> bool:
    return any(str(k).startswith("residual_encoder_body.") for k in sd)


def _is_actor_critic_checkpoint(sd: dict[str, Any]) -> bool:
    return any(str(k).startswith("actor.") for k in sd)


def apply_vr_policy_warmstart(policy: nn.Module, checkpoint_path: str) -> None:
    """Load actor (+ ``std`` / ``log_std``) weights when tensor shapes match.

    Supported checkpoints:

    - **LatentBottleneckAnyBody** (masked partial KP): copies
      ``residual_encoder_body`` linears onto the **first** matching actor linears,
      ``residual_mu`` onto the **last** actor linear, and ``std`` if compatible.
    - **ActorCritic**: copies overlapping ``actor.*`` and noise parameters (no critic).

    Raises if the file is missing, the policy is not ``ActorCritic``, or required
    tensors are shape-incompatible.
    """
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"warmstart checkpoint not found: {path}")

    loaded = torch.load(str(path), map_location="cpu", weights_only=False)
    src: dict[str, Any] = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded  # type: ignore[assignment]
    if not isinstance(src, dict):
        raise TypeError("Checkpoint must contain a dict model_state_dict or be a flat state dict.")

    if not isinstance(policy, ActorCritic):
        raise TypeError(
            "VR policy warmstart only supports ActorCritic "
            f"(got {type(policy).__name__})."
        )
    if getattr(policy, "ref_vel_skip_first_layer", False):
        raise NotImplementedError("VR warmstart does not support ref_vel_skip_first_layer ActorCritic.")

    if _is_residual_student_checkpoint(src):
        _warmstart_from_residual_student(policy, src)
    elif _is_actor_critic_checkpoint(src):
        _warmstart_from_actor_critic_state_dict(policy, src)
    else:
        sample = list(src.keys())[:24]
        raise ValueError(
            "Unrecognized warmstart checkpoint: expected keys like "
            "'residual_encoder_body.*' (masked partial KP student) or 'actor.*' (PPO). "
            f"Sample keys: {sample}"
        )


def _warmstart_from_residual_student(policy: ActorCritic, src: dict[str, Any]) -> None:
    if not isinstance(policy.actor, nn.Sequential):
        raise TypeError("Expected policy.actor to be nn.Sequential for residual-student warmstart.")

    dev = next(policy.parameters()).device
    src_enc_indices = _state_dict_linear_child_indices(src, "residual_encoder_body")
    if not src_enc_indices:
        raise ValueError("Checkpoint has no residual_encoder_body.*.weight tensors.")
    rank0 = int(__import__("os").environ.get("RANK", "0")) == 0
    if rank0:
        print("[INFO] VR warmstart checkpoint tensor shapes:")
        for src_i in src_enc_indices:
            for suf in ("weight", "bias"):
                k = f"residual_encoder_body.{src_i}.{suf}"
                v = src.get(k)
                shape = tuple(v.shape) if isinstance(v, torch.Tensor) else "missing"
                print(f"  - {k}: {shape}")
        for k in ("residual_mu.weight", "residual_mu.bias", "std", "log_std"):
            v = src.get(k)
            if isinstance(v, torch.Tensor):
                print(f"  - {k}: {tuple(v.shape)}")

    dst_lin = _linear_child_indices(policy.actor)
    dst_linear_modules = _linear_children(policy.actor)
    n_enc = len(src_enc_indices)
    if len(dst_linear_modules) < n_enc + 1:
        raise ValueError(
            f"Actor has too few linear layers for student warmstart: actor_linears={len(dst_linear_modules)}, "
            f"student_encoder_linears={n_enc} (need at least encoder+1 for residual_mu)."
        )

    dst_enc_targets = dst_linear_modules[:n_enc]
    # Faithful mapping: checkpoint has an explicit residual_mu head (no index),
    # so map it to the actor's final linear output head.
    dst_mu_layer = dst_linear_modules[-1]

    def _copy_seq_linear(sd: dict[str, Any], src_prefix: str, src_i: int, dst_layer: nn.Linear) -> None:
        for suf in ("weight", "bias"):
            sk = f"{src_prefix}.{src_i}.{suf}"
            if sk not in sd:
                raise KeyError(f"Missing {sk} in warmstart checkpoint")
            sv = sd[sk]
            tv = getattr(dst_layer, suf)
            if sv.shape != tv.shape:
                raise ValueError(
                    f"Shape mismatch warmstarting {sk} -> actor.<linear>.{suf}: "
                    f"checkpoint {tuple(sv.shape)} vs policy {tuple(tv.shape)}"
                )
            with torch.no_grad():
                tv.copy_(sv.to(device=dev, dtype=tv.dtype))

    for src_i, dst_layer in zip(src_enc_indices, dst_enc_targets):
        _copy_seq_linear(src, "residual_encoder_body", src_i, dst_layer)

    for suf in ("weight", "bias"):
        sk = f"residual_mu.{suf}"
        if sk not in src:
            raise KeyError(f"Missing {sk} in warmstart checkpoint")
        sv = src[sk]
        tv = getattr(dst_mu_layer, suf)
        if sv.shape != tv.shape:
            raise ValueError(
                f"Shape mismatch warmstarting {sk} -> actor.<last_linear>.{suf}: "
                f"checkpoint {tuple(sv.shape)} vs policy {tuple(tv.shape)}"
            )
        with torch.no_grad():
            tv.copy_(sv.to(device=dev, dtype=tv.dtype))

    if "std" in src and hasattr(policy, "std"):
        sv, tv = src["std"], policy.std
        if sv.shape == tv.shape:
            with torch.no_grad():
                tv.copy_(sv.to(device=dev, dtype=tv.dtype))
    elif "log_std" in src and hasattr(policy, "log_std"):
        sv, tv = src["log_std"], policy.log_std
        if sv.shape == tv.shape:
            with torch.no_grad():
                tv.copy_(sv.to(device=dev, dtype=tv.dtype))

    if rank0:
        mid = dst_lin[n_enc:-1] if len(dst_lin) > n_enc + 1 else []
        print(
            "[INFO] VR policy warmstart (masked partial KP student): "
            f"copied residual_encoder_body indices {src_enc_indices} -> first {n_enc} actor linear layers, "
            "residual_mu -> last actor linear layer; "
            f"actor layer indices left randomly initialized: {mid}"
        )


def _warmstart_from_actor_critic_state_dict(policy: ActorCritic, src: dict[str, Any]) -> None:
    dev = next(policy.parameters()).device
    dst = policy.state_dict()
    to_load: dict[str, torch.Tensor] = {}
    mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for k, v in src.items():
        if str(k).startswith("critic."):
            continue
        if k not in dst:
            continue
        if not isinstance(v, torch.Tensor):
            continue
        if dst[k].shape != v.shape:
            mismatches.append((k, tuple(v.shape), tuple(dst[k].shape)))
            continue
        to_load[k] = v.to(device=dev, dtype=dst[k].dtype)

    actor_keys = [k for k in to_load if str(k).startswith("actor.")]
    if not actor_keys:
        raise ValueError(
            "ActorCritic warmstart found no compatible actor.* tensors "
            f"(shape mismatches: {mismatches[:8]}{'...' if len(mismatches) > 8 else ''})."
        )
    policy.load_state_dict(to_load, strict=False)
    rank0 = int(__import__("os").environ.get("RANK", "0")) == 0
    if rank0:
        print(
            f"[INFO] VR policy warmstart (ActorCritic checkpoint): loaded {len(to_load)} tensors "
            f"({len(actor_keys)} actor keys). Skipped critic."
        )
