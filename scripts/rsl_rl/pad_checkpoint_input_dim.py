#!/usr/bin/env python
"""Pad an RSL-RL checkpoint's actor input dimension.

Use case: warm-start a re-trained teacher after adding new obs terms at the END of the
policy obs group. Old actor `Linear(770, 1024)` becomes new `Linear(785, 1024)`; new
input columns are initialized to **zero** so the actor's behavior at iteration 0 is
bitwise identical to the pre-pad checkpoint. PPO then learns to use the new obs during
fine-tuning.

What gets padded (read-only — script writes a new .pt, never overwrites the source):
- ``model_state_dict["actor.0.weight"]``                     : new cols = 0
- ``obs_norm_state_dict["_mean"]``                           : new cols = 0
- ``obs_norm_state_dict["_var"]``                            : new cols = 1
- ``obs_norm_state_dict["_std"]``                            : new cols = 1
- ``optimizer_state_dict["state"][i]["exp_avg"]``            : new cols = 0  (Adam 1st moment)
- ``optimizer_state_dict["state"][i]["exp_avg_sq"]``         : new cols = 0  (Adam 2nd moment)
   (only for the state entry whose tensor shape matches the old actor.0.weight; identified
   by shape match — any tensor of shape exactly equal to the old actor.0.weight is padded.)

What is preserved unchanged: actor.0.bias, all later actor layers, all critic layers,
privileged_obs_norm_state_dict, iter, infos.

Critic is not padded — the privileged (critic) obs group is unchanged.

Example:
    python scripts/rsl_rl/pad_checkpoint_input_dim.py \\
        --input  logs/.../mosaic_hybrid/<run>/model_55000.pt \\
        --output logs/.../mosaic_hybrid/<run>/model_55000__padded_785.pt \\
        --old_dim 770 --new_dim 785
"""
from __future__ import annotations

import argparse
import os

import torch


def _pad_last_dim(tensor: torch.Tensor, new_size: int, fill: float) -> torch.Tensor:
    if tensor.shape[-1] >= new_size:
        raise ValueError(
            f"Expected last dim < new_size; got tensor shape {tuple(tensor.shape)} vs new_size={new_size}."
        )
    pad_cols = new_size - tensor.shape[-1]
    pad_shape = list(tensor.shape)
    pad_shape[-1] = pad_cols
    pad = torch.full(pad_shape, float(fill), dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Source .pt path.")
    p.add_argument("--output", required=True, help="Destination .pt path (must differ from input).")
    p.add_argument("--old_dim", type=int, required=True, help="Current actor input dim (e.g. 770).")
    p.add_argument("--new_dim", type=int, required=True, help="New actor input dim (e.g. 785).")
    p.add_argument(
        "--actor_first_weight_key",
        default="actor.0.weight",
        help="Key of the actor's first linear weight in model_state_dict.",
    )
    args = p.parse_args()

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        raise ValueError("--input and --output must differ.")
    if args.new_dim <= args.old_dim:
        raise ValueError(f"new_dim ({args.new_dim}) must be greater than old_dim ({args.old_dim}).")

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    msd = ckpt.get("model_state_dict")
    if msd is None or args.actor_first_weight_key not in msd:
        raise KeyError(
            f"{args.actor_first_weight_key!r} not found in model_state_dict. "
            f"Available first-layer-ish keys: "
            f"{[k for k in (msd or {}) if k.endswith('0.weight')]}"
        )

    # ---- Actor first layer ----
    w_old = msd[args.actor_first_weight_key]
    if w_old.shape[-1] != args.old_dim:
        raise ValueError(
            f"{args.actor_first_weight_key} last dim is {w_old.shape[-1]}, expected --old_dim={args.old_dim}."
        )
    w_new = _pad_last_dim(w_old, args.new_dim, fill=0.0)
    msd[args.actor_first_weight_key] = w_new
    print(f"[pad] {args.actor_first_weight_key}: {tuple(w_old.shape)} → {tuple(w_new.shape)} (zero-pad)")

    # ---- Optimizer state (Adam moments) ----
    # Adam tracks per-parameter exp_avg / exp_avg_sq buffers. The buffer for actor.0.weight
    # is now shape-mismatched against the padded param; pad its last dim with zeros (Adam's
    # default init for new params is zero on both moments).
    opt = ckpt.get("optimizer_state_dict")
    old_actor_shape = tuple(w_old.shape)  # e.g. (1024, 770)
    if opt is not None and "state" in opt:
        n_padded = 0
        for sidx, entry in opt["state"].items():
            for buf_key in ("exp_avg", "exp_avg_sq"):
                t = entry.get(buf_key)
                if t is None or not hasattr(t, "shape"):
                    continue
                if tuple(t.shape) != old_actor_shape:
                    continue
                entry[buf_key] = _pad_last_dim(t, args.new_dim, fill=0.0)
                print(
                    f"[pad] optimizer.state[{sidx}].{buf_key}: "
                    f"{tuple(t.shape)} → {tuple(entry[buf_key].shape)} (zero-pad)"
                )
                n_padded += 1
        if n_padded == 0:
            print(
                "[pad] WARNING: optimizer_state_dict had no tensors matching old actor.0.weight "
                f"shape {old_actor_shape!r} — resume will likely error in optimizer.step()."
            )
        elif n_padded != 2:
            print(
                f"[pad] WARNING: padded {n_padded} optimizer buffers; expected exactly 2 "
                "(exp_avg + exp_avg_sq for actor.0.weight)."
            )
    else:
        print("[pad] optimizer_state_dict not present — skipping optimizer pad.")

    # ---- Obs normalizer ----
    onsd = ckpt.get("obs_norm_state_dict")
    if onsd is not None:
        fills = {"_mean": 0.0, "_var": 1.0, "_std": 1.0}
        for key, fill in fills.items():
            if key not in onsd:
                continue
            t_old = onsd[key]
            if t_old.shape[-1] != args.old_dim:
                print(f"[pad] WARNING: obs_norm.{key} last dim {t_old.shape[-1]} ≠ old_dim — skipping.")
                continue
            onsd[key] = _pad_last_dim(t_old, args.new_dim, fill=fill)
            print(f"[pad] obs_norm.{key}: {tuple(t_old.shape)} → {tuple(onsd[key].shape)} (fill={fill})")
    else:
        print("[pad] obs_norm_state_dict not present — skipping normalizer pad.")

    torch.save(ckpt, args.output)
    print(f"[pad] wrote {args.output}")


if __name__ == "__main__":
    main()
