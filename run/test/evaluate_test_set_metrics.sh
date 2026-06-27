#!/usr/bin/env bash
set -euo pipefail

# Evaluate a trained policy checkpoint on a prepared motion test set (single GPU).
#
# Same as `run/evaluate_test_set_multiple_gpus.sh`, but one process and a default
# physical GPU index (CUDA_VISIBLE_DEVICES=7 when GPUs 0–6 are busy training).
#
# Override via environment variables:
#   CUDA_VISIBLE_DEVICES  (default: 7)
#   MOTION_DIR, LOAD_RUN, CHECKPOINT
#   NUM_ENVS, EVAL_ITERS
#
# Partial-mask tasks: pin one keypoint mask mode or sweep all modes in one wandb run:
#   EVAL_MASK_MODE=name_or_index   — e.g. full, upper, 0 (single mode for whole eval)
#   EVAL_ALL_MASK_MODES=1          — eval each mode for EVAL_ITERS iters; metrics under eval_mask/<name>/...

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MOTION_DIR="${MOTION_DIR:-/home/lsn/Datasets/SONIC_npzs/g1/npz_splits_filtered_small/test}"
LOAD_RUN="${LOAD_RUN:-2026-04-09_16-15-13_sample_mask_modes}"
CHECKPOINT="${CHECKPOINT:-model_15000.pt}"
NUM_ENVS="${NUM_ENVS:-12000}"
EVAL_ITERS="${EVAL_ITERS:-120}"
# Set EVAL_ALL_MASK_MODES=1 to evaluate every mask mode (EVAL_ITERS each) in one run; use EVAL_ITERS=20 for quick sweeps.
EVAL_EXTRA=()
if [[ "${EVAL_ALL_MASK_MODES:-0}" == "1" ]]; then
  EVAL_EXTRA+=(--eval_all_mask_modes)
elif [[ -n "${EVAL_MASK_MODE:-}" ]]; then
  EVAL_EXTRA+=(--eval_mask_mode "${EVAL_MASK_MODE}")
fi

TASK="Partial-Masked-Residual-Latent-Distill-2B-General-Tracking-Flat-G1-v0"
#"Residual-Latent-Distill-2B-General-Tracking-Flat-G1-v0"
#"General-Tracking-Flat-G1-Wo-State-Estimation-v0-World-Coordinate-Reward"

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/rsl_rl/evaluate_policy.py \
  --task="${TASK}" \
  --motion="${MOTION_DIR}" \
  --num_envs="${NUM_ENVS}" \
  --load_run="${LOAD_RUN}" \
  --checkpoint="${CHECKPOINT}" \
  --eval_iters="${EVAL_ITERS}" \
  "${EVAL_EXTRA[@]}" \
  --headless \
  --logger wandb \
  --log_project_name test_Partial_Masked_Residual_Latent_Distill_2B \
  --run_name sample_mask_modes


# EVAL_ITERS=20 EVAL_ALL_MASK_MODES=1 bash run/evaluate_test_set_single_gpu.sh

