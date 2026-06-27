#!/usr/bin/env bash
# Prior-only rollout evaluation: eval_iters motion clips (outer loop), rollout_length env steps
# per clip (like play.py --video_length, default 500). Wandb project evaluate_prior_rollout.
# Uses streaming metrics (no full-horizon tensor hoard). LOG_INTERVAL prints like training.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

TASK="${TASK:-PULSE-Distill-General-Tracking-Flat-G1-v0}"
MOTION_DIR="${MOTION_DIR:-/home/lsn/Datasets/bones-seed/g1/npz_splits/test}"
LOAD_RUN="${LOAD_RUN:-2026-03-28_08-13-44_PULSE_sonic_data}"
CHECKPOINT="${CHECKPOINT:-model_6000.pt}"
NUM_ENVS="${NUM_ENVS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-15}"
ROLLOUT_LENGTH="${ROLLOUT_LENGTH:-300}"
LOG_INTERVAL="${LOG_INTERVAL:-30}"

HYDRA_FULL_ERROR=1 python scripts/rsl_rl/evaluate_prior_rollout.py \
  --task "${TASK}" \
  --motion "${MOTION_DIR}" \
  --num_envs "${NUM_ENVS}" \
  --eval_iters "${EVAL_ITERS}" \
  --rollout_length "${ROLLOUT_LENGTH}" \
  --log_interval "${LOG_INTERVAL}" \
  --load_run "${LOAD_RUN}" \
  --checkpoint "${CHECKPOINT}" \
  --headless \
  --logger wandb \
  --log_project_name evaluate_prior_rollout \
  --run_name "prior_eval_${CHECKPOINT%.pt}" \
  --disable_motion_group_sampling \
  "$@"

# EVAL_ITERS=120 ROLLOUT_LENGTH=500 bash run/evaluate_prior_rollout.sh
# MOTION_DIR=/path/to/one_clip.npz  # repeats the same clip EVAL_ITERS times
