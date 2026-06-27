#!/usr/bin/env bash
set -euo pipefail

# Evaluate the JC MUSE-Transformer teacher on a prepared motion test set and push
# all metrics to wandb (single GPU).
#
# Modeled on run/test/evaluate_test_set_metrics.sh, but for the
# MUSE-Transformer-Distill (joint-command) teacher trained 2026-05-18.
#
# Test-time semantics (per request):
#   - EVERY timestep visible: --p_mask 0.0. evaluate_policy.py disables the
#     goal-mask curriculum (which otherwise ramps p_mask 0->0.5 by iter ~2000)
#     and pins motion.p_mask=0.0 so the goal is never masked.
#   - A dedicated wandb panel ``eval_world/`` is logged by WorldPoiMetricLogger:
#       eval_world/anchor_pos      world-frame L2 anchor (torso) pos error
#       eval_world/anchor_lin_vel  world-frame L2 anchor lin-vel error
#       eval_world/poi_pos         mean world-frame L2 pos error over the 5 POI
#                                  (torso + L/R wrist + L/R ankle), measured vs
#                                  the RAW clip world reference (NOT re-anchored
#                                  to the robot)
#       eval_world/poi_lin_vel     mean world-frame L2 lin-vel error over the 5 POI
#       eval_world/steps           samples accumulated this iteration
#
# Override via environment variables:
#   CUDA_VISIBLE_DEVICES  (default: 0)
#   MOTION_DIR, LOAD_RUN, CHECKPOINT
#   NUM_ENVS, EVAL_ITERS
#   WANDB_PROJECT, WANDB_RUN_NAME

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

MOTION_DIR="${MOTION_DIR:-/home/lsn/Datasets/SONIC_npzs/g1/npz_by_motion/train}"
LOAD_RUN="${LOAD_RUN:-2026-05-18_20-38-29_muse_transformer_det_cosine_w_0.1}"
CHECKPOINT="${CHECKPOINT:-model_10000.pt}"
NUM_ENVS="${NUM_ENVS:-12000}"
EVAL_ITERS="${EVAL_ITERS:-120}"
WANDB_PROJECT="${WANDB_PROJECT:-test_MUSE_Transformer_teacher}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-muse_transformer_det_cosine_w0.1}"

TASK="MUSE-Transformer-Distill-General-Tracking-Flat-G1-v0"

if [[ ! -d "$MOTION_DIR" ]]; then
  echo "Error: MOTION_DIR=$MOTION_DIR not found." >&2
  exit 1
fi

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/rsl_rl/evaluate_policy.py \
  --task="${TASK}" \
  --motion="${MOTION_DIR}" \
  --num_envs="${NUM_ENVS}" \
  --load_run="${LOAD_RUN}" \
  --checkpoint="${CHECKPOINT}" \
  --eval_iters="${EVAL_ITERS}" \
  --p_mask=0.0 \
  --headless \
  --logger wandb \
  --log_project_name "${WANDB_PROJECT}" \
  --run_name "${WANDB_RUN_NAME}"

# CUDA_VISIBLE_DEVICES=0 EVAL_ITERS=100 NUM_ENVS=4096 bash run/test/evaluate_muse_transformer_test_metrics.sh
