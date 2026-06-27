#!/usr/bin/env bash
# MUSE distillation training (delta-command + random goal masking; no separate prior).
#
# Required env vars:
#   TEACHER_CHECKPOINT   path to stage-1 teacher .pt (e.g. mosaic_hybrid run)
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES, MOTION_DIR, NUM_ENVS, RUN_NAME, RESUME_CHECKPOINT
#
# Example:
#   TEACHER_CHECKPOINT=logs/rsl_rl/g1_flat_mosaic_hybrid/<run>/model_55000.pt \
#   bash run/train/run_muse_distillation.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=4
    
TEACHER_CHECKPOINT="logs/rsl_rl/g1_flat_mosaic_hybrid/2026-04-15_17-11-44_medium_5_step_1024_1024_512_512_256_256/model_55000.pt"

MOTION_DIR="/home/lsn/Datasets/SONIC_npzs/g1/npz_splits_filtered_medium/train"
NUM_ENVS=4096

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    EXTRA_ARGS+=(--resume_student_checkpoint "$RESUME_CHECKPOINT")
fi

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/rsl_rl/train.py \
    --task=MUSE-Distill-General-Tracking-Flat-G1-v0 \
    --distributed \
    --num_envs="$NUM_ENVS" \
    --motion "$MOTION_DIR" \
    --teacher_checkpoint "$TEACHER_CHECKPOINT" \
    --headless \
    --logger wandb \
    --log_project_name MUSE_Distill \
    --run_name muse_encoder_std_0.1 \
    "${EXTRA_ARGS[@]}"

# bash run/train/run_muse_distillation.sh