#!/usr/bin/env bash
# MUSE-Transformer distillation: deterministic encoder + unit-norm latent recipe.
#
# Encoder: 1 un-masked command token + H=5 proprio tokens (+ [CLS]) -> transformer -> mu.
# Env-driven masking: env's goal_mask_history (1-step) drops the command token from attention
# via key_padding_mask. Curriculum ramps p_mask 0.0 -> 0.5 over iter 500 -> 4000.
# Latent: z := mu (no reparameterization), L2-normalized onto the unit hypersphere before decoder.
# Decoder: full proprio history flattened (450-d for G1) by default.
# Algorithm: BC + fixed-weight cosine smoothness on mu (w=0.01). No KL: framing is input-modal
# conditioning, not output sampling, so no need for a Gaussian prior. Unit-norm makes the
# Kendall-adaptive workaround unnecessary (||mu|| can't drift anymore).
#
# Required env vars:
#   TEACHER_CHECKPOINT   path to stage-1 teacher .pt (e.g. mosaic_hybrid run)
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES, MOTION_DIR, NUM_ENVS, RUN_NAME, RESUME_CHECKPOINT
#
# Example:
#   TEACHER_CHECKPOINT=logs/rsl_rl/g1_flat_mosaic_hybrid/<run>/model_55000.pt \
#   bash run/train/run_muse_transformer_distillation.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-logs/rsl_rl/g1_flat_mosaic_hybrid/2026-05-14_14-33-50_sonic_55k_teacher/model_76000.pt}"

MOTION_DIR="${MOTION_DIR:-/home/lsn/Datasets/SONIC_npzs/g1/npz_by_motion/train}"
NUM_ENVS="${NUM_ENVS:-2048}"
RUN_NAME="${RUN_NAME:-muse_transformer_det_cosine_w_0.1_55k_no_mask}"

#RESUME_CHECKPOINT="logs/rsl_rl/g1_flat_muse_transformer_distillation/2026-05-18_20-38-29_muse_transformer_det_cosine_w_0.1/model_30000.pt"

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    EXTRA_ARGS+=(--resume_student_checkpoint "$RESUME_CHECKPOINT")
fi

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/rsl_rl/train.py \
    --task=MUSE-Transformer-Distill-General-Tracking-Flat-G1-v0 \
    --distributed \
    --num_envs="$NUM_ENVS" \
    --motion "$MOTION_DIR" \
    --teacher_checkpoint "$TEACHER_CHECKPOINT" \
    --headless \
    --logger wandb \
    --log_project_name MUSE_Distill \
    --run_name "$RUN_NAME" \
    "${EXTRA_ARGS[@]}"

# bash run/train/MUSE/run_muse_transformer_distillation.sh
