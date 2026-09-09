#!/usr/bin/env bash
# Activate the Anybody conda env and start official Stage 2 in the current pane.
set -euo pipefail
export HOME="${HOME:-/data/home/chenxiangyu}"
export USER="${USER:-chenxiangyu}"
export LOGNAME="${LOGNAME:-chenxiangyu}"
source /data/home/chenxiangyu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
ROOT="/data/home/chenxiangyu/robotics/Anybody"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-$ROOT/logs/rsl_rl/g1_flat_muse_transformer_distillation/2026-08-22_08-27-02_muse_transformer_det_cosine_w0.1_102k_teacher100k/model_10000.pt}"
export MOTION_DIR="${MOTION_DIR:-$ROOT/datasets/SONIC_npzs/g1/npz_splits_loco_manip/train}"
export RUN_NAME="${RUN_NAME:-muse_kp5_latent_distill_102k_teacher10k}"
export MASTER_PORT="${MASTER_PORT:-29600}"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/gmt_stage2_muse_kp_latent.log"
echo "[INFO] Stage 2 MUSE-Kp starting $(date -Is) log=$LOG"
echo "[INFO] TEACHER_CHECKPOINT=$TEACHER_CHECKPOINT"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] python=$(which python) torchrun=$(command -v torchrun || true)"
cd "$ROOT"
bash "$ROOT/run/train/MUSE/run_muse_kp_latent_local.sh" 2>&1 | tee -a "$LOG"
