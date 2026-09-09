#!/usr/bin/env bash
# Activate the Anybody conda env and start official Stage 1 in the current pane.
set -euo pipefail
export HOME="${HOME:-/data/home/chenxiangyu}"
export USER="${USER:-chenxiangyu}"
export LOGNAME="${LOGNAME:-chenxiangyu}"
source /data/home/chenxiangyu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
ROOT="/data/home/chenxiangyu/robotics/Anybody"
export TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-$ROOT/logs/rsl_rl/g1_flat_mosaic_hybrid/2026-08-19_16-02-00_sonic_102k_teacher/model_100000.pt}"
export MOTION_DIR="${MOTION_DIR:-$ROOT/datasets/SONIC_npzs/g1/npz_splits_loco_manip/train}"
export RUN_NAME="${RUN_NAME:-muse_transformer_det_cosine_w0.1_102k_teacher100k}"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/gmt_stage1_muse_transformer.log"
echo "[INFO] Stage 1 MUSE-Transformer starting $(date -Is) log=$LOG"
echo "[INFO] TEACHER_CHECKPOINT=$TEACHER_CHECKPOINT"
echo "[INFO] MOTION_DIR=$MOTION_DIR"
echo "[INFO] python=$(which python) torchrun=$(command -v torchrun || true)"
cd "$ROOT"
bash "$ROOT/run/train/MUSE/run_muse_transformer_local.sh" 2>&1 | tee -a "$LOG"
