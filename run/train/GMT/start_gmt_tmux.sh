#!/usr/bin/env bash
# Activate the Anybody conda env and start official Stage 0 GMT in the current pane.
set -euo pipefail
export HOME="${HOME:-/data/home/chenxiangyu}"
export USER="${USER:-chenxiangyu}"
export LOGNAME="${LOGNAME:-chenxiangyu}"
source /data/home/chenxiangyu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
ROOT="/data/home/chenxiangyu/robotics/Anybody"
export MOTION_DIR="${MOTION_DIR:-$ROOT/datasets/SONIC_npzs/g1/npz_splits_loco_manip/train}"
export RUN_NAME="${RUN_NAME:-sonic_102k_teacher}"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/gmt_stage0_sonic102k.log"
echo "[INFO] GMT Stage 0 SONIC loco-manip starting $(date -Is) log=$LOG"
echo "[INFO] MOTION_DIR=$MOTION_DIR"
echo "[INFO] python=$(which python) torchrun=$(command -v torchrun || true)"
cd "$ROOT"
bash "$ROOT/run/train/GMT/run_mosaic_gmt_local.sh" 2>&1 | tee -a "$LOG"
