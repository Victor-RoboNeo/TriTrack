#!/usr/bin/env bash
# Full pipeline: ModelScope download -> extract -> loco-manip filter -> npz convert.
# Safe to run while MOSAIC GMT occupies the 8 GPUs (download is CPU/network; convert uses CPU Isaac).
set -euo pipefail
export HOME="${HOME:-/data/home/chenxiangyu}"
export USER="${USER:-chenxiangyu}"
export LOGNAME="${LOGNAME:-chenxiangyu}"
ANYBODY="/data/home/chenxiangyu/robotics/Anybody"
LOG="$ANYBODY/logs/bones_seed_modelscope.log"
mkdir -p "$ANYBODY/logs"
exec > >(tee -a "$LOG") 2>&1
echo "[INFO] start $(date -Is) pid=$$"
bash "$ANYBODY/scripts/download_bones_seed_modelscope.sh"
bash "$ANYBODY/scripts/prepare_sonic_102k_after_download.sh"
echo "[INFO] all done $(date -Is)"
