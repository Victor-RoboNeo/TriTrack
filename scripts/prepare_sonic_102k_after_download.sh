#!/usr/bin/env bash
# After ModelScope g1.tar.gz is extracted: filter loco-manip CSVs, then Isaac-convert to npz.
# Conversion uses CUDA_VISIBLE_DEVICES="" / --device cpu so MOSAIC 8-GPU training is not stolen.
set -euo pipefail

ANYBODY="${ANYBODY:-/data/home/chenxiangyu/robotics/Anybody}"
SEED="${SEED:-$ANYBODY/datasets/bones-seed}"
CSV_ROOT="${CSV_ROOT:-$SEED/g1/csv}"
SPLIT_CSV="${SPLIT_CSV:-$ANYBODY/datasets/SONIC_npzs/g1/csv_splits_loco_manip/train}"
SPLIT_NPZ="${SPLIT_NPZ:-$ANYBODY/datasets/SONIC_npzs/g1/npz_splits_loco_manip/train}"
MANIFEST="${MANIFEST:-$ANYBODY/datasets/SONIC_npzs/g1/csv_splits_loco_manip/train_manifest.csv}"
LOG="${LOG:-$ANYBODY/logs/bones_seed_prepare.log}"

mkdir -p "$(dirname "$LOG")" "$SPLIT_NPZ"
exec > >(tee -a "$LOG") 2>&1

if [[ ! -d "$CSV_ROOT" ]]; then
  echo "[ERROR] CSV root missing: $CSV_ROOT  (extract g1.tar.gz first)" >&2
  exit 1
fi

source /data/home/chenxiangyu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
export PYTHONNOUSERSITE=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y

python "$ANYBODY/scripts/filter_bones_seed_loco_manip.py" \
  --csv_root "$CSV_ROOT" \
  --out_dir "$SPLIT_CSV" \
  --manifest "$MANIFEST" \
  --mode symlink \
  --min_duration 3.0

n=$(find "$SPLIT_CSV" -name '*.csv' | wc -l)
echo "[INFO] loco-manip CSV count=$n"

# Official 8-GPU conversion (see convert_loco_manip_gpu.sh). Do not hide GPUs.
bash "$ANYBODY/scripts/convert_loco_manip_gpu.sh"
