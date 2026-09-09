#!/usr/bin/env bash
# Official-style 8-GPU Bones-SEED CSV -> NPZ conversion.
# Does NOT hide GPUs. Each shard uses CUDA_VISIBLE_DEVICES=<shard> like
# Anybody scripts/run_batch_csv_to_npz_shards.sh.
set -euo pipefail

export HOME="${HOME:-/data/home/chenxiangyu}"
export USER="${USER:-chenxiangyu}"
export LOGNAME="${LOGNAME:-chenxiangyu}"
source /data/home/chenxiangyu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
export PYTHONNOUSERSITE=1
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
unset CUDA_VISIBLE_DEVICES || true

ANYBODY="/data/home/chenxiangyu/robotics/Anybody"
SPLIT_CSV="${SPLIT_CSV:-$ANYBODY/datasets/SONIC_npzs/g1/csv_splits_loco_manip/train}"
SHARD_ROOT="${SHARD_ROOT:-$ANYBODY/datasets/SONIC_npzs/g1/csv_by_gpu}"
OUT_DIR="${OUT_DIR:-$ANYBODY/datasets/SONIC_npzs/g1/npz_splits_loco_manip/train}"
LOG="${LOG:-$ANYBODY/logs/bones_seed_convert_gpu.log}"
NPROC="${NPROC:-8}"

mkdir -p "$OUT_DIR" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[INFO] GPU convert start $(date -Is)"

if [[ ! -d "$SPLIT_CSV" ]]; then
  echo "[ERROR] filtered CSV dir missing: $SPLIT_CSV" >&2
  exit 1
fi

python - <<PY
from pathlib import Path
src_root = Path("$SPLIT_CSV")
shard_root = Path("$SHARD_ROOT")
nproc = int("$NPROC")
csvs = sorted(p for p in src_root.rglob("*.csv") if p.is_file() or p.is_symlink())
if not csvs:
    raise SystemExit(f"no csv under {src_root}")
# rebuild shards
if shard_root.exists():
    import shutil
    shutil.rmtree(shard_root)
for i in range(nproc):
    (shard_root / str(i)).mkdir(parents=True, exist_ok=True)
for idx, src in enumerate(csvs):
    rel = src.relative_to(src_root)
    dst = shard_root / str(idx % nproc) / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())
print(f"[INFO] sharded {len(csvs)} csv -> {shard_root} nproc={nproc}")
PY

run_shard() {
  local i="$1"
  local indir="${SHARD_ROOT}/${i}"
  local shard_log="$ANYBODY/logs/convert_shard_${i}.log"
  echo "[INFO] shard $i -> $shard_log"
  CUDA_VISIBLE_DEVICES="$i" \
  OMNI_USER_DIR="/tmp/isaaclab_convert_${USER}_shard${i}" \
  XDG_CACHE_HOME="/tmp/isaaclab_convert_${USER}_shard${i}/cache" \
  python "$ANYBODY/scripts/batch_csv_to_npz.py" \
    --input_dir "$indir" \
    --output_dir "$OUT_DIR" \
    --output_prefix bones_seed_g1 \
    --input_fps 120 \
    --output_fps 50 \
    --bones_seed_g1_csv \
    --min_duration 3.0 \
    --skip_existing \
    --headless \
    --device cuda:0 \
    >"$shard_log" 2>&1
}

pids=()
for i in $(seq 0 $((NPROC - 1))); do
  run_shard "$i" &
  pids+=("$!")
done

ec=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[ERROR] shard $i pid=${pids[$i]} failed, see $ANYBODY/logs/convert_shard_${i}.log" >&2
    ec=1
  fi
done

npz_n=$(find "$OUT_DIR" -name '*.npz' | wc -l)
echo "[INFO] NPZ count=$npz_n"
if [[ "$ec" -ne 0 || "$npz_n" -eq 0 ]]; then
  echo "[ERROR] GPU convert failed" >&2
  exit 1
fi
echo "[INFO] SONIC_102K_CONVERT_OK $(date -Is)"
