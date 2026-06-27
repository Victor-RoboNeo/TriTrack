#!/usr/bin/env bash
# Run batch_csv_to_npz.py on CSV shards csv_by_gpu/0 .. csv_by_gpu/7 (one Isaac process per GPU).
#
# Usage:
#   ./scripts/run_batch_csv_to_npz_shards.sh
#   MOSAIC_ROOT=... DATA_ROOT=... ./scripts/run_batch_csv_to_npz_shards.sh
#   SEQUENTIAL=1 ./scripts/run_batch_csv_to_npz_shards.sh    # one GPU job at a time
#   MAX_PARALLEL=4 ./scripts/run_batch_csv_to_npz_shards.sh  # cap concurrent Isaac instances
#
# Each worker sets CUDA_VISIBLE_DEVICES to its shard index. Ensure csv_by_gpu/N contains *.csv
# (optionally in subdirs); empty shards are skipped with a warning.

set -euo pipefail

MOSAIC_ROOT="${MOSAIC_ROOT:-/home/lsn/MOSAIC}"
DATA_ROOT="${DATA_ROOT:-/home/lsn/Datasets/bones-seed/g1}"
CSV_PARENT="${CSV_PARENT:-${DATA_ROOT}/csv_by_gpu}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/npz}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-bones_seed_g1}"
INPUT_FPS="${INPUT_FPS:-120}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
FIRST_SHARD="${FIRST_SHARD:-0}"
LAST_SHARD="${LAST_SHARD:-7}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
SEQUENTIAL="${SEQUENTIAL:-0}"

SCRIPT="${MOSAIC_ROOT}/scripts/batch_csv_to_npz.py"
if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: batch script not found: ${SCRIPT}" >&2
  exit 1
fi

has_csv_under() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 1
  find "${dir}" -type f -name '*.csv' -print -quit | grep -q .
}

run_shard() {
  local i="$1"
  local indir="${CSV_PARENT}/${i}"
  if ! has_csv_under "${indir}"; then
    echo "[WARN] No CSV files under ${indir} — skip shard ${i}" >&2
    return 0
  fi
  echo "=== Shard ${i}: CUDA_VISIBLE_DEVICES=${i} input=${indir} -> ${OUT_DIR} ==="
  CUDA_VISIBLE_DEVICES="${i}" python "${SCRIPT}" \
    --input_dir "${indir}" \
    --output_dir "${OUT_DIR}" \
    --output_prefix "${OUTPUT_PREFIX}" \
    --input_fps "${INPUT_FPS}" \
    --output_fps "${OUTPUT_FPS}" \
    --bones_seed_g1_csv \
    --headless
}

if [[ "${SEQUENTIAL}" == "1" ]]; then
  for i in $(seq "${FIRST_SHARD}" "${LAST_SHARD}"); do
    run_shard "${i}"
  done
  echo "All shard jobs finished."
  exit 0
fi

# Parallel batches (portable; does not rely on bash 5.1+ wait -n)
i="${FIRST_SHARD}"
while [[ "${i}" -le "${LAST_SHARD}" ]]; do
  pids=()
  for ((k = 0; k < MAX_PARALLEL && i <= LAST_SHARD; k++)); do
    run_shard "${i}" &
    pids+=("$!")
    i=$((i + 1))
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done

echo "All shard jobs finished."
