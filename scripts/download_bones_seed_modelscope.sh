#!/usr/bin/env bash
# Download Bones-SEED from ModelScope (not HuggingFace).
# For AnyBody sonic_102k_teacher we only need G1 CSVs + metadata.
set -euo pipefail

ROOT="${ROOT:-/data/home/chenxiangyu/robotics/Anybody/datasets/bones-seed}"
MS_DS="${MS_DS:-bones-studio/seed}"
BASE="https://www.modelscope.cn/datasets/${MS_DS}/resolve/master"
export http_proxy="${http_proxy:-http://127.0.0.1:7890}"
export https_proxy="${https_proxy:-http://127.0.0.1:7890}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"

mkdir -p "$ROOT/metadata"
cd "$ROOT"

download() {
  local rel="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"
  echo "[INFO] $(date -Is) GET $BASE/$rel -> $out"
  curl -L --fail --retry 20 --retry-all-errors --retry-delay 5 \
    --continue-at - \
    --connect-timeout 30 \
    -A "Mozilla/5.0" \
    -o "$out" \
    "$BASE/$rel"
  echo "[INFO] done $out size=$(stat -c%s "$out" 2>/dev/null || echo '?')"
}

# small files first (filter uses metadata)
download LICENSE.md "$ROOT/LICENSE.md"
download README.md "$ROOT/README.md"
download metadata/seed_metadata_v004.csv "$ROOT/metadata/seed_metadata_v004.csv"
download metadata/seed_metadata_v004.parquet "$ROOT/metadata/seed_metadata_v004.parquet"

# G1 MuJoCo-compatible CSVs (~23.5 GiB)
download g1.tar.gz "$ROOT/g1.tar.gz"

echo "[INFO] extracting g1.tar.gz (this takes a while)"
tar -xzf "$ROOT/g1.tar.gz" -C "$ROOT"
csv_n=$(find "$ROOT/g1" -name '*.csv' | wc -l)
echo "[INFO] extracted CSV count=$csv_n"
echo "[INFO] BONES_SEED_DOWNLOAD_OK"
