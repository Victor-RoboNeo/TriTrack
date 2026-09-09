#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/data/home/chenxiangyu/robotics/Anybody"
PKG="${ROOT}/scripts/flat_unified_sparse_intent_v1"
RESULTS="${FLAT_FUSI_ROOT:-${ROOT}/results/flat_unified_sparse_intent_v1}"
PY_HT="/data/home/chenxiangyu/miniconda3/envs/humantracker/bin/python"
export FLAT_FUSI_ROOT="${RESULTS}"
export PYTHONUNBUFFERED=1
mkdir -p "${RESULTS}/logs" "${RESULTS}/reports"

cd "${ROOT}"
exec "${PY_HT}" "${PKG}/orchestrator.py" --resume "$@"
