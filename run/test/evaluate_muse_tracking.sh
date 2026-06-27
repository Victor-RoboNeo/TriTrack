#!/usr/bin/env bash
set -euo pipefail
# Evaluate a MUSE-distilled policy on standard tracking with goals fully visible (p_mask = 0).
# The MUSE policy (encoder(real_goal) -> decoder) is given the real delta_command and real
# motion_anchor_ori_b each step. Records one video per motion subfolder under VISUALIZE_DIR.
#
# Required env vars:
#   LOAD_RUN   — basename under logs/rsl_rl/g1_flat_muse_distillation/
#   CHECKPOINT — e.g. model_10000.pt
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES, VISUALIZE_DIR, MAX_SUBFOLDERS, SUBFOLDER_SAMPLE_SEED,
#   PRINT_SAMPLED_SUBFOLDERS, VIDEO_LENGTH
#
# Example:
#   LOAD_RUN=2026-05-07_21-55-29_muse_initial_run_p_max_0.5 \
#   CHECKPOINT=model_10000.pt \
#   MAX_SUBFOLDERS=5 SUBFOLDER_SAMPLE_SEED=24 \
#   bash run/test/evaluate_muse_tracking.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOAD_RUN="${LOAD_RUN:-2026-05-07_21-55-29_muse_initial_run_p_max_0.5}"
CHECKPOINT="${CHECKPOINT:-model_10000.pt}"

VISUALIZE_DIR="${VISUALIZE_DIR:-/home/lsn/Datasets/SONIC_npzs/g1/npz_splits_filtered_medium/test}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
MAX_SUBFOLDERS="${MAX_SUBFOLDERS:-5}"

VIDEO_DIR_TAG="muse_tracking"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

if [[ ! -d "$VISUALIZE_DIR" ]]; then
    echo "Error: $VISUALIZE_DIR not found."
    exit 1
fi

subfolder_dirs=()
for p in "$VISUALIZE_DIR"/*/; do
    if [[ -d "$p" ]]; then
        subfolder_dirs+=("$p")
    fi
done

if [[ "${MAX_SUBFOLDERS}" != "0" && "${#subfolder_dirs[@]}" -gt "${MAX_SUBFOLDERS}" ]]; then
    echo "[evaluate_muse_tracking] Sampling MAX_SUBFOLDERS=${MAX_SUBFOLDERS} from ${#subfolder_dirs[@]} subfolders (seed='${SUBFOLDER_SAMPLE_SEED:-<none>}')."
    mapfile -t subfolder_dirs < <(
        printf "%s\n" "${subfolder_dirs[@]}" | python3 -c '
import random, sys

max_sub = int(sys.argv[1])
seed_str = sys.argv[2] if len(sys.argv) > 2 else ""
seed = int(seed_str) if seed_str != "" else None
rng = random.Random(seed)

items = [line.rstrip("\n") for line in sys.stdin if line.strip() != ""]
if max_sub <= 0 or max_sub >= len(items):
    chosen = items
else:
    chosen = rng.sample(items, max_sub)

for x in chosen:
    print(x)
' "$MAX_SUBFOLDERS" "${SUBFOLDER_SAMPLE_SEED:-}"
    )
    if [[ "${PRINT_SAMPLED_SUBFOLDERS:-0}" == "1" ]]; then
        echo "[evaluate_muse_tracking] Selected subfolders:"
        for d in "${subfolder_dirs[@]}"; do
            echo "  - ${d%/}"
        done
    fi
else
    echo "[evaluate_muse_tracking] Using all subfolders (count=${#subfolder_dirs[@]}), seed ignored."
fi

for subfolder_dir in "${subfolder_dirs[@]}"; do
    subfolder="${subfolder_dir%/}"
    name="${subfolder##*/}"
    if [[ -d "$subfolder" ]]; then
        echo "[evaluate_muse_tracking] Motion: $subfolder"
        HYDRA_FULL_ERROR=1 python scripts/rsl_rl/play.py \
            --num_envs=1 \
            --task=MUSE-Distill-General-Tracking-Flat-G1-v0 \
            --motion "$subfolder" \
            --video_dir_tag "$VIDEO_DIR_TAG" \
            --load_run="$LOAD_RUN" \
            --checkpoint="$CHECKPOINT" \
            --headless \
            --video \
            --video_length="$VIDEO_LENGTH"
        echo "[evaluate_muse_tracking] Done: $name"
    fi
done

echo "[evaluate_muse_tracking] All subfolders finished."

# CUDA_VISIBLE_DEVICES=0 MAX_SUBFOLDERS=10 SUBFOLDER_SAMPLE_SEED=42 bash run/test/evaluate_muse_tracking.sh
