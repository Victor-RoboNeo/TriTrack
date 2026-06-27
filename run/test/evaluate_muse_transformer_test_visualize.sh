#!/usr/bin/env bash
set -euo pipefail

# Visualize the JC MUSE-Transformer teacher tracking on a motion test set: one
# video per sampled motion subfolder, EVERY timestep visible, rendering ONLY the
# 5 points of interest (torso + L/R wrist + L/R ankle) as world-frame dot markers
# (green = robot, red = goal). The anchor frame triads are hidden and the goal is
# forced fully visible via --poi5_dot_vis (drops the goal-mask curriculum + pins
# motion.p_mask=0.0).
#
# Modeled on run/test/evaluate_test_set_visualize.sh.
#
# Override via environment variables:
#   CUDA_VISIBLE_DEVICES        (default: 0)
#   VISUALIZE_DIR, LOAD_RUN, CHECKPOINT, VIDEO_LENGTH
#   MAX_SUBFOLDERS              (default: 10; 0 = unlimited)
#   SUBFOLDER_SAMPLE_SEED       (optional; deterministic sampling)
#   PRINT_SAMPLED_SUBFOLDERS    (0/1)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

VISUALIZE_DIR="${VISUALIZE_DIR:-/home/lsn/Datasets/SONIC_npzs/g1/npz_by_motion/test}"
LOAD_RUN="${LOAD_RUN:-2026-05-18_20-38-29_muse_transformer_det_cosine_w_0.1}"
CHECKPOINT="${CHECKPOINT:-model_10000.pt}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
MAX_SUBFOLDERS="${MAX_SUBFOLDERS:-10}"

VIDEO_DIR_TAG="muse_transformer_poi5_test"
TASK="MUSE-Transformer-Distill-General-Tracking-Flat-G1-v0"

if [[ ! -d "$VISUALIZE_DIR" ]]; then
    echo "Error: $VISUALIZE_DIR not found." >&2
    exit 1
fi

# Collect subfolders first, then sample randomly (instead of filesystem order).
subfolder_dirs=()
for p in "$VISUALIZE_DIR"/*/; do
    if [[ -d "$p" ]]; then
        subfolder_dirs+=("$p")
    fi
done

if [[ "${MAX_SUBFOLDERS}" != "0" && "${#subfolder_dirs[@]}" -gt "${MAX_SUBFOLDERS}" ]]; then
    echo "[evaluate_muse_transformer_test_visualize] Sampling MAX_SUBFOLDERS=${MAX_SUBFOLDERS} from ${#subfolder_dirs[@]} subfolders (seed='${SUBFOLDER_SAMPLE_SEED:-<none>}')."
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
        echo "[evaluate_muse_transformer_test_visualize] Selected subfolders:"
        for d in "${subfolder_dirs[@]}"; do
            echo "  - ${d%/}"
        done
    fi
else
    echo "[evaluate_muse_transformer_test_visualize] Using all subfolders (count=${#subfolder_dirs[@]}), seed ignored."
fi

for subfolder_dir in "${subfolder_dirs[@]}"; do
    subfolder="${subfolder_dir%/}"
    name="${subfolder##*/}"
    if [[ -d "$subfolder" ]]; then
        echo "[evaluate_muse_transformer_test_visualize] Motion: $subfolder"
        HYDRA_FULL_ERROR=1 python scripts/rsl_rl/play.py \
            --num_envs=1 \
            --task="$TASK" \
            --motion "$subfolder" \
            --video_dir_tag "$VIDEO_DIR_TAG" \
            --load_run="$LOAD_RUN" \
            --checkpoint="$CHECKPOINT" \
            --headless \
            --video \
            --poi5_dot_vis \
            --video_length="$VIDEO_LENGTH"
        echo "[evaluate_muse_transformer_test_visualize] Done: $name"
    fi
done

echo "[evaluate_muse_transformer_test_visualize] All subfolders finished."
echo "  videos -> logs/rsl_rl/g1_flat_muse_transformer_distillation/${LOAD_RUN}/videos/${VIDEO_DIR_TAG}/"

# CUDA_VISIBLE_DEVICES=1 MAX_SUBFOLDERS=5 SUBFOLDER_SAMPLE_SEED=42 PRINT_SAMPLED_SUBFOLDERS=1 bash run/test/evaluate_muse_transformer_test_visualize.sh
