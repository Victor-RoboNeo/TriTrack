#!/usr/bin/env bash
# Run evaluate once per motion subfolder under MOSAIC_Dataset/visualize (e.g. dance, in_place_motions, ...)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

VISUALIZE_DIR="${VISUALIZE_DIR:-/home/lsn/Datasets/bones-seed/g1/npz_splits/train}" # "MOSAIC_Dataset/visualize"

VISUALIZE_DIR_BASENAME="${VISUALIZE_DIR%/}"
VISUALIZE_DIR_BASENAME="${VISUALIZE_DIR_BASENAME##*/}"
# Sanitized tag used to name the video root folder, e.g. videos_test or videos_MOSAIC_DATASET.
# Rules:
# - replace invalid chars with "_"
# - if it contains "_", uppercase the whole tag (e.g. MOSAIC_Dataset -> MOSAIC_DATASET)
VIDEO_DIR_TAG="$(printf "%s" "$VISUALIZE_DIR_BASENAME" | sed -E 's/[^0-9A-Za-z_-]+/_/g')"
if [[ "$VIDEO_DIR_TAG" == *"_"* ]]; then
    VIDEO_DIR_TAG="$(printf "%s" "$VIDEO_DIR_TAG" | tr '[:lower:]' '[:upper:]')"
fi
echo "[evaluate_multiple] VISUALIZE_DIR_BASENAME=${VISUALIZE_DIR_BASENAME} -> VIDEO_DIR_TAG=${VIDEO_DIR_TAG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR" || exit 1

if [[ ! -d "$VISUALIZE_DIR" ]]; then
    echo "Error: $VISUALIZE_DIR not found."
    exit 1
fi

MAX_SUBFOLDERS="${MAX_SUBFOLDERS:-10}" # Set to 0 for unlimited.

# Collect subfolders first, then sample randomly (instead of filesystem order).
subfolder_dirs=()
for p in "$VISUALIZE_DIR"/*/; do
    if [[ -d "$p" ]]; then
        subfolder_dirs+=("$p")
    fi
done

if [[ "${MAX_SUBFOLDERS}" != "0" && "${#subfolder_dirs[@]}" -gt "${MAX_SUBFOLDERS}" ]]; then
    # Optional deterministic sampling:
    #   SUBFOLDER_SAMPLE_SEED=123 MAX_SUBFOLDERS=10 bash run/evaluate_multiple.sh
    echo "[evaluate_multiple] Sampling MAX_SUBFOLDERS=${MAX_SUBFOLDERS} from ${#subfolder_dirs[@]} subfolders (seed='${SUBFOLDER_SAMPLE_SEED:-<none>}')."
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
        echo "[evaluate_multiple] Selected subfolders:"
        for d in "${subfolder_dirs[@]}"; do
            echo "  - ${d%/}"
        done
    fi
else
    echo "[evaluate_multiple] Using all subfolders (count=${#subfolder_dirs[@]}), seed ignored."
fi

tested=0

for subfolder_dir in "${subfolder_dirs[@]}"; do
    subfolder="${subfolder_dir%/}"
    name="${subfolder##*/}"
    if [[ -d "$subfolder" ]]; then
        echo "[evaluate_multiple] Running with motion: $subfolder"
        HYDRA_FULL_ERROR=1 python scripts/rsl_rl/play.py \
            --num_envs=1 \
            --task=Partial-Masked-Residual-Latent-Distill-2B-General-Tracking-Flat-G1-v0 \
            --motion "$subfolder" \
            --video_dir_tag "$VIDEO_DIR_TAG" \
            --load_run=2026-04-24_11-34-39_simple_masking \
            --checkpoint=model_4000.pt \
            --headless \
            --video \
            --partial_2b_video_keypoint_vis \
            --fixed_mask_mode=upper_end_effector \
            --video_length=200
        echo "[evaluate_multiple] Done: $name"
        tested=$((tested + 1))
        # Sampling already enforces MAX_SUBFOLDERS, so no further truncation here.
    fi
done

echo "[evaluate_multiple] All subfolders finished."

# CUDA_VISIBLE_DEVICES=7 VISUALIZE_DIR="/home/lsn/Datasets/SONIC_npzs/g1/npz_splits_filtered_medium/test" MAX_SUBFOLDERS=3 SUBFOLDER_SAMPLE_SEED=24 PRINT_SAMPLED_SUBFOLDERS=1 bash run/evaluate_multiple.sh
# CUDA_VISIBLE_DEVICES=7 VISUALIZE_DIR="/home/lsn/Datasets/SONIC_npzs/g1/npz_splits_filtered_small_walk_jog/train" MAX_SUBFOLDERS=10 SUBFOLDER_SAMPLE_SEED=24 PRINT_SAMPLED_SUBFOLDERS=1 bash run/evaluate_multiple.sh
# CUDA_VISIBLE_DEVICES=7 VISUALIZE_DIR="/home/lsn/MOSAIC/motion_for_debug" MAX_SUBFOLDERS=1 SUBFOLDER_SAMPLE_SEED=24 PRINT_SAMPLED_SUBFOLDERS=1 bash run/evaluate_multiple.sh