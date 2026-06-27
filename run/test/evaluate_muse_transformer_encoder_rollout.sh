#!/usr/bin/env bash
set -euo pipefail
# Roll out a MUSE-Transformer-distilled encoder/decoder with the trained masked-goal pathway:
# motion.p_mask is forced to 1.0, so every env's delta_command is zero and motion_anchor_ori_b is
# identity each step (matching the masked-goal frames seen during MUSE training). The MUSE
# goal-mask curriculum is disabled by play.py when --encoder_as_proprio is set on a MUSE-Distill
# task.
#
# Required env vars:
#   LOAD_RUN   — basename under logs/rsl_rl/g1_flat_muse_transformer_distillation/
#   CHECKPOINT — e.g. model_10000.pt
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES, VISUALIZE_DIR, MAX_SUBFOLDERS, SUBFOLDER_SAMPLE_SEED,
#   PRINT_SAMPLED_SUBFOLDERS, VIDEO_LENGTH, ENCODER_AS_PROPRIO_SAMPLE (0/1),
#   RANDOM_INIT_FRAME (0/1, default 1), NO_RESET_BASE_XY_TO_ORIGIN (0/1, default 0)
#
# Example:
#   LOAD_RUN=2026-05-08_19-29-29_muse_transformer_decoder_512_256_128_full_proprio_drop_regu_loss_encoder_192_768 \
#   CHECKPOINT=model_15000.pt \
#   MAX_SUBFOLDERS=5 SUBFOLDER_SAMPLE_SEED=24 \
#   bash run/test/evaluate_muse_transformer_encoder_rollout.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOAD_RUN="${LOAD_RUN:-2026-05-18_20-38-29_muse_transformer_det_cosine_w_0.1}"
CHECKPOINT="${CHECKPOINT:-model_8000.pt}"

VISUALIZE_DIR="${VISUALIZE_DIR:-/home/lsn/Datasets/SONIC_npzs/g1/npz_splits_filtered_medium/test}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
MAX_SUBFOLDERS="${MAX_SUBFOLDERS:-5}"
ENCODER_AS_PROPRIO_SAMPLE="${ENCODER_AS_PROPRIO_SAMPLE:-0}" # 0: no proprio, 1: proprio sample
RANDOM_INIT_FRAME="${RANDOM_INIT_FRAME:-1}"                 # 1: uniform random init frame per env reset
NO_RESET_BASE_XY_TO_ORIGIN="${NO_RESET_BASE_XY_TO_ORIGIN:-0}" # 1: keep sampled frame's world XY (off-camera risk)

VIDEO_DIR_TAG="muse_transformer_encoder_rollout"

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
    echo "[evaluate_muse_transformer_encoder_rollout] Sampling MAX_SUBFOLDERS=${MAX_SUBFOLDERS} from ${#subfolder_dirs[@]} subfolders (seed='${SUBFOLDER_SAMPLE_SEED:-<none>}')."
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
        echo "[evaluate_muse_transformer_encoder_rollout] Selected subfolders:"
        for d in "${subfolder_dirs[@]}"; do
            echo "  - ${d%/}"
        done
    fi
else
    echo "[evaluate_muse_transformer_encoder_rollout] Using all subfolders (count=${#subfolder_dirs[@]}), seed ignored."
fi

EXTRA_ARGS=()
if [[ "$ENCODER_AS_PROPRIO_SAMPLE" == "1" ]]; then
    EXTRA_ARGS+=(--encoder_as_proprio_sample)
fi
if [[ "$RANDOM_INIT_FRAME" == "1" ]]; then
    EXTRA_ARGS+=(--random_init_frame)
fi
if [[ "$NO_RESET_BASE_XY_TO_ORIGIN" == "1" ]]; then
    EXTRA_ARGS+=(--no_reset_base_xy_to_origin)
fi

for subfolder_dir in "${subfolder_dirs[@]}"; do
    subfolder="${subfolder_dir%/}"
    name="${subfolder##*/}"
    if [[ -d "$subfolder" ]]; then
        echo "[evaluate_muse_transformer_encoder_rollout] Motion: $subfolder"
        HYDRA_FULL_ERROR=1 python scripts/rsl_rl/play.py \
            --num_envs=1 \
            --task=MUSE-Transformer-Distill-General-Tracking-Flat-G1-v0 \
            --motion "$subfolder" \
            --video_dir_tag "$VIDEO_DIR_TAG" \
            --load_run="$LOAD_RUN" \
            --checkpoint="$CHECKPOINT" \
            --headless \
            --video \
            --video_length="$VIDEO_LENGTH" \
            --encoder_as_proprio \
            "${EXTRA_ARGS[@]}"
        echo "[evaluate_muse_transformer_encoder_rollout] Done: $name"
    fi
done

echo "[evaluate_muse_transformer_encoder_rollout] All subfolders finished."

# CUDA_VISIBLE_DEVICES=3 MAX_SUBFOLDERS=10 SUBFOLDER_SAMPLE_SEED=42 bash run/test/evaluate_muse_transformer_encoder_rollout.sh
