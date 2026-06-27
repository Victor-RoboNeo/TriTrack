#!/usr/bin/env bash
set -euo pipefail
# Record one play.py video per motion subfolder under VISUALIZE_DIR (same loop pattern as
# evaluate_test_set_visualize.sh). Anchor + pinned EE keypoints only when using --vr_video_keypoint_vis;
# goal wrists are drawn in motion world frame (see play.py).
#
# Required env vars:
#   LOAD_RUN   — basename under logs/rsl_rl/g1_flat_vr_tracking_residual_latent/
#   CHECKPOINT — e.g. model_5000.pt
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES, VISUALIZE_DIR, MAX_SUBFOLDERS, SUBFOLDER_SAMPLE_SEED, PRINT_SAMPLED_SUBFOLDERS
#   PRIOR_CHECKPOINT, WARMSTART_CHECKPOINT, USE_WARMSTART_LOCK, FIXED_VR_MASK_MODE, VIDEO_LENGTH
#
# Example:
#   LOAD_RUN=2026-05-01_12-00-00_my_run CHECKPOINT=model_5000.pt \
#   PRIOR_CHECKPOINT=logs/rsl_rl/g1_flat_pulse_distillation/.../model_10000.pt \
#   FIXED_VR_MASK_MODE=left_ee bash run/test/evaluate_vr_tracking_visualize.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOAD_RUN="${LOAD_RUN:-2026-05-02_20-08-44_actor_1024_512_256_128_critic_1024_1024_512_512_256_12_critic_warmup_200_critic_priviledged_obs}"
CHECKPOINT="${CHECKPOINT:-model_17000.pt}"

VISUALIZE_DIR="${VISUALIZE_DIR:-/home/lsn/Datasets/SONIC_npzs/g1/npz_splits_filtered_medium/test}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-logs/rsl_rl/g1_flat_pulse_distillation/2026-04-21_21-35-33_kl_0.01_regu_0.005_prior_std_0.5/model_10000.pt}"
FIXED_VR_MASK_MODE="${FIXED_VR_MASK_MODE:-both_ee}"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
MAX_SUBFOLDERS="${MAX_SUBFOLDERS:-10}"

VISUALIZE_DIR_BASENAME="${VISUALIZE_DIR%/}"
VISUALIZE_DIR_BASENAME="${VISUALIZE_DIR_BASENAME##*/}"
VIDEO_DIR_TAG="$(printf "%s" "$VISUALIZE_DIR_BASENAME" | sed -E 's/[^0-9A-Za-z_-]+/_/g')"
if [[ "$VIDEO_DIR_TAG" == *"_"* ]]; then
    VIDEO_DIR_TAG="$(printf "%s" "$VIDEO_DIR_TAG" | tr '[:lower:]' '[:upper:]')"
fi
echo "[evaluate_vr_tracking_visualize] VISUALIZE_DIR_BASENAME=${VISUALIZE_DIR_BASENAME} -> VIDEO_DIR_TAG=${VIDEO_DIR_TAG}"

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
    echo "[evaluate_vr_tracking_visualize] Sampling MAX_SUBFOLDERS=${MAX_SUBFOLDERS} from ${#subfolder_dirs[@]} subfolders (seed='${SUBFOLDER_SAMPLE_SEED:-<none>}')."
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
        echo "[evaluate_vr_tracking_visualize] Selected subfolders:"
        for d in "${subfolder_dirs[@]}"; do
            echo "  - ${d%/}"
        done
    fi
else
    echo "[evaluate_vr_tracking_visualize] Using all subfolders (count=${#subfolder_dirs[@]}), seed ignored."
fi

VR_EXTRA=(--prior_checkpoint "${PRIOR_CHECKPOINT}" --vr_compact_goal_obs false)
if [[ "${USE_WARMSTART_LOCK:-0}" == "1" ]]; then
    VR_EXTRA+=(
        --warmstart_from_masked_partial_kp_tracker
        --warmstart_checkpoint "${WARMSTART_CHECKPOINT}"
        --warmstart_lock_full_normalizer
    )
fi

for subfolder_dir in "${subfolder_dirs[@]}"; do
    subfolder="${subfolder_dir%/}"
    name="${subfolder##*/}"
    if [[ -d "$subfolder" ]]; then
        echo "[evaluate_vr_tracking_visualize] Motion: $subfolder"
        HYDRA_FULL_ERROR=1 python scripts/rsl_rl/play.py \
            --num_envs=1 \
            --task=VR-Tracking-Flat-G1-v0 \
            --motion "$subfolder" \
            --video_dir_tag "$VIDEO_DIR_TAG" \
            --load_run="${LOAD_RUN}" \
            --checkpoint="${CHECKPOINT}" \
            "${VR_EXTRA[@]}" \
            --headless \
            --video \
            --vr_video_keypoint_vis \
            --fixed_vr_mask_mode="${FIXED_VR_MASK_MODE}" \
            --video_length="${VIDEO_LENGTH}"
        echo "[evaluate_vr_tracking_visualize] Done: $name"
    fi
done

echo "[evaluate_vr_tracking_visualize] All subfolders finished."

# CUDA_VISIBLE_DEVICES=0 FIXED_VR_MASK_MODE=left_ee SUBFOLDER_SAMPLE_SEED=42 MAX_SUBFOLDERS=10 bash run/test/evaluate_vr_tracking_visualize.sh
# CUDA_VISIBLE_DEVICES=1 FIXED_VR_MASK_MODE=right_ee SUBFOLDER_SAMPLE_SEED=42 MAX_SUBFOLDERS=10 bash run/test/evaluate_vr_tracking_visualize.sh
# CUDA_VISIBLE_DEVICES=2 FIXED_VR_MASK_MODE=both_ee SUBFOLDER_SAMPLE_SEED=42 MAX_SUBFOLDERS=10 bash run/test/evaluate_vr_tracking_visualize.sh