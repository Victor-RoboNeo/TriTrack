#!/usr/bin/env bash
# Run prior-sampling rollout once per motion subfolder under MOSAIC_Dataset/visualize (e.g. dance, walk, ...).
# Uses z ~ R(proprio), decode to action (no encoder).

VISUALIZE_DIR="MOSAIC_Dataset/visualize"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR" || exit 1

if [[ ! -d "$VISUALIZE_DIR" ]]; then
    echo "Error: $VISUALIZE_DIR not found."
    exit 1
fi

for subfolder in "$VISUALIZE_DIR"/*/; do
    subfolder="${subfolder%/}"
    name="${subfolder##*/}"
    if [[ -d "$subfolder" ]]; then
        echo "[rollout_prior_multiple] Running with motion: $subfolder"
        HYDRA_FULL_ERROR=1 python scripts/rsl_rl/play.py \
            --num_envs=1 \
            --task=PULSE-Distill-General-Tracking-Flat-G1-v0 \
            --motion "$subfolder" \
            --load_run=2026-03-28_08-13-44_PULSE_sonic_data \
            --checkpoint=model_6000.pt \
            --headless \
            --prior_sample \
            --video \
            --video_length=1000 \
            --disable_motion_group_sampling
        echo "[rollout_prior_multiple] Done: $name"
    fi
done

echo "[rollout_prior_multiple] All subfolders finished."

# bash run/rollout_prior_multiple.sh
