#!/usr/bin/env bash
# Prior rollout: record MP4s only (no prior JSON metrics; faster, cleaner logs).
# Usage: bash run/rollout_prior_video.sh [NUM_OF_PRIOR_ROLLOUTS]
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

TASK="${TASK:-PULSE-Distill-General-Tracking-Flat-G1-v0}"
NUM_ENVS="${NUM_ENVS:-1}"
MOTION_DIR="${MOTION_DIR:-/home/lsn/Datasets/SONIC_npzs/g1/npz_splits_filtered_medium/test}"
LOAD_RUN="${LOAD_RUN:-2026-04-21_21-35-33_kl_0.01_regu_0.005_prior_std_0.5}"
CHECKPOINT="${CHECKPOINT:-model_10000.pt}"
# Parsed from checkpoint basename (e.g. model_20000.pt -> 20000). Videos go under .../prior_videos_model_<steps>/...
_ck_basename="${CHECKPOINT##*/}"
MODEL_STEPS="${_ck_basename#model_}"
MODEL_STEPS="${MODEL_STEPS%.pt}"
VIDEO_DIR_TAG="${VIDEO_DIR_TAG:-model_${MODEL_STEPS}}"
VIDEO_LENGTH="${VIDEO_LENGTH:-400}"

NUM_OF_PRIOR_ROLLOUTS="${1:-${NUM_OF_PRIOR_ROLLOUTS:-1}}"
PRIOR_FALL_MIN_HEIGHT="${PRIOR_FALL_MIN_HEIGHT:-0.35}"
PRIOR_FALL_BODY_NAMES="${PRIOR_FALL_BODY_NAMES:-torso_link}"
# After first fall, wait this many env steps then pause physics (video holds last frame). -1 disables.
PRIOR_FREEZE_AFTER_FALLEN="${PRIOR_FREEZE_AFTER_FALLEN:--1}"
# After freeze: default pad MP4 with ffmpeg (clone last frame) to VIDEO_LENGTH — fast. Set 0 for --no_prior_freeze_pad_tail.
PRIOR_FREEZE_PAD_TAIL="${PRIOR_FREEZE_PAD_TAIL:-1}"
# Only if pad is off: -1 = step until --video_length; 0 = stop immediately (short MP4).
PRIOR_FREEZE_MAX_EXTRA_STEPS="${PRIOR_FREEZE_MAX_EXTRA_STEPS:--1}"
PRIOR_FREEZE_LOG_EVERY="${PRIOR_FREEZE_LOG_EVERY:-50}"

# Set to 1 to overlay motion-command debug (reference vs current) in the video.
PRIOR_VIDEO_SHOW_MOTION_DEBUG_VIS="${PRIOR_VIDEO_SHOW_MOTION_DEBUG_VIS:-0}"
# Set to 1 to print [prior_fall_debug] lines (play.py heuristic; not MDP termination).
PRIOR_DEBUG_FALL="${PRIOR_DEBUG_FALL:-0}"
PRIOR_DEBUG_FALL_EVERY="${PRIOR_DEBUG_FALL_EVERY:-1}"
# 1 = sample z ~ N(μ,σ) from prior (default); 0 = deterministic z = μ (--no_prior_latent_sampling).
PRIOR_LATENT_SAMPLING="${PRIOR_LATENT_SAMPLING:-1}"
# If set, use fixed per-latent-dim std for z (see play.py --prior_rollout_fixed_latent_std).
PRIOR_ROLLOUT_FIXED_LATENT_STD="${PRIOR_ROLLOUT_FIXED_LATENT_STD:-}"

if [[ ! -d "${MOTION_DIR}" ]]; then
    echo "Error: MOTION_DIR not found: ${MOTION_DIR}"
    exit 1
fi

subfolder_dirs=()
for p in "${MOTION_DIR}"/*/; do
    if [[ -d "${p}" ]]; then
        subfolder_dirs+=("${p%/}")
    fi
done

if [[ "${#subfolder_dirs[@]}" -eq 0 ]]; then
    echo "[rollout_prior_video] No motion subfolders under MOTION_DIR; rolling out once with MOTION_DIR directly."
    subfolder_dirs=("${MOTION_DIR%/}")
fi

if [[ "${NUM_OF_PRIOR_ROLLOUTS}" != "0" && "${#subfolder_dirs[@]}" -gt "${NUM_OF_PRIOR_ROLLOUTS}" ]]; then
    echo "[rollout_prior_video] Sampling ${NUM_OF_PRIOR_ROLLOUTS} subfolders from ${#subfolder_dirs[@]} (seed='${SUBFOLDER_SAMPLE_SEED:-<none>}')."
    mapfile -t subfolder_dirs < <(
        printf "%s\n" "${subfolder_dirs[@]}" | python3 -c '
import random, sys

max_sub = int(sys.argv[1])
seed_str = sys.argv[2] if len(sys.argv) > 2 else ""
seed = int(seed_str) if seed_str != "" else None
rng = random.Random(seed)

items = [line.rstrip("\n") for line in sys.stdin if line.strip() != ""]
chosen = rng.sample(items, max_sub)
for x in chosen:
    print(x)
' "${NUM_OF_PRIOR_ROLLOUTS}" "${SUBFOLDER_SAMPLE_SEED:-}"
    )
else
    echo "[rollout_prior_video] Using all discovered subfolders (count=${#subfolder_dirs[@]})."
fi

if [[ "${PRINT_SAMPLED_SUBFOLDERS:-0}" == "1" ]]; then
    echo "[rollout_prior_video] Selected motions:"
    for d in "${subfolder_dirs[@]}"; do
        echo "  - ${d}"
    done
fi

extra_args=()
if [[ "${PRIOR_VIDEO_SHOW_MOTION_DEBUG_VIS}" == "1" ]]; then
    extra_args+=(--prior_video_show_motion_debug_vis)
fi
if [[ "${PRIOR_DEBUG_FALL}" == "1" ]]; then
    extra_args+=(--prior_debug_fall --prior_debug_fall_every "${PRIOR_DEBUG_FALL_EVERY}")
fi
if [[ "${PRIOR_LATENT_SAMPLING}" == "0" ]]; then
    extra_args+=(--no_prior_latent_sampling)
fi
if [[ -n "${PRIOR_ROLLOUT_FIXED_LATENT_STD}" ]]; then
    extra_args+=(--prior_rollout_fixed_latent_std "${PRIOR_ROLLOUT_FIXED_LATENT_STD}")
fi
freeze_args=()
if [[ "${PRIOR_FREEZE_AFTER_FALLEN}" != "-1" ]]; then
    freeze_args+=(--prior_freeze_after_fallen "${PRIOR_FREEZE_AFTER_FALLEN}")
    freeze_args+=(--prior_freeze_max_extra_steps "${PRIOR_FREEZE_MAX_EXTRA_STEPS}")
    freeze_args+=(--prior_freeze_log_every "${PRIOR_FREEZE_LOG_EVERY}")
    if [[ "${PRIOR_FREEZE_PAD_TAIL}" == "0" ]]; then
        freeze_args+=(--no_prior_freeze_pad_tail)
    fi
fi

total="${#subfolder_dirs[@]}"
for ((i = 0; i < total; i++)); do
    motion_subfolder="${subfolder_dirs[$i]}"
    echo "[rollout_prior_video] Running prior video $((i + 1))/${total}: ${motion_subfolder}"
    HYDRA_FULL_ERROR=1 python scripts/rsl_rl/play.py \
        --task "${TASK}" \
        --num_envs "${NUM_ENVS}" \
        --motion "${motion_subfolder}" \
        --load_run "${LOAD_RUN}" \
        --checkpoint "${CHECKPOINT}" \
        --headless \
        --prior_sample \
        --no_prior_metrics \
        --video \
        --video_dir_tag "${VIDEO_DIR_TAG}" \
        --video_length "${VIDEO_LENGTH}" \
        --prior_fall_min_height "${PRIOR_FALL_MIN_HEIGHT}" \
        --prior_fall_body_names "${PRIOR_FALL_BODY_NAMES}" \
        --disable_motion_group_sampling \
        "${freeze_args[@]}" \
        "${extra_args[@]}"
done

# bash run/rollout_prior_video.sh
# PRIOR_LATENT_SAMPLING=0 SUBFOLDER_SAMPLE_SEED=24 PRINT_SAMPLED_SUBFOLDERS=1 bash run/test/rollout_prior_video.sh 5
# PRIOR_LATENT_SAMPLING=0 bash run/rollout_prior_video.sh   # deterministic prior mean (reproducible)
# PRIOR_ROLLOUT_FIXED_LATENT_STD=0.5 bash run/rollout_prior_video.sh   # z ~ N(μ, 0.5²) per dim (not MLP σ)
# PRIOR_VIDEO_SHOW_MOTION_DEBUG_VIS=1 bash run/rollout_prior_video.sh