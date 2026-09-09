#!/usr/bin/env bash
# 3-point (VR torso + L/R wrist) curriculum latent-RL on GPUs 0-3.
#
# Frozen Stage-2 encoder/decoder via --encoder_decoder_warmstart.
# Residual adapter + critic train from scratch (do NOT pass a rough-RL ckpt).
# Actor obs stay 750-D (sparse KP + proprio). No height scan.

set -euo pipefail

ROOT="${ROOT:-/data/home/chenxiangyu/robotics/Anybody}"
cd "$ROOT"

STAGE2_DIR="${STAGE2_DIR:-$ROOT/logs/rsl_rl/g1_flat_muse_kp_latent_distillation/2026-08-22_16-42-45_muse_kp5_latent_distill_102k_teacher10k}"
if [[ -z "${ENCODER_DECODER_WARMSTART:-}" ]]; then
    ENCODER_DECODER_WARMSTART="$(ls -1t "$STAGE2_DIR"/model_*.pt | head -1)"
fi
if [[ ! -f "$ENCODER_DECODER_WARMSTART" ]]; then
    echo "Stage-2 ckpt not found: $ENCODER_DECODER_WARMSTART" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export LD_LIBRARY_PATH="/data/home/chenxiangyu/tools/x11libs/lib:${LD_LIBRARY_PATH:-}"
export GIT_PYTHON_REFRESH="${GIT_PYTHON_REFRESH:-quiet}"
export PATH="${PATH:-}:/usr/bin:/bin"
export ISAACLAB_PATH="${ISAACLAB_PATH:-/data/home/chenxiangyu/robotics/IsaacLab_v2.1}"

NPROC="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
MOTION_DIR="${MOTION_DIR:-$ROOT/datasets/SONIC_npzs/g1/npz_splits_loco_manip/train}"
NUM_ENVS="${NUM_ENVS:-2048}"
RUN_NAME="${RUN_NAME:-tritrack_curriculum_3pt_stage2}"
MASTER_PORT="${MASTER_PORT:-29610}"
MAX_ITERS="${MAX_ITERS:-50000}"

mkdir -p "$ROOT/logs/tritrack"

echo "[curriculum-3pt] GPUs=$CUDA_VISIBLE_DEVICES nproc=$NPROC envs/rank=$NUM_ENVS"
echo "[curriculum-3pt] warmstart=$ENCODER_DECODER_WARMSTART"
echo "[curriculum-3pt] run_name=$RUN_NAME port=$MASTER_PORT"

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    scripts/rsl_rl/train.py \
    --task=MUSE-Kp-LatentRL-Kp5-Curriculum-3pt-G1-v0 \
    --distributed \
    --num_envs="$NUM_ENVS" \
    --max_iterations="$MAX_ITERS" \
    --motion "$MOTION_DIR" \
    --headless \
    --logger tensorboard \
    --experiment_name g1_curriculum_3pt_latent_rl \
    --run_name "$RUN_NAME" \
    --encoder_decoder_warmstart "$ENCODER_DECODER_WARMSTART" \
    agent.policy.adapter=residual \
    agent.algorithm.entropy_coef=0.003 \
    env.commands.motion.debug_vis=false \
    env.scene.contact_forces.debug_vis=false
