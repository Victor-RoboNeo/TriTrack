#!/usr/bin/env bash
# Head-always + 0-2 wrists latent-RL, loco-mani clip mix, GPUs 0-3.
#
# Resume a healthy 3-pt residual ckpt (model_35000). Do NOT resume collapsed 39k+.
# Torso (headset) always visible; wrists sampled as 0/1/2; ankles never shown.
# Actor obs stay 750-D. Latent std clamped to stop the previous PPO blow-up.

set -euo pipefail

ROOT="${ROOT:-/data/home/chenxiangyu/robotics/Anybody}"
cd "$ROOT"

CKPT_DIR="${CKPT_DIR:-$ROOT/logs/rsl_rl/g1_flat_muse_kp_latent_rl/2026-08-24_14-24-02_tritrack_curriculum_3pt_stage2}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$CKPT_DIR/model_35000.pt}"
if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
    echo "Resume ckpt not found: $RESUME_CHECKPOINT" >&2
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
RUN_NAME="${RUN_NAME:-tritrack_headhands_locomani_from35000}"
MASTER_PORT="${MASTER_PORT:-29611}"
MAX_ITERS="${MAX_ITERS:-50000}"

mkdir -p "$ROOT/logs/tritrack"

echo "[headhands] GPUs=$CUDA_VISIBLE_DEVICES nproc=$NPROC envs/rank=$NUM_ENVS"
echo "[headhands] resume=$RESUME_CHECKPOINT"
echo "[headhands] run_name=$RUN_NAME port=$MASTER_PORT"

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    scripts/rsl_rl/train.py \
    --task=MUSE-Kp-LatentRL-Kp5-HeadHands-Locomani-G1-v0 \
    --distributed \
    --num_envs="$NUM_ENVS" \
    --max_iterations="$MAX_ITERS" \
    --motion "$MOTION_DIR" \
    --headless \
    --logger tensorboard \
    --run_name "$RUN_NAME" \
    --resume_student_checkpoint "$RESUME_CHECKPOINT" \
    agent.policy.adapter=residual \
    agent.policy.latent_std_min=0.04 \
    agent.policy.latent_std_max=0.20 \
    agent.algorithm.entropy_coef=0.001 \
    agent.reset_noise_std_on_resume=true \
    env.commands.motion.debug_vis=false \
    env.scene.contact_forces.debug_vis=false
