#!/usr/bin/env bash
# P2-B: elastic latent trust region from model_50000.
# Same mix / lr / PPO as P2-A. Replaces hard parent L2 with
# terrain-conditioned spring-damper on z_exec (dead-zone + damping).
# Do NOT resume 50500 or P2-A ckpts. Do NOT change orientation reward.
set -euo pipefail

ROOT="${ROOT:-/data/home/chenxiangyu/robotics/Anybody}"
cd "$ROOT"

source /data/home/chenxiangyu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT

CKPT_DIR="${CKPT_DIR:-$ROOT/logs/rsl_rl/g1_flat_muse_kp_latent_rl/2026-08-26_00-38-10_tritrack_headhands_locomani_from35000}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$CKPT_DIR/model_50000.pt}"
if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
    echo "Resume ckpt not found: $RESUME_CHECKPOINT" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export LD_LIBRARY_PATH="/data/home/chenxiangyu/tools/x11libs/lib:${LD_LIBRARY_PATH:-}"
export GIT_PYTHON_REFRESH="${GIT_PYTHON_REFRESH:-quiet}"
export PATH="${PATH:-}:/usr/bin:/bin"
export ISAACLAB_PATH="${ISAACLAB_PATH:-/data/home/chenxiangyu/robotics/IsaacLab_v2.1}"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
export HOME=/data/home/chenxiangyu

NPROC="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
MOTION_DIR="${MOTION_DIR:-$ROOT/datasets/SONIC_npzs/g1/npz_splits_loco_manip/train}"
NUM_ENVS="${NUM_ENVS:-2048}"
RUN_NAME="${RUN_NAME:-tritrack_p2b_elastic_from50000}"
MASTER_PORT="${MASTER_PORT:-29632}"
MAX_ITERS="${MAX_ITERS:-8000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_headhands_p2_gphi}"

mkdir -p "$ROOT/logs/tritrack"
LOG="$ROOT/logs/tritrack/p2b_train.log"

echo "[p2b] GPUs=$CUDA_VISIBLE_DEVICES nproc=$NPROC envs/rank=$NUM_ENVS"
echo "[p2b] resume=$RESUME_CHECKPOINT"
echo "[p2b] experiment=$EXPERIMENT_NAME run_name=$RUN_NAME iters=+$MAX_ITERS"
echo "[p2b] PPO clip=0.1 desired_kl=0.005  hard_anchor=0  elastic λs=1.0 λd=0.05"
echo "[p2b] θ_free 10°/25°  k 1.0/0.25  save_interval=250 lr=5e-5"
echo "[p2b] log=$LOG"

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    scripts/rsl_rl/train.py \
    --task=MUSE-Kp-LatentRL-Kp5-HeadHands-P2-G1-v0 \
    --distributed \
    --num_envs="$NUM_ENVS" \
    --max_iterations="$MAX_ITERS" \
    --motion "$MOTION_DIR" \
    --headless \
    --logger tensorboard \
    --experiment_name "$EXPERIMENT_NAME" \
    --run_name "$RUN_NAME" \
    --resume_student_checkpoint "$RESUME_CHECKPOINT" \
    agent.policy.adapter=residual \
    agent.policy.latent_std_min=0.04 \
    agent.policy.latent_std_max=0.20 \
    agent.algorithm.entropy_coef=0.001 \
    agent.algorithm.learning_rate=5.0e-5 \
    agent.algorithm.clip_param=0.1 \
    agent.algorithm.desired_kl=0.005 \
    agent.algorithm.parent_residual_anchor_coef=0.0 \
    agent.algorithm.elastic_latent_coef=1.0 \
    agent.algorithm.elastic_damp_coef=0.05 \
    agent.algorithm.elastic_theta_easy_deg=10.0 \
    agent.algorithm.elastic_theta_hard_deg=25.0 \
    agent.algorithm.elastic_k_easy=1.0 \
    agent.algorithm.elastic_k_hard=0.25 \
    agent.save_interval=250 \
    agent.reset_noise_std_on_resume=false \
    env.commands.motion.debug_vis=false \
    env.scene.contact_forces.debug_vis=false \
    2>&1 | tee "$LOG"
