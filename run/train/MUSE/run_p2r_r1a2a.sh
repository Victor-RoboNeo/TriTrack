#!/usr/bin/env bash
# P2-R R1a-2a: same gate / r_η / 5° / Loco / no terrain as R1a.
# ONLY change: active-only clipped R_E progress. λ_rel=0. done ⇒ r_prog=0.
# Init: model_50000 + zero-init r_η. Do NOT resume R1a.
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
RUN_NAME="${RUN_NAME:-tritrack_p2r_r1a2a_loco_from50000}"
MASTER_PORT="${MASTER_PORT:-29642}"
# +151 → range(50000, 50151) writes model_50150.pt (probes at +50/+100/+150).
MAX_ITERS="${MAX_ITERS:-151}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_headhands_p2r_r1a2}"
# Parent-calibrated: λ_p * σ(r_prog|active) ≈ 22% of canonical per-step scale 0.15.
LAMBDA_P="${LAMBDA_P:-0.2}"
CLIP_P="${CLIP_P:-1.0}"

mkdir -p "$ROOT/logs/tritrack"
LOG="$ROOT/logs/tritrack/p2r_r1a2a_train.log"

echo "[r1a2a] GPUs=$CUDA_VISIBLE_DEVICES nproc=$NPROC envs/rank=$NUM_ENVS"
echo "[r1a2a] resume=$RESUME_CHECKPOINT  (NOT R1a ckpt)"
echo "[r1a2a] experiment=$EXPERIMENT_NAME run_name=$RUN_NAME iters=+$MAX_ITERS"
echo "[r1a2a] same gate / 5° / no scan / Loco / frozen parent"
echo "[r1a2a] r' = r_can + λ_p * 1[active] clip(ΔR_E, -$CLIP_P, $CLIP_P)  λ_p=$LAMBDA_P  λ_rel=0"
echo "[r1a2a] done ⇒ r_prog=0  (no post-reset leak)"
echo "[r1a2a] log=$LOG"

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    scripts/rsl_rl/train.py \
    --task=MUSE-Kp-LatentRL-Kp5-HeadHands-P2R-Loco-G1-v0 \
    --distributed \
    --num_envs="$NUM_ENVS" \
    --max_iterations="$MAX_ITERS" \
    --motion "$MOTION_DIR" \
    --headless \
    --logger tensorboard \
    --experiment_name "$EXPERIMENT_NAME" \
    --run_name "$RUN_NAME" \
    --resume_student_checkpoint "$RESUME_CHECKPOINT" \
    agent.experiment_name="$EXPERIMENT_NAME" \
    agent.policy.adapter=residual \
    agent.policy.intent_recovery=true \
    agent.policy.intent_recovery_s_enabled=false \
    agent.policy.intent_recovery_aux_dim=0 \
    agent.policy.intent_recovery_r_max=0.0875 \
    agent.policy.intent_recovery_r_off=0.6 \
    agent.policy.intent_recovery_r_full=2.0 \
    agent.policy.intent_recovery_persist_on=3 \
    agent.policy.intent_recovery_persist_off=3 \
    agent.policy.terrain_scan_dim=0 \
    agent.policy.latent_std_min=0.04 \
    agent.policy.latent_std_max=0.20 \
    agent.algorithm.entropy_coef=0.001 \
    agent.algorithm.learning_rate=5.0e-5 \
    agent.algorithm.clip_param=0.1 \
    agent.algorithm.desired_kl=0.005 \
    agent.algorithm.critic_warmup_itrs=0 \
    agent.algorithm.parent_residual_anchor_coef=0.0 \
    agent.algorithm.elastic_latent_coef=0.0 \
    agent.algorithm.terrain_zero_coef=0.0 \
    agent.algorithm.terrain_calm_coef=0.0 \
    agent.algorithm.recovery_progress_coef="$LAMBDA_P" \
    agent.algorithm.recovery_progress_clip="$CLIP_P" \
    agent.algorithm.recovery_release_coef=0.0 \
    agent.save_interval=50 \
    agent.reset_noise_std_on_resume=false \
    env.commands.motion.debug_vis=false \
    env.scene.contact_forces.debug_vis=false \
    2>&1 | tee "$LOG"
