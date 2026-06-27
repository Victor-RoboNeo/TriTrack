#!/usr/bin/env bash
# KP5 LATENT-RL FINETUNE — OBSTACLE-REACH SPECIALIZATION (split specialist per phase).
#
# PPO over the distilled MUSE-Kp KP5 *latent* (frozen decoder = motor prior): the right
# wrist must reach a fixed point that an obstacle blocks. The robot starts ~1 m from the
# station and walks in (mask pinned right-wrist-only, inherited from the writing env).
#
# Phases (set PHASE): 0 free reach · 1 above a box · 2 into an open container · 4 under a slab.
# Each phase is a SEPARATE specialist — one run per phase, pointed at that phase's clip pool.
#
# Prereq: generate the phase pool first (constant-wrist clips + obstacle sidecars):
#   for p in 0 1 2 4; do python scripts/gen_obstacle_clips.py --phase $p --num-clips 256; done
#
# Reward / termination / adapter come from the env + runner cfg (writing reach reward +
# OBB keep-out; fall/time terminations; residual adapter). Tune via Hydra, e.g.
#   env.rewards.obstacle_keepout.weight=-20  env.rewards.poi_pos.weight=...
#
# Common overrides: CUDA_VISIBLE_DEVICES, PHASE, MOTION_DIR, NUM_ENVS, RUN_NAME,
#   ENCODER_DECODER_WARMSTART, RESUME_CHECKPOINT, ADAPTER, INIT_LATENT_STD,
#   CRITIC_WARMUP_ITRS, LEARNING_RATE
#
# Example:
#   PHASE=1 ENCODER_DECODER_WARMSTART=logs/.../muse_kp5_latent_distill/model_14000.pt \
#   bash run/train/MUSE/run_muse_kp_latent_rl_obstacle_reach.sh

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../../.. && pwd)"   # repo root

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# The distilled KP5 policy to finetune (same warmstart as the writing specialist).
ENCODER_DECODER_WARMSTART="${ENCODER_DECODER_WARMSTART:-logs/rsl_rl/g1_flat_muse_kp_latent_distillation/2026-05-22_21-41-39_muse_kp5_latent_distill_102k_demo_modes/model_18500.pt}"

PHASE="${PHASE:-1}" # 0,1,2,4
export OBSTACLE_REACH_PHASE="$PHASE"   # the command samples this phase FRESH every reset
NUM_ENVS="${NUM_ENVS:-4096}"
RUN_NAME="${RUN_NAME:-muse_kp5_latentrl_obstacle_phase${PHASE}}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
ADAPTER="${ADAPTER:-residual}"
PY="${PY:-python}"

# Per-reset sampling needs only a SINGLE standing-seed clip: it supplies the reset pose +
# the standing pose of the non-wrist bodies; the reach target P + obstacles are sampled per
# reset (NOT from any clip). Shared across phases — generated once.
SEED_DIR="$HERE/outputs/obstacle_seed/phase0"
if [[ ! -d "$SEED_DIR" || -z "$(ls "$SEED_DIR"/*.npz 2>/dev/null)" ]]; then
    echo "[obstacle_reach] generating standing-seed clip (one-time)…"
    "$PY" scripts/gen_obstacle_clips.py --phase 0 --num-clips 1 --duration 8 \
        --out-root "$HERE/outputs/obstacle_seed"
fi
MOTION_DIR="${MOTION_DIR:-$SEED_DIR}"

IFS=',' read -r -a _gpu_arr <<<"$CUDA_VISIBLE_DEVICES"
NPROC="${#_gpu_arr[@]}"

# Only go distributed for REAL multi-GPU (NPROC>1). On a single GPU, --distributed still inits
# a process group + runs the "Synchronizing parameters" broadcast (a no-op at world_size=1) —
# pure overhead that also obscures the (silent) first-rollout startup.
DIST_FLAG=()
if [[ "$NPROC" -gt 1 ]]; then
    DIST_FLAG=(--distributed)
    # Multi-GPU NCCL on this box HANGS at the first collective (the "Synchronizing parameters"
    # broadcast): IOMMU is enabled (NCCL direct-P2P stalls under IOMMU) and two Isaac procs
    # contend on Omniverse caches. Route NCCL through host memory (P2P/IB off) so it completes
    # — small cost on the param all-reduce only. Pre-set these to override.
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
fi

EXTRA_ARGS=()
[[ -n "${RESUME_CHECKPOINT:-}" ]] && EXTRA_ARGS+=(--resume_student_checkpoint "$RESUME_CHECKPOINT")
[[ -n "${ADAPTER:-}" ]] && EXTRA_ARGS+=(agent.policy.adapter="$ADAPTER")
[[ -n "${INIT_LATENT_STD:-}" ]] && EXTRA_ARGS+=(agent.policy.init_latent_std="$INIT_LATENT_STD")
[[ -n "${CRITIC_WARMUP_ITRS:-}" ]] && EXTRA_ARGS+=(agent.algorithm.critic_warmup_itrs="$CRITIC_WARMUP_ITRS")
[[ -n "${LEARNING_RATE:-}" ]] && EXTRA_ARGS+=(agent.algorithm.learning_rate="$LEARNING_RATE")
[[ -n "${ENCODER_DECODER_WARMSTART:-}" ]] && EXTRA_ARGS+=(--encoder_decoder_warmstart "$ENCODER_DECODER_WARMSTART")

echo "[obstacle_reach] phase=$PHASE motion=$MOTION_DIR run=$RUN_NAME gpus=$CUDA_VISIBLE_DEVICES"
HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/rsl_rl/train.py \
    --task=MUSE-Kp-LatentRL-Kp5-ObstacleReach-General-Tracking-Flat-G1-v0 \
    "${DIST_FLAG[@]}" \
    --num_envs="$NUM_ENVS" \
    --motion "$MOTION_DIR" \
    --headless \
    --logger wandb \
    --log_project_name Obstacle_RL \
    --run_name "$RUN_NAME" \
    "${EXTRA_ARGS[@]}"

# PHASE=0 CUDA_VISIBLE_DEVICES=1 bash run/train/MUSE/run_muse_kp_latent_rl_obstacle_reach.sh
# PHASE=1 CUDA_VISIBLE_DEVICES=0 bash run/train/MUSE/run_muse_kp_latent_rl_obstacle_reach.sh
# PHASE=2 CUDA_VISIBLE_DEVICES=2 bash run/train/MUSE/run_muse_kp_latent_rl_obstacle_reach.sh
# PHASE=4 CUDA_VISIBLE_DEVICES=3 bash run/train/MUSE/run_muse_kp_latent_rl_obstacle_reach.sh