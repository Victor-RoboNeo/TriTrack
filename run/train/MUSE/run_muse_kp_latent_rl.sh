#!/usr/bin/env bash
# MUSE-Kp LATENT-SPACE RL FINETUNE (M1: adapter=full_ft, unanchored).
#
# PPO over the distilled MUSE-Kp *latent* — the frozen decoder is a motor prior
# (ASE/PULSE lineage). The RL action IS the latent; the runner decodes
# latent->joint via the frozen decoder before env.step (training_type=latent_rl).
# See docs/latent_rl_finetune_plan.md for the full design + decisions.
#
# ONE checkpoint is loaded (no teacher / no BC — this is RL, not distillation):
#   ENCODER_DECODER_WARMSTART = the distilled MUSE-Kp .pt. The policy's loader
#   takes the full MUSE-Kp model_state_dict (encoder + decoder + frozen teacher
#   slot + slot-aware kp_proj remap). Without it RL starts from random init.
#
# Reward (decision D5): world-frame position accuracy of the VISIBLE points of
# interest (mdp.motion_visible_kp_position_error_exp_world), swapped in for the
# anchor-frame body-pos term. All stability/regularization/anchor scaffolding
# weights kept unchanged (decision D5a; POI-vs-stability ratio is tunable).
#
# Encoder/decoder shape (KP6, 0.5 s log-spaced) MUST match the warmstart ckpt;
# kept in lockstep with G1FlatMUSEKpDistillationRunnerCfg.
#
# Required env vars (or fall back to defaults below):
#   ENCODER_DECODER_WARMSTART   distilled MUSE-Kp .pt (point at a concrete model_<N>.pt)
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES, MOTION_DIR, NUM_ENVS, RUN_NAME, RESUME_CHECKPOINT
#   ADAPTER (full_ft default | lora | residual[M4]), INIT_LATENT_STD,
#   CRITIC_WARMUP_ITRS, LEARNING_RATE,
#   PRIOR_ANCHOR_COEF (D3: 0=unanchored default, e.g. 0.05=anchored full_ft),
#   LATENTRL_REF_W (2026-05-23 grace fix: 0=off default, e.g. 0.1=weak full-body
#     reference so unwatched legs/torso stay graceful; locomotion task only),
#   LORA_RANK / LORA_ALPHA / LORA_TARGETS (ADAPTER=lora; e.g. RANK=8,
#     TARGETS='[attn_qkv,attn_out,mu_head]')
#
# Example:
#   ENCODER_DECODER_WARMSTART=logs/rsl_rl/g1_flat_muse_kp_distillation/2026-05-17_09-54-11_muse_kp6_oodmix_0p5s_3phase/model_5000.pt \
#   bash run/train/MUSE/run_muse_kp_latent_rl.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=5

# Distilled MUSE-Kp checkpoint to finetune (decision: the KP6 OOD-mix 0.5s 3-phase run).
# Point at a concrete model_<N>.pt; the runner raises a clear error if it doesn't exist.
ENCODER_DECODER_WARMSTART="${ENCODER_DECODER_WARMSTART:-logs/rsl_rl/g1_flat_muse_kp_latent_distillation/2026-05-22_21-41-39_muse_kp5_latent_distill_102k_demo_modes/model_14000.pt}"

MOTION_DIR="${MOTION_DIR:-/home/lsn/Datasets/SONIC_npzs/g1/npz_splits_loco_manip/train}"
NUM_ENVS="${NUM_ENVS:-4096}"
RUN_NAME="${RUN_NAME:-muse_kp_latentrl_residual_rfw_0.125}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# --- Weak full-body reference bundle (2026-05-23 grace fix; default OFF) ------
# Master strength for the optional posture+gait reference reward in
# LatentRLRewardsCfg. Re-densifies toward the distillation reward so the UNWATCHED
# legs/torso stay anchored to the graceful reference instead of stumbling to chase
# POI accuracy. Read from the env at import time (no Hydra plumbing needed); 0.0
# reproduces the current sparse-POI run exactly. Sweep e.g. 0.05 / 0.1 / 0.2.
# NOTE: affects ONLY this locomotion task — the wrist-writing reward is a separate
# class (LatentRLKp5WritingRewardsCfg) and is intentionally untouched.
export LATENTRL_REF_W="${LATENTRL_REF_W:-0.125}"

ADAPTER="residual"

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    EXTRA_ARGS+=(--resume_student_checkpoint "$RESUME_CHECKPOINT")
fi
if [[ -n "${ADAPTER:-}" ]]; then
    EXTRA_ARGS+=(agent.policy.adapter="$ADAPTER")
fi
# LoRA knobs (M3) — only meaningful with ADAPTER=lora. lora_targets is a list;
# override via Hydra list syntax if needed, e.g. LORA_TARGETS='[attn_qkv,mu_head]'.
if [[ -n "${LORA_RANK:-}" ]]; then
    EXTRA_ARGS+=(agent.policy.lora_rank="$LORA_RANK")
fi
if [[ -n "${LORA_ALPHA:-}" ]]; then
    EXTRA_ARGS+=(agent.policy.lora_alpha="$LORA_ALPHA")
fi
if [[ -n "${LORA_TARGETS:-}" ]]; then
    EXTRA_ARGS+=(agent.policy.lora_targets="$LORA_TARGETS")
fi
if [[ -n "${INIT_LATENT_STD:-}" ]]; then
    EXTRA_ARGS+=(agent.policy.init_latent_std="$INIT_LATENT_STD")
fi
if [[ -n "${CRITIC_WARMUP_ITRS:-}" ]]; then
    EXTRA_ARGS+=(agent.algorithm.critic_warmup_itrs="$CRITIC_WARMUP_ITRS")
fi
# Decision D3: run BOTH the unanchored (default, PRIOR_ANCHOR_COEF unset/0) and
# the anchored (e.g. PRIOR_ANCHOR_COEF=0.05) full_ft variants for an honest
# safe-start comparison vs LoRA/residual.
if [[ -n "${PRIOR_ANCHOR_COEF:-}" ]]; then
    EXTRA_ARGS+=(agent.algorithm.prior_anchor_coef="$PRIOR_ANCHOR_COEF")
fi
if [[ -n "${LEARNING_RATE:-}" ]]; then
    EXTRA_ARGS+=(agent.algorithm.learning_rate="$LEARNING_RATE")
fi
if [[ -n "${ENCODER_DECODER_WARMSTART:-}" ]]; then
    # CLI flag (cli_args.py setattr) — IsaacLab's update_class_from_dict is strict-typed
    # and rejects str overrides on a None-default field via Hydra.
    EXTRA_ARGS+=(--encoder_decoder_warmstart "$ENCODER_DECODER_WARMSTART")
fi

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/rsl_rl/train.py \
    --task=MUSE-Kp-LatentRL-General-Tracking-Flat-G1-v0 \
    --distributed \
    --num_envs="$NUM_ENVS" \
    --motion "$MOTION_DIR" \
    --headless \
    --logger wandb \
    --log_project_name AnyBody_LatentRL \
    --run_name "$RUN_NAME" \
    "${EXTRA_ARGS[@]}"

# bash run/train/MUSE/run_muse_kp_latent_rl.sh
