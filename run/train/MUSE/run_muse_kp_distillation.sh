#!/usr/bin/env bash
# MUSE-Kp distillation (KP6 recipe + OOD-avoidance mix, 2026-05-16): KP-token student on a
# 6-body G1 keypoint set with a 5-mode mask spec (bernoulli + 4 single-point deploy modes)
# and a 4-phase curriculum, warmstarted from a stable MUSE-Transformer (JC) checkpoint.
#
# Two checkpoints are loaded (each fills a different role):
#   1. TEACHER_CHECKPOINT          = PHC+ stage-1 (mosaic_hybrid). Provides:
#        - actor.* -> self.teacher (MLP)            -> the action-target source
#        - obs_norm_state_dict     -> teacher_obs_normalizer (815-d sonic_55k teacher contract)
#      Required because MUSE-T's saved obs_norm_state_dict is the MUSE student normalizer and
#      would shape-mismatch our teacher_obs_normalizer.
#   2. ENCODER_DECODER_WARMSTART   = MUSE-Transformer .pt. Loaded AFTER (1); only model_state_dict
#      is consumed (no normalizer touched). Warmstarts the shared encoder + decoder; re-loads
#      teacher.* (same PHC+ weights, no-op). The KP front-end (kp_proj on the 1 s per-body
#      input, body_id_emb on 6 bodies) has no counterpart in the JC ckpt and learns fresh.
#
# Architecture:
#   - 6 KP points = mask_modes.KP6_NATIVE_BODIES: pelvis + torso + L/R wrist + L/R ankle.
#     Pelvis is FIRST so the motion-command reset writes the robot root-link reference (fixes
#     the prior root-reset bug; pelvis is a maskable root-pose token).
#   - KP encoder (trainable): per-body token transformer over 6 KP tokens. 0.5 s log-spaced
#     layout (KP_LAYOUT_0_5S, 12 slots: 3 history + 1 abs + 8 future to ~0.5 s; per-body dim
#     12×3=36). Plus H=5 proprio tokens + [CLS]. Per-body spatial masking via key_padding_mask.
#     (Shifted from the 1 s layout 2026-05-17: 0.5 s muse_kp_aux_probe_log05s tracked better;
#      kept in lockstep with rsl_rl_ppo_cfg.py kp_lookahead_steps=12/kp_layout="log_0_5s".)
#   - 5-mode OOD-avoidance spec (mask_modes.muse_kp6_ood_mix_mode_spec): 'bernoulli' (over
#     all 6) + the 4 single-point deploy modes (L wrist, R wrist, torso, pelvis). Phases 1-3
#     stay bernoulli-only (p_see scheduled); phase 4 mixes the single-point modes so the
#     single-point-visible interactive-drag deploy distribution is in-distribution.
#
# Loss: BC vs the frozen PHC+ MLP teacher's action + cosine smoothness on unit-norm μ.
#
# 4-phase curriculum (tracking_env_cfg.py:MUSEKpDistillationCurriculumCfg):
#   - Phase 1 (iter 0..1000):    bernoulli-only, p_see=1.0 — all 6 KP points visible.
#   - Phase 2 (iter 1000..2000): bernoulli-only, p_see ramps 1.0 -> 0.4 (linear).
#   - Phase 3 (iter 2000..5000): bernoulli-only, p_see held at 0.4.
#   - Phase 4 (iter 5000..end):  mask-mode sampling (0.2 each) over {bernoulli (p_see=0.4),
#                                L wrist, R wrist, torso, pelvis}.
# No student resume: JC encoder-decoder warmstart only. A fresh run executes the full
# 4-phase curriculum in order (phases 1-3 = bernoulli warmup over iter 0..5000, then the
# phase-4 mask-mode mix). kp_proj/body_id_emb learn fresh (no KP ckpt to remap).
#
# Required env vars (or fall back to defaults below):
#   TEACHER_CHECKPOINT          PHC+ stage-1 .pt
#   ENCODER_DECODER_WARMSTART   MUSE-Transformer .pt
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES, MOTION_DIR, NUM_ENVS, RUN_NAME, RESUME_CHECKPOINT, FREEZE_MODE
#   WARMUP_FREEZE_ITERS (default 200; 0 disables warmup), WARMUP_FREEZE_MODE (default decoder_only),
#   POST_WARMUP_FREEZE_MODE (default none).
#
# Example:
#   TEACHER_CHECKPOINT=logs/rsl_rl/g1_flat_mosaic_hybrid/<run>/model_75000.pt \
#   ENCODER_DECODER_WARMSTART=logs/rsl_rl/g1_flat_muse_transformer_distillation/<run>/model_3000.pt \
#   bash run/train/run_muse_kp_distillation.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1

TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-logs/rsl_rl/g1_flat_mosaic_hybrid/2026-05-14_14-33-50_sonic_55k_teacher/model_75000.pt}"
# Optional but strongly recommended: a stable MUSE-Transformer .pt to warmstart encoder + decoder.
# Without this the encoder/decoder train from scratch (the teacher MLP still works, just no warmstart).
ENCODER_DECODER_WARMSTART="${ENCODER_DECODER_WARMSTART:-logs/rsl_rl/g1_flat_muse_transformer_distillation/2026-05-14_13-18-56_muse_transformer_det_unitnorm_cosine_w0.1_55k/model_3000.pt}"


MOTION_DIR="${MOTION_DIR:-/home/lsn/Datasets/SONIC_npzs/g1/npz_by_motion/train}"
NUM_ENVS="${NUM_ENVS:-2048}"
RUN_NAME="${RUN_NAME:-muse_kp6_oodmix_0p5s_3phase}"
# No student resume by default: JC encoder-decoder warmstart only (ENCODER_DECODER_WARMSTART);
# kp_proj/body_id_emb learn fresh. Set RESUME_CHECKPOINT=... in the env to opt back in.
RESUME_CHECKPOINT="logs/rsl_rl/g1_flat_muse_kp_distillation/2026-05-17_12-57-01_muse_kp6_oodmix_0p5s_3phase/model_2000.pt"

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    EXTRA_ARGS+=(--resume_student_checkpoint "$RESUME_CHECKPOINT")
fi
if [[ -n "${FREEZE_MODE:-}" ]]; then
    # Hydra-style override into the policy cfg (sets the static, full-run freeze mode).
    EXTRA_ARGS+=(agent.policy.freeze_mode="$FREEZE_MODE")
fi
# Warmup freeze curriculum: pin the policy in WARMUP_FREEZE_MODE for the first
# WARMUP_FREEZE_ITERS PPO iterations so the new KP-input layers can catch up to the warmstarted
# decoder, then unfreeze to POST_WARMUP_FREEZE_MODE and rebuild the optimizer.
# Defaults (set in G1FlatMUSEKpDistillationRunnerCfg.algorithm): 200 iters, decoder_only -> none.
if [[ -n "${WARMUP_FREEZE_ITERS:-}" ]]; then
    EXTRA_ARGS+=(agent.algorithm.warmup_freeze_iters="$WARMUP_FREEZE_ITERS")
fi
if [[ -n "${WARMUP_FREEZE_MODE:-}" ]]; then
    EXTRA_ARGS+=(agent.algorithm.warmup_freeze_mode="$WARMUP_FREEZE_MODE")
fi
if [[ -n "${POST_WARMUP_FREEZE_MODE:-}" ]]; then
    EXTRA_ARGS+=(agent.algorithm.post_warmup_freeze_mode="$POST_WARMUP_FREEZE_MODE")
fi
if [[ -n "${ENCODER_DECODER_WARMSTART:-}" ]]; then
    # Use the CLI flag (handled in cli_args.py via setattr) instead of a Hydra override —
    # IsaacLab's update_class_from_dict is strict-typed and rejects str overrides on a None default.
    EXTRA_ARGS+=(--encoder_decoder_warmstart "$ENCODER_DECODER_WARMSTART")
fi

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/rsl_rl/train.py \
    --task=MUSE-Kp-Distill-General-Tracking-Flat-G1-v0 \
    --distributed \
    --num_envs="$NUM_ENVS" \
    --motion "$MOTION_DIR" \
    --teacher_checkpoint "$TEACHER_CHECKPOINT" \
    --headless \
    --logger wandb \
    --log_project_name MUSE_Distill \
    --run_name "$RUN_NAME" \
    "${EXTRA_ARGS[@]}"

# bash run/train/MUSE/run_muse_kp_distillation.sh

