#!/usr/bin/env bash
# MUSE-Kp LATENT-space distillation (2026-05-18): distil a KP-token student in LATENT space
# against a near-perfect joint-command (JC) MUSE-Transformer teacher, with the shared decoder
# FROZEN. Unlike run_muse_kp_distillation.sh (action-space BC vs a frozen MLP, encoder+decoder
# co-tuned):
#
#   - Teacher (frozen): the JC MUSE-T encoder → μ_jc, fed the full UNMASKED JC obs (privileged).
#   - Student (trainable): the KP encoder → μ_kp, fed masked/partial KP obs.
#   - Decoder (frozen, shared, loaded from the JC ckpt): action = decoder([μ, proprio]).
#   - Loss = 1.0·MSE(μ_kp, sg(μ_jc)) [raw μ — JC recipe is latent_normalize=False]
#          + 0.1·MSE(decoder(μ_kp), sg(action_jc))   [behavior anchor, frozen shared decoder]
#
# SINGLE checkpoint contract: TEACHER_CHECKPOINT = the JC MUSE-Transformer .pt. The runner loads
# it into ALL of: the frozen jc_encoder (teacher μ_jc) + the frozen shared decoder + a warmstart
# of the KP backbone (kp_proj/body_id_emb learn fresh) + teacher_obs_normalizer (frozen) + the
# frozen student proprio-normalizer slice (copied at the JC-goal offset so the frozen decoder
# always sees proprio in its trained distribution). The KP/mask obs prefix normalizes online.
# No --encoder_decoder_warmstart needed.
#
# Schedule:
#   - Teacher-pilot warmup: JC teacher drives the env for the first 200 iters (on-distribution
#     states while the fresh KP front-end catches up), then a hard switch to student-pilot.
#   - Freeze warmup (aligned, 200 iters): decoder + warmstarted backbone frozen, only
#     kp_proj/body_id_emb train; then decoder stays frozen forever, the full KP encoder trains.
#   - Curriculum/obs: canonical KP6 0.5 s log-spaced, 5-mode OOD-mix, 4-phase (inherited).
#
# Required env vars (or fall back to defaults below):
#   TEACHER_CHECKPOINT   the JC MUSE-Transformer .pt (latent_normalize=False recipe)
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES, MOTION_DIR, NUM_ENVS, RUN_NAME, RESUME_CHECKPOINT
#
# Example:
#   TEACHER_CHECKPOINT=logs/rsl_rl/g1_flat_muse_transformer_distillation/<run>/model_3000.pt \
#   bash run/train/MUSE/run_muse_kp_latent_distillation.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1

# The near-perfect JC MUSE-Transformer (deterministic encoder, raw latent: latent_normalize=False).
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-logs/rsl_rl/g1_flat_muse_transformer_distillation/2026-05-21_19-12-17_muse_transformer_det_cosine_w_0.1_102k_no_mask/model_10000.pt}"

MOTION_DIR="${MOTION_DIR:-/home/lsn/Datasets/SONIC_npzs/g1/npz_splits_loco_manip/train}"
NUM_ENVS="${NUM_ENVS:-1536}"
RUN_NAME="${RUN_NAME:-muse_kp5_latent_distill_102k_demo_modes}"
# No student resume by default: a fresh run runs the full 4-phase curriculum + warmups.
# Set RESUME_CHECKPOINT=... to opt back in.
#RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
RESUME_CHECKPOINT="logs/rsl_rl/g1_flat_muse_kp_latent_distillation/2026-05-22_13-03-02_muse_kp5_latent_distill_102k/model_7000.pt"

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    EXTRA_ARGS+=(--resume_student_checkpoint "$RESUME_CHECKPOINT")
fi

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/rsl_rl/train.py \
    --task=MUSE-Kp-LatentDistill-General-Tracking-Flat-G1-v0 \
    --distributed \
    --num_envs="$NUM_ENVS" \
    --motion "$MOTION_DIR" \
    --teacher_checkpoint "$TEACHER_CHECKPOINT" \
    --headless \
    --logger wandb \
    --log_project_name MUSE_Distill \
    --run_name "$RUN_NAME" \
    "${EXTRA_ARGS[@]}"

# bash run/train/MUSE/run_muse_kp_latent_distillation.sh
