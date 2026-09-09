#!/usr/bin/env bash
# Stage 2 MUSE-Kp latent distillation with official Anybody flags, local paths.
# Official script hardcodes an unpublished student resume; that flag is omitted here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-$ROOT/logs/rsl_rl/g1_flat_muse_transformer_distillation/2026-08-22_08-27-02_muse_transformer_det_cosine_w0.1_102k_teacher100k/model_10000.pt}"
MOTION_DIR="${MOTION_DIR:-$ROOT/datasets/SONIC_npzs/g1/npz_splits_loco_manip/train}"
RUN_NAME="${RUN_NAME:-muse_kp5_latent_distill_102k_teacher10k}"
NUM_ENVS="${NUM_ENVS:-1536}"
NPROC="${NPROC:-2}"
LOG_PROJECT="${LOG_PROJECT:-MUSE_Distill}"
MASTER_PORT="${MASTER_PORT:-29600}"

if [[ ! -f "$TEACHER_CHECKPOINT" ]]; then
  echo "[ERROR] TEACHER_CHECKPOINT does not exist: $TEACHER_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -d "$MOTION_DIR" ]]; then
  echo "[ERROR] MOTION_DIR does not exist: $MOTION_DIR" >&2
  exit 1
fi
if ! find "$MOTION_DIR" -name '*.npz' -print -quit | grep -q .; then
  echo "[ERROR] no .npz motions under $MOTION_DIR" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export HYDRA_FULL_ERROR=1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export PYTHONNOUSERSITE=1
export http_proxy="${http_proxy:-http://127.0.0.1:7890}"
export https_proxy="${https_proxy:-http://127.0.0.1:7890}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"

if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-}" != "online" ]]; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
  echo "[INFO] wandb has no API key; using WANDB_MODE=$WANDB_MODE (official logger still wandb)"
fi

echo "[INFO] Anybody Stage 2 MUSE-Kp latent distill"
echo "[INFO]   teacher=$TEACHER_CHECKPOINT"
echo "[INFO]   motion=$MOTION_DIR"
echo "[INFO]   nproc=$NPROC num_envs=$NUM_ENVS run_name=$RUN_NAME"
echo "[INFO]   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES master_port=$MASTER_PORT"
echo "[INFO]   npz count=$(find "$MOTION_DIR" -name '*.npz' | wc -l)"

TORCHRUN="$(command -v torchrun || true)"
if [[ -z "$TORCHRUN" ]]; then
  TORCHRUN="$(command -v python) -m torch.distributed.run"
fi

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume_student_checkpoint "$RESUME_CHECKPOINT")
fi

# Headless distill does not need Nucleus UI markers (same S3/proxy issue as Stage 1).
HYDRA_FULL_ERROR=1 $TORCHRUN --standalone --nnodes=1 --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" scripts/rsl_rl/train.py \
  --task=MUSE-Kp-LatentDistill-General-Tracking-Flat-G1-v0 \
  --distributed \
  --num_envs="$NUM_ENVS" \
  --motion "$MOTION_DIR" \
  --teacher_checkpoint "$TEACHER_CHECKPOINT" \
  --headless \
  --logger wandb \
  --log_project_name "$LOG_PROJECT" \
  --run_name "$RUN_NAME" \
  "${EXTRA_ARGS[@]}" \
  env.commands.motion.debug_vis=false \
  env.scene.contact_forces.debug_vis=false
