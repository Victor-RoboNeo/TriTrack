#!/usr/bin/env bash
# Stage 1 MUSE-Transformer distillation with official Anybody flags, local paths.
# Official script defaults to an unpublished sonic_55k teacher; those paths are omitted here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-$ROOT/logs/rsl_rl/g1_flat_mosaic_hybrid/2026-08-19_16-02-00_sonic_102k_teacher/model_100000.pt}"
MOTION_DIR="${MOTION_DIR:-$ROOT/datasets/SONIC_npzs/g1/npz_splits_loco_manip/train}"
RUN_NAME="${RUN_NAME:-muse_transformer_det_cosine_w0.1_102k_teacher100k}"
NUM_ENVS="${NUM_ENVS:-2048}"
NPROC="${NPROC:-4}"
LOG_PROJECT="${LOG_PROJECT:-MUSE_Distill}"

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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
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

echo "[INFO] Anybody Stage 1 MUSE-Transformer"
echo "[INFO]   teacher=$TEACHER_CHECKPOINT"
echo "[INFO]   motion=$MOTION_DIR"
echo "[INFO]   nproc=$NPROC num_envs=$NUM_ENVS run_name=$RUN_NAME"
echo "[INFO]   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO]   npz count=$(find "$MOTION_DIR" -name '*.npz' | wc -l)"

TORCHRUN="$(command -v torchrun || true)"
if [[ -z "$TORCHRUN" ]]; then
  TORCHRUN="$(command -v python) -m torch.distributed.run"
fi

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume_student_checkpoint "$RESUME_CHECKPOINT")
fi

# Headless distill does not need Nucleus UI markers. Isaac 4.5 frame_prim.usd is
# currently unreachable (S3/proxy 502) and would crash env construction.
HYDRA_FULL_ERROR=1 $TORCHRUN --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/rsl_rl/train.py \
  --task=MUSE-Transformer-Distill-General-Tracking-Flat-G1-v0 \
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
