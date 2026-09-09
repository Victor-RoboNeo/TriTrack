#!/usr/bin/env bash
# Stage 0 GMT teacher with official Anybody flags, local motion path, train from scratch.
# Official script (run_mosaic_gmt.sh) also sets --resume/--load_run/--checkpoint to an
# author-only SONIC 85k run that is not shipped; those flags are omitted here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

MOTION_DIR="${MOTION_DIR:-$ROOT/datasets/MOSAIC_Dataset/G1}"
RUN_NAME="${RUN_NAME:-sonic_102k_teacher}"
NUM_ENVS="${NUM_ENVS:-4096}"
NPROC="${NPROC:-8}"
LOG_PROJECT="${LOG_PROJECT:-MUSE_Distill}"

if [[ ! -d "$MOTION_DIR" ]]; then
  echo "[ERROR] MOTION_DIR does not exist: $MOTION_DIR" >&2
  exit 1
fi
if ! find "$MOTION_DIR" -name '*.npz' -print -quit | grep -q .; then
  echo "[ERROR] no .npz motions under $MOTION_DIR" >&2
  exit 1
fi

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

echo "[INFO] Anybody GMT Stage 0"
echo "[INFO]   motion=$MOTION_DIR"
echo "[INFO]   nproc=$NPROC num_envs=$NUM_ENVS run_name=$RUN_NAME"
echo "[INFO]   npz count=$(find "$MOTION_DIR" -name '*.npz' | wc -l)"

TORCHRUN="$(command -v torchrun || true)"
if [[ -z "$TORCHRUN" ]]; then
  TORCHRUN="$(command -v python) -m torch.distributed.run"
fi

HYDRA_FULL_ERROR=1 $TORCHRUN --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/rsl_rl/train.py \
  --task=General-Tracking-Flat-G1-Wo-State-Estimation-v0-World-Coordinate-Reward \
  --distributed \
  --num_envs="$NUM_ENVS" \
  --motion "$MOTION_DIR" \
  --headless \
  --logger wandb \
  --log_project_name "$LOG_PROJECT" \
  --run_name "$RUN_NAME"
