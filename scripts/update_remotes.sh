#!/usr/bin/env bash
# One-click sync for AnyBody:
#   GitHub     <- code  (respects .gitignore)
#   ModelScope <- datasets / checkpoints / results (respects .msignore)
#
# Setup (once):
#   cp scripts/sync_remotes.env.example scripts/sync_remotes.env
#   # edit scripts/sync_remotes.env  — fill USER / TOKEN
#
# Usage:
#   bash scripts/update_remotes.sh --yes
#   bash scripts/update_remotes.sh --dry-run
#   bash scripts/update_remotes.sh --github-only --yes
#   bash scripts/update_remotes.sh --modelscope-only --yes
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${SYNC_ENV:-${ROOT}/scripts/sync_remotes.env}"
GITIGNORE="${ROOT}/.gitignore"
MSIGNORE="${ROOT}/.msignore"
MAX_GITHUB_BYTES="$((90 * 1024 * 1024))"

DO_GITHUB=1
DO_MODELSCOPE=1
DRY_RUN=0
YES=0

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \?//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --github-only) DO_MODELSCOPE=0 ;;
    --modelscope-only) DO_GITHUB=0 ;;
    --dry-run) DRY_RUN=1 ;;
    --yes|-y) YES=1 ;;
    --env) ENV_FILE="$2"; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 2 ;;
  esac
  shift
done

die() { echo "[ERROR] $*" >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*"; }
need() { [[ -n "${!1:-}" && "${!1}" != YOUR_* ]] || die "set $1 in ${ENV_FILE}"; }

[[ -f "${ENV_FILE}" ]] || die "missing ${ENV_FILE}
Copy the template first:
  cp ${ROOT}/scripts/sync_remotes.env.example ${ROOT}/scripts/sync_remotes.env
then fill GITHUB_* and MODELSCOPE_*."

# shellcheck disable=SC1090
set -a
source "${ENV_FILE}"
set +a

GITHUB_USER="${GITHUB_USER:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
GITHUB_REPO="${GITHUB_REPO:-Anybody}"
GITHUB_PRIVATE="${GITHUB_PRIVATE:-1}"
GITHUB_COMMIT="${GITHUB_COMMIT:-1}"
GITHUB_COMMIT_MESSAGE="${GITHUB_COMMIT_MESSAGE:-}"
MODELSCOPE_USER="${MODELSCOPE_USER:-}"
MODELSCOPE_TOKEN="${MODELSCOPE_TOKEN:-}"
MODELSCOPE_DATASET="${MODELSCOPE_DATASET:-Anybody-data}"
MODELSCOPE_MODEL="${MODELSCOPE_MODEL:-Anybody-ckpts}"
MODELSCOPE_VISIBILITY="${MODELSCOPE_VISIBILITY:-private}"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-modelscope}"
INCLUDE_BONES_SEED="${INCLUDE_BONES_SEED:-0}"
UPLOAD_DATASETS="${UPLOAD_DATASETS:-1}"
UPLOAD_CKPTS="${UPLOAD_CKPTS:-1}"
UPLOAD_RESULTS="${UPLOAD_RESULTS:-1}"
UPLOAD_EXPORTS="${UPLOAD_EXPORTS:-1}"

if [[ -n "${HTTP_PROXY:-}" ]]; then export http_proxy="${HTTP_PROXY}" HTTP_PROXY="${HTTP_PROXY}"; fi
if [[ -n "${HTTPS_PROXY:-}" ]]; then export https_proxy="${HTTPS_PROXY}" HTTPS_PROXY="${HTTPS_PROXY}"; fi
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"

run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY-RUN'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

confirm() {
  [[ "${YES}" == "1" ]] && return 0
  if [[ ! -t 0 ]]; then
    die "non-interactive shell: pass --yes"
  fi
  read -r -p "$1 [y/N] " ans
  [[ "${ans}" == "y" || "${ans}" == "Y" ]]
}

MS_EXCLUDE=()
ms_exclude_args() {
  MS_EXCLUDE=()
  if [[ -f "${MSIGNORE}" ]]; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
      line="${line%%#*}"
      line="${line%"${line##*[![:space:]]}"}"
      line="${line#"${line%%[![:space:]]*}"}"
      [[ -z "${line}" ]] && continue
      if [[ "${INCLUDE_BONES_SEED}" == "1" && "${line}" == *bones-seed* ]]; then
        continue
      fi
      MS_EXCLUDE+=("${line}")
    done < "${MSIGNORE}"
  fi
  if [[ "${INCLUDE_BONES_SEED}" != "1" ]]; then
    MS_EXCLUDE+=("bones-seed/**" "bones-seed" "**/bones-seed/**")
  fi
}

github_ensure_repo() {
  local vis="private"
  [[ "${GITHUB_PRIVATE}" == "1" ]] || vis="public"
  local code
  code="$(curl -sS -o /tmp/anybody_gh_repo.json -w '%{http_code}' \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${GITHUB_USER}/${GITHUB_REPO}" || true)"
  if [[ "${code}" == "200" ]]; then
    log "GitHub repo exists: ${GITHUB_USER}/${GITHUB_REPO}"
    return 0
  fi
  log "creating GitHub repo ${GITHUB_USER}/${GITHUB_REPO} (${vis})"
  run curl -sS -X POST \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/user/repos" \
    -d "{\"name\":\"${GITHUB_REPO}\",\"private\":$([[ "${vis}" == "private" ]] && echo true || echo false)}" \
    >/tmp/anybody_gh_create.json
}

github_guard_large_files() {
  local f sz
  local bad=0
  while IFS= read -r f; do
    [[ -z "${f}" || ! -f "${ROOT}/${f}" ]] && continue
    sz="$(stat -c%s "${ROOT}/${f}")"
    if (( sz > MAX_GITHUB_BYTES )); then
      echo "  TOO LARGE for GitHub (${sz} bytes): ${f}"
      bad=1
    fi
  done < <(git -C "${ROOT}" diff --cached --name-only)
  [[ "${bad}" == "0" ]] || die "staged files exceed GitHub 100MB limit; they belong on ModelScope / .gitignore"
}

do_github() {
  need GITHUB_USER
  need GITHUB_TOKEN
  [[ -f "${GITIGNORE}" ]] || die "missing ${GITIGNORE}"
  command -v git >/dev/null || die "git not found"
  command -v curl >/dev/null || die "curl not found"

  local branch remote_url
  branch="$(git -C "${ROOT}" rev-parse --abbrev-ref HEAD)"
  remote_url="https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${GITHUB_REPO}.git"

  log "GitHub target: https://github.com/${GITHUB_USER}/${GITHUB_REPO}  branch=${branch}"
  github_ensure_repo

  if [[ "${GITHUB_COMMIT}" == "1" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      log "dry-run staged preview:"
      git -C "${ROOT}" add -An
    else
      git -C "${ROOT}" add -A
      github_guard_large_files
      if git -C "${ROOT}" diff --cached --quiet; then
        log "no code changes to commit"
      else
        local msg="${GITHUB_COMMIT_MESSAGE:-sync: update Anybody code $(date -u +%Y-%m-%dT%H:%MZ)}"
        log "commit: ${msg}"
        git -C "${ROOT}" commit -m "$(cat <<EOF
${msg}

EOF
)"
      fi
    fi
  fi

  log "push ${branch} -> origin"
  run git -C "${ROOT}" push "${remote_url}" "HEAD:refs/heads/${branch}"
  log "GitHub done: https://github.com/${GITHUB_USER}/${GITHUB_REPO}/tree/${branch}"
}

ms() {
  [[ -x "${MODELSCOPE_BIN}" ]] || command -v "${MODELSCOPE_BIN}" >/dev/null \
    || die "modelscope CLI not found: ${MODELSCOPE_BIN}"
  run "${MODELSCOPE_BIN}" --token "${MODELSCOPE_TOKEN}" "$@"
}

ms_ensure_repo() {
  local repo_id="$1" repo_type="$2"
  log "ensure ModelScope ${repo_type} ${repo_id}"
  ms create "${repo_id}" --repo-type "${repo_type}" \
    --visibility "${MODELSCOPE_VISIBILITY}" --exist-ok \
    --description "AnyBody ${repo_type} dump from $(hostname)" >/dev/null || true
}

ms_upload_dir() {
  local repo_id="$1" repo_type="$2" local_dir="$3" path_in_repo="$4" msg="$5"
  [[ -d "${local_dir}" ]] || { log "skip missing ${local_dir}"; return 0; }
  ms_exclude_args
  log "ModelScope upload ${local_dir} -> ${repo_id}:${path_in_repo}  (${repo_type})"
  du -sh "${local_dir}" || true
  if [[ ${#MS_EXCLUDE[@]} -gt 0 ]]; then
    ms upload "${repo_id}" "${local_dir}" "${path_in_repo}" \
      --repo-type "${repo_type}" \
      --commit-message "${msg}" \
      --max-workers 4 \
      --exclude "${MS_EXCLUDE[@]}"
  else
    ms upload "${repo_id}" "${local_dir}" "${path_in_repo}" \
      --repo-type "${repo_type}" \
      --commit-message "${msg}" \
      --max-workers 4
  fi
}

do_modelscope() {
  need MODELSCOPE_USER
  need MODELSCOPE_TOKEN
  [[ -x "${MODELSCOPE_BIN}" || -n "$(command -v "${MODELSCOPE_BIN}" || true)" ]] \
    || die "install modelscope CLI or set MODELSCOPE_BIN in ${ENV_FILE}"

  local ds="${MODELSCOPE_USER}/${MODELSCOPE_DATASET}"
  local md="${MODELSCOPE_USER}/${MODELSCOPE_MODEL}"

  log "ModelScope login check"
  ms whoami >/dev/null || true

  if [[ "${UPLOAD_DATASETS}" == "1" || "${UPLOAD_RESULTS}" == "1" ]]; then
    ms_ensure_repo "${ds}" dataset
  fi
  if [[ "${UPLOAD_CKPTS}" == "1" || "${UPLOAD_EXPORTS}" == "1" ]]; then
    ms_ensure_repo "${md}" model
  fi

  if [[ "${UPLOAD_DATASETS}" == "1" ]]; then
    if [[ "${INCLUDE_BONES_SEED}" != "1" ]]; then
      log "skipping datasets/bones-seed (~389GB; public at bones-studio/seed). Set INCLUDE_BONES_SEED=1 to upload."
    fi
    ms_upload_dir "${ds}" dataset "${ROOT}/datasets" "datasets" "sync datasets $(date -u +%F)"
  fi
  if [[ "${UPLOAD_RESULTS}" == "1" ]]; then
    ms_upload_dir "${ds}" dataset "${ROOT}/results" "results" "sync results $(date -u +%F)"
  fi
  if [[ "${UPLOAD_CKPTS}" == "1" ]]; then
    ms_upload_dir "${md}" model "${ROOT}/logs/rsl_rl" "rsl_rl" "sync checkpoints $(date -u +%F)"
    if [[ -d "${ROOT}/logs/tritrack" ]]; then
      ms_upload_dir "${md}" model "${ROOT}/logs/tritrack" "tritrack" "sync tritrack $(date -u +%F)"
    fi
  fi
  if [[ "${UPLOAD_EXPORTS}" == "1" ]]; then
    ms_upload_dir "${md}" model "${ROOT}/exports" "exports" "sync exports $(date -u +%F)"
  fi
  log "ModelScope dataset: https://modelscope.cn/datasets/${ds}"
  log "ModelScope model:    https://modelscope.cn/models/${md}"
}

preview() {
  echo "============================================================"
  echo "ROOT             : ${ROOT}"
  echo "ENV              : ${ENV_FILE}"
  echo "DRY_RUN          : ${DRY_RUN}"
  echo "GitHub           : ${DO_GITHUB}  -> ${GITHUB_USER:-?}/${GITHUB_REPO}"
  echo "ModelScope       : ${DO_MODELSCOPE}  -> data=${MODELSCOPE_USER:-?}/${MODELSCOPE_DATASET}  ckpt=${MODELSCOPE_USER:-?}/${MODELSCOPE_MODEL}"
  echo "INCLUDE_BONES_SEED: ${INCLUDE_BONES_SEED}"
  echo "UPLOAD           : datasets=${UPLOAD_DATASETS} ckpts=${UPLOAD_CKPTS} results=${UPLOAD_RESULTS} exports=${UPLOAD_EXPORTS}"
  echo "============================================================"
  echo "local sizes (what would be considered):"
  du -sh "${ROOT}/source" "${ROOT}/scripts" "${ROOT}/datasets" "${ROOT}/logs" "${ROOT}/results" "${ROOT}/exports" 2>/dev/null || true
}

preview
confirm "Proceed with this sync?" || die "aborted"

cd "${ROOT}"
[[ "${DO_GITHUB}" == "1" ]] && do_github
[[ "${DO_MODELSCOPE}" == "1" ]] && do_modelscope
log "all requested uploads finished"
