#!/usr/bin/env bash
# ============================================================
# PaperOdyssey dev container launcher.
#
# The repo is bind-mounted at /workspace, so edits on the host take effect
# immediately inside the container — no rebuild between code changes.
# Rebuild only when requirements.txt or the Dockerfile changes.
#
#   ./run_docker.sh                     # interactive shell in the container
#   ./run_docker.sh build               # (re)build the image
#   ./run_docker.sh ui                  # Streamlit UI on http://localhost:8501
#   ./run_docker.sh python scripts/sanity_check.py
#   ./run_docker.sh pytest -q           # ...or any other command
#
# Environment overrides:
#   IMAGE=paperodyssey:dev   image tag
#   PORT=8501                host port mapped to Streamlit
#   NO_GPU=1                 force a CPU-only container
#   TORCH_VARIANT=cpu        build CPU-only torch wheels (build time)
#   INSTALL_PYSERINI=0       skip Pyserini at build time (rank-bm25 fallback)
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-paperodyssey:dev}"
PORT="${PORT:-8501}"
TORCH_VARIANT="${TORCH_VARIANT:-cu128}"
INSTALL_PYSERINI="${INSTALL_PYSERINI:-1}"

log() { printf '\033[1;34m[run_docker]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[run_docker]\033[0m %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker not found on PATH"

build_image() {
  log "building ${IMAGE} (TORCH_VARIANT=${TORCH_VARIANT}, INSTALL_PYSERINI=${INSTALL_PYSERINI})"
  docker build \
    --build-arg "TORCH_VARIANT=${TORCH_VARIANT}" \
    --build-arg "INSTALL_PYSERINI=${INSTALL_PYSERINI}" \
    --build-arg "UID=$(id -u)" \
    --build-arg "GID=$(id -g)" \
    -t "${IMAGE}" \
    -f "${REPO_ROOT}/Dockerfile" \
    "${REPO_ROOT}"
}

image_exists() { docker image inspect "${IMAGE}" >/dev/null 2>&1; }

# --gpus all only if the nvidia runtime is actually registered with dockerd.
gpu_args() {
  [ -n "${NO_GPU:-}" ] && return 0
  if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    printf '%s' "--gpus all"
  else
    log "nvidia runtime not registered with docker — running CPU-only"
  fi
}

if [ "${1:-}" = "build" ]; then
  shift
  build_image
  exit 0
fi

image_exists || { log "image ${IMAGE} not found"; build_image; }

# Caches and model weights live under ./.cache (gitignored) so they persist
# across container restarts without extra volumes.
mkdir -p "${REPO_ROOT}/.cache/hf" "${REPO_ROOT}/.cache/torch"

RUN_ARGS=(
  --rm -i
  --name "paperodyssey-dev-$$"
  --shm-size=2g
  -v "${REPO_ROOT}:/workspace"
  -w /workspace
  -p "${PORT}:8501"
)

# Allocate a TTY only for a real terminal, so `./run_docker.sh python - <<EOF`
# and other piped/redirected invocations still work.
[ -t 0 ] && RUN_ARGS+=( -t )

# shellcheck disable=SC2046  # intentional word-splitting: "--gpus all" or nothing
RUN_ARGS+=( $(gpu_args) )

[ -f "${REPO_ROOT}/.env" ] && RUN_ARGS+=( --env-file "${REPO_ROOT}/.env" )

case "${1:-shell}" in
  shell)
    shift || true
    log "shell in ${IMAGE} (repo mounted at /workspace)"
    exec docker run "${RUN_ARGS[@]}" "${IMAGE}" bash
    ;;
  ui)
    shift
    log "Streamlit on http://localhost:${PORT}"
    exec docker run "${RUN_ARGS[@]}" "${IMAGE}" \
      streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0 "$@"
    ;;
  *)
    exec docker run "${RUN_ARGS[@]}" "${IMAGE}" "$@"
    ;;
esac
