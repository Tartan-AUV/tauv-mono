#!/usr/bin/env bash
set -euo pipefail

# build_osprey_docker.sh
# Builds the osprey_orin image stack and applies a non-cached user-specific config layer
# on top (osprey_orin_user). Uses Docker Buildx Bake.
#
# Usage:
#   ./build_osprey_docker.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Resolve Git identity from host configuration only
GIT_USER_NAME_VAL=$(git config --global --get user.name || true)
GIT_USER_EMAIL_VAL=$(git config --global --get user.email || true)

if [[ -z "${GIT_USER_NAME_VAL}" || -z "${GIT_USER_EMAIL_VAL}" ]]; then
  echo "Error: Git identity not found. Please configure your Git user on the host." >&2
  echo "   Hint: git config --global user.name \"Your Name\" && git config --global user.email you@example.com" >&2
  exit 1
fi

# Show what we will use
echo "Git Name : ${GIT_USER_NAME_VAL:-<empty>}"
echo "Git Email: ${GIT_USER_EMAIL_VAL:-<empty>}"

# Ensure buildx exists
if ! docker buildx version >/dev/null 2>&1; then
  echo "Docker Buildx is required. Please enable Buildx in your Docker installation." >&2
  exit 1
fi

# Invoke bake. We set build args on the osprey_orin_user target only.
# The osprey_orin_user target is marked no-cache in docker-bake.hcl so it will always rebuild.
set -x
exec docker buildx bake \
  --progress=auto \
  --file osprey-docker-bake.hcl \
  --set osprey_orin_user.output=type=docker \
  --load \
  osprey_orin_user
