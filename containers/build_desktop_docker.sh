#!/usr/bin/env bash
set -euo pipefail

# build_desktop_docker.sh
# Builds the desktop image stack and applies a non-cached user-specific config layer
# on top (desktop_nogpu_user). Uses Docker Buildx Bake.
#
# Usage:
#   ./build_desktop_docker.sh
#
# This layer is for user-specific config. We assume your host Git is configured.
# If user.name or user.email cannot be determined, the build will exit with an error.

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

# Local cache directory for BuildKit (used in addition to registry cache)
CACHE_DIR=".buildx-cache"

# Invoke bake. We set build args on the desktop_nogpu_user target only.
# The desktop_nogpu_user target is marked no-cache in docker-bake.hcl so it will always rebuild.
# The bakefile doesn't use local cache, so we add it here.
set -x
exec docker buildx bake \
  --progress=auto \
  --file docker-bake.hcl \
  --set base.cache-to+="type=local,dest=${CACHE_DIR}/base,mode=max" \
  --set base.cache-from+="type=local,src=${CACHE_DIR}/base" \
  --set common.cache-to+="type=local,dest=${CACHE_DIR}/common,mode=max" \
  --set common.cache-from+="type=local,src=${CACHE_DIR}/common" \
  --set desktop_nogpu.cache-to+="type=local,dest=${CACHE_DIR}/desktop_nogpu,mode=max" \
  --set desktop_nogpu.cache-from+="type=local,src=${CACHE_DIR}/desktop_nogpu" \
  --set desktop_nogpu_user.args.GIT_USER_NAME="${GIT_USER_NAME_VAL}" \
  --set desktop_nogpu_user.args.GIT_USER_EMAIL="${GIT_USER_EMAIL_VAL}" \
  desktop_nogpu_user
