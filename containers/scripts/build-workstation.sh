#!/bin/bash

# todo: TAG IMAGES WITH COMMIT HASH

set -e

if [ -n "$GITHUB_WORKSPACE" ]; then
  # If inside GitHub Actions, use GITHUB_WORKSPACE as the repo root
  REPO_ROOT="$GITHUB_WORKSPACE"
else
  REPO_ROOT=$(pwd)
fi

cd $REPO_ROOT/containers

docker build --file platform/Dockerfile.x86-nvidia-turbovnc --tag tauv/x86-nvidia-turbovnc-platform .
docker build --build-arg BASE_IMAGE=tauv/x86-nvidia-turbovnc-platform --file common/Dockerfile.common --tag tauv/x86-nvidia-common .
docker build --build-arg BASE_IMAGE=tauv/x86-nvidia-common --file apps/surface/Dockerfile.workstation --tag tauv/x86-nvidia-workstation .

echo -e "\n\n\nFinished building the workstation image."
