#!/usr/bin/env bash
set -euo pipefail

# build_orin_docker.sh
# Builds the Jetson Orin container image via Docker Buildx Bake.
#
# Usage:
#   ./build_orin_docker.sh
#   AWS_CREDENTIALS_FILE=~/.aws/credentials ./build_orin_docker.sh
#   OPENCV_WITH_CUDA=ON OPENCV_CUDA_ARCH_BIN=8.7 ./build_orin_docker.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${AWS_CREDENTIALS_FILE:=${HOME}/.aws/credentials}"
if [[ ! -f "${AWS_CREDENTIALS_FILE}" ]]; then
  echo "Error: AWS credentials file not found: ${AWS_CREDENTIALS_FILE}" >&2
  echo "Set AWS_CREDENTIALS_FILE=/path/to/credentials and retry." >&2
  exit 1
fi
export AWS_CREDENTIALS_FILE

: "${OPENCV_WITH_CUDA:=ON}"
: "${OPENCV_CUDA_ARCH_BIN:=8.7}"
export OPENCV_WITH_CUDA OPENCV_CUDA_ARCH_BIN

exec docker buildx bake \
  --progress=auto \
  --file docker-bake.hcl \
  --set osprey_orin.output=type=docker \
  osprey_orin
