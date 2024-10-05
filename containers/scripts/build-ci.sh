#!/bin/bash

cd ..
docker build --file platform/Dockerfile.x86-nvidia --tag tauv/x86-nvidia-platform .
docker build --build-arg BASE_IMAGE=tauv/x86-nvidia-platform --file common/Dockerfile.common --tag tauv/x86-nvidia-common .
docker build --build-arg BASE_IMAGE=tauv/x86-nvidia-common --file apps/surface/Dockerfile.ci --tag tauv/x86-nvidia-ci .

echo -e "\n\n\nFinished building the workstation image."
