#!/bin/bash

docker build --file platform/Dockerfile.x86-nvidia-turbovnc --tag tauv/x86-nvidia-turbovnc-platform .
docker build --build-arg BASE_IMAGE=tauv/x86-nvidia-turbovnc-platform --file common/Dockerfile.common --tag tauv/x86-nvidia-common .

docker build --build-arg BASE_IMAGE=tauv/x86-nvidia-common --build-arg USER_NAME="$USER" --build-arg USER_PASSWORD="$PASSWORD" --file apps/Dockerfile.workstation --tag tauv/x86-nvidia-workstation .

echo -e "\n\n\nFinished building the workstation image."

