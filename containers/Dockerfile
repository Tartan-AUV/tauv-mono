# syntax=docker/dockerfile:1.7
#
# One Dockerfile, three stages:
#
#   base    -> shared by everything: ROS 2 Humble (ros-base), colcon/rosdep, non-root user
#   desktop -> laptop / VM image (x86 or arm64 Ubuntu 22.04) + ros-humble-desktop
#   orin    -> Jetson Orin image (NVIDIA JetPack r36.x, Ubuntu 22.04)
#
# Add a package to `base` if both platforms need it, otherwise to `desktop` or `orin`.
# Prefer apt, then a pinned pip package / prebuilt wheel, then building from source.
# If you do build something from source, give it its own RUN and say why in a comment.

ARG BASE_IMAGE=ubuntu:22.04

# =============================================================================
# base
# =============================================================================
FROM ${BASE_IMAGE} AS base

# UID/GID of the container user. Should match the host user so files written to the
# mounted repo (build/, install/, log/) are owned by you instead of root.
ARG HOST_UID=1000
ARG HOST_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    ROS_DISTRO=humble

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Keep downloaded .debs around so the apt cache mounts below are actually useful.
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# --- Basic tooling ---
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        git-lfs \
        python3-pip \
        sudo

# -----------------------------
# Lightweight Python packages
# -----------------------------
RUN pip install --no-cache-dir \
    filterpy

# --- ROS 2 apt repository ---
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") main" \
        > /etc/apt/sources.list.d/ros2.list

# --- ROS 2 Humble (no GUI tools) + dev tools (colcon, rosdep, vcstool) ---
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-dev-tools \
        ros-$ROS_DISTRO-ros-base \
        # robot_localization packages
        ros-$ROS_DISTRO-angles \
        ros-$ROS_DISTRO-geographic-msgs \
        ros-$ROS_DISTRO-diagnostic-updater \
        libgeographic-dev \
        libboost-dev \
        # Foxglove
        ros-$ROS_DISTRO-foxglove-bridge \
    && rosdep init


# --- Shell setup: source ROS, and the workspace overlay if it has been built ---
RUN printf '%s\n' \
        'source /opt/ros/$ROS_DISTRO/setup.bash' \
        'if [ -f /tauv-mono/ros_ws/install/setup.bash ]; then source /tauv-mono/ros_ws/install/setup.bash; fi' \
        >> /etc/bash.bashrc

# --- Non-root user (passwordless sudo so you can `sudo apt install` while experimenting) ---
RUN (getent group "${HOST_GID}" >/dev/null || groupadd --gid "${HOST_GID}" tauv) \
    && useradd --uid "${HOST_UID}" --gid "${HOST_GID}" --create-home --shell /bin/bash tauv \
    && usermod -aG dialout,video tauv \
    && echo 'tauv ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/tauv \
    && chmod 0440 /etc/sudoers.d/tauv

# =============================================================================
# desktop: laptop / VM development image
# =============================================================================
FROM base AS desktop

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-$ROS_DISTRO-desktop
# desktop-only packages go here (desktopulator deps, clang tools, pre-commit, ...)

# CPU-only torch, decently small. If you want to use an NVIDIA GPU, drop the --index-url
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ultralytics

ENV COLCON_DEFAULTS_FILE=/tauv-mono/ros_ws/colcon_defaults.desktop.yaml

# Keep these last
USER tauv
WORKDIR /tauv-mono/ros_ws
CMD ["bash"]

# =============================================================================
# orin: Jetson Orin vehicle image
# =============================================================================
FROM base AS orin

# torch/torchvision: NVIDIA's Jetson builds (CUDA-enabled). VERY HEAVY
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision \
        --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# Ultralytics, also decently heavy, installed here under orin to make sure we use jetson torch
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ultralytics "numpy<2"

# --- Runtime deps of NVIDIA's Jetson torch wheels (jp6/cu126) ---
# Neither ships in JetPack r36.4:
#   libopenblas-dev    -> libopenblas.so.0
#   libcudss0-cuda-12  -> libcudss.so.0, from NVIDIA's Jetson CUDA repo. Must be the 0.x series:
#                         the repo's default `cudss` is now 1.x (libcudss.so.1), which torch won't load.
# cuDSS installs into a versioned subdir, so register it with the dynamic linker.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
        libopenblas-dev \
        libcudss0-cuda-12 \
    && echo /usr/lib/aarch64-linux-gnu/libcudss/12 > /etc/ld.so.conf.d/cudss.conf \
    && ldconfig

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        # xsens_mti_ros2_driver
        ros-$ROS_DISTRO-nmea-msgs \
        ros-$ROS_DISTRO-mavros-msgs
# orin-only packages go here (torch wheel, sensor drivers, ...)

ENV COLCON_DEFAULTS_FILE=/tauv-mono/ros_ws/colcon_defaults.orin.yaml

# Keep these last
USER tauv
WORKDIR /tauv-mono/ros_ws
CMD ["bash"]
