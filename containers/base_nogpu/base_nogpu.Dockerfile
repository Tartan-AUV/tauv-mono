#+ syntax=docker/dockerfile:1.7
FROM ubuntu:noble AS base

# Base desktop image for development on Ubuntu 24.04 (noble).
# Installs ROS 2 Jazzy from the official APT repository.

ENV DEBIAN_FRONTEND=noninteractive

# 1) Prerequisites and universe repo
# - Keep the set minimal; avoid pulling extra recommends.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        openssh-client \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        curl \
        gnupg2 \
        lsb-release \
        ca-certificates \
        software-properties-common \
    && add-apt-repository universe \
    && rm -rf /var/lib/apt/lists/*

# 2) ROS 2 APT repo + key (Jazzy for noble)
RUN set -eux; \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) main" \
      > /etc/apt/sources.list.d/ros2.list

# 3) Install ROS 2 Jazzy (desktop) and common dev tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-jazzy-desktop \
        ros-jazzy-robot-localization \
        ros-dev-tools \
    && rm -rf /var/lib/apt/lists/*

# 4) Convenience: source ROS in default bash environment
ENV ROS_DISTRO=jazzy
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc

# 5) Install dependencies for Stonefish
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-dev \
        libglm-dev \
        libsdl2-dev \
        libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# 6) Pre-populate known_hosts to avoid interactive host key prompts
RUN mkdir -p -m 0755 /etc/ssh \
    && touch /etc/ssh/ssh_known_hosts \
    && ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> /etc/ssh/ssh_known_hosts

WORKDIR /

CMD ["bash"]
