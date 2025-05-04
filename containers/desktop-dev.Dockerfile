FROM ros:jazzy

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages
RUN apt-get update && apt-get install -y \
    git curl wget lsb-release gnupg2 \
    build-essential cmake ninja-build \
    python3-pip python3-venv \
    clang-format clang-tidy cppcheck \
    gcc-arm-none-eabi gdb-multiarch \
    python3-colcon-common-extensions \
    python3-pytest python3.12-venv \
    tmux vim htop \
    libboost-dev libflatbuffers-dev \
    libflatbuffers2 flatbuffers-compiler \
    && rm -rf /var/lib/apt/lists/*

# Python tools in venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools \
    && pip install \
        catkin-tools \
        ruff \
        mypy \
        pre-commit \
        numpy

# Prepare workspace
ENV ROS_WS=/ros_ws
RUN mkdir -p $ROS_WS/src
RUN mkdir /tauv-mono

# Avoid Git safe.directory errors
RUN git config --system --add safe.directory /tauv-mono

# Setup environment for the user
WORKDIR /tauv-mono

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc \
    && echo "cd /ros_ws" >> /root/.bashrc

CMD [ "bash" ]