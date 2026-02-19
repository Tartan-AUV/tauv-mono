FROM base AS common

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    make \
    cmake \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc-11 \
    g++-11 \
    gcc-13 \
    g++-13 \
    libstdc++-13-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-robot-localization \
    ros-humble-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*
# -----------------------------
# Set GCC 13 as default
# -----------------------------
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 10 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 20 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 10 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 20

ENV CC=gcc-13
ENV CXX=g++-13

# -----------------------------
# Python virtual environment
# -----------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip, setuptools, wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# -----------------------------
# Python packages (ROS build + runtime)
# -----------------------------
RUN pip install --no-cache-dir \
    catkin_pkg \
    empy==3.3.4 \
    lark==1.1.1 \
    pyyaml \
    numpy \
    pyparsing==2.4.7 \
    rosdistro \
    pandas \
    matplotlib \
    scipy \
    spatialmath-python