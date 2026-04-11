FROM base AS common

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        # ROS packages
        ros-humble-robot-localization \
        ros-humble-rosbag2-storage-mcap \
        # ros-humble-foxglove-bridge \
        # Python
        python3-pip \
        python3-venv \
        # C++ build tools (GCC 11 and 13)
        gcc-11 \
        g++-11 \
        gcc-13 \
        g++-13 \
        libstdc++-13-dev \
        make \
        cmake \
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

# -----------------------------
# Build Foxglove Bridge natively from source
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/foxglove_ws/src \
    && cd /opt/foxglove_ws/src \
    && git clone --depth 1 https://github.com/foxglove/foxglove-sdk.git \
    && cd /opt/foxglove_ws \
    && apt-get update \
    && rosdep init || true \
    && rosdep update \
    && rosdep install -y --from-paths src --ignore-src --rosdistro humble \
    && rm -rf /var/lib/apt/lists/* \
    && /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release"

RUN echo "source /opt/foxglove_ws/install/setup.bash" >> ~/.bashrc