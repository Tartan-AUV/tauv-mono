#+ syntax=docker/dockerfile:1.7
FROM ubuntu:jammy AS base

# Base desktop image for development on Ubuntu 22.04 (jammy).
# Mirrors base_orin with CPU-only dependencies.

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    libopenblas-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    openssh-client \
    libgtk-3-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    python3-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libv4l-dev \
    v4l-utils \
    qv4l2 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Build and install OpenCV 4.12.0 (CPU-only)
RUN mkdir -p /tmp/opencv_build && cd /tmp/opencv_build && \
    curl -L https://github.com/opencv/opencv/archive/4.12.0.zip -o opencv-4.12.0.zip && \
    curl -L https://github.com/opencv/opencv_contrib/archive/4.12.0.zip -o opencv_contrib-4.12.0.zip && \
    unzip opencv-4.12.0.zip && \
    unzip opencv_contrib-4.12.0.zip && \
    rm opencv-4.12.0.zip opencv_contrib-4.12.0.zip && \
    cd opencv-4.12.0 && \
    mkdir release && \
    cd release && \
    cmake -D WITH_CUDA=OFF \
          -D WITH_CUDNN=OFF \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.12.0/modules \
          -D WITH_GSTREAMER=ON \
          -D WITH_LIBV4L=ON \
          -D BUILD_opencv_python3=ON \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          .. && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /tmp/opencv_build

# Set OpenCV environment variables
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages/

# Allow pip global installs
RUN pip config set global.break-system-packages true

# Install PyTorch CPU wheel
RUN python3 -m pip install --upgrade pip; \
    python3 -m pip install numpy=='1.26.1'; \
    python3 -m pip install --no-cache --index-url https://download.pytorch.org/whl/cpu torch==2.5.0

ENV TZ="America/New_York"
ENV DEBIAN_FRONTEND="noninteractive"

# Install dependencies for Stonefish
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-dev \
        libglm-dev \
        libsdl2-dev \
        libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

CMD ["bash"]
