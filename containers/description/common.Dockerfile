FROM base AS common


### INSTALL ROS2 HUMBLE ###

ENV ROS_DISTRO=humble

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository universe \
    && apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb && \
    rm -f /tmp/ros2-apt-source.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base \
    ros-humble-rqt \
    ros-humble-rqt-common-plugins \
    ros-humble-foxglove-bridge \
    ros-dev-tools \
    python3-rosdep \
    python3-vcstool \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init 2>/dev/null || true && rosdep update

WORKDIR /


### BUILD OPENCV FROM SOURCE ###

ARG OPENCV_VERSION=4.12.0
ARG OPENCV_WITH_CUDA=OFF
ARG OPENCV_CUDA_ARCH_BIN=

RUN apt-get update && apt-get install -y --no-install-recommends \
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
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    mkdir -p /tmp/opencv_build; \
    cd /tmp/opencv_build; \
    curl -fsSL "https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip" -o "opencv-${OPENCV_VERSION}.zip"; \
    curl -fsSL "https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip" -o "opencv_contrib-${OPENCV_VERSION}.zip"; \
    unzip "opencv-${OPENCV_VERSION}.zip"; \
    unzip "opencv_contrib-${OPENCV_VERSION}.zip"; \
    rm -f "opencv-${OPENCV_VERSION}.zip" "opencv_contrib-${OPENCV_VERSION}.zip"; \
    cd "opencv-${OPENCV_VERSION}"; \
    mkdir -p release; \
    cd release; \
    CUDA_FLAGS=""; \
    if [ "${OPENCV_WITH_CUDA}" = "ON" ]; then \
      CUDA_FLAGS="-D WITH_CUDA=ON -D WITH_CUDNN=ON -D CUDA_ARCH_PTX="; \
      if [ -n "${OPENCV_CUDA_ARCH_BIN}" ]; then \
        CUDA_FLAGS="${CUDA_FLAGS} -D CUDA_ARCH_BIN=${OPENCV_CUDA_ARCH_BIN}"; \
      fi; \
    else \
      CUDA_FLAGS="-D WITH_CUDA=OFF -D WITH_CUDNN=OFF"; \
    fi; \
    cmake -G Ninja \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D OPENCV_EXTRA_MODULES_PATH="../../opencv_contrib-${OPENCV_VERSION}/modules" \
      -D WITH_GSTREAMER=ON \
      -D WITH_LIBV4L=ON \
      -D BUILD_opencv_python3=ON \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/opt/opencv-custom \
      ${CUDA_FLAGS} \
      ..; \
    ninja -j"$(nproc)"; \
    ninja install; \
    rm -rf /tmp/opencv_build; \
    echo "/opt/opencv-custom/lib" > /etc/ld.so.conf.d/opencv-custom.conf; \
    ldconfig

ENV OpenCV_DIR=/opt/opencv-custom/lib/cmake/opencv4
ENV PKG_CONFIG_PATH=/opt/opencv-custom/lib/pkgconfig
ENV LD_LIBRARY_PATH=/opt/opencv-custom/lib
ENV PYTHONPATH=/opt/opencv-custom/lib/python3.10/dist-packages
ENV CMAKE_PREFIX_PATH=/opt/opencv-custom


### BUILD ROS2 OVERLAY ###

# We intentionally build `vision_opencv` from source so `cv_bridge` links against our manually
# installed OpenCV instead of the Ubuntu-packaged OpenCV.

# ANY other packages that go into the overlay should be built here.

RUN set -eux; \
    mkdir -p /opt/ros/overlay; \
    mkdir -p /tmp/overlay_ws/src; \
    git clone --depth 1 --branch "${ROS_DISTRO}" https://github.com/ros-perception/vision_opencv.git /tmp/overlay_ws/src/vision_opencv; \
    set +u; . "/opt/ros/${ROS_DISTRO}/setup.sh"; set -u; \
    apt-get update && \
    rosdep install --from-paths /tmp/overlay_ws/src --ignore-src -r -y --rosdistro "${ROS_DISTRO}" --skip-keys="libopencv-contrib-dev libopencv-dev libopencv-imgproc-dev"; \
    colcon build \
      --base-paths /tmp/overlay_ws/src \
      --merge-install \
      --install-base "/opt/ros/overlay" \
      --cmake-args \
        "-DOpenCV_DIR=${OpenCV_DIR}"; \
    rm -rf /tmp/overlay_ws /var/lib/apt/lists/*


### INSTALL PIP-ONLY PYTHON PACKAGES ###

# This section should ONLY install python packages that are not available through APT.
# For packages already installed through apt, constrain their versions.
RUN <<'EOF' cat > /pip-constraints.txt
numpy==1.21.5
scipy==1.8.0
matplotlib==3.5.1
EOF

RUN python3 -m pip install --no-cache-dir -c /pip-constraints.txt --target /opt/python-extras \
    spatialmath-python \
    pipx \
    && rm /pip-constraints.txt
ENV PYTHONPATH=/opt/python-extras:${PYTHONPATH}

RUN python3 -m pipx ensurepath --global

### INSTALL CLI TOOLS ###

# Codex
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update \
    && apt-get install -y nodejs\
    && rm -rf /var/lib/apt/lists/*
RUN npm install -g @openai/codex


### CONFIGURE ENVIRONMENT ###

COPY description/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]

