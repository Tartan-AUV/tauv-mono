FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0 AS base

# set debian non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    python3-pip \
    libopenblas-dev \
    wget \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    python3.10-dev \
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

# Install cuSPARSELt manually
RUN wget -q https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.8.1.1_cuda12-archive.tar.xz && \
    tar -xf libcusparse_lt-linux-aarch64-0.8.1.1_cuda12-archive.tar.xz && \
    cp libcusparse_lt-linux-aarch64-0.8.1.1_cuda12-archive/lib/libcusparseLt.so* /usr/lib/aarch64-linux-gnu/ && \
    cp libcusparse_lt-linux-aarch64-0.8.1.1_cuda12-archive/include/cusparseLt* /usr/include/ && \
    rm -rf libcusparse_lt-linux-aarch64-0.8.1.1_cuda12-archive* libcusparse_lt-linux-aarch64-0.8.1.1_cuda12-archive.tar.xz

# Build and install OpenCV 4.12.0 with CUDA support
RUN mkdir -p /tmp/opencv_build && cd /tmp/opencv_build && \
    curl -L https://github.com/opencv/opencv/archive/4.12.0.zip -o opencv-4.12.0.zip && \
    curl -L https://github.com/opencv/opencv_contrib/archive/4.12.0.zip -o opencv_contrib-4.12.0.zip && \
    unzip opencv-4.12.0.zip && \
    unzip opencv_contrib-4.12.0.zip && \
    rm opencv-4.12.0.zip opencv_contrib-4.12.0.zip && \
    cd opencv-4.12.0 && \
    mkdir release && \
    cd release && \
    cmake -D WITH_CUDA=ON \
          -D WITH_CUDNN=ON \
          -D CUDA_ARCH_BIN="8.7" \
          -D CUDA_ARCH_PTX="" \
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
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="/usr/local/lib/python3.10/site-packages/:${PYTHONPATH}"
# Install PyTorch from NVIDIA wheel
RUN python3 -m pip install --upgrade pip; python3 -m pip install numpy=='1.26.1'; python3 -m pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl


ENV TZ="America/New_York"
ENV DEBIAN_FRONTEND="noninteractive"

# Install ROS2 Humble
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

RUN export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb

RUN apt-get update && apt-get install -y \
    python3-flake8-docstrings \
    python3-pip \
    python3-pytest-cov \
    ros-dev-tools \
    python3-flake8-blind-except \
    python3-flake8-builtins \
    python3-flake8-class-newline \
    python3-flake8-comprehensions \
    python3-flake8-deprecated \
    python3-flake8-import-order \
    python3-flake8-quotes \
    python3-pytest-repeat \
    python3-pytest-rerunfailures && \
    rm -rf /var/lib/apt/lists/*     

RUN mkdir -p /tmp/ros2_humble/src

WORKDIR /tmp/ros2_humble

#ROSDEP MASTER KEYS EXCLUDE OPENCV-  python path cuda cv 
RUN apt-get update && \
    vcs import --input https://raw.githubusercontent.com/ros2/ros2/humble/ros2.repos src && \
    rosdep init && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -y --os=ubuntu:jammy --skip-keys "fastcdr rti-connext-dds-6.0.1 urdfdom_headers python3-pytest-timeout graphviz opencv" && \
    rm -rf /var/lib/apt/lists/*     

RUN colcon build --merge-install --install-base /opt/ros/humble && \
    rm -rf /var/lib/apt/lists/* /tmp/ros2_humble /tmp/ros2-apt-source.deb

# Download and install ArenaSDK from S3
# RUN --mount=type=secret,id=aws_credentials \
#     mkdir -p /root/.aws && \
#     cp /run/secrets/aws_credentials /root/.aws/credentials && \
#     mkdir -p /opt/arena && \
#     aws s3 cp s3://tauv-build-assets/ArenaSDK_v0.1.78_Linux_ARM64.tar.gz /opt/arena/ArenaSDK.tar.gz

# RUN cd /opt/arena/ && \
#     tar -xzf ArenaSDK.tar.gz && \
#     cd ArenaSDK_Linux_ARM64 && \
#     chmod +x Arena_SDK_ARM64.conf && \
#     sh Arena_SDK_ARM64.conf && \
#     rm /root/.aws/credentials

RUN apt-get update && apt-get install -y \
    libboost-all-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
	ros-humble-nmea-msgs \
	ros-humble-mavros-msgs \
    python3-smbus2 \
    && rm -rf /var/lib/apt/lists/*

# Build and install cv_bridge from source
RUN mkdir -p /tmp/cv_bridge_build && cd /tmp/cv_bridge_build && \
    git clone https://github.com/ros-perception/vision_opencv.git -b humble src/vision_opencv && \
    bash -c 'source /opt/ros/humble/setup.bash && colcon build --packages-select cv_bridge --merge-install --install-base /opt/ros/humble' && \
    rm -rf /tmp/cv_bridge_build

#DroneCAN
RUN python3 -m pip install dronecan=="1.0.27" pyserial transform3d

#Depthai
RUN python3 -m pip install "numpy<2.0.0" \ depthai

#RTAB-map and pointcloud stuff
RUN apt-get update \
    && apt install -y --no-install-recommends ros-humble-rtabmap-ros

RUN apt-get update \ 
    && apt install -y --no-install-recommends ros-humble-depth-image-proc