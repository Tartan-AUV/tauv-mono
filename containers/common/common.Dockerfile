#+ syntax=docker/dockerfile:1.7
FROM base AS common_rosx

# Ensure pip is available for Python package installs
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --break-system-packages \
    spatialmath-python \
    scipy

# Build and install ROS 2 Humble (ros_core + rviz + rqt) into /opt/ros/humble
ENV ROS_DISTRO=humble

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository universe \
    && apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb

RUN apt-get update && apt-get install -y \
    python3-flake8-docstrings \
    python3-pytest-cov \
    ros-dev-tools \
    python3-rosinstall-generator \
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

RUN rosinstall_generator ros_core rosidl_default_generators rosidl_default_runtime rosidl_generator_py rviz2 rqt rqt_common_plugins --rosdistro ${ROS_DISTRO} --deps --tar > /tmp/ros2_humble/ros2-minimal.rosinstall && \
    vcs import src < /tmp/ros2_humble/ros2-minimal.rosinstall && \
    rosdep init || true && \
    rosdep update && \
    apt-get update && \
    rosdep install --from-paths src --ignore-src -y --rosdistro ${ROS_DISTRO} --os=ubuntu:jammy --skip-keys "libopencv-contrib-dev libopencv-dev libopencv-imgproc-dev rti-connext-dds-6.0.1" && \
    rm -rf /var/lib/apt/lists/*

RUN colcon build --merge-install --install-base /opt/ros/humble --packages-up-to ros_core rviz2 rqt rqt_common_plugins && \
    rm -rf /var/lib/apt/lists/* /tmp/ros2_humble /tmp/ros2-apt-source.deb

RUN apt-get update && apt-get install -y \
    libboost-all-dev && \
    rm -rf /var/lib/apt/lists/*

# Build and install common_interfaces (message targets) into /opt/ros/humble
RUN mkdir -p /tmp/ros2_msgs_ws/src && \
    rosinstall_generator rosidl_generator_py rosidl_generator_c rosidl_generator_cpp common_interfaces --rosdistro ${ROS_DISTRO} --deps --tar > /tmp/ros2_msgs_ws/ros2-msgs.rosinstall && \
    vcs import /tmp/ros2_msgs_ws/src < /tmp/ros2_msgs_ws/ros2-msgs.rosinstall && \
    apt-get update && \
    rosdep install --from-paths /tmp/ros2_msgs_ws/src --ignore-src -y --rosdistro ${ROS_DISTRO} --os=ubuntu:jammy --skip-keys "libopencv-contrib-dev libopencv-dev libopencv-imgproc-dev rti-connext-dds-6.0.1" && \
    bash -c 'source /opt/ros/humble/setup.bash && colcon build --merge-install --install-base /opt/ros/humble --packages-up-to common_interfaces --base-paths /tmp/ros2_msgs_ws/src' && \
    rm -rf /tmp/ros2_msgs_ws /var/lib/apt/lists/*

# Build and install cv_bridge from source
RUN mkdir -p /tmp/cv_bridge_build && cd /tmp/cv_bridge_build && \
    git clone https://github.com/ros-perception/vision_opencv.git -b humble src/vision_opencv && \
    bash -c 'source /opt/ros/humble/setup.bash && colcon build --packages-select cv_bridge --merge-install --install-base /opt/ros/humble' && \
    rm -rf /tmp/cv_bridge_build

WORKDIR /

# Build and install rosx_introspection into /opt/ros/humble
RUN mkdir -p /tmp/rosx_ws/src && \
    git clone https://github.com/facontidavide/rosx_introspection.git /tmp/rosx_ws/src/rosx_introspection && \
    apt-get update && \
    bash -c 'source /opt/ros/humble/setup.bash && rosdep install --from-paths /tmp/rosx_ws/src --ignore-src -y --rosdistro ${ROS_DISTRO} --os=ubuntu:jammy --skip-keys "libopencv-contrib-dev libopencv-dev libopencv-imgproc-dev"' && \
    bash -c 'source /opt/ros/humble/setup.bash && colcon build --merge-install --install-base /opt/ros/humble --base-paths /tmp/rosx_ws/src' && \
    rm -rf /tmp/rosx_ws /var/lib/apt/lists/*


# Build and install foxglove_bridge into /opt/ros/humble
RUN mkdir -p /tmp/foxglove_ws/src; \
    # Prefer the ROS distro-pinned source from rosdistro instead of cloning a moving upstream default branch.
    # This avoids CMake/ament target mismatches (e.g., geometry_msgs__rosidl_generator_cpp) when building
    # against the Humble message interface packages in this image.
    if rosinstall_generator foxglove_bridge --rosdistro "${ROS_DISTRO}" --deps --tar > /tmp/foxglove_ws/foxglove_bridge.rosinstall; then \
      vcs import /tmp/foxglove_ws/src < /tmp/foxglove_ws/foxglove_bridge.rosinstall; \
    else \
      git clone --depth 1 -b "${ROS_DISTRO}" https://github.com/foxglove/ros-foxglove-bridge.git /tmp/foxglove_ws/src/ros-foxglove-bridge; \
    fi; \
    apt-get update; \
    bash -c 'source /opt/ros/humble/setup.bash && rosdep install --from-paths /tmp/foxglove_ws/src --ignore-src -y --rosdistro ${ROS_DISTRO} --os=ubuntu:jammy --skip-keys "rti-connext-dds-6.0.1 libopencv-contrib-dev libopencv-dev libopencv-imgproc-dev"' && \
    bash -c 'source /opt/ros/humble/setup.bash && colcon build --merge-install --install-base /opt/ros/humble --base-paths /tmp/foxglove_ws/src --packages-up-to foxglove_bridge --cmake-args -DBUILD_TESTING=OFF' && \
    rm -rf /tmp/foxglove_ws /var/lib/apt/lists/*

FROM common_rosx AS common

# Codex never hurts
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs
RUN npm install -g @openai/codex
