FROM nvcr.io/nvidia/isaac-sim:4.5.0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN apt-get update && apt-get install -y \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository universe \
    && rm -rf /var/lib/apt/lists* 

# Add ROS 2 GPG key and repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y --allow-downgrades libbrotli1=1.0.9-2build6 \
    && rm -rf /var/lib/apt/lists/*

# Install ROS 2 Humble
RUN apt-get update && apt-get install -y \
    ros-humble-desktop-full \
    && rm -rf /var/lib/apt/lists/*

# Source ROS 2 setup script
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

WORKDIR /isaacsim

# Set entrypoint
ENTRYPOINT ["/bin/bash"]
