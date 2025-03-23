FROM ubuntu:22.04

# Set noninteractive mode to avoid tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install prerequisites
RUN apt update && apt install -y \ 
    curl \ 
    gnupg2 \ 
    lsb-release \ 
    software-properties-common \ 
    vim \ 
    tmux \ 
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Add ROS 2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list

# Install ROS 2 Humble Desktop Full
RUN apt update && apt install -y ros-humble-desktop-full \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y ros-dev-tools \
    && rm -rf /var/lib/apt/lists/*

# Source ROS 2 setup script
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

RUN mkdir -p /ros_ws/src
WORKDIR /ros_ws

# Set entrypoint
CMD ["/bin/bash"]

