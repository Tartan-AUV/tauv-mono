FROM base AS osprey_orin

# Source ROS2 Humble environment
RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc
