FROM base AS desktop_nogpu

# Install git, git-lfs
RUN apt-get update && apt-get install -y \
    git git-lfs

# Install pre-commit
RUN python3 -m pip install --break-system-packages \
    pre-commit

ENV COLCON_DEFAULTS_FILE=/ros_ws/src/config/colcon_defaults.sim.yaml

WORKDIR /ros_ws
