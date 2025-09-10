FROM base AS desktop_nogpu

# Install git, git-lfs
RUN apt-get update && apt-get install -y \
    git git-lfs \
    vim \
    less \
  && rm -rf /var/lib/apt/lists/*

# Install pre-commit
RUN python3 -m pip install --break-system-packages \
    pre-commit

# Ensure interactive git commands open an editor inside the container
ENV EDITOR=vim
ENV VISUAL=vim
ENV GIT_EDITOR=vim

# New workspace layout: repository mounted at /tauv-mono, workspace in /tauv-mono/ros_ws
ENV COLCON_DEFAULTS_FILE=/tauv-mono/ros_ws/colcon_defaults.sim.yaml

WORKDIR /tauv-mono/ros_ws

# Install C++ and Python tooling for pre-commit hooks
# - clang-format, clang-tidy from apt
# - ruff, pyright from pip
RUN apt-get update && apt-get install -y \
    clang-format \
    clang-tidy \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages \
    ruff \
    pyright
