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

# Configure global git hook to run pre-commit on commit for all repos
# This applies to the mounted repo at /tauv-mono without needing per-repo `pre-commit install`
RUN mkdir -p /root/.git-hooks \
  && printf '%s\n' '#!/usr/bin/env sh' 'exec pre-commit run --hook-stage pre-commit "$@"' > /root/.git-hooks/pre-commit \
  && chmod +x /root/.git-hooks/pre-commit \
  && git config --global core.hooksPath /root/.git-hooks \
  && git config --global --add safe.directory /tauv-mono \
  && git config --global push.autoSetupRemote true
