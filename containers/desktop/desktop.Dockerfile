FROM base AS desktop_nogpu

ARG TARGETARCH
ARG NVIM_VERSION=v0.11.5

#Install git, git - lfs
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    less \
    wl-clipboard \
    ripgrep \
    tmux \
    locales \
    unzip \
    clangd \
  && rm -rf /var/lib/apt/lists/*

# Set the locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG=en_US.UTF-8  
ENV LANGUAGE=en_US:en  
ENV LC_ALL=en_US.UTF-8  

# Install pre-commit
RUN python3 -m pip install --break-system-packages \
    pre-commit

# Ensure interactive git commands open an editor inside the container
ENV EDITOR=nvim
ENV VISUAL=nvim
ENV GIT_EDITOR=nvim

# install neovim
RUN set -eux; \
    case "${TARGETARCH}" in \
      amd64)  NVIM_ARCH="x86_64" ;; \
      arm64)  NVIM_ARCH="arm64" ;; \
      *) echo "Unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/nvim.tar.gz \
      "https://github.com/neovim/neovim/releases/download/${NVIM_VERSION}/nvim-linux-${NVIM_ARCH}.tar.gz"; \
    tar -C /opt -xzf /tmp/nvim.tar.gz; \
    rm -f /tmp/nvim.tar.gz; \
    ln -sf /opt/nvim-linux*/bin/nvim /usr/local/bin/nvim; \
    nvim --version | head -n 2

#New workspace layout: repository mounted at /tauv-mono, workspace in /tauv-mono/ros_ws
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

RUN cat <<'EOF' > /etc/profile.d/ros2_foxglove.sh
#!/usr/bin/env bash

if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi

case "$-" in
  *i*) ;;
  *) return 0 ;;
esac

if [ -z "${FOXGLOVE_BRIDGE_DISABLE:-}" ] && command -v ros2 >/dev/null 2>&1; then
  if ros2 pkg prefix foxglove_bridge >/dev/null 2>&1; then
    if [ ! -f /tmp/foxglove_bridge.started ]; then
      touch /tmp/foxglove_bridge.started
      ros2 run foxglove_bridge foxglove_bridge >/tmp/foxglove_bridge.log 2>&1 &
    fi
  fi
fi
EOF

RUN chmod +x /etc/profile.d/ros2_foxglove.sh
