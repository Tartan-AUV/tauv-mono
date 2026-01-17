FROM base AS desktop

ARG TARGETARCH
ARG NVIM_VERSION=v0.11.5


### INSTALL MISC TOOLS & STONEFISH DEPS (APT) ###

RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    less \
    pipx \
    wget \
    wl-clipboard \
    ripgrep \
    tmux \
    locales \
    unzip \
    clangd \
    clang-format \
    clang-tidy \
    libglm-dev \
    libsdl2-dev \
    libfreetype6-dev \
  && rm -rf /var/lib/apt/lists/*


### INSTALL MISC TOOLS (PIPX) ###

RUN pipx install \
    pre-commit \
    ruff \
    pyright

ENV PATH=${PATH}:/root/.local/bin


### LOCALES ###

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG=en_US.UTF-8  
ENV LANGUAGE=en_US:en  
ENV LC_ALL=en_US.UTF-8  


### CONFIGURE GIT ### 

RUN git config --global --add safe.directory /tauv-mono &&\
    git config --global push.autoSetupRemote true


### NEOVIM ###

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


### MISC CONFIGURATION ###

# Set colcon config for sim builds 
ENV COLCON_DEFAULTS_FILE=/tauv-mono/ros_ws/colcon_defaults.sim.yaml

WORKDIR /tauv-mono/ros_ws
CMD ["bash"]
