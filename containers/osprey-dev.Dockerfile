FROM tartanauv/tauv-orin-base:r36.4.3

RUN apt-get update && apt-get install -y \
    vim tmux \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
