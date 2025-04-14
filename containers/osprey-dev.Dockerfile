FROM tartanauv/tauv-orin-base:r36.4.3

RUN apt-get update && apt-get install -y \
    vim tmux \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libflatbuffers-dev \
    libflatbuffers2 \
    flatbuffers-compiler \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /ros_ws && cd ros_ws && mkdir install build logs src
WORKDIR /ros_ws
