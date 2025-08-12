FROM tartanauv/tauv-orin-base:r36.4.3

RUN apt-get update && apt-get install -y \
    vim tmux \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libflatbuffers-dev \
    libflatbuffers2 \
    flatbuffers-compiler \
    unzip \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY ./arena-sdk/ /arena-sdk
WORKDIR /arena-sdk
RUN mkdir arena-python && \
    unzip arena_api-2.7.1-py3-none-any.zip -d arena-python/ && \
    tar -xf ArenaSDK_v0.1.78_Linux_ARM64.tar.gz && \
    chmod +x ArenaSDK_Linux_ARM64/Arena_SDK_ARM64.conf && \
    ./ArenaSDK_Linux_ARM64/Arena_SDK_ARM64.conf && \
    python3 -m pip install arena-python/arena_api-2.7.1-py3-none-any.whl --break-system-packages

RUN echo "alias 'build'='colcon build --symlink-install --merge-install'" >> /root/.bashrc
    
RUN mkdir /ros_ws
WORKDIR /ros_ws
