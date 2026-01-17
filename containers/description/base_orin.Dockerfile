FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0 AS base

# set debian non-interactive mode
ENV DEBIAN_FRONTEND noninteractive


RUN apt-get update && apt-get install -y \
    python3-pip \
    libopenblas-dev \
    wget \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# Install cuSPARSELt manually
RUN wget -q https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.8.1.1_cuda12-archive.tar.xz && \
    tar -xf libcusparse_lt-linux-aarch64-0.8.1.1_cuda12-archive.tar.xz && \
    cp libcusparse_lt-linux-aarch64-0.8.1.1_cuda12-archive/lib/libcusparseLt.so* /usr/lib/aarch64-linux-gnu/ && \
    cp libcusparse_lt-linux-aarch64-0.8.1.1_cuda12-archive/include/cusparseLt* /usr/include/ && \
    rm -rf libcusparse_lt-linux-aarch64-0.8.1.1_cuda12-archive* libcusparse_lt-linux-aarch64-0.8.1.1_cuda12-archive.tar.xz

# Install PyTorch from NVIDIA wheel
RUN python3 -m pip install --upgrade pip; python3 -m pip install numpy=='1.26.1'; python3 -m pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl


ENV TZ="America/New_York"
ENV DEBIAN_FRONTEND="noninteractive"

# Download and install ArenaSDK from S3
RUN --mount=type=secret,id=aws_credentials \
    mkdir -p /root/.aws && \
    cp /run/secrets/aws_credentials /root/.aws/credentials && \
    mkdir -p /opt/arena && \
    aws s3 cp s3://tauv-build-assets/ArenaSDK_v0.1.78_Linux_ARM64.tar.gz /opt/arena/ArenaSDK.tar.gz

RUN cd /opt/arena/ && \
    tar -xzf ArenaSDK.tar.gz && \
    cd ArenaSDK_Linux_ARM64 && \
    chmod +x Arena_SDK_ARM64.conf && \
    sh Arena_SDK_ARM64.conf && \
    rm /root/.aws/credentials

#DroneCAN
RUN python3 -m pip install dronecan=="1.0.27"
