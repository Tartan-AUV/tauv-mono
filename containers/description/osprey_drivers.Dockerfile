FROM base AS osprey_drivers

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
