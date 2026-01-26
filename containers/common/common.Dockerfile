FROM base AS common

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-jazzy-robot-localization \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is available for Python package installs
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pandas \
        python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

# PLEASE FIX THIS AT SOME POINT??
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    spatialmath-python \
    scipy
