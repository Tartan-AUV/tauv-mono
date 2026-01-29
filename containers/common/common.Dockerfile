FROM base AS common

# System dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-robot-localization \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip tooling (optional but recommended)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# All Python packages live in the venv
RUN pip install --no-cache-dir \
    pandas \
    matplotlib \
    scipy \
    spatialmath-python