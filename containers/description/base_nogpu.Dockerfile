FROM ubuntu:jammy AS base

ENV TZ="America/New_York"
ENV DEBIAN_FRONTEND="noninteractive"


### INSTALL PIP ###

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


### INSTALL TORCH ###

RUN python3 -m pip install --no-cache-dir --target /opt/python-extras \ 
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.9.0

