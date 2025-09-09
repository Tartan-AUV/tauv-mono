# Layer that applies per-user Git config on top of desktop_nogpu
# IMPORTANT: This Dockerfile is built with Buildx Bake target "user_config"
# and maps its "base" context to the previously built "desktop_nogpu" target.

FROM base AS user_config

# Build-time arguments for Git identity
ARG GIT_USER_NAME
ARG GIT_USER_EMAIL

# Apply git config only if args are provided
RUN set -eux; \
    if [ -n "${GIT_USER_NAME:-}" ]; then git config --global user.name "${GIT_USER_NAME}"; fi; \
    if [ -n "${GIT_USER_EMAIL:-}" ]; then git config --global user.email "${GIT_USER_EMAIL}"; fi

# Keep the same working directory convention
WORKDIR /tauv-mono/ros_ws
