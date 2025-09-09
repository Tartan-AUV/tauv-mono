# User-specific configuration layer for the desktop image
# This layer is intentionally minimal and intended for per-user customization.
# We currently set Git identity if provided via build args; you can extend this
# with more user-specific config later.

FROM base AS desktop_nogpu_user

# Build-time arguments for Git identity
ARG GIT_USER_NAME
ARG GIT_USER_EMAIL

# Apply git config only if args are provided
RUN set -eux; \
    if [ -n "${GIT_USER_NAME:-}" ]; then git config --global user.name "${GIT_USER_NAME}"; fi; \
    if [ -n "${GIT_USER_EMAIL:-}" ]; then git config --global user.email "${GIT_USER_EMAIL}"; fi

WORKDIR /tauv-mono/ros_ws
