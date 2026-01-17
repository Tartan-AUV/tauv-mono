# Layer that applies per-user Git config on top of desktop_nogpu

FROM base AS user_config

ARG GIT_USER_NAME
ARG GIT_USER_EMAIL

RUN set -eux; \
    if [ -n "${GIT_USER_NAME:-}" ]; then git config --global user.name "${GIT_USER_NAME}"; fi; \
    if [ -n "${GIT_USER_EMAIL:-}" ]; then git config --global user.email "${GIT_USER_EMAIL}"; fi

WORKDIR /tauv-mo