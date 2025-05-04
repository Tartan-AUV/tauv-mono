#!/bin/bash
cat <<EOF > devcontainer.env
USERNAME=$USER
USER_UID=$(id -u)
USER_GID=$(id -g)
EOF

echo "✅ Wrote .devcontainer.env:"
cat .devcontainer/devcontainer.env
