#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
source /opt/ros/overlay/setup.bash

if [ -f /tauv-mono/ros_ws/install/setup.bash ]; then
  source /tauv-mono/ros_ws/install/setup.bash
fi

exec "$@"

