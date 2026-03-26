#!/usr/bin/env bash
# Rebuild the ROS2 workspace and source the install.
#
# MUST be sourced so the install stays active in your shell:
#   source /tauv-mono/ros_ws/build.sh
#
# Optional: pass colcon args through, e.g.
#   source build.sh --packages-select tauv_autonomy tauv_sim

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

COLCON_DEFAULTS_FILE=colcon_defaults.sim.yaml colcon build "$@"

source "$SCRIPT_DIR/install/setup.bash"

echo "Build complete. Install sourced."
