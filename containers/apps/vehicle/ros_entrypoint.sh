#!/bin/bash
set -e

ros_env_setup="/opt/ros/$ROS_DISTRO/setup.bash"
ros_pkg_dir="/opt/tauv/packages/setup.bash"
echo "sourcing   $ros_env_setup"
source "$ros_env_setup"

echo "sourcing $ros_pkg_dir" 
source "$ros_pkg_dir"

echo "ROS_ROOT   $ROS_ROOT"
echo "ROS_DISTRO $ROS_DISTRO"

exec "$@"