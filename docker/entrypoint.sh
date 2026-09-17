#!/usr/bin/env bash
set -e

source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"

cd /home/ros
colcon build --symlink-install

source /home/ros/install/setup.bash

exec "$@"
