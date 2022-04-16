mkfile_dir := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))

default: ros

###############
## ROS Build ##
###############
.PHONY: ros
ros:
	catkin build


##################
## Dependencies ##
##################
.PHONY: deps
deps:
	setup/toolchain-setup.sh
	setup/apt-setup-base.sh
	setup/apt-setup-sim.sh
	setup/pip-setup.sh
