
# TAUV-ROS-Packages

This is the monorepo for all TAUV ROS packages. Each package contains its own ROS nodes or other common code.

## Setup and File Hierarchy
### This stuff all relies on ROS, so install from here:

http://wiki.ros.org/melodic/Installation/Ubuntu
(Use the full install, since we need gazebo and stuff)

### These ROS packages should live in your `catkin_ws` folder. To install:

    cd ~ # or wherever else you want software to live
    mkdir -p catkin_ws/src
    cd catkin_ws/src
    git clone --recurse-submodules https://github.com/Tartan-AUV/TAUV-ROS-Packages
    
Your folder structure should look something like this:

    - catkin_ws/
	- src/
		- TAUV_ROS_Packages/
			- tauv_common
			- tauv_mission
			- tauv_vehicle
			- uuv_simulator

### Add this to your bashrc (or zshrc), or run it every boot:

    source /opt/ros/melodic/setup.bash
Or (if you use zsh):

    source /opt/ros/melodic/setup.zsh

### To build your ROS project:

    cd path/to/catkin_ws
    catkin build vortex_msgs
    catkin build

## Dependencies

ROS Package dependencies MUST be acyclic. Therefore, only create new ros packages when you really want to encapsulate something that does not need to be tightly coupled to the rest of the system.

Current dependency tree:

    tauv_mission
	    - tauv_common
	    - tauv_vehicle
		    - tauv_common
# Package list
## tauv_common
This package is the common shared nodes between all the other packages. Most other packages will depend on tauv_common, but tauv_common should not depend on any other tauv packages.
For example, exception and watchdog stuff should live here, as well as ui interfacing junk.
## tauv_mission
This is where autonomy and mission code lives

## tauv_vehicle
This is where controls code and driver code lives. Abstracts the vehicle from the mission.

## tauv_commander
This is the package for the operator interface. TODO.

## uuv-simulator
This is the package that handles the simulator.
