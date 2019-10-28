

# TAUV-ROS-Packages

This is the monorepo for all TAUV ROS packages. Each package contains its own ROS nodes or other common code.

## Setup and File Hierarchy
### This stuff all relies on ROS, so install from here:

http://wiki.ros.org/melodic/Installation/Ubuntu
(Use the full install, since we need gazebo and stuff)

### Next, make sure to install catkin_tools:
	sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list'
(If you are not on ubuntu, change 'lsb_release -sc' to 'bionic' without quotes)
	
	wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
	sudo apt-get update
	sudo apt-get install python-catkin-tools
	

### These ROS packages should live in your `catkin_ws` folder. To install:

    cd ~ # or wherever else you want software to live
    mkdir -p catkin_ws/src
    cd catkin_ws/src
    git clone --recurse-submodules https://github.com/Tartan-AUV/TAUV-ROS-Packages
    rosdep update
    sudo apt-get update
    rosdep install --from-paths TAUV-ROS-Packages --ignore-src -r -y

Your folder structure should look something like this:

    - catkin_ws/
      - src/
        - TAUV_ROS_Packages/
          - tauv_common
          - tauv_mission
          - tauv_vehicle
          - uuv_simulator
To update dependencies (eg, after pulling in a large change) run this command again:

    rosdep update
    sudo apt-get update
    rosdep install --from-paths path/to/TAUV-ROS-Packages --ignore-src -r -y
If you are using linux mint (or another unsupported OS), you will need to add the following line to your bashrc and source it again, otherwise commands like rosdep won't work. If your distro doesn't use 18.04 as the upstream, you will need to change this to the right upstream os version.

    export ROS_OS_OVERRIDE=ubuntu:18.04:bionic

### Add this to your bashrc (or zshrc), or run it every boot:

    source /opt/ros/melodic/setup.bash
Or (if you use zsh):

    source /opt/ros/melodic/setup.zsh

### To build your ROS project:

    cd path/to/catkin_ws
    catkin build
    source devel/setup.bash #(or setup.zsh)

## Dependencies

ROS Package dependencies MUST be acyclic. Therefore, only create new ros packages when you really want to encapsulate something that does not need to be tightly coupled to the rest of the system.

Current dependency tree:

    tauv_mission
	    - tauv_common
	    - tauv_vehicle
		    - tauv_common
# Package list
Each package contains a more detailed readme in their folder.

## tauv_common
This package is the common shared nodes between all the other packages. Most other packages will depend on tauv_common, but tauv_common should not depend on any other tauv packages with the exception of tauv_config.
Core frameworks like watchdogs, exceptions, thruster managers, and things like that live in common. In addition, reusable navigation, controls, state estimation, planning, etc code should live here. Common perception code should live here, such as object detection, point cloud registration, and things that can be reused.

## tauv_mission
This is where mission code lives. Mission code is defined as code that is specific to a single mission. For example, code to navigate dice, hit buoys, pick up garlic, or go through the gate is mission-specific. If it is specific to a single competition element, it belongs in here.
In addition to mission-code, the mission package also contains system-wide launchfiles, such as system.launch.

## tauv_vehicle
This is where driver code lives. Abstracts the vehicle from the mission. This package is designed to mirror the behavior of the simulator, but with real hardware. Things like thruster drivers, sensor drivers, etc should live here. Vehicle-specific hardware launchfiles live here as well.

## tauv_config
this folder contains one package for each vehicle as well as a tauv_config package that simply declares dependencies on the other packages. Packages relying on vehicle configuration info should depend on the tauv_config package, and use the model_name arg when determining paths to configuration files. Vehicle_description packages contain URDFs, config yamls, thruster allocation matrices (TAMs), meshes for gazebo, and other vehicle-specific info.

## tauv_gui
This is the package for the operator interface. Currently a WIP.

## uuv-simulator
This folder contains all of the simulator packages and gazebo plugins necessary to simulate the vehicle. UUV-Simulator depends on the vehicle_description packages to describe gazebo meshes and URDFs including hydrodynamics information.
