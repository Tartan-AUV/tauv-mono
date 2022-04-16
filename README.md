
# TAUV-ROS-Packages
[![Build Status](https://travis-ci.com/Tartan-AUV/TAUV-ROS-Packages.svg?token=FrwKiSXG1hQbYsyh6LNc&branch=master)](https://travis-ci.com/Tartan-AUV/TAUV-ROS-Packages)

This is the monorepo for all TAUV ROS packages. Each package contains its own ROS nodes or other common code.

## Setup and File Hierarchy
### This stuff all relies on ROS, so install from here:

http://wiki.ros.org/melodic/Installation/Ubuntu
(Use the full install, since we need gazebo and stuff)

To install:

    cd ~ # or wherever else you want software to live
    git clone --recurse-submodules https://github.com/Tartan-AUV/TAUV-ROS-Packages

### Add this to your bashrc (or zshrc), or run it every boot:

    source /opt/ros/noetic/setup.bash
Or (if you use zsh):

    source /opt/ros/noetic/setup.zsh

### To build your ROS project:

    cd path/to/TAUV_ROS_Packages
    make
    source devel/setup.bash #(or setup.zsh)
    
### Other ways to build:

    catkin build # (This is what "make" calls)
    tauvmake # (If you sourced aliases.sh)
    
### The setup script:
You need to source devel/setup.zsh every time you build and every time you open a terminal. This is annoying. Consider adding:

    source ~/TAUV_ROS_Packages/aliases.sh # (or whatever the path is for you)
    tauvsh
to your ~/.zshrc. This will automatically source it. The aliases.sh file exposes three nice commands you can run from anywhere (not just the base of the repo:

 * `tauvsh` sources devel/setup.zsh, allowing you to use ros shell commands.
 * `tauvclean` cleans the build and devel folders. Consider running if you have weird build errors and need to build from scratch
 * `tauvmake` builds the repo.

## Conventions
We use NED for most things. (If you see ENU somewhere, flag it since we should update all code to be consistent with the NED frame system)
![NED Frame](https://www.researchgate.net/publication/324590547/figure/fig3/AS:616757832200198@1524057934794/Body-frame-and-NED-frame-representation-of-linear-velocities-u-v-w-forces-X-Y-Z.png)

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
This is the package for the operator interface. ROS Multimaster is used to connect the gui to the sub. Both platforms need to be running avahi for this to work, and you need to run the setup script in the gui package before launching it.

## uuv-simulator
This folder contains all of the simulator packages and gazebo plugins necessary to simulate the vehicle. UUV-Simulator depends on the vehicle_description packages to describe gazebo meshes and URDFs including hydrodynamics information.
