# TAUV-ROS-Packages 
[![Build Status](https://travis-ci.com/Tartan-AUV/TAUV-ROS-Packages.svg?token=FrwKiSXG1hQbYsyh6LNc&branch=master)](https://travis-ci.com/Tartan-AUV/TAUV-ROS-Packages)

This is the monorepo for all TAUV ROS packages. Each package contains its own ROS nodes or other common code.

*Disclaimer: whenever `/path/to/TAUV-ROS-Packages/` is written, this should be the file path to the location of the `TAUV-ROS-Packages` cloned directory on your local computer.

# Setup
The code in this repo runs ROS Noetic on Ubuntu 20.04. Below find instructions for getting this environment installed on your computer. Either use Google or [the running setup debugging list](https://github.com/Tartan-AUV/TAUV-ROS-Packages/wiki/Installation-and-Setup-Debugging-Help) for issues encountered along the way.

## Ubuntu 20.04 + ROS Noetic Install
The easiest way to get up and running is through virtualization of an Ubuntu 20.04 environment. An alternative and often higher-performance solution (if your hardware supports it) is dual-booting your native OS and Ubuntu 20.04. [Google Drive](https://drive.google.com/drive/folders/1KdAnfuahlqWfyaMwDTwxQsJ7HDAdvSMs?usp=sharing) containing VM images -- download and save it someplace safe!

**M1 Mac Users**: Download [UTM VM Software](https://mac.getutm.app) and the `.utm` file from the Google Drive

**Windows 11 Users**: Download [Oracle VirtualBox](https://www.virtualbox.org) and the `.ova` file from the Google Drive

These VMs have ROS Noetic installed on Ubuntu 20.04. The username, computer name, and password are all `sam` (Submarine Application Machine!).

### M1 Mac Users
1. Open up UTM
2. Click "Create a New Virtual Machine"
3. Under the "Existing" tab, click "Open"
4. Find the `.utm` file you downloaded
5. Launch the VM

### Windows 11 Users
TODO

## Configure `git`
Sign up for GitHub if you don't already have an account. Follow [this tutorial](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account#:~:text=In%20the%20%22Access%22%20section%20of,this%20key%20%22Personal%20laptop%22.) for adding an SSH key to your account. Once set up, configure your username and email:
```bash
git config --global user.name "Submarine Guy"
git config --global user.email submarine@guy.com
```

In the home directory of your VM there should be a folder called `catkin_ws` (`/home/sam/catkin_ws`). All ROS code, whether from this repository or another, should live in the `src` sub-directory of the `catkin_ws` folder. Navigate to the `src` directory and clone the repository:
```bash
cd ~/catkin_ws/src
git clone --recurse-submodules git@github.com:Tartan-AUV/TAUV-ROS-Packages.git
```

## Editing the `~/.bashrc`
If you followed the ROS installation tutorial, this line might already be in your `./bashrc` file. If not, you should:
```bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
```

# Darknet installation
To use darknet in sim, you will need to [build it from source](https://github.com/leggedrobotics/darknet_ros).
```bash
# <path to catkin_ws/src>
git clone --recurse-submodules git@github.com:leggedrobotics/darknet_ros.git
catkin build darknet_ros -DCMAKE_BUILD_TYPE=Release
```

# Building Your ROS Project
When you make changes like adding new message types or add new dependencies to a CMake file, etc. you must rebuild the package with:
```bash
cd <path to catkin_ws>
catkin build
source devel/setup.bash
```
    
If the above `catkin build` command fails, try toubleshooting using these answers: https://github.com/catkin/catkin_tools/issues/525

# The Setup Script - THIS NEEDS TO BE FIXED
You need to `source devel/setup.zsh` every time you build and every time you open a terminal. This is annoying. Consider adding:
```bash
source <path to TAUV-ROS-Packages/aliases.sh>
tauvsh
```
to your `~/.zshrc`. This will automatically source it. The `aliases.sh` file exposes three nice commands you can run from anywhere (not just the base of the repo:

 * `tauvsh` sources devel/setup.zsh, allowing you to use ros shell commands.
 * `tauvclean` cleans the build and devel folders. Consider running if you have weird build errors and need to build from scratch
 * `tauvmake` builds the repo.

# Conventions
We use NED for most things. (If you see ENU somewhere, flag it since we should update all code to be consistent with the NED frame system)
![NED Frame](https://www.researchgate.net/publication/324590547/figure/fig3/AS:616757832200198@1524057934794/Body-frame-and-NED-frame-representation-of-linear-velocities-u-v-w-forces-X-Y-Z.png)

TODO: move this somewhere else

# Dependencies

ROS Package dependencies MUST be acyclic. Therefore, only create new ros packages when you really want to encapsulate something that does not need to be tightly coupled to the rest of the system.

Current dependency tree:

```
tauv_mission
- tauv_common
- tauv_vehicle
	- tauv_common
```

TODO: is this even accurate..?

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
