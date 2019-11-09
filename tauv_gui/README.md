# tauv_gui
This is the spot for GUI code that is intended to run off-system on the command computer. ROS_multimaster_fkie is used to synchronize the GUI and the vehicle. Nearly all topics, services, and frames are synchronized between the two nodes, as if they were on the same ROS network.

This folder contains multiple packages, including the tauv_gui package itself, as well as additional packages for any rqt plugins, and external packages such as joystick drivers.

# Multimaster Setup
To install, you need to clone/build the whole repo, just as described in the root README file.
Currently the GUI is only supported on **Ubuntu**.
**If you are only running locally on the simulator, you can skip all these steps.**
Once installed, you need to prepare your system to support multimaster, which involves 4 things:

## Manual setup

 - Enable IP forwarding on your system
 - Enable multicast on your system
 - Verify avahi dns resolution
	 - Run `avahi-resolve-host-name $HOST.local` and verify that it prints your hostname.local and an ip address. (might be ipv6)
	 - If the command times out, you need to [setup avahi.](http://wiki.ros.org/ROS/NetworkSetup#Using_machinename.local) If avahi is active, but the command still fails, try running `avahi-resolve-address <ip_address>` with your ip address from ifconfig. If you have underscores or other invalid characters, Avahi will omit them from your URI name and it will break the system. Please [change your hostname](https://www.cyberciti.biz/faq/ubuntu-change-hostname-command/) to something with just letters, dashes, and numbers.
 - Adjust ROS environment variables
	 - You need to add two lines to your bashrc (or zshrc) to tell ROS to use the Avahi (zeroconf) hostname instead of localhost for the ros master's URI. This will enable the vehicle to listen to published topics without needing to reconfigure routing or DNS settings on the vehicle:
	 - `export ROS_HOSTNAME=$HOST.local`
	 - `export ROS_MASTER_URI=http://$HOST.local:11311`

## Simple setup script
To avoid doing this by hand, I created a handy script `gui_setup.bash` (or `gui_setup.zsh`) that will do these steps for you. The script only temporarily enables IP forwarding and multicast, so you will have to either run it each time you reboot, or follow a tutorial to permanently enable those settings. (It isn't hard.)
To run:

    cd tauv-ros-packages/tauv_gui
    ./gui_setup.bash # if your're using bash, otherwise:
    ./gui_setup.zsh # if you use zsh

Congrats, you should have multimaster working now with a minimally invasive setup. Talk to Tom if you need help or if things aren't working. Possible issues:

## Troubleshooting multimaster

 - Default gateway is not the same as the vehicle's gateway
	 - Make sure your default gateway is the robosub router, and that you're using the correct interface.
	 - If you are wired to the router, try disabling wifi.
 - Avahi is not working
	 - Make sure your hostname is only valid alphanumeric characters and dashes
 - I can see the topics with rostopic list, but there's no data
	 - This means there's a routing error between your computer and the vehicle, which usually means that avahi is not running or that your ROS_MASTER_URI is not set to your avahi hostname. If you echo $ROS_MASTER_URI, you should see http://\<hostname\>.local:11311.
 - I can't see the topics, and the sub is just not connected
	 - Make sure that master_discovery is in fact running on the vehicle
	 - Make sure that both computers can ping each other:
		 - `ping 224.0.0.1` should resolve **both** the vehicle's IP address and your computer's. To check the vehicle's IP address, try running `avahi-resolve-host-name tauv-<vehicle_name>.local`. If that fails, then either the avahi daemon is not running on the sub, or the sub is not connected to the correct gateway. Talk to Tom.

# Launch Files
To launch the gui, use `tauv_gui/launch/gui.launch`. Once RQt is running, you may need to load the perspective `gui.perspective` in the settings.

# Plugins
## joy_vis
This is a quick RQt joystick visualizer. Use it to connect to standard, ubuntu-compatible joysticks such as an xbox360 controller, logitech controller, or wiimote. You may need to use an [external tool](https://jstest-gtk.gitlab.io/) to remap axes if they are not correct.
