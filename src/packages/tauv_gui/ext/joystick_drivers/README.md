# ROS Joystick Drivers Stack #

A simple set of nodes for supporting various types of joystick inputs and producing ROS messages from the underlying OS messages.

# TAUV Modifications
This is copied from the main ROS Joy source, with a few modifications to support runtime connect/disconnect for joysticks. This adds three services: connect, disconnect, and shutdown. These services allow the node to be reconfigured dynamically. The device-related rosparams are no longer used. Instead, devices are specified in the "dev" field of the connect service.
