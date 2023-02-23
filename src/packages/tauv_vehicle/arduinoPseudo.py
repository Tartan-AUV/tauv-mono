# this is pseudocode for writing the ROS node controlling comms
# between the ROS environment and the Arduino

# node should:
# 1- recognize which service it has received
# 2- know where to publish the result
# 3- have a binary conversion system for the command
# 4- have a system in place to assign a checksum
# 5- know how to process received data from the Arduino
# 6- publish the result

# 1- recognize which service it has received: make a universally agreed 
# upon system assigning request codes to ROS services; also make ROS topics for
# each specific command type;
# (how do we go about doing that? where do we define these inherent 
# system properties? have the ROS services already been implemented?) 


# COMMENTS ON STEP 1: try to find existing communication protocols implemented
# on ROS and what framework they've followed; if ROS topics are not the way to
# go, read about writing ROS services. 

# ANSWERS TO STEP 1: ROS topics are only suitable for continuous data streams
# and so do not suit our purpose -> our node shall be handling ROS services;
# ROS services are provided by a "providing node" and is defined by a pair
# of messages, a "request" and a "reply"; 

# WRITING A ROS SERVICE: how a ROS service works is through the provider node
# (handles requests and defines the callback function); 
# http://wiki.ros.org/rospy/Overview/Services contains more detailed info
# ROS service thus works like any other method, so the problem of knowing where
# post the result is resolved. (YAY)

# IDEA: the node must be a provider of all possible ROS services, and behaviour
# must be defined in the callback functions (MORE YAY); hopefully, I should 
# not have a problem defining multiple services in the same node

################################################################

# 2- know where to publish the result: the ROS topic associated with the
# service (do ROS services even use ROS topics?)


# COMMENTS ON STEP 2: dependent on answer to step 1

################################################################


# 3- have a binary conversion system for the command; (does PySerial have a 
# system for this? If not, how to do this?)


# COMMENTS ON STEP 3: check up on whether codes are to be assigned to 
# ROS services or ROS topics; YES! PySerial does have a system to 
# read serial data as bytes, manipulate them and decode them as string output
# OH NO! we have to deal with concurrent Arduino side functionalities too
# OH NO! we have to consider how the checksum will be added from the ROS
# and Arduino sides; 
# OH NO! we need to figure out the entire process of checksum incorporation
# BACKBURNER: IMPLEMENT CHECKSUM POST PROTOCOL


# 4- have a system in place to assign a checksum (again, do we need a specific
# algorithm here? does PySerial offer some help in this process?)

# 5- know how to process received data from the Arduino (if there is a key
# of ROS topics associated with request codes, can that help here? Does
# PySerial offer some help here?)

# 6- results are published on the ROS topic, depending on whether ROS services
# even use ROS topics or not

# TESTING

# 7- make an assignment to LEDs on a breadboard using the existing
# protocol code framework (if that is the right way to go)

# DEVELOPMENT

# 8- developing on GitHub right at outset has no immediate benefits
