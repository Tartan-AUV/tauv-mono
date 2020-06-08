# arming_server.py
#
# NOTE: FOR SIMULATION ONLY!
#
# This is a simple server for the /arm service to allow clients to successfully connect to an arming server.
# This server has no current functionality, and only serves as a placeholder for use in simulation runs.
# This should *ONLY* be used in simulation!!
# On real hardware, the actuators.py node provides an arming server that actually arms/disarms the real thrusters.
#
# Author: Tom Scherlis

import rospy
from std_srvs.srv import SetBool, SetBoolResponse


class ArmingServer:
    def __init__(self):
        rospy.Service('/arm', SetBool, self.srv_arm)

    def srv_arm(self, req):
        res = SetBoolResponse()
        res.success = False
        res.message = "Arm/Disarm not supported on this hardware!"
        # TODO: use the "set_thruster_state" service to actually arm/disarm the simulated sub


def main():
    armserver = ArmingServer()
    rospy.spin()