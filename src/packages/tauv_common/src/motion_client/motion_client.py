import rospy
from enum import IntEnum

from tauv_msgs.srv import GetTrajectory
from std_msgs.srv import SetBool

class MotionClient:

    def __init__(self):
        self._get_trajectory_server: rospy.Service = rospy.Service('gnc/get_trajectory', GetTrajectory, self._handle_get_trajectory)

        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('vehicle/thrusters/arm', SetBool)

    def _handle_get_trajectory(self, req: GetTrajectory.Request) -> GetTrajectory.Response:
        pass

    def arm(self, arm: bool):
        self._arm_srv(arm)

    def goto(self):
        pass

    def goto_relative(self):
        pass

    def cancel(self):
        # Reset target to current position
        pass