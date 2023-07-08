import rospy
from enum import IntEnum

from tauv_msgs.srv import GetTrajectory
from std_msgs.srv import SetBool

class MotionClient:

    class GotoResult(IntEnum):
        pass

    def __init__(self):
        self._get_trajectory_server: rospy.Service = rospy.Service('gnc/get_trajectory', GetTrajectory, self._handle_get_trajectory)

        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('vehicle/thrusters/arm', SetBool)

    def _handle_get_trajectory(self, req: GetTrajectory.Request) -> GetTrajectory.Response:
        pass

    def goto(self):
        pass

    def cancel(self):
        pass