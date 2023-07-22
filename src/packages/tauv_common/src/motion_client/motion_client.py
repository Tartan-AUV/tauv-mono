import rospy
from enum import IntEnum
from spatialmath import SE3

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

    def goto(self, pose: SE3, v_max: float, a_max: float, flat: bool = True):
        # Make a trapezoidal velocity curve trajectory from current pose and velocity to target pose with zero velocity
        # Relax a_max if necessary so trajectory is always feasible
        pass

    def goto_relative(self, pose: SE3, v_max: float, a_max: float, flat: bool = True):
        # Apply pose to current pose
        # Then call goto
        pass

    def cancel(self):
        # Reset target to current position
        pass