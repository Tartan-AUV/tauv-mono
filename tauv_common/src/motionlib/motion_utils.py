# motion_utils.y
#
# Simple python library intended to expose motion control of the AUV to mission scripts.
# Only one MotionUtils class should be instantiated at a time. (TODO: make this a python singleton)
# MotionUtils is intended to be used in conjunction with the mpc_trajectory_follower planner.
#
#
# To use: one MotionUtils instance should be created at sub startup, then passed to any mission scripts
# from the mission manager node.
# The MotionUtils class will automatically provide updates to the trajectory follower.
# Use update_trajectory to update the reference trajectory object to be used by motion_utils.
#
#

import rospy
from tauv_msgs.srv import GetTraj, GetTrajResponse
from std_srvs.srv import SetBool, SetBoolRequest
from nav_msgs.msg import Path
from trajectories import Trajectory, TrajectoryStatus


class MotionUtils:
    def __init__(self):
        self.traj_service = rospy.Service('/gnc/trajectory_service', GetTraj, self._traj_callback)

        self.arm_proxy = rospy.ServiceProxy('/arm', SetBool)
        self.traj = None
        self.path_pub = rospy.Publisher('/gnc/path_viz', Path)

        # 10Hz status update loop:
        rospy.Timer(rospy.Duration.from_sec(0.1), self._update_status())

    def update_trajectory(self, traj):
        assert isinstance(traj, Trajectory)
        self.traj = traj

    def arm(self, armed):
        self.arm_proxy(SetBoolRequest(armed))

    def _update_status(self, timer_event):
        path = Path()
        path.header.frame_id = 'odom'

        if self.traj is not None:
            path = self.traj.as_path(dt=0.1)

        self.path_pub.publish(path)

        # TODO: also post current status, such as eta for current trajectory, percent done, etc.

    def _traj_callback(self, req):
        response = GetTrajResponse()
        if self.traj is None:
            response.success = False
            return response

        return self.traj.get_points(req)
