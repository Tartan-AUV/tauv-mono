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
from nav_msgs.msg import Path, Odometry
from trajectories import Trajectory, TrajectoryStatus

class MotionUtils:
    def __init__(self):
        self.initialized = False
        self.traj_service = rospy.Service('/gnc/get_traj', GetTraj, self._traj_callback)

        self.arm_proxy = rospy.ServiceProxy('/arm', SetBool)
        self.traj = None
        self.path_pub = rospy.Publisher('/gnc/path_viz', Path, queue_size=10)

        self.pose = None
        self.twist = None
        self.odom_sub = rospy.Subscriber('/gnc/odom', Odometry, self._odom_callback)

        # 10Hz status update loop:
        rospy.Timer(rospy.Duration.from_sec(0.1), self._update_status)
        while not self.initialized:
            rospy.sleep(0.05)

    def set_trajectory(self, traj):
        assert isinstance(traj, Trajectory)
        self.traj = traj

    def get_robot_state(self):
        return self.pose, self.twist

    def get_motion_status(self):
        if self.traj is None:
            return TrajectoryStatus.TIMEOUT

    def arm(self, armed):
        self.arm_proxy(SetBoolRequest(armed))

    def _update_status(self, timer_event):
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = rospy.Time.now()

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

    def _odom_callback(self, msg):
        assert(isinstance(msg, Odometry))
        self.pose = msg.pose.pose
        self.twist = msg.twist.twist
        self.initialized=True