import unittest
import rospy
import numpy as np
import matplotlib.pyplot as plt

from geometry_msgs.msg import Pose, Twist, Vector3, Quaternion

from tauv_msgs.srv import GetTrajRequest, GetTrajResponse
from tauv_util.types import tl, tm
from tauv_util.transforms import rpy_to_quat, quat_to_rpy

from .new_linear_trajectory import NewLinearTrajectory
from .new_min_snap_trajectory import NewMinSnapTrajectory


class TestLinearTrajectory(unittest.TestCase):

    def setUp(self):
        rospy.init_node('test_linear_trajectory')

    def test_basic(self):
        start_pose = Pose(
            position=tm([0, 0, 0], Vector3),
            orientation=rpy_to_quat([0, 0, 0])
        )
        start_twist = Twist(
            linear=tm([0.1, 0, 0], Vector3),
            angular=tm([0, 0, 0], Vector3)
        )
        poses = [
            Pose(
                position=tm([1, 1, 1], Vector3),
                orientation=rpy_to_quat([0.0, 0.15, 0.0])
            ),
            Pose(
                position=tm([2, 1, 1], Vector3),
                orientation=rpy_to_quat([0.0, 0.2, 0.0])
            ),
            Pose(
                position=tm([10, 1, 1], Vector3),
                orientation=rpy_to_quat([0.2, 0.0, 0.0])
            ),
        ]

        # traj = NewLinearTrajectory(start_pose, start_twist, end_pose, end_twist)
        traj = NewMinSnapTrajectory(start_pose, start_twist, poses)
        print(traj._position_traj.get_times())

        req: GetTrajRequest = GetTrajRequest()
        req.curr_time = rospy.Time.now()
        req.len = 1000
        req.dt = 0.01
        res: GetTrajResponse = traj.get_points(req)
        print(res)

        self._plot_trajectory(res.poses, res.twists, req.len, req.dt)

    def _plot_trajectory(self, poses, twists, len, dt):
        t = np.linspace(0, len * dt, len, endpoint=False)
        positions = np.array([tl(pose.position) for pose in poses])
        orientations = np.array([quat_to_rpy(pose.orientation) for pose in poses])

        plt.plot(t, positions[:,0])
        plt.plot(t, positions[:,1])
        plt.plot(t, positions[:,2])
        plt.show()

        plt.plot(t, orientations[:,0])
        plt.plot(t, orientations[:,1])
        plt.plot(t, orientations[:,2])
        plt.show()

if __name__ == '__main__':
    unittest.main()
