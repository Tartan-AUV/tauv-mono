import rospy

from geometry_msgs.msg import Pose, Twist, Vector3

from tauv_util.types import tm
from tauv_util.transforms import rpy_to_quat

from motion import MotionUtils, trajectories
from motion.trajectories.trajectories import TrajectoryStatus


class TestMission:

    def __init__(self):
        self.mu = MotionUtils()

        self.poses: [Pose] = [
            Pose(
                position=tm([2, 0, 0], Vector3),
                orientation=rpy_to_quat([0, 0, 0])
            ),
            Pose(
                position=tm([2, 2, 0], Vector3),
                orientation=rpy_to_quat([0, 0, 1.57])
            ),
            Pose(
                position=tm([4, 2, 2], Vector3),
                orientation=rpy_to_quat([0, 0, 1.57])
            ),
            Pose(
                position=tm([5, 2, 2], Vector3),
                orientation=rpy_to_quat([0, 0, 1.57])
            ),
            Pose(
                position=tm([5, -1, 1], Vector3),
                orientation=rpy_to_quat([0, 0, 1.57])
            ),
        ]
        self.started = False

    def start(self):
        # rospy.Timer(rospy.Duration.from_sec(0.1), self._update)
        self._update(None)
        rospy.spin()

    def _update(self, timer_event):
        # if self.mu.get_motion_status() != TrajectoryStatus.STABILIZED and self.started:
        #     return

        self.started = True

        current_pose, current_twist = self.mu.get_robot_state()

        # try:
        print('creating trajectory')
        traj = trajectories.NewMinSnapTrajectory(current_pose, current_twist, self.poses, 0.1, 0.1)
        print(traj._position_traj.get_times())
        print(traj._orientation_traj.get_times())
        print('trajectory created')
        self.mu.set_trajectory(traj)
        # except Exception as e:
        #     print('trajectory failed')
        #     print(e)
        #     return


def main():
    rospy.init_node('test_mission')
    tm = TestMission()
    tm.start()
