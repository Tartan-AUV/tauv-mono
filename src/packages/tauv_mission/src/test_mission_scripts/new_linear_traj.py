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
                position=tm([2, 0, 0], Vector3),
                orientation=rpy_to_quat([0, 0, 1.57])
            ),
            Pose(
                position=tm([2, 2, 0], Vector3),
                orientation=rpy_to_quat([0, 0, 1.57])
            ),
            Pose(
                position=tm([2, 2, 2], Vector3),
                orientation=rpy_to_quat([0, 0, 1.57])
            ),
            Pose(
                position=tm([2, 2, 2], Vector3),
                orientation=rpy_to_quat([0, 0.2, 1.57])
            ),
            Pose(
                position=tm([2, 2, 0], Vector3),
                orientation=rpy_to_quat([0, 0, 1.57])
            ),
            Pose(
                position=tm([2, 2, 0], Vector3),
                orientation=rpy_to_quat([0, 0, 0])
            ),
            Pose(
                position=tm([0, 0, 0], Vector3),
                orientation=rpy_to_quat([0, 0, 0])
            ),
        ]
        self.i = 0
        self.started = False

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(0.1), self._update)
        rospy.spin()

    def _update(self, timer_event):
        if self.mu.get_motion_status() != TrajectoryStatus.STABILIZED and self.started:
            return

        current_pose, current_twist = self.mu.get_robot_state()

        dest_pose = self.poses[self.i]
        dest_twist = Twist()

        try:
            traj = trajectories.NewLinearTrajectory(current_pose, current_twist, dest_pose, dest_twist)
            self.mu.set_trajectory(traj)
        except:
            print('trajectory failed')
            return

        self.i = (self.i + 1) % len(self.poses)
        self.started = True

def main():
    rospy.init_node('test_mission')
    tm = TestMission()
    tm.start()
