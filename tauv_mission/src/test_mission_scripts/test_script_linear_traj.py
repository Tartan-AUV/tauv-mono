import rospy
from motionlib import MotionUtils, trajectories
from motionlib.trajectories import TrajectoryStatus
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from geometry_msgs.msg import Point


class TestMission(object):
    def __init__(self):
        self.mu = MotionUtils()
        self.points = [[2, -2, -1], [1, 1, -1], [-3, 1, -1], [-1, -1, -1]]
        self.i = 0
        curr_pos, curr_twist = self.mu.get_robot_state()
        self.traj = trajectories.LinearTrajectory(curr_pos, curr_twist, [self.points[self.i]])
        self.mu.set_trajectory(self.traj)
        self.run()

    def update(self, timer_event):
        if self.mu.get_motion_status() == TrajectoryStatus.FINISHED \
                or self.mu.get_motion_status == TrajectoryStatus.STABILIZED:
            curr_pos, curr_twist = self.mu.get_robot_state()
            self.i = (self.i + 1) % len(self.points)
            self.traj = trajectories.LinearTrajectory(curr_pos, curr_twist, [self.points[self.i]])
            self.mu.set_trajectory(self.traj)

    def run(self):
        rospy.Timer(rospy.Duration.from_sec(0.2), self.update)


def main():
    rospy.init_node('test_mission')
    tm = TestMission()
    rospy.spin()


if __name__ == "__main__":
    main()