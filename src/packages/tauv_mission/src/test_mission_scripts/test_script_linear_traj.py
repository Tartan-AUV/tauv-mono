import rospy
from motion import MotionUtils, trajectories
from motion.trajectories import TrajectoryStatus
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from geometry_msgs.msg import Point


class TestMission(object):
    def __init__(self):
        self.mu = MotionUtils()
        self.points = [[2, 0, 0], [2, 0, 0], [0, 0, 0]]
        self.orientations = [[0, 0, 0], [0.001, 0.001, 1.57], [0, 0, 0]]
        # self.points = [[0, 0, 0], [0, 0, 0]]
        # self.orientations = [[0.5, 0, 0], [0, 0, 0]]
        self.i = 0
        self.active = 0
        curr_pos, curr_twist = self.mu.get_robot_state()
        print(curr_pos, curr_twist)
        self.traj = trajectories.LinearTrajectory(curr_pos, curr_twist, [self.points[0]], [self.orientations[0]], v=0.1, a=0.05)
        self.mu.set_trajectory(self.traj)
        self.run()

    def update(self, timer_event):
        if self.mu.get_motion_status() == TrajectoryStatus.STABILIZED:
            curr_pos, curr_twist = self.mu.get_robot_state()

            print(curr_pos)
            print(curr_twist)

            if self.active == self.i:
                self.i = (self.i + 1) % len(self.points)
            try:
                self.traj = trajectories.LinearTrajectory(curr_pos, curr_twist, [self.points[self.i]], [self.orientations[self.i]], v=0.1, a=0.05)
                self.mu.set_trajectory(self.traj)
                self.active = self.i
            except:
                print('bad trajectory')

    def run(self):
        rospy.Timer(rospy.Duration.from_sec(0.2), self.update)


def main():
    rospy.init_node('test_mission')
    tm = TestMission()
    rospy.spin()


if __name__ == "__main__":
    main()