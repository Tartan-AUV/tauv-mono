import rospy
from motionlib import MotionUtils, trajectories
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from geometry_msgs.msg import Point

class TestMission(object):
    def __init__(self):
        self.mu = MotionUtils()
        # self.run_service = rospy.Service('/mission/test_mission_run', Trigger, self.run)
        self.run(None)

    def update(self, timer_event):
        res = TriggerResponse
        res.success = True
        res.message = "run success!"

        curr_pos, curr_twist = self.mu.get_robot_state()
        traj = trajectories.MinSnapTrajectory(curr_pos, curr_twist, [Point(1, 0, -2)])

        self.mu.set_trajectory(traj)

    def run(self, req):
        rospy.Timer(rospy.Duration.from_sec(0.2), self.update)

def main():
    rospy.init_node('test_mission')
    tm = TestMission()
    rospy.spin()


if __name__ == "__main__":
    main()