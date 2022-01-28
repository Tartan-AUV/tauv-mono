import rospy
from motion import MotionUtils, trajectories
from motion.trajectories.trajectories import TrajectoryStatus

class HoldMission:
    def __init__(self):
        self.mu = MotionUtils()

    def start(self):
        # curr_pos, curr_twist = self.mu.get_robot_state()
        # try:
        #     traj = trajectories.LinearTrajectory(curr_pos, curr_twist, [[0, 0, 0]], [[0, 0, 0]], v=0.1, a=0.05)
        #     self.mu.set_trajectory(traj)
        # except:
        #     print('trajectory failed')
        rospy.Timer(rospy.Duration.from_sec(0.1), self._update)
        rospy.spin()

    def _update(self, timer_event):
        if self.mu.get_motion_status() not in [TrajectoryStatus.FINISHED, TrajectoryStatus.TIMEOUT]:
            return

        curr_pos, curr_twist = self.mu.get_robot_state()
        try:
            traj = trajectories.LinearTrajectory(curr_pos, curr_twist, [[0, 0, 0]], [[0, 0, 0]], v=0.1, a=0.05)
            self.mu.set_trajectory(traj)
        except:
            print('trajectory failed')

def main():
    rospy.init_node('hold_mission')
    hm = HoldMission()
    hm.start()
