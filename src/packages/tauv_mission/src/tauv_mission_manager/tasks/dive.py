from motion.trajectories.trajectories import TrajectoryStatus
import rospy
from tauv_mission_manager.mission_utils import Task, TaskParams
from motion.trajectories.linear_trajectory import LinearTrajectory, Waypoint

class Dive(Task):
    def __init__(self, params: TaskParams) -> None:
        self.mu = params.mu
        self.cancelled = False
        self.status = params.status

    def run(self, depth):
        pos = self.mu.get_position()
        start = Waypoint(pos)
        target = Waypoint((pos[0], pos[1], depth))
        traj = LinearTrajectory([start, target])

        self.status(f"Starting trajectory! ETA: {traj.get_duration():.1f}s")
        self.do_traj(traj)
            
    def do_traj(self, traj: LinearTrajectory):
        self.mu.set_trajectory(traj)
        while self.mu.get_motion_status() < TrajectoryStatus.FINISHED \
            and not self.cancelled:
            rospy.sleep(rospy.Duration(0.5))
            # TODO: print eta

    def cancel(self):
        self.mu.abort()
        self.cancelled = True
