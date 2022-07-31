from motion.trajectories.trajectories import TrajectoryStatus
import rospy
from tauv_mission_manager.mission_utils import Task, TaskParams

class Dive(Task):
    def __init__(self, params: TaskParams) -> None:
        self.mu = params.mu
        self.cancelled = False
        self.status = params.status

    def run(self, depth):
        if (self.cancelled): return
        

        self.status(f"Starting trajectory! ETA: {traj.get_duration():.1f}s")
        self.do_traj(traj)
        self.status(f"Dive done :)")
            
    def cancel(self):
        self.mu.abort()
        self.cancelled = True
