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
        self.mu.goto_relative((0.5, 0, depth), )
        self.status(f"Dive done :)")
            
    def cancel(self):
        self.mu.abort()
        self.cancelled = True
