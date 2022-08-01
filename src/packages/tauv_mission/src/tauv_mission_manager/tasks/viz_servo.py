from importlib.abc import Finder
from motion.trajectories.trajectories import TrajectoryStatus
import rospy
from tauv_mission_manager.mission_utils import Task, TaskParams
from vision.detectors.finder import Finder

class VizServo(Task):
    def __init__(self, params: TaskParams) -> None:
        self.mu = params.mu
        self.cancelled = False
        self.status = params.status
        self.f = Finder()

    def run(self, tag):
        if (self.cancelled): return
        
        self.status(f"Looking for a {tag}")
        hunting = True
        while hunting:
            dets:  = self.f.find_by_tag(tag)
            # Choose the right buouy

        
        # self.mu.goto_relative((0.5, 0, depth))
        # self.status(f"Dive done :)")
            
    def cancel(self):
        self.mu.abort()
        self.cancelled = True
