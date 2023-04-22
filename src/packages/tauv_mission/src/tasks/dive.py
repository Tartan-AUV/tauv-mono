from mission_manager.utils import Task, TaskParams
from enum import Enum

class Dive(Task):
    class DiveStatus(Enum):
        SUCCESS = 0
        FAILURE = 1
    
    def __init__(self, params: TaskParams, depth: float) -> None:
        self.mu = params.mu
        self.depth = depth
    
    def run(self) -> DiveStatus:
        if self.mu.goto_relative((0, 0, self.depth)) == None:
            return self.DiveStatus.SUCCESS
        else:
            return self.DiveStatus.FAILURE