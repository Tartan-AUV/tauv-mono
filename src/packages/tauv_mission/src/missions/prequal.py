from mission_manager.utils import Mission, TaskParams
from motion.motion_utils import MotionUtils
from tasks.dive import Dive
from tasks.gotorelative import GoToRelative
from enum import Enum
from math import pi

class Prequal(Mission):
    class PrequalState(Enum):
        DIVING = 0
        FORWARD = 1
        BACKWARD = 2
        FINISH = 3

    def __init__(self) -> None:
        self.mu = MotionUtils()
        self.params = TaskParams(self.mu)
        
        self.start_state = self.PrequalState.DIVING

        self.tasks = {
            self.PrequalState.DIVING: Dive(self.params, depth=5),
            self.PrequalState.FORWARD: GoToRelative(self.params, pos=(10, 0, self.mu.get_position()[2]), heading=0),
            self.PrequalState.BACKWARD: GoToRelative(self.params, pos=(-10, 0, self.mu.get_position()[2]), heading=pi)
        }

        self.transitions = {
            self.PrequalState.DIVING: self.diveTransition,
            self.PrequalState.FORWARD: self.forwardTransition,
            self.PrequalState.BACKWARD: self.backwardTransition
        }
    
    def diveTransition(self, status: Dive.DiveStatus) -> PrequalState:
        if status == Dive.DiveStatus.FAILURE:
            return self.PrequalState.DIVING

        return self.PrequalState.FORWARD
    
    def forwardTransition(self, status: GoToRelative.GoToRelativeStatus) -> PrequalState:
        if status == GoToRelative.GoToRelativeStatus.FAILURE:
            return self.PrequalState.FORWARD

        return self.PrequalState.BACKWARD
    
    def backwardTransition(self, status: GoToRelative.GoToRelativeStatus) -> PrequalState:
        if status == GoToRelative.GoToRelativeStatus.FAILURE:
            return self.PrequalState.BACKWARD

        return self.PrequalState.FINISH
    
    def get_finish_state(self):
        return self.PrequalState.FINISH