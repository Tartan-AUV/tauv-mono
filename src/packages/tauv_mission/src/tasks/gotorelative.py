from mission_manager.utils import Task, TaskParams
from enum import Enum
from typing import Tuple

class GoToRelative(Task):
    class GoToRelativeStatus(Enum):
        SUCCESS = 0
        FAILURE = 1
    
    def __init__(self, params: TaskParams, pos: Tuple[float], heading: float) -> None:
        self.mu = params.mu
        self.pos = pos
        self.heading = heading

    def run(self) -> GoToRelativeStatus:
        if self.mu.goto_relative(self.pos, heading=self.heading):
            return self.GoToRelativeStatus.SUCCESS
        else:
            return self.GoToRelativeStatus.FAILURE