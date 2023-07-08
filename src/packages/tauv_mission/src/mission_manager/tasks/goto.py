from dataclasses import dataclass
from typing import Tuple
from mission_manager.task import Task, TaskResources, TaskStatus, TaskResult
from motion_client.motion_client import MotionClient


class GotoStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1


@dataclass
class GotoResult(TaskResult):
    pass


class Goto(Task):

    def __init__(self, position: Tuple[float, float, float]):
        self._position: Tuple[float, float, float] = position

    def run(self, resources: TaskResources) -> GotoResult:
        goto_result = resources.motion.goto(

        )
        if goto_result == MotionClient.Result.CANCELLED:
            return GotoResult(GotoStatus.FAILURE)
        if goto_result == MotionClient.Result.STABILIZATION_TIMED_OUT:
            return GotoResult(GotoStatus.FAILURE)

        return GotoResult(GotoStatus.FAILURE)

    def cancel(self, resources: TaskResources):
        resources.motion.cancel()