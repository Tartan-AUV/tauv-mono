import time
from dataclasses import dataclass
from typing import Tuple
from mission_manager.task import Task, TaskResources, TaskStatus, TaskResult


class GotoStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1


@dataclass
class GotoResult(TaskResult):
    status: GotoStatus


class Goto(Task):

    def __init__(self, position: Tuple[float, float, float]):
        super().__init__()

        self._position: Tuple[float, float, float] = position

    def run(self, resources: TaskResources) -> GotoResult:
        print("Running Goto...")
        time.sleep(5.0)
        if self._check_cancel(resources): return GotoResult(GotoStatus.FAILURE)
        return GotoResult(GotoStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        print("Cancelling Goto...")
        time.sleep(1.0)