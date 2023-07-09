import time
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult


class DiveStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1


@dataclass
class DiveResult(TaskResult):
    status: DiveStatus


class Dive(Task):


    def __init__(self, depth: float):
        super().__init__()

        self._depth: float = depth

    def run(self, resources: TaskResources) -> DiveResult:
        time.sleep(5.0)
        if self._check_cancel(resources): return DiveResult(DiveStatus.FAILURE)
        return DiveResult(DiveStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        time.sleep(1.0)
