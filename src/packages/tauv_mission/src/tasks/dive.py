import time
import rospy
from spatialmath import SE2
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
        resources.motion.goto_relative_with_depth(SE2(), self._depth)

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return DiveResult(DiveStatus.FAILURE)

        return DiveResult(DiveStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()

