import time
import rospy
from dataclasses import dataclass
from spatialmath import SE3
from tasks.task import Task, TaskResources, TaskStatus, TaskResult


class GotoStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1


@dataclass
class GotoResult(TaskResult):
    status: GotoStatus


class Goto(Task):

    def __init__(self, pose: SE3):
        super().__init__()

        self._pose: SE3 = pose

    def run(self, resources: TaskResources) -> GotoResult:
        resources.motion.goto(self._pose)

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return GotoResult(GotoStatus.FAILURE)

        return GotoResult(GotoStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()