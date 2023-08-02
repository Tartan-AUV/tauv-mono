import rospy
from spatialmath import SE2
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult


class TorpedoStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1


@dataclass
class TorpedoResult(TaskResult):
    status: TorpedoStatus


class Torpedo(Task):

    def __init__(self):
        super().__init__()

    def run(self, resources: TaskResources) -> TorpedoResult:
        # Look for closest circle to sub

        # While true,
        # Look for closest circle to target circle
        # Compute position along circle normal with distance controlled by current distance from circle normal
        # Goto this position
        # Sleep
        # Once we reach target, shoot
        pass

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()