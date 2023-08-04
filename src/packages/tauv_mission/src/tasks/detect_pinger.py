import time
import rospy
from spatialmath import SE3, SE3, SO3
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import numpy as np

class DetectPingerStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1

@dataclass
class DetectPingerResult(TaskResult):
    status: DetectPingerStatus

class DetectPinger(Task):

    def __init__(self):
        super().__init__()

    def run(self, resources: TaskResources): -> DetectPingerResult:
        pass

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()