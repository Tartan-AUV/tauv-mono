import rclpy
from dataclasses import dataclass
from enum import IntEnum
from mission_manager.task import Task, TaskResources
from motion_client.motion_client import MotionClient


class Dive(Task):

    class Result(IntEnum):
        SUCCESS = 0
        FAILURE = 1

    def __init__(self, depth: float):
        super().__init__()

        self._depth: float = depth

    def run(self, resources: TaskResources) -> Result:
        goto_result = resources.motion.goto_relative(

        )
        if goto_result == MotionClient.Result.CANCELLED:
            return self.Result.FAILURE

        return self.Result.SUCCESS

    def cancel(self, resources: TaskResources):
        resources.motion.cancel()