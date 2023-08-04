import time
import rospy
from spatialmath import SE2, SE3, SO3
import numpy as np
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult


class ScanTranslateStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1


@dataclass
class ScanTranslateResult(TaskResult):
    status: ScanTranslateStatus


class ScanTranslate(Task):

    def __init__(self, x_range: float, y_range: float):
        super().__init__()

        self._x_range: float = x_range
        self._y_range: float = y_range

    def run(self, resources: TaskResources) -> ScanTranslateResult:
        odom_t_vehicle_initial = resources.transforms.get_a_t_b('kf/odom', 'kf/vehicle')

        for i in range(4):
            for j in range(4):
                x = ((i / 4) - 0.5) * self._x_range
                y = ((i / 4) - 0.5) * self._y_range

                resources.motion.goto(odom_t_vehicle_initial * SE3.Rt(SO3(), np.array([x, y, 0])))

                while True:
                    if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                        break

                    if self._check_cancel(resources): return ScanTranslateResult(ScanTranslateStatus.FAILURE)

                time.sleep(5.0)

        return ScanTranslateResult(ScanTranslateStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()