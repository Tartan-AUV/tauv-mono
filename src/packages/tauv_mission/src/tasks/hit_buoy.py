import rospy
import numpy as np
from spatialmath import SE3, SO3
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult


class HitBuoyStatus(TaskStatus):
    SUCCESS = 0
    BUOY_NOT_FOUND = 1

@dataclass
class HitBuoyResult(TaskResult):
    status: HitBuoyStatus


class HitBuoy(Task):

    def __init__(self, tag: str):
        super().__init__()

        self._tag = tag

    def run(self, resources: TaskResources) -> HitBuoyResult:
        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        odom_t_buoy = resources.map.find_closest(self._tag, odom_t_vehicle.t)

        resources.motion.cancel()

        if odom_t_buoy is None:
            return HitBuoyResult(status=HitBuoyStatus.BUOY_NOT_FOUND)

        while True:
            odom_t_buoy = resources.map.find_closest('chevron', odom_t_buoy.t)

            if np.linalg.norm(odom_t_buoy.t - odom_t_vehicle.t) < 0.1:
                break

            target_odom_t_vehicle = SE3.Rt(odom_t_vehicle.R, np.array([
                odom_t_buoy.t
            ]))

            resources.motion.goto(target_odom_t_vehicle)

        resources.motion.goto_relative(SE3.Rt(SO3(), np.array([-1, 0, 0])))

        return HitBuoyResult(status=HitBuoyStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
