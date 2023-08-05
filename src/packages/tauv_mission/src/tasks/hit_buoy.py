import rospy
import numpy as np
from spatialmath import SE3, SO3
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import time
from tauv_util.spatialmath import flatten_se3
from math import atan2


class HitBuoyStatus(TaskStatus):
    SUCCESS = 0
    CANCELLED = 1
    TIMEOUT = 2
    BUOY_NOT_FOUND = 3
    BUOY_LOST = 4

@dataclass
class HitBuoyResult(TaskResult):
    status: HitBuoyStatus


class HitBuoy(Task):

    def __init__(self, tag: str, timeout: float, distance: float, error_threshold: float):
        super().__init__()

        self._tag = tag
        self._timeout = timeout
        self._distance = distance
        self._error_threshold = error_threshold

    def run(self, resources: TaskResources) -> HitBuoyResult:
        timeout_time = rospy.Time.now() + rospy.Duration.from_sec(self._timeout)

        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        buoy_detection = resources.map.find_closest(self._tag, odom_t_vehicle.t)

        buoy_t_vehicle = SE3.Tx(-self._distance)

        if buoy_detection is None:
            return HitBuoyResult(status=HitBuoyStatus.BUOY_NOT_FOUND)

        resources.motion.cancel()

        odom_t_buoy = buoy_detection.pose

        while rospy.Time.now() < timeout_time:
            odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
            buoy_detection = resources.map.find_closest(self._tag, odom_t_buoy.t)

            if buoy_detection is None:
                return HitBuoyResult(status=HitBuoyStatus.BUOY_LOST)

            odom_t_buoy = buoy_detection.pose

            odom_t_vehicle_goal = odom_t_buoy * buoy_t_vehicle

            if np.linalg.norm(odom_t_vehicle.t - odom_t_vehicle_goal.t) < self._error_threshold:
                break

            resources.transforms.set_a_to_b('kf/odom', 'buoy', odom_t_buoy)

            resources.transforms.set_a_to_b('kf/odom', 'vehicle_goal', odom_t_vehicle_goal)

            resources.motion.goto(odom_t_vehicle_goal, params=resources.motion.get_params("feedback"))

            if self._check_cancel(resources): return HitBuoyResult(status=HitBuoyStatus.CANCELLED)

            rospy.sleep(self._period)

        resources.motion.goto_relative(SE3.Rt(SO3(), (-1, 0, 0)))

        if not self._spin_cancel(resources, lambda: resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)), timeout_time):
            return HitBuoyResult(status=HitBuoyStatus.TIMEOUT)

        return HitBuoyResult(status=HitBuoyStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
