import rospy
import numpy as np
from spatialmath import SE3, SO3
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
from typing import Optional


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

    def __init__(self, tag: str, timeout: float, frequency: float, distance: float, error_a: float, error_b: float, error_threshold: float, shoot_torpedo: Optional[int] = None):
        super().__init__()

        self._tag = tag
        self._timeout = timeout
        self._period = rospy.Duration.from_sec(1 / frequency)
        self._distance = distance
        self._error_a = error_a
        self._error_b = error_b
        self._error_threshold = error_threshold
        self._shoot_torpedo = shoot_torpedo

    def run(self, resources: TaskResources) -> HitBuoyResult:
        timeout_time = rospy.Time.now() + rospy.Duration.from_sec(self._timeout)

        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        buoy_detection = resources.map.find_closest(self._tag, odom_t_vehicle.t)

        buoy_t_buoy_aligned = SE3(SO3.TwoVectors(x="-z", y="x"))
        buoy_aligned_t_vehicle_goal = SE3.Tx(-self._distance)

        if buoy_detection is None:
            return HitBuoyResult(status=HitBuoyStatus.BUOY_NOT_FOUND)

        resources.motion.cancel()

        odom_t_buoy = buoy_detection.pose

        if np.linalg.norm(odom_t_buoy.t - odom_t_vehicle.t) > 5:
            return HitBuoyResult(status=HitBuoyStatus.BUOY_NOT_FOUND)

        while rospy.Time.now() < timeout_time:
            odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
            buoy_detection = resources.map.find_closest(self._tag, odom_t_buoy.t)

            if buoy_detection is None:
                return HitBuoyResult(status=HitBuoyStatus.BUOY_LOST)

            odom_t_buoy = buoy_detection.pose
            odom_t_buoy_aligned = odom_t_buoy * buoy_t_buoy_aligned
            odom_t_vehicle_goal = odom_t_buoy_aligned * buoy_aligned_t_vehicle_goal

            if np.linalg.norm(odom_t_vehicle.t - odom_t_vehicle_goal.t) < self._error_threshold:
                if self._shoot_torpedo is not None:
                    resources.actuators.shoot_torpedo(self._shoot_torpedo)
                break

            buoy_aligned_t_vehicle = odom_t_buoy_aligned.inv() * odom_t_vehicle

            orthogonal_error = np.linalg.norm(buoy_aligned_t_vehicle.t[1:3] - buoy_aligned_t_vehicle_goal.t[1:3]) + abs(buoy_aligned_t_vehicle.rpy()[2] - buoy_aligned_t_vehicle_goal.rpy()[2])

            x = -self._error_a * (1 - np.exp(-self._error_b * orthogonal_error))

            buoy_aligned_t_vehicle_target = SE3.Rt(SO3(), (x, buoy_aligned_t_vehicle_goal.t[1], buoy_aligned_t_vehicle_goal.t[2]))
            odom_t_vehicle_target = odom_t_buoy_aligned * buoy_aligned_t_vehicle_target

            resources.transforms.set_a_to_b('kf/odom', 'buoy', odom_t_buoy)

            resources.transforms.set_a_to_b('kf/odom', 'vehicle_goal', odom_t_vehicle_goal)

            resources.transforms.set_a_to_b('kf/odom', 'target', odom_t_vehicle_target)

            resources.motion.goto(odom_t_vehicle_target, params=resources.motion.get_trajectory_params("feedback"))

            if self._check_cancel(resources): return HitBuoyResult(status=HitBuoyStatus.CANCELLED)

            rospy.sleep(self._period)

        resources.motion.goto_relative(SE3.Rt(SO3(), (-2, 0, 0)))

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return HitBuoyResult(status=HitBuoyStatus.FAILURE)

        return HitBuoyResult(status=HitBuoyStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
