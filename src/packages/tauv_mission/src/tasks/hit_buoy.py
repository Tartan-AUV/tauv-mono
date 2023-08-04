import rospy
import numpy as np
from spatialmath import SE3, SO3
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import time
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

    def __init__(self, tag: str, timeout: float):
        super().__init__()

        self._tag = tag
        self._timeout = timeout

    def run(self, resources: TaskResources) -> HitBuoyResult:
        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        buoy_detection = resources.map.find_closest(self._tag, odom_t_vehicle.t)
        # buoy_t_contact = SE3(SO3.TwoVectors(x="-z", y="x"))
        buoy_t_contact = SE3()
        vehicle_t_contact = SE3.Tx(0.6)

        resources.motion.cancel()

        if buoy_detection is None:
            return HitBuoyResult(status=HitBuoyStatus.BUOY_NOT_FOUND)

        odom_t_buoy = buoy_detection.pose

        start_time = time.time()

        while time.time() - start_time < self._timeout:
            odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
            buoy_detection = resources.map.find_closest(self._tag, odom_t_buoy.t)

            odom_t_buoy = buoy_detection.pose
            # odom_t_buoy_yaw = odom_t_buoy.rpy()[2]
            # odom_t_buoy = SE3.Rt(SO3.RPY((0, 0, odom_t_buoy_yaw))
            odom_t_contact = odom_t_buoy * buoy_t_contact
            odom_t_contact_yaw = odom_t_contact.rpy()[2]
            odom_t_contact = SE3.Rt(SO3.RPY((0, 0, odom_t_contact_yaw)), odom_t_contact.t)
            # odom_t_contact = SE3.Rt(SO3.Rz(atan2(odom_t_buoy.t[1] - odom_t_vehicle.t[1], odom_t_buoy.t[0] - odom_t_vehicle.t[0])), odom_t_buoy.t)

            if buoy_detection is None:
                return HitBuoyResult(status=HitBuoyStatus.BUOY_LOST)

            target_odom_t_vehicle = odom_t_contact * vehicle_t_contact.inv()

            resources.transforms.set_a_to_b('kf/odom', 'buoy', odom_t_buoy)

            resources.transforms.set_a_to_b('kf/odom', 'contact', odom_t_contact)

            resources.transforms.set_a_to_b('kf/odom', 'target', target_odom_t_vehicle)

            resources.motion.goto(target_odom_t_vehicle)

            if np.linalg.norm(odom_t_vehicle.t - target_odom_t_vehicle.t) < 0.1:
                break

            if self._check_cancel(resources): return HitBuoyResult(status=HitBuoyStatus.CANCELLED)

            time.sleep(0.1)

        resources.motion.goto_relative(SE3.Rt(SO3(), np.array([-1, 0, 0])))

        if time.time() > start_time + self._timeout:
            return HitBuoyResult(status=HitBuoyStatus.TIMEOUT)

        return HitBuoyResult(status=HitBuoyStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
