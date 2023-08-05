import time
import rospy
from spatialmath import SE2, SE3, SO3
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import numpy as np
from typing import Dict, Union

class FindStatus(TaskStatus):
    FOUND = 0
    NOT_FOUND = 1
    CANCELLED = 2
    TIMEOUT = 3

@dataclass
class FindResult(TaskResult):
    status: FindStatus
    tag : str

class Find(Task):
    def __init__(self, tag : str, pos_estimate : Dict, params : Dict, timeout = None):
        super().__init__(timeout)

        print(params)

        self._resources = None

        self._tag = tag
        self._approx_go_to = SE3.Rt(SO3.RPY(0,0,pos_estimate["theta"]), t=[pos_estimate["x"], pos_estimate["y"], pos_estimate["z"]])
        
        self._rot_scan_resolution = (int)(params["rot_scan_resolution"])
        self._trans_scan_resolution = (int)(params["trans_scan_resolution"])
        self._x_range = (float)(params["x_range"])
        self._y_range = (float)(params["y_range"])
        self._theta_range = (float)(params["theta_range"])

    def _go_to_until_found(self, transform) -> FindResult:
        self._resources.motion.goto(transform)

        #start scanning until we timeout or find the tagged object
        detection = self._resources.map.find_closest(self._tag, transform.t)
        while((not self._check_timeout()) and (detection is None)):
            if self._resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                return FindResult(FindStatus.NOT_FOUND, self._tag)

            detection = self._resources.map.find_closest(self._tag, transform.t)

        if(detection is None):
            return FindResult(FindStatus.TIMEOUT, self._tag)

        return FindResult(FindStatus.FOUND, self._tag)

    def _generate_scan(self, x_range, y_range, radius):
        #get the initial position
        odom_t_vehicle_initial = self._resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')

        positions = []
        #cycle around and scan
        for x in range(self._trans_scan_resolution):
            for y in range(self._trans_scan_resolution):
                for theta in range(self._rot_scan_resolution):
                    o = theta * radius / self._rot_scan_resolution
                    i = ((x / self._trans_scan_resolution) - 0.5) * x_range
                    j = ((y / self._trans_scan_resolution) - 0.5) * y_range
                    positions.append(odom_t_vehicle_initial * SE3.Rt(SO3.RPY(0,0,o), np.array([i, j, 0])))

        return positions

    def run(self, resources: TaskResources) -> FindResult:
        #setup
        self._resources = resources
        result = FindResult(FindStatus.NOT_FOUND, self._tag)

        #set timeout to start scanning
        self._set_timeout()

        #go to the initial position estimate
        odom_t_vehicle = self._resources.transforms.get_a_to_b('kf/odom', 'kf/start')
        initial_go_to = odom_t_vehicle * self._approx_go_to

        result = self._go_to_until_found(initial_go_to)
        if(result.status!=FindStatus.NOT_FOUND):
            return result

        transforms = self._generate_scan(self._x_range, self._y_range, self._theta_range)
        for transform in transforms:
            result = self._go_to_until_found(transform)

            if(result.status!=FindStatus.NOT_FOUND):
                return result

        return result

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
