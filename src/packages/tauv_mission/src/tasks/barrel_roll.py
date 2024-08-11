import rospy
import numpy as np
from math import pi
from spatialmath import SE3, SO3
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import time
from tauv_util.spatialmath import flatten_se3


class BarrelRollStatus(TaskStatus):
    SUCCESS = 0
    CANCELLED = 1


@dataclass
class BarrelRollResult(TaskResult):
    status: BarrelRollStatus


class BarrelRoll(Task):
    def __init__(self, n_roll_waypoints=8):
        super().__init__()

        self._n_roll_waypoints = n_roll_waypoints

    def run(self, resources: TaskResources) -> BarrelRollResult:
        odom_t_vehicle_start = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        
        angle_eval_points = np.linspace(0, 2*pi, self._n_roll_waypoints)
        
        for roll_angle_rad in angle_eval_points:
            print("Attempting to reach roll angle (deg): ", np.rad2deg(roll_angle_rad))
            
            start_R_goal = SO3.Rx(roll_angle_rad)
            odom_R_goal = odom_t_vehicle_start.R * start_R_goal
            
            pose_desired = SE3.Rt(odom_R_goal, odom_t_vehicle_start.t)
            
            resources.motion.goto(
                pose=pose_desired,
                params=resources.motion.get_trajectory_params("rapid"),
                flat=False
            )
            
            while True:
                if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                    break

                if self._check_cancel(resources): return BarrelRollResult(status=BarrelRollStatus.CANCELLED)

        return BarrelRollResult(status=BarrelRollStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
