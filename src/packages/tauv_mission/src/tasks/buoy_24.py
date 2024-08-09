import rospy
import numpy as np
from spatialmath import SE3, SO3, SE2
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import time
from tauv_util.spatialmath import flatten_se3
from math import atan2


class CircleBuoyStatus(TaskStatus):
    SUCCESS = 0
    CANCELLED = 1
    TIMEOUT = 2
    BUOY_NOT_FOUND = 3
    BUOY_LOST = 4

@dataclass
class CircleBuoyResult(TaskResult):
    status: CircleBuoyStatus


class CircleBuoy(Task):
    def __init__(self, tag, circle_radius, circle_ccw=True, waypoint_every_n_meters=0.75):
        super().__init__()
        
        self._circle_radius = circle_radius
        self._circle_ccw = circle_ccw
        self._tag = tag # TODO: fix

        arclength = 2 * np.pi * self._circle_radius
        waypoint_every_n_meters = waypoint_every_n_meters
        self._n_waypoints_along_circle_trajectory = int(arclength / waypoint_every_n_meters)

    def run(self, resources: TaskResources) -> CircleBuoyResult:
        timeout_time = rospy.Time.now() + rospy.Duration.from_sec(10000)

        resources.motion.cancel()

        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        # buoy_detection = resources.map.find_closest(self._tag, odom_t_vehicle.t)
        buoy_detection = resources.map.find_one(self._tag)

        if buoy_detection is None:
            print(">> !! BUOY NOT FOUND")
            return CircleBuoyResult(status=CircleBuoyStatus.BUOY_NOT_FOUND)
        
        odom_t_buoy = buoy_detection.pose

        if np.linalg.norm(odom_t_buoy.t - odom_t_vehicle.t) > 10:
            print(">> !! BUOY NOT FOUND - more than 10m away")
            return CircleBuoyResult(status=CircleBuoyStatus.BUOY_NOT_FOUND)

        odom_r_vehicle = odom_t_vehicle.t
        odom_r_bouy = odom_t_buoy.t
        
        vec_vehicle_to_bouy = odom_r_bouy - odom_r_vehicle
        
        # Get desired orientation for approach
        vec_vehicle_to_bouy_planar = vec_vehicle_to_bouy * np.array([1, 1, 0])
        vec_vehicle_to_bouy_planar /= np.linalg.norm(vec_vehicle_to_bouy_planar)
        
        e3 = np.array([0, 0, 1])
        odom_R_goal = np.array([vec_vehicle_to_bouy_planar, np.cross(e3, vec_vehicle_to_bouy_planar), e3]).T
        
        # Get desired final approach position
        vec_vehicle_to_bouy_norm = np.linalg.norm(vec_vehicle_to_bouy)
        vec_vehicle_to_bouy_hat = vec_vehicle_to_bouy / vec_vehicle_to_bouy_norm
        
        odom_r_goal = odom_r_vehicle + vec_vehicle_to_bouy_hat * (vec_vehicle_to_bouy_norm - self._circle_radius)

        
        # Final desired goal transform
        odom_t_goal = SE3.Rt(odom_R_goal, odom_r_goal)

        if vec_vehicle_to_bouy_norm > self._circle_radius:
            print('approaching')
            resources.motion.goto(odom_t_goal)

            while True:
                if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                    break

                if self._check_cancel(resources): return CircleBuoyResult(CircleBuoyStatus.TIMEOUT)
        
        # Get waypoints along the circle trajectory around the bouy
        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        odom_r_vehicle = odom_t_vehicle.t
        odom_R_vehicle = odom_t_vehicle.R

        circle_fn = lambda t : odom_r_bouy + np.array([self._circle_radius * np.cos(t), self._circle_radius * np.sin(t), 0])
        intersection_t = np.arctan2(odom_r_vehicle[1] - odom_r_bouy[1], odom_r_vehicle[0] - odom_r_bouy[0])

        if self._circle_ccw:
            final_t = intersection_t - 2 * np.pi
        else:
            final_t = intersection_t + 2 * np.pi
        
        eval_t_array = np.linspace(intersection_t, final_t, self._n_waypoints_along_circle_trajectory)
        circle_points = np.array([circle_fn(t) for t in eval_t_array])

        print("STARTING CIRCLING")
        print(f"Sub position {odom_r_vehicle}")
        for odom_r_circle_point in circle_points:
            print('circling')
            print(f'Buoy: {odom_r_bouy}')
            print(f'Goal: {odom_r_circle_point}')
            print(f'Diff norm: {np.linalg.norm(odom_r_circle_point - odom_r_bouy)}')
            print(f'Diff vec: {odom_r_circle_point - odom_r_bouy}')
            print()
            odom_r_circle_point_fixed_depth = np.array([odom_r_circle_point[0], odom_r_circle_point[1], 1.5])

            odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
            vec_vehicle_to_bouy = odom_r_bouy - odom_t_vehicle.t
            vec_vehicle_to_bouy_planar = vec_vehicle_to_bouy * np.array([1, 1, 0])
            vec_vehicle_to_bouy_planar /= np.linalg.norm(vec_vehicle_to_bouy_planar)

            e3 = np.array([0, 0, 1])
            odom_R_goal = np.array([vec_vehicle_to_bouy_planar, np.cross(e3, vec_vehicle_to_bouy_planar), e3]).T

            odom_t_circle_point = SE3.Rt(odom_R_goal, odom_r_circle_point_fixed_depth)
            resources.motion.goto(odom_t_circle_point, params=resources.motion.get_trajectory_params("rapid"))

            while True:
                if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                    break

                if self._check_cancel(resources): return CircleBuoyResult(CircleBuoyStatus.TIMEOUT)

        print('done')
        return CircleBuoyResult(status=CircleBuoyStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
