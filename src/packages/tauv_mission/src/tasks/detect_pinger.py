import time
import rospy
from spatialmath import SE3, SE3, SO3, SE2
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
from std_srvs.srv import SetBool
from tauv_msgs.msg import PingDetection
from math import atan2, cos, sin
from tauv_util.spatialmath import ros_vector3_to_r3
from trajectories.constant_acceleration_trajectory import ConstantAccelerationTrajectoryParams
import numpy as np

class DetectPingerStatus(TaskStatus):
    SUCCESS = 0
    TIMEOUT = 1
    CANCELLED = 2

@dataclass
class DetectPingerResult(TaskResult):
    status: DetectPingerStatus

class DetectPinger(Task):

    def __init__(self, frequency: float, timeout: float, depth: float):
        super().__init__()

        self._frequency = frequency
        self._timeout = timeout
        self._depth = depth

        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('vehicle/thrusters/arm', SetBool)

    def run(self, resources: TaskResources) -> DetectPingerResult:
        timeout_time = rospy.Time.now() + rospy.Duration.from_sec(self._timeout)
        avg_z = 0

        while rospy.Time.now() < timeout_time:
            try:
                self._arm_srv.call(False)
                detection = rospy.wait_for_message('vehicle/pinger_localizer/detection', PingDetection, timeout=3.0)
                self._arm_srv.call(True)
            except Exception:
                pass

            try:
                self._arm_srv.call(True)
            except Exception:
                pass

            if abs(detection.frequency - self._frequency) > 500:
                continue

            rospy.loginfo(detection)

            odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')

            direction = ros_vector3_to_r3(detection.direction)

            direction_yaw = atan2(direction[1], direction[0])
            avg_z = 0.5 * avg_z + 0.5 * direction[2]
            if avg_z > 0.8:
                resources.motion.cancel()
                return DetectPingerResult(DetectPingerStatus.SUCCESS)

            goal_odom_t_vehicle = odom_t_vehicle * SE3.Rt(SO3.Rz(direction_yaw), np.array([direction[0], direction[1], 0]))
            goal_odom_t_vehicle.t[2] = self._depth

            resources.motion.goto(goal_odom_t_vehicle, params=ConstantAccelerationTrajectoryParams(v_max_linear=0.5, a_linear=1.0, v_max_angular=0.5, a_angular=1.0))
            time.sleep(3.0)

            if self._check_cancel(resources): return DetectPingerResult(DetectPingerStatus.CANCELLED)

        if rospy.Time.now() > timeout_time:
            return DetectPingerResult(DetectPingerStatus.TIMEOUT)

        return DetectPingerResult(DetectPingerStatus.SUCCESS)



    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()