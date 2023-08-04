import rospy
import numpy as np
from math import pi
from spatialmath import SE3, SO3
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import time
from tauv_util.spatialmath import flatten_se3


class GateStatus(TaskStatus):
    SUCCESS = 0
    CANCELLED = 1
    GATE_NOT_FOUND = 2


@dataclass
class GateResult(TaskResult):
    status: GateStatus


class Gate(Task):

    def __init__(self):
        super().__init__()

    def run(self, resources: TaskResources) -> GateResult:
        odom_t_vehicle = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        '''
        scan_thetas = np.linspace(-pi / 2, pi / 2, 4)

        for scan_theta in scan_thetas:
            odom_t_vehicle_goal = odom_t_vehicle * SE3.Rz(scan_theta)

            resources.motion.goto(odom_t_vehicle_goal)

            while True:
                if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                    break

                if self._check_cancel(resources): return GateResult(status=GateStatus.CANCELLED)

            time.sleep(2.0)

        resources.motion.goto(odom_t_vehicle)

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return GateResult(status=GateStatus.CANCELLED)
        '''

        odom_t_course = resources.transforms.get_a_to_b('kf/odom', 'kf/course')
        # odom_t_vehicle = flatten_se3(odom_t_vehicle)

        '''
        gate_detection = resources.map.find_closest('gate', odom_t_vehicle.t)

        if gate_detection is None:
            return GateResult(status=GateStatus.GATE_NOT_FOUND)

        odom_t_gate = gate_detection.pose
        '''
        # odom_t_gate = odom_t_course * SE3.Rt(SO3.Rz(1.57), (5, 2, 2))
        odom_t_gate = odom_t_course * SE3.Rt(SO3(), (12, 0, 1))

        gate_t_spin = SE3.Rt(SO3(), (-2, 0, 0.5))
        gate_t_through = SE3.Rt(SO3(), (2, 0, 0.5))

        odom_t_spin = odom_t_gate * gate_t_spin
        odom_t_through = odom_t_gate * gate_t_through

        resources.transforms.set_a_to_b('kf/odom', 'gate', odom_t_gate)
        resources.transforms.set_a_to_b('kf/odom', 'spin', odom_t_spin)
        resources.transforms.set_a_to_b('kf/odom', 'through', odom_t_through)

        resources.motion.goto(odom_t_spin)

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return GateResult(status=GateStatus.CANCELLED)

        spin_thetas = np.concatenate((
            np.linspace(0, 2 * pi, 4),
            np.linspace(0, 2 * pi, 4),
        ))

        for spin_theta in spin_thetas:
            print(spin_theta)
            odom_t_vehicle_goal = odom_t_spin * SE3.Rz(spin_theta, unit='rad')
            print(odom_t_vehicle)
            print(odom_t_vehicle_goal)
            # odom_t_vehicle_goal = odom_t_spin

            resources.motion.goto(odom_t_vehicle_goal)

            while True:
                if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                    break

                if self._check_cancel(resources): return GateResult(status=GateStatus.CANCELLED)

        resources.motion.goto(odom_t_through)

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return GateResult(status=GateStatus.CANCELLED)

        odom_t_octagon = odom_t_gate * SE3.Tx(12.0)

        resources.motion.goto(odom_t_octagon)

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return GateResult(status=GateStatus.CANCELLED)

        resources.motion.arm(False)

        return GateResult(status=GateStatus.SUCCESS)



    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()
