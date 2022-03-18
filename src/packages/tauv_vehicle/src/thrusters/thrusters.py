import rospy
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from math import floor
from typing import Dict

from .maestro import Maestro
from tauv_util.types import tl
from geometry_msgs.msg import Vector3
from tauv_msgs.msg import Battery as BatteryMsg
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from geometry_msgs.msg import Wrench


class Thrusters:

    def __init__(self):
        self._dt: float = 0.02
        self._timeout: float = 5.0

        self._load_config()

        self._maestro = Maestro(ttyStr=self._maestro_port)
        print('initialized maestro')

        self._is_armed: bool = True
        self._arm_service: rospy.Service = rospy.Service('arm', SetBool, self._handle_arm)

        self._battery_sub: rospy.Subscriber = rospy.Subscriber('battery', BatteryMsg, self._handle_battery)
        self._wrench_sub: rospy.Subscriber = rospy.Subscriber('wrench', Wrench, self._handle_wrench)

        self._battery_voltage: float = 13.7
        self._wrench: Wrench = Wrench()
        self._wrench_update_time: rospy.Time = rospy.Time.now()

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        if (rospy.Time.now() - self._wrench_update_time).to_sec() > self._timeout \
                or not self._is_armed:
            self._wrench = Wrench()
            self._wrench_update_time = rospy.Time.now()

        thrusts = self._get_thrusts(self._wrench)

        for (thruster, thrust) in enumerate(thrusts):
            print(f'setting ${thruster} to ${thrust}')
            self._set_thrust(thruster, thrust)

    def _handle_arm(self, req: SetBoolRequest):
        print('arming')
        self._is_armed = req.data
        return SetBoolResponse(True, '')

    def _handle_battery(self, msg: BatteryMsg):
        self._battery_voltage = msg.voltage

    def _handle_wrench(self, msg: Wrench):
        self._wrench = msg
        self._wrench_update_time = rospy.Time.now()

    def _set_thrust(self, thruster: int, thrust: float):
        pwm_speed = self._get_pwm_speed(thruster, thrust)

        self._maestro.setTarget(pwm_speed * 4, self._thruster_channels[thruster])

    def _get_pwm_speed(self, thruster: int, thrust: float) -> int:
        pwm_speed = 1500

        thrust = thrust * self._thrust_inversions[thruster]

        if thrust < 0 and -self._negative_max_thrust < thrust < -self._negative_min_thrust:
            thrust_curve = Polynomial(
                (self._negative_thrust_coefficients[0]
                    + self._negative_thrust_coefficients[1] * self._battery_voltage
                    + self._negative_thrust_coefficients[2] * self._battery_voltage ** 2
                    - thrust,
                 self._negative_thrust_coefficients[3],
                 self._negative_thrust_coefficients[4]),
            )

            target_pwm_speed = floor(thrust_curve.roots()[0])

            if self._minimum_pwm_speed < target_pwm_speed < self._maximum_pwm_speed:
                pwm_speed = target_pwm_speed

        elif thrust > 0 and self._positive_min_thrust < thrust < self._positive_max_thrust:
            thrust_curve = Polynomial(
                (self._positive_thrust_coefficients[0]
                    + self._positive_thrust_coefficients[1] * self._battery_voltage
                    + self._positive_thrust_coefficients[2] * self._battery_voltage ** 2
                    - thrust,
                 self._positive_thrust_coefficients[3],
                 self._positive_thrust_coefficients[4]),
            )

            target_pwm_speed = floor(thrust_curve.roots()[1])

            if self._minimum_pwm_speed < target_pwm_speed < self._maximum_pwm_speed:
                pwm_speed = target_pwm_speed

        return pwm_speed

    def _get_thrusts(self, wrench: Wrench) -> np.array:
        return self._tam @ np.concatenate((tl(wrench.force), tl(wrench.torque)))

    def _load_config(self):
        self._maestro_port: str = rospy.get_param('~maestro_port')
        self._thruster_channels: [int] = rospy.get_param('~thruster_channels')
        self._minimum_pwm_speed: float = rospy.get_param('~minimum_pwm_speed')
        self._maximum_pwm_speed: float = rospy.get_param('~maximum_pwm_speed')
        self._negative_min_thrust: float = rospy.get_param('~negative_min_thrust')
        self._negative_max_thrust: float = rospy.get_param('~negative_max_thrust')
        self._positive_min_thrust: float = rospy.get_param('~positive_min_thrust')
        self._positive_max_thrust: float = rospy.get_param('~positive_max_thrust')
        self._positive_thrust_coefficients: np.array = np.array(rospy.get_param('~positive_thrust_coefficients'))
        self._negative_thrust_coefficients: np.array = np.array(rospy.get_param('~negative_thrust_coefficients'))
        self._thrust_inversions: [float] = rospy.get_param('~thrust_inversions')
        self._tam: np.array = np.linalg.pinv(np.array(rospy.get_param('~tam')))


def main():
    rospy.init_node('thrusters')
    t = Thrusters()
    t.start()