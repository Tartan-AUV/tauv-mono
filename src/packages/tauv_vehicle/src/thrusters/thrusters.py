import rospy
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from math import floor
from typing import Dict

from .maestro import Maestro
from tauv_util.types import tl
from geometry_msgs.msg import Vector3
from tauv_msgs.msg import Battery as BatteryMsg, Servos as ServosMsg
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from geometry_msgs.msg import Wrench
from std_msgs.msg import Bool

from tauv_alarms import Alarm, AlarmClient


class Thrusters:

    def __init__(self):
        self._ac: AlarmClient = AlarmClient()

        self._dt: float = 0.02
        self._timeout: float = 1.0

        self._load_config()

        while not self._try_init():
            rospy.sleep(0.5)
        rospy.loginfo('initialized maestro')

        self._is_armed: bool = False
        self._arm_service: rospy.Service = rospy.Service('arm', SetBool, self._handle_arm)

        self._servos_sub: rospy.Subscriber = rospy.Subscriber('servos', ServosMsg, self._handle_servos)
        self._battery_sub: rospy.Subscriber = rospy.Subscriber('battery', BatteryMsg, self._handle_battery)
        self._wrench_sub: rospy.Subscriber = rospy.Subscriber('wrench', Wrench, self._handle_wrench)

        self._battery_voltage: float = self._default_battery_voltage
        self._wrench: Wrench = Wrench()
        self._wrench_update_time: rospy.Time = rospy.Time.now()

        self._killed_pub : rospy.Publisher = rospy.Publisher('killed', Bool, queue_size=1)

    def _try_init(self):
        try:
            self._maestro = Maestro(ttyStr=self._maestro_port)
            return True
        except Exception as e:
            print(e)
            return False

    def start(self):
        rospy.loginfo('start')
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        _killed = True
        try:
            kval = self._maestro.getPosition(self._kill_channel)
            _killed = kval < 400
        except TypeError:
            rospy.logwarn("read error")

        self._killed_pub.publish(Bool(_killed))
        self._ac.set(Alarm.KILL_SWITCH_ACTIVE, value=_killed)
        self._ac.set(Alarm.SUB_DISARMED, value=not self._is_armed)

        if (rospy.Time.now() - self._wrench_update_time).to_sec() > self._timeout \
                or not self._is_armed or _killed:
            self._wrench = Wrench()
            self._wrench_update_time = rospy.Time.now()

        thrusts = self._get_thrusts(self._wrench)

        self._maestro.clearErrors()

        for (thruster, thrust) in enumerate(thrusts):
            self._set_thrust(thruster, thrust)

        self._ac.clear(Alarm.THRUSTERS_NOT_INITIALIZED)

    def _handle_arm(self, req: SetBoolRequest):
        rospy.loginfo('armed' if req.data else 'disarmed')
        self._is_armed = req.data
        return SetBoolResponse(True, '')

    def _handle_battery(self, msg: BatteryMsg):
        self._battery_voltage = msg.voltage

    def _handle_wrench(self, msg: Wrench):
        self._wrench = msg
        self._wrench_update_time = rospy.Time.now()

    def _handle_servos(self, msg: ServosMsg):
        for i in range(len(self._servo_channels)):
            self._maestro.setTarget(floor(msg.targets[i] / 180 * 1000 + 1500) * 4, self._servo_channels[i])

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
        self._servo_channels: [int] = rospy.get_param('~servo_channels')
        self._default_battery_voltage: float = rospy.get_param('~default_battery_voltage')
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
        self._kill_channel : int = rospy.get_param('~kill_channel')

def main():
    rospy.init_node('thrusters')
    t = Thrusters()
    t.start()