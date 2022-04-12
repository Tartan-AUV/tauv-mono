import rospy
import numpy as np

from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from tauv_util.types import tl
from std_msgs.msg import Float64
from geometry_msgs.msg import Wrench
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped


class Thrusters:

    def __init__(self):
        self._dt: float = 0.02
        self._timeout: float = 1.0

        self._load_config()

        self._is_armed: bool = False
        self._arm_service: rospy.Service = rospy.Service('arm', SetBool, self._handle_arm)

        self._wrench_sub: rospy.Subscriber = rospy.Subscriber('wrench', Wrench, self._handle_wrench)
        self._thruster_pubs: [rospy.Publisher] = list(map(
            lambda i: rospy.Publisher(f'/kingfisher/thrusters/{i}/input', FloatStamped, queue_size=10),
            range(8),
        ))

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
            self._set_thrust(thruster, thrust)

    def _handle_arm(self, req: SetBoolRequest):
        print('arming')
        self._is_armed = req.data
        return SetBoolResponse(True, '')

    def _handle_wrench(self, msg: Wrench):
        self._wrench = msg
        self._wrench_update_time = rospy.Time.now()

    def _set_thrust(self, thruster: int, thrust: float):
        f = FloatStamped()
        f.data = thrust
        self._thruster_pubs[thruster].publish(f)

    def _get_thrusts(self, wrench: Wrench) -> np.array:
        return self._tam @ np.concatenate((tl(wrench.force), tl(wrench.torque)))

    def _load_config(self):
        self._maestro_port: str = rospy.get_param('~maestro_port')
        self._thruster_channels: [int] = rospy.get_param('~thruster_channels')
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

def main():
    rospy.init_node('thrusters')
    t = Thrusters()
    t.start()
