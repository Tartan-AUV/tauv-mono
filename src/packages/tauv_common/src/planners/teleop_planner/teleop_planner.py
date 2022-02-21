import rospy
import numpy as np
from enum import Enum
from typing import Dict

from geometry_msgs.msg import Vector3, Wrench
from sensor_msgs.msg import Joy as JoyMsg
from std_srvs.srv import SetBool


class ButtonInput(Enum):
    ESTOP = 'estop'
    ARM = 'arm'


class AxisInput(Enum):
    X = 'x'
    Y = 'y'
    Z = 'z'
    ROLL = 'roll'
    PITCH = 'pitch'
    YAW = 'yaw'


class TeleopPlanner:
    def __init__(self):
        self._dt: float = 0.05
        self._command_timeout: float = 0.5

        self._joy_sub: rospy.Subscriber = rospy.Subscriber('joy', JoyMsg, self._handle_joy)
        self._wrench_pub: rospy.Publisher = rospy.Publisher('wrench', Wrench, queue_size=10)
        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('arm', SetBool)

        self._command: np.array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._command_timestamp: rospy.Time = rospy.Time.now()

        self._parse_config()

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        if (rospy.Time.now() - self._command_timestamp).to_sec() > self._command_timeout:
            self._command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        wrench: Wrench = Wrench()
        wrench.force = Vector3(self._command[0], self._command[1], self._command[2])
        wrench.torque = Vector3(self._command[3], self._command[4], self._command[5])
        self._wrench_pub.publish(wrench)

    def _handle_joy(self, msg: JoyMsg):
        if self._is_pressed(msg, ButtonInput.ESTOP):
            print('ESTOP')
            self._arm(False)
        elif self._is_pressed(msg, ButtonInput.ARM):
            print('ARM')
            self._arm(True)

        self._command = np.array([
            self._axis_value(msg, AxisInput.X),
            self._axis_value(msg, AxisInput.Y),
            self._axis_value(msg, AxisInput.Z),
            self._axis_value(msg, AxisInput.ROLL),
            self._axis_value(msg, AxisInput.PITCH),
            self._axis_value(msg, AxisInput.YAW)
        ])
        self._command_timestamp = rospy.Time.now()

    def _arm(self, arm: bool):
        try:
            resp = self._arm_srv(arm)
            if not resp.success:
                rospy.logwarn('Arm request failed')
        except rospy.ServiceException:
            rospy.logwarn('Arm server not responding')

    def _axis_value(self, msg: JoyMsg, axis: AxisInput):
        return self._axis_scales[axis.value] * msg.axes[self._axis_ids[axis.value]]

    def _is_pressed(self, msg: JoyMsg, button: ButtonInput):
        print(msg, button)
        return msg.buttons[self._button_ids[button.value]] == 1

    def _parse_config(self):
        self._button_ids: Dict[str, int] = rospy.get_param('~button_ids')
        self._axis_ids: Dict[str, int] = rospy.get_param('~axis_ids')
        self._axis_scales: Dict[str, int] = rospy.get_param('~axis_scales')
        print(self._button_ids)
        print(self._axis_ids)
        print(self._axis_scales)


def main():
    rospy.init_node('teleop_planner')
    p = TeleopPlanner()
    p.start()