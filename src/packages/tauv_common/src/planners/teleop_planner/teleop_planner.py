import rospy
import numpy as np
from enum import Enum
from typing import Dict, Optional

from tauv_msgs.msg import ControllerCommand
from tauv_msgs.srv import SetTargetPose, SetTargetPoseRequest, SetTargetPoseResponse
from tauv_util.types import tm, tl
from tauv_util.transforms import rpy_to_quat
from geometry_msgs.msg import Pose, Vector3
from sensor_msgs.msg import Joy
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse


class ButtonInput(Enum):
    ESTOP = 'estop'
    ARM = 'arm'
    MANUAL = 'manual'
    AUTO = 'auto'

class Mode(Enum):
    MANUAL = 'manual'
    AUTO = 'auto'

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

        self._parse_config()

        self._is_armed: bool = False
        self._mode: Mode = Mode.MANUAL

        self._joy_cmd: Optional[ControllerCommand] = None
        self._joy_cmd_timestamp: Optional[rospy.Time] = None
        self._joy_cmd_timeout: float = 1.0

        self._planner_cmd: Optional[ControllerCommand] = None
        self._planner_cmd_timestamp: Optional[rospy.Time] = None
        self._planner_cmd_timeout: float = 1.0

        self._joy_sub: rospy.Subscriber = rospy.Subscriber('joy', Joy, self._handle_joy)
        self._planner_cmd_sub: rospy.Subscriber = rospy.Subscriber('gnc/controller/planner_command', ControllerCommand, self._handle_planner_cmd)
        self._cmd_pub: rospy.Publisher = rospy.Publisher('gnc/controller/controller_command', ControllerCommand, queue_size=10)
        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('vehicle/thrusters/arm', SetBool)

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        time = rospy.Time.now()

        is_joy_cmd_valid = self._joy_cmd is not None \
            and self._joy_cmd_timestamp is not None \
            and (time - self._joy_cmd_timestamp).to_sec() < self._joy_cmd_timeout

        is_planner_cmd_valid = self._planner_cmd is not None \
            and self._planner_cmd_timestamp is not None \
            and (time - self._planner_cmd_timestamp).to_sec() < self._planner_cmd_timeout

        cmd = ControllerCommand()

        if self._mode == Mode.AUTO and is_planner_cmd_valid:
            cmd = self._planner_cmd
        elif self._mode == Mode.MANUAL and is_joy_cmd_valid:
            cmd = self._joy_cmd

        self._cmd_pub.publish(cmd)

    def _handle_joy(self, msg: Joy):
        if self._is_pressed(msg, ButtonInput.ESTOP):
            self._set_arm(False)
        elif self._is_pressed(msg, ButtonInput.ARM):
            self._set_arm(True)

        if self._is_pressed(msg, ButtonInput.MANUAL):
            self._mode = Mode.MANUAL
        elif self._is_pressed(msg, ButtonInput.AUTO):
            self._mode = Mode.AUTO

        joy_cmd = ControllerCommand()
        joy_cmd.a_x = self._axis_value(msg, AxisInput.X)
        joy_cmd.a_y = self._axis_value(msg, AxisInput.Y)
        joy_cmd.a_z = self._axis_value(msg, AxisInput.Z)
        joy_cmd.a_yaw = self._axis_value(msg, AxisInput.YAW)
        self._joy_cmd = joy_cmd
        self._joy_cmd_timestamp = rospy.Time.now()

    def _handle_planner_cmd(self, msg: ControllerCommand):
        self._planner_cmd = msg
        self._planner_cmd_timestamp = rospy.Time.now()

    def _set_arm(self, arm: bool):
        try:
            res: SetBoolResponse = self._arm_srv.call(arm)
            if not res.success:
                rospy.logwarn('Arm request failed')
            else:
                self._is_armed = arm
        except rospy.ServiceException:
            rospy.logwarn('Arm server not responding')

    def _axis_value(self, msg: Joy, axis: AxisInput):
        return self._axis_scales[axis.value] * msg.axes[self._axis_ids[axis.value]]

    def _is_pressed(self, msg: Joy, button: ButtonInput):
        return msg.buttons[self._button_ids[button.value]] == 1

    def _parse_config(self):
        self._button_ids: Dict[str, int] = rospy.get_param('~button_ids')
        self._axis_ids: Dict[str, int] = rospy.get_param('~axis_ids')
        self._axis_scales: Dict[str, int] = rospy.get_param('~axis_scales')


def main():
    rospy.init_node('teleop_planner')
    p = TeleopPlanner()
    p.start()