import rospy
import numpy as np
from enum import Enum
from typing import Dict, Optional

from tauv_msgs.msg import ControllerCmd as ControllerCmdMsg, Pose as PoseMsg
from tauv_msgs.srv import SetTargetPose, SetTargetPoseRequest, SetTargetPoseResponse
from tauv_util.types import tm, tl
from tauv_util.transforms import rpy_to_quat
from geometry_msgs.msg import Pose, Vector3
from sensor_msgs.msg import Joy as JoyMsg
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse


class ButtonInput(Enum):
    ESTOP = 'estop'
    ARM = 'arm'
    MANUAL = 'manual'
    MPC = 'mpc'
    HOLD = 'hold'

class Mode(Enum):
    MANUAL = 'manual'
    MPC = 'mpc'
    HOLD = 'hold'

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
        self._is_hold_enabled: bool = False
        self._mode: Mode = Mode.MANUAL

        self._joy_cmd: Optional[np.array] = None
        self._joy_cmd_timestamp: Optional[rospy.Time] = None
        self._joy_cmd_timeout: float = 0.1

        self._mpc_cmd: Optional[np.array] = None
        self._mpc_cmd_timestamp: Optional[rospy.Time] = None
        self._mpc_cmd_timeout: float = 0.5

        self._position: Optional[np.array] = None
        self._orientation: Optional[np.array] = None

        self._joy_sub: rospy.Subscriber = rospy.Subscriber('joy', JoyMsg, self._handle_joy)
        self._mpc_cmd_sub: rospy.Subscriber = rospy.Subscriber('mpc_cmd', ControllerCmdMsg, self._handle_mpc_cmd)
        self._pose_sub: rospy.Subscriber = rospy.Subscriber('pose', PoseMsg, self._handle_pose)
        self._cmd_pub: rospy.Publisher = rospy.Publisher('cmd', ControllerCmdMsg, queue_size=10)
        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('arm', SetBool)

        self._target_pose_srv: rospy.ServiceProxy = rospy.ServiceProxy('set_target_pose', SetTargetPose)
        self._hold_z_srv: rospy.ServiceProxy = rospy.ServiceProxy('set_hold_z', SetBool)

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        time = rospy.Time.now()

        is_joy_cmd_valid = self._joy_cmd is not None \
            and self._joy_cmd_timestamp is not None \
            and (time - self._joy_cmd_timestamp).to_sec() < self._joy_cmd_timeout

        is_mpc_cmd_valid = self._mpc_cmd is not None \
            and self._mpc_cmd_timestamp is not None \
            and (time - self._mpc_cmd_timestamp).to_sec() < self._mpc_cmd_timeout

        cmd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # if not self._is_armed:
        #     cmd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if self._mode == Mode.MPC and is_mpc_cmd_valid:
            cmd = self._mpc_cmd
        elif (self._mode == Mode.MANUAL or self._mode == Mode.HOLD) and is_joy_cmd_valid:
            cmd = self._joy_cmd

        msg: ControllerCmdMsg = ControllerCmdMsg()
        msg.a_x = cmd[0]
        msg.a_y = cmd[1]
        msg.a_z = cmd[2]
        msg.a_roll = cmd[3]
        msg.a_pitch = cmd[4]
        msg.a_yaw = cmd[5]
        self._cmd_pub.publish(msg)

    def _handle_joy(self, msg: JoyMsg):
        if self._is_pressed(msg, ButtonInput.ESTOP):
            self._set_arm(False)
        elif self._is_pressed(msg, ButtonInput.ARM):
            self._set_arm(True)

        if self._is_pressed(msg, ButtonInput.MANUAL):
            self._mode = Mode.MANUAL
            self._set_hold(False)
        elif self._is_pressed(msg, ButtonInput.MPC):
            self._mode = Mode.MPC
            self._set_hold(False)
        elif self._is_pressed(msg, ButtonInput.HOLD):
            self._mode = Mode.HOLD
            self._set_hold(True)

        self._joy_cmd = np.array([
            self._axis_value(msg, AxisInput.X),
            self._axis_value(msg, AxisInput.Y),
            self._axis_value(msg, AxisInput.Z),
            self._axis_value(msg, AxisInput.ROLL),
            self._axis_value(msg, AxisInput.PITCH),
            self._axis_value(msg, AxisInput.YAW)
        ])
        self._joy_cmd_timestamp = rospy.Time.now()

    def _handle_mpc_cmd(self, msg: ControllerCmdMsg):
        self._mpc_cmd = np.array([
            msg.a_x,
            msg.a_y,
            msg.a_z,
            msg.a_roll,
            msg.a_pitch,
            msg.a_yaw
        ])
        self._mpc_cmd_timestamp = rospy.Time.now()

    def _handle_pose(self, msg: PoseMsg):
        self._position = tl(msg.position)
        self._orientation = tl(msg.orientation)

    def _set_arm(self, arm: bool):
        try:
            res: SetBoolResponse = self._arm_srv.call(arm)
            if not res.success:
                rospy.logwarn('Arm request failed')
            else:
                self._is_armed = arm
        except rospy.ServiceException:
            rospy.logwarn('Arm server not responding')

    def _set_hold(self, enable: bool):
        if self._position is None \
                or self._orientation is None \
                or self._is_hold_enabled == enable:
            return

        if enable:
            pose: Pose = Pose()
            pose.position = tm(self._position, Vector3)
            pose.orientation = rpy_to_quat([0.0, 0.0, self._orientation[2]])
            req = SetTargetPoseRequest()
            req.pose = pose

            try:
                res: SetTargetPoseResponse = self._target_pose_srv.call(req)
                if not res.success:
                    rospy.logerr('Set target pose request failed')
                    return

            except rospy.ServiceException:
                rospy.logerr('Set target pose service not responding')
                return

        req = SetBoolRequest(enable)

        try:
            res: SetBoolResponse = self._hold_z_srv.call(req)
            if not res.success:
                rospy.logwarn('Set hold z request failed')

            self._is_hold_enabled = enable
        except rospy.ServiceException:
            rospy.logwarn('Set hold z service not responding')


    def _axis_value(self, msg: JoyMsg, axis: AxisInput):
        return self._axis_scales[axis.value] * msg.axes[self._axis_ids[axis.value]]

    def _is_pressed(self, msg: JoyMsg, button: ButtonInput):
        return msg.buttons[self._button_ids[button.value]] == 1

    def _parse_config(self):
        self._button_ids: Dict[str, int] = rospy.get_param('~button_ids')
        self._axis_ids: Dict[str, int] = rospy.get_param('~axis_ids')
        self._axis_scales: Dict[str, int] = rospy.get_param('~axis_scales')


def main():
    rospy.init_node('teleop_planner')
    p = TeleopPlanner()
    p.start()