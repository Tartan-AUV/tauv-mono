import rospy
import argparse
import numpy as np
from typing import Optional

from tauv_msgs.msg import ControlsTunings, DynamicsTunings
from tauv_msgs.srv import TuneControls, TuneControlsRequest, TuneControlsResponse, TuneDynamics, TuneDynamicsRequest, TuneDynamicsResponse, GetTraj, GetTrajRequest, GetTrajResponse, SetTargetPose, SetTargetPoseRequest, SetTargetPoseResponse
from geometry_msgs.msg import Pose, Twist, Vector3
from nav_msgs.msg import Odometry as Odom, Path
from std_srvs.srv import SetBool
from tauv_util.transforms import rpy_to_quat, quat_to_rpy
from motion.motion_utils import MotionUtils


class ArgumentParserError(Exception): pass


class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)


class TeleopMission:

    def __init__(self):
        self._parser = self._build_parser()

        self._motion = MotionUtils()

        self._pose: Optional[Pose] = None
        self._twist: Optional[Twist] = None

        self._tune_controls_srv: rospy.ServiceProxy = rospy.ServiceProxy('tune_controls', TuneControls)
        self._tune_dynamics_srv: rospy.ServiceProxy = rospy.ServiceProxy('tune_dynamics', TuneDynamics)
        self._target_pose_srv: rospy.ServiceProxy = rospy.ServiceProxy('set_target_pose', SetTargetPose)
        self._hold_z_srv: rospy.ServiceProxy = rospy.ServiceProxy('set_hold_z', SetBool)
        self._hold_xy_srv: rospy.ServiceProxy = rospy.ServiceProxy('set_hold_xy', SetBool)
        self._hold_yaw_srv: rospy.ServiceProxy = rospy.ServiceProxy('set_hold_yaw', SetBool)

        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('arm', SetBool)

    def start(self):
        while True:
            cmd = input('>>> ')
            try:
                args = self._parser.parse_args(cmd.split())
                args.func(args)
            except ArgumentParserError as e:
                print('error:', e)
                continue

    def _handle_tune_controls(self, args):
        print('tune_controls', args.roll, args.pitch, args.z)

        t = ControlsTunings()
        if args.roll is not None:
            t.update_roll = True
            t.roll_tunings = args.roll

        if args.roll_limits is not None:
            t.update_roll_limits = True
            t.roll_limits = args.roll_limits

        if args.pitch is not None:
            t.update_pitch = True
            t.pitch_tunings = args.pitch

        if args.pitch_limits is not None:
            t.update_pitch_limits = True
            t.pitch_limits = args.pitch_limits

        if args.yaw is not None:
            t.update_yaw = True
            t.yaw_tunings = args.yaw

        if args.yaw_limits is not None:
            t.update_yaw_limits = True
            t.yaw_limits = args.yaw_limits

        if args.xy is not None:
            t.update_x = True
            t.x_tunings = args.x
            t.update_y = True
            t.y_tunings = args.y

        if args.z is not None:
            t.update_z = True
            t.z_tunings = args.z

        if args.z_limits is not None:
            t.update_z_limits = True
            t.z_limits = args.z_limits

        if args.tau is not None:
            t.update_tau = True
            t.tau = args.tau

        req: TuneControlsRequest = TuneControlsRequest()
        req.tunings = t
        self._tune_controls_srv.call(req)

    def _handle_tune_dynamics(self, args):
        print('tune_dynamics', args.mass, args.volume, args.water_density, args.center_of_gravity, args.center_of_buoyancy, args.moments, args.linear_damping, args.quadratic_damping, args.added_mass)

        t = DynamicsTunings()
        if args.mass is not None:
            t.update_mass = True
            t.mass = args.mass

        if args.volume is not None:
            t.update_volume = True
            t.volume = args.volume

        if args.water_density is not None:
            t.update_water_density = True
            t.water_density = args.water_density

        if args.center_of_gravity is not None:
            t.update_center_of_gravity = True
            t.center_of_gravity = args.center_of_gravity

        if args.center_of_buoyancy is not None:
            t.update_center_of_buoyancy = True
            t.center_of_buoyancy = args.center_of_buoyancy

        if args.moments is not None:
            t.update_moments = True
            t.moments = args.moments

        if args.linear_damping is not None:
            t.update_linear_damping = True
            t.linear_damping = args.linear_damping

        if args.quadratic_damping is not None:
            t.update_quadratic_damping = True
            t.quadratic_damping = args.quadratic_damping

        if args.added_mass is not None:
            t.update_added_mass = True
            t.added_mass = args.added_mass

        req: TuneDynamicsRequest = TuneDynamicsRequest()
        req.tunings = t
        self._tune_dynamics_srv.call(req)

    def _handle_goto(self, args):
        v = args.v if args.v is not None else .4
        a = args.a if args.a is not None else .4
        j = args.j if args.j is not None else .4

        try:
            self._motion.goto(
                (args.x, args.y, args.z),
                args.yaw,
                v=v,
                a=a,
                j=j
            )
        except Exception as e:
            print(e)

    def _handle_goto_relative(self, args):
        v = args.v if args.v is not None else .4
        a = args.a if args.a is not None else .4
        j = args.j if args.j is not None else .4

        try:
            self._motion.goto_relative(
                (args.x, args.y, args.z),
                args.yaw,
                v=v,
                a=a,
                j=j
            )
        except Exception as e:
            print(e)

    def _handle_enable_pids(self, args):
        pose = Pose()
        pose.position = Vector3(args.x, args.y, args.z)
        pose.orientation = rpy_to_quat(np.array([0, 0, args.yaw]))

        req: SetTargetPoseRequest = SetTargetPoseRequest()
        req.pose = pose
        self._target_pose_srv.call(req)

        self._hold_xy_srv.call(args.enable_xy)
        self._hold_z_srv.call(args.enable_z)
        self._hold_yaw_srv.call(args.enable_yaw)

    def _handle_arm(self, args):
        print('arm')

        self._arm_srv.call(True)

    def _handle_disarm(self, args):
        print('disarm')

        self._arm_srv.call(False)

    def _handle_enable(self, args):
        print('enable')

        self._motion.enable()

    def _handle_disable(self, args):
        print('disable')

        self._motion.disable()

    def _handle_retare(self, args):
        print('retare')

        self._motion.retare(0, 0, 0)

    def _handle_goto_pid(self, args):
        print('goto_pid')

        try:
            self._motion.goto_pid(
                (args.x, args.y, args.z),
                args.yaw,
            )
        except Exception as e:
            print(e)

    def _handle_start_mission(self, args):
        print('start mission')

        # start mission with args.delay

    def _handle_odom(self, msg: Odom):
        self._pose = msg.pose.pose
        self._twist = msg.twist.twist

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = ThrowingArgumentParser(prog="teleop_mission")
        subparsers = parser.add_subparsers()

        tune_controls = subparsers.add_parser('tune_controls')
        tune_controls.add_argument('--roll', type=float, nargs=3)
        tune_controls.add_argument('--roll-limits', type=float, nargs=2)
        tune_controls.add_argument('--pitch', type=float, nargs=3)
        tune_controls.add_argument('--pitch-limits', type=float, nargs=2)
        tune_controls.add_argument('--yaw', type=float, nargs=3)
        tune_controls.add_argument('--yaw-limits', type=float, nargs=2)
        tune_controls.add_argument('--xy', type=float, nargs=3)
        tune_controls.add_argument('--z', type=float, nargs=3)
        tune_controls.add_argument('--z-limits', type=float, nargs=2)
        tune_controls.add_argument('--tau', type=float, nargs=6)
        tune_controls.set_defaults(func=self._handle_tune_controls)

        tune_dynamics = subparsers.add_parser('tune_dynamics')
        tune_dynamics.add_argument('--mass', type=float)
        tune_dynamics.add_argument('--volume', type=float)
        tune_dynamics.add_argument('--water_density', type=float)
        tune_dynamics.add_argument('--center_of_gravity', type=float, nargs=3)
        tune_dynamics.add_argument('--center_of_buoyancy', type=float, nargs=3)
        tune_dynamics.add_argument('--moments', type=float, nargs=6)
        tune_dynamics.add_argument('--linear_damping', type=float, nargs=6)
        tune_dynamics.add_argument('--quadratic_damping', type=float, nargs=6)
        tune_dynamics.add_argument('--added_mass', type=float, nargs=6)
        tune_dynamics.set_defaults(func=self._handle_tune_dynamics)

        goto = subparsers.add_parser('goto')
        goto.add_argument('x', type=float)
        goto.add_argument('y', type=float)
        goto.add_argument('z', type=float)
        goto.add_argument('yaw', type=float)
        goto.add_argument('--v', type=float)
        goto.add_argument('--a', type=float)
        goto.add_argument('--j', type=float)
        goto.set_defaults(func=self._handle_goto)

        goto_relative = subparsers.add_parser('goto_relative')
        goto_relative.add_argument('x', type=float)
        goto_relative.add_argument('y', type=float)
        goto_relative.add_argument('z', type=float)
        goto_relative.add_argument('yaw', type=float)
        goto_relative.add_argument('--v', type=float)
        goto_relative.add_argument('--a', type=float)
        goto_relative.add_argument('--j', type=float)
        goto_relative.set_defaults(func=self._handle_goto_relative)

        goto_pid = subparsers.add_parser('goto_pid')
        goto_pid.add_argument('x', type=float)
        goto_pid.add_argument('y', type=float)
        goto_pid.add_argument('z', type=float)
        goto_pid.add_argument('yaw', type=float)
        goto_pid.set_defaults(func=self._handle_goto_pid)

        enable_pids = subparsers.add_parser('enable_pids')
        enable_pids.add_argument('x',  type=float)
        enable_pids.add_argument('y',  type=float)
        enable_pids.add_argument('z',  type=float)
        enable_pids.add_argument('yaw',  type=float)
        enable_pids.add_argument('--enable_xy', action='store_true')
        enable_pids.add_argument('--enable_z', action='store_true')
        enable_pids.add_argument('--enable_yaw', action='store_true')
        enable_pids.set_defaults(func=self._handle_enable_pids)

        arm = subparsers.add_parser('arm')
        arm.set_defaults(func=self._handle_arm)

        arm = subparsers.add_parser('disarm')
        arm.set_defaults(func=self._handle_disarm)

        enable = subparsers.add_parser('enable')
        enable.set_defaults(func=self._handle_enable)

        disable = subparsers.add_parser('disable')
        disable.set_defaults(func=self._handle_disable)

        retare = subparsers.add_parser('retare')
        retare.set_defaults(func=self._handle_retare)

        return parser


def main():
    rospy.init_node('teleop_mission')
    m = TeleopMission()
    m.start()