import rospy
import argparse
import numpy as np
from typing import Optional

from tauv_msgs.msg import PIDTuning, DynamicsTuning
from tauv_msgs.srv import TuneController, TuneControllerRequest, TuneControllerResponse, TunePIDPlanner, TunePIDPlannerRequest, TunePIDPlannerResponse, TuneDynamics, TuneDynamicsRequest, TuneDynamicsResponse
from geometry_msgs.msg import Pose, Twist, Vector3
from std_srvs.srv import SetBool
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

        self._tune_controller_srv: rospy.ServiceProxy = rospy.ServiceProxy('/gnc/controller/tune_controller', TuneController)
        self._tune_pid_planner_srv: rospy.ServiceProxy = rospy.ServiceProxy('/gnc/pid_planner/tune_pid_planner', TunePIDPlanner)
        self._tune_dynamics_srv: rospy.ServiceProxy = rospy.ServiceProxy('/gnc/controller/tune_dynamics', TuneDynamics)
        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('/vehicle/thrusters/arm', SetBool)

    def start(self):
        while True:
            cmd = input('>>> ')
            try:
                args = self._parser.parse_args(cmd.split())
                args.func(args)
            except ArgumentParserError as e:
                print('error:', e)
                continue

    def _handle_tune_controller(self, args):
        print('tune_controller', args.x, args.y, args.z, args.roll, args.pitch, args.yaw)

        pid_tunings = []

        if args.x is not None:
            p = PIDTuning(
               axis="x",
               kp=args.x[0],
               ki=args.x[1],
               kd=args.x[2],
               tau=args.x[3],
               limits=args.x[4:6]
            )
            pid_tunings.append(p)

        if args.y is not None:
            p = PIDTuning(
                axis="y",
                kp=args.y[0],
                ki=args.y[1],
                kd=args.y[2],
                tau=args.y[3],
                limits=args.y[4:6]
            )
            pid_tunings.append(p)

        if args.z is not None:
            p = PIDTuning(
                axis="z",
                kp=args.z[0],
                ki=args.z[1],
                kd=args.z[2],
                tau=args.z[3],
                limits=args.z[4:6]
            )
            pid_tunings.append(p)

        if args.yaw is not None:
            p = PIDTuning(
                axis="yaw",
                kp=args.yaw[0],
                ki=args.yaw[1],
                kd=args.yaw[2],
                tau=args.yaw[3],
                limits=[args.yaw[4], args.yaw[5]]
            )
            pid_tunings.append(p)

        req = TunePIDPlannerRequest()
        req.tunings = pid_tunings
        try:
            self._tune_pid_planner_srv.call(req)
        except Exception as e:
            print(e)

        pid_tunings = []

        if args.roll is not None:
            p = PIDTuning(
                axis="roll",
                kp=args.roll[0],
                ki=args.roll[1],
                kd=args.roll[2],
                tau=args.roll[3],
                limits=args.roll[4:6]
            )
            pid_tunings.append(p)

        if args.pitch is not None:
            p = PIDTuning(
                axis="pitch",
                kp=args.pitch[0],
                ki=args.pitch[1],
                kd=args.pitch[2],
                tau=args.pitch[3],
                limits=args.pitch[4:6]
            )
            pid_tunings.append(p)

        if args.roll is not None:
            p = PIDTuning()
            pid_tunings.append(p)

        req = TuneControllerRequest()
        req.tunings = pid_tunings
        try:
            self._tune_controller_srv.call(req)
        except Exception as e:
            print(e)

    def _handle_tune_dynamics(self, args):
        print('tune_dynamics', args.mass, args.volume, args.water_density, args.center_of_gravity, args.center_of_buoyancy, args.moments, args.linear_damping, args.quadratic_damping, args.added_mass)

        t = DynamicsTuning()
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
        req.tuning = t
        self._tune_dynamics_srv.call(req)

    def _handle_goto(self, args):
        v = args.v if args.v is not None else .1
        a = args.a if args.a is not None else .1
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
        v = args.v if args.v is not None else .1
        a = args.a if args.a is not None else .1
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

    def _handle_arm(self, args):
        print('arm')

        self._arm_srv.call(True)

    def _handle_disarm(self, args):
        print('disarm')

        self._arm_srv.call(False)

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = ThrowingArgumentParser(prog="teleop_mission")
        subparsers = parser.add_subparsers()

        tune_controller = subparsers.add_parser('tune_controller')
        tune_controller.add_argument('--roll', type=float, nargs=6)
        tune_controller.add_argument('--pitch', type=float, nargs=6)
        tune_controller.add_argument('--yaw', type=float, nargs=6)
        tune_controller.add_argument('--x', type=float, nargs=6)
        tune_controller.add_argument('--y', type=float, nargs=6)
        tune_controller.add_argument('--z', type=float, nargs=6)
        tune_controller.set_defaults(func=self._handle_tune_controller)

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

        arm = subparsers.add_parser('arm')
        arm.set_defaults(func=self._handle_arm)

        arm = subparsers.add_parser('disarm')
        arm.set_defaults(func=self._handle_disarm)

        return parser


def main():
    rospy.init_node('teleop_mission')
    m = TeleopMission()
    m.start()