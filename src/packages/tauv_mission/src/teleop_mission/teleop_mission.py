import rospy
import argparse
from typing import Optional
import numpy as np
from math import atan2, cos, sin

from tauv_msgs.msg import PIDTuning, DynamicsTuning, DynamicsParameterConfigUpdate
from tauv_msgs.srv import \
    TuneController, TuneControllerRequest,\
    TunePIDPlanner, TunePIDPlannerRequest,\
    TuneDynamics, TuneDynamicsRequest,\
    UpdateDynamicsParameterConfigs, UpdateDynamicsParameterConfigsRequest, UpdateDynamicsParameterConfigsResponse
from geometry_msgs.msg import Pose, Twist, Point
from std_srvs.srv import SetBool
from std_msgs.msg import Float64
from tauv_msgs.srv import MapFind, MapFindRequest
from motion.motion_utils import MotionUtils
from motion.trajectories import TrajectoryStatus


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

        self._tune_controller_srv: rospy.ServiceProxy = rospy.ServiceProxy('gnc/tune_controller', TuneController)
        self._tune_pid_planner_srv: rospy.ServiceProxy = rospy.ServiceProxy('gnc/tune_pid_planner', TunePIDPlanner)
        self._tune_dynamics_srv: rospy.ServiceProxy = rospy.ServiceProxy('gnc/tune_dynamics', TuneDynamics)
        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('vehicle/thrusters/arm', SetBool)
        self._update_dynamics_parameter_configs_srv: rospy.ServiceProxy = rospy.ServiceProxy(
            'gnc/update_dynamics_parameter_configs', UpdateDynamicsParameterConfigs
        )

        self._torpedo_servo_pub: rospy.Publisher = rospy.Publisher('vehicle/servos/0/target_position', Float64, queue_size=10)

        self._goto_circle_timer: rospy.Timer = None
        self._goto_circle_v = 0.1
        self._goto_circle_a = 0.1
        self._goto_circle_j = 0.4

        self._find_srv = rospy.ServiceProxy("global_map/find", MapFind)

    def start(self):
        while not rospy.is_shutdown():
            cmd = input('>>> ')
            try:
                args = self._parser.parse_args(cmd.split())
                args.func(args)
            except ArgumentParserError as e:
                print('error:', e)
                continue

    def _handle_tune_controller(self, args):
        print('tune_controller', args.z, args.roll, args.pitch)
        pid_tunings = []

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

        req = TuneControllerRequest()
        req.tunings = pid_tunings
        try:
            self._tune_controller_srv.call(req)
        except Exception as e:
            print(e)

    def _handle_tune_pid_planner(self, args):
        print('tune_pid_planner', args.x, args.y, args.z, args.roll, args.pitch, args.yaw)

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

    def _handle_tune_dynamics(self, args):
        print('tune_dynamics', args.mass, args.volume, args.water_density, args.center_of_gravity,
              args.center_of_buoyancy, args.moments, args.linear_damping, args.quadratic_damping, args.added_mass)

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
                j=j,
                block=TrajectoryStatus.EXECUTING
            )
        except Exception as e:
            print("Exception from teleop_mission! (Gleb)")
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

    def _handle_goto_circle(self, args):
        v = args.v if args.v is not None else .1
        a = args.a if args.a is not None else .1
        j = args.j if args.j is not None else .4

        self._goto_circle_v = v
        self._goto_circle_a = a
        self._goto_circle_j = j

        self._goto_circle_timer = rospy.Timer(rospy.Duration(5.0), self._update_goto_circle)

    def _handle_stop_goto_circle(self, args):
        if self._goto_circle_timer is not None:
            self._goto_circle_timer.shutdown()

    def _update_goto_circle(self, timer_event):
        req = MapFindRequest()
        req.tag = 'circle'
        # detections, success = self._find_srv.call()
        resp = self._find_srv.call(req)

        if not resp.success or len(resp.detections) == 0:
            print('no circle')
            return

        detection = resp.detections[0]

        # Use detection
        circle_position = np.array([detection.position.x, detection.position.y, detection.position.z])
        sub_position = self._motion.get_position()

        delta_position = circle_position - sub_position
        circle_yaw = atan2(delta_position[1], delta_position[0])

        # self._motion.goto(
        #     (sub_position[0], sub_position[1], sub_position[2]),
        #     circle_yaw,
        #     v=self._goto_circle_v,
        #     a=self._goto_circle_a,
        #     j=self._goto_circle_j,
        #     block=TrajectoryStatus.FINISHED
        # )

        target_position = circle_position + np.array([
            -0.7 * cos(circle_yaw),
            -0.7 * sin(circle_yaw),
            0.2
        ])
        target_yaw = circle_yaw

        self._motion.goto(
            (target_position[0], target_position[1], target_position[2]),
            target_yaw,
            v=self._goto_circle_v,
            a=self._goto_circle_a,
            j=self._goto_circle_j,
            block=TrajectoryStatus.EXECUTING
        )

    def _handle_shoot_torpedo(self, args):
        print(f'shoot torpedo {args.torpedo}')

        if args.torpedo == 0:
            self._torpedo_servo_pub.publish(90)
            rospy.sleep(1.0)
            self._torpedo_servo_pub.publish(0)
        elif args.torpedo == 1:
            self._torpedo_servo_pub.publish(-90)
            rospy.sleep(1.0)
            self._torpedo_servo_pub.publish(0)
        else:
            self._torpedo_servo_pub.publish(0)

    def _handle_arm(self, args):
        print('arm')

        self._arm_srv.call(True)

    def _handle_disarm(self, args):
        print('disarm')

        self._arm_srv.call(False)

    def _handle_config_param_est(self, args):
        req = UpdateDynamicsParameterConfigsRequest()
        update = DynamicsParameterConfigUpdate()
        update.name = args.name

        if args.initial_value is not None:
            update.update_initial_value = True
            update.initial_value = args.initial_value

        if args.fixed is not None:
            update.update_fixed = True
            update.fixed = args.fixed == "true"

        if args.initial_covariance is not None:
            update.update_initial_covariance = True
            update.initial_covariance = args.initial_covariance

        if args.process_covariance is not None:
            update.update_process_covariance = True
            update.process_covariance = args.process_covariance

        if args.limits is not None:
            update.update_limits = True
            update.limits = args.limits

        update.reset = args.reset

        req.updates = [update]

        self._update_dynamics_parameter_configs_srv.call(req)

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = ThrowingArgumentParser(prog="teleop_mission")
        subparsers = parser.add_subparsers()

        tune_controller = subparsers.add_parser('tune_controller')
        tune_controller.add_argument('--z', type=float, nargs=6)
        tune_controller.add_argument('--roll', type=float, nargs=6)
        tune_controller.add_argument('--pitch', type=float, nargs=6)
        tune_controller.set_defaults(func=self._handle_tune_controller)

        tune_pid_planner = subparsers.add_parser('tune_pid_planner')
        tune_pid_planner.add_argument('--x', type=float, nargs=6)
        tune_pid_planner.add_argument('--y', type=float, nargs=6)
        tune_pid_planner.add_argument('--z', type=float, nargs=6)
        tune_pid_planner.add_argument('--roll', type=float, nargs=6)
        tune_pid_planner.add_argument('--pitch', type=float, nargs=6)
        tune_pid_planner.add_argument('--yaw', type=float, nargs=6)
        tune_pid_planner.set_defaults(func=self._handle_tune_pid_planner)

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

        goto_circle = subparsers.add_parser('goto_circle')
        goto_circle.add_argument('--v', type=float)
        goto_circle.add_argument('--a', type=float)
        goto_circle.add_argument('--j', type=float)
        goto_circle.set_defaults(func=self._handle_goto_circle)

        stop_goto_circle = subparsers.add_parser('stop_goto_circle')
        stop_goto_circle.set_defaults(func=self._handle_stop_goto_circle)

        shoot_torpedo = subparsers.add_parser('shoot_torpedo')
        shoot_torpedo.add_argument('torpedo', type=int)
        shoot_torpedo.set_defaults(func=self._handle_shoot_torpedo)

        arm = subparsers.add_parser('arm')
        arm.set_defaults(func=self._handle_arm)

        disarm = subparsers.add_parser('disarm')
        disarm.set_defaults(func=self._handle_disarm)

        config_param_est = subparsers.add_parser('config_param_est')
        config_param_est.add_argument('name', type=str)
        config_param_est.add_argument('--initial_value', type=float)
        config_param_est.add_argument('--fixed', type=str, choices=('true', 'false'))
        config_param_est.add_argument('--initial_covariance', type=float)
        config_param_est.add_argument('--process_covariance', type=float)
        config_param_est.add_argument('--limits', type=float, nargs=2)
        config_param_est.add_argument('--reset', default=False, action='store_true')
        config_param_est.set_defaults(func=self._handle_config_param_est)

        return parser


def main():
    rospy.init_node('teleop_mission')
    m = TeleopMission()
    m.start()