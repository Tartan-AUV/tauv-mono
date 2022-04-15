import rospy
import argparse
import numpy as np
from typing import Optional

from tauv_msgs.msg import ControlsTunings, DynamicsTunings
from tauv_msgs.srv import TuneControls, TuneControlsRequest, TuneControlsResponse, TuneDynamics, TuneDynamicsRequest, TuneDynamicsResponse, GetTraj, GetTrajRequest, GetTrajResponse, HoldPose, HoldPoseRequest, HoldPoseResponse
from geometry_msgs.msg import Pose, Twist, Vector3
from nav_msgs.msg import Odometry as Odom, Path
from tauv_util.transforms import rpy_to_quat
from motion.trajectories.linear_trajectory import Waypoint, LinearTrajectory


class ArgumentParserError(Exception): pass


class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)


class TeleopMission:

    def __init__(self):
        self._parser = self._build_parser()

        self._traj: Optional[LinearTrajectory] = None
        self._pose: Optional[Pose] = None
        self._twist: Optional[Twist] = None

        self._get_traj_srv: rospy.Service = rospy.Service('get_traj', GetTraj, self._handle_get_traj)

        self._tune_controls_srv: rospy.ServiceProxy = rospy.ServiceProxy('tune_controls', TuneControls)
        self._tune_dynamics_srv: rospy.ServiceProxy = rospy.ServiceProxy('tune_dynamics', TuneDynamics)
        self._hold_pose_srv: rospy.ServiceProxy = rospy.ServiceProxy('hold_pose', HoldPose)

        self._odom_sub: rospy.Subscriber = rospy.Subscriber('odom', Odom, self._handle_odom)

        self._path_pub: rospy.Publisher = rospy.Publisher('path', Path, queue_size=10)

    def start(self):
        while True:
            cmd = input('>>> ')
            try:
                args = self._parser.parse_args(cmd.split())
                args.func(args)
            except ArgumentParserError as e:
                print('error:', e)
                continue

    def _handle_get_traj(self, req: GetTrajRequest) -> GetTrajResponse:
        response = GetTrajResponse()

        if self._traj is None:
            response.success = False
            return response

        return self._traj.get_points(req)

    def _handle_tune_controls(self, args):
        print('tune_controls', args.roll, args.pitch, args.z)

        t = ControlsTunings()
        if args.roll is not None:
            t.update_roll = True
            t.roll_tunings = args.roll

        if args.pitch is not None:
            t.update_pitch = True
            t.pitch_tunings = args.pitch

        if args.z is not None:
            t.update_z = True
            t.z_tunings = args.z

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
        print('goto', args.x, args.y, args.z, args.roll, args.pitch, args.yaw, args.l, args.a)

        start_waypoint = Waypoint(self._pose, 0.1, 0.1)

        end_pose = Pose()
        end_pose.position = Vector3(args.x, args.y, args.z)
        end_pose.orientation = rpy_to_quat(np.array([args.roll, args.pitch, args.yaw]))

        end_waypoint = Waypoint(end_pose, 0.1, 0.1)

        try:
            self._traj = LinearTrajectory(
                [start_waypoint, end_waypoint],
                tuple(args.l),
                tuple(args.a),
            )

            self._traj.set_executing()

            self._path_pub.publish(self._traj.as_path())
        except Exception as e:
            print(e)

    def _handle_hold_pose(self, args):
        print('hold_pose', args.enable, args.roll, args.pitch, args.z)

        pose = Pose()
        pose.position = Vector3(0.0, 0.0, args.z)
        pose.orientation = rpy_to_quat(np.array([args.roll, args.pitch, 0.0]))

        req: HoldPoseRequest = HoldPoseRequest()
        req.enable = args.enable is not None
        req.pose = pose
        self._hold_pose_srv.call(req)


    def _handle_odom(self, msg: Odom):
        self._pose = msg.pose.pose
        self._twist = msg.twist.twist

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = ThrowingArgumentParser(prog="teleop_mission")
        subparsers = parser.add_subparsers()

        tune_controls = subparsers.add_parser('tune_controls')
        tune_controls.add_argument('--roll', type=float, nargs=3)
        tune_controls.add_argument('--pitch', type=float, nargs=3)
        tune_controls.add_argument('--z', type=float, nargs=3)
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
        goto.add_argument('roll', type=float)
        goto.add_argument('pitch', type=float)
        goto.add_argument('yaw', type=float)
        goto.add_argument('--l', type=float, nargs=3, default=[0.2, 0.2, 100.0])
        goto.add_argument('--a', type=float, nargs=3, default=[0.1, 0.2, 100.0])
        goto.set_defaults(func=self._handle_goto)

        hold_pose = subparsers.add_parser('hold_pose')
        hold_pose.add_argument('roll', type=float)
        hold_pose.add_argument('pitch', type=float)
        hold_pose.add_argument('z', type=float)
        hold_pose.add_argument('--enable', action='store_true')
        hold_pose.set_defaults(func=self._handle_hold_pose)

        return parser


def main():
    rospy.init_node('teleop_mission')
    m = TeleopMission()
    m.start()