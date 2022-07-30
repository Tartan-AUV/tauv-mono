import rospy
import argparse
import numpy as np
from typing import Optional
from math import pi

from tauv_msgs.msg import ControlsTunings, DynamicsTunings
from tauv_msgs.srv import TuneControls, TuneControlsRequest, TuneControlsResponse, TuneDynamics, TuneDynamicsRequest, TuneDynamicsResponse, GetTraj, GetTrajRequest, GetTrajResponse, SetTargetPose, SetTargetPoseRequest, SetTargetPoseResponse
from geometry_msgs.msg import Pose, Twist, Vector3
from nav_msgs.msg import Odometry as Odom, Path
from tauv_util.types import tl, tm
from std_srvs.srv import SetBool
from tauv_util.transforms import rpy_to_quat, quat_to_rpy
from scipy.spatial.transform import Rotation
from motion.trajectories.linear_trajectory import Waypoint, LinearTrajectory
from motion.motion_utils import MotionUtils


class ArgumentParserError(Exception): pass


class ThrowingArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)


class TeleopMission:

    def __init__(self):
        self._parser = self._build_parser()

        self._motion = MotionUtils()

        self._traj: Optional[LinearTrajectory] = None
        self._pose: Optional[Pose] = None
        self._twist: Optional[Twist] = None

        # self._get_traj_srv: rospy.Service = rospy.Service('get_traj', GetTraj, self._handle_get_traj)

        self._tune_controls_srv: rospy.ServiceProxy = rospy.ServiceProxy('tune_controls', TuneControls)
        self._tune_dynamics_srv: rospy.ServiceProxy = rospy.ServiceProxy('tune_dynamics', TuneDynamics)
        self._target_pose_srv: rospy.ServiceProxy = rospy.ServiceProxy('set_target_pose', SetTargetPose)
        self._hold_z_srv: rospy.ServiceProxy = rospy.ServiceProxy('set_hold_z', SetBool)

        self._odom_sub: rospy.Subscriber = rospy.Subscriber('odom', Odom, self._handle_odom)

        self._path_pub: rospy.Publisher = rospy.Publisher('path', Path, queue_size=10)

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

        if args.roll_limits is not None:
            t.update_roll_limits = True
            t.roll_limits = args.roll_limits

        if args.pitch is not None:
            t.update_pitch = True
            t.pitch_tunings = args.pitch

        if args.pitch_limits is not None:
            t.update_pitch_limits = True
            t.pitch_limits = args.pitch_limits

        if args.z is not None:
            t.update_z = True
            t.z_tunings = args.z

        if args.z_limits is not None:
            t.update_z_limits = True
            t.z_limits = args.z_limits

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

        req: SetTargetPoseRequest = SetTargetPoseRequest()
        req.pose = pose
        self._target_pose_srv.call(req)

        self._hold_z_srv.call(args.enable)

    def _handle_go(self, args):
        print('go', args.x, args.y, args.z, args.yaw)

        start_waypoint = Waypoint(self._pose, 0.1, 0.1)

        R = Rotation.from_quat(tl(self._pose.orientation))
        end_position = tl(self._pose.position) + R.apply(np.array([args.x, args.y, args.z]))

        end_pose = Pose()
        end_pose.position = tm(end_position, Vector3)
        end_pose.orientation = rpy_to_quat(np.array([0.0, 0.0, quat_to_rpy(self._pose.orientation)[2] + args.yaw]))

        end_waypoint = Waypoint(end_pose, 0.1, 0.1)

        try:
            self._traj = LinearTrajectory(
                [start_waypoint, end_waypoint],
                tuple(args.l),
                tuple(args.a),
            )

            self._motion.set_trajectory(self._traj)

            # self._traj.start()

            # self._path_pub.publish(self._traj.as_path())
        except Exception as e:
            print(e)

    def _handle_arm(self, args):
        print('arm', args.arm)

        self._arm_srv.call(args.arm)

    def _handle_prequal(self, args):
        print('prequal')

        R = Rotation.from_quat(tl(self._pose.orientation))

        start_position = tl(self._pose.position)
        start_orientation = quat_to_rpy(self._pose.orientation)
        start_pose = Pose(tm(start_position, Vector3), rpy_to_quat(start_orientation))

        position_1 = start_position + R.apply(np.array([6.0, 0.01, 0.01]))
        orientation_1 = start_orientation + np.array([0.01, 0.01, 0.01])
        pose_1 = Pose(tm(position_1, Vector3), rpy_to_quat(orientation_1))
        position_2 = start_position + R.apply(np.array([6.01, 0.02, 0.02]))
        orientation_2 = start_orientation + np.array([0.01, 0.01, pi])
        pose_2 = Pose(tm(position_2, Vector3), rpy_to_quat(orientation_2))
        position_3 = start_position + R.apply(np.array([0.01, 0.01, 0.01]))
        orientation_3 = start_orientation + np.array([0.01, 0.01, pi + 0.01])
        pose_3 = Pose(tm(position_3, Vector3), rpy_to_quat(orientation_3))

        start_waypoint = Waypoint(start_pose, 0.2, 0.2)
        waypoint_1 = Waypoint(pose_1, 0.2, 0.2)
        waypoint_2 = Waypoint(pose_2, 0.2, 0.2)
        waypoint_3 = Waypoint(pose_3, 0.2, 0.2)

        try:
            self._traj = LinearTrajectory(
                [start_waypoint, waypoint_1, waypoint_2, waypoint_3],
                tuple(args.l),
                tuple(args.a),
            )

            self._traj.set_executing()

            self._path_pub.publish(self._traj.as_path())
        except Exception as e:
            print(e)

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
        tune_controls.add_argument('--z', type=float, nargs=3)
        tune_controls.add_argument('--z-limits', type=float, nargs=2)
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

        go = subparsers.add_parser('go')
        go.add_argument('x', type=float)
        go.add_argument('y', type=float)
        go.add_argument('z', type=float)
        go.add_argument('yaw', type=float)
        go.add_argument('--l', type=float, nargs=3, default=[0.2, 0.2, 100.0])
        go.add_argument('--a', type=float, nargs=3, default=[0.1, 0.2, 100.0])
        go.set_defaults(func=self._handle_go)

        hold_pose = subparsers.add_parser('hold_pose')
        hold_pose.add_argument('roll', type=float)
        hold_pose.add_argument('pitch', type=float)
        hold_pose.add_argument('z', type=float)
        hold_pose.add_argument('--enable', action='store_true')
        hold_pose.set_defaults(func=self._handle_hold_pose)

        arm = subparsers.add_parser('arm')
        arm.add_argument('--arm', action='store_true')
        arm.set_defaults(func=self._handle_arm)

        prequal = subparsers.add_parser('prequal')
        prequal.add_argument('--l', type=float, nargs=3, default=[0.2, 0.2, 100.0])
        prequal.add_argument('--a', type=float, nargs=3, default=[0.2, 0.2, 100.0])
        prequal.set_defaults(func=self._handle_prequal)

        return parser


def main():
    rospy.init_node('teleop_mission')
    m = TeleopMission()
    m.start()