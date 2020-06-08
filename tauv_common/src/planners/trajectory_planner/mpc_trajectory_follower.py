# MPC Trajectory Follower
#
# This is the bridge between reference trajectories and the low-level attitude controller.
# Uses an Model Predictive Controller (MPC) to compute optimal accelerations to follow the
# reference trajectory.
#
# Must be connected to a trajectory server which provides reference trajectories as requested
# by the MpcTrajectoryFollower.
#
# Author: Tom Scherlis
#

import rospy

from mpc.mpc import MPC
import numpy as np
from scipy.spatial.transform import Rotation

from tauv_msgs.srv import GetTraj, GetTrajRequest, GetTrajResponse
from tauv_msgs.msg import ControllerCmd
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header


class MpcTrajectoryFollower:
    def __init__(self):
        self.dt = 0.05  # 20Hz

        self.odom = rospy.get_param('~world_frame', 'odom')

        # rospy.wait_for_service('get_traj', 3)
        # self.get_traj_service = rospy.ServiceProxy('get_traj', GetTraj)
        self.pub_control = rospy.Publisher('controller_cmd', ControllerCmd, queue_size=10)

        self.p = None
        self.p_d = None
        self.ready = False

        self.sub_odom = rospy.Subscriber('odom', Odometry, self.odometry_callback)

        self.prediction_pub = rospy.Publisher('mpc_pred', Path, queue_size=10)
        self.reference_pub = rospy.Publisher('mpc_ref', Path, queue_size=10)

        # Dynamics:

        # x = [x, y, z, psi, x_d, y_d, z_d, psi_d].T
        # x_d = [x_d, y_d, z_d, psi_d, x_dd, y_dd, z_dd, psi_dd].T
        # u = [x_dd, y_dd, z_dd, psi_dd].T

        # dead-simple 4x double integrator dynamics:
        self.A = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])

        # inputs are just accelerations:
        self.B = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # TODO: load these from a config.yaml
        self.Q = np.diag([1, 1, 100, 1, .1, .1, .1, .1])
        self.R = np.diag([1, 1, 10, 1]) * 1
        self.S = self.Q * 30

        # TODO: load these from a config.yaml
        INF = 1e10
        xcon = np.array([
            [INF],
            [INF],
            [INF],
            [INF],
            [.5],
            [.5],
            [.5],
            [0.75],
        ])

        ucon = np.array([
            [INF],
            [INF],
            [1],
            [INF]
        ])

        self.u_constraints = np.hstack((-1 * ucon, ucon))
        self.x_constraints = np.hstack((-1 * xcon, xcon))

        # horizon = 20 * 0.01 = 2 seconds
        self.N = 20
        self.tstep = 0.1

        # build MPC solver object:
        self.mpc = MPC(A=self.A,
                       B=self.B,
                       Q=self.Q,
                       R=self.R,
                       S=self.S,
                       N=self.N,
                       dt=self.tstep,
                       x_constraints=self.x_constraints,
                       u_constraints=self.u_constraints)

    def update(self, timer_event):
        if not self.ready:
            rospy.logwarn_throttle(3, '[MPC Trajectory Follower] No odometry received yet! Waiting...')
            return

        req = GetTrajRequest()
        req.curr_pose = self.p
        req.curr_twist = self.p_d
        req.len = self.N + 1
        req.dt = self.tstep
        req.header.stamp = rospy.Time.now()
        req.header.frame_id = self.odom

        # traj_response = self.get_traj_service(req)
        traj_response = make_test_traj(req)

        if not traj_response.success:
            rospy.logwarn_throttle(3, '[MPC Trajectory Follower] Trajectory failure!')
            return

        poses = traj_response.poses
        twists = traj_response.twists

        x = np.zeros((8, 1))
        ref_traj = np.zeros((8, self.N + 1))

        x[:, 0] = self.to_x(self.p, self.p_d)

        for i in range(self.N + 1):
            gpose = poses[i]

            if traj_response.auto_twists:
                gtwist = None
            else:
                gtwist = twists[i]

            ref_traj[:, i] = self.to_x(gpose, gtwist)

        if traj_response.auto_twists:
            gdiff = np.diff(ref_traj, 1) * self.tstep
            ref_traj[4:8, :] = np.pad(gdiff[0:4], ((0, 0), (0, 1)), 'edge')

        u_mpc, x_mpc = self.mpc.solve(x, ref_traj)

        if u_mpc is None:
            rospy.logerr("[MPC Controller] Error computing MPC trajectory!")
            u_mpc = np.zeros((6, self.N))
            x_mpc = None

        u = u_mpc[:, 0]

        self.reference_pub.publish(self.mpc.to_path(ref_traj, start_time=rospy.Time.now(), frame=self.odom))

        if x_mpc is not None:
            self.prediction_pub.publish(self.mpc.to_path(x_mpc, start_time=rospy.Time.now(), frame=self.odom))

        ref_rpy = Rotation.from_quat(tl(traj_response.poses[0].orientation)).as_euler('xyz')

        cmd = ControllerCmd()
        cmd.a_x = u[0]
        cmd.a_y = u[1]
        cmd.a_z = u[2]
        cmd.a_yaw = u[3]
        cmd.p_roll = ref_rpy[0]
        cmd.p_pitch = ref_rpy[1]

        self.pub_control.publish(cmd)

    def to_x(self, pose, twist):
        yaw = Rotation.from_quat(tl(pose.orientation)).as_euler('ZYX')[0]

        if twist is None:
            twist = Twist()

        x = [pose.position.x, pose.position.y, pose.position.z, yaw,
             twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.z]
        return x

    def odometry_callback(self, msg):
        p = msg.pose.pose
        v = msg.twist.twist

        self.p = p

        # TODO: why is ground truth published in wrong frame???
        # R = Rotation.from_quat(tl(p.orientation))
        R = Rotation.from_quat([0, 0, 0, 1])

        lin_vel = R.apply(tl(v.linear))
        ang_vel = R.apply(tl(v.angular))
        p_d = Twist()
        p_d.linear = Vector3(lin_vel[0], lin_vel[1], lin_vel[2])
        p_d.angular = Vector3(ang_vel[0], ang_vel[1], ang_vel[2])
        self.p_d = p_d

        self.ready = True

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self.dt), self.update)
        rospy.spin()


def make_test_traj(req):
    res = GetTrajResponse()
    res.success = True
    res.auto_twists = True

    pos = [10, 10, -5]

    poses = []
    for i in range(req.len):
        p = Pose()
        p.position.x = pos[0]
        p.position.y = pos[1]
        p.position.z = pos[2]
        p.orientation.w = 1
        poses.append(p)

    res.poses = poses
    return res


def tl(v):
    if isinstance(v, Quaternion):
        return [v.x, v.y, v.z, v.w]
    if isinstance(v, Vector3):
        return [v.x, v.y, v.z]


def main():
    rospy.init_node('mpc_trajectory_follower')
    mtf = MpcTrajectoryFollower()
    mtf.start()
