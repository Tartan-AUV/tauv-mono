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

        rospy.wait_for_service('get_traj', 3)
        self.get_traj_service = rospy.ServiceProxy('get_traj', GetTraj)
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

        self.Q = np.diag([1, 1, 1, 1])
        self.R = np.diag([1, 1, 1, 1])
        self.S = self.R * 10

        # horizon = 20 * 0.05 = 1 second
        self.N = 20
        self.tstep = self.dt

        # build MPC solver object:
        self.mpc = MPC(A=self.A,
                       B=self.B,
                       Q=self.Q,
                       R=self.R,
                       S=self.S,
                       N=self.N,
                       dt=self.tstep)

    def update(self, timer_event):
        if not self.ready:
            rospy.logwarn_throttle(3, '[MPC Trajectory Follower] No odometry received yet! Waiting...')
            return

        req = GetTrajRequest()
        req.start_pose = self.p
        req.start_twist = self.p_d
        req.len = self.N + 1
        req.dt = self.tstep
        req.header.stamp = rospy.Time.now()
        req.header.frame_id = self.odom

        traj_response = self.get_traj_service(req)

        if not traj_response.success:
            rospy.logwarn_throttle(3, '[MPC Trajectory Follower] Trajectory failure!')
            return

        poses = traj_response.poses
        twists = traj_response.twists

        x = np.zeros((8, 1))
        ref_traj = np.zeros((8, self.N + 1))

        x[:,0] = self.to_x(self.p, self.p_d)

        for i in range(self.N + 1):
            gpose = poses[i]

            if traj_response.auto_twists:
                gtwist = None
            else:
                gtwist = twists[i]

            ref_traj[:,i] = self.to_x(gpose, gtwist)

        if traj_response.auto_twists:
            gdiff = np.diff(ref_traj, 1) * self.tstep
            ref_traj[4:8, :] = np.pad(gdiff, ((0, 0), (0, 1)), 'edge')

        u_mpc, x_mpc = self.mpc.solve(x, ref_traj)

        self.reference_pub.publish(self.mpc.to_path(np.array(ref_traj).transpose(), start_time=rospy.Time.now(), frame=self.odom))
        self.prediction_pub.publish(self.mpc.to_path(x_mpc, start_time=rospy.Time.now(), frame=self.odom))

        ref_rpy = Rotation.from_quat(tl(traj_response.poses[0].orientation)).as_euler('xyz')

        cmd = ControllerCmd()
        cmd.a_x = u_mpc[0]
        cmd.a_y = u_mpc[1]
        cmd.a_z = u_mpc[2]
        cmd.a_yaw = u_mpc[3]
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

        R = Rotation.from_quat(tl(p.orientation))

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


def tl(v):
    if isinstance(v, Quaternion):
        return [v.x, v.y, v.z, v.w]
    if isinstance(v, Vector3):
        return [v.x, v.y, v.z]


def main():
    rospy.init_node('mpc_trajectory_follower')
    mtf = MpcTrajectoryFollower()
    mtf.start()
