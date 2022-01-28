import rospy
import numpy as np
from tauv_msgs.srv import GetTraj, GetTrajRequest, GetTrajResponse
from tauv_msgs.msg import ControllerCmd as ControllerCmdMsg
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry as OdometryMsg
from typing import Optional
from tauv_util.types import tl, tm
from tauv_util.transforms import quat_to_rpy, twist_body_to_world, twist_world_to_body

from .mpc.mpc import MPC


class MPCPlanner:
    def __init__(self):
        self._dt: float = 0.05

        self._get_traj_service: rospy.ServiceProxy = rospy.ServiceProxy('get_traj', GetTraj)

        self._command_pub: rospy.Publisher = rospy.Publisher('controller_cmd', ControllerCmdMsg, queue_size=10)
        self._odom_sub: rospy.Subscriber = rospy.Subscriber('odom', OdometryMsg, self._handle_odom)

        self._pose: Optional[Pose] = None
        self._world_twist: Optional[Twist] = None

        self._n: int = rospy.get_param('~n')
        self._tstep: float = rospy.get_param('~tstep')

        self._A: np.array = np.vstack((
            np.hstack((np.zeros((6, 6)), np.identity(6))),
            np.zeros((6, 12))
        ))

        self._B: np.array = np.vstack((
            np.zeros((6, 6)),
            np.identity(6)
        ))

        self._Q: np.array = np.diag(np.array(rospy.get_param('~Q')))
        self._R: np.array = np.diag(np.array(rospy.get_param('~R')))
        self._S: np.array = rospy.get_param('~S_coeff') * self._Q

        xcon: np.array = np.zeros((12, 1))
        xcon[:, 0] = rospy.get_param('~xcon')
        ucon: np.array = np.zeros((6, 1))
        ucon[:, 0] = rospy.get_param('~ucon')

        self._u_constraints: np.array = np.hstack((-1 * ucon, ucon))
        self._x_constraints: np.array = np.hstack((-1 * xcon, xcon))

        self._mpc: MPC = MPC(A=self._A,
                             B=self._B,
                             Q=self._Q,
                             R=self._R,
                             S=self._S,
                             N=self._n,
                             dt=self._tstep,
                             x_constraints=self._x_constraints,
                             u_constraints=self._u_constraints)

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        if self._pose is None or self._world_twist is None:
            return

        traj_response = self._get_traj()

        if traj_response is None:
            return

        poses = traj_response.poses
        twists = traj_response.twists

        body_twist = twist_world_to_body(self._pose, self._world_twist)
        current_twist = Twist()
        current_twist.linear = self._world_twist.linear
        current_twist.angular = body_twist.angular

        x = np.zeros((12, 1))
        x[:, 0] = self._to_x(self._pose, current_twist)

        ref_traj = np.zeros((12, self._n + 1))

        for i in range(self._n + 1):
            gpose = poses[i]

            if traj_response.auto_twists:
                gtwist = None
            else:
                gtwist = twists[i]

            ref_traj[:, i] = self._to_x(gpose, gtwist)

        # TODO: Re-wind orientation

        # Re-wind yaw to prevent jumps when winding:
        # psi = x[3]
        # if ref_traj[3, 0] - psi < -np.pi:
        #     ref_traj[3, 0] += 2 * np.pi
        # if ref_traj[3, 0] - psi > np.pi:
        #     ref_traj[3, 0] -= 2 * np.pi
        # for i in range(len(ref_traj[3, 0:-1])):
        #     if ref_traj[3, i + 1] - ref_traj[3, i] < -np.pi:
        #         ref_traj[3, i + 1] += 2 * np.pi
        #     if ref_traj[3, i + 1] - ref_traj[3, i] > np.pi:
        #         ref_traj[3, i + 1] -= 2 * np.pi

        u_mpc, x_mpc = self._mpc.solve(x, ref_traj)

        if u_mpc is None:
            self._send_command(np.zeros(6))
            return

        u = u_mpc[:, 0]

        self._send_command(u)

    def _send_command(self, cmd: np.array):
        msg: ControllerCmdMsg = ControllerCmdMsg()
        msg.a_x = cmd[0]
        msg.a_y = cmd[1]
        msg.a_z = cmd[2]
        msg.a_roll = cmd[3]
        msg.a_pitch = cmd[4]
        msg.a_yaw = cmd[5]
        self._command_pub.publish(msg)

    def _get_traj(self) -> Optional[GetTrajResponse]:
        req = GetTrajRequest()

        req.curr_pose = self._pose
        req.curr_twist = self._world_twist
        req.len = self._n + 1
        req.dt = self._tstep
        req.header.stamp = rospy.Time.now()
        req.header.frame_id = 'odom'
        req.curr_time = rospy.Time.now()

        try:
            res =  self._get_traj_service(req)
        except:
            return None

        if not res.success:
            return None

        return res

    def _handle_odom(self, msg: OdometryMsg):
        self._pose = msg.pose.pose
        self._world_twist = twist_body_to_world(msg.pose.pose, msg.twist.twist)

    def _to_x(self, pose: Pose, twist: Optional[Twist]) -> np.array:
        if twist is None:
            twist = Twist()

        x = np.concatenate((
            tl(pose.position),
            quat_to_rpy(pose.orientation),
            tl(twist.linear),
            tl(twist.angular)
        ))

        return x

def main():
    rospy.init_node('mpc_planner')
    p = MPCPlanner()
    p.start()
