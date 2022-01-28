from typing import Optional

import rospy
import numpy as np
from dynamics.dynamics import Dynamics
from geometry_msgs.msg import Pose, Twist, WrenchStamped, Vector3
from tauv_msgs.msg import ControllerCmd as ControllerCmdMsg
from tauv_util.types import tl, tm
from tauv_util.transforms import quat_to_rpy
from nav_msgs.msg import Odometry as OdometryMsg
from scipy.spatial.transform import Rotation


class Controller:
    def __init__(self):
        self._dt: float = 0.02

        self._pose: Optional[Pose] = None
        self._body_twist: Optional[Twist] = None

        self._odom_sub: rospy.Subscriber = rospy.Subscriber('odom', OdometryMsg, self._handle_odom)
        self._command_sub: rospy.Subscriber = rospy.Subscriber('controller_cmd', ControllerCmdMsg, self._handle_command)
        self._wrench_pub: rospy.Publisher = rospy.Publisher('wrench', WrenchStamped, queue_size=10)

        self._max_wrench: np.array = np.array(rospy.get_param('~max_wrench'))

        self._dyn: Dynamics = Dynamics(
            m=rospy.get_param('~dynamics/mass'),
            v=rospy.get_param('~dynamics/volume'),
            rho=rospy.get_param('~dynamics/water_density'),
            r_G=np.array(rospy.get_param('~dynamics/center_of_gravity')),
            r_B=np.array(rospy.get_param('~dynamics/center_of_buoyancy')),
            I=np.array(rospy.get_param('~dynamics/moments')),
            D=np.array(rospy.get_param('~dynamics/linear_damping')),
            D2=np.array(rospy.get_param('~dynamics/quadratic_damping')),
            Ma=np.array(rospy.get_param('~dynamics/added_mass')),
        )

        self._state: Optional[np.array] = None
        self._target_acceleration: Optional[np.array] = None

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        if self._pose is None or self._body_twist is None or self._target_acceleration is None:
            return

        R = Rotation.from_quat(tl(self._pose.orientation)).inv()

        eta = np.concatenate((
            tl(self._pose.position),
            quat_to_rpy(self._pose.orientation)
        ))
        v = np.concatenate((
            tl(self._body_twist.linear),
            tl(self._body_twist.angular)
        ))
        vd = np.concatenate((
            R.apply(self._target_acceleration[0:3]),
            self._target_acceleration[3:6]
        ))

        tau = self._dyn.compute_tau(eta, v, vd)
        bounded_tau = np.sign(tau) * np.minimum(np.abs(tau), self._max_wrench)

        wrench: WrenchStamped = WrenchStamped()
        wrench.header.stamp = rospy.Time.now()
        wrench.header.frame_id = 'kingfisher/base_link_ned'
        wrench.wrench.force = Vector3(bounded_tau[0], bounded_tau[1], bounded_tau[2])
        wrench.wrench.torque = Vector3(bounded_tau[3], bounded_tau[4], bounded_tau[5])
        self._wrench_pub.publish(wrench)

    def _handle_odom(self, msg: OdometryMsg):
        self._pose = msg.pose.pose
        self._body_twist = msg.twist.twist

    def _handle_command(self, msg: ControllerCmdMsg):
        self._target_acceleration = np.array([
            msg.a_x,
            msg.a_y,
            msg.a_z,
            msg.a_roll,
            msg.a_pitch,
            msg.a_yaw,
        ])

def main():
    rospy.init_node('controller')
    c = Controller()
    c.start()