# hybrid_controller.py
#
# This is an attitude controller for roll and pitch
# Also accepts a world-space linear acceleration and yaw acceleration command
# Uses the vehicle dynamics to provide more accurate acceleration control.
#
# NOTE: Acceleration control should only be used with a closed loop controller, as
# inverse dynamics can cause a positive feedback loop if dynamics are not modelled correctly:
# eg, estimated damping too high can result in rapid uncontrolled acceleration!
#

import rospy
from simple_pid import PID
from dynamics.dynamics import Dynamics
from tauv_msgs.msg import ControllerCmd
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import WrenchStamped, Wrench, Vector3, Quaternion
from nav_msgs.msg import Odometry
import control
import copy


class AttitudeController:
    def __init__(self):
        self.dt = 0.02  # 50Hz
        self.dyn = Dynamics()
        self.target_acc = None
        self.target_yaw_acc = None
        self.target_roll = None
        self.target_pitch = None
        self.last_updated = None

        self.eta = None
        self.v = None
        self.eta_d = None

        self.pub_wrench = rospy.Publisher('wrench', WrenchStamped, queue_size=10)

        # TODO: load from rosparams
        self.timeout_duration = 2.0  # timeout is 2 seconds.
        self.base_link = 'base_link'

        # NA = 1
        # self.Q = np.diag([NA, NA, NA, 300, 300, NA, NA, NA, NA, 5, 5, NA])
        # self.R = np.eye(6) * 1

        # TODO: expose ability to tune via ros service
        self._build_pids([50, 10, 0], [50, 10, 0])

        self.sub_odom = rospy.Subscriber('odom', Odometry, self.odometry_callback)
        self.sub_command = rospy.Subscriber('controller_cmd', ControllerCmd, self.plan_callback)

    def _build_pids(self, roll_tunings, pitch_tunings):
        self.roll_pid = PID(Kp=roll_tunings[0],
                            Kd=roll_tunings[1],
                            Ki=roll_tunings[2],
                            setpoint=0,
                            sample_time=None,
                            output_limits=(None, None),
                            auto_mode=True,
                            proportional_on_measurement=False)
        self.pitch_pid = PID(Kp=pitch_tunings[0],
                             Kd=pitch_tunings[1],
                             Ki=pitch_tunings[2],
                             setpoint=0,
                             sample_time=None,
                             output_limits=(None, None),
                             auto_mode=True,
                             proportional_on_measurement=False)

    def control_update(self, timer_event):
        failsafe = False

        if self.eta is None or self.v is None:
            rospy.logwarn_throttle_identical(3, 'Odometry not yet received: Controller waiting...')
            failsafe = True

        if self.last_updated is None or rospy.Time.now().to_sec() - self.last_updated > self.timeout_duration:
            failsafe = True
            rospy.logwarn_throttle_identical(3, 'No controller command received recently: entering failsafe mode!')

        if not failsafe:
            eta = self.eta
            v = self.v
            ta = self.target_acc
            ty = self.target_yaw_acc

            roll_error = self.target_roll - eta[3]
            pitch_error = self.target_pitch - eta[4]

            roll_effort = self.roll_pid(-roll_error)
            pitch_effort = self.pitch_pid(-pitch_error)

            vd_pid = [0, 0, 0, roll_effort, pitch_effort, 0]

            R = Rotation.from_euler('xyz', eta[3:6]).inv()
            target_acc_body = R.apply(ta)
            target_yaw_body = R.apply([0, 0, ty])
            vd_command = np.hstack((target_acc_body, target_yaw_body))

            vd = np.array(vd_pid) + np.array(vd_command)

            tau = self.dyn.compute_tau(eta, v, vd)
        else:
            tau = [0] * 6

        wrench = WrenchStamped()
        wrench.header.stamp = rospy.Time.now()
        wrench.header.frame_id = self.base_link
        wrench.wrench.force = Vector3(tau[0], -tau[1], -tau[2])
        wrench.wrench.torque = Vector3(tau[3], -tau[4], -tau[5])

        self.pub_wrench.publish(wrench)

    def odometry_callback(self, msg):
        p = msg.pose.pose.position
        q = tl(msg.pose.pose.orientation)
        v_l = msg.twist.twist.linear
        v_a = msg.twist.twist.angular

        R = Rotation.from_quat(q)
        rpy = R.as_euler('xyz')

        self.eta = [p.x, -p.y, -p.z, rpy[0], -rpy[1], -rpy[2]]

        # TODO: why is odometry published in the wrong frame??
        v_l = R.inv().apply(tl(v_l))
        v_a = R.inv().apply(tl(v_a))

        self.v = [v_l[0], -v_l[1], -v_l[2], v_a[0], -v_a[1], -v_a[2]]
        self.eta_d = self.dyn.get_eta_d(self.eta, self.v)

    def plan_callback(self, msg):
        self.target_acc = [msg.a_x, -msg.a_y, -msg.a_z]
        self.target_yaw_acc = -msg.a_yaw
        self.target_roll = msg.p_roll
        self.target_pitch = -msg.p_pitch

        self.last_updated = rospy.Time.now().to_sec()

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self.dt), self.control_update)
        rospy.spin()


def tl(v):
    if isinstance(v, Quaternion):
        return [v.x, v.y, v.z, v.w]
    if isinstance(v, Vector3):
        return [v.x, v.y, v.z]


def main():
    rospy.init_node('attitude_controller')
    ac = AttitudeController()
    ac.start()
