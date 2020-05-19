# hybrid_controller.py
#
# This is an attitute
#
#

import rospy
from simple_pid import PID
from dynamics.dynamics import Dynamics
from tauv_msgs.msg import ControllerCmd
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import WrenchStamped, Wrench, Vector3
from nav_msgs.msg import Odometry


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

        # TODO: expose ability to tune via ros service
        self._build_pids([1, 0.1, 0], [1, 0.1, 0])

        self.sub_odom = rospy.Subscriber('odom', Odometry, self.odometry_callback)
        self.sub_command = rospy.Subscriber('controller_cmd', ControllerCmd, self.plan_callback)

    def _build_pids(self, roll_tunings, pitch_tunings):
        self.roll_pid = PID(Kp=roll_tunings[0],
                            Kd=roll_tunings[1],
                            Ki=roll_tunings[2],
                            setpoint=0,
                            sample_time=None,
                            output_limits=None,
                            auto_mode=True,
                            proportional_on_measurement=False)
        self.pitch_pid = PID(Kp=pitch_tunings[0],
                             Kd=pitch_tunings[1],
                             Ki=pitch_tunings[2],
                             setpoint=0,
                             sample_time=None,
                             output_limits=None,
                             auto_mode=True,
                             proportional_on_measurement=False)

    def control_update(self):
        failsafe = False

        if self.eta is None or self.v is None:
            rospy.logwarn_throttle_identical(3, 'Odometry not yet received: Controller waiting...')
            failsafe = True

        if self.last_updated is None or rospy.Time.now().to_sec() - self.last_updated > self.timeout_duration:
            failsafe = True
            rospy.logwarn_throttle_identical(3, 'No controller command received recently: entering failsafe mode!')

        if not failsafe:
            roll_error = self.target_roll - self.eta[3]
            pitch_error = self.target_pitch - self.eta[4]

            roll_effort = self.roll_pid(roll_error)
            pitch_effort = self.pitch_pid(pitch_error)

            vd_pid = [0, 0, 0, roll_effort, pitch_effort, 0]
            eta_dd_pid = self.dyn.get_eta_dd(self.eta, self.eta_d, vd_pid)
            eta_dd_command = [self.target_acc[0], self.target_acc[1], self.target_acc[2], 0, 0, self.target_yaw_acc]

            eta_dd = np.array(eta_dd_pid) + np.array(eta_dd_command)

            tau = self.dyn.compute_tau(self.eta, self.eta_d, eta_dd)
        else:
            # TODO: support more complex failsafe behaviors, eg: don't publish
            tau = [0]*6

        wrench = WrenchStamped()
        wrench.header.stamp = rospy.Time.now()
        wrench.header.frame_id = self.base_link
        wrench.wrench.force = Vector3(tau[0], -tau[1], -tau[2])
        wrench.wrench.torque = Vector3(tau[3], -tau[4], -tau[5])
        self.pub_wrench.publish((wrench))

    def odometry_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v_l = msg.twist.twist.linear
        v_a = msg.twist.twist.angular

        rpy = Rotation.from_quat(q).as_euler('xyz')

        self.eta = [p.x, -p.y, -p.z, rpy[0], -rpy[1], -rpy[2]]
        self.v = [v_l.x, -v_l.y, -v_l.z, v_a.x, -v_a.y, -v_a.z]
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


def main():
    rospy.init_node('attitude_controller')
    ac = AttitudeController()
    ac.start()

