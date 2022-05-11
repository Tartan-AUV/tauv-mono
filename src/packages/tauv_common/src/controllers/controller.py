import rospy
import numpy as np
from typing import Optional
from math import pi
from simple_pid import PID

from dynamics.dynamics import Dynamics
from geometry_msgs.msg import Pose, Twist, Wrench, Vector3
from tauv_msgs.msg import ControllerCmd as ControllerCmdMsg
from tauv_msgs.srv import HoldPose, HoldPoseRequest, HoldPoseResponse, TuneDynamics, TuneDynamicsRequest, TuneDynamicsResponse, TuneControls, TuneControlsRequest, TuneControlsResponse
from tauv_util.types import tl, tm
from tauv_util.transforms import quat_to_rpy, twist_body_to_world
from nav_msgs.msg import Odometry as OdometryMsg
from scipy.spatial.transform import Rotation


class Controller:
    def __init__(self):
        self._dt: float = 1.0 / rospy.get_param('~frequency')

        self._pose: Optional[Pose] = None
        self._body_twist: Optional[Twist] = None
        self._hold_pose: Optional[Pose] = None

        self._roll_tunings: np.array = np.array(rospy.get_param('~roll_tunings'))
        self._pitch_tunings: np.array = np.array(rospy.get_param('~pitch_tunings'))
        self._z_tunings: np.array = np.array(rospy.get_param('~z_tunings'))

        self._build_pids()

        self._hold_pose_srv: rospy.Service = rospy.Service('hold_pose', HoldPose, self._handle_hold_pose)
        self._tune_dynamics_srv: rospy.Service = rospy.Service('tune_dynamics', TuneDynamics, self._handle_tune_dynamics)
        self._tune_pids_srv: rospy.Service = rospy.Service('tune_controls', TuneControls, self._handle_tune_controls)

        self._odom_sub: rospy.Subscriber = rospy.Subscriber('odom', OdometryMsg, self._handle_odom)
        self._command_sub: rospy.Subscriber = rospy.Subscriber('controller_cmd', ControllerCmdMsg, self._handle_command)
        self._wrench_pub: rospy.Publisher = rospy.Publisher('wrench', Wrench, queue_size=10)

        self._max_wrench: np.array = np.array(rospy.get_param('~max_wrench'))

        self._m = rospy.get_param('~dynamics/mass')
        self._v = rospy.get_param('~dynamics/volume')
        self._rho = rospy.get_param('~dynamics/water_density')
        self._r_G = np.array(rospy.get_param('~dynamics/center_of_gravity'))
        self._r_B = np.array(rospy.get_param('~dynamics/center_of_buoyancy'))
        self._I = np.array(rospy.get_param('~dynamics/moments'))
        self._D = np.array(rospy.get_param('~dynamics/linear_damping'))
        self._D2 = np.array(rospy.get_param('~dynamics/quadratic_damping'))
        self._Ma = np.array(rospy.get_param('~dynamics/added_mass'))

        self._dyn: Dynamics = Dynamics(
            m=self._m,
            v=self._v,
            rho=self._rho,
            r_G=self._r_G,
            r_B=self._r_B,
            I=self._I,
            D=self._D,
            D2=self._D2,
            Ma=self._Ma,
        )

        self._cmd_acceleration: Optional[np.array] = None

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        if self._pose is None or self._body_twist is None or (self._cmd_acceleration is None and self._hold_pose is None):
            return

        eta = np.concatenate((
            tl(self._pose.position),
            quat_to_rpy(self._pose.orientation)
        ))
        v = np.concatenate((
            tl(self._body_twist.linear),
            tl(self._body_twist.angular),
            # np.flip(tl(self._body_twist.angular))
        ))
        vd = self._get_acceleration()

        tau = self._dyn.compute_tau(eta, v, vd)
        while not np.allclose(np.minimum(np.abs(tau), self._max_wrench), np.abs(tau)):
            tau = 0.75 * tau
        # bounded_tau = np.sign(tau) * np.minimum(np.abs(tau), self._max_wrench)

        wrench: Wrench = Wrench()
        wrench.force = Vector3(tau[0], tau[1], tau[2])
        wrench.torque = Vector3(tau[3], tau[4], tau[5])
        self._wrench_pub.publish(wrench)

    def _get_acceleration(self) -> np.array:
        efforts = self._get_efforts()

        if self._hold_pose is not None:
            R = Rotation.from_quat(tl(self._pose.orientation)).inv()

            world_acceleration = np.array([0.0, 0.0, efforts[2]])
            body_acceleration = R.apply(world_acceleration)

            vd = np.concatenate((
                body_acceleration,
                np.array([efforts[0], efforts[1], 0.0])
            ))
        else:
            vd = np.array([
                self._cmd_acceleration[0],
                self._cmd_acceleration[1],
                self._cmd_acceleration[2],
                efforts[0],
                efforts[1],
                self._cmd_acceleration[5],
            ])

        return vd

    def _get_efforts(self):
        if self._pose is None or self._body_twist is None:
            return np.array([0.0, 0.0, 0.0])

        targets = np.array([0.0, 0.0, 0.0])

        if self._hold_pose is not None:
            rot = quat_to_rpy(self._hold_pose.orientation)
            targets = np.array([
                rot[0],
                rot[1],
                self._hold_pose.position.z,
            ])

        rot = quat_to_rpy(self._pose.orientation)
        pos = np.array([
            rot[0],
            rot[1],
            self._pose.position.z,
        ])

        err = targets - pos
        err = (err + pi) % (2 * pi) - pi

        efforts = np.array([
            self._roll_pid(err[0]),
            self._pitch_pid(err[1]),
            self._z_pid(err[2]),
        ])

        return efforts

    def _handle_odom(self, msg: OdometryMsg):
        self._pose = msg.pose.pose
        self._body_twist = msg.twist.twist

    def _handle_command(self, msg: ControllerCmdMsg):
        self._cmd_acceleration = np.array([
            msg.a_x,
            msg.a_y,
            msg.a_z,
            msg.a_roll,
            msg.a_pitch,
            msg.a_yaw,
        ])

    def _handle_hold_pose(self, req: HoldPoseRequest) -> HoldPoseResponse:
        self._hold_pose = req.pose if req.enable else None
        return HoldPoseResponse(True)

    def _handle_tune_dynamics(self, req: TuneDynamicsRequest) -> TuneDynamicsResponse:
        if req.tunings.update_mass:
            self._m = req.tunings.mass

        if req.tunings.update_volume:
            self._v = req.tunings.volume

        if req.tunings.update_water_density:
            self._rho = req.tunings.water_density

        if req.tunings.update_center_of_gravity:
            self._r_G = req.tunings.center_of_gravity

        if req.tunings.update_center_of_buoyancy:
            self._r_B = req.tunings.center_of_buoyancy

        if req.tunings.update_moments:
            self._I = req.tunings.moments

        if req.tunings.update_linear_damping:
            self._D = req.tunings.linear_damping

        if req.tunings.update_quadratic_damping:
            self._D2 = req.tunings.quadratic_damping

        if req.tunings.update_added_mass:
            self._Ma = req.tunings.added_mass

        self._dyn: Dynamics = Dynamics(
            m = self._m,
            v = self._v,
            rho = self._rho,
            r_G = self._r_G,
            r_B = self._r_B,
            I = self._I,
            D = self._D,
            D2 = self._D2,
            Ma = self._Ma,
        )
        return TuneDynamicsResponse(True)

    def _handle_tune_controls(self, req: TuneControlsRequest) -> TuneControlsResponse:
        if req.tunings.update_roll:
            self._roll_tunings = req.tunings.roll_tunings
        if req.tunings.update_pitch:
            self._pitch_tunings = req.tunings.pitch_tunings
        if req.tunings.update_z:
            self._z_tunings = req.tunings.z_tunings
        self._build_pids()
        return TuneControlsResponse(True)

    def _build_pids(self):
        def pi_clip(angle):
            if angle > 0:
                if angle > pi:
                    return angle - 2*pi
            else:
                if angle < -pi:
                    return angle + 2*pi
            return angle

        self._roll_pid: PID = PID(
            Kp=self._roll_tunings[0],
            Ki=self._roll_tunings[1],
            Kd=self._roll_tunings[2],
            error_map=pi_clip,
            proportional_on_measurement=False,
            sample_time=0.05,
        )
        self._pitch_pid: PID = PID(
            Kp=self._pitch_tunings[0],
            Ki=self._pitch_tunings[1],
            Kd=self._pitch_tunings[2],
            error_map=pi_clip,
            proportional_on_measurement=False,
            sample_time=0.05,
        )
        self._z_pid: PID = PID(
            Kp=self._z_tunings[0],
            Ki=self._z_tunings[1],
            Kd=self._z_tunings[2],
            proportional_on_measurement=False,
            sample_time=0.05,
        )


def main():
    rospy.init_node('controller')
    c = Controller()
    c.start()