import rospy
import numpy as np
from typing import Optional
from math import pi
from simple_pid import PID

from dynamics.dynamics import Dynamics
from geometry_msgs.msg import Pose, Twist, Wrench, Vector3, Quaternion
from tauv_msgs.msg import ControllerCmd as ControllerCmdMsg
from tauv_msgs.msg import ControllerDebug
from tauv_msgs.srv import SetTargetPose, SetTargetPoseRequest, SetTargetPoseResponse, TuneDynamics, TuneDynamicsRequest, TuneDynamicsResponse, TuneControls, TuneControlsRequest, TuneControlsResponse
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from std_msgs.msg import Bool
from tauv_util.types import tl, tm
from tauv_util.transforms import quat_to_rpy, twist_body_to_world
from nav_msgs.msg import Odometry as OdometryMsg
from scipy.spatial.transform import Rotation

from tauv_alarms import Alarm, AlarmClient


class Controller:
    def __init__(self):
        self._ac = AlarmClient()

        self._dt: float = 1.0 / rospy.get_param('~frequency')

        self._is_active = True

        self._pose: Optional[Pose] = None
        self._body_twist: Optional[Twist] = None
        self._target_pose: Pose = Pose()
        self._target_pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        self._hold_z: bool = False

        self._roll_tunings: np.array = np.array(rospy.get_param('~roll_tunings'))
        self._pitch_tunings: np.array = np.array(rospy.get_param('~pitch_tunings'))
        self._z_tunings: np.array = np.array(rospy.get_param('~z_tunings'))

        self._roll_limits: np.array = np.array(rospy.get_param('~roll_limits'))
        self._pitch_limits: np.array = np.array(rospy.get_param('~pitch_limits'))
        self._z_limits: np.array = np.array(rospy.get_param('~z_limits'))

        self._build_pids()

        self._target_pose_srv: rospy.Service = rospy.Service('set_target_pose', SetTargetPose, self._handle_target_pose)
        self._hold_z_srv: rospy.Service = rospy.Service('set_hold_z', SetBool, self._handle_hold_z)
        self._tune_dynamics_srv: rospy.Service = rospy.Service('tune_dynamics', TuneDynamics, self._handle_tune_dynamics)
        self._tune_pids_srv: rospy.Service = rospy.Service('tune_controls', TuneControls, self._handle_tune_controls)

        self._odom_sub: rospy.Subscriber = rospy.Subscriber('odom', OdometryMsg, self._handle_odom)
        self._command_sub: rospy.Subscriber = rospy.Subscriber('controller_cmd', ControllerCmdMsg, self._handle_command)
        self._wrench_pub: rospy.Publisher = rospy.Publisher('wrench', Wrench, queue_size=10)
        self._debug_pub = rospy.Publisher('debug', ControllerDebug, queue_size=10)
        self._active_sub = rospy.Subscriber('active', Bool, self._handle_active)

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
        # print(self._pose, self._body_twist, self._cmd_acceleration)
        if self._pose is None or self._body_twist is None or self._cmd_acceleration is None:
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

        # while not np.allclose(np.minimum(np.abs(tau), self._max_wrench), np.abs(tau)):
        #     tau = 0.75 * tau
        #     # TODO FIX THIS
        #     # tau = self._dyn.compute_tau(eta, v, vd)
        tau = np.sign(tau) * np.minimum(np.abs(tau), self._max_wrench)

        wrench: Wrench = Wrench()
        wrench.force = Vector3(tau[0], tau[1], tau[2])
        wrench.torque = Vector3(tau[3], tau[4], tau[5])
        self._wrench_pub.publish(wrench)

        self._ac.clear(Alarm.CONTROLLER_NOT_INITIALIZED)

    def _get_acceleration(self) -> np.array:
        efforts = self._get_efforts()

        if self._hold_z:
            R = Rotation.from_quat(tl(self._pose.orientation)).inv()

            world_acceleration = np.array([0.0, 0.0, efforts[2]])
            body_acceleration = R.apply(world_acceleration)

            vd = np.array([
                body_acceleration[0] + self._cmd_acceleration[0],
                body_acceleration[1] + self._cmd_acceleration[1],
                body_acceleration[2] + self._cmd_acceleration[2],
                efforts[0],
                efforts[1],
                self._cmd_acceleration[5],
            ])
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
        if self._pose is None or self._body_twist is None or not self._is_active:
            return np.array([0.0, 0.0, 0.0])

        target_rpy = quat_to_rpy(self._target_pose.orientation)
        target = np.array([
            target_rpy[0],
            target_rpy[1],
            self._target_pose.position.z,
        ])

        current_rpy = quat_to_rpy(self._pose.orientation)
        current = np.array([
            current_rpy[0],
            current_rpy[1],
            self._pose.position.z
        ])

        err = current - target
        err[0:2] = (err[0:2] + pi) % (2 * pi) - pi

        efforts = np.array([
            self._roll_pid(err[0]),
            self._pitch_pid(err[1]),
            self._z_pid(err[2]),
        ])

        cd = ControllerDebug()
        cd.ang_err_x = err[0]
        cd.ang_err_y = err[1]
        cd.ang_err_z = err[2]
        self._debug_pub.publish(cd)

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

    def _handle_target_pose(self, req: SetTargetPoseRequest) -> SetTargetPoseResponse:
        self._target_pose = req.pose
        return SetTargetPoseResponse(True)

    def _handle_hold_z(self, req: SetBoolRequest) -> SetBoolResponse:
        self._hold_z = req.data
        return SetBoolResponse(True, "")

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
        if req.tunings.update_roll_limits:
            self._roll_limits = req.tunings.roll_limits
        if req.tunings.update_pitch:
            self._pitch_tunings = req.tunings.pitch_tunings
        if req.tunings.update_pitch_limits:
            self._pitch_limits = req.tunings.pitch_limits
        if req.tunings.update_z:
            self._z_tunings = req.tunings.z_tunings
        if req.tunings.update_z_limits:
            self._z_limits = req.tunings.z_limits
        self._build_pids()
        return TuneControlsResponse(True)

    def _build_pids(self):
        def pi_clip(angle):
            if angle > pi:
                return angle - 2*pi
            if angle < -pi:
                return angle + 2*pi
            return angle

        self._roll_pid: PID = PID(
            Kp=self._roll_tunings[0],
            Ki=self._roll_tunings[1],
            Kd=self._roll_tunings[2],
            error_map=pi_clip,
            output_limits=self._roll_limits,
            proportional_on_measurement=False,
            sample_time=0.02,
        )
        self._pitch_pid: PID = PID(
            Kp=self._pitch_tunings[0],
            Ki=self._pitch_tunings[1],
            Kd=self._pitch_tunings[2],
            error_map=pi_clip,
            output_limits=self._pitch_limits,
            proportional_on_measurement=False,
            sample_time=0.02,
        )
        self._z_pid: PID = PID(
            Kp=self._z_tunings[0],
            Ki=self._z_tunings[1],
            Kd=self._z_tunings[2],
            output_limits=self._z_limits,
            proportional_on_measurement=False,
            # sample_time=0.,
        )

    def _handle_active(self, msg: Bool):
        self._is_active = msg.data


def main():
    rospy.init_node('controller')
    c = Controller()
    c.start()