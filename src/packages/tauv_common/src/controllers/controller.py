import rospy
import numpy as np
from typing import Optional
from math import pi
from controllers.tpid import PID

from dynamics.dynamics import Dynamics
from geometry_msgs.msg import Twist, Wrench, Vector3, Quaternion
from geometry_msgs.msg import Pose as GeoPose
from tauv_msgs.msg import ControllerCmd as ControllerCmdMsg
from tauv_msgs.msg import ControllerDebug, Pose
from tauv_msgs.srv import SetTargetPose, SetTargetPoseRequest, SetTargetPoseResponse, TuneDynamics, TuneDynamicsRequest, TuneDynamicsResponse, TuneControls, TuneControlsRequest, TuneControlsResponse
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from std_msgs.msg import Bool
from tauv_util.types import tl, tm
from tauv_util.transforms import quat_to_rpy, twist_body_to_world
from nav_msgs.msg import Odometry as OdometryMsg
from scipy.spatial.transform import Rotation
from tauv_msgs.msg import TrajPoint

from tauv_alarms import Alarm, AlarmClient

import tf2_ros
import tf

class Controller:
    def __init__(self):
        self._ac = AlarmClient()

        self._dt: float = 1.0 / rospy.get_param('~frequency')

        self._is_active = False

        self._pose: Optional[Pose] = None
        self._target_pose: GeoPose = GeoPose()
        self._target_pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        self._hold_x: bool = False
        self._hold_y: bool = False
        self._hold_z: bool = False
        self._hold_yaw: bool = False

        self._roll_tunings: np.array = np.array(rospy.get_param('~roll_tunings'))
        self._pitch_tunings: np.array = np.array(rospy.get_param('~pitch_tunings'))
        self._yaw_tunings: np.array = np.array(rospy.get_param('~yaw_tunings'))
        self._x_tunings: np.array = np.array(rospy.get_param('~x_tunings'))
        self._y_tunings: np.array = np.array(rospy.get_param('~y_tunings'))
        self._z_tunings: np.array = np.array(rospy.get_param('~z_tunings'))

        self._roll_limits: np.array = np.array(rospy.get_param('~roll_limits'))
        self._pitch_limits: np.array = np.array(rospy.get_param('~pitch_limits'))
        self._yaw_limits: np.array = np.array(rospy.get_param('~yaw_limits'))
        self._x_limits: np.array = np.array(rospy.get_param('~x_limits'))
        self._y_limits: np.array = np.array(rospy.get_param('~y_limits'))
        self._z_limits: np.array = np.array(rospy.get_param('~z_limits'))

        self._tau_x: np.array = np.array(rospy.get_param('~tau_x'))
        self._tau_y: np.array = np.array(rospy.get_param('~tau_y'))
        self._tau_z: np.array = np.array(rospy.get_param('~tau_z'))
        self._tau_roll: np.array = np.array(rospy.get_param('~tau_roll'))
        self._tau_pitch: np.array = np.array(rospy.get_param('~tau_pitch'))
        self._tau_yaw: np.array = np.array(rospy.get_param('~tau_yaw'))
        
        self.goal_pub = rospy.Publisher('goal', GeoPose, queue_size=10)

        self._build_pids()

        self._target_pose_srv: rospy.Service = rospy.Service('set_target_pose', SetTargetPose, self._handle_target_pose)
        self._hold_z_srv: rospy.Service = rospy.Service('set_hold_z', SetBool, self._handle_hold_z)
        self._hold_z_srv: rospy.Service = rospy.Service('set_hold_yaw', SetBool, self._handle_hold_yaw)
        self._hold_z_srv: rospy.Service = rospy.Service('set_hold_xy', SetBool, self._handle_hold_xy)
        self._tune_dynamics_srv: rospy.Service = rospy.Service('tune_dynamics', TuneDynamics, self._handle_tune_dynamics)
        self._tune_pids_srv: rospy.Service = rospy.Service('tune_controls', TuneControls, self._handle_tune_controls)

        self._odom_sub: rospy.Subscriber = rospy.Subscriber('pose', Pose, self._handle_pose)
        self._command_sub: rospy.Subscriber = rospy.Subscriber('controller_cmd', ControllerCmdMsg, self._handle_command)
        self._wrench_pub: rospy.Publisher = rospy.Publisher('wrench', Wrench, queue_size=10)
        self._debug_pub = rospy.Publisher('debug', ControllerDebug, queue_size=10)
        self._active_sub = rospy.Subscriber('active', Bool, self._handle_active)
        self._trajpoint_sub = rospy.Subscriber('traj_target', TrajPoint, self._handle_target_trajpoint)

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
        if self._pose is None or self._cmd_acceleration is None:
            return

        eta = np.concatenate((
            tl(self._pose.position),
            tl(self._pose.orientation)
        ))
        v = np.concatenate((
            tl(self._pose.velocity),
            tl(self._pose.angular_velocity),
            # np.flip(tl(self._body_twist.angular))
        ))
        v_feedforward, vd = self._get_inputs()

        tau = self._dyn.compute_tau(eta, v + v_feedforward, vd)

        # while not np.allclose(np.minimum(np.abs(tau), self._max_wrench), np.abs(tau)):
        #     tau = 0.75 * tau
        #     # TODO FIX THIS
        #     # tau = self._dyn.compute_tau(eta, v, vd)
        # tau = np.sign(tau) * np.minimum(np.abs(tau), self._max_wrench)

        wrench: Wrench = Wrench()
        wrench.force = Vector3(tau[0], tau[1], tau[2])
        wrench.torque = Vector3(tau[3], tau[4], tau[5])
        self._wrench_pub.publish(wrench)

        self._ac.clear(Alarm.CONTROLLER_NOT_INITIALIZED)

    def _get_inputs(self) -> np.array:
        efforts = self._get_efforts()

        world_velocity = tl(self._target_twist.linear)
        yaw_velocity = tl(self._target_twist.angular)[2]

        R = Rotation.from_euler('ZYX', np.flip(tl(self._pose.orientation))).inv()

        if not self._hold_x:
            world_velocity[0] = 0
        if not self._hold_y:
            world_velocity[1] = 0
        if not self._hold_z:
            world_velocity[2] = 0
        if not self._hold_yaw:
            yaw_velocity = 0

        world_acceleration = np.array([0.0, 0.0, 0.0])

        if self._hold_x:
            world_acceleration[0] = efforts[3]
        if self._hold_y:
            world_acceleration[1] = efforts[4]
        if self._hold_z:
            world_acceleration[2] = efforts[5]
        yaw_effort = 0

        if self._hold_yaw:
            yaw_effort = efforts[2]

        body_velocity = R.apply(world_velocity)
        body_acceleration = R.apply(world_acceleration)

        v = np.array([
            body_velocity[0],
            body_velocity[1],
            body_velocity[2],
            0,
            0,
            yaw_velocity
        ])

        vd = np.array([
            body_acceleration[0] + self._cmd_acceleration[0],
            body_acceleration[1] + self._cmd_acceleration[1],
            body_acceleration[2] + self._cmd_acceleration[2],
            efforts[0],
            efforts[1],
            self._cmd_acceleration[5] + yaw_effort,
        ])

        return v, vd

    def _get_efforts(self):
        cd = ControllerDebug()

        target_rpy = quat_to_rpy(self._target_pose.orientation)
        target = np.array([
            target_rpy[0],
            target_rpy[1],
            target_rpy[2],
            self._target_pose.position.x,
            self._target_pose.position.y,
            self._target_pose.position.z,
        ])

        self.goal_pub.publish(self._target_pose)

        # current_rpy = quat_to_rpy(self._pose.orientation)
        current = np.array([
            self._pose.orientation.x,
            self._pose.orientation.y,
            self._pose.orientation.z,
            self._pose.position.x,
            self._pose.position.y,
            self._pose.position.z
        ])

        err = current - target
        err[0:3] = (err[0:3] + pi) % (2 * pi) - pi

        cd.ang_x = current[0]
        cd.ang_y = current[1]
        cd.ang_z = current[2]
        cd.x = current[3]
        cd.y = current[4]
        cd.z = current[5]

        cd.ang_target_x = target[0]
        cd.ang_target_y = target[1]
        cd.ang_target_z = target[2]
        cd.target_x = target[3]
        cd.target_y = target[4]
        cd.target_z = target[5]

        cd.ang_err_x = err[0]
        cd.ang_err_y = err[1]
        cd.ang_err_z = err[2]
        cd.err_x = err[3]
        cd.err_y = err[4]
        cd.err_z = err[5]

        cd.ang_i_x = self._roll_pid._integral
        cd.ang_i_y = self._pitch_pid._integral
        cd.ang_i_z = self._yaw_pid._integral
        cd.i_x = self._x_pid._integral
        cd.i_y = self._y_pid._integral
        cd.i_z = self._z_pid._integral

        cd.ang_d_x = self._roll_pid._derivative
        cd.ang_d_y = self._pitch_pid._derivative
        cd.ang_d_z = self._yaw_pid._derivative
        cd.d_x = self._x_pid._derivative
        cd.d_y = self._y_pid._derivative
        cd.d_z = self._z_pid._derivative

        if self._pose is None or not self._is_active:
            self._debug_pub.publish(cd)
            cd.enable = False
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        cd.enable = True
        efforts = np.array([
            self._roll_pid(err[0], dt=self._dt),
            self._pitch_pid(err[1], dt=self._dt),
            self._yaw_pid(err[2], dt=self._dt),
            self._x_pid(err[3], dt=self._dt),
            self._y_pid(err[4], dt=self._dt),
            self._z_pid(err[5], dt=self._dt),
        ])

        cd.e_ang_x = efforts[0]
        cd.e_ang_y = efforts[1]
        cd.e_ang_z = efforts[2]
        cd.e_x = efforts[3]
        cd.e_y = efforts[4]
        cd.e_z = efforts[5]

        self._debug_pub.publish(cd)

        return efforts

    def _handle_pose(self, msg: Pose):
        self._pose = msg

    def _handle_command(self, msg: ControllerCmdMsg):
        self._cmd_acceleration = np.array([
            msg.a_x,
            msg.a_y,
            msg.a_z,
            msg.a_roll,
            msg.a_pitch,
            msg.a_yaw,
        ])

    def _handle_target_trajpoint(self, msg: TrajPoint):
        self._target_pose = msg.pose
        self._target_twist = msg.twist
        self._hold_x = True
        self._hold_y = True
        self._hold_yaw = True

    def _handle_target_pose(self, req: SetTargetPoseRequest) -> SetTargetPoseResponse:
        self._target_pose = req.pose
        return SetTargetPoseResponse(True)

    def _handle_hold_z(self, req: SetBoolRequest) -> SetBoolResponse:
        self._hold_z = req.data
        return SetBoolResponse(True, "")

    def _handle_hold_xy(self, req: SetBoolRequest) -> SetBoolResponse:
        self._hold_y = req.data
        self._hold_x = req.data
        return SetBoolResponse(True, "")

    def _handle_hold_yaw(self, req: SetBoolRequest) -> SetBoolResponse:
        self._hold_yaw = req.data
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
        if req.tunings.update_pitch:
            self._pitch_tunings = req.tunings.pitch_tunings
        if req.tunings.update_yaw:
            self._yaw_tunings = req.tunings.yaw_tunings
        if req.tunings.update_roll_limits:
            self._roll_limits = req.tunings.roll_limits
        if req.tunings.update_pitch_limits:
            self._pitch_limits = req.tunings.pitch_limits
        if req.tunings.update_yaw_limits:
            self._yaw_limits = req.tunings.yaw_limits
        if req.tunings.update_x:
            self._x_tunings = req.tunings.x_tunings
        if req.tunings.update_y:
            self._y_tunings = req.tunings.y_tunings
        if req.tunings.update_z:
            self._z_tunings = req.tunings.z_tunings
        if req.tunings.update_z_limits:
            self._z_limits = req.tunings.z_limits
        if req.tunings.update_tau:
            self._tau_roll = req.tunings.tau[0]
            self._tau_pitch = req.tunings.tau[1]
            self._tau_yaw = req.tunings.tau[2]
            self._tau_x = req.tunings.tau[3]
            self._tau_y = req.tunings.tau[4]
            self._tau_z = req.tunings.tau[5]
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
            d_alpha=self._dt/self._tau_roll if self._tau_roll > 0 else 1
        )
        self._pitch_pid: PID = PID(
            Kp=self._pitch_tunings[0],
            Ki=self._pitch_tunings[1],
            Kd=self._pitch_tunings[2],
            error_map=pi_clip,
            output_limits=self._pitch_limits,
            proportional_on_measurement=False,
            sample_time=0.02,
            d_alpha=self._dt/self._tau_pitch if self._tau_pitch > 0 else 1
        )
        self._yaw_pid: PID = PID(
            Kp=self._yaw_tunings[0],
            Ki=self._yaw_tunings[1],
            Kd=self._yaw_tunings[2],
            error_map=pi_clip,
            output_limits=self._yaw_limits,
            proportional_on_measurement=False,
            sample_time=0.02,
            d_alpha=self._dt/self._tau_yaw if self._tau_yaw > 0 else 1
        )
        self._x_pid: PID = PID(
            Kp=self._x_tunings[0],
            Ki=self._x_tunings[1],
            Kd=self._x_tunings[2],
            output_limits=self._x_limits,
            proportional_on_measurement=False,
            d_alpha=self._dt/self._tau_x if self._tau_x > 0 else 1
        )
        self._y_pid: PID = PID(
            Kp=self._y_tunings[0],
            Ki=self._y_tunings[1],
            Kd=self._y_tunings[2],
            output_limits=self._y_limits,
            proportional_on_measurement=False,
            d_alpha=self._dt/self._tau_y if self._tau_y > 0 else 1
        )
        self._z_pid: PID = PID(
            Kp=self._z_tunings[0],
            Ki=self._z_tunings[1],
            Kd=self._z_tunings[2],
            output_limits=self._z_limits,
            proportional_on_measurement=False,
            d_alpha=self._dt/self._tau_z if self._tau_z > 0 else 1
        )

    def _handle_active(self, msg: Bool):
        self._is_active = msg.data


def main():
    rospy.init_node('controller')
    c = Controller()
    c.start()