import rospy
import numpy as np
from typing import Optional
from simple_pid import PID
from math import pi

from dynamics.dynamics import Dynamics
from geometry_msgs.msg import Pose, Twist, Wrench, Vector3, Quaternion
from tauv_msgs.msg import ControllerCmd as ControllerCmdMsg
from tauv_msgs.srv import HoldPose, HoldPoseRequest, HoldPoseResponse, TuneDynamics, TuneDynamicsRequest, TuneDynamicsResponse, TunePids, TunePidsRequest, TunePidsResponse
from tauv_util.types import tl, tm
from tauv_util.transforms import quat_to_rpy
from nav_msgs.msg import Odometry as OdometryMsg
from scipy.spatial.transform import Rotation


class Controller:
    def __init__(self):
        self._dt: float = 1.0 / rospy.get_param('~frequency')

        self._pose: Optional[Pose] = None
        self._body_twist: Optional[Twist] = None
        self._hold_pose: Optional[Pose] = None
        self._hold_pose: Optional[Pose] = Pose()
        self._hold_pose.position = Vector3(0.0, 0.0, 0.5)
        self._hold_pose.orientation = Quaternion(1.0, 0.0, 0.0, 1.0)

        self._roll_tunings: np.array = np.array(rospy.get_param('~roll_tunings'))
        self._pitch_tunings: np.array = np.array(rospy.get_param('~pitch_tunings'))
        self._z_tunings: np.array = np.array(rospy.get_param('~z_tunings'))

        self._roll_pid: PID = self._build_pid(self._roll_tunings)
        self._pitch_pid: PID = self._build_pid(self._pitch_tunings)
        self._z_pid: PID = self._build_pid(self._z_tunings)

        self._hold_pose_srv: rospy.Service = rospy.Service('hold_pose', HoldPose, self._handle_hold_pose)
        self._tune_dynamics_srv: rospy.Service = rospy.Service('tune_dynamics', TuneDynamics, self._handle_tune_dynamics)
        self._tune_pids_srv: rospy.Service = rospy.Service('tune_pids', TunePids, self._handle_tune_pids)

        self._odom_sub: rospy.Subscriber = rospy.Subscriber('odom', OdometryMsg, self._handle_odom)
        self._command_sub: rospy.Subscriber = rospy.Subscriber('controller_cmd', ControllerCmdMsg, self._handle_command)
        self._wrench_pub: rospy.Publisher = rospy.Publisher('wrench', Wrench, queue_size=10)

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
            tl(self._body_twist.angular)
        ))
        vd = self._get_acceleration()

        tau = self._dyn.compute_tau(eta, v, vd)
        bounded_tau = np.sign(tau) * np.minimum(np.abs(tau), self._max_wrench)

        wrench: Wrench = Wrench()
        wrench.force = Vector3(bounded_tau[0], bounded_tau[1], bounded_tau[2])
        wrench.torque = Vector3(bounded_tau[3], bounded_tau[4], bounded_tau[5])
        self._wrench_pub.publish(wrench)

    def _get_acceleration(self) -> np.array:
        if self._hold_pose is not None:
            z_error = self._hold_pose.position.z - self._pose.position.z
            z_effort = self._z_pid(-z_error)

            R = Rotation.from_quat(tl(self._pose.orientation)).inv()

            world_acceleration = np.array([0.0, 0.0, z_effort])
            body_acceleration = R.apply(world_acceleration)

            hold_pose_rpy = quat_to_rpy(self._hold_pose.orientation)
            pose_rpy = quat_to_rpy(self._pose.orientation)

            roll_error = self._clamp_angle_error(hold_pose_rpy[0] - pose_rpy[0])
            roll_effort = self._roll_pid(-roll_error)

            pitch_error = self._clamp_angle_error(hold_pose_rpy[1] - pose_rpy[1])
            pitch_effort = self._pitch_pid(-pitch_error)

            vd = np.concatenate((
                body_acceleration + np.array([self._cmd_acceleration[0], self._cmd_acceleration[1], self._cmd_acceleration[2]]),
                np.array([roll_effort, pitch_effort, 0.0])
            ))
        else:
            pose_rpy = quat_to_rpy(self._pose.orientation)

            roll_error = self._clamp_angle_error(-pose_rpy[0])
            roll_effort = self._roll_pid(-roll_error)

            pitch_error = self._clamp_angle_error(-pose_rpy[1])
            pitch_effort = self._pitch_pid(-pitch_error)

            vd = np.array([
                self._cmd_acceleration[0],
                self._cmd_acceleration[1],
                self._cmd_acceleration[2],
                roll_effort,
                pitch_effort,
                self._cmd_acceleration[5]
            ])

        return vd

    def _clamp_angle_error(self, angle: float) -> float:
        return (angle + pi) % (2 * pi) - pi

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
        self._dyn: Dynamics = Dynamics(
            m=req.mass,
            v=req.volume,
            rho=req.water_density,
            r_G=req.center_of_gravity,
            r_B=req.center_of_buoyancy,
            I=req.moments,
            D=req.linear_damping,
            D2=req.quadratic_damping,
            Ma=req.added_mass
        )
        return TuneDynamicsResponse(True)

    def _handle_tune_pids(self, req: TunePidsRequest) -> TunePidsResponse:
        self._roll_pid.tunings = req.roll_tunings
        self._pitch_pid.tunings = req.pitch_tunings
        self._z_pid.tunings = req.z_tunings
        return TunePidsResponse(True)

    def _build_pid(self, tunings: np.array) -> PID:
        pid = PID(
            Kp=tunings[0],
            Ki=tunings[1],
            Kd=tunings[2],
            setpoint=0,
            sample_time=None,
            output_limits=(tunings[3], tunings[4]),
            auto_mode=True,
            proportional_on_measurement=False
        )

        return pid


def main():
    rospy.init_node('controller')
    c = Controller()
    c.start()