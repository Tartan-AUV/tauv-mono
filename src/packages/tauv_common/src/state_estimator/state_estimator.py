import rospy
import numpy as np
from threading import Lock

from geometry_msgs.msg import WrenchStamped, Vector3
from tauv_msgs.msg import XsensImuData, FluidDepth, NavigationState
from tauv_util.types import tl, tm
from tauv_util.transforms import euler_velocity_to_axis_velocity

from state_estimator.ekf import StateIndex, EKF
from dynamics_parameter_estimator.dynamics import get_acceleration
from tauv_util.transforms import quat_to_rotm, rpy_to_quat
from geometry_msgs.msg import Quaternion
from tf.transformations import *

class StateEstimator:

    def __init__(self):
        self._dt: float = 0.02

        self._lock: Lock = Lock()
        self._lock.acquire()

        self._process_covariance: np.array = np.array([])
        self._imu_covariance: np.array = np.array([])
        self._depth_covariance: np.array = np.array([])
        self._wrench_covariance: np.array = np.array([])
        self._dynamics_parameters: np.array = np.array([])

        self._imu_orientation_rotm = numpy.array([[-1.0, 0.0, 0.0],
                                                  [0.0, 1.0, 0.0],
                                                  [0.0, 0.0, 1.0]])
        self._imu_transform_quat = np.array([0.0, 1.0, 0.0, 0.0])

        self._load_config()

        self._navigation_state_pub: rospy.Publisher = rospy.Publisher('gnc/estimated_navigation_state',
                                                                      NavigationState, queue_size=10)

        self._imu_sub: rospy.Subscriber = rospy.Subscriber('vehicle/xsens_imu/raw_data',
                                                           XsensImuData, self._handle_imu)
        self._depth_sub: rospy.Subscriber = rospy.Subscriber('vehicle/arduino/depth', FluidDepth, self._handle_depth)
        self._wrench_sub: rospy.Subscriber = rospy.Subscriber('gnc/target_wrench', WrenchStamped, self._handle_wrench)

        self._measured_acceleration_pub = rospy.Publisher('gnc/measurement', Vector3, queue_size=10)

        self._free_acceleration_pub = rospy.Publisher('gnc/imu_free_acceleration', Vector3,
                                                      queue_size=10)

        self._ekf: EKF = EKF(process_covariance=self._process_covariance)

        self._lock.release()

    def _update(self, timer_event):
        self._lock.acquire()

        current_time = rospy.Time.now()

        state = self._ekf.get_state(current_time.to_sec())
        if state is None:
            self._lock.release()
            return

        nav_state = NavigationState()
        nav_state.header.stamp = current_time
        nav_state.position = tm(state[[StateIndex.X, StateIndex.Y, StateIndex.Z]], Vector3)
        nav_state.linear_velocity = tm(state[[StateIndex.VX, StateIndex.VY, StateIndex.VZ]],
                                       Vector3)
        # nav_state.linear_velocity.z = 0.0
        nav_state.linear_acceleration = tm(state[[StateIndex.AX, StateIndex.AY, StateIndex.AZ]], Vector3)
        # nav_state.linear_acceleration.z = 0.0
        nav_state.orientation = tm(state[[StateIndex.ROLL, StateIndex.PITCH, StateIndex.YAW]],
                                    Vector3)
        nav_state.euler_velocity = tm(state[[StateIndex.VROLL, StateIndex.VPITCH, StateIndex.VYAW]], Vector3)
        self._navigation_state_pub.publish(nav_state)

        self._lock.release()

    def start(self):
        rospy.Timer(rospy.Duration(self._dt), self._update)
        rospy.spin()

    def _handle_imu(self, msg: XsensImuData):
        self._lock.acquire()

        fields = [StateIndex.YAW, StateIndex.PITCH, StateIndex.ROLL,
                  StateIndex.VYAW, StateIndex.VPITCH, StateIndex.VROLL,
                  StateIndex.AX, StateIndex.AY, StateIndex.AZ]

        rotm = quat_to_rotm(tl(rpy_to_quat(tl(msg.orientation))))
        world_gravity = np.array([0.0, 0.0, 9.81])
        body_gravity = np.linalg.inv(rotm) @ world_gravity

        linear_acceleration = tl(msg.linear_acceleration)
        free_acceleration = linear_acceleration - body_gravity

        #imu_q = tl(rpy_to_quat(tl(msg.orientation)))
        #imu_q[3] = -imu_q[3]
        #imu_gravity = self._get_gravity_vector(imu_q)
        #imu_free_acceleration = tl(msg.linear_acceleration) + imu_gravity
        # print(f'{tl(msg.linear_acceleration)=} {imu_gravity=}')
        #free_acceleration = quat_to_rotm(self._imu_transform_quat) @ imu_free_acceleration

        self._free_acceleration_pub.publish(tm(free_acceleration, Vector3))

        # time = msg.header.stamp
        time = rospy.Time.now()
        measurement = np.concatenate((
            np.flip(tl(msg.orientation)),
            np.flip(tl(msg.rate_of_turn)),
            free_acceleration
        ))
        # Do not want free acceleration here

        self._ekf.handle_measurement(time.to_sec(), fields, measurement, self._imu_covariance)

        self._lock.release()

    def _handle_depth(self, msg: FluidDepth):
        self._lock.acquire()

        fields = [StateIndex.Z]

        # time = msg.header.stamp
        time = rospy.Time.now()

        measurement = np.array([msg.depth])

        self._ekf.handle_measurement(time.to_sec(), fields, measurement, self._depth_covariance)

        self._lock.release()

    def _handle_wrench(self, msg: WrenchStamped):
        self._lock.acquire()

        fields = [StateIndex.AX, StateIndex.AY, StateIndex.AZ]

        # time = msg.header.stamp
        time = rospy.Time.now()

        state = self._ekf.get_state(time.to_sec())
        if state is None:
            self._lock.release()
            return

        orientation = state[[StateIndex.ROLL, StateIndex.PITCH, StateIndex.YAW]]
        velocity = state[[StateIndex.VX, StateIndex.VY, StateIndex.VZ]]
        euler_velocity = state[[StateIndex.VROLL, StateIndex.VPITCH, StateIndex.VYAW]]
        axis_velocity = euler_velocity_to_axis_velocity(orientation, euler_velocity)

        dynamics_state = np.concatenate((
            orientation,
            velocity,
            axis_velocity
        ))

        # dynamics_state[3:6] = 0.0

        wrench = np.concatenate((
            tl(msg.wrench.force),
            tl(msg.wrench.torque),
        ))

        measurement = get_acceleration(self._dynamics_parameters, dynamics_state, wrench)[0:3]

        a = Vector3()
        a.x = measurement[0]
        a.y = measurement[1]
        a.z = measurement[2]
        self._measured_acceleration_pub.publish(a)

        self._ekf.handle_measurement(time.to_sec(), fields, measurement, self._wrench_covariance)

        self._lock.release()

    def _load_config(self):
        self._process_covariance = np.diag([
            1e-9, 1e-9, 1e-9,
            1e-9, 1e-9, 1e-9,
            1e-9, 1e-9, 1e-9,
            1e-9, 1e-9, 1e-9,
            1e-9, 1e-9, 1e-9,
        ])
        self._imu_covariance = np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-6, 1e-6, 1e-6])
        self._depth_covariance = np.array([1e-9])
        self._wrench_covariance = np.array([1e-9, 1e-9, 1e-9])
        self._dynamics_parameters = np.concatenate((
            (
                rospy.get_param('~dynamics/mass'),
                rospy.get_param('~dynamics/volume'),
            ),
            rospy.get_param('~dynamics/center_of_gravity'),
            rospy.get_param('~dynamics/center_of_buoyancy'),
            rospy.get_param('~dynamics/moments'),
            rospy.get_param('~dynamics/linear_damping'),
            rospy.get_param('~dynamics/quadratic_damping'),
            rospy.get_param('~dynamics/added_mass'),
        ))

        print("State estimator config loaded")


def main():
    rospy.init_node('state_estimator')
    n = StateEstimator()
    n.start()
