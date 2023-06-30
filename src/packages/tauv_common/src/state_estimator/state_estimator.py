import rospy
import numpy as np
from threading import Lock

from geometry_msgs.msg import WrenchStamped, Vector3
from tauv_msgs.msg import XsensImuData, FluidDepth, NavigationState
from tauv_util.types import tl, tm

from state_estimator.ekf import StateIndex, EKF
from dynamics_parameter_estimator.dynamics import get_acceleration


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

        self._load_config()

        self._navigation_state_pub: rospy.Publisher = rospy.Publisher('gnc/estimated_state', NavigationState, queue_size=10)

        self._imu_sub: rospy.Subscriber = rospy.Subscriber('vehicle/xsens_imu/data_raw', XsensImuData, self._handle_imu)
        self._depth_sub: rospy.Subscriber = rospy.Subscriber('vehicle/arduino/depth', FluidDepth, self._handle_depth)
        self._wrench_sub: rospy.Subscriber = rospy.Subscriber('gnc/target_wrench', WrenchStamped, self._handle_wrench)

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
        nav_state.linear_velocity = tm(state[[StateIndex.VX, StateIndex.VY, StateIndex.VZ]], Vector3)
        nav_state.linear_acceleration = tm(state[[StateIndex.AX, StateIndex.AY, StateIndex.AZ]], Vector3)
        nav_state.orientation = tm(state[[StateIndex.ROLL, StateIndex.PITCH, StateIndex.YAW]], Vector3)
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

        time = msg.header.stamp

        measurement = np.concatenate((
            np.flip(tl(msg.orientation)),
            np.flip(tl(msg.rate_of_turn)),
            tl(msg.free_acceleration)
        )) + np.random.normal(scale=0.01, size=9)

        self._ekf.handle_measurement(time.to_sec(), fields, measurement, self._imu_covariance)

        self._lock.release()

    def _handle_depth(self, msg: FluidDepth):
        self._lock.acquire()

        fields = [StateIndex.Z]

        time = msg.header.stamp

        measurement = np.array([msg.depth])

        self._ekf.handle_measurement(time.to_sec(), fields, measurement, self._depth_covariance)

        self._lock.release()

    def _handle_wrench(self, msg: WrenchStamped):
        self._lock.acquire()

        fields = [StateIndex.AX, StateIndex.AY, StateIndex.AZ]

        time = msg.header.stamp

        state = self._ekf.get_state(time.to_sec())
        if state is None:
            self._lock.release()
            return

        dynamics_state = state[[StateIndex.ROLL, StateIndex.PITCH, StateIndex.YAW,
                                StateIndex.VX, StateIndex.VY, StateIndex.VZ,
                                StateIndex.VROLL, StateIndex.VPITCH, StateIndex.VYAW]]

        wrench = np.concatenate((
            tl(msg.wrench.force),
            tl(msg.wrench.torque),
        ))

        measurement = get_acceleration(self._dynamics_parameters, dynamics_state, wrench)[0:3]

        self._ekf.handle_measurement(time.to_sec(), fields, measurement, self._wrench_covariance)

        self._lock.release()

    def _load_config(self):
        self._process_covariance = 1e-12 * np.ones((15,), dtype=np.float32)
        self._imu_covariance = 1e-4 * np.ones((9,), dtype=np.float32)
        self._depth_covariance = 1e-4 * np.ones((1,), dtype=np.float32)
        self._wrench_covariance = 1e-4 * np.ones((3,), dtype=np.float32)

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
        print(self._dynamics_parameters)


def main():
    rospy.init_node('state_estimator')
    n = StateEstimator()
    n.start()
