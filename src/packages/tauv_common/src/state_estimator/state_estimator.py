import rospy
import numpy as np
from threading import Lock
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Union, Optional, Callable

from geometry_msgs.msg import WrenchStamped, Vector3
from tauv_msgs.msg import XsensImuData, FluidDepth, NavigationState
from tauv_util.types import tl, tm
from tauv_util.transforms import euler_velocity_to_axis_velocity

from state_estimator.ekf import StateIndex, EKF
from dynamics_parameter_estimator.dynamics import get_acceleration
from tauv_util.transforms import quat_to_rotm, rpy_to_quat


@dataclass(order=True)
class StampedMsg:
    time: float
    msg: Union[XsensImuData, FluidDepth, WrenchStamped] = field(compare=False)


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

        self._queue: PriorityQueue[StampedMsg] = PriorityQueue()
        
        self._navigation_state_pub: rospy.Publisher = rospy.Publisher('gnc/navigation_state',
                                                                      NavigationState, queue_size=10)

        self._imu_sub: rospy.Subscriber = rospy.Subscriber('vehicle/xsens_imu/raw_data',
                                                           XsensImuData, self._handle_msg)
        self._depth_sub: rospy.Subscriber = rospy.Subscriber('vehicle/arduino/depth', FluidDepth, self._handle_msg)
        self._wrench_sub: rospy.Subscriber = rospy.Subscriber('gnc/target_wrench', WrenchStamped, self._handle_msg)

        self._measured_acceleration_pub = rospy.Publisher('gnc/measurement', Vector3, queue_size=10)

        self._free_acceleration_pub = rospy.Publisher('gnc/imu_free_acceleration', Vector3,
                                                      queue_size=10)

        self._ekf: EKF = EKF(process_covariance=self._process_covariance)

        self._lock.release()

    def _sweep_queue(self, time: float, func: Optional[Callable[[Union[XsensImuData, FluidDepth, WrenchStamped]], None]] = None):
        while True:
            if self._queue.empty():
                break

            msg = self._queue.get_nowait()
            if msg.time >= time:
                self._queue.put_nowait(msg)
                break

            if func is not None:
                func(msg.msg)

    def _update(self, timer_event):
        self._lock.acquire()

        current_time = rospy.Time.now()
        horizon_time = current_time - rospy.Duration.from_sec(0.1)

        # Wipe all messages before ekf time
        ekf_time = self._ekf.get_time()
        if ekf_time is not None:
            self._sweep_queue(ekf_time, lambda msg: print(f"dropping {type(msg)}"))

        # Apply all messages before horizon time
        self._sweep_queue(horizon_time.to_sec(), self._apply_msg)

        state = self._ekf.get_state(horizon_time.to_sec())
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

    def _handle_msg(self, msg: Union[XsensImuData, FluidDepth, WrenchStamped]):
        self._lock.acquire()

        stamped_msg = StampedMsg(msg.header.stamp.to_sec(), msg)
        self._queue.put(stamped_msg)

        self._lock.release()

    def _apply_msg(self, msg: Union[XsensImuData, FluidDepth, WrenchStamped]):
        if isinstance(msg, XsensImuData):
            self._apply_imu(msg)
        elif isinstance(msg, FluidDepth):
            self._apply_depth(msg)
        elif isinstance(msg, WrenchStamped):
            self._apply_wrench(msg)

    def _apply_imu(self, msg: XsensImuData):
        fields = [StateIndex.YAW, StateIndex.PITCH, StateIndex.ROLL,
                  StateIndex.VYAW, StateIndex.VPITCH, StateIndex.VROLL,
                  StateIndex.AX, StateIndex.AY, StateIndex.AZ]

        rotm = quat_to_rotm(tl(rpy_to_quat(tl(msg.orientation))))
        world_gravity = np.array([0.0, 0.0, -9.81])
        body_gravity = np.linalg.inv(rotm) @ world_gravity

        linear_acceleration = tl(msg.linear_acceleration)
        free_acceleration = linear_acceleration - body_gravity

        self._free_acceleration_pub.publish(tm(free_acceleration, Vector3))

        time = msg.header.stamp
        # time = rospy.Time.now()
        measurement = np.concatenate((
            np.flip(tl(msg.orientation)),
            np.flip(tl(msg.rate_of_turn)),
            free_acceleration
        ))

        self._ekf.handle_measurement(time.to_sec(), fields, measurement, self._imu_covariance)

    def _apply_depth(self, msg: FluidDepth):
        fields = [StateIndex.Z]

        time = msg.header.stamp
        # time = rospy.Time.now()

        measurement = np.array([msg.depth])

        self._ekf.handle_measurement(time.to_sec(), fields, measurement, self._depth_covariance)

    def _apply_wrench(self, msg: WrenchStamped):
        fields = [StateIndex.AX, StateIndex.AY, StateIndex.AZ]

        time = msg.header.stamp
        # time = rospy.Time.now()

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

    def _load_config(self):
        self._process_covariance = np.diag([
            1e-1, 1e-1, 1e-1,
            1e-1, 1e-1, 1e-1,
            1e-5, 1e-5, 1e-5,
            1e-1, 1e-1, 1e-1,
            1e-1, 1e-1, 1e-1,
        ])
        self._imu_covariance = np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-3, 1e-3, 1e-3])
        self._depth_covariance = np.array([1e-9])
        self._wrench_covariance = np.array([1e-4, 1e-4, 1e-4])
        self._dynamics_parameters = np.concatenate((
            (
                rospy.get_param('~dynamics/mass'),
                rospy.get_param('~dynamics/volume'),
            ),
            rospy.get_param('~dynamics/center_of_gravity'),
            rospy.get_param('~dynamics/center_of_buoyancy'),
            np.array(rospy.get_param('~dynamics/moments'))[[0, 3, 4, 1, 5, 2]],
            rospy.get_param('~dynamics/linear_damping'),
            rospy.get_param('~dynamics/quadratic_damping'),
            rospy.get_param('~dynamics/added_mass'),
        ))

        print("State estimator config loaded")


def main():
    rospy.init_node('state_estimator')
    n = StateEstimator()
    n.start()
