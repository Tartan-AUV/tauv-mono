import logging
from logging import Logger

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.subscription import Subscription
from rclpy.publisher import Publisher
from rclpy.time import Time

import numpy as np
from typing import Optional, Tuple, Dict, List, Deque, Any, Union
from dataclasses import asdict, dataclass
from enum import Enum
import math
from collections import deque
import threading
from threading import Thread
import queue
from queue import Queue

from spatialmath import SE3, SO3, UnitQuaternion

# ROS2 messages
from sensor_msgs.msg import Imu as ImuMsg
from nav_msgs.msg import Odometry
from tauv_common.util.geometry import numpify, msgify
from tauv_msgs.msg import Depth as DepthMsg
from tauv_msgs.msg import WaterlinkedDvlFrame as DvlMsg
from tauv_msgs.msg import NavigationState
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3, Point, Pose
from builtin_interfaces.msg import Time as TimeMsg

# TF2
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformException

def stamp_to_nanos(stamp: Any) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec

@dataclass
class EkfControl:
    odom_R_sensor: SO3
    a_S: np.ndarray
    omega_S: np.ndarray

    @staticmethod
    def from_msg(msg: ImuMsg) -> 'EkfControl':
        return EkfControl(
            odom_R_sensor=SO3(numpify(msg.orientation).R),  # type: ignore
            a_S=numpify(msg.linear_acceleration),
            omega_S=numpify(msg.angular_velocity),
        )

    def is_valid(self) -> bool:
        return self.a_S.shape == self.omega_S.shape == (3, 1)

@dataclass
class DvlInput:
    """DVL measurement input"""
    v_dvl_V: np.ndarray  # Velocity in DVL frame
    R: np.ndarray        # Measurement covariance

    @staticmethod
    def from_msg(msg: DvlMsg) -> 'DvlInput':
        return DvlInput(
            v_dvl_V=np.array([msg.vx, msg.vy, msg.vz])[:, np.newaxis],
            R=msg.covariance.reshape(3, 3)
        )


@dataclass
class DepthInput:
    """Depth measurement input"""
    z: np.ndarray          # Depth measurement (1, 1)
    R: np.ndarray          # Measurement variance (1, 1)

    @staticmethod
    def from_msg(msg: DepthMsg) -> 'DepthInput':
        return DepthInput(
            z=np.array([[msg.depth]]),
            R=np.array([[msg.variance]]),
        )


class EkfInput(Enum):
    """Types of EKF inputs"""
    IMU = "imu"
    DVL = "dvl"
    DEPTH = "depth"


class MeasurementType(Enum):
    """Types of measurements"""
    DVL = "dvl"
    DEPTH = "depth"
    NONE = "none"


@dataclass
class EkfParams:
    """EKF parameters"""
    initial_position_stddev_m: float
    initial_velocity_stddev_mps: float
    process_noise_density_pos: float
    process_noise_density_vel: float
    gravity: float
    history_length: int
    body_frame: str
    dvl_frame: str
    depth_frame: str


@dataclass
class EkfStaticTransforms:
    """Static transforms for the EKF"""
    r_body_depth_B: np.ndarray
    body_T_dvl: SE3
    body_T_imu: SE3

class EkfHistory:
    """History management for the EKF"""
    
    def __init__(self, t_depth: int, t_imu: int, depth: DepthInput, imu: EkfControl,
                params: EkfParams, max_length: int, logger: Logger,
                t_dvl: Optional[int] = None, dvl: Optional[DvlInput] = None):

        # Initialize
        self.control_history: Deque[Tuple[int, EkfControl]] = deque(maxlen=max_length)
        # t -> (MeasurementType, Measurement, State, Covariance)
        self.state_history: Dict[int, Tuple[MeasurementType, Union[DepthInput, DvlInput], np.ndarray, np.ndarray]] = {}
        self.last_depth_t: int = t_depth
        self.last_dvl_t: int = t_dvl if t_dvl is not None else 0
        self.last_imu_t: int = t_imu

        # Initialize state using depth measurement
        # State [r_bo_O, v_bo_B]
        state = np.array([0.0, 0.0, depth.z[0, 0], 0.0, 0.0, 0.0])[:, np.newaxis]
        var_r = params.initial_position_stddev_m ** 2
        var_v = params.initial_velocity_stddev_mps ** 2
        cov = np.diag([var_r, var_r, depth.R[0, 0], var_v, var_v, var_v])

        self.control_history.append((t_imu, imu))

        # Add initial states
        self.state_history[t_depth] = (MeasurementType.DEPTH, depth, state, cov)
        
        # Only add DVL state if DVL measurement is provided
        if t_dvl is not None and dvl is not None:
            self.state_history[t_dvl] = (MeasurementType.DVL, dvl, state, cov)

        self._logger = logger

    def add_imu_measurement(self, t: int, imu: EkfControl) -> None:
        """Add IMU measurement to history"""
        if t <= self.last_imu_t:
            raise ValueError(f"IMU measurement at {t} is not newer than last IMU at {self.last_imu_t}")
        
        if t <= self.last_dvl_t:
            return 
            # TODO: Handle this case
        
        self.control_history.append((t, imu))
        self.last_imu_t = t
    
    def add_depth_measurement(self, t: int, depth: DepthInput, ekf: 'Ekf') -> None:
        """Add depth measurement"""
        # Check constraints
        if t <= self.last_depth_t:
            raise ValueError("Depth measurement timestamp not newer than last depth")
        if t <= self.last_dvl_t:
            return
            # TODO: Handle this case
            # raise ValueError("Depth measurement timestamp not newer than last DVL")
        
        # Find latest state
        state_t, state_est = self._find_latest_state_before(t)
        if state_est is None:
            return
        
        # Find the closest control
        closest_control_t, closest_control = self._find_closest_control(t)
        
        # Predict from state_t to t
        dt = t - state_t
        x_pred = ekf.predict(state_est[2], closest_control, dt)
        P_pred = ekf.predict_cov(state_est[3], closest_control, dt)
        
        # Apply depth update
        z_pred: np.ndarray = ekf.h_depth(x_pred, closest_control)

        x_updated, P_updated = ekf.update(x_pred, P_pred, depth.z, depth.R, z_pred, ekf.H_depth)

        # Store state
        assert x_updated.shape == (6, 1)
        assert P_updated.shape == (6, 6)
        self.state_history[t] = (MeasurementType.DEPTH, depth, x_updated, P_updated)
        self.last_depth_t = t
        
        # Cleanup old states
        self._cleanup()

    def add_dvl_measurement(self, t: int, dvl: DvlInput, ekf: 'Ekf') -> None:
        """Add DVL measurement and perform update, replaying all subsequent states."""
        import logging

        # DVL measurements must be in order: reject if t <= any existing DVL measurement
        dvl_times = [tt for tt, (mtype, _, _, _) in self.state_history.items() if mtype == MeasurementType.DVL]
        if any(t <= tt for tt in dvl_times):
            logging.warning(f"DVL measurement at {t} is not newer than existing DVL measurements. Rejecting.")
            raise ValueError("DVL measurement timestamp not newer than existing DVL measurements")
        
        # 1. Find the latest state estimate before t
        state_t, state_est = self._find_latest_state_before(t)
        if state_est is None:
            return
        
        # 2. Find closest IMU/control for t
        closest_control_t, closest_control = self._find_closest_control(t)
        
        # 3. Predict from state_t to t
        dt = t - state_t
        if dt < 0:
            raise ValueError("Negative time delta in prediction")
        
        x_pred = ekf.predict(state_est[2], closest_control, dt)
        P_pred = ekf.predict_cov(state_est[3], closest_control, dt)
        
        # 4. Apply DVL update using analytic Jacobian
        z_pred = ekf.h_dvl(x_pred, closest_control)
        z = dvl.v_dvl_V
        R = dvl.R

        x_updated, P_updated = ekf.update(x_pred, P_pred, z, R, z_pred, ekf.H_dvl)
        
        # 5. Insert the DVL measurement and updated state
        assert x_updated.shape == (6, 1)
        assert P_updated.shape == (6, 6)
        self.state_history[t] = (MeasurementType.DVL, dvl, x_updated, P_updated)
        self.last_dvl_t = t
        
        # 6. Replay all subsequent measurements
        subsequent_times = sorted([tt for tt in self.state_history.keys() if tt > t])
        for replay_t in subsequent_times:
            mtype, meas, x, P = self.state_history[replay_t]
            if mtype == MeasurementType.DVL:
                # Error: encountered another DVL during replay
                raise ValueError(f"Encountered another DVL measurement at {replay_t} during replay. DVL measurements must be in order.")
            elif mtype == MeasurementType.DEPTH:
                # For simplicity, use the predicted depth value as a placeholder
                latest_t, latest_est = self._find_latest_state_before(replay_t)
                if latest_est is None:
                    return
                closest_control_t, closest_control_data = self._find_closest_control(replay_t)
                dt_replay = replay_t - latest_t
                x_pred = ekf.predict(latest_est[2], closest_control_data, dt_replay)
                P_pred = ekf.predict_cov(latest_est[3], closest_control_data, dt_replay)
                
                # Predict the depth
                z_pred = ekf.h_depth(x_pred, closest_control_data)
                z = meas.z
                R = meas.R

                # Apply depth update
                x_updated_replay, P_updated_replay = ekf.update(x_pred, P_pred, z, R, z_pred, ekf.H_depth)
                
                # Update the state at replay_t
                assert x_updated_replay.shape == (6, 1)
                assert P_updated_replay.shape == (6, 6)
                self.state_history[replay_t] = (MeasurementType.DEPTH, meas, x_updated_replay, P_updated_replay)
        
        # Cleanup old states
        self._cleanup()
    
    def _find_closest_control(self, t: int) -> Tuple[int, EkfControl]:
        """Find closest control input by timestamp"""
        if not self.control_history:
            raise ValueError("No control inputs in history")
        
        # Find closest control using binary search
        times = [tc[0] for tc in self.control_history]
        idx = min(range(len(times)), key=lambda i: abs(times[i] - t))
        return self.control_history[idx]
    
    def _find_latest_state_before(self, t: int) -> Tuple[int, Tuple[MeasurementType, Union[DepthInput, DvlInput], np.ndarray, np.ndarray]]:
        """Find latest state estimate before given time"""
        valid_times = [time for time in self.state_history.keys() if time < t]
        if not valid_times:
            self._logger.warning(f"No state estimate found before the given time {t}")
            return None, None
        
        latest_time = max(valid_times)
        return latest_time, self.state_history[latest_time]
    
    def get_latest_state(self) -> Optional[Tuple[int, Tuple[MeasurementType, Union[DepthInput, DvlInput], np.ndarray, np.ndarray]]]:
        """Get the current best estimate (latest state)"""
        if not self.state_history:
            return None
        
        latest_time = max(self.state_history.keys())
        return latest_time, self.state_history[latest_time]

    def _cleanup(self) -> None:
        """Cleanup old state estimates"""
        if self.control_history:
            oldest_control_time = self.control_history[0][0]
            self.state_history = {k: v for k, v in self.state_history.items() 
                                if k >= oldest_control_time}


class Ekf:
    """Extended Kalman Filter implementation"""
    
    def __init__(self, params: EkfParams, transforms: EkfStaticTransforms, logger: Logger):
        """Initialize EKF with parameters and transforms"""
        self._r_body_depth_B: np.ndarray = transforms.r_body_depth_B
        dvl_T_body: SE3 = transforms.body_T_dvl.inv()
        self._dvl_J_body: np.ndarray = dvl_T_body.jacob()
        # TODO: remove this
        assert not np.allclose(transforms.body_T_imu.t, 0)
        self._body_T_imu: SE3 = transforms.body_T_imu
        
        # Depth measurement Jacobian
        self._H_depth = np.zeros((1, 6))
        self._H_depth[0, 2] = 1.0  # dh/dz = 1

        # DVL Jacobian
        self._H_dvl = np.hstack([np.zeros((3, 3)), self._dvl_J_body[:3, :3]])

        # Process noise
        var_r = params.process_noise_density_pos**2
        var_v = params.process_noise_density_vel**2
        self._Qc = np.diag([var_r, var_r, var_r, var_v, var_v, var_v])
        
        # Gravity vector (pointing up in odom frame)
        self._a_g_O = np.array([0.0, 0.0, params.gravity])[:, np.newaxis]

        # Logger
        self._logger = logger
    
    def predict(self, xkm1: np.ndarray, uk: EkfControl, dt: int) -> np.ndarray:
        assert xkm1.shape == (6, 1) and uk.is_valid() and dt > 0

        dt_seconds = dt * 1e-9

        # Sensor frame origin is body frame origin
        body_R_imu = SO3(self._body_T_imu.R)
        odom_R_body: SO3 = uk.odom_R_sensor * body_R_imu.inv()
        omega_B = body_R_imu * uk.omega_S
        r_sensor__body_B = self._body_T_imu.t

        omega_B_flat = omega_B.reshape(3)
        a_body_B = body_R_imu * uk.a_S - np.cross(omega_B_flat, np.cross(omega_B_flat, r_sensor__body_B)).reshape(3, 1)
        
        a_body_O = odom_R_body * a_body_B
        # a_body_O_free = a_body_O + self._a_g_O
        a_body_O_free = a_body_O


        # Position update: r = r + v*dt + 0.5*a*dt^2
        r_body_km1_O, v_body_km1_B = xkm1[:3], xkm1[3:]
        v_body_km1_O = odom_R_body * v_body_km1_B
        r_body_km1_body_k_O = v_body_km1_O * dt_seconds + 0.5 * a_body_O_free * dt_seconds**2
        r_body_k_O = r_body_km1_O + r_body_km1_body_k_O
        
        # Velocity update: v = v + a*dt
        v_body_k_O = v_body_km1_O + a_body_O_free * dt_seconds
        v_body_k_B = odom_R_body.inv() * v_body_k_O

        assert r_body_k_O.shape == v_body_k_B.shape == (3, 1)

        return np.vstack((r_body_k_O, v_body_k_B))

    def predict_cov(self, Pkm1: np.ndarray, uk: EkfControl, dt: int) -> np.ndarray:
        """Predict covariance"""
        assert Pkm1.shape == (6, 6)

        body_R_imu = SO3(self._body_T_imu.R)
        odom_R_body: SO3 = uk.odom_R_sensor * body_R_imu.inv()

        dt_seconds = dt * 1e-9

        F = np.block([[np.eye(3), odom_R_body.A * dt_seconds],
                      [np.zeros((3, 3)), np.eye(3)]])
        
        Pk = F @ Pkm1 @ F.T + self._Qc * np.sqrt(dt)
        return Pk

    def update(self, xk_hat: np.ndarray, Pk_hat: np.ndarray, zk: np.ndarray,
               Rk: np.ndarray, zk_hat: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update state and covariance with measurement"""
        assert xk_hat.shape == (6, 1) and Pk_hat.shape == (6, 6)
        assert zk.shape == zk_hat.shape and zk.shape[1] == 1 and len(zk.shape) == 2
        assert H.shape == (zk.shape[0], xk_hat.shape[0])
        assert Pk_hat.shape == (xk_hat.shape[0], xk_hat.shape[0])
        assert Rk.shape == (zk.shape[0], zk.shape[0])



        yk_hat = zk - zk_hat
        Sk = H @ Pk_hat @ H.T + Rk

        try:
            Sk_inv = np.linalg.inv(Sk)
        except np.linalg.LinAlgError:
            raise ValueError("Sk singular")


        Kk = Pk_hat @ H.T @ Sk_inv
        xk = xk_hat + Kk @ yk_hat
        Pk = (np.eye(6) - Kk @ H) @ Pk_hat

        return xk, Pk
    
    def h_dvl(self, xk: np.ndarray, uk: EkfControl) -> np.ndarray:
        assert xk.shape == (6, 1) and uk.is_valid()

        # Transform velocity to body frame
        v_body_B = xk[3:]
        body_R_imu = SO3(self._body_T_imu.R)
        omega_body_B = body_R_imu * uk.omega_S
        
        # Create twist vector
        V_body_B = np.vstack([v_body_B, omega_body_B])
        
        # Transform to DVL frame
        V_dvl_V = (self._dvl_J_body @ V_body_B)
        return V_dvl_V[:3]

    def h_depth(self, xk: np.ndarray, uk: EkfControl) -> np.ndarray:
        assert xk.shape == (6, 1) and uk.is_valid()

        r_body_O = xk[:3]
        body_R_imu = SO3(self._body_T_imu.R)
        odom_R_body: SO3 = uk.odom_R_sensor * body_R_imu.inv()

        # Transform depth sensor position to odom frame
        r_body_depth_O = odom_R_body * self._r_body_depth_B
        r_odom_depth_O = r_body_O + r_body_depth_O
        return np.array(r_odom_depth_O[2:3])

    @property
    def H_depth(self) -> np.ndarray:
        return self._H_depth

    @property
    def H_dvl(self) -> np.ndarray:
        return self._H_dvl


class StateEstimatorEkf(Node):

    def __init__(self) -> None:
        super().__init__('state_estimator_ekf')
        
        # Declare parameters
        self.body_frame = self.declare_parameter('body_frame', 'os/body').value
        self.depth_frame = self.declare_parameter('depth_frame', 'os/depth').value
        self.dvl_frame = self.declare_parameter('dvl_frame', 'os/dvl').value
        self.imu_frame = self.declare_parameter('imu_frame', 'os/imu').value
        self.initial_position_stddev_m = self.declare_parameter('initial_position_stddev_m', 0.01).value
        self.initial_velocity_stddev_mps = self.declare_parameter('initial_velocity_stddev_mps', 0.1).value
        self.process_noise_density_pos = self.declare_parameter('process_noise_density_pos_m_per_sqrt_s', 0.01).value
        self.process_noise_density_vel = self.declare_parameter('process_noise_density_vel_mps_per_sqrt_s', 0.01).value
        self.gravity = self.declare_parameter('g', 9.85).value
        self.history_length = self.declare_parameter('history_length', 20).value
        
        # Create parameters object
        self.params = EkfParams(
            initial_position_stddev_m=self.initial_position_stddev_m,
            initial_velocity_stddev_mps=self.initial_velocity_stddev_mps,
            process_noise_density_pos=self.process_noise_density_pos,
            process_noise_density_vel=self.process_noise_density_vel,
            gravity=self.gravity,
            history_length=self.history_length,
            body_frame=self.body_frame,
            dvl_frame=self.dvl_frame,
            depth_frame=self.depth_frame
        )
        
        # Setup TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.input_queue: Optional[Queue] = None
        self.imu_sub: Optional[Subscription] = None
        self.depth_sub: Optional[Subscription] = None
        self.dvl_sub: Optional[Subscription] = None
        self.nav_state_pub: Optional[Publisher] = None
        self.processing_thread: Optional[Thread] = None

        # State
        self.ekf: Optional[Ekf] = None
        self.history: Optional[EkfHistory] = None
        self.static_transforms: Optional[EkfStaticTransforms] = None
        self._transforms_ready = False

        # Timer for checking static transforms
        self._static_tf_timer = self.create_timer(0.1, self._static_tf_timer_callback)
    
    def _static_tf_timer_callback(self) -> None:
        # Only run if not already ready
        self.get_logger().debug(f"Looking up transforms: {self.body_frame}, {self.depth_frame}, {self.dvl_frame}")
        try:
            now = self.get_clock().now()
            # Look up depth transform
            depth_tf = self.tf_buffer.lookup_transform(
                self.body_frame, self.depth_frame, now
            )
            # Look up DVL transform
            dvl_tf = self.tf_buffer.lookup_transform(
                self.body_frame, self.dvl_frame, now
            )
            # Look up IMU transform
            imu_tf = self.tf_buffer.lookup_transform(
                self.body_frame, self.imu_frame, now
            )
            # Extract transforms
            r_body_depth_B = np.array([
                depth_tf.transform.translation.x,
                depth_tf.transform.translation.y,
                depth_tf.transform.translation.z
            ])
            # Convert DVL transform to isometry matrix
            body_T_dvl = numpify(dvl_tf.transform)
            body_T_imu = numpify(imu_tf.transform)
            # Log the DVL and IMU transforms for debugging and verification
            self.get_logger().info(f"DVL Transform (body_T_dvl):\n{body_T_dvl}")
            self.get_logger().info(f"IMU Transform (body_T_imu):\n{body_T_imu}")
            self.static_transforms = EkfStaticTransforms(r_body_depth_B, body_T_dvl, body_T_imu)
            self._transforms_ready = True
            # Now initialize EKF and processing thread
            self.ekf = Ekf(self.params, self.static_transforms, self.get_logger())
            self.get_logger().info("got all transforms :D")
            self.get_logger().info("EKF initialized")
            # Processing thread
            self.start_processing()
            # Stop the timer
            self._static_tf_timer.cancel()
        except Exception as e:
            self.get_logger().warn(f"Transforms not available yet, waiting... ({e})")

    def start_processing(self) -> None:
        self.input_queue = Queue()
        # Setup publishers and subscribers
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.imu_sub = self.create_subscription(ImuMsg, 'imu', self.imu_callback, qos)
        self.depth_sub = self.create_subscription(DepthMsg, 'depth', self.depth_callback, qos)
        self.dvl_sub = self.create_subscription(DvlMsg, 'dvl', self.dvl_callback, qos)
        self.nav_state_pub = self.create_publisher(NavigationState, 'navigation_state', 10)
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    def imu_callback(self, msg: ImuMsg) -> None:
        """IMU message callback"""
        t = stamp_to_nanos(msg.header.stamp)
        u = EkfControl.from_msg(msg)
        if self.input_queue is not None:
            self.input_queue.put((EkfInput.IMU, t, u))

    def depth_callback(self, msg: DepthMsg) -> None:
        """Depth message callback"""
        t = stamp_to_nanos(msg.header.stamp)
        depth_input = DepthInput.from_msg(msg)
        if self.input_queue is not None:
            self.input_queue.put((EkfInput.DEPTH, t, depth_input))

    def dvl_callback(self, msg: DvlMsg) -> None:
        """DVL message callback"""
        t = stamp_to_nanos(msg.header.stamp)
        dvl_input = DvlInput.from_msg(msg)
        if self.input_queue is not None:
            self.input_queue.put((EkfInput.DVL, t, dvl_input))

    def _processing_loop(self) -> None:
        """Main processing loop"""
        self.get_logger().info("EKF processing thread started, waiting for first measurements...")
        
        # Wait for initial measurements (only depth and IMU required)
        depth_input_stamped = None
        imu_input_stamped = None
        
        # Store any DVL measurements that arrive before initialization
        pending_dvl_measurements = []
        
        while rclpy.ok():
            try:
                if self.input_queue is None:
                    continue
                input_type, t, data = self.input_queue.get(timeout=1.0)
                
                if input_type == EkfInput.DEPTH:
                    depth_input_stamped = (t, data)
                elif input_type == EkfInput.DVL:
                    # Store DVL measurements that arrive before initialization
                    pending_dvl_measurements.append((t, data))
                elif input_type == EkfInput.IMU:
                    imu_input_stamped = (t, data)
                
                # Only wait for depth and IMU to start
                if (depth_input_stamped is not None and 
                    imu_input_stamped is not None):
                    break
                    
            except queue.Empty:
                continue
        
        # Initialize history with depth and IMU only
        if (depth_input_stamped is None or imu_input_stamped is None):
            self.get_logger().error("Failed to get initial depth and IMU measurements")
            return
            
        t_depth, depth_input = depth_input_stamped
        t_imu, imu_input = imu_input_stamped
        
        try:
            self.history = EkfHistory(
                t_depth,
                t_imu,
                depth_input,
                imu_input,
                self.params,
                50,
                self.get_logger()
            )
            self.get_logger().info("EKF history initialized without DVL - ready to publish navigation state")
            
            # Publish initial navigation state
            self._publish_navigation_state()
            
        except ValueError as e:
            self.get_logger().error(f"Failed to initialize EKF history: {e}")
            return
        
        # Process any pending DVL measurements that arrived before initialization
        for t_dvl, dvl_data in pending_dvl_measurements:
            try:
                if self.ekf is not None:
                    self.history.add_dvl_measurement(t_dvl, dvl_data, self.ekf)
                    self._publish_navigation_state()
                    self.get_logger().info(f"Processed pending DVL measurement from {t_dvl}")
            except Exception as e:
                self.get_logger().warn(f"Failed to process pending DVL measurement: {e}")
        
        # Main processing loop
        while rclpy.ok():
            try:
                if self.input_queue is None:
                    continue
                input_type, t, data = self.input_queue.get(timeout=1.0)
                
                if input_type == EkfInput.IMU:
                    self.history.add_imu_measurement(t, data)

                elif input_type == EkfInput.DEPTH:
                    # try:
                    if self.ekf is not None:
                        self.history.add_depth_measurement(t, data, self.ekf)
                    self._publish_navigation_state()
                    # except ValueError as e:
                    #     self.get_logger().warn(f"Depth measurement rejected: {e}")
                
                elif input_type == EkfInput.DVL:
                    if self.ekf is not None:
                        self.history.add_dvl_measurement(t, data, self.ekf)
                    self._publish_navigation_state()

            except queue.Empty:
                continue
            # except Exception as e:
            #     self.get_logger().error(f"Error in processing loop: {e}")
    
    def _publish_navigation_state(self) -> None:
        """Publish current navigation state estimate"""
        if self.history is None:
            return
        
        latest = self.history.get_latest_state()
        if latest is None:
            return
        
        t, state_est = latest
        
        # Get latest IMU data for acceleration and angular velocity
        if not self.history.control_history:
            return
        
        # TODO: we should be using closest control to the state estimate
        latest_imu_t_ns, latest_imu = self.history.control_history[-1]

        # Calculate the body orientation
        odom_R_sensor = latest_imu.odom_R_sensor
        body_T_sensor = self.static_transforms.body_T_imu
        body_R_sensor = SO3(body_T_sensor)
        odom_R_body = odom_R_sensor * body_R_sensor.inv()
    
        # Accel
        a_gravity_O = np.c_[[0.0, 0.0, self.params.gravity]]
        a_gravity_B = odom_R_body.inv() * a_gravity_O
        a_sensor_S = latest_imu.a_S
        glebs_magic_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        a_body_B = glebs_magic_matrix @ a_sensor_S # + a_gravity_B

        # We assume that the body frame and IMU frame have the same origin
        omega_sensor_S = latest_imu.omega_S
        omega_body_B = body_R_sensor * omega_sensor_S
        
        # Create NavigationState message
        nav_msg = NavigationState()
        nav_msg.header.stamp = TimeMsg(sec=latest_imu_t_ns // 1_000_000_000, nanosec=latest_imu_t_ns % 1_000_000_000)
        nav_msg.header.frame_id = "odom"
        
        # Set pose (odom_t_sub)
        nav_msg.body_pose = Pose()
        nav_msg.body_pose.position.x = float(state_est[2][0, 0])
        nav_msg.body_pose.position.y = float(state_est[2][1, 0])
        nav_msg.body_pose.position.z = float(state_est[2][2, 0])
        
        # Set orientation from IMU (quaternion from IMU sensor frame to odom frame)
        odom_q_body = UnitQuaternion(odom_R_body).vec_xyzs  # quaternion [x, y, z, w]
        nav_msg.body_pose.orientation.x = float(odom_q_body[0])
        nav_msg.body_pose.orientation.y = float(odom_q_body[1])
        nav_msg.body_pose.orientation.z = float(odom_q_body[2])
        nav_msg.body_pose.orientation.w = float(odom_q_body[3])
        
        # Set body twist
        V_B = np.vstack([state_est[2][3:6], omega_body_B])
        nav_msg.body_twist = msgify(V_B, message_type="Twist")
        
        # Set acceleration in body frame (a_b)
        # Transform IMU acceleration from sensor frame to body frame
        nav_msg.a_b = Vector3()
        nav_msg.a_b.x = float(a_body_B[0, 0])
        nav_msg.a_b.y = float(a_body_B[1, 0])
        nav_msg.a_b.z = float(a_body_B[2, 0])
        
        if self.nav_state_pub is not None:
            self.nav_state_pub.publish(nav_msg)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = StateEstimatorEkf()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
