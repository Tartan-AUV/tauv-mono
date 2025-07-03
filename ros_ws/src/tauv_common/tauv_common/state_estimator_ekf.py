import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.parameter import ParameterType
from rclpy.parameter import ParameterValue

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Dict, List, Deque, Any
from dataclasses import dataclass
from enum import Enum
import math
from collections import deque
import threading
import queue
import time

# ROS2 messages
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from tauv_msgs.msg import Depth, WaterlinkedDvlFrame
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3, Point
from std_msgs.msg import Header

# TF2
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformException

# Type aliases for clarity
Vector3 = NDArray[np.float64]  # 3D vector
Vector6 = NDArray[np.float64]  # 6D vector (position + velocity)
Matrix3 = NDArray[np.float64]  # 3x3 matrix
Matrix6 = NDArray[np.float64]  # 6x6 matrix
Matrix3x6 = NDArray[np.float64]  # 3x6 matrix
Rotation = NDArray[np.float64]  # 3x3 rotation matrix
Isometry = NDArray[np.float64]  # 4x4 transformation matrix


@dataclass
class EkfControl:
    """Control input for the EKF (from IMU)"""
    odom_R_body: Rotation  # Rotation from body to odom frame
    a_body_B: Vector3      # Linear acceleration in body frame
    omega_body_B: Vector3  # Angular velocity in body frame
    
    @classmethod
    def from_imu_msg(cls, msg: Imu) -> 'EkfControl':
        """Create EkfControl from IMU message"""
        # Extract quaternion and convert to rotation matrix
        q = msg.orientation
        odom_R_body = cls._quaternion_to_rotation_matrix(q)
        
        # Extract linear acceleration and angular velocity
        a_body_B = np.array([msg.linear_acceleration.x, 
                            msg.linear_acceleration.y, 
                            msg.linear_acceleration.z], dtype=np.float64)
        omega_body_B = np.array([msg.angular_velocity.x, 
                                msg.angular_velocity.y, 
                                msg.angular_velocity.z], dtype=np.float64)
        
        return cls(odom_R_body, a_body_B, omega_body_B)
    
    @staticmethod
    def _quaternion_to_rotation_matrix(q: Quaternion) -> Rotation:
        """Convert quaternion to rotation matrix"""
        # Normalize quaternion
        norm = math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
        qw, qx, qy, qz = q.w/norm, q.x/norm, q.y/norm, q.z/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*qx**2 - 2*qz**2, 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*qx**2 - 2*qy**2]
        ], dtype=np.float64)
        
        return R


@dataclass
class EkfState:
    """EKF state vector [position; velocity]"""
    data: Vector6  # [x, y, z, vx, vy, vz]
    
    def __init__(self, r_body_O: Vector3, v_body_O: Vector3):
        """Initialize state with position and velocity"""
        self.data = np.concatenate([r_body_O, v_body_O])
    
    @classmethod
    def zeros(cls) -> 'EkfState':
        """Create zero state"""
        return cls(np.zeros(3), np.zeros(3))
    
    def r_body_O(self) -> Vector3:
        """Get position vector"""
        return self.data[:3]
    
    def v_body_O(self) -> Vector3:
        """Get velocity vector"""
        return self.data[3:6]
    
    def __getitem__(self, key):
        return self.data[key]


@dataclass
class DvlInput:
    """DVL measurement input"""
    v_dvl_V: Vector3  # Velocity in DVL frame
    R: Matrix3        # Measurement covariance


@dataclass
class DepthInput:
    """Depth measurement input"""
    z: float          # Depth measurement
    R: float          # Measurement variance


class EkfInput(Enum):
    """Types of EKF inputs"""
    IMU = "imu"
    DVL = "dvl"
    DEPTH = "depth"


@dataclass
class TimestampedControl:
    """Control input with timestamp"""
    t: float
    control: EkfControl


@dataclass
class StateEstimate:
    """State estimate with covariance"""
    state: EkfState
    cov: Matrix6


class MeasurementType(Enum):
    """Types of measurements"""
    DVL = "dvl"
    DEPTH = "depth"


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
    r_body_depth_B: Vector3
    body_T_dvl: Isometry


class EkfHistory:
    """History management for the EKF"""
    
    def __init__(self, max_control_history: int):
        self.control_history: Deque[TimestampedControl] = deque(maxlen=max_control_history)
        self.state_history: Dict[float, Tuple[MeasurementType, StateEstimate]] = {}
        self.last_dvl_t: float = 0.0
        self.last_depth_t: float = 0.0
        self.last_imu_t: float = 0.0
    
    @classmethod
    def try_new(cls, t_depth: float, t_dvl: float, t_imu: float,
                depth: DepthInput, dvl: DvlInput, imu: EkfControl,
                params: EkfParams) -> 'EkfHistory':
        """Create new EKF history with initial measurements"""
        max_dt = 0.2  # 200ms maximum time difference
        
        if (abs(t_depth - t_dvl) > max_dt or 
            abs(t_depth - t_imu) > max_dt or 
            abs(t_imu - t_dvl) > max_dt):
            raise ValueError("Initial measurements are too far apart in time")
        
        # Initialize state using depth measurement
        state = EkfState(np.array([0.0, 0.0, depth.z]), np.zeros(3))
        var_r = params.initial_position_stddev_m**2
        var_v = params.initial_velocity_stddev_mps**2
        cov = np.diag([var_r, var_r, depth.R, var_v, var_v, var_v])
        
        history = cls(params.history_length)
        history.control_history.append(TimestampedControl(t_imu, imu))
        
        # Add initial states
        history.state_history[t_depth] = (MeasurementType.DEPTH, StateEstimate(state, cov))
        history.state_history[t_dvl] = (MeasurementType.DVL, StateEstimate(state, cov))
        
        history.last_depth_t = t_depth
        history.last_dvl_t = t_dvl
        history.last_imu_t = t_imu
        
        return history
    
    def add_imu_measurement(self, t: float, imu: EkfControl) -> None:
        """Add IMU measurement to history"""
        if t <= self.last_imu_t:
            raise ValueError(f"IMU measurement at {t} is not newer than last IMU at {self.last_imu_t}")
        
        if t <= self.last_dvl_t:
            raise ValueError(f"IMU measurement at {t} is not newer than last DVL at {self.last_dvl_t}")
        
        self.control_history.append(TimestampedControl(t, imu))
        self.last_imu_t = t
    
    def add_depth_measurement(self, t: float, depth: DepthInput, ekf: 'Ekf') -> None:
        """Add depth measurement"""
        # Check constraints
        if t <= self.last_depth_t:
            raise ValueError("Depth measurement timestamp not newer than last depth")
        if t <= self.last_dvl_t:
            raise ValueError("Depth measurement timestamp not newer than last DVL")
        
        # Find latest state
        state_t, state_est = self._find_latest_state_before(t)
        
        # Find closest control
        closest_imu = self._find_closest_control(t)
        
        # Predict from state_t to t
        dt = t - state_t
        x_pred = ekf.predict(state_est.state, closest_imu.control, dt)
        P_pred = ekf.predict_cov(state_est.cov, dt)
        
        # Apply depth update
        z_pred = ekf.h_depth(x_pred, closest_imu.control)
        z = np.array([depth.z])
        R = np.array([[depth.R]])
        H = np.zeros((1, 6))
        H[0, 2] = 1.0  # dh/dz = 1
        
        x_updated, P_updated = ekf.update(x_pred, P_pred, z, R, np.array([z_pred]), H)
        
        # Store state
        self.state_history[t] = (MeasurementType.DEPTH, StateEstimate(x_updated, P_updated))
        self.last_depth_t = t
        
        # Cleanup old states
        self._cleanup()

    def add_dvl_measurement(self, t: float, dvl: DvlInput, ekf: 'Ekf') -> None:
        """Add DVL measurement and perform update, replaying all subsequent states."""
        import logging
        
        # DVL measurements must be in order: reject if t <= any existing DVL measurement
        dvl_times = [tt for tt, (mtype, _) in self.state_history.items() if mtype == MeasurementType.DVL]
        if any(t <= tt for tt in dvl_times):
            logging.warning(f"DVL measurement at {t} is not newer than existing DVL measurements. Rejecting.")
            raise ValueError("DVL measurement timestamp not newer than existing DVL measurements")
        
        # 1. Find the latest state estimate before t
        state_t, state_est = self._find_latest_state_before(t)
        
        # 2. Find closest IMU/control for t
        closest_imu = self._find_closest_control(t)
        
        # 3. Predict from state_t to t
        dt = t - state_t
        if dt < 0.0:
            raise ValueError("Negative time delta in prediction")
        
        x_pred = ekf.predict(state_est.state, closest_imu.control, dt)
        P_pred = ekf.predict_cov(state_est.cov, dt)
        
        # 4. Apply DVL update using analytic Jacobian
        z_pred = ekf.h_dvl(x_pred, closest_imu.control)
        z = dvl.v_dvl_V
        R = dvl.R
        H = ekf.F_dvl  # Use analytic Jacobian from Ekf class
        
        x_updated, P_updated = ekf.update(x_pred, P_pred, z, R, z_pred, H)
        
        # 5. Insert the DVL measurement and updated state
        self.state_history[t] = (MeasurementType.DVL, StateEstimate(x_updated, P_updated))
        self.last_dvl_t = t
        
        # 6. Replay all subsequent measurements
        subsequent_times = sorted([tt for tt in self.state_history.keys() if tt > t])
        for replay_t in subsequent_times:
            mtype, _ = self.state_history[replay_t]
            if mtype == MeasurementType.DVL:
                # Error: encountered another DVL during replay
                raise ValueError(f"Encountered another DVL measurement at {replay_t} during replay. DVL measurements must be in order.")
            elif mtype == MeasurementType.DEPTH:
                # For simplicity, use the predicted depth value as a placeholder
                latest_t, latest_est = self._find_latest_state_before(replay_t)
                closest_control = self._find_closest_control(replay_t)
                dt_replay = replay_t - latest_t
                x_pred_replay = ekf.predict(latest_est.state, closest_control.control, dt_replay)
                P_pred_replay = ekf.predict_cov(latest_est.cov, dt_replay)
                
                # Use predicted depth as measurement (placeholder)
                z_pred_depth = ekf.h_depth(x_pred_replay, closest_control.control)
                depth_placeholder = DepthInput(z=z_pred_depth, R=0.01)  # Small variance
                
                # Apply depth update
                z_depth = np.array([depth_placeholder.z])
                R_depth = np.array([[depth_placeholder.R]])
                H_depth = np.zeros((1, 6))
                H_depth[0, 2] = 1.0
                
                x_updated_replay, P_updated_replay = ekf.update(x_pred_replay, P_pred_replay, z_depth, R_depth, np.array([z_pred_depth]), H_depth)
                
                # Update the state at replay_t
                self.state_history[replay_t] = (MeasurementType.DEPTH, StateEstimate(x_updated_replay, P_updated_replay))
        
        # Cleanup old states
        self._cleanup()
    
    def _find_closest_control(self, t: float) -> TimestampedControl:
        """Find closest control input by timestamp"""
        if not self.control_history:
            raise ValueError("No control inputs in history")
        
        # Find closest control using binary search
        times = [tc.t for tc in self.control_history]
        idx = min(range(len(times)), key=lambda i: abs(times[i] - t))
        return self.control_history[idx]
    
    def _find_latest_state_before(self, t: float) -> Tuple[float, StateEstimate]:
        """Find latest state estimate before given time"""
        valid_times = [time for time in self.state_history.keys() if time < t]
        if not valid_times:
            raise ValueError("No state estimate found before the given time")
        
        latest_time = max(valid_times)
        return latest_time, self.state_history[latest_time][1]
    
    def get_latest_state(self) -> Optional[Tuple[float, StateEstimate]]:
        """Get the current best estimate (latest state)"""
        if not self.state_history:
            return None
        
        latest_time = max(self.state_history.keys())
        return latest_time, self.state_history[latest_time][1]

    def _cleanup(self):
        """Cleanup old state estimates"""
        if self.control_history:
            oldest_control_time = self.control_history[0].t
            self.state_history = {k: v for k, v in self.state_history.items() 
                                if k >= oldest_control_time}


class Ekf:
    """Extended Kalman Filter implementation"""
    
    def __init__(self, params: EkfParams, transforms: EkfStaticTransforms):
        """Initialize EKF with parameters and transforms"""
        self.r_body_depth_B = transforms.r_body_depth_B
        self.body_T_dvl = transforms.body_T_dvl
        
        # DVL adjoint matrix and measurement Jacobian
        body_T_dvl_inv = np.linalg.inv(self.body_T_dvl)
        self.Ad_dvl_T_body = self._adjoint_matrix(body_T_dvl_inv)
        self.F_dvl = np.block([[np.zeros((3, 3)), self.Ad_dvl_T_body[:3, :3]]])
        
        # Process noise
        var_r = params.process_noise_density_pos**2
        var_v = params.process_noise_density_vel**2
        self.Qc = np.diag([var_r, var_r, var_r, var_v, var_v, var_v])
        
        # Gravity vector (pointing up in odom frame)
        self.a_g_O = np.array([0.0, 0.0, params.gravity])
    
    def predict(self, xkm1: EkfState, uk: EkfControl, dt: float) -> EkfState:
        """Predict next state"""
        # Transform acceleration to odom frame
        a_body_O = uk.odom_R_body @ uk.a_body_B

        # Position update: r = r + v*dt + 0.5*a*dt^2
        r_body_km1_body_k_O = xkm1.v_body_O() * dt + 0.5 * a_body_O * dt**2
        r_body_k_O = xkm1.r_body_O() + r_body_km1_body_k_O
        
        # Velocity update: v = v + a*dt
        v_body_k_O = xkm1.v_body_O() + (a_body_O + self.a_g_O) * dt
        
        return EkfState(r_body_k_O, v_body_k_O)
    
    def predict_cov(self, Pkm1: Matrix6, dt: float) -> Matrix6:
        """Predict covariance"""
        I3 = np.eye(3)
        dr_dv = I3 * dt
        F = np.block([[I3, dr_dv],
                     [np.zeros((3, 3)), I3]])
        
        Pk = F @ Pkm1 @ F.T
        return Pk
    
    def update(self, xk_hat: EkfState, Pk_hat: Matrix6, zk: NDArray, 
               Rk: NDArray, zk_hat: NDArray, H: NDArray) -> Tuple[EkfState, Matrix6]:
        """Update state and covariance with measurement"""
        yk_hat = zk - zk_hat
        Sk = H @ Pk_hat @ H.T + Rk
        
        try:
            Sk_inv = np.linalg.inv(Sk)
        except np.linalg.LinAlgError:
            raise ValueError("Sk singular")
        
        Kk = Pk_hat @ H.T @ Sk_inv
        xk = xk_hat.data + Kk @ yk_hat
        Pk = (np.eye(6) - Kk @ H) @ Pk_hat
        
        return EkfState(xk[:3], xk[3:6]), Pk
    
    def h_dvl(self, xk: EkfState, uk: EkfControl) -> Vector3:
        """DVL measurement function"""
        # Transform velocity to body frame
        v_body_B = uk.odom_R_body.T @ xk.v_body_O()
        
        # Create twist vector
        xi_body_B = np.concatenate([v_body_B, uk.omega_body_B])
        
        # Transform to DVL frame
        v_dvl_V = (self.Ad_dvl_T_body @ xi_body_B)[:3]
        return v_dvl_V
    
    def h_depth(self, xk: EkfState, uk: EkfControl) -> float:
        """Depth measurement function"""
        # Transform depth sensor position to odom frame
        r_body_depth_O = uk.odom_R_body @ self.r_body_depth_B
        r_odom_depth_O = xk.r_body_O() + r_body_depth_O
        return r_odom_depth_O[2]
    
    @staticmethod
    def _adjoint_matrix(T: Isometry) -> Matrix6:
        """Compute adjoint matrix of transformation"""
        R = T[:3, :3]
        t = T[:3, 3]
        
        # Skew-symmetric matrix of translation
        t_skew = np.array([[0, -t[2], t[1]],
                          [t[2], 0, -t[0]],
                          [-t[1], t[0], 0]])
        
        Ad = np.block([[R, t_skew @ R],
                      [np.zeros((3, 3)), R]])
        return Ad


class StateEstimatorEkf(Node):
    """ROS2 node for EKF state estimation"""
    
    def __init__(self):
        super().__init__('state_estimator_ekf')
        
        # Declare parameters
        self.body_frame = self.declare_parameter('body_frame', 'body').value
        self.depth_frame = self.declare_parameter('depth_frame', 'depth').value
        self.dvl_frame = self.declare_parameter('dvl_frame', 'dvl').value
        self.initial_position_stddev_m = self.declare_parameter('initial_position_stddev_m', 0.01).value
        self.initial_velocity_stddev_mps = self.declare_parameter('initial_velocity_stddev_mps', 0.1).value
        self.process_noise_density_pos = self.declare_parameter('process_noise_density_pos_m_per_sqrt_s', 0.001).value
        self.process_noise_density_vel = self.declare_parameter('process_noise_density_vel_mps_per_sqrt_s', 0.001).value
        self.gravity = self.declare_parameter('g', 9.79596).value
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
        
        # Setup publishers and subscribers
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.imu_sub = self.create_subscription(Imu, 'imu', self.imu_callback, qos)
        self.depth_sub = self.create_subscription(Depth, 'depth', self.depth_callback, qos)
        self.dvl_sub = self.create_subscription(WaterlinkedDvlFrame, 'dvl', self.dvl_callback, qos)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        
        # State
        self.ekf: Optional[Ekf] = None
        self.history: Optional[EkfHistory] = None
        self.static_transforms: Optional[EkfStaticTransforms] = None
        self._transforms_ready = False

        # Timer for checking static transforms
        self.get_logger().info("Waiting for static transforms...")
        self._static_tf_timer = self.create_timer(0.1, self._static_tf_timer_callback)
    
    def _static_tf_timer_callback(self):
        # Only run if not already ready
        if self._transforms_ready:
            return
        self.get_logger().debug(f"Loooking up transforms: {self.body_frame}, {self.depth_frame}, {self.dvl_frame}")
        try:
            # Look up depth transform
            depth_tf = self.tf_buffer.lookup_transform(
                self.body_frame, self.depth_frame, Time())
            # Look up DVL transform
            dvl_tf = self.tf_buffer.lookup_transform(
                self.body_frame, self.dvl_frame, Time())
            # Extract transforms
            r_body_depth_B = np.array([
                depth_tf.transform.translation.x,
                depth_tf.transform.translation.y,
                depth_tf.transform.translation.z
            ])
            # Convert DVL transform to isometry matrix
            body_T_dvl = self._transform_to_isometry(dvl_tf.transform)
            self.static_transforms = EkfStaticTransforms(r_body_depth_B, body_T_dvl)
            self._transforms_ready = True
            self.get_logger().info("Static transforms received")
            # Now initialize EKF and processing thread
            self.ekf = Ekf(self.params, self.static_transforms)
            self.get_logger().info("EKF initialized")
            # Processing thread
            self.input_queue = queue.Queue()
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            # Stop the timer
            self._static_tf_timer.cancel()
        except Exception as e:
            self.get_logger().debug(f"Transforms not available yet, waiting... ({e})")
    
    def _transform_to_isometry(self, transform) -> Isometry:
        """Convert Transform to isometry matrix"""
        # Extract translation
        t = np.array([
            transform.translation.x,
            transform.translation.y,
            transform.translation.z
        ])
        
        # Extract rotation (quaternion to rotation matrix)
        q = transform.rotation
        R = self._quaternion_to_rotation_matrix(q)
        
        # Create isometry matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T
    
    def _quaternion_to_rotation_matrix(self, q: Quaternion) -> Rotation:
        """Convert quaternion to rotation matrix"""
        # Normalize quaternion
        norm = math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
        qw, qx, qy, qz = q.w/norm, q.x/norm, q.y/norm, q.z/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*qx**2 - 2*qz**2, 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*qx**2 - 2*qy**2]
        ])
        
        return R
    
    def imu_callback(self, msg: Imu):
        """IMU message callback"""
        try:
            control = EkfControl.from_imu_msg(msg)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.input_queue.put((EkfInput.IMU, t, control))
        except Exception as e:
            self.get_logger().error(f"Error processing IMU message: {e}")
    
    def depth_callback(self, msg: Depth):
        """Depth message callback"""
        try:
            depth_input = DepthInput(z=msg.depth, R=msg.variance)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.input_queue.put((EkfInput.DEPTH, t, depth_input))
        except Exception as e:
            self.get_logger().error(f"Error processing depth message: {e}")
    
    def dvl_callback(self, msg: WaterlinkedDvlFrame):
        """DVL message callback"""
        try:
            # Extract velocity
            v_dvl_V = np.array([msg.vx, msg.vy, msg.vz])
            
            # Extract covariance (3x3 matrix from flat array)
            R = np.array(msg.covariance).reshape(3, 3)
            
            dvl_input = DvlInput(v_dvl_V=v_dvl_V, R=R)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.input_queue.put((EkfInput.DVL, t, dvl_input))
        except Exception as e:
            self.get_logger().error(f"Error processing DVL message: {e}")
    
    def _processing_loop(self):
        """Main processing loop"""
        self.get_logger().info("EKF processing thread started, waiting for first measurements...")
        
        # Wait for initial measurements
        depth_input_stamped = None
        dvl_input_stamped = None
        imu_input_stamped = None
        
        while rclpy.ok():
            try:
                input_type, t, data = self.input_queue.get(timeout=1.0)
                
                if input_type == EkfInput.DEPTH:
                    depth_input_stamped = (t, data)
                elif input_type == EkfInput.DVL:
                    dvl_input_stamped = (t, data)
                elif input_type == EkfInput.IMU:
                    imu_input_stamped = (t, data)
                
                if (depth_input_stamped is not None and 
                    dvl_input_stamped is not None and 
                    imu_input_stamped is not None):
                    break
                    
            except queue.Empty:
                continue
        
        # Initialize history
        t_depth, depth_input = depth_input_stamped
        t_dvl, dvl_input = dvl_input_stamped
        t_imu, imu_input = imu_input_stamped
        
        try:
            self.history = EkfHistory.try_new(
                t_depth, t_dvl, t_imu, depth_input, dvl_input, imu_input, self.params)
            self.get_logger().info("EKF history initialized")
        except ValueError as e:
            self.get_logger().error(f"Failed to initialize EKF history: {e}")
            return
        
        # Main processing loop
        while rclpy.ok():
            try:
                input_type, t, data = self.input_queue.get(timeout=1.0)
                
                if input_type == EkfInput.IMU:
                    try:
                        self.history.add_imu_measurement(t, data)
                    except ValueError as e:
                        self.get_logger().warn(f"IMU measurement rejected: {e}")
                
                elif input_type == EkfInput.DEPTH:
                    try:
                        self.history.add_depth_measurement(t, data, self.ekf)
                        self._publish_odometry()
                    except ValueError as e:
                        self.get_logger().warn(f"Depth measurement rejected: {e}")
                
                elif input_type == EkfInput.DVL:
                    try:
                        self.history.add_dvl_measurement(t, data, self.ekf)
                        self._publish_odometry()
                    except ValueError as e:
                        self.get_logger().warn(f"DVL measurement rejected: {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error in processing loop: {e}")
    
    def _publish_odometry(self):
        """Publish current odometry estimate"""
        if self.history is None:
            return
        
        latest = self.history.get_latest_state()
        if latest is None:
            return
        
        t, state_est = latest
        
        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = self.body_frame
        
        # Set position
        odom_msg.pose.pose.position.x = state_est.state.r_body_O()[0]
        odom_msg.pose.pose.position.y = state_est.state.r_body_O()[1]
        odom_msg.pose.pose.position.z = state_est.state.r_body_O()[2]
        
        # Set orientation (identity for now - only position/velocity tracking)
        odom_msg.pose.pose.orientation.w = 1.0
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = 0.0
        
        # Set velocity
        odom_msg.twist.twist.linear.x = state_est.state.v_body_O()[0]
        odom_msg.twist.twist.linear.y = state_est.state.v_body_O()[1]
        odom_msg.twist.twist.linear.z = state_est.state.v_body_O()[2]
        
        # Set covariance (flatten 6x6 matrix)
        pose_cov = state_est.cov[:3, :3].flatten().tolist()
        twist_cov = state_est.cov[3:, 3:].flatten().tolist()
        odom_msg.pose.covariance = pose_cov + [0.0] * 27  # 6x6 matrix
        odom_msg.twist.covariance = [0.0] * 27 + twist_cov  # 6x6 matrix
        
        self.odom_pub.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateEstimatorEkf()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main() 