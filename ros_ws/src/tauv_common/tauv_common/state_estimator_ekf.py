import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.subscription import Subscription
from rclpy.publisher import Publisher
from rclpy.time import Time

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Dict, List, Deque, Any
from dataclasses import dataclass
from enum import Enum
import math
from collections import deque
import threading
from threading import Thread
import queue
from queue import Queue

from spatialmath import SE3, SO3

# ROS2 messages
from sensor_msgs.msg import Imu as ImuMsg
from nav_msgs.msg import Odometry
from tauv_common.util.geometry import numpify
from tauv_msgs.msg import Depth as DepthMsg
from tauv_msgs.msg import WaterlinkedDvlFrame as DvlMsg
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3, Point

# TF2
import tf2_ros
import tf2_geometry_msgs
from tf2_ros import TransformException

def stamp_to_nanos(stamp) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec

@dataclass
class EkfControl:
    odom_R_sensor: SO3
    a_sensor_S: NDArray
    omega_sensor_S: NDArray

    @staticmethod
    def from_msg(msg: ImuMsg) -> 'EkfControl':
        return EkfControl(
            odom_R_sensor=numpify(msg.orientation).SO3(),
            a_sensor_S=numpify(msg.linear_acceleration),
            omega_sensor_S=numpify(msg.angular_velocity),
        )

@dataclass
class DvlInput:
    """DVL measurement input"""
    v_dvl_V: NDArray  # Velocity in DVL frame
    R: NDArray        # Measurement covariance

    @staticmethod
    def from_msg(msg: DvlMsg) -> 'DvlInput':
        return DvlInput(
            v_dvl_V=np.array([msg.vx, msg.vy, msg.vz]),
            R=msg.covariance
        )


@dataclass
class DepthInput:
    """Depth measurement input"""
    z: float          # Depth measurement
    R: float          # Measurement variance

    @staticmethod
    def from_msg(msg: DepthMsg) -> 'DepthInput':
        return DepthInput(
            z=msg.depth,
            R=msg.variance,
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
    r_body_depth_B: NDArray
    body_T_dvl: SE3
    body_T_imu: SE3

class EkfHistory:
    """History management for the EKF"""
    
    def __init__(self, t_depth: int, t_dvl: int, t_imu: int,
                depth: DepthInput, dvl: DvlInput, imu: NDArray,
                params: EkfParams, max_length: int):

        # Validate inputs
        max_dt: int = 200_000_000  # 200ms maximum time difference between measurements
        if (abs(t_depth - t_dvl) > max_dt or
                abs(t_depth - t_imu) > max_dt or
                abs(t_imu - t_dvl) > max_dt):
            raise ValueError("Initial measurements are too far apart in time")

        # Initialize
        self.control_history: Deque[(int, NDArray)] = deque(maxlen=max_length)
        self.state_history: Dict[int, Tuple[MeasurementType, NDArray, NDArray]] = {}
        self.last_depth_t: int = t_depth
        self.last_dvl_t: int = t_dvl
        self.last_imu_t: int = t_imu

        # Initialize state using depth measurement
        # State [r_bo_O, v_bo_B]
        state = np.array([0.0, 0.0, depth.z, 0.0, 0.0, 0.0])
        var_r = params.initial_position_stddev_m ** 2
        var_v = params.initial_velocity_stddev_mps ** 2
        cov = np.diag([var_r, var_r, depth.R, var_v, var_v, var_v])

        self.control_history.append((t_imu, imu))

        # Add initial states
        self.state_history[t_depth] = (MeasurementType.DEPTH, state, cov)
        self.state_history[t_dvl] = (MeasurementType.DVL, state, cov)

    def add_imu_measurement(self, t: int, imu: NDArray) -> None:
        """Add IMU measurement to history"""
        if t <= self.last_imu_t:
            raise ValueError(f"IMU measurement at {t} is not newer than last IMU at {self.last_imu_t}")
        
        if t <= self.last_dvl_t:
            raise ValueError(f"IMU measurement at {t} is not newer than last DVL at {self.last_dvl_t}")
        
        self.control_history.append((t, imu))
        self.last_imu_t = t
    
    def add_depth_measurement(self, t: int, depth: DepthInput, ekf: 'Ekf') -> None:
        """Add depth measurement"""
        # Check constraints
        if t <= self.last_depth_t:
            raise ValueError("Depth measurement timestamp not newer than last depth")
        if t <= self.last_dvl_t:
            raise ValueError("Depth measurement timestamp not newer than last DVL")
        
        # Find latest state
        state_t, state_est = self._find_latest_state_before(t)
        
        # Find the closest control
        closest_imu = self._find_closest_control(t)
        
        # Predict from state_t to t
        dt = t - state_t
        x_pred = ekf.predict(state_est.state, closest_imu.control, dt)
        P_pred = ekf.predict_cov(state_est.cov, dt)
        
        # Apply depth update
        z_pred = ekf.h_depth(x_pred, closest_imu.control)
        z = np.array([depth.z])
        R = np.array([[depth.R]])

        x_updated, P_updated = Ekf.update(x_pred, P_pred, z, R, np.array([z_pred]), ekf.H_depth)

        # Store state
        self.state_history[t] = (MeasurementType.DEPTH, x_updated, P_updated)
        self.last_depth_t = t
        
        # Cleanup old states
        self._cleanup()

    def add_dvl_measurement(self, t: int, dvl: DvlInput, ekf: 'Ekf') -> None:
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
        if dt < 0:
            raise ValueError("Negative time delta in prediction")
        
        x_pred = ekf.predict(state_est.state, closest_imu.control, dt)
        P_pred = ekf.predict_cov(state_est.cov, dt)
        
        # 4. Apply DVL update using analytic Jacobian
        z_pred = ekf.h_dvl(x_pred, closest_imu.control)
        z = dvl.v_dvl_V
        R = dvl.R

        x_updated, P_updated = ekf.update(x_pred, P_pred, z, R, z_pred, ekf.H_dvl)
        
        # 5. Insert the DVL measurement and updated state
        self.state_history[t] = (MeasurementType.DVL, x_updated, P_updated)
        self.last_dvl_t = t
        
        # 6. Replay all subsequent measurements
        subsequent_times = sorted([tt for tt in self.state_history.keys() if tt > t])
        for replay_t in subsequent_times:
            mtype, _, _ = self.state_history[replay_t]
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
                self.state_history[replay_t] = (MeasurementType.DEPTH, x_updated_replay, P_updated_replay)
        
        # Cleanup old states
        self._cleanup()
    
    def _find_closest_control(self, t: float) -> (int, NDArray):
        """Find closest control input by timestamp"""
        if not self.control_history:
            raise ValueError("No control inputs in history")
        
        # Find closest control using binary search
        times = [tc.t for tc in self.control_history]
        idx = min(range(len(times)), key=lambda i: abs(times[i] - t))
        return self.control_history[idx]
    
    def _find_latest_state_before(self, t: int) -> Tuple[int, (MeasurementType, NDArray, NDArray)]:
        """Find latest state estimate before given time"""
        valid_times = [time for time in self.state_history.keys() if time < t]
        if not valid_times:
            raise ValueError("No state estimate found before the given time")
        
        latest_time = max(valid_times)
        return latest_time, self.state_history[latest_time]
    
    def get_latest_state(self) -> Optional[Tuple[float, (MeasurementType, NDArray, NDArray)]]:
        """Get the current best estimate (latest state)"""
        if not self.state_history:
            return None
        
        latest_time = max(self.state_history.keys())
        return latest_time, self.state_history[latest_time]

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
        self._r_body_depth_B: NDArray = transforms.r_body_depth_B
        dvl_T_body: SE3 = transforms.body_T_dvl.inv()
        self._dvl_J_body: NDArray = dvl_T_body.jacob()
        assert np.allclose(transforms.body_T_imu.t, 0)
        self._imu_R_body: SO3 = transforms.body_T_imu.R.inv()
        
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
        self._a_g_O = np.array([0.0, 0.0, params.gravity])
    
    def predict(self, xkm1: NDArray, uk: EkfControl, dt: int) -> NDArray:
        dt = dt * 1e-9
        # Transform acceleration to odom frame
        a_sensor_O = uk.odom_R_sensor @ uk.a_sensor_S

        # Sensor frame origin is body frame origin
        a_body_O = a_sensor_O

        odom_R_body: SO3 = uk.odom_R_sensor * self._imu_R_body

        # Position update: r = r + v*dt + 0.5*a*dt^2
        r_body_km1_O, v_body_km1_B = xkm1[:3], xkm1[3:]
        v_body_km1_O = odom_R_body * v_body_km1_B
        r_body_km1_body_k_O = v_body_km1_O * dt + 0.5 * a_body_O * dt**2
        r_body_k_O = r_body_km1_O + r_body_km1_body_k_O
        
        # Velocity update: v = v + a*dt
        v_body_k_O = v_body_km1_O + (a_body_O + self._a_g_O) * dt
        v_body_k_B = odom_R_body.inv() * v_body_k_O

        return np.hstack((r_body_k_O, v_body_k_B))

    @staticmethod
    def predict_cov(Pkm1: NDArray, dt: float) -> NDArray:
        """Predict covariance"""
        I3 = np.eye(3)
        dr_dv = I3 * dt
        F = np.block([[I3, dr_dv],
                     [np.zeros((3, 3)), I3]])
        
        Pk = F @ Pkm1 @ F.T
        return Pk

    @staticmethod
    def update(xk_hat: NDArray, Pk_hat: NDArray, zk: NDArray,
               Rk: NDArray, zk_hat: NDArray, H: NDArray) -> Tuple[NDArray, NDArray]:
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
        
        return xk, Pk
    
    def h_dvl(self, xk: NDArray, uk: EkfControl) -> NDArray:
        """DVL measurement function"""
        # Transform velocity to body frame
        v_body_B = xk[3:]
        omega_body_B = self._imu_R_body.inv() * uk.omega_sensor_S
        
        # Create twist vector
        V_body_B = np.hstack([v_body_B, omega_body_B])
        
        # Transform to DVL frame
        V_dvl_V = (self._dvl_J_body @ V_body_B)
        return V_dvl_V[:3]
    
    def h_depth(self, xk: NDArray, uk: EkfControl) -> float:
        """Depth measurement function"""
        r_body_O = xk[:3]
        odom_R_body: SO3 = uk.odom_R_sensor * self._imu_R_body

        # Transform depth sensor position to odom frame
        r_body_depth_O = odom_R_body * self._r_body_depth_B
        r_odom_depth_O = r_body_O + r_body_depth_O
        return r_odom_depth_O[2]

    @property
    def H_depth(self):
        return self._H_depth

    @property
    def H_dvl(self):
        return self._H_dvl


class StateEstimatorEkf(Node):

    def __init__(self):
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
        self.odom_pub: Optional[Publisher] = None
        self.processing_thread: Optional[Thread] = None

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
            self.static_transforms = EkfStaticTransforms(r_body_depth_B, body_T_dvl, body_T_imu)
            self._transforms_ready = True
            self.get_logger().info("Static transforms received")
            # Now initialize EKF and processing thread
            self.ekf = Ekf(self.params, self.static_transforms)
            self.get_logger().info("EKF initialized")
            # Processing thread
            self.start_processing()
            # Stop the timer
            self._static_tf_timer.cancel()
        except Exception as e:
            self.get_logger().debug(f"Transforms not available yet, waiting... ({e})")

    def start_processing(self):
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
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    def imu_callback(self, msg: ImuMsg):
        """IMU message callback"""
        try:
            t = stamp_to_nanos(msg.header.stamp)
            u = EkfControl.from_msg(msg)
            self.input_queue.put((EkfInput.IMU, t, u))
        except Exception as e:
            self.get_logger().error(f"Error processing IMU message: {e}")
    
    def depth_callback(self, msg: DepthMsg):
        """Depth message callback"""
        try:
            t = stamp_to_nanos(msg.header.stamp)
            depth_input = DepthInput.from_msg(msg)
            self.input_queue.put((EkfInput.DEPTH, t, depth_input))
        except Exception as e:
            self.get_logger().error(f"Error processing depth message: {e}")
    
    def dvl_callback(self, msg: DvlMsg):
        """DVL message callback"""
        try:
            t = stamp_to_nanos(msg.header.stamp)
            dvl_input = DvlInput.from_msg(msg)
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
            self.history = EkfHistory(
                t_depth,
                t_dvl,
                t_imu,
                depth_input,
                dvl_input,
                imu_input,
                self.params,
                50,
            )
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