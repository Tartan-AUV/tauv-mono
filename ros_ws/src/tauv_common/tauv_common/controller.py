"""
INDI (Incremental Nonlinear Dynamic Inversion) Controller for AUV

This controller implements INDI control for an autonomous underwater vehicle.
It takes acceleration commands and current state as inputs and produces 
wrench commands for the thruster allocation system.
"""

from copy import deepcopy
import logging
import rclpy
from rclpy.node import Node
from rclpy.subscription import Subscription
from rclpy.publisher import Publisher
import numpy as np
from typing import Optional
from dataclasses import dataclass
from scipy import signal
from scipy.signal import ellipord, ellip, sosfilt, sosfilt_zi

from geometry_msgs.msg import WrenchStamped, Wrench, Vector3, Accel
from tauv_msgs.msg import NavigationState, ControllerDebug, ControllerCommand
from tauv_common.util.geometry import numpify, msgify
from tauv_common.util.time import Time
from spatialmath import SO3, SE3, UnitQuaternion
from std_msgs.msg import Header
import rclpy.logging

@dataclass
class INDIParams:
    """Parameters for the INDI controller"""

    # Mass matrix
    M: np.ndarray  # 6x6 matrix
    
    # Proportional gain for outer loop
    K_p: np.ndarray

    # Control limits
    max_force: float = 1000.0   # Maximum force in N
    max_torque: float = 500.0   # Maximum torque in N⋅m

    # Filtering parameters for acceleration measurements
    # Elliptical filter parameters
    linear_accel_cutoff: float = 4.0   # Passband edge frequency for linear acceleration filter (Hz)
    angular_accel_cutoff: float = 4.0  # Passband edge frequency for angular acceleration filter (Hz)
    linear_accel_stopband: float = 20.0  # Stopband edge frequency for linear acceleration filter (Hz)
    angular_accel_stopband: float = 20.0 # Stopband edge frequency for angular acceleration filter (Hz)
    passband_ripple: float = 1.0        # Passband ripple in dB
    stopband_attenuation: float = 60.0  # Stopband attenuation in dB
    filter_order: int = 3               # Fixed elliptical filter order for low latency
    sampling_freq: float = 50.0         # Expected sampling frequency (Hz)

    @classmethod
    def default(cls) -> 'INDIParams':
        """Create default INDI parameters with a simple diagonal control effectiveness matrix"""
        # Simple diagonal matrix - assumes direct relationship between forces/torques and accelerations
        # This is a rough approximation that should be identified from system data
        mass = np.diag([27.0, 27.0, 27.0])  # kg - estimated vehicle mass
        inertia = np.diag([0.566407, 0.556752, 0.824859])  # kg⋅m² - estimated moments of inertia
        
        M = np.zeros((6, 6))
        M[0:3, 0:3] = mass 
        M[3:6, 3:6] = inertia  

        K_p = np.c_[[0.3, 0.3, 0.3, 0.1, 0.1, 0.1]]

        return cls(M=M, K_p=K_p)

class Controller(Node):
    """
    INDI Controller Node
    
    Subscribes to:
    - /gnc/controller_command (tauv_msgs/ControllerCommand): Controller commands including target twist and feedforward
    - /gnc/navigation_state (tauv_msgs/NavigationState): Current vehicle state
    
    Publishes to:
    - /gnc/target_wrench (geometry_msgs/WrenchStamped): Control wrench for thruster allocation
    - /gnc/controller_debug (tauv_msgs/ControllerDebug): Debug information from the controller
    """
    
    def __init__(self):
        super().__init__("indi_controller")
        
        # Parameters
        self.params = INDIParams.default()
        self._frame_id = "os/body"
        
        # State variables
        self._F_target_prev: np.ndarray = np.zeros((6, 1)) # Previous control input (6x1)
        self._V_dI_B_filtered: Optional[np.ndarray] = None   # Filtered acceleration measurements (6x1)
        self._last_nav_state: Optional[NavigationState] = None
        self._odom_T_body_latched: Optional[SE3] = None
        self._cmd: Optional[ControllerCommand] = None
        
        # Initialize Elliptical SOS filters for acceleration
        # Create separate filters for linear and angular acceleration
        nyquist_freq = self.params.sampling_freq / 2.0
        
        # Linear acceleration filter (3 channels)
        linear_wp = self.params.linear_accel_cutoff / nyquist_freq  # normalized passband edge
        linear_ws = self.params.linear_accel_stopband / nyquist_freq  # normalized stopband edge
        self._linear_filter_sos = ellip(
            self.params.filter_order,
            self.params.passband_ripple,
            self.params.stopband_attenuation,
            linear_wp,
            btype='lowpass',
            output='sos'
        )
        # Initialize filter states for 3 linear acceleration channels
        self._linear_filter_states = [
            sosfilt_zi(self._linear_filter_sos) * 0.0 
            for _ in range(3)
        ]
        
        # Angular acceleration filter (3 channels)
        angular_wp = self.params.angular_accel_cutoff / nyquist_freq  # normalized passband edge
        angular_ws = self.params.angular_accel_stopband / nyquist_freq  # normalized stopband edge
        self._angular_filter_sos = ellip(
            self.params.filter_order,
            self.params.passband_ripple,
            self.params.stopband_attenuation,
            angular_wp,
            btype='lowpass',
            output='sos'
        )
        # Initialize filter states for 3 angular acceleration channels
        self._angular_filter_states = [
            sosfilt_zi(self._angular_filter_sos) * 0.0 
            for _ in range(3)
        ]
        
        # ROS2 interfaces
        self._cmd_sub = self.create_subscription(
            ControllerCommand, 'gnc/controller_command', self._handle_cmd, 10
        )
        
        self._last_nav_state_sub = self.create_subscription(
            NavigationState, 'gnc/navigation_state', self._handle_nav_state, 10
        )
        
        self._wrench_pub = self.create_publisher(
            WrenchStamped, 'gnc/target_wrench_test', 10
        )
        
        self._debug_pub = self.create_publisher(
            ControllerDebug, 'gnc/controller_debug', 10
        )
        
        # Control loop timer (50 Hz)
        self._control_timer = self.create_timer(0.02, self._control_loop)
        
        self.get_logger().info(f"Controller initialized with elliptical SOS filters: "
                              f"linear_cutoff={self.params.linear_accel_cutoff}Hz, "
                              f"linear_stopband={self.params.linear_accel_stopband}Hz, "
                              f"angular_cutoff={self.params.angular_accel_cutoff}Hz, "
                              f"angular_stopband={self.params.angular_accel_stopband}Hz, "
                              f"order={self.params.filter_order}, "
                              f"passband_ripple={self.params.passband_ripple}dB, "
                              f"stopband_attenuation={self.params.stopband_attenuation}dB, "
                              f"sampling_freq={self.params.sampling_freq}Hz")
    
    def _handle_cmd(self, msg: ControllerCommand):
        """Handle incoming acceleration command"""
        self._cmd = msg
        
    def _handle_nav_state(self, nav_state: NavigationState):
        """Handle incoming navigation state"""
        # Estimate angular acceleration
        angular_accel = self._estimate_angular_acceleration(nav_state, self._last_nav_state)
        
        self._last_nav_state = deepcopy(nav_state)

        if angular_accel is None:
            self.get_logger().warn("Controller: could not estimate angular acceleration")
            return

        # Extract and filter acceleration measurements
        # Convert from geometry_msgs to numpy arrays
        linear_accel = numpify(nav_state.a_b)
        measured_accel = np.vstack([linear_accel, angular_accel])

        self._update_filtered_acceleration(measured_accel)
        
    def _update_filtered_acceleration(self, measured_accel: np.ndarray):
        """Apply SOS elliptical filters to acceleration measurements
        
        Uses separate elliptical filters for linear and angular acceleration components
        with configurable passband and stopband frequencies, providing sharp cutoff
        characteristics with minimal passband ripple.
        """
        assert measured_accel.shape == (6, 1)
        assert np.all(np.isfinite(measured_accel))

        # Initialize filtered acceleration array
        filtered_accel = np.zeros((6, 1))
        
        # Filter linear acceleration (first 3 components)
        for i in range(3):
            # Apply elliptical SOS filter to each channel separately
            filtered_value, self._linear_filter_states[i] = sosfilt(
                self._linear_filter_sos,
                [measured_accel[i, 0]],  # Input as 1D array
                zi=self._linear_filter_states[i]
            )
            filtered_accel[i, 0] = filtered_value[0]
        
        # Filter angular acceleration (last 3 components)
        for i in range(3):
            # Apply elliptical SOS filter to each channel separately
            filtered_value, self._angular_filter_states[i] = sosfilt(
                self._angular_filter_sos,
                [measured_accel[i+3, 0]],  # Input as 1D array
                zi=self._angular_filter_states[i]
            )
            filtered_accel[i+3, 0] = filtered_value[0]
        
        # Update the filtered acceleration state
        if self._V_dI_B_filtered is None:
            # First measurement - initialize with filtered value
            self._V_dI_B_filtered = filtered_accel
        else:
            self._V_dI_B_filtered = filtered_accel
    
    @staticmethod
    def _estimate_angular_acceleration(current_nav_state: NavigationState, last_nav_state: Optional[NavigationState]) -> Optional[np.ndarray]:
        """
        Compute angular acceleration from angular velocity with finite differences
        """
        if last_nav_state is not None:
            curr_timestamp = Time.from_msg(current_nav_state.header.stamp)
            prev_timestamp = Time.from_msg(last_nav_state.header.stamp)
            dt = (curr_timestamp - prev_timestamp).to_sec()
            if dt <= 0.0:
                logging.warning("Controller: dt <= 0.0")
                return None
            omega_b = numpify(current_nav_state.body_twist.angular)
            omega_b_prev = numpify(last_nav_state.body_twist.angular)
            angular_accel = (omega_b - omega_b_prev) / 0.01
            return angular_accel
        else:
            logging.warning("Controller: last_nav_state is None")
            return None
    
    def _control_loop(self):
        """Main control loop - runs at fixed frequency"""
        # Check if we have required inputs
        if self._last_nav_state is None:
            self.get_logger().warn("Controller: Missing navigation state")
            self._publish_wrench(np.zeros((6, 1)))
            return
        elif self._V_dI_B_filtered is None:
            self._publish_wrench(np.zeros((6, 1)))
            return
        # elif self._cmd is None:
        #     self.get_logger().warn("Controller: Missing controller command")

        # Run outer loop (velocity)
        V_dI_B_target, velocity_error = self._get_target_acceleration()

        # Run inner loop (INDI)
        F_target, dF_target = self._get_target_wrench_with_indi(V_dI_B_target)

        self._F_target_prev = F_target.copy()
        
        self._publish_wrench(F_target)
        self._publish_debug_message(V_dI_B_target, velocity_error, dF_target)

    def _get_target_acceleration(self) -> tuple[np.ndarray, np.ndarray]:
        """Outer loop: compute acceleration command from a body twist command
        
        This is a simple proportional controller with optional additive feedforward.

        V_dI_B = K_p * (V_B_target - V_B_current) + V_dI_B_feedforward
        
        Returns:
            tuple: (target_acceleration, velocity_error)
        """
        # assert self._cmd is not None

        V_B_current = numpify(self._last_nav_state.body_twist)
        if self._cmd is None:
            V_B_target = np.c_[[0.0, 0.0, -0.1, 0.0, 0.0, 0.0]]
            V_dI_B_ff = np.zeros((6, 1))
        else:
            V_B_target = numpify(self._cmd.target_twist)
            V_dI_B_ff = np.vstack([
                numpify(self._cmd.feedforward_linear_accel), 
                numpify(self._cmd.feedforward_angular_accel)
            ])
        assert V_B_target.shape == (6, 1)
        assert V_B_current.shape == (6, 1)
        assert self.params.K_p.shape == (6, 1)

        velocity_error = V_B_target - V_B_current
        V_dI_B_fb = self.params.K_p * velocity_error
        assert V_dI_B_fb.shape == (6, 1)
        assert V_dI_B_ff.shape == (6, 1)
        V_dI_B_target = V_dI_B_fb + V_dI_B_ff
        return V_dI_B_target, velocity_error

    def _get_target_wrench_with_indi(self, V_dI_B_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inner loop: compute wrench command from acceleration command"""
        assert V_dI_B_target.shape == (6, 1)

        # Acceleration error
        V_dI_B_error = V_dI_B_target - self._V_dI_B_filtered
        assert np.all(np.isfinite(V_dI_B_target))
        assert np.all(np.isfinite(self._V_dI_B_filtered))

        dF_target = self.params.M @ V_dI_B_error
        assert np.all(np.isfinite(dF_target))
        F_target = self._F_target_prev + dF_target
        assert np.all(np.isfinite(F_target))

        # Apply control limits
        F_target, hit_limits = self._apply_limits(F_target)
        # if hit_limits:
        #     self.get_logger().warn("Controller: Hit limits")

        return F_target, dF_target
    
    def _apply_limits(self, wrench: np.ndarray) -> np.ndarray:
        """Apply force and torque limits to the wrench command"""
        limited_wrench = wrench.copy()
        hit_limits = False
        
        # Limit forces (first 3 elements)
        force_magnitude = np.linalg.norm(limited_wrench[0:3])
        if force_magnitude > self.params.max_force:
            limited_wrench[0:3] *= self.params.max_force / force_magnitude
            hit_limits = True
        
        # Limit torques (last 3 elements)
        torque_magnitude = np.linalg.norm(limited_wrench[3:6])
        if torque_magnitude > self.params.max_torque:
            limited_wrench[3:6] *= self.params.max_torque / torque_magnitude
            hit_limits = True
        
        return limited_wrench, hit_limits
    
    def _publish_wrench(self, wrench: np.ndarray, timestamp: Optional[Time] = None):
        """Publish the computed wrench command"""
        wrench_msg = WrenchStamped()
        if timestamp is None:
            timestamp = self.get_clock().now()
        wrench_msg.header = Header()
        wrench_msg.header.stamp = timestamp.to_msg()
        wrench_msg.header.frame_id = self._frame_id
        wrench_msg.wrench = msgify(wrench, message_type="Wrench")
        
        self._wrench_pub.publish(wrench_msg)
    
    def _publish_debug_message(self, V_dI_B_target: np.ndarray, velocity_error: np.ndarray, dF_target: np.ndarray, timestamp: Optional[Time] = None):
        """Publish debug information from the controller"""
        debug_msg = ControllerDebug()
        
        if timestamp is None:
            timestamp = self.get_clock().now()
        debug_msg.header = Header()
        debug_msg.header.stamp = timestamp.to_msg()
        debug_msg.header.frame_id = self._frame_id
        
        # Velocity errors (outer loop)
        debug_msg.linear_velocity_error = msgify(velocity_error[0:3], message_type="Vector3")
        debug_msg.angular_velocity_error = msgify(velocity_error[3:6], message_type="Vector3")
        
        # Filtered accelerations (measured)
        debug_msg.filtered_linear_acceleration = msgify(self._V_dI_B_filtered[0:3], message_type="Vector3")
        debug_msg.filtered_angular_acceleration = msgify(self._V_dI_B_filtered[3:6], message_type="Vector3")
        
        # Desired accelerations (from outer loop)
        debug_msg.desired_linear_acceleration = msgify(V_dI_B_target[0:3], message_type="Vector3")
        debug_msg.desired_angular_acceleration = msgify(V_dI_B_target[3:6], message_type="Vector3")
        
        # Acceleration errors (inner loop)
        V_dI_B_error = V_dI_B_target - self._V_dI_B_filtered
        debug_msg.linear_acceleration_error = msgify(V_dI_B_error[0:3], message_type="Vector3")
        debug_msg.angular_acceleration_error = msgify(V_dI_B_error[3:6], message_type="Vector3")
        
        # Control output (wrench increment from INDI)
        debug_msg.wrench_increment = msgify(dF_target, message_type="Wrench")
        
        self._debug_pub.publish(debug_msg)


def main():
    """Main entry point"""
    rclpy.init()
    
    try:
        controller = Controller()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        if 'controller' in locals():
            controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


