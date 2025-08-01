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
from numpy.typing import NDArray
from typing import Optional
from dataclasses import dataclass

from geometry_msgs.msg import WrenchStamped, Wrench, Vector3, Accel
from tauv_msgs.msg import NavigationState, ControllerDebug, ControllerCommand
from tauv_common.util.geometry import numpify, msgify
from tauv_common.util.time import Time
from spatialmath import SO3, SE3, UnitQuaternion
from std_msgs.msg import Header

@dataclass
class INDIParams:
    """Parameters for the INDI controller"""

    # Mass matrix
    M: NDArray  # 6x6 matrix
    
    # Proportional gain for outer loop
    K_p: NDArray

    # Control limits
    max_force: float = 1000.0   # Maximum force in N
    max_torque: float = 500.0   # Maximum torque in N⋅m

    # Filtering parameters for acceleration measurements
    accel_filter_alpha: float = 0.3  # Low-pass filter coefficient (0 = no filter, 1 = full filter)

    @classmethod
    def default(cls) -> 'INDIParams':
        """Create default INDI parameters with a simple diagonal control effectiveness matrix"""
        # Simple diagonal matrix - assumes direct relationship between forces/torques and accelerations
        # This is a rough approximation that should be identified from system data
        mass = np.diag([23.0, 23.0, 23.0])  # kg - estimated vehicle mass
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
        self._F_target_prev: NDArray = np.zeros((6, 1)) # Previous control input (6x1)
        self._V_dI_B_filtered: Optional[NDArray] = None   # Filtered acceleration measurements (6x1)
        self._last_nav_state: Optional[NavigationState] = None
        self._odom_T_body_latched: Optional[SE3] = None
        self._cmd: Optional[ControllerCommand] = None
        
        # ROS2 interfaces
        self._cmd_sub = self.create_subscription(
            ControllerCommand, 'gnc/controller_command', self._handle_cmd, 10
        )
        
        self._last_nav_state_sub = self.create_subscription(
            NavigationState, 'gnc/navigation_state', self._handle_nav_state, 10
        )
        
        self._wrench_pub = self.create_publisher(
            WrenchStamped, 'gnc/target_wrench', 10
        )
        
        self._debug_pub = self.create_publisher(
            ControllerDebug, 'gnc/controller_debug', 10
        )
        
        # Control loop timer (50 Hz)
        self._control_timer = self.create_timer(0.02, self._control_loop)
        
        self.get_logger().info("Controller initialized")
    
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
        
    def _update_filtered_acceleration(self, measured_accel: NDArray):
        assert measured_accel.shape == (6, 1)
        assert np.all(np.isfinite(measured_accel))

        if self._V_dI_B_filtered is None:
            self._V_dI_B_filtered = measured_accel
        else:
            alpha = self.params.accel_filter_alpha
            self._V_dI_B_filtered = alpha * measured_accel + (1 - alpha) * self._V_dI_B_filtered
    
    @staticmethod
    def _estimate_angular_acceleration(current_nav_state: NavigationState, last_nav_state: Optional[NavigationState]) -> Optional[NDArray]:
        """
        Compute angular acceleration from angular velocity with finite differences
        """
        if last_nav_state is not None:
            curr_timestamp = Time.from_msg(current_nav_state.header.stamp)
            prev_timestamp = Time.from_msg(last_nav_state.header.stamp)
            dt = (curr_timestamp - prev_timestamp).to_sec()
            if dt <= 0.0:
                return None
            omega_b = numpify(current_nav_state.omega_b)
            omega_b_prev = numpify(last_nav_state.omega_b)
            angular_accel = (omega_b - omega_b_prev) / dt
            return angular_accel
        else:
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
        elif self._cmd is None:
            self.get_logger().warn("Controller: Missing controller command")
            self._publish_wrench(np.zeros((6, 1)))
            return

        # Run outer loop (velocity)
        V_dI_B_target, velocity_error = self._get_target_acceleration()

        # Run inner loop (INDI)
        F_target, dF_target = self._get_target_wrench_with_indi(V_dI_B_target)

        self._F_target_prev = F_target.copy()
        
        self._publish_wrench(F_target)
        self._publish_debug_message(V_dI_B_target, velocity_error, dF_target)

    def _get_target_acceleration(self) -> tuple[NDArray, NDArray]:
        """Outer loop: compute acceleration command from a body twist command
        
        This is a simple proportional controller with optional additive feedforward.

        V_dI_B = K_p * (V_B_target - V_B_current) + V_dI_B_feedforward
        
        Returns:
            tuple: (target_acceleration, velocity_error)
        """
        assert self._cmd is not None

        V_B_target = numpify(self._cmd.target_twist)
        V_B_current = numpify(self._last_nav_state.body_twist)
        assert V_B_target.shape == (6, 1)
        assert V_B_current.shape == (6, 1)
        assert self.params.K_p.shape == (6, 1)

        velocity_error = V_B_target - V_B_current
        V_dI_B_fb = self.params.K_p * velocity_error
        assert V_dI_B_fb.shape == (6, 1)
        V_dI_B_ff = np.vstack([
            numpify(self._cmd.feedforward_linear_accel), 
            numpify(self._cmd.feedforward_angular_accel)
        ])
        assert V_dI_B_ff.shape == (6, 1)
        V_dI_B_target = V_dI_B_fb + V_dI_B_ff
        return V_dI_B_target, velocity_error

    def _get_target_wrench_with_indi(self, V_dI_B_target: NDArray) -> tuple[NDArray, NDArray]:
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
        if hit_limits:
            self.get_logger().warn("Controller: Hit limits")

        return F_target, dF_target
    
    def _apply_limits(self, wrench: NDArray) -> NDArray:
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
    
    def _publish_wrench(self, wrench: NDArray, timestamp: Optional[Time] = None):
        """Publish the computed wrench command"""
        wrench_msg = WrenchStamped()
        if timestamp is None:
            timestamp = self.get_clock().now()
        wrench_msg.header = Header()
        wrench_msg.header.stamp = timestamp.to_msg()
        wrench_msg.header.frame_id = self._frame_id
        wrench_msg.wrench = msgify(wrench, message_type="Wrench")
        
        self._wrench_pub.publish(wrench_msg)
    
    def _publish_debug_message(self, V_dI_B_target: NDArray, velocity_error: NDArray, dF_target: NDArray, timestamp: Optional[Time] = None):
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


