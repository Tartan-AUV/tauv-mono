"""
INDI (Incremental Nonlinear Dynamic Inversion) Controller for AUV

This controller implements INDI control for an autonomous underwater vehicle.
It takes acceleration commands and current state as inputs and produces 
wrench commands for the thruster allocation system.
"""

import rclpy
from rclpy.node import Node
from rclpy.subscription import Subscription
from rclpy.publisher import Publisher
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from dataclasses import dataclass

from geometry_msgs.msg import AccelStamped, WrenchStamped, Wrench, Vector3, Accel
from tauv_msgs.msg import NavigationState
from tauv_common.util.geometry import numpify
from spatialmath import SO3, UnitQuaternion
from std_msgs.msg import Header

@dataclass
class INDIParams:
    """Parameters for the INDI controller"""
    # Control effectiveness matrix - relates wrench inputs to accelerations
    # For a 6DOF system: [fx, fy, fz, tx, ty, tz] -> [ax, ay, az, alphax, alphay, alphaz]
    control_effectiveness: NDArray  # 6x6 matrix
    
    # Filtering parameters for acceleration measurements
    accel_filter_alpha: float = 0.8  # Low-pass filter coefficient (0 = no filter, 1 = full filter)
    
    # Control limits
    max_force: float = 100.0   # Maximum force in N
    max_torque: float = 50.0   # Maximum torque in N⋅m
    
    # Controller gains
    kp_pos: float = 1.0    # Proportional gain for position control
    kd_pos: float = 0.5    # Derivative gain for position control

    @classmethod
    def default(cls) -> 'INDIParams':
        """Create default INDI parameters with a simple diagonal control effectiveness matrix"""
        # Simple diagonal matrix - assumes direct relationship between forces/torques and accelerations
        # This is a rough approximation that should be identified from system data
        mass = 50.0  # kg - estimated vehicle mass
        inertia = np.diag([10.0, 15.0, 12.0])  # kg⋅m² - estimated moments of inertia
        
        G = np.zeros((6, 6))
        G[0:3, 0:3] = np.eye(3) / mass  # Force to linear acceleration
        G[3:6, 3:6] = np.linalg.inv(inertia)  # Torque to angular acceleration
        
        return cls(control_effectiveness=G)

class INDIController(Node):
    """
    INDI Controller Node
    
    Subscribes to:
    - /gnc/acceleration_command (geometry_msgs/AccelStamped): Desired acceleration
    - /gnc/navigation_state (tauv_msgs/NavigationState): Current vehicle state
    
    Publishes to:
    - /gnc/target_wrench (geometry_msgs/WrenchStamped): Control wrench for thruster allocation
    """
    
    def __init__(self):
        super().__init__("indi_controller")
        
        # Parameters
        self.params = INDIParams.default()
        
        # State variables
        self._previous_wrench: Optional[NDArray] = None  # Previous control input (6x1)
        self._filtered_accel: Optional[NDArray] = None   # Filtered acceleration measurements (6x1)
        self._nav_state: Optional[NavigationState] = None
        self._accel_command: Optional[AccelStamped] = None
        
        # Control effectiveness matrix inverse (for computational efficiency)
        try:
            self._G_inv = np.linalg.inv(self.params.control_effectiveness)
        except np.linalg.LinAlgError:
            self.get_logger().error("Control effectiveness matrix is singular! Using pseudo-inverse.")
            self._G_inv = np.linalg.pinv(self.params.control_effectiveness)
        
        # ROS2 interfaces
        self._accel_cmd_sub = self.create_subscription(
            AccelStamped, 'gnc/acceleration_command', self._handle_accel_command, 10
        )
        
        self._nav_state_sub = self.create_subscription(
            NavigationState, 'gnc/navigation_state', self._handle_nav_state, 10
        )
        
        self._wrench_pub = self.create_publisher(
            WrenchStamped, 'gnc/target_wrench', 10
        )
        
        # Control loop timer (50 Hz)
        self._control_timer = self.create_timer(0.02, self._control_loop)
        
        self.get_logger().info("INDI Controller initialized")
    
    def _handle_accel_command(self, msg: AccelStamped):
        """Handle incoming acceleration command"""
        self._accel_command = msg
        
    def _handle_nav_state(self, msg: NavigationState):
        """Handle incoming navigation state"""
        self._nav_state = msg
        
        # Extract and filter acceleration measurements
        # Convert from geometry_msgs to numpy arrays
        linear_accel = numpify(msg.a_b)  # Linear acceleration in body frame
        angular_accel = self._compute_angular_acceleration(msg)  # Angular acceleration
        
        # Combine into 6DOF acceleration vector
        measured_accel = np.vstack([linear_accel, angular_accel])
        
        # Apply low-pass filter
        if self._filtered_accel is None:
            self._filtered_accel = measured_accel
        else:
            alpha = self.params.accel_filter_alpha
            self._filtered_accel = alpha * self._filtered_accel + (1 - alpha) * measured_accel
    
    def _compute_angular_acceleration(self, nav_state: NavigationState) -> NDArray:
        """
        Compute angular acceleration from angular velocity
        
        This is a simple numerical differentiation. In practice, you might want to use
        a more sophisticated approach or get this directly from IMU if available.
        """
        # For now, we'll assume zero angular acceleration as a placeholder
        # In a real implementation, you'd differentiate omega_b or get it from sensors
        return np.zeros((3, 1))
    
    def _control_loop(self):
        """Main control loop - runs at fixed frequency"""
        # Check if we have required inputs
        if self._accel_command is None or self._nav_state is None or self._filtered_accel is None:
            return
        
        # Extract desired acceleration from command
        accel_cmd = self._accel_command.accel
        desired_linear_accel = numpify(accel_cmd.linear)
        desired_angular_accel = numpify(accel_cmd.angular)
        desired_accel = np.vstack([desired_linear_accel, desired_angular_accel])
        
        # INDI control law
        if self._previous_wrench is None:
            # Initialize with zero wrench
            self._previous_wrench = np.zeros((6, 1))
        
        # Compute acceleration error
        accel_error = desired_accel - self._filtered_accel
        
        # INDI update: u_new = u_prev + G^(-1) * (a_desired - a_measured)
        wrench_increment = self._G_inv @ accel_error
        new_wrench = self._previous_wrench + wrench_increment
        
        # Apply control limits
        new_wrench = self._apply_limits(new_wrench)
        
        # Store for next iteration
        self._previous_wrench = new_wrench.copy()
        
        # Publish wrench command
        self._publish_wrench(new_wrench)
        
        # Debug logging
        self.get_logger().debug(
            f"Accel error: [{accel_error[0,0]:.3f}, {accel_error[1,0]:.3f}, {accel_error[2,0]:.3f}] m/s²"
        )
        self.get_logger().debug(
            f"Wrench: F=[{new_wrench[0,0]:.1f}, {new_wrench[1,0]:.1f}, {new_wrench[2,0]:.1f}] N, "
            f"T=[{new_wrench[3,0]:.1f}, {new_wrench[4,0]:.1f}, {new_wrench[5,0]:.1f}] N⋅m"
        )
    
    def _apply_limits(self, wrench: NDArray) -> NDArray:
        """Apply force and torque limits to the wrench command"""
        limited_wrench = wrench.copy()
        
        # Limit forces (first 3 elements)
        force_magnitude = np.linalg.norm(limited_wrench[0:3])
        if force_magnitude > self.params.max_force:
            limited_wrench[0:3] *= self.params.max_force / force_magnitude
        
        # Limit torques (last 3 elements)
        torque_magnitude = np.linalg.norm(limited_wrench[3:6])
        if torque_magnitude > self.params.max_torque:
            limited_wrench[3:6] *= self.params.max_torque / torque_magnitude
        
        return limited_wrench
    
    def _publish_wrench(self, wrench: NDArray):
        """Publish the computed wrench command"""
        wrench_msg = WrenchStamped()
        wrench_msg.header = Header()
        wrench_msg.header.stamp = self.get_clock().now().to_msg()
        wrench_msg.header.frame_id = "os/body"  # Body frame
        
        # Convert numpy array to Wrench message
        wrench_msg.wrench.force.x = float(wrench[0, 0])
        wrench_msg.wrench.force.y = float(wrench[1, 0])
        wrench_msg.wrench.force.z = float(wrench[2, 0])
        wrench_msg.wrench.torque.x = float(wrench[3, 0])
        wrench_msg.wrench.torque.y = float(wrench[4, 0])
        wrench_msg.wrench.torque.z = float(wrench[5, 0])
        
        self._wrench_pub.publish(wrench_msg)


def main():
    """Main entry point"""
    rclpy.init()
    
    try:
        controller = INDIController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        if 'controller' in locals():
            controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


