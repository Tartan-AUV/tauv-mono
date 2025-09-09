"""
Thruster controller that converts thrust commands to angular velocity setpoints.
Publishes ThrusterSetpoint messages for the depth_actuators_i2c node.
"""

import numpy as np
import rclpy
from rclpy.node import Node

from tauv_msgs.msg import TargetThrust, ThrusterSetpoint

# Thruster model parameters
THRUST_COEFF_FWD = 3.645e-3  # N/(rad/s)^2
THRUST_COEFF_REV = 2.905e-3  # N/(rad/s)^2
MAX_SETPOINT = 362.9  # rad/s




class ThrusterController(Node):
    """
    ROS2 node that converts thrust commands to angular velocity setpoints.
    """
    
    def __init__(self):
        super().__init__("thruster_controller")
        
        # Get number of thrusters from parameter (default to 8)
        self.declare_parameter('num_thrusters', 8)
        self._num_thrusters = self.get_parameter('num_thrusters').value
        
        self.get_logger().info(f"Thruster controller initialized for {self._num_thrusters} thrusters")
        
        # Subscribe to target thrust commands
        self._target_thrust_sub = self.create_subscription(
            TargetThrust, 
            "target_thrust", 
            self._handle_target_thrust, 
            10
        )
        
        # Publisher for thruster setpoints
        self._thruster_setpoint_pub = self.create_publisher(
            ThrusterSetpoint,
            "thruster_setpoint",
            10
        )

    def _thrust_to_omega(self, thrust_n: float) -> float:
        """
        Convert thrust in Newtons to angular velocity in rad/s.
        
        Args:
            thrust_n: Thrust in Newtons (positive = forward, negative = reverse)
            
        Returns:
            Angular velocity in rad/s
        """
        if abs(thrust_n) < 1e-6:  # Near zero thrust
            return 0.0
            
        # Use appropriate thrust coefficient based on direction
        thrust_coeff = THRUST_COEFF_FWD if thrust_n > 0 else THRUST_COEFF_REV
        
        # Calculate required angular velocity
        # Thrust = thrust_coeff * omega^2
        # omega = sqrt(|thrust| / thrust_coeff) * sign(thrust)
        omega_radps = np.sqrt(min(abs(thrust_n) / thrust_coeff, MAX_SETPOINT**2)) * np.sign(thrust_n)
        
        # Clamp to valid range
        return np.clip(omega_radps, -MAX_SETPOINT, MAX_SETPOINT)

    def _handle_target_thrust(self, msg: TargetThrust):
        """Handle incoming thrust commands and publish angular velocity setpoints."""
        thrusts = msg.target_thrust
        
        # Ensure we don't try to control more thrusters than initialized
        num_to_control = min(len(thrusts), self._num_thrusters)
        
        if len(thrusts) > self._num_thrusters:
            self.get_logger().warn(
                f"Received {len(thrusts)} thrust commands but only {self._num_thrusters} thrusters configured"
            )
        
        # Create thruster setpoint message
        setpoint_msg = ThrusterSetpoint()
        
        # Convert thrust to omega for each thruster
        omega_values = []
        enables = []
        
        for i in range(self._num_thrusters):
            if i < num_to_control:
                omega = self._thrust_to_omega(thrusts[i])
                omega_values.append(omega)
                # Enable thruster if thrust is non-zero
                enables.append(1 if abs(thrusts[i]) > 1e-6 else 0)
            else:
                # Set remaining thrusters to zero/disabled
                omega_values.append(0.0)
                enables.append(0)
        
        setpoint_msg.omega_radps = omega_values
        setpoint_msg.enables = enables
        
        # Publish the setpoint
        self._thruster_setpoint_pub.publish(setpoint_msg)

    def destroy_node(self):
        """Publish neutral setpoints before shutting down."""
        self.get_logger().info("Shutting down thruster controller, publishing neutral setpoints")
        
        # Create neutral setpoint message
        setpoint_msg = ThrusterSetpoint()
        setpoint_msg.omega_radps = [0.0] * self._num_thrusters
        setpoint_msg.enables = [0] * self._num_thrusters
        
        # Publish neutral setpoint
        self._thruster_setpoint_pub.publish(setpoint_msg)
        
        super().destroy_node()


def main():
    rclpy.init()
    node = ThrusterController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
