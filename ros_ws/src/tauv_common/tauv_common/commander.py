"""
Commander Module for AUV Velocity and Attitude Control

This module implements a high-level commander that accepts velocity and attitude 
commands and generates acceleration commands for the INDI controller. It serves 
as the outer control loop in a cascaded control architecture.

The commander implements:
- Proportional velocity control for linear acceleration commands
- Quaternion-based attitude control for angular acceleration commands
- Feedforward acceleration support
- Individual enable flags for velocity and attitude control
"""

import rclpy
from rclpy.node import Node
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from dataclasses import dataclass

from geometry_msgs.msg import AccelStamped, Vector3, Accel, Quaternion
from tauv_msgs.msg import NavigationState, VelocityAttitudeCommand
from tauv_common.util.geometry import numpify
from spatialmath import SO3, UnitQuaternion
from std_msgs.msg import Header

@dataclass
class CommanderParams:
    """Parameters for the commander control laws"""
    
    # Velocity control gains
    kp_velocity: float = 2.0        # Proportional gain for velocity error [1/s]
    kd_velocity: float = 0.1        # Derivative gain for velocity control [1]
    
    # Attitude control gains  
    kp_attitude: float = 1.5        # Proportional gain for attitude error [1/s²]
    kd_attitude: float = 0.3        # Derivative gain for attitude control [1/s]
    
    # Control limits
    max_linear_accel: float = 2.0   # Maximum linear acceleration command [m/s²]
    max_angular_accel: float = 1.0  # Maximum angular acceleration command [rad/s²]
    
    # Velocity filtering (for derivative estimation)
    velocity_filter_alpha: float = 0.9  # Low-pass filter for velocity measurements
    
    @classmethod
    def default(cls) -> 'CommanderParams':
        """Create default commander parameters"""
        return cls()

class Commander(Node):
    """
    AUV Commander Node
    
    This node implements velocity and attitude control by generating acceleration
    commands for the INDI controller. It operates as the outer loop in a cascaded
    control system.
    
    Subscribes to:
    - /gnc/velocity_attitude_command (tauv_msgs/VelocityAttitudeCommand): Target velocity and attitude
    - /gnc/navigation_state (tauv_msgs/NavigationState): Current vehicle state
    
    Publishes to:
    - /gnc/acceleration_command (geometry_msgs/AccelStamped): Acceleration commands for INDI controller
    """
    
    def __init__(self):
        super().__init__("commander")
        
        # Parameters
        self.params = CommanderParams.default()
        
        # State variables
        self._nav_state: Optional[NavigationState] = None
        self._velocity_attitude_cmd: Optional[VelocityAttitudeCommand] = None
        self._previous_velocity: Optional[NDArray] = None  # For derivative estimation
        self._previous_angular_velocity: Optional[NDArray] = None
        self._filtered_velocity: Optional[NDArray] = None
        self._filtered_angular_velocity: Optional[NDArray] = None
        
        # ROS2 interfaces
        self._cmd_sub = self.create_subscription(
            VelocityAttitudeCommand, 'gnc/velocity_attitude_command', 
            self._handle_velocity_attitude_command, 10
        )
        
        self._nav_state_sub = self.create_subscription(
            NavigationState, 'gnc/navigation_state', 
            self._handle_nav_state, 10
        )
        
        self._accel_cmd_pub = self.create_publisher(
            AccelStamped, 'gnc/acceleration_command', 10
        )
        
        # Control loop timer (50 Hz)
        self._control_timer = self.create_timer(0.02, self._control_loop)
        
        self.get_logger().info("Commander initialized")
    
    def _handle_velocity_attitude_command(self, msg: VelocityAttitudeCommand):
        """Handle incoming velocity and attitude command"""
        self._velocity_attitude_cmd = msg
        
    def _handle_nav_state(self, msg: NavigationState):
        """Handle incoming navigation state and update filtered estimates"""
        self._nav_state = msg
        
        # Extract current velocities
        current_velocity = numpify(msg.v_b)  # Linear velocity in body frame
        current_angular_velocity = numpify(msg.omega_b)  # Angular velocity in body frame
        
        # Apply filtering for smoother derivative estimation
        alpha = self.params.velocity_filter_alpha
        
        if self._filtered_velocity is None:
            self._filtered_velocity = current_velocity
            self._filtered_angular_velocity = current_angular_velocity
        else:
            self._filtered_velocity = alpha * self._filtered_velocity + (1 - alpha) * current_velocity
            self._filtered_angular_velocity = alpha * self._filtered_angular_velocity + (1 - alpha) * current_angular_velocity
    
    def _control_loop(self):
        """Main control loop - generates acceleration commands"""
        # Check if we have required inputs
        if self._nav_state is None or self._velocity_attitude_cmd is None:
            return
        
        cmd = self._velocity_attitude_cmd
        nav = self._nav_state
        
        # Initialize acceleration command
        linear_accel_cmd = np.zeros((3, 1))
        angular_accel_cmd = np.zeros((3, 1))
        
        # Velocity control
        if cmd.velocity_control_enabled and self._filtered_velocity is not None:
            linear_accel_cmd = self._compute_velocity_control(cmd, nav)
        
        # Attitude control  
        if cmd.attitude_control_enabled and self._filtered_angular_velocity is not None:
            angular_accel_cmd = self._compute_attitude_control(cmd, nav)
        
        # Add feedforward acceleration if provided
        if cmd.velocity_control_enabled:
            feedforward_accel = numpify(cmd.feedforward_acceleration)
            linear_accel_cmd += feedforward_accel
        
        # Apply limits
        linear_accel_cmd = self._limit_acceleration(linear_accel_cmd, self.params.max_linear_accel)
        angular_accel_cmd = self._limit_acceleration(angular_accel_cmd, self.params.max_angular_accel)
        
        # Publish acceleration command
        self._publish_acceleration_command(linear_accel_cmd, angular_accel_cmd)
        
        # Debug logging
        if cmd.velocity_control_enabled:
            vel_error = numpify(cmd.target_velocity) - self._filtered_velocity
            self.get_logger().debug(
                f"Velocity error: [{vel_error[0,0]:.3f}, {vel_error[1,0]:.3f}, {vel_error[2,0]:.3f}] m/s"
            )
        
        self.get_logger().debug(
            f"Accel cmd: linear=[{linear_accel_cmd[0,0]:.2f}, {linear_accel_cmd[1,0]:.2f}, {linear_accel_cmd[2,0]:.2f}] m/s², "
            f"angular=[{angular_accel_cmd[0,0]:.2f}, {angular_accel_cmd[1,0]:.2f}, {angular_accel_cmd[2,0]:.2f}] rad/s²"
        )
    
    def _compute_velocity_control(self, cmd: VelocityAttitudeCommand, nav: NavigationState) -> NDArray:
        """
        Compute linear acceleration command from velocity error
        
        Uses proportional-derivative control:
        a_cmd = Kp * (v_desired - v_current) + Kd * (a_desired - a_current)
        
        Where a_desired is approximated by differentiating v_desired
        """
        # Compute velocity error in body frame
        target_velocity = numpify(cmd.target_velocity)
        velocity_error = target_velocity - self._filtered_velocity
        
        # Proportional term
        accel_cmd = self.params.kp_velocity * velocity_error
        
        # Derivative term (if we have previous velocity measurements)
        if self._previous_velocity is not None:
            dt = 0.02  # Control loop period
            
            # Estimate acceleration from velocity derivative
            current_accel_estimate = (self._filtered_velocity - self._previous_velocity) / dt
            
            # For simplicity, assume zero desired acceleration
            # (In practice, you might differentiate the velocity command)
            desired_accel = np.zeros_like(current_accel_estimate)
            accel_error = desired_accel - current_accel_estimate
            
            accel_cmd += self.params.kd_velocity * accel_error
        
        # Store for next iteration
        self._previous_velocity = self._filtered_velocity.copy()
        
        return accel_cmd
    
    def _compute_attitude_control(self, cmd: VelocityAttitudeCommand, nav: NavigationState) -> NDArray:
        """
        Compute angular acceleration command from attitude error
        
        Uses quaternion-based attitude control with proportional-derivative structure.
        The attitude error is computed using quaternion multiplication and converted
        to a rotation vector for control purposes.
        """
        # Get current and desired orientations as quaternions
        current_quat = numpify(nav.body_pose.orientation)  # UnitQuaternion
        target_quat = numpify(cmd.target_attitude)         # UnitQuaternion
        
        assert isinstance(current_quat, UnitQuaternion)
        assert isinstance(target_quat, UnitQuaternion)
        
        # Compute attitude error quaternion: q_error = q_target * q_current^(-1)
        # This gives the rotation needed to go from current to target orientation
        error_quat = target_quat * current_quat.inv()
        
        # Convert to rotation vector (axis-angle representation)
        # For small angles, this approximates the angular error
        error_rotation_vector = error_quat.log().vec.reshape((3, 1))
        
        # Proportional term
        angular_accel_cmd = self.params.kp_attitude * error_rotation_vector
        
        # Derivative term (angular velocity error)
        # Desired angular velocity is typically zero for regulation
        desired_angular_velocity = np.zeros((3, 1))
        angular_velocity_error = desired_angular_velocity - self._filtered_angular_velocity
        angular_accel_cmd += self.params.kd_attitude * angular_velocity_error
        
        return angular_accel_cmd
    
    def _limit_acceleration(self, acceleration: NDArray, max_magnitude: float) -> NDArray:
        """Apply magnitude limits to acceleration commands"""
        magnitude = np.linalg.norm(acceleration)
        if magnitude > max_magnitude:
            return acceleration * (max_magnitude / magnitude)
        return acceleration
    
    def _publish_acceleration_command(self, linear_accel: NDArray, angular_accel: NDArray):
        """Publish the computed acceleration command"""
        accel_msg = AccelStamped()
        accel_msg.header = Header()
        accel_msg.header.stamp = self.get_clock().now().to_msg()
        accel_msg.header.frame_id = "os/body"  # Body frame
        
        # Convert numpy arrays to Vector3 messages
        accel_msg.accel.linear.x = float(linear_accel[0, 0])
        accel_msg.accel.linear.y = float(linear_accel[1, 0])
        accel_msg.accel.linear.z = float(linear_accel[2, 0])
        
        accel_msg.accel.angular.x = float(angular_accel[0, 0])
        accel_msg.accel.angular.y = float(angular_accel[1, 0])
        accel_msg.accel.angular.z = float(angular_accel[2, 0])
        
        self._accel_cmd_pub.publish(accel_msg)


def main():
    """Main entry point"""
    rclpy.init()
    
    try:
        commander = Commander()
        rclpy.spin(commander)
    except KeyboardInterrupt:
        pass
    finally:
        if 'commander' in locals():
            commander.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 