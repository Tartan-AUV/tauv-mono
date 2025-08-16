"""
PID Position Hold Controller for TartanAUV vehicles.

This controller implements cascaded PID control loops to hold the vehicle
at a fixed pose:
- Outer loop: Position control (generates velocity commands)
- Inner loop: Velocity control (generates wrench commands)

For rotation control, Kd is ignored since angular acceleration is not available.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from spatialmath import SE3, SO3, UnitQuaternion

from geometry_msgs.msg import WrenchStamped, Pose, Twist, Vector3
from tauv_msgs.msg import NavigationState
from std_msgs.msg import Header
from tauv_common.util.geometry import numpify, msgify

# ---------------------------------------------------------------------------
# PID Controller Parameters
# ---------------------------------------------------------------------------

class PIDGains:
    """PID gains for position and velocity control."""
    
    def __init__(self):
        # Position control gains (outer loop)
        # These generate velocity commands from position errors
        self.position_kp = np.array([0.3, 0.3, 0.3])  # x, y, z
        self.position_ki = np.array([0.01, 0.01, 0.02])
        self.position_kd = np.array([0.1, 0.1, 0.15])
        
        # Orientation control gains (outer loop)
        # These generate angular velocity commands from orientation errors
        self.orientation_kp = np.array([0.3, 0.3, 0.3])  # roll, pitch, yaw
        self.orientation_ki = np.array([0.0, 0.0, 0.0))
        # No Kd for orientation as we don't have angular acceleration
        
        # Velocity control gains (inner loop)
        # These generate forces from velocity errors
        self.velocity_kp = np.array([100.0, 100.0, 100.0])  # x, y, z
        self.velocity_ki = np.array([0.0, 0.0, 0.0])
        self.velocity_kd = np.array([10.0, 10.0, 10.0])
        
        # Angular velocity control gains (inner loop)
        # These generate torques from angular velocity errors
        self.angular_velocity_kp = np.array([50.0, 50.0, 50.0])  # roll, pitch, yaw
        self.angular_velocity_ki = np.array([0.0, 0.0, 0.0])
        # No Kd for angular velocity as we don't have angular acceleration
        
        # Control limits
        self.max_linear_velocity = 1.0  # m/s
        self.max_angular_velocity = 0.5  # rad/s
        self.max_force = 100.0  # N
        self.max_torque = 50.0  # N⋅m
        
        # Integral windup limits
        self.position_integral_limit = 0.5
        self.orientation_integral_limit = 0.2
        self.velocity_integral_limit = 20.0
        self.angular_velocity_integral_limit = 5.0

# ---------------------------------------------------------------------------
# Controller Node
# ---------------------------------------------------------------------------

class Controller(Node):
    """
    PID Position Hold Controller Node.
    
    Subscribes to:
    - NavigationState: Current vehicle state (pose, twist, accelerations)
    
    Publishes to:
    - target_wrench: Wrench command for thruster allocation
    """
    
    def __init__(self) -> None:
        super().__init__("controller")
        
        # ------------------------------------------------------------------
        # Target pose (hardcoded)
        # ------------------------------------------------------------------
        self.target_position = np.array([0.0, 0.0, 1.0])  # x, y, z in meters
        # Identity quaternion - vehicle level and facing forward
        self.target_orientation = UnitQuaternion()  # Identity quaternion (w=1, x=y=z=0)
        
        # ------------------------------------------------------------------
        # PID gains and parameters
        # ------------------------------------------------------------------
        self.gains = PIDGains()
        
        # ------------------------------------------------------------------
        # PID state variables
        # ------------------------------------------------------------------
        # Position control
        self.position_integral = np.zeros(3)
        self.last_position_error = np.zeros(3)
        
        # Orientation control
        self.orientation_integral = np.zeros(3)
        self.last_orientation_error = np.zeros(3)
        
        # Velocity control
        self.velocity_integral = np.zeros(3)
        self.last_velocity_error = np.zeros(3)
        self.last_linear_acceleration = np.zeros(3)
        
        # Angular velocity control
        self.angular_velocity_integral = np.zeros(3)
        self.last_angular_velocity_error = np.zeros(3)
        
        # Timing
        self.last_control_time = None
        
        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------
        self._nav_state_sub: Subscription = self.create_subscription(
            NavigationState,
            "navigation_state",
            self._navigation_callback,
            10,
        )
        
        self._wrench_pub: Publisher = self.create_publisher(
            WrenchStamped,
            "target_wrench",
            10,
        )
        
        # Control loop timer at 50 Hz
        self._control_timer = self.create_timer(0.02, self._control_loop)
        
        self.get_logger().info(
            f"PID Controller initialized - Target pose: "
            f"position=({self.target_position[0]:.2f}, {self.target_position[1]:.2f}, {self.target_position[2]:.2f}), "
            f"orientation=(w={self.target_orientation.s:.2f}, x={self.target_orientation.vec[0]:.2f}, "
            f"y={self.target_orientation.vec[1]:.2f}, z={self.target_orientation.vec[2]:.2f})"
        )
        
        # Store latest navigation state
        self.current_nav_state = None
    
    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    
    def _navigation_callback(self, msg: NavigationState) -> None:
        """Store the latest navigation state."""
        self.current_nav_state = msg
    
    def _control_loop(self) -> None:
        """Main control loop - runs at 50 Hz."""
        if self.current_nav_state is None:
            return
        
        # Calculate dt
        current_time = self.get_clock().now()
        if self.last_control_time is None:
            dt = 0.02  # Assume first iteration is at expected rate
        else:
            dt = (current_time - self.last_control_time).nanoseconds / 1e9
            if dt <= 0:
                return
        self.last_control_time = current_time
        
        # Extract current state using numpify
        current_pose = numpify(self.current_nav_state.body_pose)  # SE3
        current_position = current_pose.t  # Translation vector
        current_orientation = current_pose.UnitQuaternion()  # Rotation as quaternion
        
        # Extract velocities and accelerations
        current_twist = numpify(self.current_nav_state.body_twist)  # 6x1 array
        current_linear_velocity = current_twist[0:3, 0]  # First 3 elements
        current_angular_velocity = current_twist[3:6, 0]  # Last 3 elements
        
        current_linear_acceleration = numpify(self.current_nav_state.a_b).flatten()
        
        # ------------------------------------------------------------------
        # Outer Loop: Position to Velocity
        # ------------------------------------------------------------------
        target_linear_velocity = self._position_pid(
            current_position, self.target_position, dt
        )
        
        target_angular_velocity = self._orientation_pid(
            current_orientation, self.target_orientation, dt
        )
        
        # ------------------------------------------------------------------
        # Inner Loop: Velocity to Wrench
        # ------------------------------------------------------------------
        force = self._velocity_pid(
            current_linear_velocity, target_linear_velocity, 
            current_linear_acceleration, dt
        )
        
        torque = self._angular_velocity_pid(
            current_angular_velocity, target_angular_velocity, dt
        )
        
        # Store acceleration for next iteration
        self.last_linear_acceleration = current_linear_acceleration
        
        # ------------------------------------------------------------------
        # Publish wrench command
        # ------------------------------------------------------------------
        self._publish_wrench(force, torque)
    
    # ------------------------------------------------------------------
    # PID control methods
    # ------------------------------------------------------------------
    
    def _position_pid(self, current: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        """
        Position PID controller (outer loop).
        Generates velocity commands from position errors.
        """
        error = target - current
        
        # Proportional term
        p_term = self.gains.position_kp * error
        
        # Integral term
        self.position_integral += error * dt
        self.position_integral = np.clip(
            self.position_integral,
            -self.gains.position_integral_limit,
            self.gains.position_integral_limit
        )
        i_term = self.gains.position_ki * self.position_integral
        
        # Derivative term
        if np.any(self.last_position_error != 0):
            d_term = self.gains.position_kd * (error - self.last_position_error) / dt
        else:
            d_term = np.zeros(3)
        self.last_position_error = error
        
        # Combine terms and apply limits
        velocity_cmd = p_term + i_term + d_term
        velocity_cmd = np.clip(
            velocity_cmd,
            -self.gains.max_linear_velocity,
            self.gains.max_linear_velocity
        )
        
        return velocity_cmd
    
    def _orientation_pid(self, current: UnitQuaternion, target: UnitQuaternion, dt: float) -> np.ndarray:
        """
        Orientation PID controller (outer loop) using quaternions.
        Generates angular velocity commands from quaternion orientation errors.
        Note: Kd term is omitted as we don't have angular acceleration.
        """
        # Calculate quaternion error
        # error_quat = target * current.inv() gives rotation needed to reach target
        error_quat = target * current.inv()
        
        error_angle, error_vec = error_quat.angvec()
        
        # Proportional term
        p_term = self.gains.orientation_kp * error_vec * error_angle
        
        # Integral term
        self.orientation_integral += error_vec * dt
        self.orientation_integral = np.clip(
            self.orientation_integral,
            -self.gains.orientation_integral_limit,
            self.gains.orientation_integral_limit
        )
        i_term = self.gains.orientation_ki * self.orientation_integral
        
        # No derivative term for orientation (as specified)
        
        # Combine terms and apply limits
        angular_velocity_cmd = p_term + i_term
        angular_velocity_cmd = np.clip(
            angular_velocity_cmd,
            -self.gains.max_angular_velocity,
            self.gains.max_angular_velocity
        )
        
        return angular_velocity_cmd
    
    def _velocity_pid(self, current: np.ndarray, target: np.ndarray, 
                     current_accel: np.ndarray, dt: float) -> np.ndarray:
        """
        Velocity PID controller (inner loop).
        Generates force commands from velocity errors.
        """
        error = target - current
        
        # Proportional term
        p_term = self.gains.velocity_kp * error
        
        # Integral term
        self.velocity_integral += error * dt
        self.velocity_integral = np.clip(
            self.velocity_integral,
            -self.gains.velocity_integral_limit,
            self.gains.velocity_integral_limit
        )
        i_term = self.gains.velocity_ki * self.velocity_integral
        
        # Derivative term (using acceleration)
        d_term = -self.gains.velocity_kd * current_accel
        
        # Combine terms and apply limits
        force = p_term + i_term + d_term
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > self.gains.max_force:
            force = force * (self.gains.max_force / force_magnitude)
        
        return force
    
    def _angular_velocity_pid(self, current: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        """
        Angular velocity PID controller (inner loop).
        Generates torque commands from angular velocity errors.
        Note: Kd term is omitted as we don't have angular acceleration.
        """
        error = target - current
        
        # Proportional term
        p_term = self.gains.angular_velocity_kp * error
        
        # Integral term
        self.angular_velocity_integral += error * dt
        self.angular_velocity_integral = np.clip(
            self.angular_velocity_integral,
            -self.gains.angular_velocity_integral_limit,
            self.gains.angular_velocity_integral_limit
        )
        i_term = self.gains.angular_velocity_ki * self.angular_velocity_integral
        
        # No derivative term (as specified)
        
        # Combine terms and apply limits
        torque = p_term + i_term
        torque_magnitude = np.linalg.norm(torque)
        if torque_magnitude > self.gains.max_torque:
            torque = torque * (self.gains.max_torque / torque_magnitude)
        
        return torque
    
    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    
    def _publish_wrench(self, force: np.ndarray, torque: np.ndarray) -> None:
        """Publish wrench command using msgify utility."""
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "body"
        
        # Combine force and torque into 6D wrench vector and use msgify
        wrench_vec = np.concatenate([force, torque])
        msg.wrench = msgify(wrench_vec, message_type="Wrench")
        
        self._wrench_pub.publish(msg)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the PID controller node."""
    rclpy.init()
    controller = Controller()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
