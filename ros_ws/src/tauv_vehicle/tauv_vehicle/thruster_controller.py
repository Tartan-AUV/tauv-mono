"""
Thruster controller that directly controls thrusters using Adafruit PWM board.
Maps thrust commands to PWM signals.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from adafruit_servokit import ServoKit
from adafruit_extended_bus import ExtendedI2C as I2C

from tauv_msgs.msg import TargetThrust

# I2C configuration
I2C_BUS = 7
PCA9685_ADDRESS = 0x40
PWM_FREQUENCY = 50  # Hz

# Thruster model parameters
THRUST_COEFF_FWD = 3.645e-3  # N/(rad/s)^2
THRUST_COEFF_REV = 2.905e-3  # N/(rad/s)^2
MAX_SETPOINT = 362.9  # rad/s

# PWM settings
PULSE_WIDTH_MIN_US = 1000
PULSE_WIDTH_MAX_US = 2000
NEUTRAL_FRACTION = 0.5
FULL_REVERSE_FRACTION = 0.0
FULL_FORWARD_FRACTION = 1.0


class ThrusterController(Node):
    
    def __init__(self):
        super().__init__("thruster_controller")
        
        # Initialize I2C and PWM controller
        self._i2c = I2C(I2C_BUS)
        self._kit = ServoKit(channels=16, i2c=self._i2c, address=PCA9685_ADDRESS)
        self._kit._pca.frequency = PWM_FREQUENCY
        
        # Get number of thrusters from parameter (default to 8)
        self.declare_parameter('num_thrusters', 8)
        self._num_thrusters = self.get_parameter('num_thrusters').value
        
        # Initialize all thrusters to neutral
        for i in range(self._num_thrusters):
            self._kit.servo[i].set_pulse_width_range(PULSE_WIDTH_MIN_US, PULSE_WIDTH_MAX_US)
            self._kit.servo[i].angle = None  # Disable angle mapping
            self._kit.servo[i].fraction = NEUTRAL_FRACTION
            
        self.get_logger().info(f"Initialized {self._num_thrusters} thrusters on I2C bus {I2C_BUS}")
        
        # Subscribe to target thrust commands
        self._target_thrust_sub = self.create_subscription(
            TargetThrust, 
            "target_thrust", 
            self._handle_target_thrust, 
            10
        )

    def _thrust_to_fraction(self, thrust_n: float) -> float:
        """
        Convert thrust in Newtons to PWM fraction (0.0 to 1.0).
        
        Args:
            thrust_n: Thrust in Newtons (positive = forward, negative = reverse)
            
        Returns:
            PWM fraction where 0.5 is neutral, 0.0 is full reverse, 1.0 is full forward
        """
        if abs(thrust_n) < 1e-6:  # Near zero thrust
            return NEUTRAL_FRACTION
            
        # Use appropriate thrust coefficient based on direction
        thrust_coeff = THRUST_COEFF_FWD if thrust_n > 0 else THRUST_COEFF_REV
        
        # Calculate required angular velocity
        omega_radps = np.sqrt(min(abs(thrust_n) / thrust_coeff, MAX_SETPOINT**2)) * np.sign(thrust_n)
        
        # Map to fraction: omega ranges from -MAX_SETPOINT to +MAX_SETPOINT
        # PWM fraction ranges from 0.0 (full reverse) to 1.0 (full forward)
        normalized = omega_radps / MAX_SETPOINT  # -1 to 1
        fraction = (normalized + 1.0) / 2.0  # 0 to 1
        
        # Clamp to valid range
        return np.clip(fraction, FULL_REVERSE_FRACTION, FULL_FORWARD_FRACTION)

    def _handle_target_thrust(self, msg: TargetThrust):
        """Handle incoming thrust commands and set PWM outputs."""
        thrusts = msg.target_thrust
        
        # Ensure we don't try to control more thrusters than initialized
        num_to_control = min(len(thrusts), self._num_thrusters)
        
        if len(thrusts) > self._num_thrusters:
            self.get_logger().warn(
                f"Received {len(thrusts)} thrust commands but only {self._num_thrusters} thrusters configured"
            )
        
        # Set PWM for each thruster
        for i in range(num_to_control):
            fraction = self._thrust_to_fraction(thrusts[i])
            self._kit.servo[i].fraction = fraction
            
        # Set any remaining configured thrusters to neutral
        for i in range(num_to_control, self._num_thrusters):
            self._kit.servo[i].fraction = NEUTRAL_FRACTION

    def destroy_node(self):
        """Ensure thrusters are set to neutral before shutting down."""
        self.get_logger().info("Shutting down thruster controller, setting all thrusters to neutral")
        for i in range(self._num_thrusters):
            try:
                self._kit.servo[i].fraction = NEUTRAL_FRACTION
            except Exception as e:
                self.get_logger().error(f"Failed to set thruster {i} to neutral: {e}")
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
