"""
Combined depth sensor and thruster controller using unified I2C driver.
Reads depth sensor data and controls thrusters through a single I2C interface.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

from tauv_msgs.msg import ThrusterSetpoint, DepthSensorFrame
from .i2c_drivers import UnifiedI2CDriver, UNITS_Pa, UNITS_Centigrade, DENSITY_FRESHWATER

# I2C configuration
I2C_BUS = 7
PCA9685_ADDRESS = 0x40
PWM_FREQUENCY = 50  # Hz

# Thruster model parameters
THRUST_COEFF_FWD = 3.645e-3  # N/(rad/s)^2
THRUST_COEFF_REV = 2.905e-3  # N/(rad/s)^2
MAX_SETPOINT = 362.9  # rad/s

# PWM settings for 50Hz frequency
# At 50Hz, the period is 20ms, and PCA9685 has 12-bit resolution (4096 counts)
# 1ms pulse = 205 counts, 1.5ms = 307 counts, 2ms = 409 counts
PULSE_WIDTH_MIN_US = 1000  # 1ms pulse
PULSE_WIDTH_MAX_US = 2000  # 2ms pulse
NEUTRAL_PULSE_US = 1500    # 1.5ms pulse

# Convert pulse widths to PCA9685 counts (12-bit resolution at 50Hz)
PWM_PERIOD_US = 20000  # 20ms period at 50Hz
PWM_RESOLUTION = 4096  # 12-bit resolution
PWM_COUNT_MIN = int(PULSE_WIDTH_MIN_US * PWM_RESOLUTION / PWM_PERIOD_US)    # ~205
PWM_COUNT_MAX = int(PULSE_WIDTH_MAX_US * PWM_RESOLUTION / PWM_PERIOD_US)    # ~409
PWM_COUNT_NEUTRAL = int(NEUTRAL_PULSE_US * PWM_RESOLUTION / PWM_PERIOD_US)  # ~307


class DepthActuatorsI2C(Node):
    """
    ROS2 node that combines depth sensor reading and thruster control
    through a unified I2C interface.
    """
    
    def __init__(self):
        super().__init__("depth_actuators_i2c")
        
        # Initialize unified I2C driver
        self._driver = UnifiedI2CDriver(
            bus_number=I2C_BUS,
            pca9685_address=PCA9685_ADDRESS
        )
        
        # Initialize both devices
        ms5837_ok = self._driver.init_ms5837()
        pca9685_ok = self._driver.init_pca9685()
        
        if not ms5837_ok:
            self.get_logger().warn("MS5837 depth sensor initialization failed")
        else:
            self.get_logger().info("MS5837 depth sensor initialized successfully")
            # Set fluid density (default to freshwater)
            self._driver.set_fluid_density(DENSITY_FRESHWATER)
        
        if not pca9685_ok:
            self.get_logger().error("PCA9685 PWM controller initialization failed")
            raise RuntimeError("Failed to initialize PCA9685 PWM controller")
        else:
            self.get_logger().info("PCA9685 PWM controller initialized successfully")
            # Set PWM frequency for servo control
            self._driver.set_pwm_frequency(PWM_FREQUENCY)
        
        self._ms5837_ok = ms5837_ok
        self._pca9685_ok = pca9685_ok
        
        # Get number of thrusters from parameter (default to 8)
        self.declare_parameter('num_thrusters', 8)
        self._num_thrusters = self.get_parameter('num_thrusters').value
        
        # Depth sensor reading rate (Hz)
        self.declare_parameter('depth_sensor_rate', 10.0)
        depth_sensor_rate = self.get_parameter('depth_sensor_rate').value
        
        # Initialize all thrusters to neutral
        for i in range(self._num_thrusters):
            self._driver.set_pwm(i, PWM_COUNT_NEUTRAL)
            
        self.get_logger().info(
            f"Initialized {self._num_thrusters} thrusters on I2C bus {I2C_BUS}, "
            f"PCA9685 at address 0x{PCA9685_ADDRESS:02X}"
        )
        
        # Subscribe to thruster setpoint commands
        self._thruster_setpoint_sub = self.create_subscription(
            ThrusterSetpoint,
            "thruster_setpoint",
            self._handle_thruster_setpoint,
            10
        )
        
        # Publisher for depth sensor data
        self._depth_sensor_pub = self.create_publisher(
            DepthSensorFrame,
            "depth",
            10
        )
        
        # Timer for periodic depth sensor reading
        if self._ms5837_ok:
            self._depth_timer = self.create_timer(
                1.0 / depth_sensor_rate,
                self._read_depth_sensor
            )
        
        self.get_logger().info("Depth actuators I2C node initialized")

    def _omega_to_pwm_count(self, omega_radps: float, enabled: bool = True) -> int:
        """
        Convert angular velocity in rad/s to PWM count value.
        
        Args:
            omega_radps: Angular velocity in rad/s (positive = forward, negative = reverse)
            enabled: Whether the thruster is enabled
            
        Returns:
            PWM count value (0-4095) where neutral is ~307
        """
        if not enabled or abs(omega_radps) < 1e-6:  # Disabled or near zero
            return PWM_COUNT_NEUTRAL
        
        # Clamp omega to valid range
        omega_clamped = np.clip(omega_radps, -MAX_SETPOINT, MAX_SETPOINT)
        
        # Map omega to PWM count
        # omega ranges from -MAX_SETPOINT to +MAX_SETPOINT
        # PWM count ranges from PWM_COUNT_MIN to PWM_COUNT_MAX
        normalized = omega_clamped / MAX_SETPOINT  # -1 to 1
        
        # Linear interpolation between min and max counts
        if normalized < 0:
            # Reverse: interpolate between neutral and min
            pwm_count = PWM_COUNT_NEUTRAL + normalized * (PWM_COUNT_NEUTRAL - PWM_COUNT_MIN)
        else:
            # Forward: interpolate between neutral and max
            pwm_count = PWM_COUNT_NEUTRAL + normalized * (PWM_COUNT_MAX - PWM_COUNT_NEUTRAL)
        
        # Ensure integer and clamp to valid range
        return int(np.clip(pwm_count, PWM_COUNT_MIN, PWM_COUNT_MAX))

    def _handle_thruster_setpoint(self, msg: ThrusterSetpoint):
        """Handle incoming thruster setpoint commands and set PWM outputs."""
        omega_values = msg.omega_radps
        enables = msg.enables
        
        # Ensure we don't try to control more thrusters than initialized
        num_to_control = min(len(omega_values), self._num_thrusters)
        
        if len(omega_values) > self._num_thrusters:
            self.get_logger().warn(
                f"Received {len(omega_values)} thruster commands but only "
                f"{self._num_thrusters} thrusters configured"
            )
        
        # Set PWM for each thruster
        for i in range(num_to_control):
            # Check if we have an enable flag for this thruster
            enabled = enables[i] if i < len(enables) else True
            pwm_count = self._omega_to_pwm_count(omega_values[i], enabled)
            self._driver.set_pwm(i, pwm_count)
            
        # Set any remaining configured thrusters to neutral
        for i in range(num_to_control, self._num_thrusters):
            self._driver.set_pwm(i, PWM_COUNT_NEUTRAL)

    def _read_depth_sensor(self):
        """Read depth sensor data and publish it."""
        if not self._ms5837_ok:
            return
        
        # Read sensor data
        if not self._driver.read_depth_sensor():
            self.get_logger().warn("Failed to read depth sensor")
            return
        
        # Get sensor readings
        pressure_pa = self._driver.get_pressure(UNITS_Pa)
        temperature_c = self._driver.get_temperature(UNITS_Centigrade)
        depth_m = self._driver.get_depth()
        
        # Create and publish depth sensor message
        msg = DepthSensorFrame()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "depth_sensor"
        
        msg.depth = float(depth_m)
        msg.pressure = float(pressure_pa)
        msg.temperature = float(temperature_c)
        
        self._depth_sensor_pub.publish(msg)

    def destroy_node(self):
        """Ensure thrusters are set to neutral before shutting down."""
        self.get_logger().info(
            "Shutting down depth actuators I2C node, setting all thrusters to neutral"
        )
        
        if self._pca9685_ok:
            for i in range(self._num_thrusters):
                try:
                    self._driver.set_pwm(i, PWM_COUNT_NEUTRAL)
                except Exception as e:
                    self.get_logger().error(f"Failed to set thruster {i} to neutral: {e}")
        
        # Close I2C bus
        try:
            self._driver.close()
        except Exception as e:
            self.get_logger().error(f"Failed to close I2C bus: {e}")
        
        super().destroy_node()


def main():
    rclpy.init()
    node = DepthActuatorsI2C()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
