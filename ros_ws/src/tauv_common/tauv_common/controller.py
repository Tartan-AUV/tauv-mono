"""
PD Yaw Controller with Wrench Test Sequence for TartanAUV vehicles.

This controller implements:
1. A timed wrench test sequence that applies different wrenches over time
2. A proportional-derivative control loop for yaw orientation
3. The final output is the sum of test wrench and yaw control wrench

Test sequence:
1. Wait phase: Output zero wrench (overrides yaw control)
2. Down phase: Apply downward force
3. Roll phase: Apply roll torque
4. Forward phase: Apply forward force with downward component
5. Stopped phase: Return to normal yaw control only
"""

from __future__ import annotations

import time
from enum import Enum

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.client import Client
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from std_srvs.srv import Empty
from tauv_msgs.msg import NavigationState

from tauv_common.util.geometry import msgify, numpify

# ---------------------------------------------------------------------------
# Test Phase Enum
# ---------------------------------------------------------------------------


class TestPhase(Enum):
    """Test phases for the wrench test sequence"""

    WAIT1 = 1
    WAIT2 = 2
    ROTATE = 3
    DOWN = 4
    ROLL = 5
    FORWARD = 6
    STOPPED = 7


# ---------------------------------------------------------------------------
# Yaw Controller Parameters
# ---------------------------------------------------------------------------


class YawControllerGains:
    """Proportional and derivative gains for yaw control only."""

    def __init__(self):
        # Yaw control gains
        self.yaw_kp = 50.0  # Proportional gain for yaw control
        self.yaw_kd = 5.0  # Derivative (damping) gain for yaw control

        # Control limits
        self.max_torque = 10000.0  # N⋅m


# ---------------------------------------------------------------------------
# Controller Node
# ---------------------------------------------------------------------------


class Controller(Node):
    """
    PD Yaw Controller with Wrench Test Sequence.

    Runs a timed wrench test sequence while maintaining yaw control.
    The output wrench is the sum of test wrench and yaw control wrench,
    except during WAITING phase where output is zero.

    Test sequence:
    1. Wait phase (5s): Zero wrench output
    2. Down phase (4s): Downward force
    3. Roll phase (3s): Roll torque
    4. Forward phase (6s): Forward force with downward component
    5. Stopped phase: Return to yaw control only

    Subscribes to:
    - NavigationState: Current vehicle state (yaw from pose, yaw rate from twist)

    Publishes to:
    - target_wrench: Combined wrench command for thruster allocation
    """

    def __init__(self) -> None:
        super().__init__("yaw_controller")

        # ------------------------------------------------------------------
        # Target yaw angle (hardcoded)
        # ------------------------------------------------------------------
        self.target_yaw = 0.0  # Target yaw angle in radians

        # ------------------------------------------------------------------
        # Controller gains and parameters
        # ------------------------------------------------------------------
        self.gains = YawControllerGains()

        # ------------------------------------------------------------------
        # Test sequence timing and parameters
        # ------------------------------------------------------------------
        self.start_time = time.time()
        self.wait1_duration = 10.0  # seconds
        self.wait2_duration = 15.0  # seconds
        self.rotate_duration = 15.0  # seconds
        self.down_duration = 4.0  # seconds
        self.roll_duration = 3.0  # seconds
        self.forward_duration = 6.0  # seconds
        self.current_phase = TestPhase.WAIT1
        self.service_called = False  # Flag to track if retare service was called

        # Down phase wrench values (in NED frame)
        self.down_force_x = 0.0  # North (N)
        self.down_force_y = 0.0  # East (N)
        self.down_force_z = 600.0  # Down (N)
        self.down_torque_x = 0.0  # Roll (Nm)
        self.down_torque_y = 0.0  # Pitch (Nm)
        self.down_torque_z = 0.0  # Yaw (Nm)

        # Roll phase wrench values (in NED frame)
        self.roll_force_x = 0.0  # North (N)
        self.roll_force_y = 0.0  # East (N)
        self.roll_force_z = 240.0  # Down (N)
        self.roll_torque_x = 600.0  # Roll (Nm)
        self.roll_torque_y = 0.0  # Pitch (Nm)
        self.roll_torque_z = 0.0  # Yaw (Nm)

        # Rotate phase wrench values (in NED frame)
        self.rotate_force_x = 0.0  # North (N)
        self.rotate_force_y = 0.0  # East (N)
        self.rotate_force_z = 0.0  # Down (N)
        self.rotate_torque_x = 0.0  # Roll (Nm)
        self.rotate_torque_y = 0.0  # Pitch (Nm)
        self.rotate_torque_z = 0.0  # Yaw torque for rotation (Nm)

        # Forward phase wrench values (in NED frame)
        self.forward_force_x = 600.0  # North (N)
        self.forward_force_y = 0.0  # East (N)
        self.forward_force_z = 240.0  # Down (N)
        self.forward_torque_x = 0.0  # Roll (Nm)
        self.forward_torque_y = 0.0  # Pitch (Nm)
        self.forward_torque_z = 0.0  # Yaw (Nm)

        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------
        self._nav_state_sub: Subscription = self.create_subscription(
            NavigationState,
            "gnc/navigation_state",
            self._navigation_callback,
            10,
        )

        self._wrench_pub: Publisher = self.create_publisher(
            WrenchStamped,
            "gnc/target_wrench",
            10,
        )

        # Service client for retaring local frame
        self._retare_client: Client = self.create_client(Empty, "/os/retare_local_frame")

        # Control loop timer at 50 Hz (changed from 100 Hz in wrench_test)
        self._control_timer = self.create_timer(0.02, self._control_loop)

        # ------------------------------------------------------------------
        # Initialization logging
        # ------------------------------------------------------------------
        self.get_logger().info("PD Yaw Controller with Wrench Test initialized")
        self.get_logger().info(
            f"Yaw control - Target: {self.target_yaw:.2f} rad ({np.degrees(self.target_yaw):.1f} deg), "
            f"Kp: {self.gains.yaw_kp:.1f}, Kd: {self.gains.yaw_kd:.1f}"
        )
        self.get_logger().info("Test sequence:")
        self.get_logger().info(f"  1. WAIT1 for {self.wait1_duration} seconds (zero wrench)")
        self.get_logger().info("  2. Call retare_local_frame service")
        self.get_logger().info(f"  3. WAIT2 for {self.wait2_duration} seconds (zero wrench)")
        self.get_logger().info(f"  4. ROTATE wrench for {self.rotate_duration} seconds")
        self.get_logger().info(f"  5. DOWN wrench for {self.down_duration} seconds")
        self.get_logger().info(f"  6. ROLL wrench for {self.roll_duration} seconds")
        self.get_logger().info(f"  7. FORWARD wrench for {self.forward_duration} seconds")
        self.get_logger().info("  8. STOPPED (yaw control only)")
        self.get_logger().info(
            f"  Rotate Force: [{self.rotate_force_x:.1f}, {self.rotate_force_y:.1f}, {self.rotate_force_z:.1f}] N"
        )
        self.get_logger().info(
            f"  Rotate Torque: [{self.rotate_torque_x:.1f}, {self.rotate_torque_y:.1f}, {self.rotate_torque_z:.1f}] Nm"
        )
        self.get_logger().info(
            f"  Down Force: [{self.down_force_x:.1f}, {self.down_force_y:.1f}, {self.down_force_z:.1f}] N"
        )
        self.get_logger().info(
            f"  Down Torque: [{self.down_torque_x:.1f}, {self.down_torque_y:.1f}, {self.down_torque_z:.1f}] Nm"
        )
        self.get_logger().info(
            f"  Roll Force: [{self.roll_force_x:.1f}, {self.roll_force_y:.1f}, {self.roll_force_z:.1f}] N"
        )
        self.get_logger().info(
            f"  Roll Torque: [{self.roll_torque_x:.1f}, {self.roll_torque_y:.1f}, {self.roll_torque_z:.1f}] Nm"
        )
        self.get_logger().info(
            f"  Forward Force: [{self.forward_force_x:.1f}, {self.forward_force_y:.1f}, {self.forward_force_z:.1f}] N"
        )
        self.get_logger().info(
            f"  Forward Torque: [{self.forward_torque_x:.1f}, {self.forward_torque_y:.1f}, {self.forward_torque_z:.1f}] Nm"
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
        """Main control loop - runs at 50 Hz. Combines test wrench with yaw control."""
        # Update test phase based on elapsed time
        self.update_test_phase()

        # ------------------------------------------------------------------
        # Get test sequence wrench based on current phase
        # ------------------------------------------------------------------
        test_wrench = self._get_test_wrench()

        # ------------------------------------------------------------------
        # Calculate yaw control wrench (if nav state available and not in WAITING)
        # ------------------------------------------------------------------
        yaw_wrench = np.zeros(6)

        if self.current_phase == TestPhase.WAIT1 or self.current_phase == TestPhase.WAIT2:
            # During WAIT phases, output only zero wrench
            combined_wrench = np.zeros(6)
        else:
            # Calculate yaw control if navigation state is available
            if self.current_nav_state is not None:
                current_yaw = self._extract_yaw_from_nav_state()
                current_yaw_rate = self._extract_yaw_rate_from_nav_state()
                yaw_torque = self._yaw_pd_control(current_yaw, current_yaw_rate)
                yaw_wrench[5] = yaw_torque  # Only yaw torque (tz) component

            # Combine test wrench and yaw control wrench
            combined_wrench = test_wrench + yaw_wrench

        # ------------------------------------------------------------------
        # Publish combined wrench command
        # ------------------------------------------------------------------
        self._publish_combined_wrench(combined_wrench)

    def _extract_yaw_from_nav_state(self) -> float:
        """Extract yaw angle (in radians) from the NavigationState body_pose quaternion."""
        pose = numpify(self.current_nav_state.body_pose.orientation)

        # Convert quaternion to euler angles (roll, pitch, yaw)
        _, _, yaw = pose.rpy()
        return yaw

    def _extract_yaw_rate_from_nav_state(self) -> float:
        """Extract yaw rate (angular velocity around z-axis) from NavigationState body_twist."""
        return self.current_nav_state.body_twist.angular.z

    def _yaw_pd_control(self, current_yaw: float, current_yaw_rate: float) -> float:
        """
        Proportional-derivative controller for yaw orientation with damping.
        Returns yaw torque command.
        """
        # Calculate yaw error (handle angle wrapping)
        yaw_error = self.target_yaw - current_yaw

        # Wrap error to [-pi, pi]
        while yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        while yaw_error < -np.pi:
            yaw_error += 2 * np.pi

        # Proportional control + derivative (damping) term
        # PD control: u = Kp * error - Kd * rate (negative because we want to oppose the rate)
        yaw_torque = self.gains.yaw_kp * yaw_error - self.gains.yaw_kd * current_yaw_rate

        # Apply torque limits
        yaw_torque = np.clip(yaw_torque, -self.gains.max_torque, self.gains.max_torque)

        return yaw_torque

    # ------------------------------------------------------------------
    # Test sequence methods
    # ------------------------------------------------------------------

    def update_test_phase(self):
        """Update the current test phase based on elapsed time"""
        elapsed_time = time.time() - self.start_time
        previous_phase = self.current_phase

        if elapsed_time < self.wait1_duration:
            self.current_phase = TestPhase.WAIT1
        elif elapsed_time < self.wait1_duration + self.wait2_duration:
            self.current_phase = TestPhase.WAIT2
            # Call retare service after WAIT1 ends (once only)
            if previous_phase == TestPhase.WAIT1 and not self.service_called:
                self._call_retare_service()
        elif elapsed_time < self.wait1_duration + self.wait2_duration + self.rotate_duration:
            self.current_phase = TestPhase.ROTATE
        elif (
            elapsed_time
            < self.wait1_duration + self.wait2_duration + self.rotate_duration + self.down_duration
        ):
            self.current_phase = TestPhase.DOWN
        elif (
            elapsed_time
            < self.wait1_duration
            + self.wait2_duration
            + self.rotate_duration
            + self.down_duration
            + self.roll_duration
        ):
            self.current_phase = TestPhase.ROLL
        elif (
            elapsed_time
            < self.wait1_duration
            + self.wait2_duration
            + self.rotate_duration
            + self.down_duration
            + self.roll_duration
            + self.forward_duration
        ):
            self.current_phase = TestPhase.FORWARD
        else:
            self.current_phase = TestPhase.STOPPED

        # Log phase transitions
        if self.current_phase != previous_phase:
            if self.current_phase == TestPhase.WAIT2:
                self.get_logger().info(
                    f"Phase transition: Starting WAIT2 phase (t={elapsed_time:.1f}s)"
                )
            elif self.current_phase == TestPhase.ROTATE:
                self.get_logger().info(
                    f"Phase transition: Starting ROTATE wrench (t={elapsed_time:.1f}s)"
                )
            elif self.current_phase == TestPhase.DOWN:
                self.get_logger().info(
                    f"Phase transition: Starting DOWN wrench (t={elapsed_time:.1f}s)"
                )
            elif self.current_phase == TestPhase.ROLL:
                self.get_logger().info(
                    f"Phase transition: Starting ROLL wrench (t={elapsed_time:.1f}s)"
                )
            elif self.current_phase == TestPhase.FORWARD:
                self.get_logger().info(
                    f"Phase transition: Starting FORWARD wrench (t={elapsed_time:.1f}s)"
                )
            elif self.current_phase == TestPhase.STOPPED:
                self.get_logger().info(
                    f"Phase transition: Test complete, yaw control only (t={elapsed_time:.1f}s)"
                )

    def _get_test_wrench(self) -> np.ndarray:
        """Get the test wrench based on current phase. Returns 6D wrench vector."""
        wrench = np.zeros(6)  # [fx, fy, fz, tx, ty, tz]

        if self.current_phase == TestPhase.DOWN:
            wrench[0] = self.down_force_x
            wrench[1] = self.down_force_y
            wrench[2] = self.down_force_z
            wrench[3] = self.down_torque_x
            wrench[4] = self.down_torque_y
            wrench[5] = self.down_torque_z
        elif self.current_phase == TestPhase.ROTATE:
            wrench[0] = self.rotate_force_x
            wrench[1] = self.rotate_force_y
            wrench[2] = self.rotate_force_z
            wrench[3] = self.rotate_torque_x
            wrench[4] = self.rotate_torque_y
            wrench[5] = self.rotate_torque_z
        elif self.current_phase == TestPhase.ROLL:
            wrench[0] = self.roll_force_x
            wrench[1] = self.roll_force_y
            wrench[2] = self.roll_force_z
            wrench[3] = self.roll_torque_x
            wrench[4] = self.roll_torque_y
            wrench[5] = self.roll_torque_z
        elif self.current_phase == TestPhase.FORWARD:
            wrench[0] = self.forward_force_x
            wrench[1] = self.forward_force_y
            wrench[2] = self.forward_force_z
            wrench[3] = self.forward_torque_x
            wrench[4] = self.forward_torque_y
            wrench[5] = self.forward_torque_z
        # For WAIT1, WAIT2 and STOPPED phases, return zero wrench

        return wrench

    def _call_retare_service(self):
        """Call the retare local frame service asynchronously"""
        if not self._retare_client.service_is_ready():
            self.get_logger().warn("Retare service not available, skipping call")
            self.service_called = True  # Mark as called to avoid repeated attempts
            return

        req = Empty.Request()
        future = self._retare_client.call_async(req)

        def service_callback(future):
            try:
                response = future.result()
                self.get_logger().info("Successfully called retare_local_frame service")
            except Exception as e:
                self.get_logger().error(f"Failed to call retare_local_frame service: {e}")

        future.add_done_callback(service_callback)
        self.service_called = True
        self.get_logger().info("Called retare_local_frame service")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _publish_combined_wrench(self, wrench_vec: np.ndarray) -> None:
        """Publish combined wrench command."""
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "os/body"

        # Convert numpy array to Wrench message
        msg.wrench = msgify(wrench_vec, message_type="Wrench")

        self._wrench_pub.publish(msg)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the yaw controller node."""
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
