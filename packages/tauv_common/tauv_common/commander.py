"""
Simplified Commander node for TartanAUV vehicles.

This node implements a simple timed behavior:
1. Wait for 20 seconds
2. Publish a fixed target twist for 20 seconds
3. Stop publishing

Notes
-----
This is a minimal timing-based commander for basic vehicle testing.
"""

from __future__ import annotations

from enum import Enum
import time

import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher

from tauv_msgs.msg import ControllerCommand
from geometry_msgs.msg import Twist

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class CommanderState(Enum):
    """States for the commander node."""
    WAITING = "waiting"      # 0-20s: Don't publish anything
    PUBLISHING = "publishing"  # 20-40s: Publish fixed twist
    STOPPED = "stopped"      # 40s+: Stop publishing

# ---------------------------------------------------------------------------
# Commander node
# ---------------------------------------------------------------------------

class Commander(Node):
    """Simplified commander node with timed twist publishing."""

    def __init__(self) -> None:
        super().__init__("commander")

        # ------------------------------------------------------------------
        # Timing and state
        # ------------------------------------------------------------------
        self.start_time = time.time()
        self.state = CommanderState.WAITING
        
        # Phase durations in seconds
        self.wait_duration = 1.0
        self.publish_duration = 20.0

        # ------------------------------------------------------------------
        # Fixed twist command to publish
        # ------------------------------------------------------------------
        self.target_twist = Twist()
        # Set desired linear and angular velocities
        self.target_twist.linear.x = 0.0   # 0.5 m/s forward
        self.target_twist.linear.y = 0.0   # No sideways motion
        self.target_twist.linear.z = 0.05   # 
        self.target_twist.angular.x = 0.0  # No roll rate
        self.target_twist.angular.y = 0.0  # No pitch rate
        self.target_twist.angular.z = 0.0  # 0.2 rad/s yaw rate

        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------
        self._cmd_pub: Publisher = self.create_publisher(
            ControllerCommand,
            "gnc/controller_command",
            10,
        )

        # Timer to check state and publish at 50 Hz
        self._timer_period = 0.02
        self.create_timer(self._timer_period, self._timer_callback)

        self.get_logger().info(
            "Commander initialized – Will wait 20s, publish twist for 20s, then stop"
        )
        self.get_logger().info(
            f"Fixed twist: linear=({self.target_twist.linear.x}, {self.target_twist.linear.y}, {self.target_twist.linear.z}), "
            f"angular=({self.target_twist.angular.x}, {self.target_twist.angular.y}, {self.target_twist.angular.z})"
        )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _timer_callback(self) -> None:
        """Timer callback to handle state transitions and publishing."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # State machine logic
        if self.state == CommanderState.WAITING:
            if elapsed_time >= self.wait_duration:
                self.state = CommanderState.PUBLISHING
                self.get_logger().info("Starting to publish fixed twist command")
        elif self.state == CommanderState.PUBLISHING:
            if elapsed_time >= (self.wait_duration + self.publish_duration):
                self.state = CommanderState.STOPPED
                self.get_logger().info("Stopping twist publishing")
            else:
                # Publish the fixed twist command
                self._publish_twist_command()

        # Log state transitions (only once per transition)
        if self.state == CommanderState.WAITING and elapsed_time < 1.0:
            if int(elapsed_time * 50) % 50 == 0:  # Log once per second during wait
                remaining = self.wait_duration - elapsed_time
                self.get_logger().info(f"Waiting... {remaining:.1f}s remaining")

    def _publish_twist_command(self) -> None:
        """Create and publish the controller command with fixed twist."""
        cmd = ControllerCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "body"  # Twist is in body frame
        cmd.target_twist = self.target_twist
        
        self._cmd_pub.publish(cmd)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the simplified commander node."""
    rclpy.init()
    commander = Commander()
    
    try:
        rclpy.spin(commander)
    except KeyboardInterrupt:
        pass
    finally:
        commander.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()