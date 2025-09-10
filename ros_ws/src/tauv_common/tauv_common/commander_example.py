#!/usr/bin/env python3
"""
Commander Example Script

This script demonstrates how to send velocity and attitude commands to the
commander module. It can be used for testing the commander's functionality
or as a reference for higher-level control systems.

Usage:
    ros2 run tauv_common commander_example

This will send a sequence of commands to demonstrate:
1. Forward velocity command
2. Turning maneuver (yaw rotation)
3. Combined velocity and attitude control
4. Emergency stop
"""

import numpy as np
import rclpy
from rclpy.node import Node
from spatialmath import UnitQuaternion
from std_msgs.msg import Header
from tauv_msgs.msg import VelocityAttitudeCommand


class CommanderExample(Node):
    """Example node for sending commands to the commander"""

    def __init__(self):
        super().__init__("commander_example")

        # Publisher for velocity/attitude commands
        self._cmd_pub = self.create_publisher(
            VelocityAttitudeCommand, '/os/gnc/velocity_attitude_command', 10
        )

        # Timer for sending demo commands
        self._demo_timer = self.create_timer(2.0, self._demo_sequence)
        self._demo_step = 0

        self.get_logger().info("Commander Example started - sending demo commands...")

    def _demo_sequence(self):
        """Send a sequence of demo commands"""

        if self._demo_step == 0:
            # Step 1: Forward motion at 0.5 m/s
            self.get_logger().info("Demo Step 1: Forward velocity 0.5 m/s")
            self._send_velocity_command(forward_velocity=0.5)

        elif self._demo_step == 1:
            # Step 2: Turn right (positive yaw)
            self.get_logger().info("Demo Step 2: Turn right 30 degrees")
            yaw_angle = np.deg2rad(30)  # 30 degrees in radians
            target_quat = UnitQuaternion.Rz(yaw_angle)  # Rotation about Z-axis
            self._send_attitude_command(target_quat)

        elif self._demo_step == 2:
            # Step 3: Combined forward motion and attitude hold
            self.get_logger().info("Demo Step 3: Forward motion with attitude hold")
            yaw_angle = np.deg2rad(30)
            target_quat = UnitQuaternion.Rz(yaw_angle)
            self._send_combined_command(forward_velocity=0.3, target_quat=target_quat)

        elif self._demo_step == 3:
            # Step 4: Turn left to return to center
            self.get_logger().info("Demo Step 4: Return to center heading")
            target_quat = UnitQuaternion()  # Identity quaternion (zero rotation)
            self._send_attitude_command(target_quat)

        elif self._demo_step == 4:
            # Step 5: Stop all motion
            self.get_logger().info("Demo Step 5: Emergency stop")
            self._send_stop_command()

        else:
            # Demo complete
            self.get_logger().info("Demo sequence complete. Repeating...")
            self._demo_step = -1  # Will be incremented to 0

        self._demo_step += 1

    def _send_velocity_command(
        self,
        forward_velocity: float = 0.0,
        side_velocity: float = 0.0,
        vertical_velocity: float = 0.0,
    ):
        """Send a pure velocity command (attitude control disabled)"""

        cmd = VelocityAttitudeCommand()
        cmd.header = Header()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "os/body"

        # Set target velocity in body frame
        cmd.target_velocity.x = forward_velocity  # Forward/backward
        cmd.target_velocity.y = side_velocity  # Left/right
        cmd.target_velocity.z = vertical_velocity  # Up/down

        # Set target attitude (not used since attitude control is disabled)
        cmd.target_attitude.w = 1.0
        cmd.target_attitude.x = 0.0
        cmd.target_attitude.y = 0.0
        cmd.target_attitude.z = 0.0

        # Zero feedforward acceleration
        cmd.feedforward_acceleration.x = 0.0
        cmd.feedforward_acceleration.y = 0.0
        cmd.feedforward_acceleration.z = 0.0

        # Enable velocity control only
        cmd.velocity_control_enabled = True
        cmd.attitude_control_enabled = False

        self._cmd_pub.publish(cmd)

    def _send_attitude_command(self, target_quat: UnitQuaternion):
        """Send a pure attitude command (velocity control disabled)"""

        cmd = VelocityAttitudeCommand()
        cmd.header = Header()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "os/body"

        # Set target velocity (not used since velocity control is disabled)
        cmd.target_velocity.x = 0.0
        cmd.target_velocity.y = 0.0
        cmd.target_velocity.z = 0.0

        # Set target attitude
        target_quat_vec = target_quat.vec_xyzs
        cmd.target_attitude.x = target_quat_vec[0]  # x component
        cmd.target_attitude.y = target_quat_vec[1]  # y component
        cmd.target_attitude.z = target_quat_vec[2]  # z component
        cmd.target_attitude.w = target_quat_vec[3]  # w component

        # Zero feedforward acceleration
        cmd.feedforward_acceleration.x = 0.0
        cmd.feedforward_acceleration.y = 0.0
        cmd.feedforward_acceleration.z = 0.0

        # Enable attitude control only
        cmd.velocity_control_enabled = False
        cmd.attitude_control_enabled = True

        self._cmd_pub.publish(cmd)

    def _send_combined_command(
        self,
        forward_velocity: float = 0.0,
        side_velocity: float = 0.0,
        vertical_velocity: float = 0.0,
        target_quat: UnitQuaternion = None,
    ):
        """Send combined velocity and attitude command"""

        if target_quat is None:
            target_quat = UnitQuaternion()  # Identity quaternion

        cmd = VelocityAttitudeCommand()
        cmd.header = Header()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "os/body"

        # Set target velocity
        cmd.target_velocity.x = forward_velocity
        cmd.target_velocity.y = side_velocity
        cmd.target_velocity.z = vertical_velocity

        # Set target attitude
        target_quat_vec = target_quat.vec_xyzs
        cmd.target_attitude.x = target_quat_vec[0]  # x component
        cmd.target_attitude.y = target_quat_vec[1]  # y component
        cmd.target_attitude.z = target_quat_vec[2]  # z component
        cmd.target_attitude.w = target_quat_vec[3]  # w component

        # Zero feedforward acceleration
        cmd.feedforward_acceleration.x = 0.0
        cmd.feedforward_acceleration.y = 0.0
        cmd.feedforward_acceleration.z = 0.0

        # Enable both velocity and attitude control
        cmd.velocity_control_enabled = True
        cmd.attitude_control_enabled = True

        self._cmd_pub.publish(cmd)

    def _send_stop_command(self):
        """Send emergency stop command (zero velocity, current attitude)"""

        cmd = VelocityAttitudeCommand()
        cmd.header = Header()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "os/body"

        # Zero target velocity
        cmd.target_velocity.x = 0.0
        cmd.target_velocity.y = 0.0
        cmd.target_velocity.z = 0.0

        # Identity quaternion (maintain current attitude approximately)
        cmd.target_attitude.w = 1.0
        cmd.target_attitude.x = 0.0
        cmd.target_attitude.y = 0.0
        cmd.target_attitude.z = 0.0

        # Zero feedforward acceleration
        cmd.feedforward_acceleration.x = 0.0
        cmd.feedforward_acceleration.y = 0.0
        cmd.feedforward_acceleration.z = 0.0

        # Enable velocity control only for stopping
        cmd.velocity_control_enabled = True
        cmd.attitude_control_enabled = False

        self._cmd_pub.publish(cmd)


def main():
    """Main entry point"""
    rclpy.init()

    try:
        example = CommanderExample()
        rclpy.spin(example)
    except KeyboardInterrupt:
        pass
    finally:
        if 'example' in locals():
            example.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
