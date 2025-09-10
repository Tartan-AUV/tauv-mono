#!/usr/bin/env python3

import time
from enum import Enum

import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node


class TestPhase(Enum):
    """Test phases for the wrench test sequence"""

    WAITING = 1
    DOWN = 2
    ROLL = 3
    FORWARD = 4
    STOPPED = 5


class WrenchTestNode(Node):
    """
    ROS2 node for executing a timed wrench test sequence.

    Sequence:
    1. Wait for 20 seconds (publish zero wrench)
    2. Command down wrench for specified duration
    3. Command forward wrench for specified duration
    4. Command no wrench (zero) indefinitely

    Publishes wrench commands at 100Hz throughout the sequence.
    """

    def __init__(self):
        super().__init__('wrench_test_node')

        # Publisher for wrench commands
        self.publisher_ = self.create_publisher(WrenchStamped, '/os/gnc/target_wrench', 10)

        # Timer for publishing at regular intervals
        self.timer = self.create_timer(0.01, self.publish_wrench)  # 100 Hz

        # Test sequence timing
        self.start_time = time.time()
        self.wait_duration = 5.0  # seconds
        self.down_duration = 4.0  # seconds
        self.roll_duration = 3  # seconds
        self.forward_duration = 6.0  # seconds
        self.current_phase = TestPhase.WAITING

        # Down phase wrench values (in NED frame)
        self.down_force_x = 0.0  # North (N)
        self.down_force_y = 0.0  # East (N)
        self.down_force_z = 600.0  # Up (N, negative down)
        self.down_torque_x = 0.0  # Roll (Nm)
        self.down_torque_y = 0.0  # Pitch (Nm)
        self.down_torque_z = 0.0  # Yaw (Nm)

        # Roll phase wrench values (in NED frame)
        self.roll_force_x = 0.0  # North (N)
        self.roll_force_y = 0.0  # East (N)
        self.roll_force_z = 0.0  # Up (N, negative down)
        self.roll_torque_x = 0.0  # Roll (Nm)
        self.roll_torque_y = 10000.0  # Pitch (Nm)
        self.roll_torque_z = 0.0  # Yaw (Nm)

        # Forward phase wrench values (in NED frame)
        self.forward_force_x = 600.0  # North (N)
        self.forward_force_y = 0.0  # East (N)
        self.forward_force_z = 240.0  # Up (N, negative down)
        self.forward_torque_x = 0.0  # Roll (Nm)
        self.forward_torque_y = 0.0  # Pitch (Nm)
        self.forward_torque_z = 0.0  # Yaw (Nm)

        self.get_logger().info("Wrench Test Node started")
        self.get_logger().info("Test sequence:")
        self.get_logger().info(f"  1. Wait for {self.wait_duration} seconds (zero wrench)")
        self.get_logger().info(f"  2. Down wrench for {self.down_duration} seconds")
        self.get_logger().info(f"  3. Forward wrench for {self.forward_duration} seconds")
        self.get_logger().info("  4. Stop (zero wrench)")
        self.get_logger().info(
            f"  Down Force: [{self.down_force_x:.1f}, {self.down_force_y:.1f}, {self.down_force_z:.1f}] N"
        )
        self.get_logger().info(
            f"  Down Torque: [{self.down_torque_x:.1f}, {self.down_torque_y:.1f}, {self.down_torque_z:.1f}] Nm"
        )
        self.get_logger().info(
            f"  Forward Force: [{self.forward_force_x:.1f}, {self.forward_force_y:.1f}, {self.forward_force_z:.1f}] N"
        )
        self.get_logger().info(
            f"  Forward Torque: [{self.forward_torque_x:.1f}, {self.forward_torque_y:.1f}, {self.forward_torque_z:.1f}] Nm"
        )

    def publish_wrench(self):
        """Publish wrench values at 100Hz based on current test phase"""
        # Update test phase based on elapsed time
        self.update_test_phase()

        # Create and publish wrench message
        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = self.get_clock().now().to_msg()
        wrench_msg.header.frame_id = 'os/body'

        # Set force values based on current test phase (NED frame)
        if self.current_phase == TestPhase.DOWN:
            wrench_msg.wrench.force.x = self.down_force_x
            wrench_msg.wrench.force.y = self.down_force_y
            wrench_msg.wrench.force.z = self.down_force_z
            wrench_msg.wrench.torque.x = self.down_torque_x
            wrench_msg.wrench.torque.y = self.down_torque_y
            wrench_msg.wrench.torque.z = self.down_torque_z
        elif self.current_phase == TestPhase.ROLL:
            wrench_msg.wrench.force.x = self.roll_force_x
            wrench_msg.wrench.force.y = self.roll_force_y
            wrench_msg.wrench.force.z = self.roll_force_z
            wrench_msg.wrench.torque.x = self.roll_torque_x
            wrench_msg.wrench.torque.y = self.roll_torque_y
            wrench_msg.wrench.torque.z = self.roll_torque_z
        elif self.current_phase == TestPhase.FORWARD:
            wrench_msg.wrench.force.x = self.forward_force_x
            wrench_msg.wrench.force.y = self.forward_force_y
            wrench_msg.wrench.force.z = self.forward_force_z
            wrench_msg.wrench.torque.x = self.forward_torque_x
            wrench_msg.wrench.torque.y = self.forward_torque_y
            wrench_msg.wrench.torque.z = self.forward_torque_z
        else:
            # Publish zero wrench during WAITING and STOPPED phases
            wrench_msg.wrench.force.x = 0.0
            wrench_msg.wrench.force.y = 0.0
            wrench_msg.wrench.force.z = 0.0
            wrench_msg.wrench.torque.x = 0.0
            wrench_msg.wrench.torque.y = 0.0
            wrench_msg.wrench.torque.z = 0.0

        self.publisher_.publish(wrench_msg)

    def update_test_phase(self):
        """Update the current test phase based on elapsed time"""
        elapsed_time = time.time() - self.start_time
        previous_phase = self.current_phase

        if elapsed_time < self.wait_duration:
            self.current_phase = TestPhase.WAITING
        elif elapsed_time < self.wait_duration + self.down_duration:
            self.current_phase = TestPhase.DOWN
        elif elapsed_time < self.wait_duration + self.down_duration + self.roll_duration:
            self.current_phase = TestPhase.ROLL
        elif (
            elapsed_time
            < self.wait_duration + self.down_duration + self.roll_duration + self.forward_duration
        ):
            self.current_phase = TestPhase.FORWARD
        else:
            self.current_phase = TestPhase.STOPPED

        # Log phase transitions
        if self.current_phase != previous_phase:
            if self.current_phase == TestPhase.DOWN:
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
                    f"Phase transition: Stopping all wrenches (t={elapsed_time:.1f}s)"
                )
                self.get_logger().info("Test sequence complete")


def main(args=None):
    rclpy.init(args=args)

    try:
        node = WrenchTestNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
