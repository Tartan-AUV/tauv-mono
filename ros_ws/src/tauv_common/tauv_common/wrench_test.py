#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped

class WrenchTestNode(Node):
    """
    ROS2 node for publishing hardcoded WrenchStamped messages at 100Hz.
    
    Publishes a constant wrench command for testing purposes.
    """
    
    def __init__(self):
        super().__init__('wrench_test_node')
        
        # Publisher for wrench commands
        self.publisher_ = self.create_publisher(WrenchStamped, '/os/gnc/target_wrench', 10)
        
        # Timer for publishing at regular intervals
        self.timer = self.create_timer(0.01, self.publish_wrench)  # 100 Hz
        
        # Hardcoded wrench values (in NED frame)
        self.force_x = -20.0   # North (N)
        self.force_y = 0.0    # East (N)
        self.force_z = -10.0   # Up (N, negative down)
        self.torque_x = 0.0   # Roll (Nm)
        self.torque_y = 0.0   # Pitch (Nm)
        self.torque_z = 0.0   # Yaw (Nm)
        
        self.get_logger().info("Wrench Test Node started")
        self.get_logger().info("Publishing hardcoded wrench at 100Hz:")
        self.get_logger().info(f"  Force: [{self.force_x:.1f}, {self.force_y:.1f}, {self.force_z:.1f}] N")
        self.get_logger().info(f"  Torque: [{self.torque_x:.1f}, {self.torque_y:.1f}, {self.torque_z:.1f}] Nm")
        
    def publish_wrench(self):
        """Publish hardcoded wrench values at 100Hz"""
        # Create and publish wrench message
        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = self.get_clock().now().to_msg()
        wrench_msg.header.frame_id = 'os/body'
        
        # Set force values (NED frame)
        wrench_msg.wrench.force.x = self.force_x
        wrench_msg.wrench.force.y = self.force_y
        wrench_msg.wrench.force.z = self.force_z
        
        # Set torque values (NED frame)
        wrench_msg.wrench.torque.x = self.torque_x
        wrench_msg.wrench.torque.y = self.torque_y
        wrench_msg.wrench.torque.z = self.torque_z
        
        self.publisher_.publish(wrench_msg)

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
