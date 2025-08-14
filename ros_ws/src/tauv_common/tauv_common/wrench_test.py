#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped

class WrenchTestNode(Node):
    """
    ROS2 node for publishing WrenchStamped messages that toggle on/off every second.
    
    Publishes wrench commands at 100Hz, but toggles between active wrench values
    and zero values with a 1 second interval.
    """
    
    def __init__(self):
        super().__init__('wrench_test_node')
        
        # Publisher for wrench commands
        self.publisher_ = self.create_publisher(WrenchStamped, '/os/gnc/target_wrench', 10)
        
        # Timer for publishing at regular intervals
        self.timer = self.create_timer(0.01, self.publish_wrench)  # 100 Hz
        
        # Timer for toggling wrench on/off
        self.toggle_timer = self.create_timer(1000.0, self.toggle_wrench)  # 1 Hz toggle
        
        # Wrench toggle state
        self.wrench_enabled = True
        
        # Hardcoded wrench values (in NED frame)
        self.force_x = 0.0   # North (N)
        self.force_y = 0.0    # East (N)
        self.force_z = -30.0   # Up (N, negative down)
        self.torque_x = 0.0   # Roll (Nm)
        self.torque_y = 0.0   # Pitch (Nm)
        self.torque_z = 0.0   # Yaw (Nm)
        
        self.get_logger().info("Wrench Test Node started")
        self.get_logger().info("Publishing wrench at 100Hz with 1-second on/off toggle:")
        self.get_logger().info(f"  Active Force: [{self.force_x:.1f}, {self.force_y:.1f}, {self.force_z:.1f}] N")
        self.get_logger().info(f"  Active Torque: [{self.torque_x:.1f}, {self.torque_y:.1f}, {self.torque_z:.1f}] Nm")
        
    def publish_wrench(self):
        """Publish wrench values at 100Hz, using toggle state to determine if active or zero"""
        # Create and publish wrench message
        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = self.get_clock().now().to_msg()
        wrench_msg.header.frame_id = 'os/body'
        
        # Set force values based on toggle state (NED frame)
        if self.wrench_enabled:
            wrench_msg.wrench.force.x = self.force_x
            wrench_msg.wrench.force.y = self.force_y
            wrench_msg.wrench.force.z = self.force_z
            wrench_msg.wrench.torque.x = self.torque_x
            wrench_msg.wrench.torque.y = self.torque_y
            wrench_msg.wrench.torque.z = self.torque_z
        else:
            # Publish zero wrench when disabled
            wrench_msg.wrench.force.x = 0.0
            wrench_msg.wrench.force.y = 0.0
            wrench_msg.wrench.force.z = 0.0
            wrench_msg.wrench.torque.x = 0.0
            wrench_msg.wrench.torque.y = 0.0
            wrench_msg.wrench.torque.z = 0.0
        
        self.publisher_.publish(wrench_msg)
    
    def toggle_wrench(self):
        """Toggle wrench on/off state every second"""
        self.wrench_enabled = not self.wrench_enabled
        state_str = "ON" if self.wrench_enabled else "OFF"
        self.get_logger().info(f"Wrench toggled {state_str}")

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
