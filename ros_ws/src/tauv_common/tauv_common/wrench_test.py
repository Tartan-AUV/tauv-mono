#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
import sys
import select
import termios
import tty

class WrenchKeyboardNode(Node):
    """
    ROS2 node for publishing WrenchStamped messages from keyboard commands.
    
    Controls:
    - WASD: Force in North/South/West/East directions
    - QZ: Force in Up/Down directions  
    - Arrow keys: Pitch/Yaw torque
    - ,.: Roll torque
    - Space: Stop all forces/torques
    """
    
    def __init__(self):
        super().__init__('wrench_keyboard_node')
        
        # Publisher for wrench commands
        self.publisher_ = self.create_publisher(WrenchStamped, '/os/gnc/target_wrench', 10)
        
        # Timer for publishing at regular intervals
        self.timer = self.create_timer(0.1, self.publish_wrench)  # 10 Hz
        
        # Current wrench values (in NED frame)
        self.force_x = 0.0  # North
        self.force_y = 0.0  # East
        self.force_z = 0.0  # Down
        self.torque_x = 0.0  # Roll
        self.torque_y = 0.0  # Pitch
        self.torque_z = 0.0  # Yaw
        
        # Magnitudes
        self.force_magnitude = 300.0  # N
        self.torque_magnitude = 100.0  # Nm
        
        # Terminal settings for keyboard input
        self.settings = termios.tcgetattr(sys.stdin)
        
        self.get_logger().info("Wrench Keyboard Node started")
        self.get_logger().info("Controls (NED Frame):")
        self.get_logger().info("  W/S: Force North/South (±X)")
        self.get_logger().info("  A/D: Force West/East (±Y)")
        self.get_logger().info("  Q/Z: Force Up/Down (±Z)")
        self.get_logger().info("  ↑/↓: Pitch torque (±Y)")
        self.get_logger().info("  ←/→: Yaw torque (±Z)")
        self.get_logger().info("  ,/.: Roll torque (±X)")
        self.get_logger().info("  Space: Stop all forces/torques")
        self.get_logger().info("  Ctrl+C: Exit")
        
    def get_key(self):
        """Get a single keypress from stdin"""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key
        
    def publish_wrench(self):
        """Publish current wrench values"""
        # Get keyboard input
        key = self.get_key()
        
        # Process key input
        if key:
            self.process_key(key)
            
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
        
    def process_key(self, key):
        """Process keyboard input and update wrench values"""
        # Reset all values first
        self.force_x = 0.0
        self.force_y = 0.0
        self.force_z = 0.0
        self.torque_x = 0.0
        self.torque_y = 0.0
        self.torque_z = 0.0
        
        # Force controls (WASD + QZ)
        if key == 'w':
            self.force_x = self.force_magnitude  # North
        elif key == 's':
            self.force_x = -self.force_magnitude  # South
        elif key == 'd':
            self.force_y = self.force_magnitude  # East
        elif key == 'a':
            self.force_y = -self.force_magnitude  # West
        elif key == 'q':
            self.force_z = -self.force_magnitude  # Up (negative down)
        elif key == 'z':
            self.force_z = self.force_magnitude  # Down
            
        # Torque controls (arrow keys + ,.)
        elif key == '\x1b':  # Escape sequence for arrow keys
            key2 = self.get_key()
            if key2 == '[':
                key3 = self.get_key()
                if key3 == 'A':  # Up arrow
                    self.torque_y = self.torque_magnitude  # Pitch up
                elif key3 == 'B':  # Down arrow
                    self.torque_y = -self.torque_magnitude  # Pitch down
                elif key3 == 'D':  # Left arrow
                    self.torque_z = -self.torque_magnitude  # Yaw left
                elif key3 == 'C':  # Right arrow
                    self.torque_z = self.torque_magnitude  # Yaw right
                    
        elif key == ',':
            self.torque_x = -self.torque_magnitude  # Roll left
        elif key == '.':
            self.torque_x = self.torque_magnitude  # Roll right
            
        # Stop all
        elif key == ' ':
            pass  # Already reset all values above
            
        # Exit
        elif key == '\x03':  # Ctrl+C
            self.get_logger().info("Exiting...")
            rclpy.shutdown()
            
        # Display current values for debugging
        if key and key != '\x03':
            self.get_logger().info(f"Force: [{self.force_x:.1f}, {self.force_y:.1f}, {self.force_z:.1f}] N")
            self.get_logger().info(f"Torque: [{self.torque_x:.1f}, {self.torque_y:.1f}, {self.torque_z:.1f}] Nm")
            
    def __del__(self):
        """Restore terminal settings on exit"""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = WrenchKeyboardNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
