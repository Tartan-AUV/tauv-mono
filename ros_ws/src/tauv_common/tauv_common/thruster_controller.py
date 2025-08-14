"""
This is a simple open-loop thruster controller that uses a first-order deadband model of the thruster
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.subscription import Subscription
from rclpy.publisher import Publisher

from tauv_msgs.msg import TargetThrust, ThrusterSetpoint

THRUST_COEFF_FWD = 3.645e-3 # N/(rad/s)^2
THRUST_COEFF_REV = 2.905e-3 # N/(rad/s)^2
MAX_SETPOINT = 362.9 # rad/s

class ThrusterController(Node):
    
    def __init__(self):
        super().__init__("thruster_controller")

        self._target_thrust_sub = self.create_subscription(TargetThrust, "target_thrust", self._handle_target_thrust, 10)
        self._thruster_setpoint_pub = self.create_publisher(ThrusterSetpoint, "thruster_setpoint", 10)

    def _get_omega(self, f: float) -> float:
        return np.sqrt(min(MAX_SETPOINT, abs(f)) / THRUST_COEFF_REV) * np.sign(f)

    def _handle_target_thrust(self, target_thrust: TargetThrust):
        # Get target thrust
        f = target_thrust.target_thrust  # (n_thrusters,)
        thruster_setpoint = ThrusterSetpoint()
        thruster_setpoint.omega_radps = [self._get_omega(f_i) for f_i in f]
        thruster_setpoint.enables = [True] * len(f)
        self._thruster_setpoint_pub.publish(thruster_setpoint)

def main():
    rclpy.init()
    node = ThrusterController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
