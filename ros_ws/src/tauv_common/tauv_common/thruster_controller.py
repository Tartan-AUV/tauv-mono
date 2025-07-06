"""
This is a simple open-loop thruster controller that uses a first-order deadband model of the thruster
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.subscription import Subscription
from rclpy.publisher import Publisher

from tauv_msgs.msg import TargetThrust, RpmCommand

THRUST_COEFF_FWD = 0.000371
THRUST_COEFF_REV = 0.000297
DEADBAND_LOWER = -47.0
DEADBAND_UPPER = 39.2
MAX_SETPOINT = 362.0

class ThrusterController(Node):
    
    def __init__(self):
        super().__init__("thruster_controller")

        self._target_thrust_sub = self.create_subscription(TargetThrust, "gnc/target_thrust", self._handle_target_thrust, 10)
        self._rpm_command_pub = self.create_publisher(RpmCommand, "vehicle/actuators/thruster_setpoint", 10)

    def _get_rpm(self, f: float) -> float:
        if f < DEADBAND_LOWER:
            f = abs(f)
            return -np.sqrt(min(MAX_SETPOINT, f) / THRUST_COEFF_REV)
        elif f > DEADBAND_UPPER:
            return np.sqrt(min(MAX_SETPOINT, f) / THRUST_COEFF_FWD)
        else:
            return 0

    def _handle_target_thrust(self, target_thrust: TargetThrust):
        # Get target thrust
        f = target_thrust.target_thrust  # (n_thrusters,)
        rpm_command = RpmCommand()
        rpm_command.rpms = [self._get_rpm(f_i) for f_i in f]
        rpm_command.enables = [True] * len(f)
        self._rpm_command_pub.publish(rpm_command)


def main():
    rclpy.init()
    node = ThrusterController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
