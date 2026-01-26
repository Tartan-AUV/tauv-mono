#!/usr/bin/env python3
"""Teleop CLI node for TartanAUV vehicles.

This node starts an interactive command-line interface that allows an operator
to send *goto* commands to the commander via the ``gnc/goto`` service.

The CLI supports the following command (more can be added later):

    goto x y z heading [pitch roll] [velocity]

All positional arguments are expressed in the odometry (world) frame.
Angles are given in *degrees* for convenience; they are internally converted to
radians.  Velocity is optional – specifying 0 (or omitting it) lets the
commander use its default velocity.

Arrow-key line editing and history navigation are provided via the *readline*
module that ships with Python on POSIX systems.
"""

from __future__ import annotations

import cmd
import math
import threading
import time

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from tauv_msgs.srv import Goto

try:
    # Enables arrow-key editing & history on POSIX
    import readline  # noqa: F401
except ImportError:
    # On non-POSIX OSes the CLI will still work, just without fancy editing
    pass


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert roll, pitch, yaw (radians) → quaternion (x, y, z, w)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


class TeleopNode(Node):
    """ROS2 node wrapping a Goto service client."""

    def __init__(self) -> None:
        super().__init__('teleop')
        self._goto_client = self.create_client(Goto, 'gnc/goto')
        if not self._goto_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Goto service "gnc/goto" not available – CLI will still run.')
        else:
            self.get_logger().info('Connected to "gnc/goto" service.')

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def send_goto(self, pose: Pose, velocity: float = 0.0) -> tuple[bool, str]:
        """Send a *Goto* request and wait for the response.

        Args:
            pose: Target pose in the odometry frame.
            velocity: Desired linear velocity in m/s (≤0 → use default).

        Returns:
            (success flag, message string) tuple from the service response.
        """
        if not self._goto_client.service_is_ready():
            return False, 'Goto service not available.'

        req = Goto.Request()
        req.target_pose = pose
        req.velocity = float(velocity)

        future = self._goto_client.call_async(req)
        while rclpy.ok() and not future.done():
            time.sleep(0.05)

        if future.result() is not None:
            return future.result().success, future.result().message
        if future.exception() is not None:
            return False, f'Exception: {future.exception()}'
        return False, 'Unknown error.'


class TeleopCLI(cmd.Cmd):
    """Interactive command-line interface for vehicle teleoperation."""

    intro = 'Teleop CLI – type "help" for available commands. Press Ctrl-D or type "exit" to quit.'
    prompt = '(teleop) '

    def __init__(self, node: TeleopNode):
        super().__init__()
        self.node = node

    # ------------------------------------------------------------------
    # CLI commands
    # ------------------------------------------------------------------

    def do_goto(self, arg: str):  # noqa: D401 – kept for cmd API
        """goto x y z heading [pitch roll] [velocity]

        Send a *goto* command to the commander.

        • x, y, z – Position in metres (odometry frame)
        • heading – Yaw angle in *degrees* (0° = facing +X)
        • pitch, roll – Optional angles in degrees (default 0)
        • velocity – Optional linear speed in m/s (default: commander default)
        """
        tokens = arg.split()
        if len(tokens) < 4:
            print('Usage: goto x y z heading [pitch roll] [velocity]')
            return

        try:
            x = float(tokens[0])
            y = float(tokens[1])
            z = float(tokens[2])
            heading_deg = float(tokens[3])
            pitch_deg = float(tokens[4]) if len(tokens) >= 5 else 0.0
            roll_deg = float(tokens[5]) if len(tokens) >= 6 else 0.0
            velocity = float(tokens[6]) if len(tokens) >= 7 else 0.0
        except ValueError as exc:
            print(f'Invalid numeric value: {exc}')
            return

        # Convert degrees → radians and then to quaternion
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(heading_deg)
        qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw

        ok, message = self.node.send_goto(pose, velocity)
        status = '✓' if ok else '✗'
        print(f'[{status}] {message}')

    # ------------------------------------------------------------------
    # Misc / shell housekeeping
    # ------------------------------------------------------------------

    def do_exit(self, arg):  # noqa: D401 – kept for cmd API
        """Exit the CLI."""
        return self._exit()

    def do_quit(self, arg):  # noqa: D401 – alias for exit
        return self._exit()

    def do_EOF(self, arg):  # Ctrl-D handler
        return self._exit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _exit(self):
        print('Exiting...')
        return True


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main() -> None:  # noqa: D401 – simple description
    """Run the teleop CLI node."""
    rclpy.init()
    node = TeleopNode()

    # Spin ROS node in a background thread so that service responses are handled
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        TeleopCLI(node).cmdloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
