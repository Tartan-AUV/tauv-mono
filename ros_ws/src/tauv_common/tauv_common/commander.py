"""
Simplified Commander node for TartanAUV vehicles.

This node implements station keeping at a hardcoded position.
The vehicle will maintain a fixed pose in the odom frame using
proportional control.

Notes
-----
This is a simplified version that only performs station keeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from tauv_msgs.msg import NavigationState, ControllerCommand
from tauv_common.util.geometry import numpify, msgify
from spatialmath import UnitQuaternion, SO3, SE3

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Params:
    """Controller parameters for station keeping."""
    
    # Proportional gains
    kp_pos: float = 0.3  # Position control gain
    kp_att: float = 0.3  # Attitude control gain

# ---------------------------------------------------------------------------
# Commander node
# ---------------------------------------------------------------------------

class Commander(Node):
    """Simplified commander node for station keeping at a fixed position."""

    def __init__(self) -> None:
        super().__init__("commander")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.params = Params()
        
        # Translation offset from initial position
        # This will be added to the initial pose to get the target
        # Example: [0, 0, 0.5] means hold position but go 0.5m deeper
        self.position_offset = np.array([0.0, 0.0, 0.5])  # [x, y, z] in meters
        
        # Whether to maintain initial orientation (True) or use a fixed orientation (False)
        self.maintain_initial_orientation = True
        
        # Target pose will be set once we receive the first navigation state
        self.odom_T_body_target: Optional[SE3] = None
        self.target_initialized = False

        # ------------------------------------------------------------------
        # Internal state
        # ------------------------------------------------------------------
        self._last_nav: Optional[NavigationState] = None
        self._last_cmd: Optional[ControllerCommand] = None

        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------
        self._nav_state_sub: Subscription = self.create_subscription(
            NavigationState,
            "gnc/navigation_state",
            self._handle_nav_state,
            10,
        )
        self._cmd_pub: Publisher = self.create_publisher(
            ControllerCommand,
            "gnc/controller_command",
            10,
        )

        # Timer to republish last command at fixed rate (50 Hz)
        self._timer_period = 0.02
        self.create_timer(self._timer_period, self._timer_callback)

        self.get_logger().info(
            f"Commander initialized – Will station keep with offset: "
            f"x={self.position_offset[0]:.2f}, "
            f"y={self.position_offset[1]:.2f}, "
            f"z={self.position_offset[2]:.2f}m from initial position"
        )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _handle_nav_state(self, msg: NavigationState) -> None:
        """Process navigation state and publish station keeping command."""
        self._last_nav = msg
        
        # Initialize target pose on first message
        if not self.target_initialized:
            self._initialize_target_pose(msg)
        
        # Only compute and publish commands after target is initialized
        if self.target_initialized:
            cmd_msg = self._compute_station_keep_command(msg)
            self._cmd_pub.publish(cmd_msg)
            self._last_cmd = cmd_msg  # Cache for timer

    def _timer_callback(self) -> None:
        """Republish the last command at fixed rate."""
        if self._last_cmd is not None:
            self._cmd_pub.publish(self._last_cmd)

    # ------------------------------------------------------------------
    # Station keeping logic
    # ------------------------------------------------------------------

    def _compute_station_keep_command(self, nav: NavigationState) -> ControllerCommand:
        """Compute controller command for station keeping at hardcoded position."""
        # Extract current pose
        p_O: np.ndarray = np.array([
            nav.body_pose.position.x,
            nav.body_pose.position.y,
            nav.body_pose.position.z,
        ])[:, None]  # shape (3,1)
        q_OB: UnitQuaternion = numpify(nav.body_pose.orientation)  # odom->body

        # Target pose
        q_OB_target: UnitQuaternion = self.odom_T_body_target.UnitQuaternion()
        r_body_target_O: np.ndarray = self.odom_T_body_target.t.reshape(3, 1)

        # Position error in odom
        e_pos_O: np.ndarray = r_body_target_O - p_O  # (3,1)

        # Convert to body frame: e_B = R_BO * e_O
        R_OB = q_OB.SO3()  # rotation odom->body
        e_pos_B: np.ndarray = R_OB.inv() * e_pos_O

        # Linear velocity target
        v_B_target: np.ndarray = self.params.kp_pos * e_pos_B  # (3,1)

        # Attitude error – quaternion representing rotation from current to desired
        q_err: UnitQuaternion = q_OB_target * q_OB.inv()
        theta, v = q_err.angvec()
        vec_err = v * theta

        w_B_target: np.ndarray = self.params.kp_att * vec_err[:, np.newaxis]  # (3,1)

        # Assemble ControllerCommand message
        cmd = ControllerCommand()
        cmd.header = nav.header  # Reuse navigation timestamp
        cmd.target_twist = msgify(np.vstack((v_B_target, w_B_target)), message_type="Twist")
        return cmd


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