"""
Commander node for TartanAUV vehicles.

Initial implementation supports a single operating mode – station keeping –
which attempts to hold the vehicle at a hard-coded pose in the odom frame.

Inputs
------
/gnc/navigation_state   (tauv_msgs/NavigationState)
    Estimated vehicle state from the EKF.

Outputs
-------
/gnc/controller_command (tauv_msgs/ControllerCommand)
    Twist and feed-forward accelerations for the INDI controller.

The node maps position/orientation error to desired body-frame twist using
simple proportional control:

v_B_target = K_p_pos * R_BO * (r_O_des − r_O)
ω_B_target = 2 * K_p_att * rotVec(q_des * q_BO⁻¹)

where R_BO is the rotation from body to odom and q_BO is the corresponding
quaternion.  Feed-forward accelerations are zero for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from geometry_msgs.msg import Twist, Vector3
from tauv_msgs.msg import NavigationState, ControllerCommand
from tauv_common.util.geometry import numpify, msgify
from spatialmath import UnitQuaternion, SO3


@dataclass
class StationKeepingParams:
    """Gains and set-point for station keeping"""

    # Desired pose expressed in the *odom* frame
    r_body_desired_O: NDArray  # (3, 1)
    odom_q_body_desired: UnitQuaternion  # desired body orientation expressed as odom->body

    # Proportional gains
    kp_pos: float = 0.3  # m/s per metre of position error
    kp_att: float = 0.7  # rad/s per rad of attitude error

    @classmethod
    def default(cls) -> "StationKeepingParams":
        return cls(
            r_body_desired_O=np.c_[[0.0, 0.0, -8.0]],
            odom_q_body_desired=UnitQuaternion(),  # identity – body aligned with odom
        )


class Commander(Node):
    """Commander node generating set-points for the INDI controller."""

    def __init__(self) -> None:
        super().__init__("commander")

        # Parameters
        self.params = StationKeepingParams.default()

        # Subscriptions and publishers
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

        self._last_nav: Optional[NavigationState] = None

        # Timer to republish the last command at fixed rate (optional)
        self._timer_period = 0.1  # 10 Hz – sufficient for outer loop
        self.create_timer(self._timer_period, self._timer_callback)

        self.get_logger().info("Commander initialised (station-keeping mode)")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _handle_nav_state(self, msg: NavigationState) -> None:  # noqa: N803 – ROS callback naming
        """Process latest navigation state and publish controller command."""
        self._last_nav = msg
        cmd_msg = self._compute_command(msg)
        self._cmd_pub.publish(cmd_msg)
        self._last_cmd: ControllerCommand = cmd_msg  # cache for timer

    def _timer_callback(self) -> None:
        """Publish the last command if no new nav messages have arrived."""
        if hasattr(self, "_last_cmd"):
            self._cmd_pub.publish(self._last_cmd)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _compute_command(self, nav: NavigationState) -> ControllerCommand:
        """Generate a ControllerCommand for station keeping."""
        # Extract current pose
        p_O: NDArray = np.array([
            nav.body_pose.position.x,
            nav.body_pose.position.y,
            nav.body_pose.position.z,
        ])[:, None]  # shape (3,1)

        q_OB: UnitQuaternion = numpify(nav.body_pose.orientation)  # odom->body

        # Position error expressed in odom
        e_pos_O: NDArray = self.params.r_body_desired_O - p_O  # (3,1)

        # Convert to body frame: e_B = R_BO * e_O
        R_OB = q_OB.SO3()  # rotation odom->body
        e_pos_B: NDArray = R_OB.inv() * e_pos_O 

        # Linear velocity target
        v_B_target: NDArray = self.params.kp_pos * e_pos_B  # (3,1)

        # Attitude error – quaternion representing rotation from current to desired
        q_err: UnitQuaternion = self.params.odom_q_body_desired * q_OB.inv()

        theta, v = q_err.angvec()
        vec_err = v * theta

        w_B_target: NDArray = self.params.kp_att * vec_err[:, np.newaxis]  # (3,1)

        # ------------------------------------------------------------------
        # Assemble ControllerCommand message
        # ------------------------------------------------------------------
        cmd = ControllerCommand()
        cmd.header = nav.header  # reuse navigation timestamp
        cmd.target_twist = msgify(np.vstack((v_B_target, w_B_target)), message_type="Twist")

        return cmd


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------


def main() -> None:
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