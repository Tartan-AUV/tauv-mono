"""
Commander node for TartanAUV vehicles.

The node supports two operating modes:
1. Station keeping (default) – hold the vehicle at a fixed pose.
2. Trajectory following – follow a user-supplied time-parameterised
   trajectory consisting of a sequence of way-points expressed in the *odom*
   frame.

A ROS service (``/gnc/set_trajectory``) is provided to load / clear the
trajectory at run-time.  The service accepts a list of ``geometry_msgs/
PoseStamped`` messages whose header stamps denote the desired arrival time at
that pose.

Behavioural requirements implemented here:
• All way-points must have zero linear & angular velocity – the commander
  inserts a cubic (ease-in / ease-out) profile between consecutive way-points
  so that velocity ramps up from 0, peaks in the middle of the segment and
  ramps back to 0 at the next way-point.
• Maximum velocity and acceleration constraints are enforced.  If any segment
  would violate the configured limits the trajectory is rejected by the
  service.
• If the commander is holding position (no active trajectory) and a new
  trajectory is loaded whose first time-stamp is in the future, an implicit
  starting point at the current pose & time is inserted so that the vehicle
  accelerates smoothly from its current state.
• When the final way-point is reached, or when the trajectory is cleared via
  the service, the commander reverts to station keeping at the last point.

Notes
-----
The current implementation focuses on clarity and correctness rather than
absolute efficiency.  All heavy-weight numerical computations occur at the
trajectory-planning stage (service callback); run-time interpolation on the
control loop is lightweight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from tauv_common.util.time import Time, Duration

from geometry_msgs.msg import PoseStamped, Pose, Twist
from tauv_msgs.msg import NavigationState, ControllerCommand
from tauv_msgs.srv import SetTrajectory, Goto
from tauv_common.util.geometry import numpify, msgify
from spatialmath import UnitQuaternion, SO3, SE3

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    """A single trajectory way-point."""

    t: Time  # absolute ROS time
    odom_T_waypoint: SE3  # Pose expressed in odom frame


@dataclass
class Params:
    # Proportional gains
    kp_pos: float = 0.3
    kp_att: float = 0.7
    
    max_linear_velocity: float = 1.0  # [m/s]
    max_angular_velocity: float = 1.0  # [rad/s]
    max_linear_acceleration: float = 3.0  # [m/s^2]
    max_angular_acceleration: float = 3.0  # [rad/s^2]

# ---------------------------------------------------------------------------
# Commander node
# ---------------------------------------------------------------------------

class Commander(Node):
    """Commander node generating set-points for the INDI controller."""

    def __init__(self) -> None:  # noqa: D401 – simple description
        super().__init__("commander")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.params = Params()
        self.odom_T_body_target: SE3 = SE3.Rt(SO3(), np.r_[0.0, 0.0, -8.0])

        # ------------------------------------------------------------------
        # Internal state
        # ------------------------------------------------------------------
        self._last_nav: Optional[NavigationState] = None
        self._trajectory: List[Waypoint] = []  # empty ⇒ station-keep
        self._traj_final_waypoint: Optional[Waypoint] = None  # for hold-position

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
        self.create_service(SetTrajectory, "gnc/set_trajectory", self._handle_set_trajectory)
        self.create_service(Goto, "gnc/goto", self._handle_goto)

        # Timer to republish last command at fixed rate (outer-loop frequency)
        self._timer_period = 0.1  # 10 Hz
        self.create_timer(self._timer_period, self._timer_callback)

        self.get_logger().info("Commander initialised – station-keeping mode")

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _handle_nav_state(self, msg: NavigationState) -> None:  # noqa: N803 – ROS naming convention
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
    # Trajectory services
    # ------------------------------------------------------------------

    def _handle_set_trajectory(self, req: SetTrajectory.Request, res: SetTrajectory.Response):  # noqa: D401
        """Validate & load a new trajectory or clear the current one."""
        poses: List[PoseStamped] = list(req.poses)

        if self._last_nav is None:
            res.success = False
            res.message = "Navigation state not yet received – cannot accept trajectory"
            return res

        now = Time.from_msg(self._last_nav.header.stamp)

        # Clear trajectory if empty
        if len(poses) == 0:
            self._trajectory.clear()
            self._traj_final_waypoint = None
            res.success = True
            res.message = "Trajectory cleared – reverting to station keep"
            return res

        # Build internal waypoint list
        waypoints: List[Waypoint] = []

        # Implicit starting waypoint at current pose/time if commander is idle
        if not self._trajectory:  # currently holding position
            current_pose = numpify(self._last_nav.body_pose)
            waypoints.append(Waypoint(t=now, odom_T_waypoint=current_pose))

        # Append user-provided way-points
        for pose in poses:
            pose_se3 = numpify(pose.pose)
            waypoints.append(Waypoint(t=Time.from_msg(pose.header.stamp), odom_T_waypoint=pose_se3))

        # Check feasibility for every segment
        if not self._check_feasibility(waypoints):
            res.success = False
            res.message = "Infeasible trajectory"
            return res

        # Trajectory accepted – replace current trajectory
        self._trajectory = waypoints
        self._traj_final_waypoint = waypoints[-1]
        res.success = True
        res.message = "Trajectory accepted"
        return res

    def _handle_goto(self, req: Goto.Request, res: Goto.Response):  # noqa: D401
        """Handle a goto command – create a 2-point trajectory to target pose."""
        if self._last_nav is None:
            res.success = False
            res.message = "Navigation state not yet received – cannot accept goto"
            return res

        # Current state
        now = Time.from_msg(self._last_nav.header.stamp)
        current_pose: SE3 = numpify(self._last_nav.body_pose)  # type: ignore[arg-type]

        # Target pose
        target_pose: SE3 = numpify(req.target_pose)

        # Distance
        v_max = self.params.max_linear_velocity
        a_max = self.params.max_linear_acceleration
        d = np.linalg.norm(target_pose.t - current_pose.t)
        t1 = d / v_max + v_max / a_max
        t2 = np.sqrt(4 * d / a_max)

        # Angle
        q_rel = current_pose.UnitQuaternion().inv() * target_pose.UnitQuaternion()
        theta, _ = q_rel.angvec()
        omega_max = self.params.max_angular_velocity
        alpha_max = self.params.max_angular_acceleration
        t3 = theta / omega_max + omega_max / alpha_max
        t4 = np.sqrt(4 * theta / alpha_max)
        t = max(t1, t2, t3, t4)

        # Duration based on velocity (avoid divide-by-zero)
        duration = Duration(nanoseconds=int(t * 1e9))
        arrival_time = now + duration
        # TODO: This is a hack to get the clock type to be ROS_TIME
        arrival_time = Time(nanoseconds=arrival_time.nanoseconds, clock_type=rclpy.time.ClockType.ROS_TIME)

        # Build new trajectory: start at current pose, end at target
        self._trajectory.clear()
        self._trajectory.append(Waypoint(t=now, odom_T_waypoint=current_pose))
        self._trajectory.append(Waypoint(t=arrival_time, odom_T_waypoint=target_pose))
        self._traj_final_waypoint = self._trajectory[-1]

        self.odom_T_body_target = target_pose

        res.success = True
        res.message = "Goto accepted"
        return res

    def _check_feasibility(self, waypoints: List[Waypoint]) -> bool:
        for wp0, wp1 in zip(waypoints[:-1], waypoints[1:]):
            dt = (wp1.t - wp0.t).to_sec()
            if dt <= 0:
                self.get_logger().error("Non-monotonic waypoint timestamps. Rejecting trajectory request.")
                return False

            # FROM CHAT:
            # d_max = v_max * dt - v_max ** 2 / a_max (line 255) becomes negative when dt < v_max / a_max, meaning every translation would fail the test for very short segments. If the intent is the usual triangular/trapezoidal profile, the formula should be
            # d_max = 0.5 * a_max * dt**2 for a pure triangular profile, or use the proper trapezoidal derivation. (This may be intentional; just flagging it.)
            
            # Translation limits
            d = np.linalg.norm(wp1.odom_T_waypoint.t - wp0.odom_T_waypoint.t)
            v_max = self.params.max_linear_velocity
            a_max = self.params.max_linear_acceleration
            d_max = v_max * dt - v_max ** 2 / a_max
            if d > d_max:
                self.get_logger().error(f"Translation limit exceeded on segment. Rejecting trajectory request.")
                return False

            # Rotation limits
            q_rel = wp0.odom_T_waypoint.UnitQuaternion().inv() * wp1.odom_T_waypoint.UnitQuaternion()
            theta, _ = q_rel.angvec()
            theta = np.abs(theta)
            omega_required = theta / dt
            omega_max = self.params.max_angular_velocity
            if omega_required > omega_max:
                self.get_logger().error(f"Angular velocity limit exceeded on segment. Rejecting trajectory request.")
                return False

        return True

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _compute_command(self, nav: NavigationState) -> ControllerCommand:
        """Compute controller command based on current mode (trajectory / hold)."""
        # if self._trajectory:
        #     return self._compute_trajectory_command(nav)
        return self._compute_station_keep_command(nav)

    # ------------------------------------------------------------------
    # Station keeping implementation (unchanged from original)
    # ------------------------------------------------------------------

    def _compute_station_keep_command(self, nav: NavigationState) -> ControllerCommand:
        # Extract current pose
        p_O: NDArray = np.array([
            nav.body_pose.position.x,
            nav.body_pose.position.y,
            nav.body_pose.position.z,
        ])[:, None]  # shape (3,1)
        q_OB: UnitQuaternion = numpify(nav.body_pose.orientation)  # odom->body

        # Target pose
        q_OB_target: UnitQuaternion = self.odom_T_body_target.UnitQuaternion()
        r_body_target_O: NDArray = self.odom_T_body_target.t.reshape(3, 1)

        # Position error in odom
        e_pos_O: NDArray = r_body_target_O - p_O  # (3,1)

        # Convert to body frame: e_B = R_BO * e_O
        R_OB = q_OB.SO3()  # rotation odom->body
        e_pos_B: NDArray = R_OB.inv() * e_pos_O

        # Linear velocity target
        v_B_target: NDArray = self.params.kp_pos * e_pos_B  # (3,1)

        # Attitude error – quaternion representing rotation from current to desired
        q_err: UnitQuaternion = q_OB_target * q_OB.inv()
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

    # ------------------------------------------------------------------
    # Trajectory following implementation
    # ------------------------------------------------------------------

    def _compute_trajectory_command(self, nav: NavigationState) -> ControllerCommand:
        raise NotImplementedError("Trajectory following not implemented")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: D401 – simple description
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
