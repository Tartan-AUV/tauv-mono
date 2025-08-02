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
# Utility helpers
# ---------------------------------------------------------------------------

def smooth_step(u: float) -> float:
    """Cubic smooth-step: 3u² − 2u³ for u∈[0,1]."""
    return 3 * u * u - 2 * u * u * u


def smooth_step_derivative(u: float) -> float:
    """Derivative of *smooth_step* w.r.t u."""
    return 6 * u * (1 - u)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    """A single trajectory way-point."""

    t: Time  # absolute ROS time
    pose_O: SE3  # Pose expressed in odom frame


@dataclass
class StationKeepingParams:
    """Gains and set-point for station keeping."""

    # Desired pose (odom frame)
    r_body_desired_O: NDArray  # (3, 1)
    odom_q_body_desired: UnitQuaternion

    # Proportional gains
    kp_pos: float = 0.3  # m/s per metre of position error
    kp_att: float = 0.7  # rad/s per rad of attitude error

    @classmethod
    def default(cls) -> "StationKeepingParams":
        return cls(
            r_body_desired_O=np.c_[[0.0, 0.0, -8.0]],
            odom_q_body_desired=UnitQuaternion(),  # identity – body aligned with odom
        )


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
        self.params = StationKeepingParams.default()

        # Motion limits (could be exposed as ROS parameters)
        self._max_velocity: float = 0.6  # m/s & rad/s
        self._max_acceleration: float = 0.3  # m/s² & rad/s²

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
            waypoints.append(Waypoint(t=now, pose_O=current_pose))

        # Append user-provided way-points
        for pose in poses:
            pose_se3 = numpify(pose.pose)
            waypoints.append(Waypoint(t=Time.from_msg(pose.header.stamp), pose_O=pose_se3))

        # Check feasibility for every segment
        for wp0, wp1 in zip(waypoints[:-1], waypoints[1:]):
            dt = (wp1.t - wp0.t).to_sec()
            if dt <= 0:
                res.success = False
                res.message = "Way-point time-stamps must be strictly increasing"
                return res

            # Translation limits
            dist = float(np.linalg.norm(wp1.pose_O.t - wp0.pose_O.t))
            vmax_req = 1.5 * dist / dt  # peak of cubic profile
            amax_req = 6.0 * dist / (dt * dt)

            # Orientation limits (use rotation angle)
            q_rel = wp0.pose_O.UnitQuaternion().inv() * wp1.pose_O.UnitQuaternion()
            angle, _ = q_rel.angvec()
            vmax_ang_req = 1.5 * angle / dt
            amax_ang_req = 6.0 * angle / (dt * dt)

            if vmax_req > self._max_velocity or vmax_ang_req > self._max_velocity:
                res.success = False
                res.message = "Velocity limit exceeded on segment"
                return res
            if amax_req > self._max_acceleration or amax_ang_req > self._max_acceleration:
                res.success = False
                res.message = "Acceleration limit exceeded on segment"
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

        # Velocity (m/s). Use default if not set or invalid.
        velocity = req.velocity if req.velocity > 0.0 else 0.5

        # Distance to travel (translation only)
        dist = float(np.linalg.norm(target_pose.t - current_pose.t))

        # Duration based on velocity (avoid divide-by-zero)
        travel_time_sec = dist / velocity if velocity > 1e-6 else 0.0
        duration = Duration(nanoseconds=int(travel_time_sec * 1e9))
        arrival_time = now + duration
        # TODO: This is a hack to get the clock type to be ROS_TIME
        arrival_time = Time(nanoseconds=arrival_time.nanoseconds, clock_type=rclpy.time.ClockType.ROS_TIME)
        self.get_logger().info(f"arrival_time clock type: {arrival_time.clock_type}")

        # Build new trajectory: start at current pose, end at target
        self._trajectory.clear()
        self._trajectory.append(Waypoint(t=now, pose_O=current_pose))
        self._trajectory.append(Waypoint(t=arrival_time, pose_O=target_pose))
        self._traj_final_waypoint = self._trajectory[-1]

        res.success = True
        res.message = "Goto accepted"
        return res

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _compute_command(self, nav: NavigationState) -> ControllerCommand:
        """Compute controller command based on current mode (trajectory / hold)."""
        if self._trajectory:
            return self._compute_trajectory_command(nav)
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

        # Position error in odom
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

    # ------------------------------------------------------------------
    # Trajectory following implementation
    # ------------------------------------------------------------------

    def _compute_trajectory_command(self, nav: NavigationState) -> ControllerCommand:
        assert self._trajectory  # non-empty

        t_now = Time.from_msg(nav.header.stamp)

        # If trajectory is finished → hold last point
        self.get_logger().info(f"t_now clock type: {t_now.clock_type}")
        self.get_logger().info(f"self._trajectory[-1].t clock type: {self._trajectory[-1].t.clock_type}")
        if t_now >= self._trajectory[-1].t:
            self._trajectory.clear()
            return self._compute_station_keep_at(self._traj_final_waypoint, nav)

        # Find the active segment index i such that t_i ≤ t < t_{i+1}
        for i in range(len(self._trajectory) - 1):
            if self._trajectory[i].t <= t_now < self._trajectory[i + 1].t:
                break
        else:  # should never happen due to earlier check
            return self._compute_station_keep_at(self._traj_final_waypoint, nav)

        wp0 = self._trajectory[i]
        wp1 = self._trajectory[i + 1]
        dt = (wp1.t - wp0.t).to_sec()
        u = (t_now - wp0.t).to_sec() / dt  # ∈ [0,1)

        S = smooth_step(u)
        dSdt = smooth_step_derivative(u) / dt

        # ------------------------------------------------------------------
        # Position & velocity (odom)
        # ------------------------------------------------------------------
        pos0 = np.c_[wp0.pose_O.t]
        pos1 = np.c_[wp1.pose_O.t]
        p_O_target = pos0 + S * (pos1 - pos0)
        v_O_target = dSdt * (pos1 - pos0)

        # ------------------------------------------------------------------
        # Orientation & angular velocity (odom)
        # ------------------------------------------------------------------
        q0 = wp0.pose_O.UnitQuaternion()
        q1 = wp1.pose_O.UnitQuaternion()
        q_rel = q0.inv() * q1
        angle, axis = q_rel.angvec()
        if angle < 1e-6:
            q_target = q0
            w_O_target = np.zeros((3, 1))
        else:
            # Interpolate orientation using spherical linear interpolation (slerp)
            q_target = q0.interp(q1, S)
            w_O_target = (axis * (angle * dSdt))[:, None]  # (3,1)

        # ------------------------------------------------------------------
        # Convert to body-frame targets
        # ------------------------------------------------------------------
        R_OB = q_target.SO3()
        v_B_target: NDArray = R_OB.inv() * v_O_target  # (3,1)
        w_B_target: NDArray = R_OB.inv() * w_O_target  # (3,1)

        # ------------------------------------------------------------------
        # Assemble ControllerCommand message
        # ------------------------------------------------------------------
        cmd = ControllerCommand()
        cmd.header = nav.header
        cmd.target_twist = msgify(np.vstack((v_B_target, w_B_target)), message_type="Twist")
        return cmd

    # ------------------------------------------------------------------
    # Helper – station keep at an arbitrary pose
    # ------------------------------------------------------------------

    def _compute_station_keep_at(self, waypoint: Optional[Waypoint], nav: NavigationState) -> ControllerCommand:
        if waypoint is None:
            # Fallback – keep current pose
            self.params.r_body_desired_O = np.c_[np.array([
                nav.body_pose.position.x,
                nav.body_pose.position.y,
                nav.body_pose.position.z,
            ])]
            self.params.odom_q_body_desired = numpify(nav.body_pose.orientation)
        else:
            self.params.r_body_desired_O = np.c_[waypoint.pose_O.t]
            self.params.odom_q_body_desired = waypoint.pose_O.UnitQuaternion()
        return self._compute_station_keep_command(nav)


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
