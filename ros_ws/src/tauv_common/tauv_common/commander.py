"""
Commander node for TartanAUV vehicles.

The node supports two operating modes:
1. Station keeping (default) – hold the vehicle at a fixed pose.
2. Goto mode – move to a target pose at a specified velocity using a ROS2 action.

A ROS action (``/gnc/goto_velocity``) is provided to command the vehicle to
move to a target pose at a specified velocity. The action provides feedback
on the current pose, distance remaining, and current velocity.

Behavioural requirements:
• The commander interpolates a trajectory from the current pose to the target
  pose at the specified velocity.
• When the target is reached or the action is cancelled, the commander reverts
  to station keeping at the current pose.
• The trajectory uses simple linear interpolation for position and spherical
  linear interpolation (SLERP) for orientation.

Notes
-----
The current implementation focuses on clarity and correctness. Velocity
ramp-up and ramp-down are not implemented in this version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional
import asyncio
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.executors import MultiThreadedExecutor
from tauv_common.util.time import Time, Duration

from geometry_msgs.msg import PoseStamped, Pose, Twist
from tauv_msgs.msg import NavigationState, ControllerCommand
from tauv_msgs.action import GotoVelocity
from tauv_common.util.geometry import numpify, msgify
from spatialmath import UnitQuaternion, SO3, SE3

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GotoTarget:
    """Target for goto action."""
    
    target_pose: SE3  # Target pose in odom frame
    velocity: float  # Desired velocity in m/s
    start_pose: SE3  # Starting pose when action was accepted
    start_time: Time  # Time when action was accepted
    total_distance: float  # Total distance to travel
    duration: float  # Expected duration in seconds


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
        self._active_goto: Optional[GotoTarget] = None  # Current goto target
        self._goal_handle: Optional[ServerGoalHandle] = None  # Current action goal handle
        self._goal_lock = threading.Lock()  # Thread safety for goal handling

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
        
        # Create action server for goto with velocity
        self._goto_action_server = ActionServer(
            self,
            GotoVelocity,
            "gnc/goto_velocity",
            execute_callback=self._execute_goto,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
        )

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
        
        # Send feedback if action is active
        with self._goal_lock:
            if self._goal_handle and self._active_goto:
                self._send_feedback()

    def _timer_callback(self) -> None:
        """Publish the last command if no new nav messages have arrived."""
        if hasattr(self, "_last_cmd"):
            self._cmd_pub.publish(self._last_cmd)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Action callbacks
    # ------------------------------------------------------------------
    
    def _goal_callback(self, goal_request) -> GoalResponse:
        """Accept or reject a new goal."""
        # Always accept new goals (preempts previous goal if any)
        self.get_logger().info("Received new goto goal")
        return GoalResponse.ACCEPT
    
    def _cancel_callback(self, goal_handle) -> CancelResponse:
        """Accept or reject a cancel request."""
        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT
    
    async def _execute_goto(self, goal_handle: ServerGoalHandle) -> GotoVelocity.Result:
        """Execute the goto action."""
        self.get_logger().info("Executing goto action")
        
        # Check if navigation state is available
        if self._last_nav is None:
            goal_handle.abort()
            result = GotoVelocity.Result()
            result.success = False
            result.message = "Navigation state not yet received"
            return result
        
        # Extract goal parameters
        target_pose: SE3 = numpify(goal_handle.request.target_pose)
        velocity = goal_handle.request.velocity
        
        # Validate velocity
        if velocity <= 0:
            goal_handle.abort()
            result = GotoVelocity.Result()
            result.success = False
            result.message = "Velocity must be positive"
            return result
        
        # Get current state
        current_pose: SE3 = numpify(self._last_nav.body_pose)
        start_time = Time.from_msg(self._last_nav.header.stamp)
        
        # Calculate distance and duration
        distance = np.linalg.norm(target_pose.t - current_pose.t)
        duration = distance / velocity if distance > 0.01 else 0.1  # Avoid divide by zero
        
        # Create goto target
        with self._goal_lock:
            self._active_goto = GotoTarget(
                target_pose=target_pose,
                velocity=velocity,
                start_pose=current_pose,
                start_time=start_time,
                total_distance=distance,
                duration=duration
            )
            self._goal_handle = goal_handle
            self.odom_T_body_target = target_pose
        
        self.get_logger().info(f"Starting goto: distance={distance:.2f}m, velocity={velocity:.2f}m/s, duration={duration:.2f}s")
        
        # Wait for completion or cancellation
        while True:
            # Check if cancelled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                with self._goal_lock:
                    self._active_goto = None
                    self._goal_handle = None
                result = GotoVelocity.Result()
                result.success = False
                result.message = "Goal cancelled"
                self.get_logger().info("Goto action cancelled")
                return result
            
            # Check if reached target
            with self._goal_lock:
                if self._active_goto and self._last_nav:
                    current_pose = numpify(self._last_nav.body_pose)
                    distance_remaining = np.linalg.norm(
                        self._active_goto.target_pose.t - current_pose.t
                    )
                    
                    # Check if close enough to target (within 10cm)
                    if distance_remaining < 0.1:
                        goal_handle.succeed()
                        self._active_goto = None
                        self._goal_handle = None
                        result = GotoVelocity.Result()
                        result.success = True
                        result.message = "Target reached"
                        self.get_logger().info("Goto action completed successfully")
                        return result
            
            # Sleep briefly
            await asyncio.sleep(0.1)
    
    def _send_feedback(self) -> None:
        """Send feedback for the active action."""
        if not self._goal_handle or not self._active_goto or not self._last_nav:
            return
        
        feedback = GotoVelocity.Feedback()
        
        # Current pose
        current_pose: SE3 = numpify(self._last_nav.body_pose)
        feedback.current_pose = msgify(current_pose, message_type="Pose")
        
        # Distance remaining
        feedback.distance_remaining = float(
            np.linalg.norm(self._active_goto.target_pose.t - current_pose.t)
        )
        
        # Current velocity (simplified - just use the commanded velocity)
        # In a real implementation, you'd compute actual velocity from state estimator
        if feedback.distance_remaining > 0.1:
            feedback.current_velocity = self._active_goto.velocity
        else:
            feedback.current_velocity = 0.0
        
        self._goal_handle.publish_feedback(feedback)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _compute_command(self, nav: NavigationState) -> ControllerCommand:
        """Compute controller command based on current mode (goto / hold)."""
        with self._goal_lock:
            if self._active_goto:
                return self._compute_goto_command(nav)
        return self._compute_station_keep_command(nav)

    # ------------------------------------------------------------------
    # Station keeping implementation (unchanged from original)
    # ------------------------------------------------------------------

    def _compute_station_keep_command(self, nav: NavigationState) -> ControllerCommand:
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

        # ------------------------------------------------------------------
        # Assemble ControllerCommand message
        # ------------------------------------------------------------------
        cmd = ControllerCommand()
        cmd.header = nav.header  # reuse navigation timestamp
        cmd.target_twist = msgify(np.vstack((v_B_target, w_B_target)), message_type="Twist")
        return cmd

    # ------------------------------------------------------------------
    # Goto implementation with velocity control
    # ------------------------------------------------------------------

    def _compute_goto_command(self, nav: NavigationState) -> ControllerCommand:
        """Compute command for goto mode with specified velocity."""
        if not self._active_goto:
            return self._compute_station_keep_command(nav)
        
        # Get current state
        current_pose: SE3 = numpify(nav.body_pose)
        current_time = Time.from_msg(nav.header.stamp)
        
        # Calculate elapsed time
        elapsed = (current_time - self._active_goto.start_time).to_sec()
        
        # Simple linear interpolation for position (no ramp-up/down for now)
        # Calculate progress along trajectory (0 to 1)
        if self._active_goto.duration > 0:
            progress = min(elapsed / self._active_goto.duration, 1.0)
        else:
            progress = 1.0
        
        # Interpolate position
        start_pos = self._active_goto.start_pose.t
        target_pos = self._active_goto.target_pose.t
        interpolated_pos = start_pos + progress * (target_pos - start_pos)
        
        # SLERP for orientation
        start_quat = self._active_goto.start_pose.UnitQuaternion()
        target_quat = self._active_goto.target_pose.UnitQuaternion()
        interpolated_quat = start_quat.interp(target_quat, s=progress)
        
        # Create interpolated target pose
        interpolated_target = SE3.Rt(interpolated_quat.SO3(), interpolated_pos)
        
        # Calculate errors in body frame
        p_O = current_pose.t.reshape(3, 1)
        q_OB = current_pose.UnitQuaternion()
        
        # Position error
        e_pos_O = interpolated_target.t.reshape(3, 1) - p_O
        R_OB = q_OB.SO3()
        e_pos_B = R_OB.inv() * e_pos_O
        
        # Velocity command based on specified velocity and direction
        # Scale by remaining distance to avoid overshoot
        distance_remaining = np.linalg.norm(e_pos_O)
        if distance_remaining > 0.01:
            # Direction vector in body frame
            direction_B = e_pos_B / np.linalg.norm(e_pos_B)
            # Apply specified velocity
            v_B_target = self._active_goto.velocity * direction_B
            # Add proportional term for trajectory tracking
            v_B_target += self.params.kp_pos * e_pos_B
        else:
            v_B_target = self.params.kp_pos * e_pos_B
        
        # Attitude error
        q_OB_target = interpolated_target.UnitQuaternion()
        q_err = q_OB_target * q_OB.inv()
        theta, v = q_err.angvec()
        vec_err = v * theta
        w_B_target = self.params.kp_att * vec_err[:, np.newaxis]
        
        # Assemble command
        cmd = ControllerCommand()
        cmd.header = nav.header
        cmd.target_twist = msgify(np.vstack((v_B_target, w_B_target)), message_type="Twist")
        return cmd


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: D401 – simple description
    rclpy.init()
    commander = Commander()
    
    # Use MultiThreadedExecutor for action server
    executor = MultiThreadedExecutor()
    executor.add_node(commander)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        commander.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()