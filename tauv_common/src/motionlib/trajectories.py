# Trajectories.py
#
# This is the home for the Tartan AUV Trajectory profiles!
#
# Each profile extends the abstract base class (abc) Trajectory, and is used to
# describe a single controlled motion.
#
# Useful resource on trajectory planning:
# https://medium.com/mathworks/trajectory-planning-for-robot-manipulators-522404efb6f0
#
# Author: Tom Scherlis
#

import rospy
import abc
from enum import Enum
from tauv_msgs.srv import GetTrajResponse, GetTrajRequest
from geometry_msgs.msg import Pose, Vector3, Quaternion, Point, Twist
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
import numpy as np
from collections import Iterable
from tauv_util.types import tl, tm

# imports for optimal spline lib:
from python_optimal_splines.OptimalTrajectory import OptimalTrajectory
from python_optimal_splines.OptimalSpline import OptimalSpline
from python_optimal_splines.OptimalSplineGen import Waypoint, compute_min_derivative_spline
from python_optimal_splines.TrajectoryWaypoint import TrajectoryWaypoint

# math
from math import sin, cos, atan2


class TrajectoryStatus(Enum):
    PENDING = 0      # Trajectory has not been initialized yet
    INITIALIZED = 1  # Trajectory has been initialized but has not started tracking yet.
    EXECUTING = 2    # MPC controller is actively tracking this trajectory
    FINISHED = 3     # Trajectory has been completed (ie, all useful points have been published to controller)
    STABILIZED = 4   # AUV has settled to within some tolerance of the final goal location.
    TIMEOUT = 5      # Trajectory has finished, but timed out while waiting to stabilize.


class Trajectory(object):
    __metaclass__ = abc.ABCMeta

    # @abc.abstractmethod
    # def initialize(self, pose, twist):
    #     # Compute the initial trajectory if necessary. This may be useful for complex
    #     # trajectories that require expensive precomputing before use online.
    #     #
    #     # Note: this function is different from __init__, since it is called after
    #     # instantiation and is provided pose and twist. Some trajectories do not care
    #     # about initial pose and twist, so all precomputation could happen in __init__
    #     # instead.
    #     #
    #     # Returns nothing.
    #     #
    #     pass

    @abc.abstractmethod
    def get_points(self, request):
        # Return n points for the mpc controller to use as its reference trajectory.
        # Should correspond to the current time/position, with a horizon provided
        # in the request.
        #
        # This function should be fast enough to run real-time. Most trajectories
        # should be precomputed in the "initialize" function.
        #
        # request is a GetTrajRequest. Returns: GetTrajResponse
        pass

    @abc.abstractmethod
    def duration(self):
        # Return an estimate of the duration of the trajectory. (float)
        # This should only be called after initialize().
        pass

    @abc.abstractmethod
    def time_remaining(self):
        # Return an estimate of the remaining time in the trajectory (float)
        # This should only be called after initialize().
        pass

    @abc.abstractmethod
    def get_status(self):
        # Return a TrajectoryStatus enum indicating progress
        pass

    @abc.abstractmethod
    def as_path(self, dt=0.1):
        # Return a Path object for visualization in rviz
        pass


# MinSnapTrajectory uses quadratic programming to solve for a 5th order spline
# with minimal snap, which is the 4th derivative of position w.r.t. time.
# The MinSnapTrajectory does not currently support roll/pitch maneuvers, and is parameterized
# by positions and yaw headings.
#
# The trajectory will stop at the last waypoint, ending with zero velocity.
#
#
# Note that while MinSnapTrajectory will use an approximate velocity, that is only used as an average
# velocity for the trajectory. True maximum velocity may exceed that, and no guarantees are made.
#
class MinSnapTrajectory(Trajectory):
    def __init__(self, curr_pose, curr_twist, positions, headings=None, directions=None, velocities=0.5, autowind_headings=True, max_alt=-0.5):
        # MinSnapTrajectory allows a good deal of customization:
        #
        # - curr_pose: unstamped Pose
        # - curr_twist: unstamped Twist (measured in world frame!)
        # - positions: The position waypoints as a Point[].
        # - headings: heading angles corresponding to positions as a float[].
        #             If None: auto-orient forwards for full trajectory.
        # - directions: directions of motion at given waypoints. Velocity vector will be
        #               parallel to this direction at corresponding waypoints. (or zero.)
        #               If None: directions will be chosen arbitrarily based on minimum snap.
        #               Quaternion[] or list[]
        # - autowind_headings: If true, the headings will be automatically winded to reduce the amount of yaw required
        #                      to achieve them. If false, then the robot will use absolute headings which may result in
        #                      yaw motions greater than pi radians. Setting this to false can be useful for forcing the
        #                      robot to do a 360 spin, for example.
        # - max_alt: This parameter enforces a maximum altitude on the resulting trajectory. Useful for ensuring that
        #            the robot will not surface accidentally, regardless of the output trajectory result.
        #
        self.frame = "odom"  # TODO: support other frames?
        self.max_alt = max_alt
        self.autowind_headings = autowind_headings

        self.status = TrajectoryStatus.PENDING
        num_wp = len(positions)
        assert(num_wp > 0)
        assert(all([len(e) == 3 for e in positions]))
        if isinstance(velocities, list) or isinstance(velocities, tuple):
            assert(len(velocities) == len(positions))

        self.start_pose = curr_pose
        self.start_twist = curr_twist

        start_pos = tl(curr_pose.position)
        q_start = tl(curr_pose.orientation)
        start_psi = Rotation.from_quat(q_start).as_euler("ZYX")[0]

        # TODO: fix these to proper frame when switching away from ground truth data.
        start_vel = tl(curr_twist.linear)
        start_ang_vel = tl(curr_twist.angular)

        waypoints = []

        self.T = self._compute_duration([start_pos, positions], velocities)

        # create initial waypoint:
        p = start_pos
        v = start_vel
        wp = TrajectoryWaypoint(tuple(p))
        wp.add_hard_constraints(order=1, values=tuple(v))
        waypoints.append(wp)

        # create the rest of the waypoints:
        for i in range(len(positions)):
            p = positions[i]
            d = None
            if directions is not None:
                d = tl(directions[i])

            p = tl(p)
            wp = TrajectoryWaypoint(tuple(p))

            # Last waypoint has zero velocity
            if i == len(positions) - 1:
                wp.add_hard_constraints(order=1, values=[0, 0, 0])

            # Add directional constraints if necessary:
            if d is not None:
                dirvec = Rotation.from_quat(d).apply([1, 0, 0])
                wp.add_hard_directional_constraint(1, dirvec)

            waypoints.append(wp)

        self.traj = OptimalTrajectory(
            order=5,
            ndims=3,
            waypoints=waypoints,
            min_derivative_order=4,
            continuity_order=2,
            constraint_check_dt=0.05
        )

        self.traj.solve(aggressiveness=None,
                        time_opt_order=None,
                        use_faster_ts=True,
                        T=self.T)

        times = self.traj.get_times()
        last_psi = start_psi
        if headings is not None:
            # Create another optimal spline for heading interpolation. Just make this 4th order, minimizing jerk.
            # This is a 1d spline without any fancy 3d features so just use the base OptimalSpline class, rather than
            # OptimalTrajectory.
            wpts = []
            # Create initial heading:
            h = start_psi
            h_d = start_ang_vel[2]
            wp = Waypoint(times[0])
            wp.add_hard_constraint(order=0, value=h)
            wp.add_hard_constraint(order=1, value=h_d)
            wpts.append(wp)

            # add middle waypoints:
            for i in range(len(times))[1:-1]:
                if headings[i] is None:
                    continue
                psi = headings[i]

                if self.autowind_headings:
                    psi = (psi + np.pi) % 2 * np.pi - np.pi
                    last_psi = (last_psi + np.pi) % 2 * np.pi - np.pi
                    if psi - last_psi < -np.pi:
                        psi += 2*np.pi
                    if psi - last_psi > np.pi:
                        psi -= 2*np.pi
                    last_psi = psi

                wp = Waypoint(times[i])
                wp.add_hard_constraint(order=0, value=psi)
                wpts.append(wp)

            wp = Waypoint(times[-1])
            if headings[-1] is not None:
                wp.add_hard_constraint(order=0, value=headings[-1])
            wp.add_hard_constraint(order=0, value=0)
            wpts.append(wp)

            self.heading_traj = compute_min_derivative_spline(order=4,
                                                              min_derivative_order=3,
                                                              continuity_order=2,
                                                              waypoints=wpts)
        else:
            self.heading_traj = None

        self.start_time = rospy.Time.now().to_sec()
        self.status = TrajectoryStatus.INITIALIZED

    def _compute_duration(self, positions, velocity):
        v = velocity
        if not isinstance(v, Iterable):
            v = [velocity] * (len(positions) - 1)

        T = 0
        for i in range(len(positions) - 1):
            p0 = np.array(tl(positions[i]))
            p1 = np.array(tl(positions[i+1]))
            d = np.dot(p1-p0, p1-p0)
            T += d/v[i]
        return T

    def get_points(self, request):
        assert(isinstance(request, GetTrajRequest))

        res = GetTrajResponse()

        poses = []
        twists = []

        elapsed = rospy.Time.now().to_sec() - self.start_time

        lasth = Rotation.from_quat(tl(request.curr_pose.orientation)).as_euler("ZYX")[0]

        for i in range(request.len):
            t = request.dt * i + elapsed
            if t > self.T:
                t = self.T

            # Create position data
            pt = tm(self.traj.val(t, dim=None, order=0), Point)
            pt.z = min(pt.z, self.max_alt)
            v = tm(self.traj.val(t, dim=None, order=1), Vector3)

            # Create attitude data
            if self.heading_traj is not None:
                h = self.heading_traj.val(0, t)
                dh = self.heading_traj.val(1, t)
            else:
                # Compute heading to point along trajectory using atan2
                dx = v.x
                dy = v.y
                a = self.traj.val(t, dim=None, order=2)
                ddx = a[0]
                ddy = a[1]

                if dx == 0 and dy == 0:
                    h = lasth
                    dh = 0
                else:
                    h = atan2(dy, dx)
                    dh = -dy/(dx**2 + dy**2) * ddx + dx/(dx**2 + dy**2) * ddy
                    # see: https://stackoverflow.com/questions/52176354/sympy-can-i-safely-differentiate-atan2
                    lasth = h

            q = tm(Rotation.from_euler("ZYX", [h, 0, 0]), Quaternion)
            av = Vector3(0, 0, dh)

            p = Pose(pt, q)
            t = Twist(v, av)

            poses.append(p)
            twists.append(t)

        res.twists = twists
        res.poses = poses
        res.auto_twists = False
        res.success = True
        return res

    def duration(self):
        return rospy.Duration(self.T)

    def time_remaining(self):
        end_time = rospy.Time(self.start_time) + self.duration()
        return end_time - rospy.Time.now()

    def get_status(self):
        return self.status

    def as_path(self, dt=0.1):
        request = GetTrajRequest()
        request.curr_pose = self.start_pose
        request.curr_twist = self.start_twist
        request.len = self.T/dt
        request.dt = dt
        res = self.get_points(request)

        path = Path()
        path.header.frame_id = self.frame
        path.poses = res.poses
        return path



# Uses linear segments between waypoints, with a trapezoidal velocity
# profile for each waypoint. These trajectories are NOT smooth, and require
# stopping at each point. Initial velocity in the direction of the first segment is
# accounted for, however velocity perpendicular to the first segment is not.
# class LinearTrapezoidalTrajectory(Trajectory):
#     def __init__(self, waypoints, speed=1, stab_threshold=0.05, timeout=5):
#         self.state = TrajectoryStatus.PENDING
#
#     def get_points(self, request):
#         pass
