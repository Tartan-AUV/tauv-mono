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
from geometry_msgs.msg import Pose, Vector3, Quaternion, Point
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
import numpy as np
from collections import Iterable

# imports for optimal spline lib:
from python_optimal_splines.OptimalTrajectory import OptimalTrajectory
from python_optimal_splines.OptimalSpline import OptimalSpline
from python_optimal_splines.OptimalSplineGen import Waypoint, compute_min_derivative_spline
from python_optimal_splines.TrajectoryWaypoint import TrajectoryWaypoint


def tl(o):
    if isinstance(o, Vector3):
        return [o.x, o.y, o.z]
    if isinstance(o, Point):
        return [o.x, o.y, o.z]
    if isinstance(o, Quaternion):
        return [o.x, o.y, o.z, o.w]
    if isinstance(o, list):
        return o
    assert(False, "Unsupported type for tl!")


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
    def __init__(self, curr_pose, curr_twist, positions, headings=None, directions=None, velocities=0.5, autowind_heading=True):
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
        #
        self.status = TrajectoryStatus.PENDING
        num_wp = len(positions)
        assert(num_wp > 0)
        assert(all([len(e) == 3 for e in positions]))
        if isinstance(velocities, list) or isinstance(velocities, tuple):
            assert(len(velocities) == len(positions))

        start_pos = tl(curr_pose.position)
        q_start = tl(curr_pose.orientation)
        start_psi = Rotation.from_quat(q_start).as_euler("ZYX")[0]
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

        if headings is not None:
            # Create another optimal spline for heading interpolation. Just make this 4th order, minimizing jerk.
            wpts = []
            # Create initial heading:
            h = start_psi
            h_d = start_ang_vel[2]
            wp = Waypoint(times[0])
            wp.add_hard_constraint(order=0, value=h)
            wp.add_hard_constraint(order=1, value=h_d)
            wpts.append(wp)

            # add middle waypoints:
            for i in range(times)[1:-1]:
                if headings[i] is None:


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

        elapsed = rospy.Time.now().to_sec() - self.start_time
        for i in range(request.len):
            t = request.dt * i + elapsed
            if t > self.T:
                t = self.T
            pt = Point(self.traj.val(t, dim=None, order=0))
            v = Vector3(self.traj.val(t, dim=None, order=1))


    def duration(self):
        return rospy.Duration(self.T)
        pass

    def time_remaining(self):
        end_time = rospy.Time(self.start_time) + self.duration()
        return end_time - rospy.Time.now()

    def get_status(self):
        return self.status

    def as_path(self, dt=0.1):
        pass


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
