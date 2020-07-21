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

import abc
from enum import Enum
from nav_msgs.msg import Path
from collections import Iterable

# imports for optimal spline lib:
from python_optimal_splines import OptimalSpline
from python_optimal_splines import TrajectoryWaypoint

# math


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
