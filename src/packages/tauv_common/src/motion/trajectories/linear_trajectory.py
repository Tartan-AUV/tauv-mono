import rospy
from .trajectories import TrajectoryStatus, Trajectory

from tauv_msgs.srv import GetTrajResponse, GetTrajRequest
from geometry_msgs.msg import Pose, Vector3, Quaternion, Point, Twist, PoseStamped
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
import numpy as np
from collections import Iterable
from tauv_util.types import tl, tm
from tauv_util.transforms import twist_world_to_body

# imports for s-curve traj lib:
from .pyscurve import ScurvePlanner

# math
from math import sin, cos, atan2, sqrt, ceil
import collections

ScurveParams = collections.namedtuple('ScurveParams', 'v_max a_max j_max')


class LinearSegment(object):
    def __init__(self, p0, p1, h0, h1, params_lin, params_ang, start_velocity=(0, 0, 0), start_angular_velocity=(0, 0, 0)):
        p = ScurvePlanner()
        # Build s-curve 1-dof trajectory for position:
        q0 = p0
        q1 = p1
        v0 = start_velocity
        v1 = [0, 0, 0]
        v_max = params_lin.v_max
        a_max = params_lin.a_max
        j_max = params_lin.j_max
        print(q0, q1, v0, v1)
        self.tr_lin = p.plan_trajectory(q0, q1, v0, v1, v_max, a_max, j_max)

        # Build s-curve 1-dof trajectory for heading:
        q0 = h0
        q1 =  h1
        v0 = start_angular_velocity
        v1 = [0, 0, 0]
        v_max = params_ang.v_max
        a_max = params_ang.a_max
        j_max = params_ang.j_max
        print(q0, q1, v0, v1)
        self.tr_ang = p.plan_trajectory(q0, q1, v0, v1, v_max, a_max, j_max)

    def duration(self):
        return max(self.tr_lin.time[0], self.tr_ang.time[0])

    def __call__(self, t, order=0):
        if t < 0:
            t = 0

        p = self.tr_lin(min(t, self.tr_lin.time[0]))
        h = self.tr_ang(min(t, self.tr_ang.time[0]))

        res = [p[0][2 - order], p[1][2 - order], p[2][2 - order], h[0][2 - order], h[1][2 - order], h[2][2 - order]]

        return res


# Uses linear segments between waypoints, with a s-curve velocity
# profile for each waypoint. These trajectories are NOT smooth, and require
# stopping at each point. Initial velocity in the direction of the first segment is
# accounted for, however velocity perpendicular to the first segment is not.
class LinearTrajectory(Trajectory):
    def __init__(self, start_pose, start_twist, positions, orientations, v=0.4, a=0.4, j=0.4):
        # MinSnapTrajectory allows a good deal of customization:
        #
        # - start_pose: unstamped Pose TODO: support stamped poses/twists in other frames!
        # - start_twist: unstamped Twist (measured in world frame!)
        # - positions: The position waypoints as a Point[].
        # - headings: heading angles corresponding to positions as a float[].
        #             If None: auto-orient forwards for full trajectory.
        # - v, a, j: max velocity, acceleration, and jerk respectively. (Applies to linear *and* angular trajectories)
        # - autowind_headings: If true, the headings will be automatically winded to reduce the amount of yaw required
        #                      to achieve them. If false, then the robot will use absolute headings which may result in
        #                      yaw motions greater than pi radians. Setting this to false can be useful for forcing the
        #                      robot to do a 360 spin, for example.
        #
        self.status = TrajectoryStatus.PENDING

        positions = [tl(p) for p in positions]

        assert(len(positions) > 0)
        assert(all([len(e) == 3 for e in positions]))
        # if headings is not None:
        #     assert(len(headings) == len(positions))

        start_pos = tl(start_pose.position)
        start_orientation = np.flip(Rotation.from_quat(tl(start_pose.orientation)).as_euler("ZYX"))
        start_vel = tl(start_twist.linear)
        start_ang_vel = tl(start_twist.angular)

        # Create list of waypoints (position and heading + velocities for each)
        p = [start_pos] + positions
        pv = [start_vel] + [[0, 0, 0]] * len(positions)

        o = [start_orientation] + orientations
        ov = [start_ang_vel] + [[0, 0, 0]] * len(positions)

        #     if autowind_headings:
        #         psi = (psi + np.pi) % (2 * np.pi) - np.pi
        #         last_psi = (last_psi + np.pi) % (2 * np.pi) - np.pi
        #         if psi - last_psi < -np.pi:
        #             psi += 2 * np.pi
        #         if psi - last_psi > np.pi:
        #             psi -= 2 * np.pi
        #         last_psi = psi

        # Create trajectories for each segment:
        lin_params = ScurveParams(v, a, j)
        ang_params = ScurveParams(0.05, 0.02, j)

        self.segments = []
        self.ts = []

        for i in range(len(p) - 1):
            s = LinearSegment(p[i], p[i+1], o[i], o[i+1], lin_params, ang_params,
                              start_velocity=pv[i], start_angular_velocity=ov[i])
            t = s.duration()
            if i > 0:
                t += self.ts[i-1]
            self.ts.append(t)
            self.segments.append(s)

        self.start_time = rospy.Time.now().to_sec()
        self.status = TrajectoryStatus.INITIALIZED

    def get_points(self, request):
        assert(isinstance(request, GetTrajRequest))

        res = GetTrajResponse()

        poses = []
        twists = []

        elapsed = request.curr_time.to_sec() - self.start_time
        T = self.duration().to_sec()

        for i in range(request.len):
            t = request.dt * i + elapsed
            if t > T:
                t = T

            # Find appropriate segment:
            seg = 0
            while self.ts[seg] <= t:
                seg += 1
                if seg > len(self.ts) - 1:
                    seg = len(self.ts) - 1
                    break

            p = self.segments[seg](t, order=0)
            v = self.segments[seg](t, order=1)

            pose = Pose(tm(p[0:3], Point), tm(Rotation.from_euler("ZYX", [p[5], p[4], p[3]]).as_quat(), Quaternion))
            twist = Twist(tm(v[0:3], Vector3), tm([v[3], v[4], v[5]], Vector3))
            poses.append(pose)
            twists.append(twist)

        res.twists = twists
        res.poses = poses
        res.auto_twists = False
        res.success = True
        return res

    def duration(self):
        return rospy.Duration(self.ts[-1])

    def time_remaining(self):
        end_time = rospy.Time(self.start_time) + self.duration()
        return end_time - rospy.Time.now()

    def set_executing(self):
        self.status = TrajectoryStatus.EXECUTING

    def get_status(self):
        if self.time_remaining().to_sec() <= 0:
            self.status = TrajectoryStatus.FINISHED

        # TODO: determine if stabilized, timed out.

        # if self.time_remaining().to_sec() <= -1:
        #     self.status = TrajectoryStatus.STABILIZED

        return self.status

    def as_path(self, dt=0.1):
        request = GetTrajRequest()
        request.curr_pose = Pose()
        request.curr_twist = Twist()
        request.len = int(ceil(self.duration().to_sec()/dt))
        request.dt = dt
        request.curr_time = rospy.Time.from_sec(self.start_time)
        res = self.get_points(request)

        start_time = rospy.Time.now()

        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = start_time

        stamped_poses = []
        for i, p in enumerate(res.poses):
            ps = PoseStamped()
            ps.header.stamp = start_time + rospy.Duration.from_sec(dt * i)
            ps.pose = p
            stamped_poses.append(ps)
        path.poses = stamped_poses
        return path
