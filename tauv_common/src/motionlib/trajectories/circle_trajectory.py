import rospy
from trajectories import TrajectoryStatus, Trajectory

from tauv_msgs.srv import GetTrajResponse, GetTrajRequest
from geometry_msgs.msg import Pose, Vector3, Quaternion, Point, Twist, PoseStamped
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
import numpy as np
from collections import Iterable
from tauv_util.types import tl, tm

# imports for s-curve traj lib:
from linear_trajectory import LinearSegment

# math
from math import sin, cos, atan2, sqrt, ceil
import collections

ScurveParams = collections.namedtuple('ScurveParams', 'v_max a_max j_max')


# Circle around a point.
class CircleTrajectory(Trajectory):
    def __init__(self, start_pose, start_twist, circle_pose, radius, start_theta=None, end_theta=None, leadin=True,
                 heading=np.pi, reverse=False, v=0.4, a=0.4, j=0.4, start_time=None):
        # MinSnapTrajectory allows a good deal of customization:
        #
        # - start_pose: unstamped Pose TODO: support stamped poses/twists in other frames!
        # - start_twist: unstamped Twist (measured in world frame!)
        # - circle_pose: Define the position and orientation of the circle:
        #      circle_pose.position: position of the centroid of the circle
        #      circle_pose.orientation: orientation of the circle. The circle will be about the Z axis of the frame
        #                               created by circle_pose, with theta relative to the X axis. Eg: if set to the
        #                               unit quaternion (0,0,0,1), ie no roll, pitch, or yaw, then the circle will be
        #                               horizontal, with theta=0 laying on the X axis of the frame.
        # - radius: radius to circle at
        # - start_theta: Initial theta about axis. If set to None, use current position to determine.
        # - end_theta: Final theta about axis. If set to None, perform exactly one rotation (ie, start_theta + 2*pi)
        # - heading: heading angle relative to radius vector. eg: pi means look at centroid. Defined relative to axis.
        # - v, a, j: max velocity, acceleration, and jerk respectively. (Applies to linear *and* angular trajectories)
        # - start_time: manually override the start time of the trajectory. If set to None, it will use the current
        #               time. This is extremely useful for "hot-starting" trajectories. This is useful if your target
        #               moves, and you need to regenerate the trajectory about a different point while executing
        #               without restarting the trajectory.
        self.status = TrajectoryStatus.PENDING

        start_pos = tl(start_pose.position)
        start_psi = Rotation.from_quat(tl(start_pose.orientation)).as_euler("ZYX")[0]
        start_vel = tl(start_twist.linear)
        start_ang_vel = tl(start_twist.angular)



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

            pose = Pose(tm(p[0:3], Point), tm(Rotation.from_euler("ZYX", [p[3], 0, 0]).as_quat(), Quaternion))
            twist = Twist(tm(v[0:3], Vector3), tm([0, 0, v[3]], Vector3))
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
