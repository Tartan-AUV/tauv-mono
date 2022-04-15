import rospy
import numpy as np

from geometry_msgs.msg import Pose, PoseStamped, Twist, Vector3
from nav_msgs.msg import Path
from tauv_msgs.srv import GetTrajRequest, GetTrajResponse
from tauv_util.types import tl, tm
from tauv_util.transforms import linear_distance, yaw_distance, quat_to_rpy, rpy_to_quat, twist_world_to_body, twist_body_to_world

from .trajectories import Trajectory, TrajectoryStatus
from .pyscurve import ScurvePlanner

class Waypoint:
    def __init__(self, pose: Pose, linear_error: float, angular_error: float):
        self.pose: Pose = pose
        self.linear_error: float = linear_error
        self.angular_error: float = angular_error

    def to_pose_stamped(self) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = 'odom'
        ps.pose = self.pose
        return ps


class LinearTrajectory(Trajectory):
    def __init__(self, waypoints: [Waypoint], linear_constraints: (float, float, float), angular_constraints: (float, float, float)):
        self._waypoints: [Waypoint] = waypoints
        self._target: int = 1

        self._linear_constraints: (float, float, float) = linear_constraints
        self._angular_constraints: (float, float, float) = angular_constraints

        self._status = TrajectoryStatus.PENDING

        self._p = ScurvePlanner()
        self._plan()

        self._status = TrajectoryStatus.INITIALIZED

    def _plan(self):
        start = self._waypoints[self._target - 1]
        end = self._waypoints[self._target]

        print(start.pose.position)
        print(end.pose.position)

        start_position = tl(start.pose.position)
        end_position = tl(end.pose.position)
        start_linear_velocity = np.array([0.0, 0.0, 0.0])
        end_linear_velocity = np.array([0.0, 0.0, 0.0])

        self._linear_traj = self._p.plan_trajectory(
            start_position, end_position,
            start_linear_velocity, end_linear_velocity,
            v_max=self._linear_constraints[0],
            a_max=self._linear_constraints[1],
            j_max=self._linear_constraints[2],
        )

        start_yaw = quat_to_rpy(start.pose.orientation)[2]
        end_yaw = quat_to_rpy(end.pose.orientation)[2]

        start_yaw_velocity = 0.0
        end_yaw_velocity = 0.0

        # start_body_twist = twist_world_to_body(req.curr_pose, req.curr_twist)
        # start_yaw_velocity = tl(start_body_twist.angular)[2]

        self._yaw_traj = self._p.plan_trajectory(
            np.array([start_yaw]), np.array([end_yaw]),
            np.array([start_yaw_velocity]), np.array([end_yaw_velocity]),
            v_max=self._angular_constraints[0],
            a_max=self._angular_constraints[1],
            j_max=self._angular_constraints[2],
        )

        self._start_time = rospy.Time.now()


    def get_points(self, req: GetTrajRequest) -> GetTrajResponse:
        if self._status == TrajectoryStatus.STABILIZED:
            res = GetTrajResponse()
            res.success = False
            return res

        target_waypoint = self._waypoints[self._target]

        target_reached = linear_distance(req.curr_pose, target_waypoint.pose) < target_waypoint.linear_error and \
                         yaw_distance(req.curr_pose, target_waypoint.pose) < target_waypoint.angular_error and \
                            np.linalg.norm(tl(req.curr_twist.linear)) < 0.05

        if target_reached and self._target == len(self._waypoints) - 1:
            self._status = TrajectoryStatus.STABILIZED
            res = GetTrajResponse()
            res.success = False
            return res
        elif target_reached:
            self._target += 1
            self._plan()

        poses = [None] * req.len
        twists = [None] * req.len

        linear_duration = self._linear_traj.time[0]
        yaw_duration = self._yaw_traj.time[0]

        elapsed = (req.curr_time - self._start_time).to_sec()

        for i in range(req.len):
            t = elapsed + req.dt * i
            t_linear = min(t, linear_duration)
            t_yaw = min(t, yaw_duration)

            position = self._linear_traj(t_linear)[:,2]
            linear_velocity = self._linear_traj(t_linear)[:,1]

            yaw = self._yaw_traj(t_yaw)[0,2]
            yaw_velocity = self._yaw_traj(t_yaw)[0,1]

            pose = Pose()
            pose.position = tm(position, Vector3)
            pose.orientation = rpy_to_quat(np.array([0.0, 0.0, yaw]))

            twist = Twist()
            twist.linear = tm(linear_velocity, Vector3)
            twist.angular = tm(np.array([0.0, 0.0, yaw_velocity]), Vector3)

            poses[i] = pose
            twists[i] = twist

        res: GetTrajResponse = GetTrajResponse()
        res.poses = poses
        res.twists = twists
        res.auto_twists = False
        res.success = True
        return res

    def get_duration(self) -> rospy.Duration:
        return rospy.Duration.from_sec(0)

    def get_time_remaining(self) -> rospy.Duration:
        return rospy.Duration.from_sec(0)

    def set_executing(self):
        self._status = TrajectoryStatus.EXECUTING

    def get_status(self) -> TrajectoryStatus:
        return self._status

    def as_path(self) -> Path:
        path = Path()
        path.header.frame_id = 'odom'
        path.poses = list(map(lambda w: w.to_pose_stamped(), self._waypoints))
        return path
