import rospy
import numpy as np
from typing import Optional

from geometry_msgs.msg import Pose, PoseArray, Twist, Vector3, Quaternion, PoseStamped
from nav_msgs.msg import Path

from tauv_msgs.srv import GetTrajRequest, GetTrajResponse
from tauv_util.types import tl, tm
from tauv_util.transforms import quat_to_rpy, rpy_to_quat, twist_world_to_body, twist_body_to_world

from .trajectories import Trajectory, TrajectoryStatus
from .pyscurve import ScurvePlanner


class NewLinearTrajectory(Trajectory):

    def __init__(self, start_pose: Pose, start_twist: Twist, end_pose: Pose, end_twist: Twist, v_l: float, a_l: float, j_l: float, v_a: float, a_a: float, j_a: float):
        self.status = TrajectoryStatus.PENDING

        self._pose: Optional[Pose] = None
        self._twist: Optional[Twist] = None

        p = ScurvePlanner()

        self._start_pose = start_pose
        self._end_pose = end_pose

        start_position = tl(start_pose.position)
        end_position = tl(end_pose.position)
        start_linear_velocity = tl(start_twist.linear)
        end_linear_velocity = tl(end_twist.linear)

        self.linear_traj = p.plan_trajectory(start_position, end_position,
                                             start_linear_velocity, end_linear_velocity,
                                             v_max=v_l if v_l is not None else 0.2,
                                             a_max=a_l if a_l is not None else 0.05,
                                             j_max=j_l if j_l is not None else 0.4)

        start_orientation = quat_to_rpy(start_pose.orientation)
        end_orientation = quat_to_rpy(end_pose.orientation)

        start_body_twist = twist_world_to_body(start_pose, start_twist)
        end_body_twist = twist_world_to_body(end_pose, end_twist)
        start_angular_velocity = tl(start_body_twist.angular)
        end_angular_velocity = tl(end_body_twist.angular)

        self.angular_traj = p.plan_trajectory(start_orientation, end_orientation,
                                              start_angular_velocity, end_angular_velocity,
                                              v_max=v_a if v_a is not None else 0.1,
                                              a_max=a_a if a_a is not None else 0.05,
                                              j_max=j_a if j_a is not None else 0.05)

        self._duration: rospy.Duration = rospy.Duration.from_sec(max(self.linear_traj.time[0], self.angular_traj.time[0]))
        self.start_time: rospy.Time = rospy.Time.now()

        self.status: TrajectoryStatus = TrajectoryStatus.INITIALIZED

    def get_points(self, request: GetTrajRequest) -> GetTrajResponse:
        elapsed = (request.curr_time - self.start_time).to_sec()

        if elapsed > self._duration.to_sec():
            res: GetTrajResponse = GetTrajResponse()
            res.success = False
            return res

        poses = [None] * request.len
        twists = [None] * request.len

        for i in range(request.len):
            t = (request.dt * i) + elapsed

            if t > self._duration.to_sec():
                t = self._duration.to_sec()

            position = self.linear_traj(t)[:,2]
            linear_velocity = self.linear_traj(t)[:,1]
            orientation = self.angular_traj(t)[:,2]
            angular_velocity = self.angular_traj(t)[:,1]

            pose = Pose()
            pose.position = tm(position, Vector3)
            pose.orientation = rpy_to_quat(orientation)

            body_twist = Twist()
            body_twist.angular = tm(angular_velocity, Vector3)
            world_twist = twist_body_to_world(pose, body_twist)
            world_twist.linear = tm(linear_velocity, Vector3)

            poses[i] = pose
            twists[i] = world_twist

        res: GetTrajResponse = GetTrajResponse()
        res.poses = poses
        res.twists = twists
        res.auto_twists = False
        res.success = True
        return res

    def duration(self) -> rospy.Duration:
        return self._duration

    def time_remaining(self) -> rospy.Duration:
        return (self.start_time + self._duration) - rospy.Time.now()

    def set_executing(self):
        self.status = TrajectoryStatus.EXECUTING

    def get_status(self) -> TrajectoryStatus:
        if self.time_remaining().to_sec() <= 0:
            self.status = TrajectoryStatus.STABILIZED

        # TODO: Add something that will actually ensure stabilization

        return self.status

    def as_path(self, dt=0.1) -> Path:
        start_pose = PoseStamped()
        start_pose.header.frame_id = 'odom'
        start_pose.pose = self._start_pose
        end_pose = PoseStamped()
        end_pose.header.frame_id = 'odom'
        end_pose.pose = self._end_pose

        path = Path()
        path.header.frame_id = 'odom'
        path.poses = [start_pose, end_pose]
        return path