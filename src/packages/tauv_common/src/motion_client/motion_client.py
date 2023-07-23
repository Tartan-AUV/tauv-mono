import rospy
from threading import Lock, Event
from enum import IntEnum
from typing import Optional
from spatialmath import SE3, SO3, SE2, Twist3

from tauv_msgs.srv import GetTrajectory, GetTrajectoryRequest, GetTrajectoryResponse
from tauv_msgs.msg import NavigationState
from std_msgs.srv import SetBool
from trajectories import Trajectory, ConstantAccelerationTrajectory, ConstantAccelerationTrajectoryParams
from tauv_util.spatialmath import ros_nav_state_to_se3, ros_nav_state_to_body_twist3, body_twist3_to_world_twist3, flatten_se3, flatten_twist3


class MotionClient:

    def __init__(self):
        self._params: ConstantAccelerationTrajectoryParams = None

        self._odom_lock: Lock = Lock()
        self._odom: Optional[SE3, Twist3] = None

        self._trajectory_lock: Lock = Lock()
        self._trajectory: Optional[Trajectory] = None
        self._trajectory_start_time: Optional[rospy.Time] = None
        self._trajectory_complete_event: Event = Event()
        self._trajectory_complete_timer: Optional[rospy.Timer] = None

        self._load_config()

        self._nav_state_sub: rospy.Subscriber = rospy.Subscriber('gnc/navigation_state', NavigationState, self._handle_nav_state)

        self._get_trajectory_server: rospy.Service = rospy.Service('gnc/get_trajectory', GetTrajectory, self._handle_get_trajectory)

        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('vehicle/thrusters/arm', SetBool)

    def _handle_get_trajectory(self, req: GetTrajectoryRequest) -> GetTrajectoryResponse:
        with self._trajectory_lock:
            res = GetTrajectoryResponse()

            if self._trajectory is None:
                if self._odom is None:
                    res.success = False
                    res.message = "no trajectory or odometry"
                    return res

                res.poses = [self._odom[0]]
                res.twists = [Twist3()]
                res.success = True
                res.message = "no trajectory, returning current pose"
                return res

            traj_time = (req.curr_time - self._trajectory_start_time).to_sec()
            poses = [None] * req.len
            twists = [None] * req.len

            for i in range(req.len):
                t = traj_time + i * req.dt
                poses[i], twists[i] = self._trajectory.evaluate(t)

            res.poses = poses
            res.twists = twists
            res.success = True
            res.message = "success"
            return res

    def arm(self, arm: bool):
        self._arm_srv(arm)

    def goto(self,
             pose: SE3,
             params: Optional[ConstantAccelerationTrajectoryParams] = None,
             flat: bool = True,
             current_time: rospy.Time = rospy.Time.now()):

        params = self._params if params is None else params

        start = self._get_start(current_time)
        if start is None:
            raise RuntimeError("no trajectory or odometry")

        start_pose, start_world_twist = start
        if flat:
            start_pose = flatten_se3(start_pose)
            start_world_twist = flatten_twist3(start_world_twist)
            pose = flatten_se3(pose)

        traj = ConstantAccelerationTrajectory(
            start_pose, start_world_twist,
            pose, Twist3(),
            params,
            relax=True,
        )

        self._set_trajectory(traj, current_time)

    def goto_relative(self,
                      relative_pose: SE3,
                      params: Optional[ConstantAccelerationTrajectoryParams] = None,
                      flat: bool = True,
                      current_time: rospy.Time = rospy.Time.now()):
        start = self._get_start(current_time)
        if start is None:
            raise RuntimeError("no trajectory or odometry")

        start_pose, _ = start
        if flat:
            start_pose = flatten_se3(start_pose)
            relative_pose = flatten_se3(relative_pose)

        pose = start_pose @ relative_pose

        self.goto(pose, params, flat=True, current_time=current_time)

    def goto_relative_with_depth(self,
                                 relative_pose: SE2,
                                 z: float,
                                 params: Optional[ConstantAccelerationTrajectoryParams] = None,
                                 current_time: rospy.Time = rospy.Time.now()):
        start = self._get_start(current_time)
        if start is None:
            raise RuntimeError("no trajectory or odometry")

        start_pose, _ = start
        start_pose = flatten_se3(start_pose)

        pose = start_pose @ SE3(relative_pose)
        t = pose.t
        t[2] = z
        pose = SE3.Rt(SO3(pose), t)

        self.goto(pose, params, flat=True, current_time=current_time)

    def wait_until_complete(self, timeout: Optional[rospy.Duration] = None):
        with self._trajectory_lock:
            if self._trajectory is None:
                return

        self._trajectory_complete_event.wait(timeout.to_sec())

    def cancel(self):
        with self._trajectory_lock:
            self._trajectory = None
            self._trajectory_start_time = None
            self._trajectory_complete_event.set()

            if self._trajectory_complete_timer is not None:
                self._trajectory_complete_timer.shutdown()

            self._trajectory_complete_timer = None

    def _set_trajectory(self, trajectory: Trajectory, current_time: rospy.Time):
        with self._trajectory_lock:
            self._trajectory_start_time = current_time
            self._trajectory = trajectory
            self._trajectory_complete_event.clear()

            if self._trajectory_complete_timer is not None:
                self._trajectory_complete_timer.shutdown()

            self._trajectory_complete_timer = rospy.Timer(
                rospy.Duration.from_sec(trajectory.duration),
                self._handle_trajectory_complete,
                oneshot=True
            )
            self._trajectory_complete_timer.run()

    def _get_start(self, time: rospy.Time) -> Optional[(SE3, Twist3)]:
        with self._trajectory_lock:
            if self._trajectory is not None:
                traj_time = (time - self._trajectory_start_time).to_sec()

                return self._trajectory.evaluate(traj_time)

        with self._odom_lock:
            if self._odom is not None:
                return self._odom

        return None

    def _handle_trajectory_complete(self, timer_event: rospy.TimerEvent):
        self._trajectory_complete_event.set()

    def _handle_nav_state(self, msg: NavigationState):
        pose = ros_nav_state_to_se3(msg)
        body_twist = ros_nav_state_to_body_twist3(msg)
        world_twist = body_twist3_to_world_twist3(pose, body_twist)
        with self._odom_lock:
            self._odom = (pose, world_twist)

    def _load_config(self):
        v_max_linear = rospy.get_param('motion/v_max_linear')
        v_max_angular = rospy.get_param('motion/v_max_angular')
        a_linear = rospy.get_param('motion/a_linear')
        a_angular = rospy.get_param('motion/a_angular')

        params = ConstantAccelerationTrajectoryParams(
            v_max_linear=v_max_linear,
            v_max_angular=v_max_angular,
            a_linear=a_linear,
            a_angular=a_angular,
        )

        self._params = params
