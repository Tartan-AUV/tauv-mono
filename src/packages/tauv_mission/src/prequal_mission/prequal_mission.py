import rospy
import numpy as np
from typing import Optional
from math import pi

from geometry_msgs.msg import Pose, Vector3
from std_srvs.srv import SetBool
from tauv_msgs.srv import GetTraj, GetTrajRequest, GetTrajResponse
from nav_msgs.msg import Odometry
from tauv_util.types import tl, tm
from tauv_util.transforms import rpy_to_quat, quat_to_rpy
from scipy.spatial.transform import Rotation
from motion.trajectories.linear_trajectory import Waypoint, LinearTrajectory


class PrequalMission:
    def __init__(self):
        self._start_pose: Optional[Pose] = None
        self._load_config()
        self._traj = None

        self._odom_sub: rospy.Subscriber = rospy.Subscriber('odom', Odometry, self._handle_odom)
        self._get_traj_srv: rospy.Service = rospy.Service('get_traj', GetTraj, self._handle_get_traj)
        self._arm_srv: rospy.ServiceProxy = rospy.ServiceProxy('arm', SetBool)

    def start(self):
        print('start')
        rospy.Timer(self._delay, self._execute_mission, oneshot=True)
        rospy.spin()

    def _execute_mission(self, timer_event):
        if self._start_pose is None:
            return

        self._start_time = rospy.Time.now()

        self._arm_srv.call(True)

        start_position = tl(self._start_pose.position)
        start_orientation = quat_to_rpy(self._start_pose.orientation)

        R = Rotation.from_quat(tl(rpy_to_quat(np.array([0.0, 0.0, start_orientation[2]]))))

        waypoints: [Waypoint] = []

        for pose in self._poses:
            rel_position = pose[0:3]
            rel_yaw = pose[3]
            linear_acceptance = pose[4]
            angular_acceptance = pose[5]

            position = start_position + R.apply(rel_position)
            orientation = start_orientation + np.array([0.0, 0.0, rel_yaw])
            orientation[2] = (orientation[2] + pi) % (2 * pi) - pi

            pose = Pose(
                position=tm(position, Vector3),
                orientation=rpy_to_quat(orientation)
            )
            waypoint = Waypoint(pose, linear_acceptance, angular_acceptance)

            print(position)
            print(orientation)

            waypoints.append(waypoint)

        try:
            self._traj = LinearTrajectory(
                waypoints,
                self._linear_params,
                self._angular_params,
            )

            self._traj.set_executing()
        except Exception as e:
            print(e)

    def _handle_get_traj(self, req: GetTrajRequest) -> GetTrajResponse:
        response = GetTrajResponse()

        if self._traj is None:
            response.success = False
            return response

        if rospy.Time.now() - self._start_time > self._timeout:
            self._arm_srv.call(False)

        return self._traj.get_points(req)

    def _handle_odom(self, msg: Odometry):
        if self._start_pose is None:
            self._start_pose = msg.pose.pose

        roll = quat_to_rpy(msg.pose.pose.orientation)[0]

        if abs(roll) > self._roll_kill_threshold:
           self._arm_srv.call(False)

    def _load_config(self):
        self._delay: rospy.Duration = rospy.Duration.from_sec(rospy.get_param('~delay'))
        self._timeout: rospy.Duration = rospy.Duration.from_sec(rospy.get_param('~timeout'))
        self._roll_kill_threshold: float = rospy.get_param('~roll_kill_threshold')
        self._linear_params: (float, float, float) = tuple(rospy.get_param('~linear_params'))
        self._angular_params: (float, float, float) = tuple(rospy.get_param('~angular_params'))
        self._poses: np.array = np.array(rospy.get_param('~poses'))

def main():
    rospy.init_node('prequal_mission')
    n = PrequalMission()
    n.start()
