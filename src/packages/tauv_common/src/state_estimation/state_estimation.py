import rospy
import tf
import numpy as np
import bisect
from math import pi

from scipy.spatial.transform import Rotation
from tauv_util.types import tl, tm
from tauv_util.transforms import rpy_to_quat
from tauv_msgs.msg import Pose as PoseMsg, XsensImuData as ImuMsg, TeledyneDvlData as DvlMsg, FluidDepth as DepthMsg
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, Quaternion, Pose, PoseWithCovariance, Twist, TwistWithCovariance
from nav_msgs.msg import Odometry as OdometryMsg

from .ekf import EKF


def extract_msg_time(msg) -> float:
    return msg.header.stamp.to_sec()


class StateEstimation:

    def __init__(self):
        self._initialized = False

        self._odom_broadcaster: tf.TransformBroadcaster = tf.TransformBroadcaster()

        self._pose_pub: rospy.Publisher = rospy.Publisher('pose', PoseMsg, queue_size=10)
        self._odom_pub: rospy.Publisher = rospy.Publisher('odom', OdometryMsg, queue_size=10)

        self._imu_sub: rospy.Subscriber = rospy.Subscriber('/xsens_imu/data', ImuMsg, self._receive_msg)
        self._dvl_sub: rospy.Subscriber = rospy.Subscriber('/teledyne_dvl/data', DvlMsg, self._receive_msg)
        self._depth_sub: rospy.Subscriber = rospy.Subscriber('depth', DepthMsg, self._receive_msg)

        self._dt: rospy.Duration = rospy.Duration.from_sec(1.0 / rospy.get_param('~frequency'))
        self._horizon_delay: rospy.Duration = rospy.Duration.from_sec(rospy.get_param('~horizon_delay'))

        dvl_offset = np.array(rospy.get_param('~dvl_offset'))
        process_covariance = np.array(rospy.get_param('~process_covariance'))

        self._imu_covariance = np.array(rospy.get_param('~imu_covariance'))
        self._dvl_covariance = np.array(rospy.get_param('~dvl_covariance'))
        self._depth_covariance = np.array(rospy.get_param('~depth_covariance'))

        self._ekf: EKF = EKF(dvl_offset, process_covariance)

        self._msg_queue = []

        self._last_horizon_time: rospy.Time = rospy.Time.now() - self._horizon_delay

        self._initialized = True

    def _update(self, timer_event):
        if not self._initialized:
            return

        current_time = rospy.Time.now()
        horizon_time = current_time - self._horizon_delay

        self._msg_queue = list(filter(lambda m: extract_msg_time(m) >= self._last_horizon_time.to_sec(), self._msg_queue))
        pending_msg_queue = list(filter(lambda m: extract_msg_time(m) < horizon_time.to_sec(), self._msg_queue))

        for msg in pending_msg_queue:
            if isinstance(msg, ImuMsg):
                self._handle_imu(msg)
            elif isinstance(msg, DvlMsg):
                self._handle_dvl(msg)
            elif isinstance(msg, DepthMsg):
                self._handle_depth(msg)

        self._publish_state(current_time)

        self._last_horizon_time = horizon_time

    def _receive_msg(self, msg):
        if not self._initialized:
            return

        self._msg_queue = sorted(self._msg_queue + [msg], key=extract_msg_time)

        # TODO: Add time sanity checks

    def _handle_imu(self, msg: ImuMsg):
        if not self._initialized:
            return

        timestamp = msg.header.stamp

        orientation = tl(msg.orientation)
        adj_orientation = [orientation[0], orientation[1], orientation[2] + pi]

        free_acceleration = tl(msg.free_acceleration)
        # Accel in world NED
        adj_free_acceleration = np.array([free_acceleration[1], -free_acceleration[0], free_acceleration[2]])
        covariance = self._imu_covariance

        linear_acceleration = Rotation.from_euler('ZYX', np.flip(adj_orientation)).inv().apply(adj_free_acceleration)

        self._ekf.handle_imu_measurement(adj_orientation, linear_acceleration, covariance, timestamp)

    def _handle_dvl(self, msg: DvlMsg):
        if not self._initialized:
            return

        timestamp = msg.header.stamp

        if not msg.is_hr_velocity_valid:
            return

        velocity = tl(msg.hr_velocity)

        beam_std_dev = sum(msg.beam_standard_deviations) / 4.0

        covariance = np.maximum(1.0e-7 * np.array([beam_std_dev, beam_std_dev, beam_std_dev]), self._dvl_covariance)

        self._ekf.handle_dvl_measurement(velocity, covariance, timestamp)

    def _get_dvl_covariance(self, msg: DvlMsg):
        return self._dvl_covariance
        # TODO: Use standard deviation

    def _handle_depth(self, msg: DepthMsg):
        if not self._initialized:
            return

        timestamp = msg.header.stamp

        depth = np.array([msg.depth])
        covariance = self._depth_covariance

        self._ekf.handle_depth_measurement(depth, covariance, timestamp)

    def _publish_state(self, time: rospy.Time):
        try:
            position, velocity, acceleration, orientation, angular_velocity = self._ekf.get_state(time)
        except ValueError as e:
            print(f'get_state: ${e}')
            return

        msg: PoseMsg = PoseMsg()
        msg.header = Header()
        msg.header.stamp = time
        msg.position = position
        msg.velocity = velocity
        msg.acceleration = acceleration
        msg.orientation = orientation
        msg.angular_velocity = angular_velocity
        self._pose_pub.publish(msg)

        orientation_quat = rpy_to_quat(tl(orientation))

        odom_msg: OdometryMsg = OdometryMsg()
        odom_msg.header = Header()
        odom_msg.header.stamp = time
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'kingfisher/base_link'
        odom_msg.pose = PoseWithCovariance()
        odom_msg.pose.pose = Pose(
            position=position,
            orientation=orientation_quat,
        )
        odom_msg.twist = TwistWithCovariance()
        odom_msg.twist.twist = Twist(
            linear=velocity,
            angular=angular_velocity
        )
        self._odom_pub.publish(odom_msg)

        self._odom_broadcaster.sendTransform(
            translation=(position.x, position.y, position.z),
            rotation=(orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w),
            time=time,
            child='kingfisher/base_link',
            parent='odom',
        )

    def start(self):
        rospy.Timer(self._dt, self._update)
        rospy.spin()


def main():
    rospy.init_node('state_estimation')
    n = StateEstimation()
    n.start()
