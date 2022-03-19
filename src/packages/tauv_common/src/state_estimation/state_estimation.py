import rospy
import tf
import numpy as np
import bisect

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

        self._imu_sub: rospy.Subscriber = rospy.Subscriber('imu', ImuMsg, self._receive_msg)
        self._dvl_sub: rospy.Subscriber = rospy.Subscriber('dvl', DvlMsg, self._receive_msg)
        self._depth_sub: rospy.Subscriber = rospy.Subscriber('depth', DepthMsg, self._receive_msg)

        self._dt: rospy.Duration = rospy.Duration.from_sec(1.0 / rospy.get_param('~frequency'))
        self._horizon_delay: rospy.Duration = rospy.Duration.from_sec(rospy.get_param('~horizon_delay'))

        dvl_offset = np.array(rospy.get_param('~dvl_offset'))
        process_covariance = np.array(rospy.get_param('~process_covariance'))

        self._imu_covariance = np.array(rospy.get_param('~imu_covariance'))
        self._dvl_covariance = np.array(rospy.get_param('~dvl_covariance'))
        self._depth_covariance = rospy.get_param('~depth_covariance')

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
        pending_msg_queue = list(filter(lambda m: extract_msg_time(m) < horizon_time, self._msg_queue))

        for msg in pending_msg_queue:
            if isinstance(msg, ImuMsg):
                self._handle_imu(msg)
            elif isinstance(msg, DvlMsg):
                self._handle_dvl(msg)
            elif isinstance(msg, DepthMsg):
                self._handle_depth(msg)

        self._publish_state(horizon_time)
        self._last_horizon_time = horizon_time

    def _receive_msg(self, msg):
        self._msg_queue = sorted(self._msg_queue.append(msg), key=extract_msg_time)

        # TODO: Add time sanity checks

    def _handle_imu(self, msg: ImuMsg):
        if not self._initialized:
            return

        timestamp = msg.header.stamp

        orientation = tl(msg.orientation)
        angular_velocity = tl(msg.rate_of_turn)
        linear_acceleration = tl(msg.free_acceleration)
        covariance = self._imu_covariance

        self._ekf.handle_imu_measurement(orientation, angular_velocity, linear_acceleration, covariance, timestamp)

    def _handle_dvl(self, msg: DvlMsg):
        if not self._initialized:
            return

        timestamp = msg.header.stamp

        if not msg.is_hr_velocity_valid:
            return

        velocity = tl(msg.hr_velocity)

        covariance = self._get_dvl_covariance(msg)

        self._ekf.handle_dvl_measurement(velocity, covariance, timestamp)

    def _get_dvl_covariance(self, msg: DvlMsg):
        return self._dvl_covariance
        # TODO: Use standard deviation

    def _handle_depth(self, msg: DepthMsg):
        if not self._initialized:
            return

        timestamp = msg.header.stamp

        depth = msg.depth
        covariance = self._depth_covariance

        self._ekf.handle_depth_measurement(depth, covariance, timestamp)

    def _publish_state(self, time: rospy.Time):
        position, velocity, acceleration, orientation, angular_velocity = self._ekf.get_state(time)

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
