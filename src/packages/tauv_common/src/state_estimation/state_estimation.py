import rospy
import tf
import numpy as np

from scipy.spatial.transform import Rotation
from tauv_util.types import tl, tm
from tauv_util.transforms import quat_to_rpy
from tauv_msgs.msg import Pose as PoseMsg, XsensImuData as ImuMsg, TeledyneDvlData as DvlMsg
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, Quaternion, Pose, PoseWithCovariance, Twist, TwistWithCovariance
from nav_msgs.msg import Odometry as OdometryMsg

from .ekf import EKF

class StateEstimation:

    def __init__(self):
        self._initialized = False

        self._odom_broadcaster: tf.TransformBroadcaster = tf.TransformBroadcaster()

        self._pose_pub: rospy.Publisher = rospy.Publisher('pose', PoseMsg, queue_size=10)
        self._odom_pub: rospy.Publisher = rospy.Publisher('odom', OdometryMsg, queue_size=10)

        self._imu_sub: rospy.Subscriber = rospy.Subscriber('imu', ImuMsg, self._handle_imu)
        self._dvl_sub: rospy.Subscriber = rospy.Subscriber('dvl', DvlMsg, self._handle_dvl)

        dvl_offset = np.array(rospy.get_param('~dvl_offset'))

        imu_covariance = np.diag(rospy.get_param('~imu_covariance'))
        dvl_covariance = np.diag(rospy.get_param('~dvl_covariance'))

        self._ekf: EKF = EKF(dvl_offset, imu_covariance, dvl_covariance)

        self._initialized = True

    def _handle_imu(self, msg: ImuMsg):
        if not self._initialized:
            return

        timestamp = msg.header.stamp

        linear_acceleration = tl(msg.linear_acceleration)

        angular_velocity = tl(msg.angular_velocity)

        orientation = quat_to_rpy(msg.orientation)
        orientation = np.array([-orientation[2], -orientation[1], -orientation[0]])
        orientation = (orientation + np.pi) % (2 * np.pi) - np.pi

        self._ekf.handle_imu_measurement(linear_acceleration, orientation, angular_velocity, timestamp)

        self._publish_state(timestamp)

    def _handle_dvl(self, msg: DvlMsg):
        if not self._initialized:
            return

        timestamp = msg.header.stamp

        velocity = tl(msg.hr_velocity)
        print(velocity)

        self._ekf.handle_dvl_measurement(velocity, timestamp)

        self._publish_state(timestamp)

    def _publish_state(self, timestamp: rospy.Time):
        state: np.array = self._ekf.get_state()

        position = Vector3(state[0], state[1], state[2])
        velocity = Vector3(state[3], state[4], state[5])
        acceleration = Vector3(state[6], state[7], state[8])
        orientation = Vector3(state[9], state[10], state[11])
        angular_velocity = Vector3(state[12], state[13], state[14])

        msg: PoseMsg = PoseMsg()
        msg.header = Header()
        msg.header.stamp = timestamp
        msg.position = position
        msg.velocity = velocity
        msg.acceleration = acceleration
        msg.orientation = orientation
        msg.angular_velocity = angular_velocity
        self._pose_pub.publish(msg)

        orientation_quat = tm(Rotation.from_euler('ZYX', state[9:12]).as_quat(), Quaternion)

        odom_msg: OdometryMsg = OdometryMsg()
        odom_msg.header = Header()
        odom_msg.header.stamp = timestamp
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
            time=timestamp,
            child='kingfisher/base_link',
            parent='odom',
        )


    def start(self):
        rospy.spin()


def main():
    rospy.init_node('state_estimation')
    n = StateEstimation()
    n.start()
