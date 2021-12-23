import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np

from tauv_msgs.msg import Pose as PoseMsg
from sensor_msgs.msg import Imu as ImuMsg
from uuv_sensor_ros_plugins_msgs.msg import DVL as DvlMsg
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, Pose, Point, Quaternion, Twist, PoseWithCovariance, TwistWithCovariance
from nav_msgs.msg import Odometry as OdometryMsg

from .ekf import EKF


class StateEstimation:

    def __init__(self):
        self._pose_pub: rospy.Publisher = rospy.Publisher('/state_estimation/pose', PoseMsg, queue_size=10)
        self._odometry_pub: rospy.Publisher = rospy.Publisher('/state_estimation/odometry', OdometryMsg, queue_size=10)

        self._imu_sub: rospy.Subscriber = rospy.Subscriber('/rexrov/imu', ImuMsg, self._handle_imu)
        self._dvl_sub: rospy.Subscriber = rospy.Subscriber('/rexrov/dvl', DvlMsg, self._handle_dvl)

        dvl_offset = np.array([1.4, 0, 0.31])
        imu_covariance = 0.0001 * np.ones(9, float)
        dvl_covariance = 0.0001 * np.ones(3, float)

        self._ekf: EKF = EKF(dvl_offset, imu_covariance, dvl_covariance)

    def _handle_imu(self, msg: ImuMsg):
        timestamp = msg.header.stamp
        linear_acceleration = np.array([msg.linear_acceleration.x, -msg.linear_acceleration.y, -msg.linear_acceleration.z + 9.8])
        angular_velocity = np.array([msg.angular_velocity.z, msg.angular_velocity.y, msg.angular_velocity.x])
        orientation_quat = msg.orientation
        orientation_eul = euler_from_quaternion([orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w])
        orientation = np.array([orientation_eul[2], orientation_eul[1], orientation_eul[0]])

        self._ekf.handle_imu_measurement(linear_acceleration, orientation, angular_velocity, timestamp)
        self._publish_state(timestamp)

    def _handle_dvl(self, msg: DvlMsg):
        timestamp = msg.header.stamp
        velocity = np.array([msg.velocity.z, -msg.velocity.y, msg.velocity.x])

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

    def start(self):
        rospy.spin()

def main():
    rospy.init_node('state_estimation')
    n = StateEstimation()
    n.start()