import rospy
import numpy as np

from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu as ImuMsg
from geometry_msgs.msg import Vector3, Quaternion
from tauv_msgs.msg import XsensImuData as XsensImuMsg
from tauv_util.types import tl, tm
from tauv_util.transforms import quat_to_rpy


class XsensImu:

    def __init__(self):
        self._sim_imu_sub: rospy.Subscriber = rospy.Subscriber('sim_imu', ImuMsg, self._handle_sim_imu)
        self._imu_pub: rospy.Publisher = rospy.Publisher('imu', XsensImuMsg, queue_size=10)

    def start(self):
        rospy.spin()

    def _handle_sim_imu(self, msg: ImuMsg):
        m = XsensImuMsg()
        m.header.stamp = msg.header.stamp
        orientation = quat_to_rpy(msg.orientation)
        m.orientation = Vector3(orientation[1], orientation[0], -orientation[2])

        m.linear_acceleration = Vector3(msg.linear_acceleration.y, msg.linear_acceleration.x, -msg.linear_acceleration.z)

        R = Rotation.from_euler('ZYX', np.flip(tl(m.orientation)))
        m.free_acceleration = tm(R.apply(np.array([msg.linear_acceleration.y, msg.linear_acceleration.x, -msg.linear_acceleration.z])) + np.array([0.0, 0.0, -9.81]), Vector3)

        m.rate_of_turn = Vector3(msg.angular_velocity.y, msg.angular_velocity.x, -msg.angular_velocity.z)
        self._imu_pub.publish(m)

def main():
    rospy.init_node('xsens_imu')
    x = XsensImu()
    x.start()