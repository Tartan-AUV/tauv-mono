import rospy
from math import pi
from tauv_msgs.msg import XsensImuData as ImuMsg, TeledyneDvlData as DvlMsg
from geometry_msgs.msg import Vector3


class ImuDvlTransformer:
    def __init__(self):
        self._imu_sub: rospy.Subscriber = rospy.Subscriber('imu', ImuMsg, self._handle_imu)
        self._dvl_sub: rospy.Subscriber = rospy.Subscriber('dvl', DvlMsg, self._handle_dvl)
        self._imu_pub: rospy.Publisher = rospy.Publisher('imu_out', ImuMsg, queue_size=10) 
        self._dvl_pub: rospy.Publisher = rospy.Publisher('dvl_out', DvlMsg, queue_size=10)

    def start(self):
        rospy.spin()

    def _handle_imu(self, msg: ImuMsg):
        msg.orientation = Vector3(-msg.orientation.x, msg.orientation.y, -msg.orientation.z)
        msg.rate_of_turn = Vector3(-msg.rate_of_turn.x, msg.rate_of_turn.y, -msg.rate_of_turn.z)
        msg.linear_acceleration = Vector3(-msg.linear_acceleration.x, msg.linear_acceleration.y, -msg.linear_acceleration.z)
        msg.free_acceleration = Vector3(msg.free_acceleration.y, -msg.free_acceleration.x, -msg.free_acceleration.z)

        msg.orientation = Vector3(-msg.orientation.x, msg.orientation.y, -msg.orientation.z)
        msg.rate_of_turn = Vector3(-msg.rate_of_turn.x, msg.rate_of_turn.y, -msg.rate_of_turn.z)
        msg.linear_acceleration = Vector3(-msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z)
        msg.free_acceleration = Vector3(0, 0, 0)

        self._imu_pub.publish(msg)

    def _handle_dvl(self, msg: DvlMsg):
        msg.hr_velocity = Vector3(msg.hr_velocity.y, msg.hr_velocity.x, msg.hr_velocity.z)

        msg.hr_velocity = Vector3(msg.hr_velocity.y, msg.hr_velocity.x, msg.hr_velocity.z)

        self._dvl_pub.publish(msg)


def main():
    rospy.init_node('imu_dvl_transformer')
    n = ImuDvlTransformer()
    n.start()
