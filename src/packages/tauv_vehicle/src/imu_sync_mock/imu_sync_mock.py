from datetime import timedelta

import rospy

from tauv_msgs.msg import ImuData as ImuDataMsg
from std_msgs.msg import Header

# Time between readings, in ms
OFFSET = 10

SKEW = 1000

class ImuSyncMock:

    def __init__(self):
        self.imu_time = rospy.Time()
        self.raw_data_pub = rospy.Publisher('/sensors/imu/raw_data', ImuDataMsg, queue_size=10)

    def send(self, timer_event):
        ros_time = rospy.Time.now()
        self.imu_time = self.imu_time + rospy.Duration(0, 10e5 * OFFSET)

        msg = ImuDataMsg()
        msg.header = Header()
        msg.header.stamp = ros_time

        msg.ros_time = ros_time
        msg.imu_time = self.imu_time

        self.raw_data_pub.publish(msg)

    def start(self):
        rospy.Timer(rospy.Duration(0, (10e5 + SKEW) * OFFSET), self.send)
        rospy.spin()

def main():
    rospy.init_node('imu_sync_mock')
    n = ImuSyncMock()
    n.start()
