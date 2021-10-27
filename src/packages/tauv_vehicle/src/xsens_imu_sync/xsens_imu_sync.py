import rospy
import numpy as np

from tauv_msgs.msg import ImuSync as ImuSyncMsg, ImuData as ImuDataMsg
from std_msgs.msg import Header

VAR_OFFSET_INIT = 2e-3 ** 2
VAR_SKEW_INIT = 1e-3 ** 2
VAR_OFFSET = 2e-3 ** 2
VAR_SKEW = 2e-6 ** 2

class ImuSync:
    def __init__(self):
        self.sync_pub = rospy.Publisher('/sensors/imu/sync', ImuSyncMsg, queue_size=10)
        self.data_pub = rospy.Publisher('/sensors/imu/data', ImuDataMsg, queue_size=10)
        self.raw_data_sub = rospy.Subscriber('/sensors/imu/raw_data', ImuDataMsg, self.handle_imu_data)

        self.x = np.array([0, 1])

        self.P = np.array([[VAR_OFFSET_INIT, 0], [0, VAR_SKEW_INIT]])

        self.H = np.array([[1, 0], [0, 0]])

        self.Q = np.array([[0, 0], [0, VAR_SKEW]])

        self.R = VAR_OFFSET

        self.last_corrected_time = None
        self.last_ros_time = None
        self.last_imu_time = None

    def handle_imu_data(self, data):
        ros_time = data.ros_time
        imu_time = data.imu_time

        if self.last_imu_time is None:
            self.last_imu_time = imu_time
            self.x = np.array([(ros_time - imu_time).to_sec(), 1])
    
        dt = (imu_time - self.last_imu_time).to_sec()

        F = np.array([[1, dt], [0, 1]])
        self.x = np.matmul(F, self.x)

        self.P = np.matmul(F, np.matmul(self.P, np.transpose(F))) + dt * self.Q

        S = np.matmul(self.H, np.matmul(self.P, np.transpose(self.H))) + self.R

        K = np.matmul(self.P, np.transpose(self.H)) / S

        residual = np.array([(ros_time - imu_time).to_sec(), 0]) - np.matmul(self.H, self.x)

        self.x = self.x + np.matmul(K, residual)
        self.P = np.matmul(np.identity(2) - np.matmul(K, self.H), self.P)

        corrected_time = self.convert_imu_time(imu_time)

        msg = ImuSyncMsg()
        msg.header = Header()
        msg.header.stamp = corrected_time
        msg.ros_time = ros_time
        msg.imu_time = imu_time
        msg.triggered_dvl = data.triggered_dvl

        if not self.last_corrected_time is None:
            msg.d_corrected = (corrected_time - self.last_corrected_time).to_sec()

        if not self.last_ros_time is None:
            msg.d_ros = (ros_time - self.last_ros_time).to_sec()

        if not self.last_imu_time is None:
            msg.d_imu = (imu_time - self.last_imu_time).to_sec()

        self.sync_pub.publish(msg)

        data_msg = ImuDataMsg()
        data_msg.header = Header()
        data_msg.header.stamp = corrected_time
        data_msg.ros_time = ros_time
        data_msg.imu_time = imu_time
        data_msg.triggered_dvl = data.triggered_dvl
        data_msg.orientation = data.orientation
        data_msg.linear_acceleration = data.linear_acceleration

        self.data_pub.publish(data_msg)

        self.last_corrected_time = corrected_time
        self.last_ros_time = ros_time
        self.last_imu_time = imu_time

    def convert_imu_time(self, imu_time):
        dt = (imu_time - self.last_imu_time).to_sec()

        converted_secs = imu_time.to_sec() + self.x[0] + dt * self.x[1]
        return rospy.Time.from_sec(converted_secs)

    def start(self):
        rospy.spin()

def main():
    rospy.init_node('imu_sync')
    n = ImuSync()
    n.start()
