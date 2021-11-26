import rospy
from tauv_msgs.msg import SonarPulse

def print_data(data):
    rospy.loginfo("angle: ", data.angle)
    rospy.loginfo("transmit duration: ", data.transmit_duration)
    rospy.loginfo("sample period: ", data.sample_period)
    rospy.loginfo("transmit frequency: ", data.transmit_frequency)
    rospy.loginfo("number of samples: ", data.number_of_samples)
    rospy.loginfo("data length: ", data.data_length)
    rospy.loginfo("data: ", data.data)

