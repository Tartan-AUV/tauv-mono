import rospy
from tauv_alarms import AlarmClient, Alarm

class MyNode:
    def __init__(self):
        self.ac = AlarmClient()
        self.ac.set(Alarm.DVL_DRIVER_NOT_INITIALIZED)
        self.ac.clear(Alarm.DVL_DRIVER_NOT_INITIALIZED)


if __name__ == '__main__':
    rospy.init_node('alarm_test_node')
    MyNode()
    rospy.spin()