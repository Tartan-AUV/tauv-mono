#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry

class TAMPrinter:
    def __init__(self):
        print('i have been created')
        self.printer = rospy.Subscriber('/kingfisher/pose_gt', Odometry, self._print_callback)

    def _print_callback(self, data):
        print(data)

if __name__ == '__main__':
    rospy.init_node('tam_printer')
    TAMPrinter()
    rospy.spin()