# control_aggregator.py
#
#


import rospy
from std_msgs.msg import WrenchStamped
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header

class ControlAggregator:
    def __init__(self):
        self.controllers = rospy.get_param('configs/controllers')
        self.topic_suffix = rospy.get_param('configs/topic_suffix')
        self.enable_suffix = rospy.get_param('enable_suffix')
        self.output_frame = rospy.get_param('configs/output_frame')

    def start(self):
        print(("controllers: {}\n"
               "topic_suffix: {}\n"
               "enable_suffix: {}\n"
               "output_frame: {}").format(
            self.controllers, self.topic_suffix, self.enable_suffix, self.output_frame))


def main():
    rospy.init_node('control_aggregator')
    ca = ControlAggregator()
    ca.start()