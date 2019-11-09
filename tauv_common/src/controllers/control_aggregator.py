# control_aggregator.py
#
#


import rospy
from std_msgs.msg import WrenchStamped
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header

class ControlAggregator:
    def __init__(self):
        self.output_frame = rospy.get_param('~/output_frame')

    def start(self):
